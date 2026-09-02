from collections import defaultdict
from typing import List, Optional
import numpy as np
from wgpy_backends.webgpu.webgpu_data_type import WebGPULogicalDType, WebGPUStorageDType
from wgpy_backends.webgpu.texture import (
    WebGPUArrayTextureShape,
    get_default_texture_shape,
)
from wgpy_backends.webgpu.platform import get_platform


performance_metrics = {
    "webgpu.buffer.create": 0,
    "webgpu.buffer.delete": 0,
    "webgpu.buffer.write_count": 0,
    "webgpu.buffer.write_size": 0,
    "webgpu.buffer.write_scalar_count": 0,
    "webgpu.buffer.read_count": 0,
    "webgpu.buffer.read_size": 0,
    "webgpu.buffer.read_scalar_count": 0,
    "webgpu.buffer.buffer_count": 0,
    "webgpu.buffer.buffer_count_max": 0,
    "webgpu.buffer.buffer_size": 0,
    "webgpu.buffer.buffer_size_max": 0,
}


class GPUBufferUsage:
    MAP_READ = 0x0001
    MAP_WRITE = 0x0002
    COPY_SRC = 0x0004
    COPY_DST = 0x0008
    INDEX = 0x0010
    VERTEX = 0x0020
    UNIFORM = 0x0040
    STORAGE = 0x0080
    INDIRECT = 0x0100
    QUERY_RESOLVE = 0x0200


_pool = defaultdict(list)

added_kernels = set()

# --- graph capture: while capturing, buffer ids must stay stable (not recycled
# into the pool), because JS replays the recorded kernel sequence against these
# exact ids. Python only holds ids; the buffers themselves live in JS.
# id -> byte length, so a release can destroy them AND keep the size accounting
# straight. ---
_capture_depth = 0
_pinned_ids = {}


def begin_capture_pin():
    global _capture_depth
    _capture_depth += 1


def end_capture_pin():
    global _capture_depth
    if _capture_depth > 0:
        _capture_depth -= 1


def reset_capture_pins():
    """Abandon any pin state (a model is being released; captures go with it).
    Does NOT dispose anything — the ids still need to be read off first."""
    global _capture_depth
    _capture_depth = 0


def _maybe_pin(buffer_id: int, byte_length: int):
    if _capture_depth > 0:
        _pinned_ids[buffer_id] = byte_length


# The pool exists to skip a createBuffer when the next tensor wants a shape we have just
# finished with. A handful of spares does that; hoarding does not. Measured on a 9.83GB
# model on a 24GB machine: the pool had grown to 14.6GB, and ONE shape was holding 259
# buffers of 27.1MB — 7GB parked, for a reuse that needs two or three. Anything past the cap
# is given back to the device instead.
_POOL_PER_SHAPE = 4
# And a ceiling on the whole pool, because per-shape alone is not one: a prefill touches
# enough distinct shapes that four spares of each came to 2.33GB parked between replies,
# on a machine where the model itself is 9.2GB of 24GB.
_POOL_MAX_BYTES = 512 * 1024 * 1024
_pool_bytes = 0


def _pool_put(texture_shape: WebGPUArrayTextureShape, buffer_id: int):
    global _pool_bytes
    if buffer_id in _pinned_ids:
        return  # pinned by an active/recorded capture — never recycle
    ids = _pool[texture_shape]
    if (len(ids) >= _POOL_PER_SHAPE
            or _pool_bytes + texture_shape.byte_length > _POOL_MAX_BYTES):
        # Bounded per SHAPE rather than by a global byte budget: a budget has to be
        # apportioned between shapes that know nothing about each other, and the waste being
        # cut here is one shape's spares, not the total.
        get_platform().disposeBuffer(buffer_id)
        performance_metrics["webgpu.buffer.delete"] += 1
        performance_metrics["webgpu.buffer.buffer_count"] -= 1
        performance_metrics["webgpu.buffer.buffer_size"] -= texture_shape.byte_length
        return
    ids.append(buffer_id)
    _pool_bytes += texture_shape.byte_length


# Buffers come back to the pool from `WebGPUBuffer.__del__`, which only runs when the last
# reference goes — and a tensor caught in a reference cycle has no such moment. Refcounting
# frees the rest promptly, so this looked fine, but the cycles accumulate: on that same
# model a single collect freed 729,934 objects and moved 72,162 buffers into the pool. Until
# then those buffers were held by nothing, reachable by nothing, and still on the device.
#
# So the collect is part of allocating, not something to hope for. It is not run per buffer
# -- it walks the whole heap -- but on a budget of BYTES since the last one.
#
# Bytes and not a count of allocations: the first version counted, every 4096, and left the
# peak at 22.4GB for a 9.8GB model, because 4096 allocations is however many gigabytes the
# shapes in front of it happen to be. What has to be bounded is how much dead memory may
# pile up between collects, and that is a byte figure.
#
# The budget scales with the model rather than being a constant, so a 0.4GB model does not
# pay a 1GB allowance and a 10GB one is not collected every few tensors. It scales off the
# LOW-WATER mark -- the least ever held after a collect -- and not off what is held right
# now. Off "now" it feeds back on itself: the ledger drifts up, the budget grows with it,
# collects get rarer, and the drift accelerates. Measured with that mistake in: the budget
# had reached 1.7GB, which is 8% of 21GB, on a model whose working set is 9.2GB.
_REAP_FRACTION = 0.08
_REAP_FLOOR = 256 * 1024 * 1024
_live_floor = 0                   # least held after any collect; 0 until the first one
_reap_budget = _REAP_FLOOR
_bytes_since_reap = 0


def _note_alloc(byte_length: int):
    global _bytes_since_reap
    _bytes_since_reap += byte_length


def reap_now():
    """Collect, and re-scale the allowance from what is actually live afterwards.

    Worth calling at the end of a generation as well as from the allocation path. A collect
    can only free what nothing refers to, and mid-computation the frames on the stack still
    refer to plenty: the same collect that freed 9.4GB once the reply had finished had been
    freeing far less while it was being written. So the allocation path bounds the growth
    within a reply, and the boundary between replies is where it actually comes back.
    """
    global _bytes_since_reap, _reap_budget, _live_floor
    _bytes_since_reap = 0
    import gc

    gc.collect()
    try:
        held = get_platform().gpuBytes()[0]
    except Exception:
        return
    if held and (_live_floor == 0 or held < _live_floor):
        _live_floor = held
    base = _live_floor or held
    _reap_budget = max(_REAP_FLOOR, int(base * _REAP_FRACTION))


def _maybe_reap():
    if _bytes_since_reap < _reap_budget:
        return
    reap_now()


def _pool_get(texture_shape: WebGPUArrayTextureShape) -> Optional[int]:
    global _pool_bytes
    if len(_pool[texture_shape]) > 0:
        _pool_bytes -= texture_shape.byte_length
        return _pool[texture_shape].pop()
    return None


def release_capture_buffers():
    """Destroy every buffer a recorded capture pinned.

    Pinned buffers never enter the reuse pool when their Python object dies —
    `__del__` drops them — so without this they stay allocated on the GPU
    forever, and JS refuses their disposeBuffer while pinned too. Called at
    model release, after the JS side has been told to drop its captures and
    pins (so the disposeBuffer messages actually land)."""
    plat = get_platform()
    for buffer_id, byte_length in list(_pinned_ids.items()):
        plat.disposeBuffer(buffer_id)
        performance_metrics["webgpu.buffer.delete"] += 1
        performance_metrics["webgpu.buffer.buffer_count"] -= 1
        performance_metrics["webgpu.buffer.buffer_size"] -= byte_length
    _pinned_ids.clear()


def release_pooled_buffers():
    """Destroy everything the reuse pools hold and empty them.

    The pools exist to skip createBuffer when the next tensor has a matching
    shape. When a model is released and a DIFFERENT one loads, most shapes
    will not match, so pooled buffers would just sit on the GPU next to the
    new model's allocations until the device runs out. Called at model
    release, after `release_capture_buffers`."""
    plat = get_platform()
    for texture_shape, ids in list(_pool.items()):
        for buffer_id in ids:
            plat.disposeBuffer(buffer_id)
            performance_metrics["webgpu.buffer.delete"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] -= 1
            performance_metrics["webgpu.buffer.buffer_size"] -= (
                texture_shape.byte_length
            )
    _pool.clear()
    global _pool_bytes
    _pool_bytes = 0
    for data, ids in list(_meta_pool.items()):
        for buffer_id in ids:
            plat.disposeBuffer(buffer_id)
            performance_metrics["webgpu.buffer.delete"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] -= 1
            performance_metrics["webgpu.buffer.buffer_size"] -= len(data)
    _meta_pool.clear()


def _get_comm_buf(byte_size: int) -> np.ndarray:
    if WebGPUBuffer._comm_buf is None or WebGPUBuffer._comm_buf.size < byte_size:
        WebGPUBuffer._comm_buf = np.empty(
            (max(byte_size, 1024 * 1024),), dtype=np.uint8
        )
        get_platform().setCommBuf(WebGPUBuffer._comm_buf)
    return WebGPUBuffer._comm_buf


class WebGPUBufferBase:
    buffer_id: int


class WebGPUBuffer(WebGPUBufferBase):
    size: int  # Logical number of elements (May differ from the number of elements in the physical buffer)
    dtype: (
        np.dtype
    )  # ndarray logical type (may be different from physical representation in WebGPU)
    texture_shape: WebGPUArrayTextureShape

    _comm_buf: Optional[np.ndarray] = None
    next_id = 1

    def __init__(
        self,
        size: int,
        dtype: np.dtype,
        texture_shape: Optional[WebGPUArrayTextureShape] = None,
    ) -> None:
        self.size = size
        self.dtype = dtype
        self.texture_shape = texture_shape or get_default_texture_shape(size, dtype)
        _maybe_reap()
        pooled_buffer_id = _pool_get(self.texture_shape)
        _note_alloc(self.texture_shape.byte_length)
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGPUBuffer.next_id
            WebGPUBuffer.next_id += 1
            get_platform().createBuffer(self.buffer_id, self.texture_shape.byte_length)
            performance_metrics["webgpu.buffer.create"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] += 1
            performance_metrics[
                "webgpu.buffer.buffer_size"
            ] += self.texture_shape.byte_length
            performance_metrics["webgpu.buffer.buffer_count_max"] = max(
                performance_metrics["webgpu.buffer.buffer_count_max"],
                performance_metrics["webgpu.buffer.buffer_count"],
            )
            performance_metrics["webgpu.buffer.buffer_size_max"] = max(
                performance_metrics["webgpu.buffer.buffer_size_max"],
                performance_metrics["webgpu.buffer.buffer_size"],
            )
        _maybe_pin(self.buffer_id, self.texture_shape.byte_length)

    def __del__(self):
        # TODO: limit pooled size
        _pool_put(self.texture_shape, self.buffer_id)
        # get_platform().disposeBuffer(self.buffer_id)

    def set_data(self, array: np.ndarray):
        if self.size == 0:
            return
        buf = _get_comm_buf(self.texture_shape.byte_length)
        packed = buf.view(self.texture_shape.storage_dtype_numpy)
        packed[: array.size] = array.ravel()
        get_platform().setData(self.buffer_id, self.texture_shape.byte_length)
        performance_metrics["webgpu.buffer.write_count"] += 1
        # physical size
        performance_metrics[
            "webgpu.buffer.write_size"
        ] += self.texture_shape.byte_length
        # logical size
        if array.size <= 1:
            performance_metrics["webgpu.buffer.write_scalar_count"] += 1

    def get_data(self) -> np.ndarray:
        return self._get_data_internal(self.dtype)

    def _get_data_internal(self, original_dtype: np.dtype):
        if self.size == 0:
            return np.zeros((0,), dtype=original_dtype)
        performance_metrics["webgpu.buffer.read_count"] += 1
        buf = _get_comm_buf(self.texture_shape.byte_length)
        get_platform().getData(self.buffer_id, self.texture_shape.byte_length)
        performance_metrics["webgpu.buffer.read_size"] += self.texture_shape.byte_length
        if self.size <= 1:
            performance_metrics["webgpu.buffer.read_scalar_count"] += 1
        view = buf.view(self.texture_shape.storage_dtype_numpy)[: self.size]

        return view.copy().astype(original_dtype, copy=False)


_meta_pool = defaultdict(list)


class WebGPUMetaBuffer(WebGPUBufferBase):
    _data: bytes

    def __init__(self, data: bytes, pooled_buffer_id: Optional[int]) -> None:
        super().__init__()

        self._data = data
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGPUBuffer.next_id
            WebGPUBuffer.next_id += 1

            buf = _get_comm_buf(len(data))
            packed = buf.view(np.uint8)
            packed[: len(data)] = np.frombuffer(data, dtype=np.uint8)
            get_platform().createMetaBuffer(self.buffer_id, len(data))

            performance_metrics["webgpu.buffer.create"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] += 1
            performance_metrics["webgpu.buffer.buffer_size"] += len(data)
            performance_metrics["webgpu.buffer.buffer_count_max"] = max(
                performance_metrics["webgpu.buffer.buffer_count_max"],
                performance_metrics["webgpu.buffer.buffer_count"],
            )
            performance_metrics["webgpu.buffer.buffer_size_max"] = max(
                performance_metrics["webgpu.buffer.buffer_size_max"],
                performance_metrics["webgpu.buffer.buffer_size"],
            )
        _maybe_pin(self.buffer_id, len(data))

    @property
    def data(self):
        return self._data

    def __del__(self):
        if self.buffer_id in _pinned_ids:
            return  # pinned by a capture — never recycle
        _meta_pool[self._data].append(self.buffer_id)


class WebGPUMetaBufferItem:
    name: str
    native_type: str
    numpy_dtype_str: str

    def __init__(
        self, name: str, native_type: str, numpy_dtype_str: Optional[str] = None
    ) -> None:
        self.name = name
        self.native_type = native_type
        self.numpy_dtype_str = (
            numpy_dtype_str or {"f32": "f4", "i32": "i4", "u32": "u4"}[native_type]
        )

    def __repr__(self) -> str:
        return f"WebGPUMetaBufferItem('{self.name}', '{self.native_type}', '{self.numpy_dtype_str}')"


def create_meta_buffer(data: bytes) -> WebGPUMetaBuffer:
    pooled = _meta_pool[data]
    pooled_buffer_id = None
    if len(pooled) > 0:
        pooled_buffer_id = pooled.pop()
    new_buf = WebGPUMetaBuffer(data, pooled_buffer_id=pooled_buffer_id)
    return new_buf


def create_meta_buffer_from_structure(data_tuple: tuple, dtype) -> WebGPUMetaBuffer:
    """
    example: data_tuple = (2, 1.5), dtype = "i4,f4"
    """
    structured_array = np.array([data_tuple], dtype=dtype)
    data = structured_array.tobytes()
    return create_meta_buffer(data)


def create_meta_buffer_from_dict(
    data_dict: dict, item_definitions: List[WebGPUMetaBufferItem]
) -> WebGPUMetaBuffer:
    """
    example: data_dict = {"a": 2, "b": 1.5}, item_definitions = [WebGPUMetaBufferItem("a", "i32"), WebGPUMetaBufferItem("b", "f32")]
    """
    dtype = np.dtype([(item.name, item.numpy_dtype_str) for item in item_definitions])
    structured_array = np.array(
        [tuple(data_dict[item.name] for item in item_definitions)], dtype=dtype
    )
    data = structured_array.tobytes()
    return create_meta_buffer(data)
