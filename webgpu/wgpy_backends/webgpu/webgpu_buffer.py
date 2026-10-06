from collections import defaultdict
from typing import List, Optional
from time import perf_counter
import struct

import numpy as np
from wgpy_backends.webgpu.webgpu_data_type import WebGPULogicalDType, WebGPUStorageDType
from wgpy_backends.webgpu.texture import (
    WebGPUArrayTextureShape,
    get_default_texture_shape,
)
from wgpy_backends.webgpu.platform import get_platform


_upload_profiles = {}


def _upload_auto(key, staged, direct):
    """Choose by measured end-to-end upload time for this physical buffer shape.

    A never-repeated shape uses the established path once. Later natural
    uploads alternate paths, one upload per call; the faster median wins.
    Calibration never makes an extra copy of a weight or uses a fixed benefit
    threshold or model-name exception.
    """
    profile = _upload_profiles.setdefault(key, {"seen": 0, "staged": [],
                                                 "direct": [], "choice": None})
    if profile["choice"] is not None:
        (direct if profile["choice"] == "direct" else staged)()
        return
    mode = "staged" if profile["seen"] % 2 == 0 else "direct"
    start = perf_counter()
    (staged if mode == "staged" else direct)()
    profile[mode].append(perf_counter() - start)
    profile["seen"] += 1
    trials = 7 if key[0] < 1024 * 1024 else 3
    if len(profile["direct"]) >= trials and len(profile["staged"]) >= trials:
        profile["choice"] = (
            "direct" if sorted(profile["direct"])[trials // 2]
            < sorted(profile["staged"])[trials // 2] else "staged")


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
# Pins belong to the RECORDING that made them, and a recording is replaced whenever its
# name is captured again. Without that, every generation pinned a fresh set of decode
# intermediates and none was ever unpinned until the model was released: the pool refused
# them (`_pool_put` returns early on a pinned id), so they could neither be reused nor
# freed. Measured on a 27B, same prompt and same reply length, only a page reload between:
# the decode step went 992ms -> 537ms. That is what was accumulating.
_capture_name = None
_pins = {}                    # capture name -> {buffer_id: byte length}
# A dead temporary may be reused later in the SAME recording: the recorded
# commands before its finalizer are already enqueued, and no subsequent Python
# operation can name the dead object. Never share it with another recording.
_capture_free = defaultdict(list)
# A buffer whose Python object died while it was pinned. `_pool_put` had to refuse it, so
# nothing will ever be called for it again -- it is reachable only from here, and this is
# where it gets freed once the recording that pinned it is gone.
_orphaned = {}


def begin_capture_pin(name=None):
    """Start pinning for `name`. Re-recording a name releases the previous recording's pins.

    Released, not destroyed. A buffer pinned by the old recording may still be held by a
    live Python object -- `_kv_reserve` re-records in the MIDDLE of a generation, while the
    previous step's logits are still referenced -- and destroying it there would leave that
    object pointing at a freed id. Unpinned, it takes the ordinary path when it dies. Only
    the ones that already died while pinned are freed here, because nothing else can.
    """
    global _capture_depth, _capture_name
    if _capture_depth == 0:
        _capture_free.clear()
        key = name if name is not None else "?"
        if key in _pins:
            release_capture_pin(key)
        _pins[key] = {}
        _capture_name = key
    _capture_depth += 1


def release_capture_pin(name):
    """Unpin one retired graph; retain IDs still used by any other recording."""
    if _capture_depth:
        raise RuntimeError("cannot release capture pins while recording")
    old = _pins.pop(name, None)
    if old is None:
        raise KeyError("capture %r has no pin record" % name)
    for bid in old:
        if any(bid in pins for pins in _pins.values()):
            continue
        _pinned_ids.pop(bid, None)
        shape = _orphaned.pop(bid, None)
        if shape is not None:
            get_platform().disposeBuffer(bid)
            performance_metrics["webgpu.buffer.delete"] += 1
            performance_metrics["webgpu.buffer.buffer_count"] -= 1
            performance_metrics["webgpu.buffer.buffer_size"] -= shape.byte_length


def end_capture_pin():
    global _capture_depth, _capture_name
    if _capture_depth > 0:
        _capture_depth -= 1
        if _capture_depth == 0:
            _capture_free.clear()
            _capture_name = None


def reset_capture_pins():
    """Abandon every recording after JS has dropped its capture references.

    A profiler resets captures while the model is still live.  Leaving
    ``_pinned_ids`` populated then makes every short-lived buffer look pinned
    forever, so neither its finalizer nor the reuse pool can return it.  Only
    orphaned buffers are destroyed here: a live Tensor still owns its buffer
    and will release it through its normal finalizer.
    """
    global _capture_depth, _capture_name
    _capture_depth = 0
    _capture_name = None
    _pins.clear()
    _pinned_ids.clear()
    _capture_free.clear()
    for bid, shape in list(_orphaned.items()):
        get_platform().disposeBuffer(bid)
        performance_metrics["webgpu.buffer.delete"] += 1
        performance_metrics["webgpu.buffer.buffer_count"] -= 1
        performance_metrics["webgpu.buffer.buffer_size"] -= shape.byte_length
    _orphaned.clear()


def _maybe_pin(buffer_id: int, byte_length: int):
    if _capture_depth > 0:
        _pinned_ids[buffer_id] = byte_length
        if _capture_name is not None:
            _pins.setdefault(_capture_name, {})[buffer_id] = byte_length


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
        # Its owner is gone but the recording still needs the id, so it can be neither
        # pooled nor freed now. Remembered here so that whoever releases that recording can
        # free it -- otherwise this is the last anyone ever hears of it.
        _orphaned[buffer_id] = texture_shape
        if (_capture_depth and _capture_name is not None
                and buffer_id in _pins.get(_capture_name, ())
                and not any(buffer_id in pins for name, pins in _pins.items()
                            if name != _capture_name)):
            _capture_free[texture_shape].append(buffer_id)
        return  # pinned: only the current recording may reuse a dead temporary
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


# Nonzero while something is being timed (`paused_reaping`): a collect that lands inside a
# timing is measured as if it were the work. In a route race it was -- every sample that
# took 1.4-2.1 ms instead of 0.5-0.7 had a young collect and a reap inside it, and two such
# samples were enough to make a 30%-faster kernel's win unprovable, so the slower one stayed.
_reap_paused = [0]


class paused_reaping(object):
    """No collection while inside -- neither this allocation path's nor Python's own. What
    became due meanwhile is collected on the way out, outside whatever was being timed."""

    def __enter__(self):
        import gc
        self._gc = gc.isenabled()
        gc.disable()
        _reap_paused[0] += 1
        return self

    def __exit__(self, *exc):
        import gc
        _reap_paused[0] -= 1
        if self._gc:
            gc.enable()
        if not _reap_paused[0]:
            _maybe_reap()
        return False


def _maybe_reap():
    global _bytes_since_reap
    if _reap_paused[0] or _bytes_since_reap < _reap_budget:
        return
    # The cycles a burst of allocation leaves behind are young, and a collect of the two
    # younger generations finds them without walking the whole heap -- the full walk was
    # 13 ms on a loaded 27B and ran 412 times in one load (5.4 s). The full collect still
    # runs whenever the young one did not bring the ledger back within the allowance, so
    # the bound on dead memory between collects is the same as before.
    import gc

    gc.collect(1)
    try:
        held = get_platform().gpuBytes()[0]
    except Exception:
        held = None
    if held is not None and _live_floor and held <= _live_floor + _reap_budget:
        _bytes_since_reap = 0
        return
    reap_now()


def _pool_get(texture_shape: WebGPUArrayTextureShape) -> Optional[int]:
    global _pool_bytes
    if _capture_depth:
        reusable = _capture_free.get(texture_shape)
        if reusable:
            buffer_id = reusable.pop()
            _orphaned.pop(buffer_id)
            return buffer_id
    if len(_pool[texture_shape]) > 0:
        _pool_bytes -= texture_shape.byte_length
        return _pool[texture_shape].pop()
    return None


def capture_pin_stats():
    """What the pins and the reuse pool hold right now: (buffers, pinned bytes, pool bytes).

    Read at each recording, because the only thing that differs between two recordings of the
    SAME graph -- same kernels, same workgroups, same dispatch shapes -- is which buffers they
    ended up addressing. A reply whose per-token cost falls three-fold the moment the graph is
    re-recorded is a reply where that difference is the whole story, and it is not visible
    from anywhere else.
    """
    return len(_pinned_ids), sum(_pinned_ids.values()), _pool_bytes


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


def release_comm_buffer():
    """Drop the Python and JS views of the largest host staging array."""
    WebGPUBuffer._comm_buf = None
    get_platform().releaseCommBuf()


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
        # Construction can fail before either attribute is assigned (for example while a
        # browser benchmark is deliberately probing an unsupported allocation).  Python
        # still invokes ``__del__`` on that partially initialised object; never turn the
        # original failure into hundreds of noisy ignored AttributeErrors.
        texture_shape = getattr(self, "texture_shape", None)
        buffer_id = getattr(self, "buffer_id", None)
        if texture_shape is not None and buffer_id is not None:
            _pool_put(texture_shape, buffer_id)
        # get_platform().disposeBuffer(self.buffer_id)

    def set_data(self, array: np.ndarray):
        if self.size == 0:
            return
        storage_dtype = self.texture_shape.storage_dtype_numpy
        direct = (array.dtype == storage_dtype and array.flags.c_contiguous
                  and array.nbytes == self.texture_shape.byte_length)
        key = (self.texture_shape.byte_length,
               self.texture_shape.storage_dtype, array.dtype.str)
        choice = _upload_profiles.get(key, {}).get("choice") if direct else "staged"
        if choice == "direct":
            get_platform().setDataFromArray(self.buffer_id, array,
                                             self.texture_shape.byte_length)
        elif choice == "staged":
            buf = _get_comm_buf(self.texture_shape.byte_length)
            packed = buf.view(storage_dtype)
            packed[: array.size] = array.ravel()
            get_platform().setData(self.buffer_id, self.texture_shape.byte_length)
        else:
            def staged_upload():
                buf = _get_comm_buf(self.texture_shape.byte_length)
                packed = buf.view(storage_dtype)
                packed[: array.size] = array.ravel()
                get_platform().setData(self.buffer_id, self.texture_shape.byte_length)
            _upload_auto(key, staged_upload,
                         lambda: get_platform().setDataFromArray(
                             self.buffer_id, array, self.texture_shape.byte_length))
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

            get_platform().createMetaBuffer(self.buffer_id, data)

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
        buffer_id = getattr(self, "buffer_id", None)
        data = getattr(self, "_data", None)
        if buffer_id is None or data is None:
            return
        if buffer_id in _pinned_ids:
            return  # pinned by a capture — never recycle
        _meta_pool[data].append(buffer_id)


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


# A packer per dtype string, built once. numpy re-parses a structured dtype string on every
# `np.array(..., dtype="u4,i4,f4")` -- 27 regex matches a parse, 32 parses a decision
# request -- for what is a fixed little-endian layout of 4-byte fields.
_META_STRUCTS = {}
_META_CODES = {"u4": "I", "i4": "i", "f4": "f"}


def create_meta_buffer_from_structure(data_tuple: tuple, dtype) -> WebGPUMetaBuffer:
    """
    example: data_tuple = (2, 1.5), dtype = "i4,f4"
    """
    packer = _META_STRUCTS.get(dtype)
    if packer is None:
        fields = [f.strip() for f in dtype.split(",")] if isinstance(dtype, str) else None
        packer = (struct.Struct("<" + "".join(_META_CODES[f] for f in fields))
                  if fields and all(f in _META_CODES for f in fields) else False)
        _META_STRUCTS[dtype] = packer
    if packer:
        return create_meta_buffer(packer.pack(*data_tuple))
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
