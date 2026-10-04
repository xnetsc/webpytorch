from collections import defaultdict
from typing import Optional
from time import perf_counter
import numpy as np
from wgpy_backends.webgl.texture import (
    WebGL2RenderingContext,
    WebGLArrayTextureShape,
    get_default_texture_shape,
)
from wgpy_backends.webgl.shader_util import (
    header,
    native_pixel_type_for_internal_format,
)
from wgpy_backends.webgl.platform import get_platform
import wgpy_backends.webgl.webgl_config as webgl_config


_upload_profiles = {}


def _upload_auto(key, staged, direct):
    """Measure repeated physical shapes and use the faster upload path.

    Natural calls alternate paths with one upload per call, so calibration
    never makes an extra copy. The faster median wins without a benefit gate.
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
    byte_length = key.element_count * texture_type_to_element_itemsize[key.type]
    trials = 7 if byte_length < 1024 * 1024 else 3
    if len(profile["direct"]) >= trials and len(profile["staged"]) >= trials:
        profile["choice"] = (
            "direct" if sorted(profile["direct"])[trials // 2]
            < sorted(profile["staged"])[trials // 2] else "staged")

performance_metrics = {
    "webgl.buffer.create": 0,
    "webgl.buffer.delete": 0,
    "webgl.buffer.write_count": 0,
    "webgl.buffer.write_size": 0,
    "webgl.buffer.write_scalar_count": 0,
    "webgl.buffer.read_count": 0,
    "webgl.buffer.read_size": 0,
    "webgl.buffer.read_scalar_count": 0,
    "webgl.buffer.buffer_count": 0,
    "webgl.buffer.buffer_count_max": 0,
    "webgl.buffer.buffer_size": 0,
    "webgl.buffer.buffer_size_max": 0,
}

texture_type_to_element_itemsize = {
    WebGL2RenderingContext.FLOAT: 4,
    WebGL2RenderingContext.HALF_FLOAT: 2,
    WebGL2RenderingContext.INT: 4,
    WebGL2RenderingContext.UNSIGNED_BYTE: 1,
}


def get_dtype_js_ctor_type(dtype):
    dtype = np.dtype(dtype)
    return {
        np.dtype(np.float32): "Float32Array",
        np.dtype(np.int32): "Int32Array",
        np.dtype(np.uint16): "Uint16Array",
        np.dtype(np.uint8): "Uint8Array",
    }[dtype]


_pool = defaultdict(list)

added_kernels = set()

# Graph capture: keep buffer ids stable (don't recycle) while a capture is being
# recorded, so the JS-side replay references the same ids. Python holds ids only.
# id -> byte size, so a release can destroy them AND keep the size accounting
# straight.
_capture_depth = 0
_pinned_ids = {}
_capture_name = None
_pins = {}
_orphaned = {}
_POOL_PER_SHAPE = 4
_POOL_MAX_BYTES = 512 * 1024 * 1024
_pool_bytes = 0


def begin_capture_pin(name=None):
    global _capture_depth, _capture_name
    if _capture_depth == 0:
        key = name if name is not None else "?"
        if key in _pins:
            release_capture_pin(key)
        _pins[key] = {}
        _capture_name = key
    _capture_depth += 1


def release_capture_pin(name):
    """Unpin one retired graph and free only its no-longer-owned buffers."""
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
            performance_metrics["webgl.buffer.delete"] += 1
            performance_metrics["webgl.buffer.buffer_count"] -= 1
            performance_metrics["webgl.buffer.buffer_size"] -= _texture_shape_byte_size(shape)


def end_capture_pin():
    global _capture_depth
    if _capture_depth > 0:
        _capture_depth -= 1


def reset_capture_pins():
    """Abandon any pin state (a model is being released; captures go with it).
    Does NOT dispose anything — the ids still need to be read off first."""
    global _capture_depth
    _capture_depth = 0
    _pins.clear()
    _orphaned.clear()


def _maybe_pin(buffer_id: int, byte_size: int):
    if _capture_depth > 0:
        _pinned_ids[buffer_id] = byte_size
        if _capture_name is not None:
            _pins.setdefault(_capture_name, {})[buffer_id] = byte_size


def _pool_put(texture_shape: WebGLArrayTextureShape, buffer_id: int):
    global _pool_bytes
    if buffer_id in _pinned_ids:
        _orphaned[buffer_id] = texture_shape
        return  # pinned by a capture — never recycle
    ids = _pool[texture_shape]
    byte_size = _texture_shape_byte_size(texture_shape)
    if len(ids) >= _POOL_PER_SHAPE or _pool_bytes + byte_size > _POOL_MAX_BYTES:
        get_platform().disposeBuffer(buffer_id)
        performance_metrics["webgl.buffer.delete"] += 1
        performance_metrics["webgl.buffer.buffer_count"] -= 1
        performance_metrics["webgl.buffer.buffer_size"] -= byte_size
        return
    ids.append(buffer_id)
    _pool_bytes += byte_size


def _pool_get(texture_shape: WebGLArrayTextureShape) -> Optional[int]:
    global _pool_bytes
    if len(_pool[texture_shape]) > 0:
        _pool_bytes -= _texture_shape_byte_size(texture_shape)
        return _pool[texture_shape].pop()
    return None


def _texture_shape_byte_size(texture_shape: WebGLArrayTextureShape) -> int:
    return (
        texture_shape.element_count
        * texture_type_to_element_itemsize[texture_shape.type]
    )


def release_capture_buffers():
    """Destroy every buffer a recorded capture pinned.

    Pinned buffers never enter the reuse pool when their Python object dies —
    `__del__` drops them — so without this they stay allocated on the GPU
    forever, and JS refuses their disposeBuffer while pinned too. Called at
    model release, after the JS side has been told to drop its captures and
    pins (so the disposeBuffer messages actually land)."""
    plat = get_platform()
    for buffer_id, byte_size in list(_pinned_ids.items()):
        plat.disposeBuffer(buffer_id)
        performance_metrics["webgl.buffer.delete"] += 1
        performance_metrics["webgl.buffer.buffer_count"] -= 1
        performance_metrics["webgl.buffer.buffer_size"] -= byte_size
    _pinned_ids.clear()


def release_pooled_buffers():
    """Destroy everything the reuse pool holds and empty it.

    The pool exists to skip createBuffer when the next tensor has a matching
    shape. When a model is released and a DIFFERENT one loads, most shapes
    will not match, so pooled buffers would just sit on the GPU next to the
    new model's allocations until the device runs out. Called at model
    release, after `release_capture_buffers`."""
    plat = get_platform()
    for texture_shape, ids in list(_pool.items()):
        byte_size = _texture_shape_byte_size(texture_shape)
        for buffer_id in ids:
            plat.disposeBuffer(buffer_id)
            performance_metrics["webgl.buffer.delete"] += 1
            performance_metrics["webgl.buffer.buffer_count"] -= 1
            performance_metrics["webgl.buffer.buffer_size"] -= byte_size
    _pool.clear()
    global _pool_bytes
    _pool_bytes = 0


def release_comm_buffer():
    """Drop the Python and JS views of the largest host staging array."""
    WebGLBuffer._comm_buf = None
    get_platform().releaseCommBuf()


class WebGLBuffer:
    buffer_id: int
    size: int  # Logical number of elements (May differ from the number of elements in the texture; for RGBA textures, 1 pixel corresponds to 4 elements)
    dtype: (
        np.dtype
    )  # logical type (may be different from physical representation in WebGL)
    texture_shape: WebGLArrayTextureShape

    _comm_buf: Optional[np.ndarray] = None
    next_id = 1

    def __init__(
        self,
        size: int,
        dtype: np.dtype,
        texture_shape: Optional[WebGLArrayTextureShape] = None,
    ) -> None:
        self.size = size
        self.dtype = dtype
        self.texture_shape = texture_shape or get_default_texture_shape(size, dtype)
        pooled_buffer_id = _pool_get(self.texture_shape)
        if pooled_buffer_id is not None:
            self.buffer_id = pooled_buffer_id
        else:
            self.buffer_id = WebGLBuffer.next_id
            WebGLBuffer.next_id += 1
            get_platform().createBuffer(self.buffer_id, self.texture_shape.to_json())
            performance_metrics["webgl.buffer.create"] += 1
            performance_metrics["webgl.buffer.buffer_count"] += 1
            performance_metrics["webgl.buffer.buffer_size"] += (
                self.texture_shape.element_count
                * texture_type_to_element_itemsize[self.texture_shape.type]
            )
            performance_metrics["webgl.buffer.buffer_count_max"] = max(
                performance_metrics["webgl.buffer.buffer_count_max"],
                performance_metrics["webgl.buffer.buffer_count"],
            )
            performance_metrics["webgl.buffer.buffer_size_max"] = max(
                performance_metrics["webgl.buffer.buffer_size_max"],
                performance_metrics["webgl.buffer.buffer_size"],
            )
        _maybe_pin(
            self.buffer_id,
            self.texture_shape.element_count
            * texture_type_to_element_itemsize[self.texture_shape.type],
        )

    def __del__(self):
        texture_shape = getattr(self, "texture_shape", None)
        buffer_id = getattr(self, "buffer_id", None)
        if texture_shape is not None and buffer_id is not None:
            _pool_put(texture_shape, buffer_id)
        # get_platform().disposeBuffer(self.buffer_id)

    def _get_comm_buf(self, byte_size: int) -> np.ndarray:
        if WebGLBuffer._comm_buf is None or WebGLBuffer._comm_buf.size < byte_size:
            WebGLBuffer._comm_buf = np.empty(
                (max(byte_size, 1024 * 1024),), dtype=np.uint8
            )
            get_platform().setCommBuf(WebGLBuffer._comm_buf)
        return WebGLBuffer._comm_buf

    def set_data(self, array: np.ndarray):
        if self.texture_shape.type == WebGL2RenderingContext.HALF_FLOAT:
            array_f16 = array.astype(np.float16, copy=False).ravel()
            size = self.texture_shape.element_count
            dtype = np.uint16
            eligible = array_f16.size == size
            choice = (_upload_profiles.get(self.texture_shape, {}).get("choice")
                      if eligible else "staged")
            if choice == "direct":
                get_platform().setDataFromArray(
                    self.buffer_id, array_f16.view(np.uint16),
                    get_dtype_js_ctor_type(dtype), size * np.dtype(dtype).itemsize)
            elif choice == "staged":
                buf = self._get_comm_buf(np.dtype(dtype).itemsize * size)
                packed = buf.view(np.float16)
                packed[: array.size] = array_f16
                get_platform().setData(self.buffer_id, get_dtype_js_ctor_type(dtype), size)
            else:
                # The float16 conversion is needed for this WebGL storage format;
                # do not copy its result once more into the reusable WASM arena.
                def staged_upload():
                    buf = self._get_comm_buf(np.dtype(dtype).itemsize * size)
                    packed = buf.view(np.float16)
                    packed[: array.size] = array_f16
                    get_platform().setData(self.buffer_id, get_dtype_js_ctor_type(dtype), size)
                _upload_auto(self.texture_shape, staged_upload,
                             lambda: get_platform().setDataFromArray(
                                 self.buffer_id, array_f16.view(np.uint16),
                                 get_dtype_js_ctor_type(dtype),
                                 size * np.dtype(dtype).itemsize))
        else:
            dtype = {
                WebGL2RenderingContext.FLOAT: np.float32,
                WebGL2RenderingContext.INT: np.int32,
                WebGL2RenderingContext.UNSIGNED_BYTE: np.uint8,
            }[self.texture_shape.type]
            size = self.texture_shape.element_count
            eligible = (array.dtype == np.dtype(dtype) and array.flags.c_contiguous
                        and array.size == size)
            choice = (_upload_profiles.get(self.texture_shape, {}).get("choice")
                      if eligible else "staged")
            if choice == "direct":
                get_platform().setDataFromArray(
                    self.buffer_id, array, get_dtype_js_ctor_type(dtype),
                    size * np.dtype(dtype).itemsize)
            elif choice == "staged":
                buf = self._get_comm_buf(np.dtype(dtype).itemsize * size)
                packed = buf.view(dtype)
                packed[: array.size] = array.ravel()
                get_platform().setData(self.buffer_id, get_dtype_js_ctor_type(dtype), size)
            else:
                def staged_upload():
                    buf = self._get_comm_buf(np.dtype(dtype).itemsize * size)
                    packed = buf.view(dtype)
                    packed[: array.size] = array.ravel()
                    get_platform().setData(self.buffer_id, get_dtype_js_ctor_type(dtype), size)
                _upload_auto(self.texture_shape, staged_upload,
                             lambda: get_platform().setDataFromArray(
                                 self.buffer_id, array, get_dtype_js_ctor_type(dtype),
                                 size * np.dtype(dtype).itemsize))
        performance_metrics["webgl.buffer.write_count"] += 1
        # physical size
        performance_metrics["webgl.buffer.write_size"] += (
            size * np.dtype(dtype).itemsize
        )
        # logical size
        if array.size <= 1:
            performance_metrics["webgl.buffer.write_scalar_count"] += 1

    def get_data(self) -> np.ndarray:
        # TODO Sorting out dtype, whether it is a WebGL internal representation or ndarray dtype.
        copied = self._copy_to_rgba_if_needed()
        if copied is not None:
            return copied._get_data_internal(
                self.texture_shape.elements_per_pixel == 1, self.dtype
            )
        else:
            return self._get_data_internal(False, self.dtype)

    def _get_data_internal(self, extract_r_from_rgba: bool, original_dtype: np.dtype):
        performance_metrics["webgl.buffer.read_count"] += 1
        if self.texture_shape.type == WebGL2RenderingContext.HALF_FLOAT:
            buf = self._get_comm_buf(
                np.dtype(np.uint16).itemsize * self.texture_shape.element_count
            )
            get_platform().getData(
                self.buffer_id,
                get_dtype_js_ctor_type(np.uint16),
                self.texture_shape.element_count,
            )
            performance_metrics["webgl.buffer.read_size"] += (
                self.texture_shape.element_count * np.dtype(np.uint16).itemsize
            )
            if self.size <= 1:
                performance_metrics["webgl.buffer.read_scalar_count"] += 1
            view = buf.view(np.float16)[: self.size]
            if extract_r_from_rgba:
                view = view[::4]

            return view.astype(np.float32).astype(original_dtype, copy=False)
        else:
            dtype = {
                WebGL2RenderingContext.FLOAT: np.float32,
                WebGL2RenderingContext.INT: np.int32,
                WebGL2RenderingContext.UNSIGNED_BYTE: np.uint8,
            }[self.texture_shape.type]
            buf = self._get_comm_buf(
                np.dtype(dtype).itemsize * self.texture_shape.element_count
            )
            get_platform().getData(
                self.buffer_id,
                get_dtype_js_ctor_type(dtype),
                self.texture_shape.element_count,
            )
            performance_metrics["webgl.buffer.read_size"] += (
                self.texture_shape.element_count * np.dtype(np.uint16).itemsize
            )
            if self.size <= 1:
                performance_metrics["webgl.buffer.read_scalar_count"] += 1
            view = buf.view(dtype)[: self.size]
            if extract_r_from_rgba:
                view = view[::4]

            return view.copy().astype(original_dtype, copy=False)

    def _copy_to_rgba_if_needed(self) -> Optional["WebGLBuffer"]:
        is_32bit = self.texture_shape.type in [
            WebGL2RenderingContext.FLOAT,
            WebGL2RenderingContext.INT,
        ]
        is_rch = self.texture_shape.elements_per_pixel == 1
        ok = True
        if not webgl_config.can_read_r_texture() and is_rch:
            ok = False
        if not webgl_config.can_read_non_32bit_texture() and not is_32bit:
            ok = False
        if ok:
            # no copy needed
            return None

        ss = self.texture_shape  # source shape
        t_internal_format = {
            WebGL2RenderingContext.R16F: WebGL2RenderingContext.RGBA32F,  # some env does not support reading RGBA16F
            WebGL2RenderingContext.R32F: WebGL2RenderingContext.RGBA32F,
            WebGL2RenderingContext.R32I: WebGL2RenderingContext.RGBA32I,
            WebGL2RenderingContext.R8UI: WebGL2RenderingContext.RGBA32I,  # some env does not support reading 8UI
            WebGL2RenderingContext.RGBA16F: WebGL2RenderingContext.RGBA32F,  # some env does not support reading RGBA16F
            WebGL2RenderingContext.RGBA32F: WebGL2RenderingContext.RGBA32F,
            WebGL2RenderingContext.RGBA32I: WebGL2RenderingContext.RGBA32I,
            WebGL2RenderingContext.RGBA8UI: WebGL2RenderingContext.RGBA32I,  # some env does not support reading 8UI
        }[ss.internal_format]
        t_format = {
            WebGL2RenderingContext.RED: WebGL2RenderingContext.RGBA,
            WebGL2RenderingContext.RED_INTEGER: WebGL2RenderingContext.RGBA_INTEGER,
            WebGL2RenderingContext.RGBA: WebGL2RenderingContext.RGBA,
            WebGL2RenderingContext.RGBA_INTEGER: WebGL2RenderingContext.RGBA_INTEGER,
        }[ss.format]
        t_type = {
            WebGL2RenderingContext.FLOAT: WebGL2RenderingContext.FLOAT,
            WebGL2RenderingContext.HALF_FLOAT: WebGL2RenderingContext.FLOAT,
            WebGL2RenderingContext.INT: WebGL2RenderingContext.INT,
            WebGL2RenderingContext.UNSIGNED_BYTE: WebGL2RenderingContext.INT,  # 8UI -> 32I
        }[ss.type]
        target_shape = WebGLArrayTextureShape(
            height=ss.height,
            width=ss.width,
            depth=ss.depth,
            dim=ss.dim,
            internal_format=t_internal_format,
            format=t_format,
            type=t_type,
        )
        target_dtype = {
            np.dtype(np.float32): np.dtype(np.float32),
            np.dtype(np.int32): np.dtype(np.int32),
            np.dtype(np.uint8): np.dtype(np.int32),
            np.dtype(np.bool_): np.dtype(np.int32),
        }[self.dtype]
        # always get vec4 even it only contains R channel
        native_pixel_type_src = {
            WebGL2RenderingContext.R32F: "vec4",
            WebGL2RenderingContext.R16F: "vec4",
            WebGL2RenderingContext.R32I: "ivec4",
            WebGL2RenderingContext.R8UI: "uvec4",
            WebGL2RenderingContext.RGBA32F: "vec4",
            WebGL2RenderingContext.RGBA16F: "vec4",
            WebGL2RenderingContext.RGBA32I: "ivec4",
            WebGL2RenderingContext.RGBA8UI: "uvec4",
        }[ss.internal_format]
        native_pixel_type_dst = native_pixel_type_for_internal_format[
            target_shape.internal_format
        ]

        sampler_type = {
            WebGL2RenderingContext.FLOAT: "sampler2D",
            WebGL2RenderingContext.HALF_FLOAT: "sampler2D",
            WebGL2RenderingContext.INT: "isampler2D",
            WebGL2RenderingContext.UNSIGNED_BYTE: "usampler2D",
        }[ss.type]
        if ss.dim == "2DArray":
            sampler_type += "Array"

        new_size = (
            self.size if self.texture_shape.elements_per_pixel == 4 else self.size * 4
        )
        target_buffer = WebGLBuffer(
            size=new_size, dtype=target_dtype, texture_shape=target_shape
        )

        kernel_source = None
        if target_shape.dim == "2D":
            kernel_name = (
                f"copy_r_to_rgba_2d_{native_pixel_type_src}_{native_pixel_type_dst}"
            )
            if kernel_name not in added_kernels:
                kernel_source = f"""{header}
uniform {sampler_type} _src_r;
out {native_pixel_type_dst} _out_color;
void main() {{
{native_pixel_type_src} v = texelFetch(_src_r, ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y)), 0);
_out_color = {native_pixel_type_dst}(v);
}}
"""
        else:
            kernel_name = f"copy_r_to_rgba_2darray_{native_pixel_type_src}_{native_pixel_type_dst}"
            if kernel_name not in added_kernels:
                kernel_source = f"""{header}
uniform {sampler_type} _src_r;
out {native_pixel_type_dst} _out_color;
uniform int _draw_depth;
void main() {{
{native_pixel_type_src} v = texelFetch(_src_r, ivec3(int(gl_FragCoord.x), int(gl_FragCoord.y), _draw_depth), 0);
_out_color = {native_pixel_type_dst}(v);
}}
"""
        if kernel_source is not None:
            get_platform().addKernel(kernel_name, {"source": kernel_source})
            added_kernels.add(kernel_name)

        get_platform().runKernel(
            {
                "name": kernel_name,
                "inputs": [{"name": "_src_r", "id": self.buffer_id}],
                "output": target_buffer.buffer_id,
                "uniforms": [],
            }
        )
        return target_buffer
