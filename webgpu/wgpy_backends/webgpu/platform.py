# platform call interface
import re

import numpy as np
from js import gpu  # Pyodide-dependent

# WebGPU caps a dispatch at 65535 workgroups PER DIMENSION. Every kernel here that walks a
# tensor one element per thread dispatches 1-D, so it stops fitting at 65535 * workgroup
# size -- 4.19M elements at the usual 64 -- and a model whose per-token work is tens of
# thousands of floats passes that as soon as the context is a few hundred tokens long. The
# device does not clamp it: the dispatch is rejected, the whole command buffer is invalidated,
# and every kernel batched behind it is dropped too, so the failure shows up as an answer
# made of garbage rather than as an error where the mistake was.
#
# Rather than teach thirty kernels to index themselves differently, a dispatch that does not
# fit is folded into a plane -- (x, 1, 1) becomes (x', 1, z) -- and ONE rewritten variant of
# that kernel is compiled which reads the fold back into the flat index it expects. The
# rewrite is mechanical and is checked at compile time by the driver like any other shader.
_DISPATCH_LIMIT = 65535


def _fold_source(source: str) -> str:
    """Rewrite a compute entry point so a folded dispatch still yields a flat index.

    `global_invocation_id.x` and `workgroup_id.x` are what these kernels index by, and both
    run out at the same place. Each is renamed, and a shadowing declaration puts the z plane
    back into x: for a workgroup id that is `+ z * num_workgroups.x`, and for a global id
    the same scaled by the workgroup size, which is exactly the flat id the unfolded
    dispatch would have produced.
    """
    wg = re.search(r"@workgroup_size\(\s*(\d+)", source)
    if not wg:
        raise ValueError("no @workgroup_size to fold against")
    wgsize = int(wg.group(1))
    ent = re.search(r"(@compute\b[\s\S]*?fn\s+\w+\s*\()([\s\S]*?)(\)\s*\{)", source)
    if not ent:
        raise ValueError("no compute entry point found")
    head, params, tail = ent.group(1), ent.group(2), ent.group(3)
    lets = []
    for builtin, scaled in (("global_invocation_id", True), ("workgroup_id", False)):
        hit = re.search(r"@builtin\(" + builtin + r"\)\s*(\w+)", params)
        if not hit:
            continue
        name = hit.group(1)
        # A kernel that already reads z means something by it, and folding would overwrite
        # that meaning. Refuse rather than silently return wrong numbers.
        if re.search(r"\b" + re.escape(name) + r"\.z\b", source):
            raise ValueError(builtin + " already reads .z")
        params = params.replace(hit.group(0), "@builtin(" + builtin + ") " + name + "_wtfold")
        plane = name + "_wtfold.z * wtfold_n.x" + (" * %du" % wgsize if scaled else "")
        lets.append("  let %s = vec3<u32>(%s_wtfold.x + %s, %s_wtfold.y, 0u);\n"
                    % (name, name, plane, name))
    if not lets:
        raise ValueError("entry point indexes by neither global_invocation_id nor workgroup_id")
    params = params.rstrip()
    if params and not params.endswith(","):
        params += ","
    params += " @builtin(num_workgroups) wtfold_n: vec3<u32>"
    return source[:ent.start()] + head + params + tail + "\n" + "".join(lets) + source[ent.end():]


class KernelUnsupported(RuntimeError):
    """A kernel this device cannot run: a feature, language feature or limit it needs is not
    there. Raised when the kernel is registered, before anything is dispatched -- a pipeline
    the device rejects otherwise fails later and asynchronously, and its dispatches write
    nothing. A caller choosing among implementations treats it as "not available here"."""


# What WebGPU guarantees every device. A limit the device did not report is taken to be
# this, never "unlimited": a kernel that needs more must be shown the device has more.
_GUARANTEED = {"maxStorageBuffers": 8, "maxWorkgroupStorage": 16384, "maxInvocations": 256,
               "maxWorkgroupSizeX": 256, "maxWorkgroupSizeY": 256, "maxWorkgroupSizeZ": 64}
_SCALAR_BYTES = {"f32": 4, "u32": 4, "i32": 4, "f16": 2, "bool": 4}


def _wgsl_bytes(t):
    """Size in bytes of a WGSL type as workgroup storage, or None if not one of the plain
    forms (scalars, vectors, matrices, atomics and fixed arrays of them)."""
    t = t.strip()
    if t in _SCALAR_BYTES:
        return _SCALAR_BYTES[t]
    m = re.match(r"atomic<\s*(\w+)\s*>$", t)
    if m:
        return 4
    m = re.match(r"vec([234])<\s*(\w+)\s*>$", t)
    if m and m.group(2) in _SCALAR_BYTES:
        n = int(m.group(1))
        return (4 if n == 3 else n) * _SCALAR_BYTES[m.group(2)]
    m = re.match(r"mat([234])x([234])<\s*(\w+)\s*>$", t)
    if m and m.group(3) in _SCALAR_BYTES:
        rows = int(m.group(2))
        return int(m.group(1)) * (4 if rows == 3 else rows) * _SCALAR_BYTES[m.group(3)]
    m = re.match(r"array<\s*(.+?)\s*,\s*(\d+)u?\s*>$", t)
    if m:
        inner = _wgsl_bytes(m.group(1))
        return None if inner is None else inner * int(m.group(2))
    return None


def kernel_requirements(source, binding_types):
    """What a WGSL compute kernel asks of the device, read from its source and bindings."""
    needs = {"f16": bool(re.search(r"^\s*enable\s+[^;]*\bf16\b", source, re.M)),
             "subgroups": bool(re.search(r"^\s*enable\s+[^;]*\bsubgroups\b", source, re.M)),
             "wgsl": [w.strip() for m in re.finditer(r"^\s*requires\s+([^;]+);", source, re.M)
                      for w in m.group(1).split(",") if w.strip()],
             "storage": sum(1 for b in binding_types or () if "storage" in str(b)),
             "workgroup_bytes": 0, "invocations": None, "size": None}
    for m in re.finditer(r"var<workgroup>\s*\w+\s*:\s*([^;]+);", source):
        b = _wgsl_bytes(m.group(1))
        if b is None:                       # a type this cannot size: do not guess
            needs["workgroup_bytes"] = None
            break
        needs["workgroup_bytes"] += (b + 15) // 16 * 16
    m = re.search(r"@workgroup_size\(([^)]*)\)", source)
    if m:
        try:
            dims = [int(d.strip().rstrip("u")) for d in m.group(1).split(",") if d.strip()]
        except ValueError:
            dims = None
        if dims:
            dims = (dims + [1, 1])[:3]
            needs["size"] = dims
            needs["invocations"] = dims[0] * dims[1] * dims[2]
    return needs


def unsupported_reason(source, binding_types, info):
    """Why this device cannot run the kernel, or None if it can."""
    need = kernel_requirements(source, binding_types)
    info = info or {}

    def limit(key):
        return int(info.get(key) or _GUARANTEED[key])
    if need["f16"] and not info.get("f16"):
        return "needs shader-f16, which this device does not have"
    if need["subgroups"] and not info.get("subgroups"):
        return "needs subgroups, which this device does not have"
    have = set(info.get("wgsl") or ())
    for w in need["wgsl"]:
        if w not in have:
            return "needs the WGSL language feature %s, which this browser does not offer" % w
    if need["storage"] > limit("maxStorageBuffers"):
        return "binds %d storage buffers, the device allows %d" % (
            need["storage"], limit("maxStorageBuffers"))
    if need["workgroup_bytes"] is not None and need["workgroup_bytes"] > limit("maxWorkgroupStorage"):
        return "uses %d bytes of workgroup memory, the device allows %d" % (
            need["workgroup_bytes"], limit("maxWorkgroupStorage"))
    if need["invocations"] is not None and need["invocations"] > limit("maxInvocations"):
        return "runs %d invocations a workgroup, the device allows %d" % (
            need["invocations"], limit("maxInvocations"))
    if need["size"] is not None:
        for d, key in zip(need["size"], ("maxWorkgroupSizeX", "maxWorkgroupSizeY",
                                          "maxWorkgroupSizeZ")):
            if d > limit(key):
                return "a workgroup dimension of %d exceeds the device's %d" % (d, limit(key))
    return None


class WebGPUPlatform:
    def __init__(self) -> None:
        self._latest_comm_buf = None
        self._kernels = {}          # name -> the descriptor it was added with
        self._folds = {}            # name -> folded variant's name, or None if it cannot be
        # What we have asked the GPU for and not given back. This is the one number about
        # GPU memory that is honest from inside a browser: the device's own utilisation and
        # footprint are not exposed to a page at all, but every buffer this backend holds
        # was allocated through the two calls below, so counting them is exact rather than
        # an estimate. Peak is kept because the interesting moment -- a model that only just
        # fits -- is over before anyone looks.
        self._gpu_bytes = 0
        self._gpu_peak = 0
        self._gpu_live = {}         # buffer_id -> its size, so a dispose subtracts the right amount

    def getDeviceInfo(self) -> dict:
        if getattr(self, "_device_info", None) is None:
            self._device_info = gpu.getDeviceInfo().to_py()
        return self._device_info

    def createBuffer(self, buffer_id: int, byte_length: int):
        self._gpu_note(buffer_id, byte_length)
        return gpu.createBuffer(buffer_id, byte_length)

    def createMetaBuffer(self, buffer_id: int, data: bytes):
        return gpu.createMetaBuffer(buffer_id, len(data), data)

    def disposeBuffer(self, buffer_id: int):
        self._gpu_note(buffer_id, None)
        return gpu.disposeBuffer(buffer_id)

    def _gpu_note(self, buffer_id, byte_length):
        """Add a buffer to the running total, or take it back out.

        Keyed by id and not just summed: buffers are pooled and reused, so the same id can
        be created again, and a dispose whose size had to be guessed would drift. An id
        that is disposed twice, or one we never saw created, changes nothing.
        """
        if byte_length is None:
            self._gpu_bytes -= self._gpu_live.pop(buffer_id, 0)
            return
        self._gpu_bytes += int(byte_length) - self._gpu_live.get(buffer_id, 0)
        self._gpu_live[buffer_id] = int(byte_length)
        if self._gpu_bytes > self._gpu_peak:
            self._gpu_peak = self._gpu_bytes

    def gpuBytes(self):
        """(held, peak, count) -- what this backend has out on the device right now."""
        return (self._gpu_bytes, self._gpu_peak, len(self._gpu_live))

    def setCommBuf(self, buffer: np.ndarray):
        self._latest_comm_buf = buffer
        return gpu.setCommBuf(buffer)

    def releaseCommBuf(self):
        self._latest_comm_buf = None
        return gpu.releaseCommBuf()

    def setData(self, buffer_id: int, byte_length: int):
        result = int(gpu.setData(buffer_id, byte_length))
        if result < 0:
            raise RuntimeError("WebGPU buffer upload failed on the browser main thread")
        if not result:
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            result = int(gpu.setData(buffer_id, byte_length))
            if result < 0:
                raise RuntimeError("WebGPU buffer upload failed on the browser main thread")
            if not result:
                raise ValueError("setData failed twice")

    def setDataFromArray(self, buffer_id: int, array: np.ndarray, byte_length: int):
        """Upload an exact, contiguous NumPy array without a Python staging copy."""
        result = int(gpu.setDataFromArray(buffer_id, array, byte_length))
        if result < 0:
            raise RuntimeError("WebGPU direct upload failed on the browser main thread")

    def getData(self, buffer_id: int, byte_length: int):
        result = int(gpu.getData(buffer_id, byte_length))
        if result < 0:
            raise RuntimeError("WebGPU buffer readback failed on the browser main thread")
        if not result:
            self.setCommBuf(self._latest_comm_buf)
            result = int(gpu.getData(buffer_id, byte_length))
            if result < 0:
                raise RuntimeError("WebGPU buffer readback failed on the browser main thread")
            if not result:
                raise ValueError("getData failed twice")

    def sampleLogits(self, buffer_id: int, byte_length: int, count: int, options: dict) -> int:
        """Select a token in JS from a GPU readback; Python handles only IDs/options."""
        return int(gpu.sampleLogits(buffer_id, byte_length, count, options))

    def routeHost(self, logits_id: int, logits_bytes: int, index_id: int,
                  weight_id: int, rows: int, experts: int, k: int, renormalize: bool):
        """Route a GPU logits buffer entirely in JS; Python passes buffer handles."""
        return gpu.routeHost(logits_id, logits_bytes, index_id, weight_id,
                             rows, experts, k, renormalize)

    def addKernel(self, name, descriptor):
        # Checked against what this device and browser actually offer before anything is
        # compiled: see `KernelUnsupported`.
        d = dict(descriptor)
        why = unsupported_reason(str(d.get("source", "")), d.get("bindingTypes"),
                                 self.getDeviceInfo())
        if why is not None:
            raise KernelUnsupported("%s %s" % (name, why))
        # Kept so a dispatch that turns out not to fit can be recompiled from the same
        # source. Nothing else reads this.
        self._kernels[name] = d
        return gpu.addKernel(name, descriptor)

    def runKernel(self, descriptor):
        wgs = descriptor.get("workGroups")
        if wgs is not None and int(wgs.get("x", 1) or 1) > _DISPATCH_LIMIT:
            descriptor = self._fold_dispatch(descriptor, wgs)
        WebGPUPlatform.dispatches += 1
        if WebGPUPlatform.count_names:
            _n = descriptor.get("name")
            WebGPUPlatform.by_name[_n] = WebGPUPlatform.by_name.get(_n, 0) + 1
        return gpu.runKernel(descriptor)

    def _fold_dispatch(self, descriptor, wgs):
        name = descriptor.get("name")
        x = int(wgs.get("x", 1) or 1)
        y = int(wgs.get("y", 1) or 1)
        z = int(wgs.get("z", 1) or 1)
        # z is where the fold goes, and y is left alone, so a dispatch already using either
        # has nowhere to put it. None do today; saying so beats folding one of them wrongly.
        if z != 1:
            raise ValueError(
                "kernel %r wants %d workgroups in x (limit %d) and already uses z=%d, "
                "so the dispatch cannot be folded" % (name, x, _DISPATCH_LIMIT, z))
        folded = self._folds.get(name, False)
        if folded is False:
            base = self._kernels.get(name)
            try:
                if base is None:
                    raise ValueError("kernel was never added through this platform")
                variant = dict(base)
                variant["source"] = _fold_source(base["source"])
                folded = name + "__fold"
                gpu.addKernel(folded, variant)
            except Exception as e:
                self._folds[name] = None
                raise ValueError(
                    "kernel %r wants %d workgroups in x, past the %d limit, and its source "
                    "could not be folded: %s" % (name, x, _DISPATCH_LIMIT, e)) from None
            self._folds[name] = folded
        if folded is None:
            raise ValueError("kernel %r wants %d workgroups in x, past the %d limit, and "
                             "cannot be folded" % (name, x, _DISPATCH_LIMIT))
        # Squared off rather than filling z with 65535-wide slabs: it keeps both dimensions
        # small, and the leftover threads -- at most one x row -- index past the end, which
        # is where every one of these kernels either returns early or has its write dropped.
        planes = (x + _DISPATCH_LIMIT - 1) // _DISPATCH_LIMIT
        per = (x + planes - 1) // planes
        out = dict(descriptor)
        out["name"] = folded
        out["workGroups"] = {"x": per, "y": y, "z": planes}
        return out

    # How many dispatches have been issued, ever. Sampled either side of a recording, it says
    # how long that recording's command list is -- and two recordings of the same graph that
    # do not agree on that are not the same graph, whatever the source says.
    dispatches = 0
    # The same count broken down by kernel. A total says two runs of one function differ; the
    # breakdown says WHICH commands the difference is, which is the question that follows.
    # Only while something asks for it: two dict operations per dispatch is nothing at a
    # recording and hundreds of times a token on a path that is not replaying one.
    by_name = {}
    count_names = False

    def beginCapture(self, name):
        # The name matters here: JS replaces the recorded command list for that name, so the
        # buffers the PREVIOUS recording pinned are no longer referenced by anything and must
        # stop being pinned. Without it every generation pinned a fresh set that was never
        # released until the model was.
        from wgpy_backends.webgpu.webgpu_buffer import begin_capture_pin
        # The worker->main channel is FIFO.  Replace the JS recording first, then send
        # disposals for orphaned ids from its predecessor; otherwise JS still refuses them.
        result = gpu.beginCapture(name)
        begin_capture_pin(name)
        return result

    def endCapture(self):
        from wgpy_backends.webgpu.webgpu_buffer import end_capture_pin
        end_capture_pin()
        return gpu.endCapture()

    def resetCaptures(self):
        """Drop every recorded capture graph and its pins, JS side included.
        JS must drop its pin set BEFORE Python disposes orphaned buffers;
        otherwise JS refuses those disposals.  This is also used after layer
        profiling while the model remains live, not only at model release."""
        from wgpy_backends.webgpu.webgpu_buffer import reset_capture_pins
        result = gpu.resetCaptures()
        reset_capture_pins()
        return result

    def releaseCapture(self, name):
        """Retire one cold graph without disabling recording for other shapes."""
        from wgpy_backends.webgpu.webgpu_buffer import release_capture_pin
        result = gpu.releaseCapture(name)
        release_capture_pin(name)
        return result

    def replay(self, name):
        return gpu.replay(name)

    def clearBuffer(self, buffer_id):
        """Zero a buffer where it lives, in command order. No host data crosses."""
        return gpu.clearBuffer(buffer_id)

    def replayStaged(self, name, buffer_id, byte_length, stage_slot, collect_slot):
        """One crossing per round of a pipelined loop: queue a replay of `name` (None for
        none) and a staged read of buffer `buffer_id`'s first `byte_length` bytes into
        `stage_slot` (-1 for none), then wait for the read staged earlier in `collect_slot`
        (-1 for none) and return its bytes (a JS Uint8Array; `.to_bytes()`)."""
        return gpu.replayStaged(name, buffer_id, byte_length, stage_slot, collect_slot)


_instance = None


def get_platform() -> WebGPUPlatform:
    global _instance
    if _instance is None:
        _instance = WebGPUPlatform()
    return _instance
