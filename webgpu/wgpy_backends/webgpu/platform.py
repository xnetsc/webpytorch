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
        return gpu.getDeviceInfo().to_py()

    def createBuffer(self, buffer_id: int, byte_length: int):
        self._gpu_note(buffer_id, byte_length)
        return gpu.createBuffer(buffer_id, byte_length)

    def createMetaBuffer(self, buffer_id: int, byte_length: int):
        return gpu.createMetaBuffer(buffer_id, byte_length)

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

    def setData(self, buffer_id: int, byte_length: int):
        if not gpu.setData(buffer_id, byte_length):
            # WASM buffer may reallocated
            self.setCommBuf(self._latest_comm_buf)
            if not gpu.setData(buffer_id, byte_length):
                raise ValueError("setData failed twice")

    def getData(self, buffer_id: int, byte_length: int):
        if not gpu.getData(buffer_id, byte_length):
            self.setCommBuf(self._latest_comm_buf)
            if not gpu.getData(buffer_id, byte_length):
                raise ValueError("getData failed twice")

    def addKernel(self, name, descriptor):
        # Kept so a dispatch that turns out not to fit can be recompiled from the same
        # source. Nothing else reads this.
        self._kernels[name] = dict(descriptor)
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
        begin_capture_pin(name)
        return gpu.beginCapture(name)

    def endCapture(self):
        from wgpy_backends.webgpu.webgpu_buffer import end_capture_pin
        end_capture_pin()
        return gpu.endCapture()

    def resetCaptures(self):
        """Drop every recorded capture graph and its pins, JS side included.
        Sent at model release, BEFORE the buffered disposeBuffer messages, so
        those are no longer refused by the JS-side pin set."""
        from wgpy_backends.webgpu.webgpu_buffer import reset_capture_pins
        reset_capture_pins()
        return gpu.resetCaptures()

    def replay(self, name):
        return gpu.replay(name)


_instance = None


def get_platform() -> WebGPUPlatform:
    global _instance
    if _instance is None:
        _instance = WebGPUPlatform()
    return _instance
