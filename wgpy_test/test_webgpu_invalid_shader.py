import pytest
import wgpy as cp


def test_invalid_wgsl_fails_explicitly_instead_of_returning_zero_tensor():
    from wgpy_backends.webgpu.platform import get_platform

    plat = get_platform()
    plat.addKernel("invalid_wgsl_must_fail", {
        "source": "@compute @workgroup_size(1) fn main() { this is not WGSL; }",
        "bindingTypes": [],
    })
    plat.runKernel({"name": "invalid_wgsl_must_fail", "tensors": [],
                    "workGroups": {"x": 1, "y": 1, "z": 1}})
    with pytest.raises(Exception, match="invalid_wgsl_must_fail"):
        cp.asnumpy(cp.zeros((4,), dtype="float32"))
