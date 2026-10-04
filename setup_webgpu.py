import os
import tempfile
from pathlib import Path

# Wheel bytes participate in the browser SDK cache key. Without a stable ZIP
# timestamp, rebuilding identical source needlessly invalidates model tuning.
os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")

from setuptools import setup

ROOT = Path(__file__).resolve().parent

# This repository's pyproject describes the public `webtorch` wheel.  Backend wheels are
# deliberately separate Pyodide packages; running setup() from the repository root makes
# modern setuptools merge that unrelated [project] table and silently emit webtorch instead
# of wgpy-webgpu.  Build from a clean temporary metadata root while reading package sources
# by absolute path, so the checked-in command produces the wheel it names.
with tempfile.TemporaryDirectory(prefix="wgpy-webgpu-build-") as tmp:
    os.chdir(tmp)
    setup(
        name="wgpy-webgpu",
        version="1.0.0",
        install_requires=["numpy"],
        description="cupy-like GPU linear algebra module on WebGPU",
        packages=[
            "cupy",
            "cupy.cuda",
            "cupyx",
            "cupyx.scipy",
            "cupy_backends",
            "cupy_backends.runtime",
            "cupy_backends.webgpu",
            "wgpy",
            "wgpy.common",
            "wgpy_backends",
            "wgpy_backends.runtime",
            "wgpy_backends.webgpu",
        ],
        package_dir={
            "": str(ROOT),
            "cupy_backends": str(ROOT / "webgpu/cupy_backends"),
            "wgpy_backends": str(ROOT / "webgpu/wgpy_backends"),
        },
        options={
            "bdist_wheel": {"dist_dir": str(ROOT / "dist")},
            "build": {"build_base": str(Path(tmp) / "build")},
        },
    )
