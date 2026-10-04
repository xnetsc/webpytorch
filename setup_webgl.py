import os
import tempfile
from pathlib import Path

# Keep same-source backend wheels byte-identical for the shared SDK stamp.
os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")

from setuptools import setup

ROOT = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory(prefix="wgpy-webgl-build-") as tmp:
    os.chdir(tmp)
    setup(
        name="wgpy-webgl",
        version="1.0.0",
        install_requires=["numpy"],
        description="cupy-like GPU linear algebra module on WebGL",
        packages=[
            "cupy",
            "cupy.cuda",
            "cupyx",
            "cupyx.scipy",
            "cupy_backends",
            "cupy_backends.runtime",
            "cupy_backends.webgl",
            "wgpy",
            "wgpy.common",
            "wgpy_backends",
            "wgpy_backends.runtime",
            "wgpy_backends.webgl",
        ],
        package_dir={
            "": str(ROOT),
            "cupy_backends": str(ROOT / "webgl/cupy_backends"),
            "wgpy_backends": str(ROOT / "webgl/wgpy_backends"),
        },
        options={
            "bdist_wheel": {"dist_dir": str(ROOT / "dist")},
            "build": {"build_base": str(Path(tmp) / "build")},
        },
    )
