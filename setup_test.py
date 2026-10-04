import os
import tempfile
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

with tempfile.TemporaryDirectory(prefix="wgpy-test-build-") as tmp:
    os.chdir(tmp)
    setup(
        name="wgpy_test",
        version="1.0.0",
        description="browser-side WgPy test package",
        packages=find_packages(where=str(ROOT), include=["wgpy_test"]),
        package_dir={"": str(ROOT)},
        options={
            "bdist_wheel": {"dist_dir": str(ROOT / "dist")},
            "build": {"build_base": str(Path(tmp) / "build")},
        },
    )
