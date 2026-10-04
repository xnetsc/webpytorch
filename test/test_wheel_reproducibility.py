"""Checked-in browser wheels must not change stamps on a same-source rebuild."""

from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ("webgpu", "webgl"))
def test_backend_wheel_uses_fixed_zip_timestamps(backend):
    wheel = ROOT / "dist" / f"wgpy_{backend}-1.0.0-py3-none-any.whl"
    with ZipFile(wheel) as archive:
        assert archive.infolist()
        assert {item.date_time for item in archive.infolist()} == {
            (2000, 1, 1, 0, 0, 0)
        }
