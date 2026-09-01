#!/usr/bin/env python3
"""Rebuild the checked-in backend wheels from the source directories beside them.

dist/wgpy_webgpu-...whl and dist/wgpy_webgl-...whl are build artifacts kept in the repo,
because the page installs them with micropip and a static host has nothing to build with.
There is no packaging step to run; this is it. Each wheel's entries are replaced with the
matching file under webgpu/ or webgl/, RECORD is recomputed, and the archive is rewritten
in place -- entry order and metadata are preserved, so a wheel whose sources have not
changed comes out byte-identical and shows as no diff.

    python3 scripts/pack-wheels.py            # rebuild both, report what changed
    python3 scripts/pack-wheels.py --check    # report only, change nothing (exit 1 if stale)
"""
import base64
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEELS = {
    'dist/wgpy_webgpu-1.0.0-py3-none-any.whl': 'webgpu',
    'dist/wgpy_webgl-1.0.0-py3-none-any.whl': 'webgl',
}


def record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
    return '%s,sha256=%s,%d' % (name, digest, len(data))


def rebuild(wheel, root, check):
    path = os.path.join(HERE, wheel)
    src_root = os.path.join(HERE, root)
    zin = zipfile.ZipFile(path)
    record_name = next((n for n in zin.namelist() if n.endswith('.dist-info/RECORD')), None)

    items, changed = [], []
    for info in zin.infolist():
        data = zin.read(info.filename)
        src = os.path.join(src_root, info.filename)
        # Only files the repo actually keeps a source for. Anything else -- the vendored
        # cupy shims, the metadata -- is carried through untouched.
        if info.filename != record_name and os.path.isfile(src):
            with open(src, 'rb') as f:
                fresh = f.read()
            if fresh != data:
                changed.append(info.filename)
                data = fresh
        items.append((info, data))
    zin.close()

    if not changed:
        print('%s: up to date' % wheel)
        return False
    print('%s: %d file(s) differ' % (wheel, len(changed)))
    for name in changed:
        print('    ' + name)
    if check:
        return True

    if record_name is not None:
        lines = [record_line(i.filename, d) for i, d in items if i.filename != record_name]
        lines.append('%s,,' % record_name)
        record = ('\n'.join(lines) + '\n').encode()
        items = [(i, record if i.filename == record_name else d) for i, d in items]

    # Written beside the target and moved into place, so an interrupted run cannot leave a
    # half-written wheel where a working one was.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.whl')
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
            for info, data in items:
                zout.writestr(info, data)
        zipfile.ZipFile(tmp).close()          # refuses to move a wheel it cannot read back
        shutil.move(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    print('    repacked')
    return True


def main():
    check = '--check' in sys.argv[1:]
    stale = False
    for wheel, root in WHEELS.items():
        stale |= rebuild(wheel, root, check)
    if check and stale:
        sys.stderr.write('\nwheels are stale: run python3 scripts/pack-wheels.py\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
