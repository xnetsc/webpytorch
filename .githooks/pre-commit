#!/usr/bin/env python3
"""Refuse a commit whose wheels do not contain the sources committed beside them.

The two backend wheels in dist/ are BUILT ARTIFACTS of webgpu/ and webgl/, and nothing in
the repo rebuilds them: they are checked in, byte for byte, and the page installs them with
micropip. So editing webgpu/wgpy_backends/... and committing is not shipping anything --
the tree says one thing, the artifact the browser actually runs says another, and the two
disagree silently. Only someone reading the installed file in a running page would notice,
which is how it was noticed.

This is a `pre-commit` hook: git runs it before writing the commit, and a non-zero exit
stops it. It compares the STAGED source against what is inside the STAGED wheel, so it
judges the commit being made rather than the working tree.

Enable it with:  git config core.hooksPath .githooks
Bypass it with:  git commit --no-verify   (when the mismatch is deliberate)
"""
import io
import subprocess
import sys
import zipfile

# dist/<wheel>  <-  the directory in the repo whose tree it packages
WHEELS = {
    'dist/wgpy_webgpu-1.0.0-py3-none-any.whl': 'webgpu/',
    'dist/wgpy_webgl-1.0.0-py3-none-any.whl': 'webgl/',
}


def staged(path):
    """The bytes of `path` as this commit will record them, or None if it has none."""
    try:
        return subprocess.run(['git', 'show', ':' + path], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
    except subprocess.CalledProcessError:
        return None


def main():
    names = subprocess.run(['git', 'diff', '--cached', '--name-only'],
                           check=True, stdout=subprocess.PIPE, text=True).stdout.split()
    problems = []
    for wheel, root in WHEELS.items():
        # Only worth checking when this commit touches one side or the other.
        touched = [n for n in names if n == wheel or n.startswith(root)]
        if not touched:
            continue
        blob = staged(wheel)
        if blob is None:
            continue                       # the wheel is being deleted, or was never here
        try:
            z = zipfile.ZipFile(io.BytesIO(blob))
        except zipfile.BadZipFile:
            problems.append('%s is not a readable wheel' % wheel)
            continue
        inside = set(z.namelist())
        for name in names:
            if not name.startswith(root) or not name.endswith('.py'):
                continue
            entry = name[len(root):]       # webgpu/wgpy_backends/x.py -> wgpy_backends/x.py
            if entry not in inside:
                continue                   # a source the wheel does not package: not its job
            src = staged(name)
            if src is None:
                continue                   # being deleted
            if z.read(entry) != src:
                problems.append('%s does not match %s inside %s' % (name, entry, wheel))

    if not problems:
        return 0
    sys.stderr.write('\ncommit refused: a wheel does not contain the source beside it\n\n')
    for p in problems:
        sys.stderr.write('  ' + p + '\n')
    sys.stderr.write("""
The wheels in dist/ are checked-in build artifacts; the browser installs them, not the
directories. Repack the changed files into the wheel and stage it, then commit again:

  python3 scripts/pack-wheels.py && git add dist/*.whl

Deliberate mismatch?  git commit --no-verify
""")
    return 1


if __name__ == '__main__':
    sys.exit(main())
