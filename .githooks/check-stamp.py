#!/usr/bin/env python3
"""Refuse a commit that changes a served script without moving its cache-busting stamp.

`scripts/stamp.sh` writes a content hash into every locally-served script URL, because a
browser's own memory cache answers a `<script src>` on a reload without consulting the
service worker: change the bytes and leave the URL alone, and the page keeps running the
old file until a SECOND reload. The failure is silent -- the page works, it is just not the
page that was committed -- and it costs an entire measurement to notice, because the code
under test is not the code running.

The stamp only helps if it is actually re-run. This check is what makes forgetting it
impossible: for every stamped URL, it hashes the STAGED bytes of the file that URL points
at and requires the URL to say the same thing.

What it does NOT do: add stamps, or demand one for a URL that has none. A reference with no
`?v=` is a deliberate choice in at least one place (`coi-serviceworker.js` registers itself
and must keep a stable URL), and a hook cannot tell that apart from an oversight.

Fix a failure with:  sh scripts/stamp.sh
This is a `pre-commit` check: it reads the STAGED tree, so it judges the commit being made.

Enable it with:  git config core.hooksPath .githooks
Bypass it with:  git commit --no-verify
"""
import hashlib
import posixpath
import re
import subprocess
import sys

# Where a stamped URL can appear. Each entry is (file that does the referencing, regex whose
# `url` group is the path as written and whose `v` group is the stamp).
SOURCES = [
    ('chat/index.html', re.compile(r'src="(?P<url>[^"?]+)\?v=(?P<v>[0-9a-f]+)"')),
    ('chat/app.js', re.compile(r"new Worker\('(?P<url>[^'?]+)\?v=(?P<v>[0-9a-f]+)'\)")),
]


def blob(path):
    """`path` as this commit will record it, in bytes, or None if the commit has no such file."""
    p = subprocess.run(['git', 'show', ':' + path], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL)
    return p.stdout if p.returncode == 0 else None


def main():
    bad = []
    for src, pattern in SOURCES:
        text = blob(src)
        if text is None:
            continue
        for m in pattern.finditer(text.decode('utf-8', 'replace')):
            url = m.group('url')
            if '://' in url:
                continue                                  # not ours to stamp
            target = posixpath.normpath(posixpath.join(posixpath.dirname(src), url))
            content = blob(target)
            if content is None:
                continue                                  # not tracked; nothing to hash
            want = hashlib.sha1(content).hexdigest()[:len(m.group('v'))]
            if want != m.group('v'):
                bad.append((src, url, m.group('v'), want))
    if bad:
        sys.stderr.write('\nstale cache-busting stamp -- the browser would keep running the '
                         'previous file:\n')
        for src, url, got, want in bad:
            sys.stderr.write('  %s: %s?v=%s  (its bytes now hash to %s)\n' % (src, url, got, want))
        sys.stderr.write('\nRun `sh scripts/stamp.sh`, stage the result, and commit again.\n\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
