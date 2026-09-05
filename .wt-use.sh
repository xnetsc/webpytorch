#!/bin/sh
# Put one commit's tree behind the baseline server, keeping the same origin so the
# model stays cached. Usage: sh .wt-use.sh <commit>
set -e
cd "$(dirname "$0")"
rm -rf .wt-base
mkdir -p .wt-base
git archive "$1" | tar -x -C .wt-base
for d in lib models node_modules; do [ -e ".wt-base/$d" ] || ln -s "$(pwd)/$d" ".wt-base/$d"; done
echo "8121 now serves $(git log --oneline -1 "$1" | cat)"
