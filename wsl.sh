#!/bin/bash
# wsl.sh "<command>" -- runs <command> (default: make) on the GPU machine with this folder as it is. With a .hosts file (not in
# git; see .hosts.example) the folder goes there as a tar on stdin and runs in /opt/ozaki/src, the command inside it as .run.sh
# (no quoting through several shells); without one, here.
set -euo pipefail
cd "$(dirname "$0")"
# set -e stops at the first failing command, a grep without a match too (29-10-2026: twice silently, the output just ended):
# the ERR trap says where it stopped
printf 'export PATH=/usr/local/cuda-12.8/bin:$PATH\nset -e\ntrap '"'"'echo "[wsl.sh] stopped at line $LINENO: $BASH_COMMAND (rc $?)" >&2'"'"' ERR\n%s\n' "${1:-make}" > .run.sh
{ git rev-parse --short HEAD; git diff --quiet HEAD || echo "+ uncommitted changes"; } > .version 2>/dev/null
trap 'rm -f .run.sh .version' EXIT
if [ -f .hosts ]; then
    . ./.hosts
    tar czf - --exclude=.git --exclude=.hosts --exclude='out_*' . | oz_gpu "mkdir -p /opt/ozaki/src && cd /opt/ozaki/src && tar xzf - && bash .run.sh" 2>&1 | tr -d '\r'
else
    bash .run.sh
fi
