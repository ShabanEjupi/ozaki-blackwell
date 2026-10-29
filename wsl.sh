#!/bin/bash
# wsl.sh "<command>" -- ships this folder to the laptop's WSL2 (imazhe, the GPU) and runs <command> in it.
# Smart App Control refuses unsigned .exe on Windows, so every binary is built and run inside WSL2.
# Path: IDE -> ampere -> reverse ssh tunnel -> laptop sshd -> wsl.exe; stdin carries the tar, the command rides
# inside it as .run.sh (no quoting through four shells).
set -euo pipefail
cd "$(dirname "$0")"
# set -e stops at the first failing command, a grep without a match too (29-10-2026: twice silently, the output just ended):
# the ERR trap says where it stopped
printf 'export PATH=/usr/local/cuda-12.8/bin:$PATH\nset -e\ntrap '"'"'echo "[wsl.sh] stopped at line $LINENO: $BASH_COMMAND (rc $?)" >&2'"'"' ERR\n%s\n' "${1:-make}" > .run.sh
{ git rev-parse --short HEAD; git diff --quiet HEAD || echo "+ uncommitted changes"; } > .version 2>/dev/null
trap 'rm -f .run.sh .version' EXIT
tar czf - --exclude=.git --exclude='out_*' . |
  ssh ampere "timeout ${TIMEOUT:-580} ssh -o BatchMode=yes -i ~/.ssh/laptop_ed25519 -p 16222 NB@127.0.0.1 \
    'wsl.exe -d imazhe -u root -- bash -c \"mkdir -p /opt/ozaki/src && cd /opt/ozaki/src && tar xzf - && bash .run.sh\"'" 2>&1 | tr -d '\r'
