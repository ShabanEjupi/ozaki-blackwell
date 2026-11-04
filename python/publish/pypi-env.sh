#!/bin/bash
# Runs on ampere: writes /tmp/pypi-reg/env (0600) from the credentials file; prints only a line count.
set -euo pipefail
K=/mnt/data/workspace/spacecode-brain/credentials.local.txt
umask 077
{ grep -E '^PYPI_(USER|EMAIL|PASS)=' "$K"; echo "EMRI=Shaban Ejupi"; echo "PRIT_MIN=60"; } > /tmp/pypi-reg/env
wc -l < /tmp/pypi-reg/env
