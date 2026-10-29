#!/bin/bash
# oz_campaign.sh [n] [points...] -- the FP64 table of README "Past the classical ceiling, in FP64": at each operating point
# (SM MHz/memory MHz, or "default"), the 2-norm-scaled 14-moduli DGEMM classical (L = 0) and with Strassen (L = 2, checked
# bit for bit against L = 0), the max-scaled 15-moduli one, and the GPU's own FP64 where NATIVE lists the point.
n=${1:-16384}; shift; pts=${@:-default 1800/9001 1500/9001 1350/9001 1200/9001}
S=/mnt/c/Windows/System32/nvidia-smi.exe
trap '"$S" -rgc >/dev/null 2>&1; "$S" -rmc >/dev/null 2>&1' EXIT
for p in $pts; do
  "$S" -rgc >/dev/null; "$S" -rmc >/dev/null
  if [ $p != default ]; then "$S" -lgc ${p%/*},${p%/*} >/dev/null; "$S" -lmc ${p#*/},${p#*/} >/dev/null; fi
  nat=0; case " ${NATIVE:-default 1800/9001 1500/9001} " in *" $p "*) nat=1;; esac
  echo "== $p"
  OZ_SCALE=norm OZ_NATIVE=$nat stdbuf -oL /opt/ozaki/b/oz_strassen $n 14 0 | grep -E 'native|^oz'
  OZ_SCALE=max OZ_NATIVE=0 OZ_L=2 stdbuf -oL /opt/ozaki/b/oz_strassen $n 15 0 | grep -E '^oz'
  nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
done
