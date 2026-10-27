#!/bin/bash
# real_run.sh NAME URL -- one real matrix end to end: download (NIST Matrix Market), exact conversion (mm2oz.py), the
# GPU solver (det.cu OZ_FILE, HOL), SPEX Left LU on the same integers (timeout $SPEX_T s, default 3600), and the FP64
# errors against the exact x (fp64_err.py). Results under /opt/ozaki/real/NAME.
set -o pipefail
N=$1; U=$2; D=/opt/ozaki/real/$N; mkdir -p $D; cd $(dirname $0)/..
[ -f $D/m.mtx.gz ] || curl -sSL -A Mozilla/5.0 -m 300 -o $D/m.mtx.gz "$U" || { echo "$N: download failed"; exit 1; }
python3 -I bench/mm2oz.py $D/m.mtx.gz $D/s > $D/conv.log || { echo "$N: conversion failed"; cat $D/conv.log; exit 1; }
read np L n0 < <(python3 -c "import struct;print(*struct.unpack('3i',open('$D/s.oz','rb').read()[4:16]))")
echo "$N: $(cat $D/conv.log)"
OZ_FILE=$D/s.oz OZ_PLANES=$L OZ_HOL=1 ./out_det $np 32 0 1 > $D/gpu.log 2>&1; echo "  GPU rc=$? $(grep -E 'file:|HOL:|head|FILE|failed|RATREC|minor' $D/gpu.log | tr '\n' ' ' | cut -c1-900)"
if [ "${SPEX_T:-3600}" != 0 ]; then timeout ${SPEX_T:-3600} ./out_spex $D/s.int > $D/spex.log 2>&1; echo "  SPEX rc=$? $(tail -1 $D/spex.log)"; fi
python3 -I bench/fp64_err.py $D $n0 $np
