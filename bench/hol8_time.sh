#!/bin/bash
# hol8_time.sh -- the exact determinants of the 8-bit residue chain (0de27f0) timed with the machine to themselves, every
# line kept: the paper's HOL table (16-bit chain, out_det_hol.txt) gets the 8-bit numbers only from this log.
#   OZ_DEST=/opt/ozaki/src-hol8 ./wsl.sh "setsid -f bash bench/hol8_time.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/hol8.log. Per size: the 8-bit chain twice, the 16-bit chain (OZ_HOL16=1) once; the heads must be the
# 16-bit chain's (and out_det_hol.txt's, which are bit-exact against FLINT).
L=/opt/ozaki/job/hol8.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
g++ -O3 -c bench/det_flint.cpp -o out_dflint.o && nvcc -O3 -arch=sm_120 -std=c++17 --default-stream per-thread -Xcompiler -fopenmp \
    -o out_det bench/det.cu out_dflint.o -lflint -lgmp -lgomp -lcusolver -lcublas >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold hol8; log "machine held"
for c in "1024 8" "1024 16" "2048 8" "2048 16" "4096 8"; do set -- $c
    for v in 8bit 8bit 16bit; do
        if [ $v = 16bit ]; then e="OZ_HOL16=1"; else e=""; fi
        r=$(env $e OZ_PLANES=$2 OZ_HOL=1 ./out_det $1 32 0 1 2>&1 | grep -E "^HOL|head" | tr '\n' ' ')
        log "n=$1 planes=$2 $v: $r"
    done
done
log "done, machine given back"
