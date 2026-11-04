#!/bin/bash
# ozprof_run.sh -- 04-11-2026: where the FP64 GEMM's time goes at n = 2048, 4096 and 8192 (OZ_PROF=1 in src/ozaki2.cuh: the GPU
# time of scale+residues, the INT8 GEMMs and the CRT, from events on the stream). fair.log had our 16 moduli at 6.73 TFLOPS
# at n = 4096 and 5.06 at 8192 while cuBLAS's emulation rose from 8.51 to 9.67; the paper says "we have not profiled the fall".
# One held window; this run's TFLOPS lines are not the paper's (each profiled call waits for its last event).
#   OZ_DEST=/opt/ozaki/src-ozprof ./wsl.sh "setsid -f bash bench/ozprof_run.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/ozprof.log
L=/opt/ozaki/job/ozprof.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
nice -n 19 nvcc -O3 -arch=sm_120 -std=c++17 -o out_test bench/ozaki_test.cu >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold ozprof; log "machine held"
OZ_PROF=1 timeout 900 ./out_test 2>&1 | grep -E '^oz1[246] (2048|4096|8192)x|^n=' >> $L; log "rc=${PIPESTATUS[0]}"
log "done, machine given back"
