#!/bin/bash
# dd_same_run.sh -- the double-double table of the paper in single runs (04-11-2026): for n = 2048, 4096, 6144, 8192 (kappa 1e20,
# seed 1, nb as in the table) one process times __float128 LU, the GPU solver after a warm-up, the CPU double-double LU
# unblocked and blocked (OZ_CPUDD_NB=64, the best of cpudd.log), so every ratio in the table is between numbers of one run.
# Until today the GPU stood beside quad in one run (0.956 s at 8192, after minutes of all-core load) and beside the blocked CPU
# LU in another (0.676 s). FP64 LU skipped (OZ_SKIPCPU=1): its error is in the text already. ~16 min, quad 12 of them.
#   OZ_DEST=/opt/ozaki/src-dd ./wsl.sh "setsid -f bash bench/dd_same_run.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/dd_same.log
L=/opt/ozaki/job/dd_same.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
{ nice -n 19 g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o &&
  nice -n 19 nvcc -O3 -arch=sm_120 -std=c++17 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off,-march=native -o out_solve \
      bench/solve.cu out_ref.o -lquadmath -lgomp; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold ddsame; log "machine held"
for n in 2048 4096 6144 8192; do
    log "== n=$n seed=1"; OZ_SKIPCPU=1 OZ_CPUDD_NB=64 timeout 3600 ./out_solve $n 1024 20 1 >> $L 2>&1; log "rc=$?"; done
log "done, machine given back"
