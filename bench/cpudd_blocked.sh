#!/bin/bash
# cpudd_blocked.sh -- the double-double LU's CPU baseline made as strong as the arithmetic allows (03-11-2026): the paper put the
# GPU solver against __float128 only (499x at n = 8192), while campaign v2 had also timed a CPU double-double LU with the same
# arithmetic -- unblocked, memory-bound (115 s at n = 8192, 64x the work of n = 2048 in 242x the time). lu_dd_cpu_blocked
# (bench/solve.cu) does the same operations in the same order, cache-blocked. With the machine held:
#  1. n = 2048 and 4096 (kappa 1e20, seed 1): blocked and unblocked, the LU and the solution compared bit for bit;
#  2. nb = 64 / 128 / 256 at n = 4096: the fastest is the CPU's, used below;
#  3. n = 1024 ... 8192, seeds 1-3 (the campaign's matrices, so its GPU and unblocked times pair with these): GPU solver
#     (after a warm-up solve) and the blocked CPU LU, kappa 1e20;
#  4. energy per solve at n = 4096 (bench/energy_cpudd.sh: RAPL package + nvidia-smi), blocked CPU vs GPU.
#   OZ_DEST=/opt/ozaki/src-cpudd ./wsl.sh "setsid -f bash bench/cpudd_blocked.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/cpudd.log
L=/opt/ozaki/job/cpudd.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
{ bash bench/build_solvers.sh solve; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold cpudd; log "machine held"
for n in 2048 4096; do
    log "== check n=$n"; OZ_CPUDD_ONLY=128 OZ_CPUDD_CHECK=1 ./out_solve $n $((n >= 4096 ? 1024 : n / 4)) 20 1 >> $L 2>&1; log "rc=$?"; done
best=128; bt=1e9
for nb in 64 128 256; do
    t=$(OZ_CPUDD_ONLY=$nb ./out_solve 4096 1024 20 1 | tee -a $L | awk '/blocked nb=/ {print $(NF-1)}')
    log "nb=$nb: ${t:-?} s"; [ -n "$t" ] && awk -v a=$t -v b=$bt 'BEGIN {exit !(a < b)}' && { best=$nb; bt=$t; }; done
log "nb = $best"
for n in 1024 2048 4096 6144 8192; do
    for seed in 1 2 3; do
        log "== n=$n seed=$seed"; OZ_CPUDD_ONLY=$best timeout 3600 ./out_solve $n $((n >= 4096 ? 1024 : n / 4)) 20 $seed >> $L 2>&1; log "rc=$?"; done; done
log "== energy per solve, n=4096 (bench/energy_cpudd.sh)"; bash bench/energy_cpudd.sh 4096 $best 8 >> $L 2>&1; log "rc=$?"
log "done, machine given back"
