#!/bin/bash
# cputd_blocked.sh -- the triple-double LU's CPU baseline (kappa 1e36) made as strong as the arithmetic allows (03-11-2026), as
# cpudd_blocked.sh does for double-double: the paper's "td LU, 24 CPU threads" was the unblocked loop (1.04 s at n = 2048, 18.0 s
# at 4096 for 8x the work: memory-bound), and solve_td.cu's documented build had no -march=native (its fma a libm call). Built
# here with -march=native; with the machine held:
#  1. n = 2048 (seed 1): unblocked and blocked (nb = 128), the LU and the solution compared bit for bit;
#  2. nb = 64 / 128 / 256 at n = 4096, blocked alone: the fastest is the CPU's, used below;
#  3. n = 1024, 2048, 4096, seeds 1-3: GPU solver (nb 256, the program's default; after a warm-up solve), the unblocked and
#     the blocked CPU LU (no MPFR).
#   OZ_DEST=/opt/ozaki/src-cputd ./wsl.sh "setsid -f bash bench/cputd_blocked.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/cputd.log
L=/opt/ozaki/job/cputd.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
bash bench/build_solvers.sh soltd >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold cputd; log "machine held"
log "== check n=2048"; OZ_NOMPFR=1 OZ_CPUTD_NB=128 ./out_soltd 2048 256 36 1 >> $L 2>&1; log "rc=$?"
best=128; bt=1e9
for nb in 64 128 256; do
    t=$(OZ_NOMPFR=1 OZ_CPUTD_NOUB=1 OZ_CPUTD_NB=$nb ./out_soltd 4096 256 36 1 | tee -a $L | awk '/blocked nb=/ {print $(NF-1)}')
    log "nb=$nb: ${t:-?} s"; [ -n "$t" ] && awk -v a=$t -v b=$bt 'BEGIN {exit !(a < b)}' && { best=$nb; bt=$t; }; done
log "nb = $best"
for n in 1024 2048 4096; do
    for seed in 1 2 3; do
        log "== n=$n seed=$seed"; OZ_NOMPFR=1 OZ_CPUTD_NB=$best timeout 3600 ./out_soltd $n 256 36 $seed >> $L 2>&1; log "rc=$?"; done; done
log "done, machine given back"
