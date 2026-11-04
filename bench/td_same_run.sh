#!/bin/bash
# td_same_run.sh -- the triple-double table of the paper in single runs (04-11-2026): for n = 1024, 2048, 4096 (kappa 1e36,
# seed 1) one process times the GPU solver, FP64 LU, the CPU triple-double LU unblocked and blocked (nb 64, the best of
# cputd.log) and MPFR at 160 bits, so every ratio in the table is between numbers of one run. Until today the MPFR column came
# from an earlier run whose GPU times (0.082 / 0.212 / 0.563 s) the current solver no longer takes (0.066 / 0.159 / 0.437 s).
#   OZ_DEST=/opt/ozaki/src-td ./wsl.sh "setsid -f bash bench/td_same_run.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/td_same.log
L=/opt/ozaki/job/td_same.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
bash bench/build_solvers.sh soltd >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold tdsame; log "machine held"
for n in 1024 2048 4096; do
    log "== n=$n seed=1"; OZ_CPUTD_NB=64 timeout 3600 ./out_soltd $n 256 36 1 >> $L 2>&1; log "rc=$?"; done
log "done, machine given back"
