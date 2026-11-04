#!/bin/bash
# energy_prof_run.sh -- 04-11-2026, two things the paper still lacked, in one held window:
#  1. energy per solve of the blocked CPU double-double LU beside the GPU solver (bench/energy_cpudd.sh, n = 4096, nb 64):
#     fair.log had 0 RAPL samples in every window although the counter streamed a sample a second -- the awk turned the
#     computed epoch into "1.7938e+09" (CONVFMT %.6g), outside every window;
#  2. where the time of today's solvers goes (OZ_PROF=1, a sync at every phase boundary): double-double at n = 8192 (nb 1024)
#     and triple-double at n = 4096 (nb 256). The paper's breakdowns came from the 0.855 s and 0.563 s solvers; the current
#     ones take 0.69 s and 0.45 s.
#   OZ_DEST=/opt/ozaki/src-eprof ./wsl.sh "setsid -f bash bench/energy_prof_run.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/eprof.log
L=/opt/ozaki/job/eprof.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
F="-O3 -arch=sm_120 -std=c++17 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off,-march=native"
{ nice -n 19 g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o &&
  nice -n 19 nvcc $F -o out_solve bench/solve.cu out_ref.o -lquadmath -lgomp &&
  nice -n 19 nvcc $F -o out_soltd bench/solve_td.cu -lmpfr -lgmp -lgomp; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold eprof; log "machine held"
if win_interop 2>> $L; then log "Windows reachable (WSL_INTEROP=$WSL_INTEROP), AC: $(battery)"; else log "Windows NOT reachable"; fi
log "== energy per solve, n=4096 (bench/energy_cpudd.sh, nb 64, 8 CPU solves)"
bash bench/energy_cpudd.sh 4096 64 8 >> $L 2>&1; log "rc=$?"
log "== dd phases, n=8192 nb=1024 (OZ_PROF=1; the first solve pays the module load)"
OZ_PROF=1 timeout 900 ./out_solve 8192 1024 20 1 >> $L 2>&1; log "rc=$?"
log "== td phases, n=4096 nb=256 (OZ_PROF=1, GPU only)"
OZ_PROF=1 OZ_GPU=1 timeout 900 ./out_soltd 4096 256 36 1 >> $L 2>&1; log "rc=$?"
log "done, machine given back"
