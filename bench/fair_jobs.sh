#!/bin/bash
# fair_jobs.sh -- 04-11-2026, everything the paper's same-laptop comparisons still lacked, in one held window:
#  1. the three FP64 emulations timed the same way: bench/ozaki_test.cu now takes the median of five calls after a warm-up at
#     n = 2048, 4096, 8192, as bench/emu_harness.h times cuBLAS 13's emulation and GEMMul8 (until today ours was the mean of
#     three at 2048 and 4096 only);
#  2. the GPU double-double solve at each n of the dd table as the median of 20 back-to-back solves after a warm-up (single
#     solves at n = 2048 ranged 0.086-0.210 s on 04-11; the blocked CPU LU beside it is stable to 2%, cpudd.log);
#  3. energy per solve of the blocked CPU double-double LU beside the GPU (bench/energy_cpudd.sh, nb 64) with Windows reachable:
#     the detached run of 04-11 01:10 had a dead WSL interop socket, 0 RAPL samples and printed "-1 + 13.8 = 12.8 W" (llm/win.sh).
#   OZ_DEST=/opt/ozaki/src-fair ./wsl.sh "setsid -f bash bench/fair_jobs.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/fair.log
L=/opt/ozaki/job/fair.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
P=/opt/ozaki/cu13/lib/python3.12/site-packages/nvidia/cu13
{ nice -n 19 nvcc -O3 -arch=sm_120 -std=c++17 -o out_test bench/ozaki_test.cu &&
  nice -n 19 g++ -O2 -std=c++17 -I$P/include -idirafter /usr/local/cuda-12.8/include bench/cublas_emu.cpp -o out_cublas_emu \
      -L$P/lib -l:libcublas.so.13 -l:libcudart.so.13 -Wl,-rpath,$P/lib &&
  nice -n 19 $P/bin/nvcc -O3 -std=c++20 -arch=sm_120 -DGEMMUL8_BUILD_INT8=1 -DGEMMUL8_BUILD_FP8=0 -I/opt/ozaki/GEMMul8/include \
      -o out_gemmul8_emu bench/gemmul8_emu.cu /opt/ozaki/GEMMul8/lib/libgemmul8.a -L$P/lib -l:libcublas.so.13 \
      -l:libcublasLt.so.13 -l:libcudart.so.13 -lcuda -ldl -Xlinker -rpath=$P/lib &&
  bash bench/build_solvers.sh solve; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold fair; log "machine held"
if win_interop 2>> $L; then log "Windows reachable (WSL_INTEROP=$WSL_INTEROP), AC: $(battery)"; else log "Windows NOT reachable"; fi
log "== ours (ozaki_test)"; ./out_test >> $L 2>&1; log "rc=$?"
log "== cuBLAS"; ./out_cublas_emu >> $L 2>&1; log "rc=$?"
log "== GEMMul8, our harness (emu_harness.h)"; ./out_gemmul8_emu >> $L 2>&1; log "rc=$?"
for n in 1024 2048 4096 6144 8192; do
    nb=$((n >= 4096 ? 1024 : n / 4)); log "== GPU dd solve, n=$n nb=$nb, 20 back to back after a warm-up"
    OZ_GPU=1 OZ_REPS=20 timeout 900 ./out_solve $n $nb 20 1 > $L.tmp 2>&1; rc=$?
    awk '/^all on GPU/ {print $4}' $L.tmp | sort -n | awk -v n=$n 'NF {a[++k] = $1} END { if (k) printf "n=%d: median %.3f s of %d solves (%.3f-%.3f)\n", n, (a[int((k + 1) / 2)] + a[int(k / 2) + 1]) / 2, k, a[1], a[k]; else print "n=" n ": no solve printed" }' >> $L
    grep -v '^all on GPU' $L.tmp | grep -v '^MARK' >> $L; log "rc=$rc"
done; rm -f $L.tmp
log "== energy per solve, n=4096 (bench/energy_cpudd.sh, nb 64, 8 CPU solves)"; bash bench/energy_cpudd.sh 4096 64 8 >> $L 2>&1; log "rc=$?"
log "done, machine given back"
