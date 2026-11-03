#!/bin/bash
# energy_emu.sh -- GFLOPS/W of every FP64 emulation on this laptop in one held window (03-11-2026): the Energy section gave ours
# against the native FP64 peak kernel only, while cuBLAS 13 emulates DGEMM itself and GEMMul8 is public. The same method as
# bench/energy_all.sh (12 s per run on bench/energy.cu's inputs, n = 4096, nvidia-smi power every 200 ms, GFLOPS/W = the run's
# GFLOPS / its mean power, three runs per workload): ours (out_energy: native peak kernel, 12/14/16 moduli), cuBLAS (native
# DGEMM, ADP, 55 and 47 bits) and GEMMul8 (fast and accurate modes, 14/16/18 moduli) through EMU_ENERGY of emu_harness.h.
#   OZ_DEST=/opt/ozaki/src-energyemu ./wsl.sh "setsid -f bash bench/energy_emu.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/energy_emu.log
L=/opt/ozaki/job/energy_emu.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
P=/opt/ozaki/cu13/lib/python3.12/site-packages/nvidia/cu13
{ nice -n 19 nvcc -O3 -arch=sm_120 -std=c++17 -o out_energy bench/energy.cu &&
  nice -n 19 g++ -O2 -std=c++17 -I$P/include -idirafter /usr/local/cuda-12.8/include bench/cublas_emu.cpp -o out_cublas_emu \
      -L$P/lib -l:libcublas.so.13 -l:libcudart.so.13 -Wl,-rpath,$P/lib &&
  nice -n 19 $P/bin/nvcc -O3 -std=c++20 -arch=sm_120 -DGEMMUL8_BUILD_INT8=1 -DGEMMUL8_BUILD_FP8=0 -I/opt/ozaki/GEMMul8/include \
      -o out_gemmul8_emu bench/gemmul8_emu.cu /opt/ozaki/GEMMul8/lib/libgemmul8.a -L$P/lib -l:libcublas.so.13 \
      -l:libcublasLt.so.13 -l:libcudart.so.13 -lcuda -ldl -Xlinker -rpath=$P/lib; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold energyemu; log "machine held"
./out_energy oz16 3 > /dev/null 2>&1   # warm-up: clocks and power settle before the first sampled run
run() {   # run <label> <command...>: three sampled runs
    local lab=$1; shift; local cmd=("$@") pw ns clk
    for r in 1 2 3; do
        S=$(mktemp)
        nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -lms 200 > "$S" 2> /dev/null & SP=$!
        out=$(set -o pipefail; "${cmd[@]}" 2>&1 | tail -n 1); rc=$?
        kill $SP 2> /dev/null; wait $SP 2> /dev/null
        g=$(printf '%s\n' "$out" | sed -nE 's/.*: ([0-9.]+) GFLOPS over.*/\1/p')
        m=$(awk -F', *' '$1 ~ /^[0-9.]+$/ {s += $1; c += $2; k++} END {if (k) printf "%.2f %d %.0f", s / k, k, c / k; else print "0 0 0"}' "$S")
        rm -f "$S"; read -r pw ns clk <<< "$m"   # not set --: that would replace the command in $@
        e=$(awk -v g="${g:-0}" -v p="$pw" 'BEGIN {if (p > 0 && g > 0) printf "%.2f GFLOPS/W", g / p; else print "NOT MEASURED"}')
        log "$lab run $r rc=$rc: $out | mean $pw W over $ns samples, SM $clk MHz | $e"
    done
}
for w in native oz16 oz14 oz12; do run "ours $w" ./out_energy $w 12; done
for c in "native" "emu ADP" "emu 55b" "emu 47b"; do run "cuBLAS $c" env EMU_ENERGY="$c" ./out_cublas_emu; done
for c in fast14 fast16 fast18 accu14 accu16 accu18; do run "GEMMul8 $c" env EMU_ENERGY=$c ./out_gemmul8_emu; done
log "done, machine given back"
