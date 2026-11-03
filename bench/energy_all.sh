#!/bin/bash
# energy_all.sh -- the Energy section's GEMM line measured again with the current code: the paper's 7.4 / 53.6 / 69.5
# GFLOPS/W (native FP64 peak kernel, Ozaki-II at 16 and 12 moduli, n = 4096) are the code of 15 October, and the FP64
# emulation has since gone from 5.56 to 6.76 TFLOPS. Same method as then (bench/energy.cu, 12 s per run, nvidia-smi power
# every 200 ms, GFLOPS/W = the run's GFLOPS / its mean power), but three runs per workload, so a range, not one number;
# machine held (llm/hold.sh: AC, no training run, no SPEX). Every run's raw output, sample count, mean W and SM clock logged.
#   OZ_DEST=/opt/ozaki/src-energy ./wsl.sh "setsid -f bash bench/energy_all.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/energy.log
L=/opt/ozaki/job/energy.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
nvcc -O3 -arch=sm_120 -std=c++17 -o out_energy bench/energy.cu >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold energy; log "machine held"
./out_energy oz16 3 > /dev/null 2>&1   # warm-up: clocks and power settle before the first sampled run
for w in native oz16 oz12; do
    for r in 1 2 3; do
        S=$(mktemp)
        nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits -lms 200 > "$S" 2> /dev/null & P=$!
        out=$(./out_energy $w 12 2>&1); rc=$?
        kill $P 2> /dev/null; wait $P 2> /dev/null
        g=$(printf '%s\n' "$out" | sed -nE 's/.*: ([0-9.]+) GFLOPS over.*/\1/p')
        # samples: "power, clock" per line; a line without a number (N/A) is skipped and the count says how many were used
        m=$(awk -F', *' '$1 ~ /^[0-9.]+$/ {s += $1; c += $2; k++} END {if (k) printf "%.2f %d %.0f", s / k, k, c / k; else print "0 0 0"}' "$S")
        rm -f "$S"; set -- $m
        e=$(awk -v g="${g:-0}" -v p="$1" 'BEGIN {if (p > 0 && g > 0) printf "%.2f GFLOPS/W", g / p; else print "NOT MEASURED (no power samples or no GFLOPS)"}')
        log "$w run $r rc=$rc: $out | mean $1 W over $2 samples, SM $3 MHz | $e"
    done
done
log "done, machine given back"
