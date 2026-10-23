#!/bin/bash
# energy_det.sh N REPS -- runs inside WSL: GPU power (nvidia-smi, 100 ms) around REPS back-to-back exact determinants on the
# GPU (method 1), then one FLINT fmpz_mat_det of the same matrix; epoch marks between. The CPU package power comes from the
# Windows RAPL counter, sampled by bench/energy_det_host.sh at the same time on the Windows side.
set -euo pipefail
n=${1:-2048}; reps=${2:-20}
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
$SMI --query-gpu=timestamp,power.draw --format=csv,noheader -lms 100 > out_gpu_power.csv &
trap 'kill $! 2>/dev/null' EXIT
sleep 8; echo "MARK gpu-start $(date +%s.%N)"
for i in $(seq $reps); do ./out_det $n 32 0 1 | grep -o "total [0-9.]* s"; done
echo "MARK gpu-end $(date +%s.%N)"; sleep 8
echo "MARK flint-start $(date +%s.%N)"; ./out_df $n 24; echo "MARK flint-end $(date +%s.%N)"; sleep 4
kill $! 2>/dev/null; sleep 0.5
echo "GPU_POWER_CSV_BEGIN"; cat out_gpu_power.csv; echo "GPU_POWER_CSV_END"
