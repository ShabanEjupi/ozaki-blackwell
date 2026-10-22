#!/bin/bash
# energy_solve.sh N [REPS] -- energy per Ax=b solve (kappa 1e20): __float128 LU on all CPU cores vs double-double LU on the GPU.
# Runs inside WSL; the program prints epoch marks around the quad solve and around REPS back-to-back GPU solves (after a
# warm-up). CPU package power: the Windows RAPL counters, sampled on the Windows side; GPU power: nvidia-smi here, 100 ms.
set -euo pipefail
n=${1:-4096}; reps=${2:-20}
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
$SMI --query-gpu=timestamp,power.draw,clocks.sm --format=csv,noheader -lms 100 > out_gpu_power.csv &
trap 'kill $! 2>/dev/null' EXIT
echo "MARK idle-start $(date +%s.%N)"; sleep 10; echo "MARK idle-end $(date +%s.%N)"
OZ_QUADONLY=1 ./out_dev $n 1024 20 1 | grep -E "MARK|LU"
sleep 5
OZ_GPU=1 OZ_REPS=$reps ./out_dev $n 1024 20 1 | grep -E "MARK|all on" | tail -3
kill $! 2>/dev/null; sleep 0.5
echo "GPU_POWER_CSV_BEGIN"; cat out_gpu_power.csv; echo "GPU_POWER_CSV_END"
