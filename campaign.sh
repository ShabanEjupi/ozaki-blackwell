#!/bin/bash
# campaign.sh -- days-long unattended run on the laptop (inside WSL2): the Ax=b grid over n, kappa and seeds, with GPU power
# sampled every 200 ms, so the paper's tables carry spreads, not single runs. Stop: touch /opt/ozaki/STOP.
# Output: /opt/ozaki/campaign/{solve.log,power.csv,done.txt}; each solve run is headed by "### n kappa seed start-epoch".
set -u
O=/opt/ozaki/campaign; mkdir -p $O; cd /opt/ozaki/src
export PATH=/usr/local/cuda-12.8/bin:$PATH
g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o &&
  nvcc -O3 -arch=sm_120 -std=c++17 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off -o out_solve bench/solve.cu out_ref.o -lquadmath -lgomp || exit 1
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
$SMI --query-gpu=timestamp,power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader -lms 200 >> $O/power.csv 2>&1 &
trap 'kill $! 2>/dev/null' EXIT
for seed in $(seq 1 1000); do
  for n in 1024 2048 4096 6144; do
    for k in 8 12 16 20 24 28; do
      [ -e /opt/ozaki/STOP ] && exit 0
      grep -qx "$n $k $seed" $O/done.txt 2>/dev/null && continue
      echo "### $n $k $seed $(date +%s)" >> $O/solve.log
      timeout 1800 ./out_solve $n 256 $k $seed >> $O/solve.log 2>&1 && echo "$n $k $seed" >> $O/done.txt
    done
  done
done
