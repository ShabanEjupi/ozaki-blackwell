#!/bin/bash
# vf_sweep.sh [n] [nm] -- joules per FP64 DGEMM against the SM clock (nvidia-smi -lgc from WSL through interop), for the
# Strassen-Ozaki path and, at three clocks, the GPU's own FP64. Under the power cap the clock is where the watts go; below
# it, energy per operation falls with the voltage, so the fewest joules per result sit at some clock below the cap.
# The lock is released on exit (trap), whatever happens.
n=${1:-16384}; nm=${2:-15}
S=/mnt/c/Windows/System32/nvidia-smi.exe
trap '"$S" -rgc >/dev/null 2>&1' EXIT
for clk in default 2400 2100 1800 1500 1200 900; do
  if [ $clk = default ]; then "$S" -rgc >/dev/null; else "$S" -lgc $clk,$clk >/dev/null; fi
  nat=0; case $clk in default|1800|1200) nat=1;; esac
  echo "== SM clock $clk"
  OZ_NATIVE=$nat OZ_L=2 stdbuf -oL /opt/ozaki/b/oz_strassen $n $nm 0 | grep -E 'native|^oz'
  nvidia-smi --query-gpu=temperature.gpu,clocks.mem --format=csv,noheader
done
