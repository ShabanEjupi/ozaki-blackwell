#!/bin/bash
# vf_mem.sh [n] [nm] -- the same as vf_sweep.sh over memory clocks (-lmc) at the efficient SM clocks: the memory draws watts
# of its own, and the modular products, not the passes, are most of the time
n=${1:-16384}; nm=${2:-15}
S=/mnt/c/Windows/System32/nvidia-smi.exe
trap '"$S" -rgc >/dev/null 2>&1; "$S" -rmc >/dev/null 2>&1' EXIT
for clk in 1500 1200; do for mem in 14001 11001 9001; do
  "$S" -lgc $clk,$clk >/dev/null; "$S" -lmc $mem,$mem >/dev/null
  echo "== SM $clk MHz, memory $mem MHz"
  OZ_NATIVE=0 OZ_L=2 stdbuf -oL /opt/ozaki/b/oz_strassen $n $nm 0 | grep -E '^oz'
done; done
