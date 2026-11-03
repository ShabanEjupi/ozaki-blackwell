#!/bin/bash
# emu_compare.sh -- FP64 emulation on INT8 tensor cores, three implementations on this laptop in one held window (03-11-2026):
# ours (bench/ozaki_test.cu: Ozaki-II, 12/14/16 moduli), NVIDIA's (cuBLAS 13, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT: ADP
# and 55/47/39 fixed bits; bench/cublas_emu.cpp against the pip wheels in /opt/ozaki/cu13) and RIKEN's GEMMul8 (Ozaki-II,
# the authors' public code, MIT; /opt/ozaki/GEMMul8 built for sm_120, INT8 backend, gemm only). The paper compared with none
# of them on the same machine: cuBLAS's emulation (CUDA 13.0 Update 2, October 2025) was not cited at all.
#   OZ_DEST=/opt/ozaki/src-emu ./wsl.sh "setsid -f bash bench/emu_compare.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/emu_compare.log
L=/opt/ozaki/job/emu_compare.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
P=/opt/ozaki/cu13/lib/python3.12/site-packages/nvidia/cu13
{ nvcc -O3 -arch=sm_120 -std=c++17 -o out_test bench/ozaki_test.cu &&
  g++ -O2 -std=c++17 -I$P/include -idirafter /usr/local/cuda-12.8/include bench/cublas_emu.cpp -o out_cublas_emu \
      -L$P/lib -l:libcublas.so.13 -l:libcudart.so.13 -Wl,-rpath,$P/lib; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
G=/opt/ozaki/GEMMul8/test   # built with nvcc 13.4 of /opt/ozaki/cu13 (its cuBLASLt batch modes need cuBLAS 13): gemmul8_build.log
. llm/hold.sh; hold emu; log "machine held"
log "== ours (ozaki_test)"; ./out_test >> $L 2>&1; log "rc=$?"
log "== cuBLAS"; ./out_cublas_emu >> $L 2>&1; log "rc=$?"
for m in "accuracy_square GEMM D transA=N transB=N" "time_square GEMM D transA=N transB=N"; do
    log "== GEMMul8: $m"; (cd $G && timeout 1800 make run MODE="$m") >> $L 2>&1; log "rc=$?"; done
log "done, machine given back"
