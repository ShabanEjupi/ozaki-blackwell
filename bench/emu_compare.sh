#!/bin/bash
# emu_compare.sh -- FP64 emulation on INT8 tensor cores, three implementations on this laptop in one held window (03-11-2026):
# ours (bench/ozaki_test.cu: Ozaki-II, 12/14/16 moduli), NVIDIA's (cuBLAS 13, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT: ADP
# and 55/47/39 fixed bits; bench/cublas_emu.cpp against the pip wheels in /opt/ozaki/cu13) and RIKEN's GEMMul8 (Ozaki-II,
# the authors' public code, MIT; /opt/ozaki/GEMMul8 built for sm_120, INT8 backend, gemm only). The paper compared with none
# of them on the same machine: cuBLAS's emulation (CUDA 13.0 Update 2, October 2025) was not cited at all.
#   OZ_DEST=/opt/ozaki/src-emu ./wsl.sh "setsid -f bash bench/emu_compare.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/emu_compare.log (EMU_LOG=); EMU_SKIP_G8TEST=1 leaves out GEMMul8's own harness (~1 h).
L=${EMU_LOG:-/opt/ozaki/job/emu_compare.log}; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
P=/opt/ozaki/cu13/lib/python3.12/site-packages/nvidia/cu13
{ nice -n 19 nvcc -O3 -arch=sm_120 -std=c++17 -o out_test bench/ozaki_test.cu &&
  nice -n 19 g++ -O2 -std=c++17 -I$P/include -idirafter /usr/local/cuda-12.8/include bench/cublas_emu.cpp -o out_cublas_emu \
      -L$P/lib -l:libcublas.so.13 -l:libcudart.so.13 -Wl,-rpath,$P/lib &&
  nice -n 19 $P/bin/nvcc -O3 -std=c++20 -arch=sm_120 -DGEMMUL8_BUILD_INT8=1 -DGEMMUL8_BUILD_FP8=0 -I/opt/ozaki/GEMMul8/include \
      -o out_gemmul8_emu bench/gemmul8_emu.cu /opt/ozaki/GEMMul8/lib/libgemmul8.a -L$P/lib -l:libcublas.so.13 \
      -l:libcublasLt.so.13 -l:libcudart.so.13 -lcuda -ldl -Xlinker -rpath=$P/lib; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
G=/opt/ozaki/GEMMul8/test   # bench/gemmul8_build.sh (nvcc 13.4 + cuBLAS 13 of /opt/ozaki/cu13; its cuBLASLt batch modes need 13)
[ -x $G/test ] || { log "GEMMul8 not built: run bench/gemmul8_build.sh first"; exit 1; }
. llm/hold.sh; hold emu; log "machine held"
log "== ours (ozaki_test)"; ./out_test >> $L 2>&1; log "rc=$?"
log "== cuBLAS"; ./out_cublas_emu >> $L 2>&1; log "rc=$?"
log "== GEMMul8, our harness (emu_harness.h)"; ./out_gemmul8_emu >> $L 2>&1; log "rc=$?"
[ "${EMU_SKIP_G8TEST:-0}" = 1 ] && { log "done (GEMMul8's own harness skipped), machine given back"; exit 0; }
# GEMMul8's own harness (double: moduli 8-20 for accuracy, 9-17 for time; fast and accurate modes; cuBLAS's emulation and native
# beside): accuracy at n = 8192,
# timing at n = 1024 ... 65536 (the sizes that do not fit in 12 GB are skipped), >= 30 runs and >= 15 s per point: ~1 h.
# Its CSV files go to /opt/ozaki/job/gemmul8/.
mkdir -p /opt/ozaki/job/gemmul8
for m in "accuracy_square GEMM D transA=N transB=N:3600" "time_square GEMM D transA=N transB=N:7200"; do
    log "== GEMMul8: ${m%:*}"; (cd $G && timeout ${m##*:} ./test ${m%:*}) >> $L 2>&1; log "rc=$?"
    mv -f $G/*.csv /opt/ozaki/job/gemmul8/ 2> /dev/null; done
log "done, machine given back"
