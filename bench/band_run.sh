#!/bin/bash
# band_run.sh -- 04-11-2026: the FP64 emulation's INT8 GEMMs walking each plane in bands of 8 tile rows (OZ_BAND, src/gemm_i8.cuh)
# against the old row-major order, in one held window, interleaved 0 8 0 8 so a thermal drift hits both alike. Per band: the
# OZ_PROF stage times at 4096/8192, then a run without OZ_PROF for the TFLOPS lines and the accuracy table -- the errors must be
# identical (a band changes which block computes a tile, not the sums). Also the GPU's L2 size and memory bus, which decide
# where row-major order starts re-reading B from DRAM (ozprof.log, 4 November: 93 TOPS at 8192 against 155-171 at 4096).
#   OZ_DEST=/opt/ozaki/src-band ./wsl.sh "setsid -f bash bench/band_run.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/band.log
L=/opt/ozaki/job/band.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
nice -n 19 nvcc -O3 -arch=sm_120 -std=c++17 -o out_test bench/ozaki_test.cu >> $L 2>&1 || { log "build FAILED"; exit 1; }
cat > out_l2q.cu <<'E'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp p; if (cudaGetDeviceProperties(&p, 0)) return 1;
    printf("%s: L2 %d KiB, persisting L2 max %d KiB, memory bus %d bit, %d SMs\n", p.name, p.l2CacheSize >> 10,
           p.persistingL2CacheMaxSize >> 10, p.memoryBusWidth, p.multiProcessorCount);
}
E
nvcc -o out_l2q out_l2q.cu >> $L 2>&1 && ./out_l2q >> $L
. llm/hold.sh; hold band; log "machine held"
for b in 0 8 0 8; do
    log "OZ_BAND=$b profile"
    OZ_BAND=$b OZ_PROF=1 timeout 900 ./out_test 2>&1 | grep -E '^oz16 (4096|8192)x' >> $L
    log "OZ_BAND=$b speed+accuracy"
    OZ_BAND=$b timeout 900 ./out_test 2>&1 | grep -E 'K=|^n=' >> $L; log "rc=${PIPESTATUS[0]}"
done
log "done, machine given back"
