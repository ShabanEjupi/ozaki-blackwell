// bw.cu -- the card's DRAM ceiling: device-to-device copy, a pure read and a pure write (int4 per thread)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void rd(const int4 *x, size_t n, int *o) { int4 a = {}; for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { int4 v = x[i]; a.x ^= v.x; a.y ^= v.y; a.z ^= v.z; a.w ^= v.w; } if (a.x == 12345 && a.y == 7) *o = a.z + a.w; }
__global__ void wr(int4 *x, size_t n) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] = make_int4(i, 0, 0, 0); }
int main() {
    const size_t B = 1ull << 30; char *a, *b; int *o; cudaMalloc(&a, B); cudaMalloc(&b, B); cudaMalloc(&o, 4);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1); float ms;
    for (int r = 0; r < (getenv("BW_REPS") ? atoi(getenv("BW_REPS")) : 2); r++) {
        cudaEventRecord(t0); cudaMemcpy(b, a, B, cudaMemcpyDeviceToDevice); cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
        printf("copy  %.0f GB/s (read + write)\n", 2 * B / ms / 1e6);
        cudaEventRecord(t0); rd<<<48 * 16, 256>>>((int4 *)a, B / 16, o); cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
        printf("read  %.0f GB/s\n", B / ms / 1e6);
        cudaEventRecord(t0); wr<<<48 * 16, 256>>>((int4 *)b, B / 16); cudaEventRecord(t1); cudaEventSynchronize(t1); cudaEventElapsedTime(&ms, t0, t1);
        printf("write %.0f GB/s\n", B / ms / 1e6);
    }
}
