// hold.cu <GB> <seconds> -- keeps <GB> of VRAM and the SMs busy for <seconds>, printing every 10 s; a probe for what stops
// the WSL VM while a long GPU job runs (27-10-2026: the VM went down 4-40 s after each model-lokal tick restarted Ollama).
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
__global__ void spin(float *p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = p[i] * 0.999f + 1.0f;
}
int main(int argc, char **argv) {
    size_t n = (size_t)(atof(argv[1]) * (1u << 30) / 4); int secs = atoi(argv[2]);
    float *p; if (cudaMalloc(&p, n * 4) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
    cudaMemset(p, 0, n * 4);
    time_t t0 = time(0), last = t0;
    while (time(0) - t0 < secs) {
        spin<<<46 * 8, 256>>>(p, n);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("CUDA error %s after %ld s\n", cudaGetErrorString(e), (long)(time(0) - t0)); return 2; }
        if (time(0) - last >= 10) { last = time(0); size_t f, t; cudaMemGetInfo(&f, &t); printf("%ld s ok, free %.2f GB\n", (long)(last - t0), f / 1e9); fflush(stdout); }
    }
    printf("done\n");
}
