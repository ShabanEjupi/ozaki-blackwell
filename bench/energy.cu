// energy.cu <ozN|native> <seconds> -- one workload in a loop; the caller samples nvidia-smi power meanwhile.
// Prints the FP64(-equivalent) GFLOPS so GFLOPS/W = this / the mean power.draw.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "../src/ozaki2.cuh"

__global__ void fp64_peak(double *out, int n) {   // the native FP64 ceiling (same kernel as peak.cu)
    double x[8]; for (int i = 0; i < 8; i++) x[i] = threadIdx.x + i;
    for (int i = 0; i < n; i++)
#pragma unroll
        for (int j = 0; j < 16; j++)
#pragma unroll
            for (int t = 0; t < 8; t++) x[t] = fma(x[t], 0.999, 1e-3);
    double s = 0; for (int t = 0; t < 8; t++) s += x[t];
    if (s == -1.2345) out[0] = s;
}

int main(int argc, char **argv) {
    const char *w = argc > 1 ? argv[1] : "oz16"; double secs = argc > 2 ? atof(argv[2]) : 10;
    const int n = 4096; double *A, *B, *C;
    cudaMalloc(&A, (size_t)n * n * 8); cudaMalloc(&B, (size_t)n * n * 8); cudaMalloc(&C, (size_t)n * n * 8);
    double *h = new double[(size_t)n * n]; unsigned long long s = 88172645463325252ull;
    for (size_t i = 0; i < (size_t)n * n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; h[i] = (double)(long long)s / 9.2e18; }
    cudaMemcpy(A, h, (size_t)n * n * 8, cudaMemcpyHostToDevice); cudaMemcpy(B, h, (size_t)n * n * 8, cudaMemcpyHostToDevice);
    bool native = !strcmp(w, "native");
    oz::Plan p{}; if (!native) p = oz::plan(n, n, n, atoi(w + 2));
    auto t0 = std::chrono::steady_clock::now(); double flops = 0, el = 0;
    cudaDeviceProp pr; cudaGetDeviceProperties(&pr, 0);
    while (el < secs) {
        if (native) { fp64_peak<<<pr.multiProcessorCount * 8, 256>>>(C, 2000); flops += 2.0 * 8 * 16 * 2000 * pr.multiProcessorCount * 8 * 256; }
        else { oz::dgemm(p, A, B, C); flops += 2.0 * n * n * n; }
        cudaDeviceSynchronize();
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    printf("%s: %.1f GFLOPS over %.1f s\n", w, flops / el / 1e9, el);
}
