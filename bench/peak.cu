// peak.cu -- what this GPU can really do: native FP64 / FP32 FMA vs INT8 tensor cores driven from PTX.
// The ratio INT8 / FP64 is the room an FP64 emulation (Ozaki scheme) has to work in.
#include <cstdio>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

template <typename T> __global__ void fma_peak(T *out, int n, T a, T b) {
    T x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
    for (int i = 0; i < n; i++) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
            x0 = fma(x0, a, b); x1 = fma(x1, a, b); x2 = fma(x2, a, b); x3 = fma(x3, a, b);
            x4 = fma(x4, a, b); x5 = fma(x5, a, b); x6 = fma(x6, a, b); x7 = fma(x7, a, b);
        }
    }
    T s = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
    if (s == (T)-1.2345) out[0] = s;   // never true, keeps the work alive
}

// m16n8k32 s8*s8 -> s32: 16*8*32 multiply-adds per warp per instruction. Four independent accumulators.
__global__ void imma_peak(int *out, int n, unsigned seed) {
    unsigned a0 = seed ^ threadIdx.x, a1 = a0 * 3, a2 = a0 * 5, a3 = a0 * 7, b0 = a0 * 11, b1 = a0 * 13;
    int c[4][4] = {};
    for (int i = 0; i < n; i++) {
#pragma unroll
        for (int j = 0; j < 16; j++) {
#pragma unroll
            for (int k = 0; k < 4; k++)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                             : "+r"(c[k][0]), "+r"(c[k][1]), "+r"(c[k][2]), "+r"(c[k][3])
                             : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    int s = 0;
    for (int k = 0; k < 4; k++) s += c[k][0] + c[k][1] + c[k][2] + c[k][3];
    if (s == 0x7f3a1b2c) out[0] = s;
}

int main() {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    int blocks = p.multiProcessorCount * 8, threads = 256;
    void *buf; CK(cudaMalloc(&buf, 64));
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    float ms;
    printf("%s, %d SMs, sm_%d%d\n", p.name, p.multiProcessorCount, p.major, p.minor);

    auto run = [&](const char *name, double opsPerIter, int n, auto launch) {
        launch(1); cudaDeviceSynchronize();   // warm-up
        cudaEventRecord(t0); launch(n); cudaEventRecord(t1); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        double t = opsPerIter * n * blocks * threads / (ms * 1e-3) / 1e12;
        printf("%-28s %9.3f T(FL)OPS  (%.1f ms)\n", name, t, ms);
        return t;
    };
    double f64 = run("FP64 FMA (native)", 2.0 * 8 * 16, 2000, [&](int n) { fma_peak<double><<<blocks, threads>>>((double *)buf, n, 0.999, 1e-3); });
    double f32 = run("FP32 FMA (native)", 2.0 * 8 * 16, 20000, [&](int n) { fma_peak<float><<<blocks, threads>>>((float *)buf, n, 0.999f, 1e-3f); });
    // per thread: ops of the warp's mma / 32 lanes
    double i8 = run("INT8 mma.sync m16n8k32", 2.0 * 16 * 8 * 32 / 32 * 4 * 16, 4000, [&](int n) { imma_peak<<<blocks, threads>>>((int *)buf, n, 12345u); });
    CK(cudaGetLastError());
    printf("INT8 / FP64 = %.0fx   FP32 / FP64 = %.0fx\n", i8 / f64, f32 / f64);
    return 0;
}
