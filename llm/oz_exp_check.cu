// oz_exp_check.cu -- proves llm/oz_exp.cuh bit for bit: host oz_exp = the host's exp(), and GPU oz_exp = host oz_exp,
//   set 0: every one of the 2^32 FP32 f at x = -(double)f (the MoE backward's swiglu_back and the router's sigmoid);
//   set 1: 2^32 doubles from a fixed hash with |x| < 2^10, exponents 2^-60 .. 2^9 equally likely (the softmax's exp(p - max));
//   set 2: 2^32 doubles from a fixed hash, any bit pattern (limits, subnormal results, overflow, inf, nan).
// NaN = NaN counts as equal (the payload is not compared). Also times the three: host libm, host oz_exp, GPU oz_exp.
//   nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_oz_exp_check llm/oz_exp_check.cu -lgomp && ./out_oz_exp_check [sets]
#include "oz_exp.cuh"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <omp.h>

__host__ __device__ inline uint64_t ozc_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
__host__ __device__ inline double ozc_input(int set, uint64_t i) {
    if (set == 0) {
        uint32_t u = (uint32_t)i; float f;
#ifdef __CUDA_ARCH__
        f = __uint_as_float(u);
#else
        memcpy(&f, &u, 4);
#endif
        return -(double)f;
    }
    uint64_t h = ozc_mix(i ^ ((uint64_t)set << 40));
    if (set == 1) return ozx_dbl((h & 0x800fffffffffffffull) | ((uint64_t)(1023 - 60 + (h >> 52 & 0x7ff) % 70) << 52));
    return ozx_dbl(h);
}
__global__ void ozc_kernel(int set, uint64_t base, size_t n, double *o) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        o[i] = oz_exp(ozc_input(set, base + i));
}
static bool same(double a, double b) { return ozx_bits(a) == ozx_bits(b) || (a != a && b != b); }

int main(int argc, char **argv) {
    const char *sets = argc > 1 ? argv[1] : "012";
    const size_t C = (size_t)1 << 23; double *d; std::vector<double> h(C);   // 64 MB: runs beside the training
    if (cudaMalloc(&d, C * 8) != cudaSuccess) { fprintf(stderr, "cudaMalloc\n"); return 1; }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int fails = 0;
    for (const char *s = sets; *s; s++) {
        int set = *s - '0';
        unsigned long long dlib = 0, dgpu = 0, nan = 0, shown = 0; double tg = 0, tlib = 0, toz = 0;
        for (uint64_t b = 0; b < (1ull << 32); b += C) {
            cudaEventRecord(e0); ozc_kernel<<<4096, 256>>>(set, b, C, d); cudaEventRecord(e1);
            if (cudaMemcpy(h.data(), d, C * 8, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "copy\n"); return 1; }
            float ms; cudaEventElapsedTime(&ms, e0, e1); tg += ms / 1e3;
            double sl = 0, so = 0;                     // sums keep the timed loops from being optimised away
            auto t0 = std::chrono::steady_clock::now();
            #pragma omp parallel for reduction(+ : sl) schedule(static)
            for (size_t i = 0; i < C; i++) { double y = exp(ozc_input(set, b + i)); sl += y == y ? (double)(ozx_bits(y) & 1) : 0; }
            auto t1 = std::chrono::steady_clock::now();
            #pragma omp parallel for reduction(+ : so) schedule(static)
            for (size_t i = 0; i < C; i++) { double y = oz_exp(ozc_input(set, b + i)); so += y == y ? (double)(ozx_bits(y) & 1) : 0; }
            auto t2 = std::chrono::steady_clock::now();
            tlib += std::chrono::duration<double>(t1 - t0).count(); toz += std::chrono::duration<double>(t2 - t1).count();
            #pragma omp parallel for reduction(+ : dlib, dgpu, nan) schedule(static)
            for (size_t i = 0; i < C; i++) {
                double x = ozc_input(set, b + i), l = exp(x), o = oz_exp(x);
                if (o != o) nan++;
                bool a = same(o, l), g = same(h[i], o);
                if (!a) dlib++;
                if (!g) dgpu++;
                if ((!a || !g) && shown < 8) {
                    #pragma omp critical
                    if (shown < 8) { shown++; printf("  set %d x=%a libm=%a oz=%a gpu=%a\n", set, x, l, o, h[i]); }
                }
            }
            if (sl < 0 || so < 0) printf("?");
        }
        printf("set %d: 2^32 inputs, host oz_exp != host exp(): %llu, GPU != host: %llu (NaN results %llu); "
               "time host libm %.1f s, host oz_exp %.1f s (%d threads), GPU %.2f s\n",
               set, dlib, dgpu, nan, tlib, toz, omp_get_max_threads(), tg);
        fflush(stdout);
        if (dlib || dgpu) fails++;
    }
    printf(fails ? "FAILED\n" : "all equal\n");
    return fails ? 1 : 0;
}
