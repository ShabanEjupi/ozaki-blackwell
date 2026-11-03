// emu_harness.h -- one harness for every other FP64 emulation measured beside ours (cublas_emu.cpp: NVIDIA's, gemmul8_emu.cu:
// RIKEN's GEMMul8), so that they see ozaki_test.cu's inputs, its reference and its timing and nothing else differs:
//  * accuracy: M = N = 256, K = 512 / 4096 at exponent spreads 2^0, 2^8, 2^16 and 2^32 (the wide ones: the hard case of
//    integer-sliced emulation), and 200 x 300 x 500 (ozaki_test.cu's cases and
//    seeds); componentwise error against |A||B| in units of 2^-53; the reference sums the exact products (TwoProd) as an
//    unrounded double-double, as ozaki_test.cu does;
//  * speed: n = 2048, 4096, 8192 (ozaki_test.cu's inputs), one warm-up and five timed calls, the median.
// Each configuration is a name and a call computing row-major C (M x N) = A (M x K) B (K x N) on device pointers; a call that
// returns false (no memory for its workspace) is printed as "-".
#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

struct EmuCfg { std::string name; std::function<bool(int M, int N, int K, const double *A, const double *B, double *C)> gemm; };

static void emu_fill(std::vector<double> &v, double spread, unsigned seed) {   // ozaki_test.cu's fill
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> u(-spread, spread);
    for (auto &x : v) x = nd(g) * exp2(u(g));
}

static void emu_accuracy(const std::vector<EmuCfg> &cfg) {
    int M = 256, N = 256;
    printf("accuracy, M=N=%d (componentwise err / unit roundoff 2^-53)\n", M);
    for (int K : {512, 4096, 500})
        for (double spread : {0.0, 8.0, 16.0, 32.0}) {
            if (K == 500) { M = 200; N = 300; if (spread) continue; }
            std::vector<double> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
            emu_fill(A, spread, 1 + K); emu_fill(B, spread, 2 + K);
            std::vector<long double> abs_((size_t)M * N);
            std::vector<double> rs((size_t)M * N), rc((size_t)M * N);   // exact products, double-double sum: hi + lo
            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++) {
                    double x = A[(size_t)i * K + k];
                    for (int j = 0; j < N; j++) {
                        double y = B[(size_t)k * N + j], p = x * y, e = std::fma(x, y, -p), s = rs[(size_t)i * N + j], t = s + p, bp = t - s;
                        rs[(size_t)i * N + j] = t; rc[(size_t)i * N + j] += ((s - (t - bp)) + (p - bp)) + e;
                        abs_[(size_t)i * N + j] += fabsl((long double)x * y);
                    }
                }
            double *dA, *dB, *dC;
            CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, B.size() * 8)); CK(cudaMalloc(&dC, C.size() * 8));
            CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
            printf("%dx%d K=%5d spread 2^±%-2.0f |", M, N, K, spread);
            for (auto &c : cfg) {
                if (!c.gemm(M, N, K, dA, dB, dC)) { printf(" %s -", c.name.c_str()); continue; }
                CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost));
                long double e = 0;
                for (size_t t = 0; t < C.size(); t++) e = fmaxl(e, fabsl(((long double)C[t] - rs[t]) - rc[t]) / abs_[t]);
                printf(" %s %.3g", c.name.c_str(), (double)(e / 0x1p-53L));
            }
            printf("\n"); fflush(stdout);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }
}

static void emu_speed(const std::vector<EmuCfg> &cfg) {
    printf("\nspeed, n x n x n (FP64-equivalent TFLOPS), median of 5 after a warm-up\n");
    for (int n : {2048, 4096, 8192}) {
        std::vector<double> A((size_t)n * n); emu_fill(A, 0, 7);
        double *dA, *dB, *dC;
        CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, A.size() * 8)); CK(cudaMalloc(&dC, A.size() * 8));
        CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        printf("n=%d |", n);
        for (auto &c : cfg) {
            if (!c.gemm(n, n, n, dA, dB, dC)) { printf(" %s -", c.name.c_str()); continue; }
            CK(cudaDeviceSynchronize());
            std::vector<float> ms(5);
            for (auto &x : ms) { cudaEventRecord(t0); c.gemm(n, n, n, dA, dB, dC); cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&x, t0, t1); }
            std::sort(ms.begin(), ms.end());
            printf(" %s %.3f TF (%.1f ms, %.1f-%.1f)", c.name.c_str(), 2.0 * n * n * n / (ms[2] * 1e-3) / 1e12, ms[2], ms[0], ms[4]);
        }
        printf("\n"); fflush(stdout);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
}

// EMU_ENERGY=<name> (03-11-2026): that configuration alone at n = 4096 on bench/energy.cu's inputs, called back to back for EMU_SECS
// seconds (default 12) after a warm-up, as bench/energy.cu runs ours -- bench/energy_emu.sh samples the board power meanwhile.
// Returns false when no configuration has that name.
static bool emu_energy(const std::vector<EmuCfg> &cfg) {
    const char *want = getenv("EMU_ENERGY"); if (!want) return false;
    const double secs = getenv("EMU_SECS") ? atof(getenv("EMU_SECS")) : 12.0; const int n = 4096;
    for (auto &c : cfg) {
        if (c.name != want) continue;
        std::vector<double> A((size_t)n * n); unsigned long long s = 88172645463325252ull;   // energy.cu's inputs, A = B
        for (auto &x : A) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; x = (double)(long long)s / 9.2e18; }
        double *dA, *dB, *dC;
        CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, A.size() * 8)); CK(cudaMalloc(&dC, A.size() * 8));
        CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        if (!c.gemm(n, n, n, dA, dB, dC)) { printf("%s: no workspace\n", want); return true; }
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::steady_clock::now(); double t = 0; long k = 0;
        while (t < secs) { c.gemm(n, n, n, dA, dB, dC); CK(cudaDeviceSynchronize()); k++;
                           t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
        printf("%s: %.1f GFLOPS over %.1f s (%ld calls, n = %d)\n", want, 2.0 * n * n * n * k / t / 1e9, t, k, n);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        return true;
    }
    printf("EMU_ENERGY=%s: no such configuration\n", want); return true;
}
