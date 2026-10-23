// big_test.cu -- exact big-integer GEMM from INT8 tensor cores (ozaki_big.cuh): bit-exact check against GMP, speed,
// and the CPU at the same job: GMP mpz inner products on every core, and FLINT's fmpz_mat_mul (multimodular + BLAS).
// g++ -O3 -c bench/big_flint.cpp -o out_flint.o && nvcc -O3 -arch=sm_120 -std=c++17 -DHAVE_FLINT -Xcompiler -fopenmp,-march=native \
//   -o out_big bench/big_test.cu out_flint.o -lflint -lgmp -lgomp
// Use: out_big [P_bits n] ...   (pairs; default ladder)
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>
#include <unistd.h>
#include "../src/ozaki_big.cuh"
#ifdef HAVE_FLINT
double flint_gemm(int n, int NL, const unsigned *A, const unsigned *B, int threads);   // big_flint.cpp
#endif
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void to_mpz(mpz_t z, const unsigned *x, int nl) {   // two's complement limbs -> mpz
    mpz_import(z, nl, -1, 4, 0, 0, x);
    if ((int)x[nl - 1] < 0) { mpz_t t; mpz_init_set_ui(t, 1); mpz_mul_2exp(t, t, 32 * nl); mpz_sub(z, z, t); mpz_clear(t); }
}

static void run(int bits, int n, std::mt19937_64 &g) {
    int NL = (bits + 9 + 31) / 32, M = n, N = n, K = n;
    std::vector<unsigned> A((size_t)M * K * NL), B((size_t)K * N * NL);
    for (auto *X : {&A, &B})
        for (size_t e = 0; e < X->size() / NL; e++) {
            for (int k = 0; k < NL - 1; k++) (*X)[e * NL + k] = (unsigned)g();
            (*X)[e * NL + NL - 1] = (unsigned)((int)(g() % (1u << 24)) - (1 << 23));   // |x| < 2^(32 NL - 9)
        }
    double t0 = now();
    oz::BigPlan pl = oz::big_plan(M, N, K, NL);
    double tplan = now() - t0;
    unsigned *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * 4); cudaMalloc(&dB, B.size() * 4); cudaMalloc(&dC, (size_t)M * N * pl.NA * 4);
    cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice);
    oz::biggemm(pl, dA, dB, dC);
    cudaError_t e = cudaDeviceSynchronize(); if (e) { printf("%s\n", cudaGetErrorString(e)); exit(1); }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
    for (int r = 0; r < 3; r++) oz::biggemm(pl, dA, dB, dC);
    cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 3;
    std::vector<unsigned> C((size_t)M * N * pl.NA);
    cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost);
    int bad = 0, checked = 0;
#pragma omp parallel for reduction(+ : bad, checked)
    for (int s = 0; s < 256; s++) {   // 256 random elements, exact
        int i = (int)((s * 2654435761u) % M), j = (int)((s * 40503u + 7) % N);
        mpz_t acc, x, y; mpz_inits(acc, x, y, NULL);
        for (int k = 0; k < K; k++) { to_mpz(x, &A[((size_t)i * K + k) * NL], NL); to_mpz(y, &B[((size_t)k * N + j) * NL], NL); mpz_addmul(acc, x, y); }
        to_mpz(x, &C[((size_t)i * N + j) * pl.NA], pl.NA);
        bad += mpz_cmp(acc, x) != 0; checked++;
        mpz_clears(acc, x, y, NULL);
    }
    double ops = 2.0 * M * N * K;
    printf("\n%d-bit entries (NL %d), n = %d: %d primes, %d bytes each (M ~ 2^%d), %d INT8 GEMMs of n^3 + 2 conversion GEMMs\n",
           32 * NL - 9, NL, n, pl.L, pl.NB, (int)(pl.NA * 32), pl.NB * pl.NB * pl.L);
    printf("  GPU: %.1f ms = %.3g big multiply-adds/s (plan %.2f s); exact: %d of %d sampled elements wrong\n", ms, ops / 2 / ms * 1e3, tplan, bad, checked);
    if (getenv("OZ_ENERGY")) {   // energy: marks (epoch s) around ~8 s of back-to-back GPU products, then around FLINT, idle gaps between
        auto mark = [](const char *w) { printf("MARK %s %.3f\n", w, std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()); fflush(stdout); };
        int reps = std::max(1, (int)(8000 / ms));
        mark("gpu-start"); for (int r = 0; r < reps; r++) oz::biggemm(pl, dA, dB, dC); cudaDeviceSynchronize(); mark("gpu-end");
        printf("  GPU %d products in the window\n", reps);
#ifdef HAVE_FLINT
        sleep(4); mark("flint-start"); double tf = 0; int fr = 0; while (tf < 8) { tf += flint_gemm(n, NL, A.data(), B.data(), omp_get_max_threads()); fr++; } mark("flint-end");
        printf("  FLINT %d products, %.2f s each\n", fr, tf / fr);
#endif
        oz::big_release(pl); cudaFree(dA); cudaFree(dB); cudaFree(dC); return;
    }
    if (getenv("OZ_GPUONLY")) { oz::big_release(pl); cudaFree(dA); cudaFree(dB); cudaFree(dC); return; }
    {   // GMP mpz inner products, every core, on an nc^3 corner, scaled to n^3
        int nc = std::min(n, bits > 8000 ? 48 : 96);
        double tg0 = now();
#pragma omp parallel
        {
            mpz_t acc, x, y; mpz_inits(acc, x, y, NULL);
#pragma omp for schedule(dynamic)
            for (int ij = 0; ij < nc * nc; ij++) {
                int i = ij / nc, j = ij % nc;
                mpz_set_ui(acc, 0);
                for (int k = 0; k < K; k++) { to_mpz(x, &A[((size_t)i * K + k) * NL], NL); to_mpz(y, &B[((size_t)k * N + j) * NL], NL); mpz_addmul(acc, x, y); }
            }
            mpz_clears(acc, x, y, NULL);
        }
        double tg = (now() - tg0) * ((double)n * n) / ((double)nc * nc);
        printf("  GMP mpz_addmul, %d threads (%dx%d of the outputs, scaled): %.2f s -> GPU %.0fx\n", omp_get_max_threads(), nc, nc, tg, tg / (ms / 1e3));
    }
#ifdef HAVE_FLINT
    {   // FLINT fmpz_mat_mul (multimodular, BLAS-style kernels), every core, at the full size
        double tf = flint_gemm(n, NL, A.data(), B.data(), omp_get_max_threads());
        printf("  FLINT fmpz_mat_mul, %d threads: %.2f s -> GPU %.0fx\n", omp_get_max_threads(), tf, tf / (ms / 1e3));
    }
#endif
    oz::big_release(pl); cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char **argv) {
    std::mt19937_64 g(11);
    if (argc >= 3) { for (int i = 1; i + 1 < argc; i += 2) run(atoi(argv[i]), atoi(argv[i + 1]), g); return 0; }
    int ladder[][2] = {{1015, 1024}, {4087, 512}, {16375, 256}, {32759, 256}};
    for (auto &r : ladder) run(r[0], r[1], g);
}
