// mp_test.cu -- arbitrary-precision GEMM from INT8 (Q doubles per number): error against an exact MPFR sum in units of
// 2^-(53 Q), speed, and MPFR GEMM at the same precision on all CPU cores.
// nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp,-march=native -o out_mp bench/mp_test.cu -lqd -lmpfr -lgmp -lgomp
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <mpfr.h>
#include <omp.h>
#include "../src/ozaki_mp.cuh"
#include <qd/qd_real.h>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <int Q> static void fill(std::vector<double> *x, size_t n, int spread, std::mt19937_64 &g) {
    std::uniform_real_distribution<double> u(-1, 1);
    for (size_t i = 0; i < n; i++) {
        x[0][i] = ldexp(u(g), spread ? (int)(g() % (2 * spread + 1)) - spread : 0);
        for (int q = 1; q < Q; q++) x[q][i] = ldexp(x[q - 1][i] * u(g), -53);
    }
}

template <int Q> static void run(std::mt19937_64 &g) {
    printf("\nQ = %d doubles (%d bits): %d slices, %d slice products x 49 INT8 GEMMs\n", Q, 53 * Q, oz::mp_slices<Q>(), oz::mp_pairs<Q>());
    for (int spread : {0, 8}) {
        int M = 128, N = 128, K = 256;
        std::vector<double> A[Q], B[Q];
        for (int q = 0; q < Q; q++) { A[q].resize(M * K); B[q].resize(K * N); }
        fill<Q>(A, M * K, spread, g); fill<Q>(B, K * N, spread, g);
        oz::MpPtr<Q> dA, dB, dC;
        for (int q = 0; q < Q; q++) {
            cudaMalloc(&dA.p[q], M * K * 8); cudaMalloc(&dB.p[q], K * N * 8); cudaMalloc(&dC.p[q], M * N * 8);
            cudaMemcpy(dA.p[q], A[q].data(), M * K * 8, cudaMemcpyHostToDevice); cudaMemcpy(dB.p[q], B[q].data(), K * N * 8, cudaMemcpyHostToDevice);
        }
        oz::MpPlan<Q> pl = oz::mp_plan<Q>(M, N, K);
        oz::mpgemm<Q>(pl, dA, dB, dC);
        cudaError_t e = cudaDeviceSynchronize(); if (e) { printf("%s\n", cudaGetErrorString(e)); exit(1); }
        std::vector<double> C[Q];
        for (int q = 0; q < Q; q++) { C[q].resize(M * N); cudaMemcpy(C[q].data(), dC.p[q], M * N * 8, cudaMemcpyDeviceToHost); }
        double w = 0;
#pragma omp parallel for reduction(max : w)
        for (int i = 0; i < M; i++) {
            mpfr_t s, a, b, t; mpfr_inits2(2400, s, a, b, t, (mpfr_ptr)0);
            for (int j = 0; j < N; j++) {
                mpfr_set_zero(s, 1); double mg = 0;
                for (int k = 0; k < K; k++) {
                    mpfr_set_d(a, A[0][i * K + k], MPFR_RNDN); mpfr_set_d(b, B[0][k * N + j], MPFR_RNDN);
                    for (int q = 1; q < Q; q++) { mpfr_add_d(a, a, A[q][i * K + k], MPFR_RNDN); mpfr_add_d(b, b, B[q][k * N + j], MPFR_RNDN); }
                    mpfr_mul(t, a, b, MPFR_RNDN); mpfr_add(s, s, t, MPFR_RNDN);
                    mg += fabs(A[0][i * K + k] * B[0][k * N + j]);
                }
                for (int q = 0; q < Q; q++) mpfr_sub_d(s, s, C[q][i * N + j], MPFR_RNDN);
                w = fmax(w, fabs(mpfr_get_d(s, MPFR_RNDN)) / mg);
            }
            mpfr_clears(s, a, b, t, (mpfr_ptr)0);
        }
        printf("  K = %d, magnitudes 2^+-%d: max |C - C_exact| / (|A||B|) = %.3g units of 2^-%d\n", K, spread, ldexp(w, 53 * Q), 53 * Q);
        oz::mp_release(pl);
        for (int q = 0; q < Q; q++) { cudaFree(dA.p[q]); cudaFree(dB.p[q]); cudaFree(dC.p[q]); }
    }
    for (int n : {1024, 2048}) {
        oz::MpPtr<Q> d[3];
        std::vector<double> h[Q]; for (int q = 0; q < Q; q++) h[q].resize((size_t)n * n);
        fill<Q>(h, (size_t)n * n, 0, g);
        for (int t = 0; t < 3; t++) for (int q = 0; q < Q; q++) { cudaMalloc(&d[t].p[q], (size_t)n * n * 8); cudaMemcpy(d[t].p[q], h[q].data(), (size_t)n * n * 8, cudaMemcpyHostToDevice); }
        oz::MpPlan<Q> pl = oz::mp_plan<Q>(n, n, n);
        oz::mpgemm<Q>(pl, d[0], d[1], d[2]); cudaDeviceSynchronize();
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
        for (int r = 0; r < 3; r++) oz::mpgemm<Q>(pl, d[0], d[1], d[2]);
        cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 3;
        printf("  GPU n = %d: %.1f ms, %.3f TFLOPS of %d-bit arithmetic\n", n, ms, 2.0 * n * n * n / ms / 1e9, 53 * Q);
        oz::mp_release(pl);
        for (int t = 0; t < 3; t++) for (int q = 0; q < Q; q++) cudaFree(d[t].p[q]);
    }
    if (Q == 4) {   // the QD library (Hida, Li, Bailey): qd_real, the classic CPU 212-bit arithmetic, every core, n = 256 measured
        int n = 256; std::vector<double> h[Q]; for (int q = 0; q < Q; q++) h[q].resize(n * n);
        fill<Q>(h, n * n, 0, g);
        std::vector<qd_real> X(n * n), Y(n * n);
        for (int i = 0; i < n * n; i++) X[i] = qd_real(h[0][i], h[1][i], h[2][i], h[3][i]);
        unsigned int cw; fpu_fix_start(&cw);
        double t0 = now();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) { qd_real s = 0.0; for (int k = 0; k < n; k++) s += X[i * n + k] * X[k * n + j]; Y[i * n + j] = s; }
        double t = now() - t0;
        fpu_fix_end(&cw);
        printf("  QD qd_real GEMM, %d threads: %.5f TFLOPS (n = 2048 would take ~%.1f s)\n", omp_get_max_threads(), 2.0 * n * n * n / t / 1e12, t * 512);
    }
    {   // MPFR GEMM at 53 Q bits, every core, n = 256 measured
        int n = 256; std::vector<double> h[Q]; for (int q = 0; q < Q; q++) h[q].resize(n * n);
        fill<Q>(h, n * n, 0, g);
        std::vector<__mpfr_struct> X(n * n);
        for (int i = 0; i < n * n; i++) { mpfr_init2(&X[i], 53 * Q); mpfr_set_d(&X[i], h[0][i], MPFR_RNDN); for (int q = 1; q < Q; q++) mpfr_add_d(&X[i], &X[i], h[q][i], MPFR_RNDN); }
        double t0 = now();
#pragma omp parallel
        {
            mpfr_t s; mpfr_init2(s, 53 * Q);
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) { mpfr_set_zero(s, 1); for (int k = 0; k < n; k++) mpfr_fma(s, &X[i * n + k], &X[k * n + j], s, MPFR_RNDN); }
            mpfr_clear(s);
        }
        double t = now() - t0;
        printf("  MPFR %d-bit GEMM, %d threads: %.5f TFLOPS (n = 2048 would take ~%.0f s)\n", 53 * Q, omp_get_max_threads(), 2.0 * n * n * n / t / 1e12, t * 512);
        for (auto &x : X) mpfr_clear(&x);
    }
}

int main() {
    std::mt19937_64 g(5);
    run<4>(g); run<6>(g); run<8>(g);
}
