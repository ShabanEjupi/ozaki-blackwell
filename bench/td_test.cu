// td_test.cu -- triple-double GEMM from INT8: error in units of 2^-159 against an exact MPFR sum, speed, and the same
// GEMM in MPFR at 160 bits on all CPU cores (what one would run today for this precision).
// nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp -o out_td bench/td_test.cu -lmpfr -lgmp -lgomp
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <mpfr.h>
#include <omp.h>
#include "../src/ozaki_td.cuh"
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Triple-double arithmetic on the CPU, QD-style: the strongest software baseline for ~159 bits (MPFR is far slower).
// s + a b down to the order-2^-106 products; the order-2^-159 terms are dropped, as a triple-double library does.
static inline void ts(double a, double b, double &s, double &e) { s = a + b; double v = s - a; e = (a - (s - v)) + (b - v); }
static inline void td_fma(double &sx, double &sy, double &sz, double ax, double ay, double az, double bx, double by, double bz) {
    double p0 = ax * bx, q0 = std::fma(ax, bx, -p0), p1 = ax * by, q1 = std::fma(ax, by, -p1), p2 = ay * bx, q2 = std::fma(ay, bx, -p2);
    double t0, e0, u, f1, f2, f3, f4;
    ts(sx, p0, t0, e0);
    ts(sy, q0, u, f1); ts(u, p1, u, f2); ts(u, p2, u, f3); ts(u, e0, u, f4);
    double w = sz + ax * bz + ay * by + az * bx + q1 + q2 + f1 + f2 + f3 + f4;
    double h, r1, r2;
    ts(t0, u, h, r1); ts(r1, w, r1, r2); ts(h, r1, sx, sy); sz = r2;
}

// random triple-doubles: hi in +-[2^-e, 1) * 2^e', mid and lo the next 53-bit pieces
static void fill(std::vector<double> &h, std::vector<double> &m, std::vector<double> &l, int spread, std::mt19937_64 &g) {
    std::uniform_real_distribution<double> u(-1, 1);
    for (size_t i = 0; i < h.size(); i++) {
        h[i] = ldexp(u(g), spread ? (int)(g() % (2 * spread + 1)) - spread : 0);
        m[i] = ldexp(h[i] * u(g), -53); l[i] = ldexp(m[i] * u(g), -53);
    }
}

int main() {
    std::mt19937_64 g(1);
    printf("accuracy: max_ij |C - C_exact| / (|A||B|)_ij in units of 2^-159, M = N = 128\n");
    printf("%6s %6s %10s %10s %10s\n", "K", "spread", "41 moduli", "45 moduli", "49 moduli");
    for (int K : {256, 1024}) for (int spread : {0, 8}) {
        int M = 128, N = 128;
        std::vector<double> Ah(M * K), Am(M * K), Al(M * K), Bh(K * N), Bm(K * N), Bl(K * N);
        fill(Ah, Am, Al, spread, g); fill(Bh, Bm, Bl, spread, g);
        // exact C and |A||B| with MPFR (600 bits holds every product and the sum exactly)
        std::vector<double> ref(M * N * 3), mag(M * N);
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < M; i++) {
            mpfr_t s, a, b, t, ab; mpfr_inits2(600, s, a, b, t, ab, (mpfr_ptr)0);
            for (int j = 0; j < N; j++) {
                mpfr_set_zero(s, 1); double mg = 0;
                for (int k = 0; k < K; k++) {
                    mpfr_set_d(a, Ah[i * K + k], MPFR_RNDN); mpfr_add_d(a, a, Am[i * K + k], MPFR_RNDN); mpfr_add_d(a, a, Al[i * K + k], MPFR_RNDN);
                    mpfr_set_d(b, Bh[k * N + j], MPFR_RNDN); mpfr_add_d(b, b, Bm[k * N + j], MPFR_RNDN); mpfr_add_d(b, b, Bl[k * N + j], MPFR_RNDN);
                    mpfr_mul(t, a, b, MPFR_RNDN); mpfr_add(s, s, t, MPFR_RNDN);
                    mg += fabs(Ah[i * K + k] * Bh[k * N + j]);
                }
                for (int c = 0; c < 3; c++) { double d = mpfr_get_d(s, MPFR_RNDN); ref[(i * N + j) * 3 + c] = d; mpfr_sub_d(s, s, d, MPFR_RNDN); }
                mag[i * N + j] = mg;
            }
            mpfr_clears(s, a, b, t, ab, (mpfr_ptr)0);
        }
        double *d[9]; for (auto &x : d) cudaMalloc(&x, (size_t)K * 128 * 8);
        cudaMemcpy(d[0], Ah.data(), M * K * 8, cudaMemcpyHostToDevice); cudaMemcpy(d[1], Am.data(), M * K * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d[2], Al.data(), M * K * 8, cudaMemcpyHostToDevice); cudaMemcpy(d[3], Bh.data(), K * N * 8, cudaMemcpyHostToDevice);
        cudaMemcpy(d[4], Bm.data(), K * N * 8, cudaMemcpyHostToDevice); cudaMemcpy(d[5], Bl.data(), K * N * 8, cudaMemcpyHostToDevice);
        printf("%6d %6d", K, spread);
        for (int nm : {41, 45, 49}) {
            oz::Plan p = oz::plan(M, N, K, nm, 160);
            if (nm == 41) oz::tdgemm_t<41>(p, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
            else if (nm == 45) oz::tdgemm_t<45>(p, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
            else oz::tdgemm_t<49>(p, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
            cudaError_t e = cudaDeviceSynchronize(); if (e) { printf(" %s\n", cudaGetErrorString(e)); return 1; }
            std::vector<double> c(M * N * 3);
            cudaMemcpy(c.data(), d[6], M * N * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(c.data() + M * N, d[7], M * N * 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(c.data() + 2 * M * N, d[8], M * N * 8, cudaMemcpyDeviceToHost);
            double w = 0;
            mpfr_t e_; mpfr_init2(e_, 600);
            for (int q = 0; q < M * N; q++) {   // (ref - c) exactly in MPFR
                mpfr_set_d(e_, ref[q * 3], MPFR_RNDN); mpfr_add_d(e_, e_, ref[q * 3 + 1], MPFR_RNDN); mpfr_add_d(e_, e_, ref[q * 3 + 2], MPFR_RNDN);
                mpfr_sub_d(e_, e_, c[q], MPFR_RNDN); mpfr_sub_d(e_, e_, c[M * N + q], MPFR_RNDN); mpfr_sub_d(e_, e_, c[2 * M * N + q], MPFR_RNDN);
                w = fmax(w, fabs(mpfr_get_d(e_, MPFR_RNDN)) / mag[q]);
            }
            mpfr_clear(e_);
            if (getenv("TD_DEBUG") && nm == 49) for (int q = 0; q < 2; q++) printf("\n  ref %.17g %.17g %.17g\n  gpu %.17g %.17g %.17g", ref[q * 3], ref[q * 3 + 1], ref[q * 3 + 2], c[q], c[M * N + q], c[2 * M * N + q]);
            printf(" %10.3g", ldexp(w, 159));
            oz::release(p);
        }
        printf("\n");
        for (auto x : d) cudaFree(x);
    }

    printf("\nspeed: triple-double GEMM, 49 moduli (TFLOPS of triple-double, 2 n^3 / s)\n");
    for (int n : {2048, 4096}) {
        double *d[9]; for (auto &x : d) cudaMalloc(&x, (size_t)n * n * 8);
        std::vector<double> h((size_t)n * n), m(h.size()), l(h.size());
        fill(h, m, l, 0, g);
        for (int t = 0; t < 6; t++) cudaMemcpy(d[t], (t % 3 == 0 ? h : t % 3 == 1 ? m : l).data(), h.size() * 8, cudaMemcpyHostToDevice);
        oz::Plan p = oz::plan(n, n, n, 49, 160);
        oz::tdgemm_t<49>(p, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]); cudaDeviceSynchronize();
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a);
        for (int r = 0; r < 3; r++) oz::tdgemm_t<49>(p, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]);
        cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); ms /= 3;
        printf("n = %d: %.1f ms, %.3f TFLOPS\n", n, ms, 2.0 * n * n * n / ms / 1e9);
        oz::release(p); for (auto x : d) cudaFree(x);
    }

    {   // triple-double on the CPU, every core: loop i, k, j so the j loop vectorizes; n = 512 measured, scaled by n^3
        int n = 512; std::vector<double> h(n * n), m(h.size()), l(h.size()); fill(h, m, l, 0, g);
        std::vector<double> Cx(h.size()), Cy(h.size()), Cz(h.size());
        double t0 = now();
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double *sx = &Cx[(size_t)i * n], *sy = &Cy[(size_t)i * n], *sz = &Cz[(size_t)i * n];
            for (int j = 0; j < n; j++) sx[j] = sy[j] = sz[j] = 0;
            for (int k = 0; k < n; k++) {
                double ax = h[i * n + k], ay = m[i * n + k], az = l[i * n + k];
                const double *bx = &h[(size_t)k * n], *by = &m[(size_t)k * n], *bz = &l[(size_t)k * n];
#pragma omp simd
                for (int j = 0; j < n; j++) td_fma(sx[j], sy[j], sz[j], ax, ay, az, bx[j], by[j], bz[j]);
            }
        }
        double t = now() - t0;
        printf("CPU triple-double GEMM, %d threads: n = 512 %.3f s = %.4f TFLOPS (n = 4096 would take ~%.1f s)\n",
               omp_get_max_threads(), t, 2.0 * n * n * n / t / 1e12, t * 512);
        double c = 0; for (double v : Cx) c += v; if (c == 12345.678) printf(" ");   // keep the result alive
    }
    // MPFR at 160 bits, every core: n = 256 measured, scaled by n^3
    {
        int n = 256; std::vector<double> h(n * n), m(h.size()), l(h.size()); fill(h, m, l, 0, g);
        std::vector<__mpfr_struct> A(n * n), B(n * n);   // mpfr_t is an array type
        for (int i = 0; i < n * n; i++) {
            mpfr_init2(&A[i], 160); mpfr_init2(&B[i], 160);
            mpfr_set_d(&A[i], h[i], MPFR_RNDN); mpfr_add_d(&A[i], &A[i], m[i], MPFR_RNDN); mpfr_add_d(&A[i], &A[i], l[i], MPFR_RNDN);
            mpfr_set(&B[i], &A[i], MPFR_RNDN);
        }
        double t0 = now();
#pragma omp parallel
        {
            mpfr_t s, t; mpfr_inits2(160, s, t, (mpfr_ptr)0);
#pragma omp for schedule(static)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    mpfr_set_zero(s, 1);
                    for (int k = 0; k < n; k++) mpfr_fma(s, &A[i * n + k], &B[k * n + j], s, MPFR_RNDN);
                }
            mpfr_clears(s, t, (mpfr_ptr)0);
        }
        double t = now() - t0;
        printf("MPFR 160-bit GEMM, %d threads: n = 256 %.2f s = %.4f TFLOPS (n = 4096 would take ~%.0f s)\n",
               omp_get_max_threads(), t, 2.0 * n * n * n / t / 1e12, t * 4096.0 * 4096 * 4096 / ((double)n * n * n));
        for (int i = 0; i < n * n; i++) { mpfr_clear(&A[i]); mpfr_clear(&B[i]); }
    }
}
