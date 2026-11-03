// dd_test.cu -- double-double GEMM from INT8: error in units of 2^-106 against the exact product (a 600-bit MPFR sum peeled
// into three doubles; the __float128 sum used before, ref_quad.cpp, is printed beside it as "quad off by": its own error
// reaches K 2^-113 |A||B| = K/128 units in the worst case), and speed.
// g++ -O3 -c bench/ref_quad.cpp -o out_rq.o && nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp -o out_dd bench/dd_test.cu out_rq.o -lquadmath -lmpfr -lgmp -lgomp
#include <cstdio>
#include <cmath>
#include <mpfr.h>
#include <random>
#include <vector>
#include "../src/ozaki_dd.cuh"
void ref_quad(const double *, const double *, const double *, const double *, int, int, int, double *, double *, double *);

template <int NM> void dd_run(oz::Plan &p, double **d) { oz::ddgemm_t<NM>(p, d[0], d[1], d[2], d[3], d[4], d[5]); }
static void dd_any(int nm, oz::Plan &p, double **d) {
    switch (nm) { case 16: dd_run<16>(p, d); break; case 20: dd_run<20>(p, d); break; case 24: dd_run<24>(p, d); break;
                  case 28: dd_run<28>(p, d); break; default: dd_run<32>(p, d); }
}

int main() {
    const int M = 128, N = 128;
    printf("double-double GEMM, componentwise err / (|A||B| * 2^-106)\n");
    for (int K : {256, 1024})
        for (double spread : {0.0, 8.0}) {
            std::mt19937_64 g(K + (int)spread);
            std::normal_distribution<double> nd; std::uniform_real_distribution<double> u(-spread, spread);
            std::vector<double> h[4] = {std::vector<double>((size_t)M * K), std::vector<double>((size_t)M * K),
                                        std::vector<double>((size_t)K * N), std::vector<double>((size_t)K * N)};
            for (int t : {0, 2})
                for (size_t i = 0; i < h[t].size(); i++) {   // a full 106-bit value: hi + a lo just under its ulp
                    h[t][i] = nd(g) * exp2(u(g));
                    h[t + 1][i] = h[t][i] * 0x1p-54 * (2 * (g() >> 11) * 0x1p-53 - 1);
                }
            std::vector<double> Rh((size_t)M * N), Rl(Rh.size()), Ab(Rh.size()), Ch(Rh.size()), Cl(Rh.size());
            ref_quad(h[0].data(), h[1].data(), h[2].data(), h[3].data(), M, N, K, Rh.data(), Rl.data(), Ab.data());
            std::vector<double> R3(Rh.size() * 3);   // the exact C: 600 bits hold every product and the sum; peeled into 3 doubles
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < M; i++) {
                mpfr_t s, a, b, t; mpfr_inits2(600, s, a, b, t, (mpfr_ptr)0);
                for (int j = 0; j < N; j++) {
                    mpfr_set_zero(s, 1);
                    for (int k = 0; k < K; k++) {
                        mpfr_set_d(a, h[0][(size_t)i * K + k], MPFR_RNDN); mpfr_add_d(a, a, h[1][(size_t)i * K + k], MPFR_RNDN);
                        mpfr_set_d(b, h[2][(size_t)k * N + j], MPFR_RNDN); mpfr_add_d(b, b, h[3][(size_t)k * N + j], MPFR_RNDN);
                        mpfr_mul(t, a, b, MPFR_RNDN); mpfr_add(s, s, t, MPFR_RNDN);
                    }
                    for (int c = 0; c < 3; c++) { double v = mpfr_get_d(s, MPFR_RNDN); R3[((size_t)i * N + j) * 3 + c] = v; mpfr_sub_d(s, s, v, MPFR_RNDN); }
                }
                mpfr_clears(s, a, b, t, (mpfr_ptr)0);
            }
            { double q = 0; for (size_t t = 0; t < Rh.size(); t++) q = fmax(q, fabs(((Rh[t] - R3[3 * t]) + (Rl[t] - R3[3 * t + 1])) - R3[3 * t + 2]) / (Ab[t] * 0x1p-106));
              printf("(quad off by %.3g) ", q); }
            double *d[6];
            for (int t = 0; t < 4; t++) { cudaMalloc(&d[t], h[t].size() * 8); cudaMemcpy(d[t], h[t].data(), h[t].size() * 8, cudaMemcpyHostToDevice); }
            cudaMalloc(&d[4], Rh.size() * 8); cudaMalloc(&d[5], Rh.size() * 8);
            printf("K=%5d spread 2^±%-2.0f |", K, spread);
            for (int nm : {16, 20, 24, 28, 32}) {
                oz::Plan p = oz::plan(M, N, K, nm, 106);
                dd_any(nm, p, d);
                cudaError_t e = cudaGetLastError(); if (e) { printf(" %s\n", cudaGetErrorString(e)); return 1; }
                cudaMemcpy(Ch.data(), d[4], Ch.size() * 8, cudaMemcpyDeviceToHost); cudaMemcpy(Cl.data(), d[5], Cl.size() * 8, cudaMemcpyDeviceToHost);
                double worst = 0;
                for (size_t t = 0; t < Ch.size(); t++) worst = fmax(worst, fabs(((Ch[t] - R3[3 * t]) + (Cl[t] - R3[3 * t + 1])) - R3[3 * t + 2]) / (Ab[t] * 0x1p-106));
                printf(" dd%d(%db) %.3g", nm, p.bits, worst);
                oz::release(p);
            }
            printf("\n");
            for (auto x : d) cudaFree(x);
        }

    printf("\nspeed (double-double-equivalent TFLOPS)\n");
    for (int n : {2048, 4096}) {
        double *d[6];
        for (auto &x : d) { cudaMalloc(&x, (size_t)n * n * 8); cudaMemset(x, 0x3f, (size_t)n * n * 8); }
        for (int nm : {24, 32}) {
            oz::Plan p = oz::plan(n, n, n, nm, 106);
            dd_any(nm, p, d);
            cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaEventRecord(t0); for (int r = 0; r < 3; r++) dd_any(nm, p, d); cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= 3;
            printf("n=%d dd%d: %.2f TF (%.1f ms)\n", n, nm, 2.0 * n * n * n / (ms * 1e-3) / 1e12, ms);
            oz::release(p);
        }
        for (auto x : d) cudaFree(x);
    }
}
