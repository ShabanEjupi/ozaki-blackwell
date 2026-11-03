// Error is measured componentwise against |A||B|, against an exact-to-2^-80 reference: each product split exactly (TwoProd,
// fma) and summed as a double-double (TwoSum), kept unrounded as hi + lo (Dot2 of Ogita, Rump and Oishi without the final
// rounding: off by at most ~(K 2^-53)^2 |A||B|). The 80-bit (long double) sum used before is printed beside it ("ref80"):
// its own error reaches K 2^-64 |A||B| in the worst case, 2 units of 2^-53 at K = 4096.// ozaki_test.cu -- how exact, and how fast, FP64 from INT8 is next to the GPU's own FP64.
// Error is measured against an 80-bit (long double) CPU reference, componentwise against |A||B|:
//   err = max_ij |C - Cref|_ij / (|A||B|)_ij   -- the bound a classic FP64 GEMM is judged by (~K * 2^-53).
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include "../src/ozaki2.cuh"
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void dgemm_native(const double *A, const double *B, double *C, int M, int N, int K) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= N) return;
    double s = 0;
    for (int k = 0; k < K; k++) s = fma(A[(size_t)i * K + k], B[(size_t)k * N + j], s);
    C[(size_t)i * N + j] = s;
}

static void fill(std::vector<double> &v, double spread, unsigned seed) {   // signs, and magnitudes over 2^±spread
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> u(-spread, spread);
    for (auto &x : v) x = nd(g) * exp2(u(g));
}

int main() {
    int M = 256, N = 256;
    printf("accuracy, M=N=%d (componentwise err / unit roundoff 2^-53)\n", M);
    for (int K : {512, 4096, 500})
        for (double spread : {0.0, 8.0, 16.0, 32.0}) {   // 16, 32 (03-11-2026): wide exponent ranges, the hard case of integer-sliced emulation
            if (K == 500) { M = 200; N = 300; if (spread) continue; }   // nothing a multiple of the 128x128x64 tile
            std::vector<double> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
            fill(A, spread, 1 + K); fill(B, spread, 2 + K);
            std::vector<long double> ref((size_t)M * N), abs_((size_t)M * N);
            std::vector<double> rs((size_t)M * N), rc((size_t)M * N);   // exact products, double-double sum: hi + lo
            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++) {
                    long double a = A[(size_t)i * K + k];
                    for (int j = 0; j < N; j++) {
                        ref[(size_t)i * N + j] += a * B[(size_t)k * N + j];
                        { double x = (double)a, y = B[(size_t)k * N + j], p = x * y, e = std::fma(x, y, -p), s = rs[(size_t)i * N + j], t = s + p, bp = t - s;
                          rs[(size_t)i * N + j] = t; rc[(size_t)i * N + j] += ((s - (t - bp)) + (p - bp)) + e; }
                        abs_[(size_t)i * N + j] += fabsl(a * B[(size_t)k * N + j]);
                    }
                }
            double *dA, *dB, *dC;
            CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, B.size() * 8)); CK(cudaMalloc(&dC, C.size() * 8));
            CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dB, B.data(), B.size() * 8, cudaMemcpyHostToDevice));
            auto err = [&]() {
                CK(cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost));
                long double e = 0;
                for (size_t t = 0; t < C.size(); t++) e = fmaxl(e, fabsl(((long double)C[t] - rs[t]) - rc[t]) / abs_[t]);
                return (double)(e / 0x1p-53L);
            };
            { long double d = 0; for (size_t t = 0; t < ref.size(); t++) d = fmaxl(d, fabsl((ref[t] - rs[t]) - rc[t]) / abs_[t]);
              printf("(ref80 off by %.3g) ", (double)(d / 0x1p-53L)); }
            dgemm_native<<<dim3((N + 127) / 128, M), 128>>>(dA, dB, dC, M, N, K); CK(cudaGetLastError());
            printf("%dx%d K=%5d spread 2^±%-2.0f  native FP64 %8.2f |", M, N, K, spread, err());
            for (int nm : {8, 10, 12, 14, 16}) {
                oz::Plan p = oz::plan(M, N, K, nm);
                oz::dgemm(p, dA, dB, dC); CK(cudaGetLastError());
                printf(" oz%d(%db) %.3g", nm, p.bits, err());
                oz::release(p);
            }
            printf("\n");
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }

    printf("\nspeed, n x n x n (FP64-equivalent TFLOPS; native FP64 peak measured 0.449)\n");
    for (int n : {2048, 4096}) {
        std::vector<double> A((size_t)n * n);
        fill(A, 0, 7);
        double *dA, *dB, *dC;
        CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, A.size() * 8)); CK(cudaMalloc(&dC, A.size() * 8));
        CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        float ms;
        dgemm_native<<<dim3(n / 128, n), 128>>>(dA, dB, dC, n, n, n);
        cudaEventRecord(t0); dgemm_native<<<dim3(n / 128, n), 128>>>(dA, dB, dC, n, n, n); cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1);
        printf("n=%d  native(naive) %.3f TF |", n, 2.0 * n * n * n / (ms * 1e-3) / 1e12);
        for (int nm : {12, 14, 16}) {
            oz::Plan p = oz::plan(n, n, n, nm);
            oz::dgemm(p, dA, dB, dC);
            cudaEventRecord(t0);
            for (int r = 0; r < 3; r++) oz::dgemm(p, dA, dB, dC);
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1); ms /= 3;
            printf(" oz%d %.2f TF (%.1f ms)", nm, 2.0 * n * n * n / (ms * 1e-3) / 1e12, ms);
            oz::release(p);
        }
        printf("\n");
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
