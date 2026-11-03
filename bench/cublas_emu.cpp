// cublas_emu.cpp -- the vendor's own FP64 emulation on the same laptop, measured the way ozaki_test.cu measures ours:
// cuBLAS (CUDA 13.0 Update 2 and later) emulates DGEMM on INT8 tensor cores with fixed-point slices (Ozaki scheme;
// NVIDIA, "Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS", 24 October 2025), compute type
// CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, its Automatic Dynamic Precision (ADP) choosing the mantissa bits that keep FP64
// accuracy (and falling back to native FP64 when it cannot), or a fixed bit count. Host code only, against the pip wheels
// nvidia-cublas / nvidia-cuda-runtime (bench/emu_compare.sh builds it); cuBLAS is column-major, so row-major C = A B is the
// column-major product C^T = B^T A^T.
//  * accuracy: ozaki_test.cu's cases and reference (M = N = 256, K = 512 / 4096 at exponent spreads 2^0 and 2^8, and
//    200 x 300 x 500), componentwise error against |A||B| in units of 2^-53, the reference exact to ~2^-80;
//  * speed: n = 2048, 4096, 8192 (ozaki_test.cu's inputs), one warm-up and five timed calls, the median.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s = (x); if (s) { printf("%s: cublas status %d\n", #x, (int)s); exit(1); } } while (0)

static void fill(std::vector<double> &v, double spread, unsigned seed) {   // ozaki_test.cu's fill
    std::mt19937_64 g(seed);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> u(-spread, spread);
    for (auto &x : v) x = nd(g) * exp2(u(g));
}
struct Mode { const char *name; bool emu; int bits; };   // bits 0: ADP (dynamic)
static const Mode MODES[] = {{"native", false, 0}, {"emu ADP", true, 0}, {"emu 55b", true, 55}, {"emu 47b", true, 47}, {"emu 39b", true, 39}};
static void set_mode(cublasHandle_t h, const Mode &m) {
    if (!m.emu) return;
    CB(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER));
    if (m.bits) { CB(cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
                  CB(cublasSetFixedPointEmulationMaxMantissaBitCount(h, m.bits)); }
    else CB(cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
}
// row-major C (M x N) = A (M x K) B (K x N)
static void gemm(cublasHandle_t h, const Mode &m, int M, int N, int K, const double *A, const double *B, double *C) {
    const double one = 1, zero = 0;
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_64F, N, A, CUDA_R_64F, K, &zero, C, CUDA_R_64F, N,
                    m.emu ? CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT : CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
}
int main() {
    int ver = 0; cublasHandle_t h0; CB(cublasCreate(&h0)); CB(cublasGetVersion(h0, &ver)); CB(cublasDestroy(h0));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    // the workspace the emulation may want (up to 8 GB): the largest of 8, 6, 4, 2 GB that the card gives
    void *ws = nullptr; size_t wsz = 0;
    for (size_t g : {8, 6, 4, 2}) if (cudaMalloc(&ws, g << 30) == cudaSuccess) { wsz = g << 30; break; } else cudaGetLastError();
    printf("cuBLAS %d on %s (sm_%d%d), workspace %zu GB\n", ver, pr.name, pr.major, pr.minor, wsz >> 30);
    std::vector<cublasHandle_t> H(sizeof MODES / sizeof MODES[0]);
    for (size_t i = 0; i < H.size(); i++) { CB(cublasCreate(&H[i])); if (wsz) CB(cublasSetWorkspace(H[i], ws, wsz)); set_mode(H[i], MODES[i]); }
    int M = 256, N = 256;
    printf("accuracy, M=N=%d (componentwise err / unit roundoff 2^-53)\n", M);
    for (int K : {512, 4096, 500})
        for (double spread : {0.0, 8.0}) {
            if (K == 500) { M = 200; N = 300; if (spread) continue; }
            std::vector<double> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
            fill(A, spread, 1 + K); fill(B, spread, 2 + K);
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
            for (size_t i = 0; i < H.size(); i++) {
                gemm(H[i], MODES[i], M, N, K, dA, dB, dC); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(C.data(), dC, C.size() * 8, cudaMemcpyDeviceToHost));
                long double e = 0;
                for (size_t t = 0; t < C.size(); t++) e = fmaxl(e, fabsl(((long double)C[t] - rs[t]) - rc[t]) / abs_[t]);
                printf(" %s %.3g", MODES[i].name, (double)(e / 0x1p-53L));
            }
            printf("\n");
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
        }
    printf("\nspeed, n x n x n (FP64-equivalent TFLOPS), median of 5 after a warm-up\n");
    for (int n : {2048, 4096, 8192}) {
        std::vector<double> A((size_t)n * n); fill(A, 0, 7);
        double *dA, *dB, *dC;
        CK(cudaMalloc(&dA, A.size() * 8)); CK(cudaMalloc(&dB, A.size() * 8)); CK(cudaMalloc(&dC, A.size() * 8));
        CK(cudaMemcpy(dA, A.data(), A.size() * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, A.data(), A.size() * 8, cudaMemcpyHostToDevice));
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        printf("n=%d |", n);
        for (size_t i = 0; i < H.size(); i++) {
            gemm(H[i], MODES[i], n, n, n, dA, dB, dC); CK(cudaDeviceSynchronize());
            std::vector<float> ms(5);
            for (auto &x : ms) { cudaEventRecord(t0); gemm(H[i], MODES[i], n, n, n, dA, dB, dC); cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&x, t0, t1); }
            std::sort(ms.begin(), ms.end());
            printf(" %s %.3f TF (%.1f ms, %.1f-%.1f)", MODES[i].name, 2.0 * n * n * n / (ms[2] * 1e-3) / 1e12, ms[2], ms[0], ms[4]);
        }
        printf("\n");
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    for (auto h : H) cublasDestroy(h);
    if (ws) cudaFree(ws);
    return 0;
}
