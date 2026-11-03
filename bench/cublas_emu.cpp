// cublas_emu.cpp -- the vendor's own FP64 emulation on the same laptop, measured the way ozaki_test.cu measures ours:
// cuBLAS (CUDA 13.0 Update 2 and later) emulates DGEMM on INT8 tensor cores with fixed-point slices (Ozaki scheme;
// NVIDIA, "Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS", 24 October 2025), compute type
// CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, its Automatic Dynamic Precision (ADP) choosing the mantissa bits that keep FP64
// accuracy (and falling back to native FP64 when it cannot), or a fixed bit count. Host code only, against the pip wheels
// nvidia-cublas / nvidia-cuda-runtime (bench/emu_compare.sh builds it); cuBLAS is column-major, so row-major C = A B is the
// column-major product C^T = B^T A^T. Cases, reference and timing: emu_harness.h (shared with gemmul8_emu.cu).
#include <cublas_v2.h>
#include "emu_harness.h"
#define CB(x) do { cublasStatus_t s = (x); if (s) { printf("%s: cublas status %d\n", #x, (int)s); exit(1); } } while (0)

struct Mode { const char *name; bool emu; int bits; };   // bits 0: ADP (dynamic)
static const Mode MODES[] = {{"native", false, 0}, {"emu ADP", true, 0}, {"emu 55b", true, 55}, {"emu 47b", true, 47}, {"emu 39b", true, 39}};
static void set_mode(cublasHandle_t h, const Mode &m) {
    if (!m.emu) return;
    CB(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER));
    if (m.bits) { CB(cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED));
                  CB(cublasSetFixedPointEmulationMaxMantissaBitCount(h, m.bits)); }
    else CB(cublasSetFixedPointEmulationMantissaControl(h, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
}
int main() {
    int ver = 0; cublasHandle_t h0; CB(cublasCreate(&h0)); CB(cublasGetVersion(h0, &ver)); CB(cublasDestroy(h0));
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    // the workspace the emulation may want (up to 8 GB): the largest of 8, 6, 4, 2 GB that the card gives
    void *ws = nullptr; size_t wsz = 0;
    for (size_t g : {8, 6, 4, 2}) if (cudaMalloc(&ws, g << 30) == cudaSuccess) { wsz = g << 30; break; } else cudaGetLastError();
    printf("cuBLAS %d on %s (sm_%d%d), workspace %zu GB\n", ver, pr.name, pr.major, pr.minor, wsz >> 30);
    std::vector<cublasHandle_t> H(sizeof MODES / sizeof MODES[0]);
    std::vector<EmuCfg> cfg;
    for (size_t i = 0; i < H.size(); i++) {
        CB(cublasCreate(&H[i])); if (wsz) CB(cublasSetWorkspace(H[i], ws, wsz)); set_mode(H[i], MODES[i]);
        cublasHandle_t h = H[i]; bool emu = MODES[i].emu;
        cfg.push_back({MODES[i].name, [h, emu](int M, int N, int K, const double *A, const double *B, double *C) {   // row-major C = A B
            const double one = 1, zero = 0;
            CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_64F, N, A, CUDA_R_64F, K, &zero, C, CUDA_R_64F, N,
                            emu ? CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT : CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT));
            return true; }});
    }
    emu_accuracy(cfg);
    emu_speed(cfg);
    for (auto h : H) cublasDestroy(h);
    if (ws) cudaFree(ws);
    return 0;
}
