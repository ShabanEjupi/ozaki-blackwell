// gemmul8_emu.cu -- RIKEN's GEMMul8 (Ozaki scheme II, the authors' public code, MIT; /opt/ozaki/GEMMul8, 30b5b5d, INT8
// backend, built by bench/gemmul8_build.sh) on the same laptop, measured the way ozaki_test.cu measures ours and
// cublas_emu.cpp NVIDIA's (emu_harness.h: the same cases, reference and timing). GEMMul8's own harness (its test/, run by
// emu_compare.sh as well) uses other matrices and another error measure, so its numbers do not sit in one table with ours;
// these do. Both of its modes (fastmode = true: the cheaper scaling bound; false: the accurate one), 12 to 20 moduli.
// Its API follows cuBLAS (column-major): row-major C = A B is the column-major product C^T = B^T A^T. The workspace is
// sized with gemmul8::workSize and kept (grown on a warm-up call, never inside a timed one).
//   nvcc 13.4 (/opt/ozaki/cu13), -I GEMMul8/include, GEMMul8/lib/libgemmul8.a, cuBLAS/cuBLASLt 13: emu_compare.sh
#include <cublas_v2.h>
#include "gemmul8.hpp"
#include "emu_harness.h"
#define CB(x) do { cublasStatus_t s = (x); if (s) { printf("%s: cublas status %d\n", #x, (int)s); exit(1); } } while (0)

static void *g_work = nullptr; static size_t g_size = 0;
static bool work_for(size_t need) {   // grow-only workspace; false when the card has no room for it
    if (need <= g_size) return true;
    if (g_work) cudaFree(g_work);
    g_work = nullptr; g_size = 0;
    if (cudaMalloc(&g_work, need) != cudaSuccess) { cudaGetLastError(); return false; }
    g_size = need; return true;
}
int main() {
    cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
    cublasHandle_t h; CB(cublasCreate(&h));
    void *cws; CK(cudaMalloc(&cws, 32 << 20)); CB(cublasSetWorkspace(h, cws, 32 << 20));   // gemmul8.hpp's advice: >= 32 MiB
    printf("GEMMul8 (INT8 backend) on %s (sm_%d%d); fast = fastmode, accu = accurate mode; number = moduli\n", pr.name, pr.major, pr.minor);
    std::vector<EmuCfg> cfg;
    for (bool fast : {true, false})
        for (int nm : {12, 14, 16, 18, 20})
            cfg.push_back({std::string(fast ? "fast" : "accu") + std::to_string(nm), [h, fast, nm](int M, int N, int K, const double *A, const double *B, double *C) {
                if (!work_for(gemmul8::workSize<false, gemmul8::Backend::INT8, gemmul8::Func::gemm>(N, M, K, nm))) return false;
                const double one = 1, zero = 0;
                gemmul8::gemm<double, gemmul8::Backend::INT8>(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N, nm, fast, g_work);
                return cudaGetLastError() == cudaSuccess; }});
    if (emu_energy(cfg)) return 0;
    emu_accuracy(cfg);
    emu_speed(cfg);
    if (g_work) cudaFree(g_work);
    cudaFree(cws); cublasDestroy(h);
    return 0;
}
