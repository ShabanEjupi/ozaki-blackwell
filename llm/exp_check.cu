// exp_check.cu -- is the GPU's exp() the host C library's on every input the MoE backward gives it? swiglu_back computes
// s = 1 / (1 + exp(-g)) for g the double of an FP32 gate output, on the host (glibc). That loop moves to the GPU with every
// bit kept only if exp(-(double)f) is the same double on both for every FP32 f -- checked here for all 2^32 of them, chunk by
// chunk (NaN inputs: NaN on both counts as equal). The other operations of the formula are single IEEE operations, the same
// on both when nothing is contracted into an FMA (__dmul_rn / __dadd_rn on the device; the host build has no FMA).
//   nvcc -O3 -arch=sm_120a -Xcompiler -fopenmp -o out_exp_check llm/exp_check.cu -lgomp && ./out_exp_check
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
__global__ void gexp(uint32_t base, size_t n, double *o) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t u = base + (uint32_t)i; float f; memcpy(&f, &u, 4); o[i] = exp(-(double)f); }
}
int main() {
    const size_t C = (size_t)1 << 26; double *d; std::vector<double> h(C);
    if (cudaMalloc(&d, C * 8) != cudaSuccess) { fprintf(stderr, "cudaMalloc\n"); return 1; }
    unsigned long long bad = 0, nan = 0, fin = 0, shown = 0, hcl = 0, gcl = 0;   // on a mismatch: closer to expl (64-bit), host or GPU
    for (uint64_t b = 0; b < (1ull << 32); b += C) {
        gexp<<<4096, 256>>>((uint32_t)b, C, d);
        if (cudaMemcpy(h.data(), d, C * 8, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "copy\n"); return 1; }
        #pragma omp parallel for reduction(+ : bad, nan, fin, hcl, gcl) schedule(static)
        for (size_t i = 0; i < C; i++) {
            uint32_t u = (uint32_t)(b + i); float f; memcpy(&f, &u, 4); double r = exp(-(double)f);
            if (r != r && h[i] != h[i]) { nan++; continue; }
            if (std::isfinite(r) && r != 0) fin++;
            if (memcmp(&r, &h[i], 8)) {
                bad++; long double x = expl(-(long double)f), eh = fabsl((long double)r - x), eg = fabsl((long double)h[i] - x);
                hcl += eh < eg; gcl += eg < eh;
                #pragma omp critical
                if (shown < 20) { shown++; printf("  f %a (0x%08x): host %a, GPU %a\n", (double)f, u, r, h[i]); }
            }
        }
    }
    printf("exp_check: %llu of 2^32 FP32 inputs give a different exp(-x) (%llu finite nonzero results, %llu NaN on both); "
           "nearer to expl: the host %llu, the GPU %llu\n", bad, fin, nan, hcl, gcl);
    return bad != 0;
}
