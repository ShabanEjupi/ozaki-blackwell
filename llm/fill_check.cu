// fill_check.cu -- oz_bf.cuh on the GPU against the host, bit for bit: oz_bfb for 2^32 hashed doubles (normal, subnormal,
// zero, inf, nan) and every FP32 value; oz_swiglu_bf for 2^31 hashed (g, u, y) triples (realistic and any bit pattern)
// against jais_lora.cu's fill_gu formula written as it is there, with ultra.cu's bf(). NaN = NaN counts as equal.
//   nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_fill_check llm/fill_check.cu -lgomp && ./out_fill_check
#define ULTRA_LIB
#include "ultra.cu"
#include "oz_bf.cuh"
#include <omp.h>
__host__ __device__ inline uint64_t fc_mix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
__host__ __device__ inline float fc_f(uint64_t h, int set) {   // set 0: |x| ~ 2^-12 .. 2^5 with sign; set 1: any bit pattern
    uint32_t u = (uint32_t)h;
    if (set == 0) u = (u & 0x807fffffu) | ((uint32_t)(127 - 12 + (h >> 40) % 18) << 23);
    float f;
#ifdef __CUDA_ARCH__
    f = __uint_as_float(u);
#else
    memcpy(&f, &u, 4);
#endif
    return f;
}
__global__ void k_bfb(uint64_t base, size_t n, uint16_t *o) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint64_t h = fc_mix(base + i); o[i] = oz_bfb(ozx_dbl((base + i) & 1 ? h : (h & 0x800fffffffffffffull) | ((uint64_t)(1023 - 140 + (h >> 52 & 0x7ff) % 160) << 52))); }
}
__global__ void k_sw(int set, uint64_t base, size_t n, uint16_t *o) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint64_t h = fc_mix(base + i), h2 = fc_mix(h), h3 = fc_mix(h2);
        oz_swiglu_bf(fc_f(h, set), fc_f(h2, set), fc_f(h3, set) * (set ? 1.f : 1e-3f), o[2 * i], o[2 * i + 1]); }
}
static bool same16(uint16_t a, uint16_t b) { return a == b || (((a & 0x7f80) == 0x7f80 && (a & 0x7f)) && ((b & 0x7f80) == 0x7f80 && (b & 0x7f))); }
int main() {
    const size_t C = (size_t)1 << 24; uint16_t *d; std::vector<uint16_t> h(2 * C);
    if (cudaMalloc(&d, 4 * C) != cudaSuccess) { fprintf(stderr, "cudaMalloc\n"); return 1; }
    unsigned long long bad = 0, shown = 0;
    for (uint64_t b = 0; b < (1ull << 32); b += C) {   // oz_bfb: GPU vs ultra.cu's bf()
        k_bfb<<<2048, 256>>>(b, C, d); cudaMemcpy(h.data(), d, 2 * C, cudaMemcpyDeviceToHost);
        #pragma omp parallel for reduction(+ : bad) schedule(static)
        for (size_t i = 0; i < C; i++) {
            uint64_t hh = fc_mix(b + i); double v = ozx_dbl((b + i) & 1 ? hh : (hh & 0x800fffffffffffffull) | ((uint64_t)(1023 - 140 + (hh >> 52 & 0x7ff) % 160) << 52));
            float f = bf(v); uint32_t u; memcpy(&u, &f, 4); uint16_t r = u >> 16;
            if (!same16(r, h[i]) || !same16(r, oz_bfb(v))) { bad++; if (shown < 6) {
                #pragma omp critical
                { shown++; printf("  bfb v=%a host %04x oz %04x gpu %04x\n", v, r, oz_bfb(v), h[i]); } } }
        }
    }
    printf("oz_bfb: 2^32 doubles, differ: %llu\n", bad); fflush(stdout);
    unsigned long long sbad[2] = {0, 0};
    for (int set = 0; set < 2; set++)
        for (uint64_t b = 0; b < (1ull << 31); b += C) {   // swiglu: GPU vs the host's fill_gu formula
            k_sw<<<2048, 256>>>(set, b, C, d); cudaMemcpy(h.data(), d, 4 * C, cudaMemcpyDeviceToHost);
            unsigned long long sb = 0;
            #pragma omp parallel for reduction(+ : sb) schedule(static)
            for (size_t i = 0; i < C; i++) {
                uint64_t h1 = fc_mix(b + i), h2 = fc_mix(h1), h3 = fc_mix(h2);
                float gf = fc_f(h1, set), uf = fc_f(h2, set), yf = fc_f(h3, set) * (set ? 1.f : 1e-3f);
                double da = 0.0; da += yf;                      // dgrad_f8g's add into the zeroed da
                double g = gf, s = 1.0 / (1.0 + oz_exp(-g));   // jais_lora.cu fill_gu, verbatim
                float f1 = bf(da * uf * s * (1.0 + g * (1.0 - s))), f2 = bf(da * g * s); uint32_t u1, u2; memcpy(&u1, &f1, 4); memcpy(&u2, &f2, 4);
                uint16_t x1, x2; oz_swiglu_bf(gf, uf, yf, x1, x2);
                if (!same16(u1 >> 16, h[2 * i]) || !same16(u2 >> 16, h[2 * i + 1]) || !same16(u1 >> 16, x1) || !same16(u2 >> 16, x2)) { sb++;
                    if (shown < 12) {
                        #pragma omp critical
                        { shown++; printf("  sw set %d g=%a u=%a y=%a host %04x %04x gpu %04x %04x\n", set, gf, uf, yf, u1 >> 16, u2 >> 16, h[2 * i], h[2 * i + 1]); } } }
            }
            sbad[set] += sb;
        }
    printf("oz_swiglu_bf: 2^31 triples realistic, differ: %llu; 2^31 any bits, differ: %llu\n", sbad[0], sbad[1]);
    printf(bad || sbad[0] || sbad[1] ? "FAILED\n" : "all equal\n");
    return bad || sbad[0] || sbad[1];
}
