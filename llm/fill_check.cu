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
static int fill_kernels();
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
    int kf = fill_kernels();
    printf(bad || sbad[0] || sbad[1] || kf ? "FAILED\n" : "all equal\n");
    return bad || sbad[0] || sbad[1] || kf;
}
// (appended 03-11) the two fill kernels of moe_dev.cuh against jais_lora.cu's fill_down and fill_gu loops, written as they
// are there, on Jais-3's shapes (D = 6144, F = 2048 here as a stand-in multiple of 128): every byte of x and every dw.
#include "moe_dev.cuh"
static int fill_kernels() {
    const int N = 2048, n = 3000, D = 6144, F = 2048; std::mt19937_64 R(7); std::normal_distribution<double> nd;
    std::vector<double> g((size_t)N * D), w(n); std::vector<int> rows(n); std::vector<float> y((size_t)n * D), gg((size_t)n * F), uu(gg.size()), yd(gg.size());
    for (auto &v : g) v = nd(R) * 1e-3; for (auto &v : y) v = (float)nd(R); for (int i = 0; i < n; i++) { rows[i] = (int)(R() % N); w[i] = std::abs(nd(R)) * 0.3; }
    for (size_t k = 0; k < gg.size(); k++) { gg[k] = (float)(nd(R) * 3); uu[k] = (float)(nd(R) * 2); yd[k] = (float)(nd(R) * 1e-2); }
    g[5] = 0; g[6] = -0.0; y[7] = -0.0f; gg[3] = 0; yd[4] = -0.0f; gg[9] = 90; gg[10] = -90; g[11] = 1e-320;   // edges
    std::vector<uint16_t> hx((size_t)n * D), hgu((size_t)n * 2 * F); std::vector<double> hdw(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {   // fill_down, verbatim
        size_t r = rows[i]; double wi = w[i], d = 0; const double *gr = g.data() + r * D; const float *yi = y.data() + (size_t)i * D; uint16_t *x = hx.data() + (size_t)i * D;
        for (int c = 0; c < D; c++) { float f = bf(wi * gr[c]); uint32_t u; memcpy(&u, &f, 4); x[c] = u >> 16; d += gr[c] * yi[c]; }
        hdw[i] = d; }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) { uint16_t *x = hgu.data() + (size_t)i * 2 * F;   // fill_gu, verbatim (da from the dgrad add)
        for (int c = 0; c < F; c++) { size_t k = (size_t)i * F + c; double da = 0.0; da += yd[k]; double gv = gg[k], s = 1.0 / (1.0 + oz_exp(-gv));
            float f1 = bf(da * uu[k] * s * (1.0 + gv * (1.0 - s))), f2 = bf(da * gv * s); uint32_t u1, u2; memcpy(&u1, &f1, 4); memcpy(&u2, &f2, 4);
            x[c] = u1 >> 16; x[F + c] = u2 >> 16; } }
    double *dg, *dw_, *dwo; int *dr; float *dy, *dgg, *duu, *dyd; uint16_t *dx, *dgu;
    cudaMalloc(&dg, g.size() * 8); cudaMalloc(&dw_, n * 8); cudaMalloc(&dwo, n * 8); cudaMalloc(&dr, n * 4); cudaMalloc(&dy, y.size() * 4);
    cudaMalloc(&dgg, gg.size() * 4); cudaMalloc(&duu, gg.size() * 4); cudaMalloc(&dyd, gg.size() * 4); cudaMalloc(&dx, hx.size() * 2); cudaMalloc(&dgu, hgu.size() * 2);
    cudaMemcpy(dg, g.data(), g.size() * 8, cudaMemcpyHostToDevice); cudaMemcpy(dw_, w.data(), n * 8, cudaMemcpyHostToDevice); cudaMemcpy(dr, rows.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dgg, gg.data(), gg.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(duu, uu.data(), gg.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dyd, yd.data(), gg.size() * 4, cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1, e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    cudaEventRecord(e0); k_fill_down<<<2048, 256>>>(dg, dr, dw_, dy, n, D, dx, dwo); cudaEventRecord(e1);
    k_fill_gu<<<2048, 256>>>(dgg, duu, dyd, n, F, dgu); cudaEventRecord(e2); cudaEventSynchronize(e2);
    if (cudaGetLastError()) { printf("kernel error\n"); return 1; }
    float t1, t2; cudaEventElapsedTime(&t1, e0, e1); cudaEventElapsedTime(&t2, e1, e2);
    std::vector<uint16_t> gx(hx.size()), ggu(hgu.size()); std::vector<double> gdw(n);
    cudaMemcpy(gx.data(), dx, gx.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(ggu.data(), dgu, ggu.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(gdw.data(), dwo, n * 8, cudaMemcpyDeviceToHost);
    size_t bx = 0, bg = 0, bd = 0;
    for (size_t k = 0; k < hx.size(); k++) bx += hx[k] != gx[k];
    for (size_t k = 0; k < hgu.size(); k++) bg += !same16(hgu[k], ggu[k]);
    for (int i = 0; i < n; i++) bd += memcmp(&hdw[i], &gdw[i], 8) != 0;
    printf("k_fill_down: %zu of %zu x differ, %zu of %d dw differ (%.2f ms); k_fill_gu: %zu of %zu x differ (%.2f ms)\n", bx, hx.size(), bd, n, t1, bg, hgu.size(), t2);
    return (bx || bd || bg) ? 1 : 0;
}
