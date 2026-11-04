// capi.cu -- the C ABI of the Python package ozaki_blackwell (libozaki_blackwell.so): FP64, double-double and triple-double
// matrix products from INT8 tensor cores (src/ozaki2.cuh, ozaki_dd.cuh, ozaki_td.cuh) behind plain C functions, so that
// ctypes can call them with NumPy (host) or CUDA (device) pointers and nobody has to compile anything. cudart is linked
// statically: the library needs only the NVIDIA driver. All matrices are row-major float64: A is M x K, B is K x N.
// Every function returns 0 or a cudaError_t / OZB_E* code; ozb_error() turns it into text. Nothing here aborts the process.
#include <algorithm>
#include <cstring>
#include <mutex>
#include <vector>
#include "../src/ozaki2.cuh"
#include "../src/ozaki_dd.cuh"
#include "../src/ozaki_td.cuh"

#define OZB_API extern "C" __attribute__((visibility("default")))
#ifndef OZB_VERSION
#define OZB_VERSION "0.0.0"
#endif
enum { OZB_EARG = 10001, OZB_EK = 10002, OZB_EARCH = 10003 };

namespace {
std::mutex mu;   // one plan cache per precision; ctypes releases the GIL, so two Python threads may call at once
struct Cache { oz::Plan p{}; int nm = 0, maxbits = 0; bool live = false; };
Cache cache[3];  // 0 FP64, 1 dd, 2 td

int check_args(int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return OZB_EARG;
    if (K >= 131072) return OZB_EK;   // int32 accumulators: 127^2 K < 2^31 (oz::bits_for would abort)
    int dev, major;
    if (cudaGetDevice(&dev) || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev)) return (int)cudaGetLastError();
    return major < 8 ? OZB_EARCH : 0;  // mma.sync.m16n8k32 s8 and cp.async need sm_80
}
oz::Plan &plan_for(int which, int M, int N, int K, int nm, int maxbits) {
    Cache &c = cache[which];
    if (c.live && (c.p.M != M || c.p.N != N || c.p.K != K || c.nm != nm || c.maxbits != maxbits)) { oz::release(c.p); c.live = false; }
    if (!c.live) { c.p = oz::plan(M, N, K, nm, maxbits); c.nm = nm; c.maxbits = maxbits; c.live = true; }
    return c.p;
}
// host pointers: the n arrays go to the device (in), the m results come back (out)
struct Staged {
    double *d[9] = {}; int err = 0;
    Staged(const double *const *in, int nin, const size_t *len_in, int nout, size_t len_out) {
        for (int i = 0; i < nin + nout && !err; i++) err = cudaMalloc(&d[i], (i < nin ? len_in[i] : len_out) * sizeof(double));
        for (int i = 0; i < nin && !err; i++) err = cudaMemcpy(d[i], in[i], len_in[i] * sizeof(double), cudaMemcpyHostToDevice);
    }
    int back(double *const *out, int nin, int nout, size_t len_out) {
        for (int i = 0; i < nout && !err; i++) err = cudaMemcpy(out[i], d[nin + i], len_out * sizeof(double), cudaMemcpyDeviceToHost);
        return err;
    }
    ~Staged() { for (double *p : d) if (p) cudaFree(p); }
};
int finish() { const int e = (int)cudaDeviceSynchronize(); return e ? e : (int)cudaGetLastError(); }
}  // namespace

OZB_API const char *ozb_version(void) { return OZB_VERSION; }

OZB_API const char *ozb_error(int e) {
    switch (e) {
    case 0: return "ok";
    case OZB_EARG: return "M, N and K must be positive";
    case OZB_EK: return "K must be below 131072 (the int32 accumulators of the INT8 GEMMs)";
    case OZB_EARCH: return "this GPU is older than sm_80 (Ampere); the INT8 mma.sync.m16n8k32 path needs sm_80 or newer";
    default: return cudaGetErrorString((cudaError_t)e);
    }
}

// name (NUL-terminated, truncated to len), compute capability as 10*major+minor, L2 size in KiB, SM count
OZB_API int ozb_device(char *name, int len, int *sm, int *l2_kib, int *sms) {
    int dev; cudaDeviceProp p;
    if (int e = cudaGetDevice(&dev)) return e;
    if (int e = cudaGetDeviceProperties(&p, dev)) return e;
    if (name && len > 0) { strncpy(name, p.name, len - 1); name[len - 1] = 0; }
    if (sm) *sm = 10 * p.major + p.minor;
    if (l2_kib) *l2_kib = p.l2CacheSize >> 10;
    if (sms) *sms = p.multiProcessorCount;
    return 0;
}

// C = A B in FP64 with `moduli` moduli (8..16; 16 = FP64 accuracy up to exponent spreads of 2^+-16, see the paper)
OZB_API int ozb_dgemm(int M, int N, int K, const double *A, const double *B, double *C, int moduli, int on_host) {
    if (int e = check_args(M, N, K)) return e;
    if (moduli < 8 || moduli > 16) return OZB_EARG;
    std::lock_guard<std::mutex> g(mu);
    if (!on_host) { oz::dgemm(plan_for(0, M, N, K, moduli, 62), A, B, C); return finish(); }
    const double *in[2] = {A, B}; const size_t len[2] = {(size_t)M * K, (size_t)K * N};
    Staged s(in, 2, len, 1, (size_t)M * N);
    if (s.err) return s.err;
    oz::dgemm(plan_for(0, M, N, K, moduli, 62), s.d[0], s.d[1], s.d[2]);
    if (int e = finish()) return e;
    double *out[1] = {C};
    return s.back(out, 2, 1, (size_t)M * N);
}

// (Ch + Cl) = (Ah + Al)(Bh + Bl) in double-double (~106 bits), 32 moduli
OZB_API int ozb_ddgemm(int M, int N, int K, const double *Ah, const double *Al, const double *Bh, const double *Bl,
                       double *Ch, double *Cl, int on_host) {
    if (int e = check_args(M, N, K)) return e;
    std::lock_guard<std::mutex> g(mu);
    if (!on_host) { oz::ddgemm_t<32>(plan_for(1, M, N, K, 32, 106), Ah, Al, Bh, Bl, Ch, Cl); return finish(); }
    const double *in[4] = {Ah, Al, Bh, Bl};
    const size_t len[4] = {(size_t)M * K, (size_t)M * K, (size_t)K * N, (size_t)K * N};
    Staged s(in, 4, len, 2, (size_t)M * N);
    if (s.err) return s.err;
    oz::ddgemm_t<32>(plan_for(1, M, N, K, 32, 106), s.d[0], s.d[1], s.d[2], s.d[3], s.d[4], s.d[5]);
    if (int e = finish()) return e;
    double *out[2] = {Ch, Cl};
    return s.back(out, 4, 2, (size_t)M * N);
}

// (C0 + C1 + C2) = (A0 + A1 + A2)(B0 + B1 + B2) in triple-double (~159 bits), 49 moduli
OZB_API int ozb_tdgemm(int M, int N, int K, const double *A0, const double *A1, const double *A2, const double *B0,
                       const double *B1, const double *B2, double *C0, double *C1, double *C2, int on_host) {
    if (int e = check_args(M, N, K)) return e;
    std::lock_guard<std::mutex> g(mu);
    if (!on_host) { oz::tdgemm_t<49>(plan_for(2, M, N, K, 49, 160), A0, A1, A2, B0, B1, B2, C0, C1, C2); return finish(); }
    const double *in[6] = {A0, A1, A2, B0, B1, B2};
    const size_t a = (size_t)M * K, b = (size_t)K * N, len[6] = {a, a, a, b, b, b};
    Staged s(in, 6, len, 3, (size_t)M * N);
    if (s.err) return s.err;
    oz::tdgemm_t<49>(plan_for(2, M, N, K, 49, 160), s.d[0], s.d[1], s.d[2], s.d[3], s.d[4], s.d[5], s.d[6], s.d[7], s.d[8]);
    if (int e = finish()) return e;
    double *out[3] = {C0, C1, C2};
    return s.back(out, 6, 3, (size_t)M * N);
}

// frees the cached plans (device memory); the next call plans again
OZB_API void ozb_release(void) {
    std::lock_guard<std::mutex> g(mu);
    for (Cache &c : cache) if (c.live) { oz::release(c.p); c.live = false; }
}

// GPU time of `reps` products of random n x n matrices on the device (no transfers), the median in ms; kind 0 FP64, 1 dd, 2 td
OZB_API int ozb_bench(int n, int kind, int reps, double *median_ms) {
    if (int e = check_args(n, n, n)) return e;
    if (kind < 0 || kind > 2 || reps < 1 || reps > 101) return OZB_EARG;
    const int parts = kind + 1; const size_t len = (size_t)n * n;
    double *d[9] = {};
    int err = 0;
    for (int i = 0; i < 3 * parts && !err; i++) err = cudaMalloc(&d[i], len * sizeof(double));
    if (!err) {   // values in [-1, 1) with lower parts 2^-53 and 2^-106 below, from a fixed integer hash
        double *h = (double *)malloc(len * sizeof(double));
        for (int i = 0; i < 2 * parts && !err; i++) {
            const double sc = i % parts == 0 ? 1.0 : (i % parts == 1 ? 0x1p-53 : 0x1p-106);
            uint64_t x = 0x9E3779B97F4A7C15ull * (i + 1);
            for (size_t j = 0; j < len; j++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; h[j] = sc * ((double)(x >> 11) * 0x1p-52 - 1.0); }
            err = cudaMemcpy(d[i < parts ? i : parts + i], h, len * sizeof(double), cudaMemcpyHostToDevice);
        }
        free(h);
    }
    // layout: A parts at d[0..parts), B parts at d[2parts..3parts), C parts at d[parts..2parts)
    std::vector<float> t;
    cudaEvent_t e0, e1;
    if (!err) { cudaEventCreate(&e0); cudaEventCreate(&e1); }
    for (int r = -1; r < reps && !err; r++) {   // r = -1: warm-up (plans, first launches)
        cudaEventRecord(e0);
        if (kind == 0) err = ozb_dgemm(n, n, n, d[0], d[2], d[1], 16, 0);
        else if (kind == 1) err = ozb_ddgemm(n, n, n, d[0], d[1], d[4], d[5], d[2], d[3], 0);
        else err = ozb_tdgemm(n, n, n, d[0], d[1], d[2], d[6], d[7], d[8], d[3], d[4], d[5], 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (r >= 0) t.push_back(ms);
    }
    if (!t.empty()) { std::sort(t.begin(), t.end()); *median_ms = t[t.size() / 2]; cudaEventDestroy(e0); cudaEventDestroy(e1); }
    for (double *p : d) if (p) cudaFree(p);
    return err;
}
