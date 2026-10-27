// ultra.cu -- NVIDIA Nemotron 3 Ultra (560 524 578 816 parameters, NVFP4 checkpoint, 352 GB) generating text token by
// token on this laptop, streamed from disk, with every linear map of the network correctly rounded.
//
// One generic GEMV covers the four weight formats of the checkpoint: BF16 (attention, latent projections, lm_head), FP32
// (router), FP8 E4M3 W8A8 (Mamba in/out, shared experts) and NVFP4 W4A4 (routed experts). Weight and activation elements
// are integers times powers of two (activations quantized first exactly as the format prescribes: FP8 with the static
// input scale, NVFP4 with a dynamic E4M3 scale per 16 and the static global scale), so a dot product is an integer sum:
// each warp accumulates it in a 128-bit fixed point anchored at the row's largest possible term, multiplies by the
// exact product of the FP32 global scales, and rounds ONCE to the output precision (BF16, FP32 for router and logits).
// Terms below the 125-bit window are counted; the rounding is certified when the interval they span rounds to one value
// (uncertified outputs are counted and reported -- none expected).
// The engine path is the same model with FP32 accumulation (FMA in k order, warp tree, split-K s slices in order, times
// the FP32-rounded global scale), as FP32-accumulating kernels compute it; s = 1 and s = 4 stand for two kernel/batch
// configurations of one serving engine.
// Everything outside the linear maps (RMSNorm, conv1d, SiLU, softplus, the SSM recurrence with its FP32 state, attention
// softmax, routing, the weighted expert sum, residual adds) is evaluated on the host in FP64 and rounded once to the
// precision the model keeps there (BF16 activations, FP32 SSM state) -- identical for both paths, so the paths differ
// only in how the linear maps are accumulated. KV cache in BF16 (the checkpoint's FP8 KV scales are not used).
//
// nvcc -O3 -arch=sm_120a -std=c++17 -o out_ultra llm/ultra.cu
// out_ultra <model dir with index.txt> <prompts file: one line of token ids per prompt> <new tokens> [modes, e.g. 0,1,4]
// prints one line per step and sequence: step prompt mode token top1 top2; stats of the shadow comparison at the end.
#define NV_NO_MAIN
#include "nvexact.cu"
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

constexpr int H = 8192, VOC = 131072, MH = 256, MD = 64, MN = 128, NG = 8, DIN = 16384, CONV = 18432, LAT = 2048,
              E = 512, TOPK = 22, NQ = 64, NKV = 2, HD = 128;
enum { F_BF16, F_F32, F_E4M3, F_NVFP4 };

// ---- the GEMV --------------------------------------------------------------------------------------------------------
struct DevW { const void *w; const uint8_t *sw; int fmt, M, K; unsigned long long Gm; int Ge; float Gf; };
__host__ __device__ inline void wdec(const DevW &W, int row, int k, int &m, int &e) {
    size_t i = (size_t)row * W.K + k;
    if (W.fmt == F_BF16) {
        uint16_t u = ((const uint16_t *)W.w)[i]; int ex = (u >> 7) & 255, f = u & 127;
        m = ex ? 128 + f : f; e = ex ? ex - 134 : -133; if (u >> 15) m = -m;
    } else if (W.fmt == F_F32) {
        uint32_t u = ((const uint32_t *)W.w)[i]; int ex = (u >> 23) & 255, f = u & 0x7FFFFF;
        m = ex ? (1 << 23) + f : f; e = ex ? ex - 150 : -149; if (u >> 31) m = -m;
    } else if (W.fmt == F_E4M3) {
        uint8_t b = ((const uint8_t *)W.w)[i]; int ex = (b >> 3) & 15, f = b & 7;
        m = ex ? 8 + f : f; e = ex ? ex - 10 : -9; if (b >> 7) m = -m;
    } else {   // NVFP4: low nibble = even k, 2*E2M1 times the E4M3 scale of its block of 16
        uint8_t b = ((const uint8_t *)W.w)[i / 2]; int nib = (k & 1) ? b >> 4 : b & 15, c = nib & 7;
        int q = c < 4 ? c : (c == 7 ? 12 : 2 * (c - 2));
        int se = e4m3_sig_eps(W.sw[(size_t)row * (W.K / 16) + k / 16]);
        m = q * (se & 255); e = (se >> 8) - 33; if (nib & 8) m = -m;
    }
}
__host__ __device__ inline int bitlen128(unsigned __int128 u) {
    uint64_t hi = (uint64_t)(u >> 64), lo = (uint64_t)u;
#ifdef __CUDA_ARCH__
    return hi ? 128 - __clzll(hi) : (lo ? 64 - __clzll(lo) : 0);
#else
    return hi ? 128 - __builtin_clzll(hi) : (lo ? 64 - __builtin_clzll(lo) : 0);
#endif
}
// u * 2^X rounded to p significant bits, nearest even, FP32 exponent range (subnormal LSB -126-(p-1))
__host__ __device__ inline double round_p(unsigned __int128 u, int X, int p) {
    if (!u) return 0.0;
    int r = bitlen128(u) - p, minlsb = -126 - (p - 1);
    if (X + r < minlsb) r = minlsb - X;
    if (r <= 0) return ldexp((double)(uint64_t)u, X);
    if (r >= 127) return 0.0;
    unsigned __int128 q = u >> r, rem = u - (q << r), half = (unsigned __int128)1 << (r - 1);
    if (rem > half || (rem == half && (q & 1))) q++;
    return ldexp((double)(uint64_t)q, X + r);
}
// (acc +- tail) * 2^lsb * Gm 2^Ge rounded to p bits; true when both ends of the interval round to the same value
__device__ inline bool cert_round(__int128 acc, int tail, int lsb, unsigned long long Gm, int Ge, int p, float &out) {
    bool neg = acc < 0;
    unsigned __int128 a = neg ? (unsigned __int128)(-acc) : (unsigned __int128)acc, t = (unsigned)tail;
    int nb = bitlen128(a);
    if (nb > 78) {   // keep 78 bits; the interval widens only if a bit is lost (an exact tie must stay certified)
        int r = nb - 78; unsigned __int128 lost = a & (((unsigned __int128)1 << r) - 1);
        a >>= r; t = (t == 0 && lost == 0) ? 0 : ((t + ((unsigned __int128)1 << r) - 1) >> r) + 1; lsb += r;
    }
    if (a <= t) { out = 0.f; return a == 0 && t == 0; }
    double lo = round_p((a - t) * Gm, lsb + Ge, p), hi = round_p((a + t) * Gm, lsb + Ge, p);
    out = (float)(neg ? -lo : lo);
    return lo == hi;
}
// one warp per row; mode 0 exact, mode s > 0 FP32 engine with split-K s
__global__ void gemv_k(DevW W, const int32_t *xm, const int16_t *xe, int hix, int p, int mode, float *y, int *nbad) {
    int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31, K = W.K;
    if (row >= W.M) return;
    if (mode == 0) {
        int hiw = -100000;
        for (int k = lane; k < K; k += 32) { int m, e; wdec(W, row, k, m, e); if (m) hiw = max(hiw, e + 32 - __clz(abs(m))); }
        for (int o = 16; o; o >>= 1) hiw = max(hiw, __shfl_xor_sync(~0u, hiw, o));
        if (hiw == -100000 || hix == -100000) { if (!lane) y[row] = 0.f; return; }
        int lsb = hiw + hix + 16 - 125;   // K <= 2^15: |sum| < 2^(hiw+hix+15) = 2^124 units
        __int128 acc = 0; int tail = 0;
        for (int k = lane; k < K; k += 32) {
            int m, e; wdec(W, row, k, m, e); int32_t a = xm[k];
            if (!m || !a) continue;
            long long pr = (long long)m * a; int sh = e + xe[k] - lsb;
            if (sh >= 0) acc += (__int128)pr << sh;
            else { tail++; acc += sh > -63 ? (__int128)(pr >> -sh) : (__int128)(pr < 0 ? -1 : 0); }
        }
        for (int o = 16; o; o >>= 1) {
            unsigned long long lo = (unsigned long long)acc, hi = (unsigned long long)((unsigned __int128)acc >> 64);
            lo = __shfl_xor_sync(~0u, lo, o); hi = __shfl_xor_sync(~0u, hi, o);
            acc += (__int128)(((unsigned __int128)hi << 64) | lo);
            tail += __shfl_xor_sync(~0u, tail, o);
        }
        if (!lane) { float r; if (!cert_round(acc, tail, lsb, W.Gm, W.Ge, p, r)) atomicAdd(nbad, 1); y[row] = r; }
    } else {
        int per = K / mode; float tot = 0.f;
        for (int c = 0; c < mode; c++) {
            float acc = 0.f;
            for (int k = c * per + lane; k < (c + 1) * per; k += 32) {
                int m, e; wdec(W, row, k, m, e);
                acc = fmaf(ldexpf((float)m, e), ldexpf((float)xm[k], xe[k]), acc);
            }
            for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(~0u, acc, o);
            tot += acc;
        }
        float r = tot * W.Gf;
        if (!lane) y[row] = p == 8 ? __bfloat162float(__float2bfloat16_rn(r)) : r;
    }
}

// ---- host numerics ---------------------------------------------------------------------------------------------------
static float bf(double v) {   // once to BF16, nearest even
    if (v == 0 || !std::isfinite(v)) return (float)v;
    int ex; frexp(v, &ex);
    int q = std::max(ex - 8, -133);
    return (float)ldexp(nearbyint(ldexp(v, -q)), q);
}
static float bf16f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }
static uint64_t fnv(uint64_t h, const std::vector<float> &v) {   // FNV-1a 64 over the bits: a transcript digest of the exact path
    for (float f : v) { uint32_t b; memcpy(&b, &f, 4); for (int i = 0; i < 4; i++) { h ^= (b >> (8 * i)) & 255; h *= 1099511628211ULL; } }
    return h;
}
static int bl32(uint32_t v) { return v ? 32 - __builtin_clz(v) : 0; }
struct Act { std::vector<int32_t> m; std::vector<int16_t> e; int hix = -100000; };
static void put(Act &a, int k, int m, int e) { a.m[k] = m; a.e[k] = (int16_t)e; if (m) a.hix = std::max(a.hix, e + bl32(std::abs(m))); }
static Act act_plain(const std::vector<float> &x) {
    Act a; a.m.resize(x.size()); a.e.resize(x.size());
    for (size_t k = 0; k < x.size(); k++) {
        if (x[k] == 0.f) { put(a, k, 0, 0); continue; }
        int ex; float f = frexpf(x[k], &ex); put(a, k, (int)ldexpf(f, 24), ex - 24);
    }
    return a;
}
static Act act_fp8(const std::vector<float> &x, float is) {
    Act a; a.m.resize(x.size()); a.e.resize(x.size());
    for (size_t k = 0; k < x.size(); k++) {
        float t = x[k] / is; uint8_t b = to_e4m3(fabsf(t));
        int se = e4m3_sig_eps(b), m = b ? (se & 255) : 0;
        put(a, k, t < 0 ? -m : m, (se >> 8) - 32);
    }
    return a;
}
static Act act_nvfp4(const std::vector<float> &x, float gA) {   // ModelOpt: sA = E4M3(amax/(6 gA)), q = 2 E2M1(x/(sA gA))
    Act a; a.m.resize(x.size()); a.e.resize(x.size());
    for (size_t b = 0; b < x.size() / 16; b++) {
        float amax = 0.f; for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(x[b * 16 + i]));
        uint8_t s = to_e4m3(amax / (6.f * gA)); float d = e4m3_val(s) * gA; int se = e4m3_sig_eps(s);
        for (int i = 0; i < 16; i++) put(a, b * 16 + i, s ? to_2e2m1(x[b * 16 + i] / d) * (se & 255) : 0, (se >> 8) - 33);
    }
    return a;
}
static void rmsnorm(const std::vector<float> &x, const std::vector<float> &w, std::vector<float> &y, size_t off, size_t n) {
    double ss = 0; for (size_t i = 0; i < n; i++) ss += (double)x[off + i] * x[off + i];
    float var = (float)(ss / n); float r = (float)(1.0 / sqrt((double)var + 1e-5));
    for (size_t i = 0; i < n; i++) y[off + i] = bf((double)w[off + i] * bf((double)x[off + i] * r));
}

// ---- weights from disk -----------------------------------------------------------------------------------------------
struct TI { std::string dt, file; size_t off, n; };
static std::unordered_map<std::string, TI> idx;
static std::map<std::string, int> fds;
static std::string DIR;
static uint8_t *pin; static size_t PIN = (size_t)300 << 20;
static double t_disk = 0, bytes_disk = 0;
static void readt(const std::string &name, void *dst, size_t want = 0) {
    auto it = idx.find(name);
    if (it == idx.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(1); }
    const TI &t = it->second;
    if (want && want != t.n) { fprintf(stderr, "%s: %zu bytes, expected %zu\n", name.c_str(), t.n, want); exit(1); }
    if (!fds.count(t.file)) fds[t.file] = open((DIR + "/" + t.file).c_str(), O_RDONLY);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t done = 0; done < t.n;) {
        ssize_t r = pread(fds[t.file], (char *)dst + done, t.n - done, t.off + done);
        if (r <= 0) { perror("pread"); exit(1); }
        done += r;
    }
    t_disk += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); bytes_disk += t.n;
}
static size_t tsize(const std::string &name) { return idx.at(name).n; }
static float scalar(const std::string &name) { float f; readt(name, &f, 4); return f; }
static std::vector<float> vec_bf16(const std::string &name) {
    std::vector<uint16_t> u(tsize(name) / 2); readt(name, u.data());
    std::vector<float> v(u.size()); for (size_t i = 0; i < u.size(); i++) v[i] = bf16f(u[i]); return v;
}
static std::vector<float> vec_f32(const std::string &name) { std::vector<float> v(tsize(name) / 4); readt(name, v.data()); return v; }

static void *dW, *dS; static int32_t *dxm; static int16_t *dxe; static float *dy; static int *dbad;
static void gscale(DevW &W, float a, float b) {   // exact a*b as a 48-bit integer times 2^Ge; engine: a*b in FP32
    int xa, xb; float fa = frexpf(a, &xa), fb = frexpf(b, &xb);
    W.Gm = (unsigned long long)ldexpf(fa, 24) * (unsigned long long)ldexpf(fb, 24); W.Ge = xa + xb - 48; W.Gf = a * b;
}
static DevW load_w(const std::string &name, int fmt, int M, int K, void *dst = nullptr) {
    DevW W{}; W.fmt = fmt; W.M = M; W.K = K; W.Gm = 1; W.Ge = 0; W.Gf = 1.f;
    size_t n = tsize(name); void *d = dst ? dst : dW;
    for (size_t o = 0; o < n; o += PIN) {   // stream through the pinned buffer
        size_t c = std::min(PIN, n - o);
        const TI t = idx.at(name); TI part = t; part.off += o; part.n = c;
        idx["__part"] = part; readt("__part", pin);
        CK(cudaMemcpy((char *)d + o, pin, c, cudaMemcpyHostToDevice));
    }
    W.w = d;
    return W;
}
// stats: exact vs engine on the same input (shadow), per format
static long long st_out[4], st_d1[4], st_d4[4], st_bad;
static std::vector<float> gemv(const DevW &W, const Act &a, int p, int mode, bool shadow = false) {
    CK(cudaMemcpy(dxm, a.m.data(), a.m.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dxe, a.e.data(), a.e.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(dbad, 0, 4));
    gemv_k<<<(W.M + 7) / 8, 256>>>(W, dxm, dxe, a.hix, p, mode, dy, dbad);
    std::vector<float> y(W.M); CK(cudaMemcpy(y.data(), dy, W.M * 4, cudaMemcpyDeviceToHost));
    int nb; CK(cudaMemcpy(&nb, dbad, 4, cudaMemcpyDeviceToHost)); st_bad += nb;
    if (shadow && mode == 0) {
        std::vector<float> z(W.M);
        for (int s : {1, 4}) {
            gemv_k<<<(W.M + 7) / 8, 256>>>(W, dxm, dxe, a.hix, p, s, dy, dbad);
            CK(cudaMemcpy(z.data(), dy, W.M * 4, cudaMemcpyDeviceToHost));
            long long d = 0; for (int i = 0; i < W.M; i++) d += z[i] != y[i];
            (s == 1 ? st_d1 : st_d4)[W.fmt] += d;
        }
        st_out[W.fmt] += W.M;
    }
    return y;
}

// ---- self-test: the kernel against an independent host superaccumulator (2560 bits, no window, no tail) ----------------
struct Big {   // nonnegative, bit i of limb j = 2^(64 j + i - OFF)
    static constexpr int NL = 40, OFF = 1280; uint64_t l[NL] = {};
    void add(unsigned __int128 v, int e) {   // v * 2^e
        int pos = e + OFF, j = pos / 64, s = pos % 64;
        unsigned __int128 lo = v << s, carry = s ? v >> (128 - s) : 0;
        uint64_t parts[3] = {(uint64_t)lo, (uint64_t)(lo >> 64), (uint64_t)carry};
        for (int q = 0; q < 3 || false; q++) {
            unsigned __int128 t = (unsigned __int128)l[j + q] + parts[q]; l[j + q] = (uint64_t)t;
            for (int c = j + q + 1; (t >> 64) && c < NL; c++) { t = (unsigned __int128)l[c] + 1; l[c] = (uint64_t)t; }
        }
    }
    bool ge(const Big &b) const { for (int j = NL - 1; j >= 0; j--) if (l[j] != b.l[j]) return l[j] > b.l[j]; return true; }
    void sub(const Big &b) { unsigned __int128 br = 0; for (int j = 0; j < NL; j++) { unsigned __int128 t = (unsigned __int128)l[j] - b.l[j] - br; l[j] = (uint64_t)t; br = (t >> 64) ? 1 : 0; } }
};
static double big_round(Big a, unsigned long long Gm, int Ge, int p) {   // a * Gm * 2^Ge to p bits, nearest even
    Big r; for (int j = 0; j < Big::NL; j++) if (a.l[j]) r.add((unsigned __int128)a.l[j] * Gm, 64 * j - Big::OFF);
    int top = -1; for (int j = Big::NL - 1; j >= 0 && top < 0; j--) if (r.l[j]) top = 64 * j + 63 - __builtin_clzll(r.l[j]);
    if (top < 0) return 0.0;
    int lsb = top - (p - 1), minlsb = -126 - (p - 1) - Ge + Big::OFF;
    if (lsb < minlsb) lsb = minlsb;
    auto bit = [&](int i) -> int { return i < 0 ? 0 : (r.l[i / 64] >> (i % 64)) & 1; };
    uint64_t q = 0; for (int i = top; i >= lsb; i--) q = q * 2 + bit(i);
    int hb = bit(lsb - 1); bool rest = false; for (int i = lsb - 2; i >= 0 && !rest; i--) rest = bit(i);
    if (hb && (rest || (q & 1))) q++;
    return ldexp((double)q, lsb - Big::OFF + Ge);
}
static int selftest() {
    std::mt19937_64 rng(2027); int M = 256, K = 8192, fails = 0; long long checked = 0;
    CK(cudaMalloc(&dW, (size_t)M * K * 4)); CK(cudaMalloc(&dS, (size_t)M * K / 16)); CK(cudaMalloc(&dxm, K * 4)); CK(cudaMalloc(&dxe, K * 2));
    CK(cudaMalloc(&dy, M * 4)); CK(cudaMalloc(&dbad, 4));
    for (int fmt = 0; fmt < 4; fmt++) for (int trial = 0; trial < (fmt == F_BF16 ? 4 : 3); trial++) {
        size_t wb = fmt == F_BF16 ? 2 : fmt == F_F32 ? 4 : 1; std::vector<uint8_t> w((size_t)M * K * wb), sw((size_t)M * K / 16);
        for (auto &b : w) b = rng();
        if (fmt == F_BF16) for (size_t i = 0; i < w.size(); i += 2) w[i + 1] = (w[i + 1] & 0x80) | (0x30 + rng() % (trial == 2 ? 0x40 : 0x10));   // exponents: narrow, or 2^-60 wide
        if (fmt == F_F32) for (size_t i = 0; i < w.size(); i += 4) w[i + 3] = (w[i + 3] & 0x80) | (0x30 + rng() % (trial == 2 ? 0x40 : 0x10));
        if (fmt == F_E4M3) for (auto &b : w) if ((b & 0x7F) == 0x7F) b ^= 1;
        for (auto &b : sw) b = 0x20 + rng() % 0x50;
        std::vector<float> x(K); std::normal_distribution<double> nd;
        for (int k = 0; k < K; k++) x[k] = bf(nd(rng) * (trial == 1 ? exp2(rng() % 40 - 20.0) : 1.0));
        if (trial == 2) for (int k = 0; k < K / 2; k++) x[K / 2 + k] = -x[k];   // heavy cancellation against w
        if (trial == 3) {   // exact ties: w0 + 2^(e0-8) (rows r odd) or w0 + 2^(e0-24), x = (1, 1, 0, ...): nearest-even decides
            std::fill(x.begin(), x.end(), 0.f); x[0] = x[1] = 1.f; std::fill(w.begin(), w.end(), 0);
            for (int r = 0; r < M; r++) { uint16_t w0 = (uint16_t)(0x3F80 | (rng() & 0x7F) | (r & 2 ? 0x8000 : 0)), w1 = (uint16_t)((w0 & 0x8000) | ((0x7F - (r & 1 ? 8 : 24)) << 7));
                memcpy(&w[(size_t)r * K * 2], &w0, 2); memcpy(&w[((size_t)r * K + 1) * 2], &w1, 2); }
        }
        Act a = fmt == F_E4M3 ? act_fp8(x, 0.0123f) : fmt == F_NVFP4 ? act_nvfp4(x, 0.00731f) : act_plain(x);
        CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dS, sw.data(), sw.size(), cudaMemcpyHostToDevice));
        DevW W{}; W.w = dW; W.sw = (uint8_t *)dS; W.fmt = fmt; W.M = M; W.K = K; W.Gm = 1; W.Ge = 0; W.Gf = 1.f;
        if (fmt >= F_E4M3) gscale(W, 0.0371f, fmt == F_E4M3 ? 0.0123f : 0.00731f);
        DevW Wh = W; Wh.w = w.data(); Wh.sw = sw.data();
        for (int p : {8, 24}) {
            st_bad = 0; auto y = gemv(W, a, p, 0);
            for (int r = 0; r < M; r++) {
                Big pos, neg;
                for (int k = 0; k < K; k++) { int m, e; wdec(Wh, r, k, m, e); long long pr = (long long)m * a.m[k]; if (!pr) continue;
                    (pr > 0 ? pos : neg).add((unsigned __int128)(pr > 0 ? pr : -pr), e + a.e[k]); }
                bool ng = !pos.ge(neg); if (ng) { neg.sub(pos); pos = neg; } else pos.sub(neg);
                double ref = big_round(pos, W.Gm, W.Ge, p); if (ng) ref = -ref;
                checked++; if ((float)ref != y[r]) { if (fails++ < 5) fprintf(stderr, "fmt %d trial %d p %d row %d: %.9g vs %.9g\n", fmt, trial, p, r, ref, y[r]); }
            }
            printf("selftest fmt %d trial %d p %2d: %d rows, uncertified %lld\n", fmt, trial, p, M, st_bad);
        }
    }
    printf("selftest: %lld outputs against the superaccumulator, %d differ\n", checked, fails);
    return fails != 0;
}

// ---- shared by the model drivers (ultra.cu, dense.cu) ----------------------------------------------------------------------
// one token's causal attention over its cache in FP64, rounded once to BF16; query head h reads KV head h / (nq / nkv)
static void attend(const std::vector<float> &q, const std::vector<float> &K, const std::vector<float> &V, int nq, int nkv, int hd,
                   std::vector<float> &o) {
    size_t T = K.size() / ((size_t)nkv * hd);
    #pragma omp parallel for schedule(static)
    for (int hq = 0; hq < nq; hq++) {
        int g = hq / (nq / nkv); std::vector<double> sc(T); double mx = -1e300;
        for (size_t u = 0; u < T; u++) { double a = 0; for (int d = 0; d < hd; d++) a += (double)q[hq * hd + d] * K[(u * nkv + g) * hd + d];
            sc[u] = a / sqrt((double)hd); mx = std::max(mx, sc[u]); }
        double den = 0; for (auto &v : sc) { v = exp(v - mx); den += v; }
        for (int d = 0; d < hd; d++) { double a = 0; for (size_t u = 0; u < T; u++) a += sc[u] * V[(u * nkv + g) * hd + d]; o[hq * hd + d] = bf(a / den); }
    }
}
// greedy: the largest logit (lowest id on a tie), printed with the runner-up
static int pick(const std::vector<float> &lg, size_t t, int prompt, int mode, int token) {
    int V = lg.size(), b1 = 0; for (int v = 1; v < V; v++) if (lg[v] > lg[b1]) b1 = v;
    int b2 = b1 ? 0 : 1; for (int v = 0; v < V; v++) if (v != b1 && lg[v] > lg[b2]) b2 = v;
    printf("step %zu prompt %d mode %d token %d next %d top1 %.6f top2 %d %.6f\n", t, prompt, mode, token, b1, lg[b1], b2, lg[b2]);
    return b1;
}
template <class S> static void report(const std::vector<S> &seqs) {   // shadow statistics, digests, token ids
    const char *fn[4] = {"BF16", "FP32", "FP8", "NVFP4"};
    for (int f = 0; f < 4; f++) if (st_out[f])
        printf("shadow %s outputs %lld engine_s1_differs %lld (%.2f%%) engine_s4_differs %lld (%.2f%%)\n", fn[f], st_out[f],
               st_d1[f], 100.0 * st_d1[f] / st_out[f], st_d4[f], 100.0 * st_d4[f] / st_out[f]);
    printf("uncertified %lld\n", st_bad);
    for (auto &s : seqs) printf("digest prompt %d mode %d: %016llx\n", s.prompt, s.mode, (unsigned long long)s.dig);
    for (auto &s : seqs) { printf("seq prompt %d mode %d:", s.prompt, s.mode); for (int v : s.toks) printf(" %d", v); printf("\n"); }
}

// ---- sequences ---------------------------------------------------------------------------------------------------------
struct Seq {
    int prompt, mode; std::vector<int> toks; size_t plen;
    std::vector<float> h;
    std::map<int, std::vector<float>> conv, ssm;            // per Mamba layer: last 3 conv inputs [CONV][3], state [MH][MD][MN]
    std::map<int, std::vector<float>> kc, vc;               // per attention layer: [t][NKV*HD]
    uint64_t dig = 1469598103934665603ULL;   // FNV-1a of every residual stream after every layer and every logit vector
};

#ifndef ULTRA_LIB
int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--selftest")) return selftest();
    if (argc < 4) { fprintf(stderr, "usage: out_ultra <model dir> <prompts> <new tokens> [modes]\n"); return 1; }
    DIR = argv[1]; int NEW = atoi(argv[3]);
    std::vector<int> modes{0, 1, 4};
    if (argc > 4) { modes.clear(); std::stringstream ss(argv[4]); std::string t; while (std::getline(ss, t, ',')) modes.push_back(atoi(t.c_str())); }
    { std::ifstream f(DIR + "/index.txt"); std::string nm, dt, sh, fl; size_t off, n;
      while (f >> nm >> dt >> sh >> fl >> off >> n) idx[nm] = {dt, fl, off, n}; }
    std::vector<std::string> types;
    { std::ifstream f(DIR + "/config.json"); std::string s((std::istreambuf_iterator<char>(f)), {});
      size_t a = s.find("\"layers_block_type\""), b = s.find(']', a); std::string l = s.substr(a, b - a);
      for (size_t i = l.find('[') + 1; (i = l.find('"', i)) != std::string::npos;) { size_t j = l.find('"', i + 1); types.push_back(l.substr(i + 1, j - i - 1)); i = j + 1; } }
    int NL = types.size();
    std::vector<Seq> seqs;
    { std::ifstream f(argv[2]); std::string line; int pi = 0;
      while (std::getline(f, line)) { if (line.empty()) continue; std::stringstream ss(line); std::vector<int> t; int v; while (ss >> v) t.push_back(v);
        for (int m : modes) { Seq s; s.prompt = pi; s.mode = m; s.toks = t; s.plen = t.size(); seqs.push_back(s); } pi++; } }
    size_t steps = 0; for (auto &s : seqs) steps = std::max(steps, s.plen + NEW);
    fprintf(stderr, "%zu tensors, %d layers, %zu sequences, %zu steps\n", idx.size(), NL, seqs.size(), steps);

    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)300 << 20)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, 32768 * 4)); CK(cudaMalloc(&dxe, 32768 * 2)); CK(cudaMalloc(&dy, VOC * 4)); CK(cudaMalloc(&dbad, 4));
    void *dLM; CK(cudaMalloc(&dLM, (size_t)VOC * H * 2));
    DevW LM = load_w("lm_head.weight", F_BF16, VOC, H, dLM);
    auto normf = vec_bf16("backbone.norm_f.weight");
    std::map<int, std::vector<float>> lnorm;   // small per-layer tensors stay on the host
    struct MambaSmall { std::vector<float> A_log, D, cb, cw, dtb, nw; float is_in, ws_in, is_out, ws_out; };
    std::map<int, MambaSmall> msm;
    struct MoeSmall { std::vector<float> bias; float sis_u, sws_u, sis_d, sws_d; };
    std::map<int, MoeSmall> moes;
    for (int L = 0; L < NL; L++) {
        std::string P = "backbone.layers." + std::to_string(L) + ".";
        lnorm[L] = vec_bf16(P + "norm.weight");
        if (types[L] == "mamba") {
            MambaSmall m; std::string Q = P + "mixer.";
            m.A_log = vec_bf16(Q + "A_log"); m.D = vec_bf16(Q + "D"); m.cb = vec_bf16(Q + "conv1d.bias"); m.cw = vec_bf16(Q + "conv1d.weight");
            m.dtb = vec_bf16(Q + "dt_bias"); m.nw = vec_bf16(Q + "norm.weight");
            m.is_in = scalar(Q + "in_proj.input_scale"); m.ws_in = scalar(Q + "in_proj.weight_scale");
            m.is_out = scalar(Q + "out_proj.input_scale"); m.ws_out = scalar(Q + "out_proj.weight_scale");
            msm[L] = m;
        } else if (types[L] == "moe") {
            MoeSmall m; std::string Q = P + "mixer.";
            m.bias = vec_f32(Q + "gate.e_score_correction_bias");
            m.sis_u = scalar(Q + "shared_experts.up_proj.input_scale"); m.sws_u = scalar(Q + "shared_experts.up_proj.weight_scale");
            m.sis_d = scalar(Q + "shared_experts.down_proj.input_scale"); m.sws_d = scalar(Q + "shared_experts.down_proj.weight_scale");
            moes[L] = m;
        }
    }
    for (auto &s : seqs) for (int L = 0; L < NL; L++) {
        if (types[L] == "mamba") { s.conv[L].assign((size_t)CONV * 3, 0.f); s.ssm[L].assign((size_t)MH * MD * MN, 0.f); }
    }
    fprintf(stderr, "resident: lm_head + small tensors (%.1f GB read)\n", bytes_disk / 1e9);

    for (size_t t = 0; t < steps; t++) {
        auto t0 = std::chrono::steady_clock::now(); double d0 = t_disk, b0 = bytes_disk;
        for (auto &s : seqs) {   // embedding row of the current token
            int tok = s.toks[t]; std::vector<uint16_t> row(H);
            const TI e = idx.at("backbone.embeddings.weight"); TI r = e; r.off += (size_t)tok * H * 2; r.n = (size_t)H * 2;
            idx["__row"] = r; readt("__row", row.data());
            s.h.resize(H); for (int i = 0; i < H; i++) s.h[i] = bf16f(row[i]);
        }
        for (int L = 0; L < NL; L++) {
            std::string Q = "backbone.layers." + std::to_string(L) + ".mixer.";
            std::vector<std::vector<float>> x(seqs.size(), std::vector<float>(H)), out(seqs.size());
            for (size_t j = 0; j < seqs.size(); j++) rmsnorm(seqs[j].h, lnorm[L], x[j], 0, H);
            if (types[L] == "mamba") {
                auto &m = msm[L];
                DevW W = load_w(Q + "in_proj.weight", F_E4M3, 2 * DIN + 2 * NG * MN + MH, H); gscale(W, m.ws_in, m.is_in);
                std::vector<std::vector<float>> g(seqs.size(), std::vector<float>(DIN));
                for (size_t j = 0; j < seqs.size(); j++) {
                    Seq &s = seqs[j];
                    auto pr = gemv(W, act_fp8(x[j], m.is_in), 8, s.mode, s.mode == 0);
                    const float *z = pr.data(), *xbc = pr.data() + DIN, *dt = pr.data() + DIN + CONV;
                    auto &cs = s.conv[L]; std::vector<float> xc(CONV);
                    for (int c = 0; c < CONV; c++) {   // causal conv1d over the last 4 inputs, then SiLU, to BF16
                        double a = m.cb[c];
                        for (int q = 0; q < 3; q++) a += (double)cs[(size_t)c * 3 + q] * m.cw[(size_t)c * 4 + q];
                        a += (double)xbc[c] * m.cw[(size_t)c * 4 + 3];
                        float af = (float)a; xc[c] = bf((double)af / (1.0 + exp(-(double)af)));
                        cs[(size_t)c * 3 + 0] = cs[(size_t)c * 3 + 1]; cs[(size_t)c * 3 + 1] = cs[(size_t)c * 3 + 2]; cs[(size_t)c * 3 + 2] = xbc[c];
                    }
                    const float *xs = xc.data(), *B = xc.data() + DIN, *C = xc.data() + DIN + NG * MN;
                    auto &S = s.ssm[L]; std::vector<float> y(DIN);
                    #pragma omp parallel for schedule(static)
                    for (int hh = 0; hh < MH; hh++) {
                        int gi = hh / (MH / NG);
                        double v = (double)dt[hh] + m.dtb[hh]; float dtv = (float)(v > 20 ? v : log1p(exp(v)));   // softplus, no clamp (dt_limit 0..inf)
                        float A = (float)-exp((double)m.A_log[hh]); float dA = (float)exp((double)dtv * A);
                        for (int d = 0; d < MD; d++) {
                            float xv = xs[hh * MD + d]; double acc = 0;
                            float *st = &S[((size_t)hh * MD + d) * MN];
                            for (int n = 0; n < MN; n++) {
                                float dB = (float)((double)B[gi * MN + n] * dtv);
                                st[n] = (float)((double)st[n] * dA + (double)dB * xv);
                                acc += (double)st[n] * C[gi * MN + n];
                            }
                            y[hh * MD + d] = bf(acc + (double)xv * m.D[hh]);
                        }
                    }
                    for (int gq = 0; gq < NG; gq++) {   // gated RMSNorm per group of 2048: y * silu(z), normalized, times weight
                        size_t o = (size_t)gq * (DIN / NG); std::vector<float> gg(DIN / NG); double ss = 0;
                        for (int i = 0; i < DIN / NG; i++) { double zz = z[o + i]; gg[i] = (float)((double)y[o + i] * (zz / (1.0 + exp(-zz)))); ss += (double)gg[i] * gg[i]; }
                        float var = (float)(ss / (DIN / NG)); float r = (float)(1.0 / sqrt((double)var + 1e-5));
                        for (int i = 0; i < DIN / NG; i++) g[j][o + i] = bf((double)m.nw[o + i] * bf((double)gg[i] * r));
                    }
                }
                W = load_w(Q + "out_proj.weight", F_E4M3, H, DIN); gscale(W, m.ws_out, m.is_out);
                for (size_t j = 0; j < seqs.size(); j++) out[j] = gemv(W, act_fp8(g[j], m.is_out), 8, seqs[j].mode, seqs[j].mode == 0);
            } else if (types[L] == "attention") {
                std::vector<std::vector<float>> q(seqs.size()), o(seqs.size(), std::vector<float>(NQ * HD));
                DevW W = load_w(Q + "q_proj.weight", F_BF16, NQ * HD, H);
                for (size_t j = 0; j < seqs.size(); j++) q[j] = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                for (const char *kv : {"k_proj", "v_proj"}) {
                    W = load_w(Q + kv + ".weight", F_BF16, NKV * HD, H);
                    for (size_t j = 0; j < seqs.size(); j++) { auto r = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                        auto &c = kv[0] == 'k' ? seqs[j].kc[L] : seqs[j].vc[L]; c.insert(c.end(), r.begin(), r.end()); }
                }
                for (size_t j = 0; j < seqs.size(); j++) attend(q[j], seqs[j].kc[L], seqs[j].vc[L], NQ, NKV, HD, o[j]);
                W = load_w(Q + "o_proj.weight", F_BF16, H, NQ * HD);
                for (size_t j = 0; j < seqs.size(); j++) out[j] = gemv(W, act_plain(o[j]), 8, seqs[j].mode, seqs[j].mode == 0);
            } else {   // moe
                auto &m = moes[L];
                DevW W = load_w(Q + "gate.weight", F_F32, E, H);
                std::vector<std::vector<int>> ex(seqs.size()); std::vector<std::vector<double>> wt(seqs.size());
                for (size_t j = 0; j < seqs.size(); j++) {
                    auto lg = gemv(W, act_plain(x[j]), 24, seqs[j].mode, seqs[j].mode == 0);
                    std::vector<float> sg(E); std::vector<std::pair<float, int>> c(E);
                    for (int e = 0; e < E; e++) { sg[e] = (float)(1.0 / (1.0 + exp(-(double)lg[e]))); c[e] = {sg[e] + m.bias[e], e}; }
                    std::partial_sort(c.begin(), c.begin() + TOPK, c.end(), [](auto &a, auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                    double sum = 0; for (int k = 0; k < TOPK; k++) sum += sg[c[k].second];
                    for (int k = 0; k < TOPK; k++) { ex[j].push_back(c[k].second); wt[j].push_back(5.0 * sg[c[k].second] / (sum + 1e-20)); }
                }
                W = load_w(Q + "fc1_latent_proj.weight", F_BF16, LAT, H);
                std::vector<std::vector<float>> lat(seqs.size()); std::vector<std::vector<double>> accd(seqs.size(), std::vector<double>(LAT, 0.0));
                for (size_t j = 0; j < seqs.size(); j++) lat[j] = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                std::set<int> uni; for (auto &v : ex) uni.insert(v.begin(), v.end());
                for (int e : uni) {   // each chosen expert read once, applied to every sequence that chose it
                    std::string EP = Q + "experts." + std::to_string(e) + ".";
                    DevW Wu = load_w(EP + "up_proj.weight", F_NVFP4, 5120, LAT); readt(EP + "up_proj.weight_scale", pin);
                    CK(cudaMemcpy(dS, pin, tsize(EP + "up_proj.weight_scale"), cudaMemcpyHostToDevice)); Wu.sw = (uint8_t *)dS;
                    float gWu = scalar(EP + "up_proj.weight_scale_2"), gAu = scalar(EP + "up_proj.input_scale"); gscale(Wu, gWu, gAu);
                    std::vector<std::vector<float>> r(seqs.size());
                    for (size_t j = 0; j < seqs.size(); j++) if (std::count(ex[j].begin(), ex[j].end(), e)) {
                        auto u = gemv(Wu, act_nvfp4(lat[j], gAu), 8, seqs[j].mode, seqs[j].mode == 0);
                        r[j].resize(u.size()); for (size_t i = 0; i < u.size(); i++) { double v = u[i] > 0 ? u[i] : 0; r[j][i] = bf(v * v); }
                    }
                    DevW Wd = load_w(EP + "down_proj.weight", F_NVFP4, LAT, 5120); readt(EP + "down_proj.weight_scale", pin);
                    CK(cudaMemcpy(dS, pin, tsize(EP + "down_proj.weight_scale"), cudaMemcpyHostToDevice)); Wd.sw = (uint8_t *)dS;
                    float gWd = scalar(EP + "down_proj.weight_scale_2"), gAd = scalar(EP + "down_proj.input_scale"); gscale(Wd, gWd, gAd);
                    for (size_t j = 0; j < seqs.size(); j++) {
                        auto it = std::find(ex[j].begin(), ex[j].end(), e); if (it == ex[j].end()) continue;
                        auto d = gemv(Wd, act_nvfp4(r[j], gAd), 8, seqs[j].mode, seqs[j].mode == 0);
                        double w = wt[j][it - ex[j].begin()];
                        for (int i = 0; i < LAT; i++) accd[j][i] += w * d[i];   // FP64 in expert order, rounded once below
                    }
                }
                W = load_w(Q + "fc2_latent_proj.weight", F_BF16, H, LAT);
                for (size_t j = 0; j < seqs.size(); j++) { std::vector<float> a(LAT); for (int i = 0; i < LAT; i++) a[i] = bf(accd[j][i]);
                    out[j] = gemv(W, act_plain(a), 8, seqs[j].mode, seqs[j].mode == 0); }
                W = load_w(Q + "shared_experts.up_proj.weight", F_E4M3, 10240, H); gscale(W, m.sws_u, m.sis_u);
                std::vector<std::vector<float>> sh(seqs.size());
                for (size_t j = 0; j < seqs.size(); j++) { auto u = gemv(W, act_fp8(x[j], m.sis_u), 8, seqs[j].mode, seqs[j].mode == 0);
                    sh[j].resize(u.size()); for (size_t i = 0; i < u.size(); i++) { double v = u[i] > 0 ? u[i] : 0; sh[j][i] = bf(v * v); } }
                W = load_w(Q + "shared_experts.down_proj.weight", F_E4M3, H, 10240); gscale(W, m.sws_d, m.sis_d);
                for (size_t j = 0; j < seqs.size(); j++) { auto d = gemv(W, act_fp8(sh[j], m.sis_d), 8, seqs[j].mode, seqs[j].mode == 0);
                    for (int i = 0; i < H; i++) out[j][i] = bf((double)out[j][i] + d[i]); }
            }
            for (size_t j = 0; j < seqs.size(); j++) for (int i = 0; i < H; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + out[j][i]);
            for (auto &s : seqs) s.dig = fnv(s.dig, s.h);
        }
        for (auto &s : seqs) {   // final norm, logits (FP32, correctly rounded on the exact path), greedy
            std::vector<float> hn(H); rmsnorm(s.h, normf, hn, 0, H);
            auto lg = gemv(LM, act_plain(hn), 24, s.mode, s.mode == 0); s.dig = fnv(s.dig, lg);
            int b1 = pick(lg, t, s.prompt, s.mode, s.toks[t]);
            if (t + 1 >= s.plen && t + 1 >= s.toks.size()) s.toks.push_back(b1);
        }
        fflush(stdout);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "step %zu: %.1f s (disk %.1f s, %.1f GB, %.2f GB/s), uncertified so far %lld\n", t, sec, t_disk - d0,
                (bytes_disk - b0) / 1e9, (bytes_disk - b0) / 1e9 / std::max(1e-9, t_disk - d0), st_bad);
    }
    report(seqs);
}
#endif
