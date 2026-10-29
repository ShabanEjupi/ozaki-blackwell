// lora.cu -- one LoRA step of a frozen linear layer y = W x + s B (A x) with every product correctly rounded once,
// on the INT8 tensor cores (exact.cuh: digit planes, 128-bit integer sums), forward and backward:
//   H  = X A^T                      (BF16)     U  = G B                      (BF16)
//   Y  = [X | sH] [W | B]^T         (FP32, the frozen product and the adapter rounded together, one rounding)
//   dX = [G | sU] [W^T | A^T]^T     (FP32, idem)
//   dB = s G^T H,  dA = s U^T X     (FP32, sums over the tokens)
// G = dL/dY. s is a power of two (alpha / r), so the scalings are exact. Every output is a function of the exact real
// value alone, hence:
//   * the rows of Y, H, U, dX of a token are the same bits whatever else is in the batch (batch size 1 or N);
//   * dA, dB are the same bits for any order of the tokens (the floating-point sums of a BF16/FP32 engine are not);
//   * sampled outputs of every GEMM are checked against an independent host fixed-point sum (640 bits, no window).
// The baseline is the same products on BF16 tensor cores with FP32 accumulation (split-K 1 and 4, ex::bf16_gemm).
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_lora llm/lora.cu ; out_lora [hidden 8192] [tokens 1024] [rank 16] [seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "exact.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

using BF = std::vector<uint16_t>;
using FV = std::vector<float>;
static uint16_t f2b(float f) {   // nearest even; the inputs here are finite
    uint32_t u; memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
static float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// ---- independent reference: the exact sum of BF16 products in fixed point (bit i = 2^(i - OFF)), then one rounding ------
struct Fx {
    static constexpr int NL = 10, OFF = 272;   // products >= 2^-266, < 2^256 times K < 2^20: bits 6 .. 547
    uint64_t l[NL] = {};
    void add(uint64_t v, int pos) {
        const int j = pos / 64, s = pos % 64;
        const unsigned __int128 t = (unsigned __int128)v << s;
        uint64_t part[2] = {(uint64_t)t, (uint64_t)(t >> 64)};
        for (int q = 0; q < 2; q++) {
            unsigned __int128 c = (unsigned __int128)l[j + q] + part[q];
            l[j + q] = (uint64_t)c;
            for (int i = j + q + 1; (c >> 64) && i < NL; i++) { c = (unsigned __int128)l[i] + 1; l[i] = (uint64_t)c; }
        }
    }
    bool ge(const Fx &b) const { for (int j = NL - 1; j >= 0; j--) if (l[j] != b.l[j]) return l[j] > b.l[j]; return true; }
    void sub(const Fx &b) { uint64_t br = 0; for (int j = 0; j < NL; j++) { unsigned __int128 t = (unsigned __int128)l[j] - b.l[j] - br; l[j] = (uint64_t)t; br = (t >> 64) ? 1 : 0; } }
    int bit(int i) const { return i < 0 ? 0 : (l[i / 64] >> (i % 64)) & 1; }
};
static float ref_dot(const uint16_t *a, const uint16_t *b, int K, int p) {
    Fx pos, neg;
    for (int k = 0; k < K; k++) {
        const int ea = (a[k] >> 7) & 255, eb = (b[k] >> 7) & 255;
        const uint64_t ma = ea ? (a[k] & 127) | 128 : a[k] & 127, mb = eb ? (b[k] & 127) | 128 : b[k] & 127;
        if (!ma || !mb) continue;
        const int e = (ea ? ea - 134 : -133) + (eb ? eb - 134 : -133);
        ((a[k] ^ b[k]) & 0x8000 ? neg : pos).add(ma * mb, e + Fx::OFF);
    }
    const bool ng = !pos.ge(neg);
    Fx d = ng ? neg : pos; d.sub(ng ? pos : neg);
    int top = -1;
    for (int j = Fx::NL - 1; j >= 0 && top < 0; j--) if (d.l[j]) top = 64 * j + 63 - __builtin_clzll(d.l[j]);
    if (top < 0) return 0.f;
    const int emin = -149 + (24 - p), L = std::max(top - Fx::OFF - (p - 1), emin), lsb = L + Fx::OFF;
    uint64_t q = 0;
    for (int i = top; i >= lsb; i--) q = 2 * q + d.bit(i);
    bool sticky = false;
    for (int i = lsb - 2; i >= 0 && !sticky; i--) sticky = d.bit(i);
    if (d.bit(lsb - 1) && (sticky || (q & 1))) q++;
    double f = ldexp((double)q, L);
    if (f > (p == 24 ? 0x1.fffffep127 : 0x1.fep127)) f = INFINITY;
    return (float)(ng ? -f : f);
}

// ---- Y[j][i] = sum_k A[i][k] X[j][k], correctly rounded to p bits (24: FP32, 8: BF16); M, N % 128, K % 64 ---------------
static int S = 3, TM = 4;   // three int8 digits per element (13-bit exponent windows), up to four tiers per row
static long long g_checked = 0, g_fail = 0, g_nan = 0;
static FV exmm(const char *name, const BF &A, const BF &X, int M, int N, int K, int p, std::mt19937_64 &rng, int samples = 4096) {
    __nv_bfloat16 *dA, *dX; int *stat, *nbad; int8_t *digA, *digX; int *BA, *BX; float *Y;
    CK(cudaMalloc(&dA, A.size() * 2)); CK(cudaMalloc(&dX, X.size() * 2));
    CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&digA, (size_t)TM * S * M * K)); CK(cudaMalloc(&digX, (size_t)TM * S * N * K));
    CK(cudaMalloc(&BA, (size_t)TM * M * 4)); CK(cudaMalloc(&BX, (size_t)TM * N * 4)); CK(cudaMalloc(&stat, 24)); CK(cudaMalloc(&nbad, 4));
    CK(cudaMemset(stat, 0, 24)); CK(cudaMemset(nbad, 0, 4)); CK(cudaMalloc(&Y, (size_t)M * N * 4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);   // the time includes the split
    int Z = 0, st[6]; float ms; char how[48];
    bool window = !getenv("OZ_DENSE");
    if (window) {   // each row's densest window in S digits + at most C sparse elements: S^2 planes instead of (T S)^2
        const int C = getenv("OZ_C") ? atoi(getenv("OZ_C")) : 64; int *em, *idx, *cnt;
        const int SX = getenv("OZ_SX") ? atoi(getenv("OZ_SX")) : S;   // the digits of X (the tokens): 2 = a 6-bit window, more sparse
        const int SA = getenv("OZ_SA") && (!strcmp(name, "Y") || !strcmp(name, "dX")) ? atoi(getenv("OZ_SA")) : S;   // the digits of the first operand (W): 1 or 2 carry a quantized base
        CK(cudaMalloc(&em, (size_t)(M + N) * 4)); CK(cudaMalloc(&idx, (size_t)(M + N) * C * 4)); CK(cudaMalloc(&cnt, (size_t)(M + N) * 4));
        int32_t *P; CK(cudaMalloc(&P, (size_t)S * S * M * N * 4));   // allocated outside the timed part
        cudaEvent_t t1, t2; cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventRecord(e0);
        ex::split_window<<<M, 256>>>(dA, digA, BA, em, idx, cnt, stat, M, K, SA, C);
        ex::split_window<<<N, 256>>>(dX, digX, BX, em + M, idx + (size_t)M * C, cnt + M, stat + 3, N, K, SX, C);
        CK(cudaMemcpy(st, stat, 24, cudaMemcpyDeviceToHost));
        if (st[2] || st[5]) { fprintf(stderr, "%s: inf/nan %d %d\n", name, st[2], st[5]); exit(1); }
        if (!st[1] && !st[4]) {
            ex::Digits Wd{digA, BA, 1, SA, M, K}, Xd{digX, BX, 1, SX, N, K};
            Z = SA * SX;
            cudaEventRecord(t1);
            ex::gemm_pairs<<<dim3(N / oz::BN, M / oz::BM, Z), oz::THREADS>>>(Wd, Xd, P, 1);
            cudaEventRecord(t2);
            ex::combine_window<<<dim3((N + 255) / 256, M), 256>>>(P, Wd, Xd, ex::Window{dA, em, idx, cnt, C, M},
                ex::Window{dX, em + M, idx + (size_t)M * C, cnt + M, C, N}, p, Y, M, nbad);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&ms, e0, e1);
            if (getenv("OZ_PROF")) { float a, b; cudaEventElapsedTime(&a, e0, t1); cudaEventElapsedTime(&b, t1, t2); printf("      split %.1f  gemm %.1f  combine %.1f ms\n", a, b, ms - a - b); }
            snprintf(how, sizeof how, "window, sparse <= %d,%d", st[0], st[3]);
        } else { window = false; CK(cudaMemset(stat, 0, 24)); cudaEventRecord(e0); }   // a row has more than C: the tiers
        for (void *q : {(void *)em, (void *)idx, (void *)cnt, (void *)P}) CK(cudaFree(q));
    }
    if (!window) {
        ex::split_rows<<<M, 256>>>(dA, digA, BA, stat, M, K, S, TM);
        ex::split_rows<<<N, 256>>>(dX, digX, BX, stat + 3, N, K, S, TM);
        CK(cudaMemcpy(st, stat, 24, cudaMemcpyDeviceToHost));
        if (st[1] || st[2] || st[4] || st[5]) { fprintf(stderr, "%s: left over %d %d, inf/nan %d %d (more tiers needed)\n", name, st[1], st[4], st[2], st[5]); exit(1); }
        ex::Digits Wd{digA, BA, std::max(st[0], 1), S, M, K}, Xd{digX, BX, std::max(st[3], 1), S, N, K};
        Z = Wd.T * S * Xd.T * S;
        int32_t *P; CK(cudaMalloc(&P, (size_t)Z * M * N * 4));
        cudaEventRecord(e0);   // the tiers are timed without their split, as before 29-10 (the window path includes its own)
        ex::gemm_pairs<<<dim3(N / oz::BN, M / oz::BM, Z), oz::THREADS>>>(Wd, Xd, P, Xd.T);
        ex::combine<<<dim3((M + 255) / 256, N), 256>>>(P, Wd, Xd, Wd.T, Xd.T, p, Y, M, nbad);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); cudaEventElapsedTime(&ms, e0, e1);
        CK(cudaFree(P));
        snprintf(how, sizeof how, "tiers %d,%d", Wd.T, Xd.T);
    }
    FV out((size_t)N * M); int bad;
    CK(cudaMemcpy(out.data(), Y, out.size() * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&bad, nbad, 4, cudaMemcpyDeviceToHost));
    for (void *q : {(void *)dA, (void *)dX, (void *)digA, (void *)digX, (void *)BA, (void *)BX, (void *)stat, (void *)nbad, (void *)Y}) CK(cudaFree(q));
    int fail = 0;
    for (int s = 0; s < samples; s++) {
        const int i = rng() % M, j = rng() % N;
        const float r = ref_dot(&A[(size_t)i * K], &X[(size_t)j * K], K, p), g = out[(size_t)j * M + i];
        if (fbits(r) != fbits(g)) { if (fail++ < 3) fprintf(stderr, "  %s (%d,%d): got %a, exact %a\n", name, i, j, g, r); }
    }
    g_checked += samples; g_fail += fail; g_nan += bad;
    printf("  %-3s %5d x %5d x %5d  %-22s %3d planes  %7.1f ms  %5.1f effective TOPS  uncertified %d  sampled %d, differ %d\n",
           name, M, N, K, how, Z, ms, 2.0 * M * N * K / ms / 1e9, bad, samples, fail);
    return out;
}

// the baseline: BF16 tensor cores, FP32 accumulation, split-K S (K / S multiple of 32), same layout as exmm
static FV bfmm(const BF &A, const BF &X, int M, int N, int K, int Sk) {
    __nv_bfloat16 *dA, *dX; float *part, *Y;
    CK(cudaMalloc(&dA, A.size() * 2)); CK(cudaMalloc(&dX, X.size() * 2)); CK(cudaMalloc(&part, (size_t)Sk * M * N * 4)); CK(cudaMalloc(&Y, (size_t)M * N * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice));
    ex::bf16_gemm<<<dim3(N / ex::BN, M / ex::BM, Sk), ex::THREADS>>>(dA, dX, part, M, N, K, Sk);
    ex::reduce_k<<<1024, 256>>>(part, Y, (size_t)M * N, Sk);
    FV out((size_t)M * N); CK(cudaMemcpy(out.data(), Y, out.size() * 4, cudaMemcpyDeviceToHost));
    for (void *q : {(void *)dA, (void *)dX, (void *)part, (void *)Y}) CK(cudaFree(q));
    return out;
}

static BF transpose(const BF &a, int R, int C) { BF t((size_t)R * C); for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) t[(size_t)c * R + r] = a[(size_t)r * C + c]; return t; }
static BF tobf(const FV &v, float s = 1.f) { BF o(v.size()); for (size_t i = 0; i < v.size(); i++) o[i] = f2b(v[i] * s); return o; }
static BF hcat(const BF &a, int Ca, const BF &b, int Cb, int R) {   // [a | b], R rows
    BF o((size_t)R * (Ca + Cb));
    for (int r = 0; r < R; r++) {
        std::copy_n(&a[(size_t)r * Ca], Ca, &o[(size_t)r * (Ca + Cb)]);
        std::copy_n(&b[(size_t)r * Cb], Cb, &o[(size_t)r * (Ca + Cb) + Ca]);
    }
    return o;
}

struct Step { FV H, U, Y, dX, dA, dBt; };
// X [N][D], G [N][D], W [D][D], Wt, Ap [RP][D] (rows >= r zero), Bp [D][RP] (columns >= r zero); s = 2^ls
static Step lora_step(const BF &X, const BF &G, const BF &W, const BF &Wt, const BF &Ap, const BF &Bp, int D, int N, int RP, int ls, std::mt19937_64 &rng) {
    Step o; const float s = ldexpf(1.f, ls);
    o.H = exmm("H", Ap, X, RP, N, D, 8, rng);                                    // [N][RP]
    const BF Bt = transpose(Bp, D, RP);
    o.U = exmm("U", Bt, G, RP, N, D, 8, rng);                                    // [N][RP]
    const BF Hs = tobf(o.H, s), Us = tobf(o.U, s);
    o.Y = exmm("Y", hcat(W, D, Bp, RP, D), hcat(X, D, Hs, RP, N), D, N, D + RP, 24, rng);
    o.dX = exmm("dX", hcat(Wt, D, transpose(Ap, RP, D), RP, D), hcat(G, D, Us, RP, N), D, N, D + RP, 24, rng);
    o.dBt = exmm("dB", transpose(G, N, D), transpose(tobf(o.H), N, RP), D, RP, N, 24, rng);   // [RP][D] = dB^T / s
    o.dA = exmm("dA", transpose(X, N, D), transpose(tobf(o.U), N, RP), D, RP, N, 24, rng);    // [RP][D] = dA / s
    for (auto *v : {&o.dBt, &o.dA}) for (auto &f : *v) f *= s;                    // exact: a power of two
    return o;
}

static int diff(const float *a, const float *b, size_t n) { int d = 0; for (size_t i = 0; i < n; i++) d += fbits(a[i]) != fbits(b[i]); return d; }
// ulps of a against the correctly rounded c (same sign region assumed; counts distance in the float lattice)
static double ulps(float a, float c) { int32_t x = (int32_t)fbits(a), y = (int32_t)fbits(c); if (x < 0) x = INT32_MIN - x; if (y < 0) y = INT32_MIN - y; return fabs((double)x - y); }

int main(int argc, char **argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 8192, N = argc > 2 ? atoi(argv[2]) : 1024, r = argc > 3 ? atoi(argv[3]) : 16;
    const uint64_t seed = argc > 4 ? atoll(argv[4]) : 2026;
    const int RP = 128, ls = 1;   // rank padded to the 128-row tile; s = alpha / r = 2
    if (D % 128 || N % 256 || r > RP) { fprintf(stderr, "hidden %% 128, tokens %% 256, rank <= 128\n"); return 1; }
    std::mt19937_64 rng(seed); std::normal_distribution<double> nd; std::student_t_distribution<double> td(4.0);
    printf("LoRA step, hidden %d, tokens %d, rank %d (padded %d), s = %d, seed %llu\n", D, N, r, RP, 1 << ls, (unsigned long long)seed);
    BF W((size_t)D * D), X((size_t)N * D), G((size_t)N * D), Ap((size_t)RP * D, 0), Bp((size_t)D * RP, 0);
    for (auto &w : W) w = f2b((float)(0.02 * td(rng)));                           // heavy-tailed frozen weights
    for (int j = 0; j < N; j++) { const double sc = exp2(rng() % 8 - 4.0); for (int k = 0; k < D; k++) X[(size_t)j * D + k] = f2b((float)(sc * nd(rng))); }
    if (const char *o = getenv("OZ_OUTLIERS"))   // LLM-style outlier channels: the same few features 2^7 larger in every token
        for (int c = 0, n = atoi(o); c < n; c++) { const int k = rng() % D; for (int j = 0; j < N; j++) X[(size_t)j * D + k] = f2b(128.f * b2f(X[(size_t)j * D + k])); }
    if (const char *wf = getenv("OZ_W")) {   // real weights and inputs (llm/jais_lora_prep.py): W [D][D] and the first N rows of X, BF16
        auto rd = [](const char *p, std::vector<uint16_t> &v) { FILE *f = fopen(p, "rb"); if (!f || fread(v.data(), 2, v.size(), f) != v.size()) { fprintf(stderr, "cannot read %zu values from %s\n", v.size(), p); exit(1); } fclose(f); };
        rd(wf, W); if (const char *xf = getenv("OZ_X")) rd(xf, X);
        printf("real W from %s%s\n", wf, getenv("OZ_X") ? ", real X" : "");
    }
    for (auto &g : G) g = f2b((float)(1e-4 * td(rng)));                           // upstream gradient
    for (int q = 0; q < r; q++) for (int k = 0; k < D; k++) Ap[(size_t)q * D + k] = f2b((float)(nd(rng) / sqrt((double)D)));
    for (int i = 0; i < D; i++) for (int q = 0; q < r; q++) Bp[(size_t)i * RP + q] = f2b((float)(1e-3 * nd(rng)));
    if (getenv("OZ_WQ"))   // the frozen base quantized to MXFP4 (QLoRA-style), held exactly in BF16: E2M1 x 2^scale per 32
        for (int i = 0; i < D; i++) for (int g = 0; g < D; g += 32) {
            float mx = 0; for (int k = g; k < g + 32; k++) mx = std::max(mx, fabsf(b2f(W[(size_t)i * D + k])));
            if (mx == 0) continue;
            const float sc = ldexpf(1.f, (int)floorf(log2f(mx)) - 2), grid[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
            for (int k = g; k < g + 32; k++) {
                const float v = b2f(W[(size_t)i * D + k]) / sc; int bq = 0;
                for (int q = 1; q < 8; q++) if (fabsf(fabsf(v) - grid[q]) < fabsf(fabsf(v) - grid[bq])) bq = q;
                W[(size_t)i * D + k] = f2b(copysignf(grid[bq] * sc, v));
            }
        }
    const BF Wt = transpose(W, D, D);

    printf("[1] the step, %d tokens\n", N);
    const Step F = lora_step(X, G, W, Wt, Ap, Bp, D, N, RP, ls, rng);

    const int n1 = 256;   // the first 256 tokens alone (smallest batch the 128-row tiles and split-K 4 take)
    printf("[2] the same layer, the first %d tokens alone\n", n1);
    const BF X1(X.begin(), X.begin() + (size_t)n1 * D), G1(G.begin(), G.begin() + (size_t)n1 * D);
    const Step F1 = lora_step(X1, G1, W, Wt, Ap, Bp, D, n1, RP, ls, rng);
    printf("    per-token rows that differ from the batch of %d: H %d  U %d  Y %d  dX %d  (of %d, %d, %d, %d)\n", N,
           diff(F1.H.data(), F.H.data(), F1.H.size()), diff(F1.U.data(), F.U.data(), F1.U.size()),
           diff(F1.Y.data(), F.Y.data(), F1.Y.size()), diff(F1.dX.data(), F.dX.data(), F1.dX.size()),
           (int)F1.H.size(), (int)F1.U.size(), (int)F1.Y.size(), (int)F1.dX.size());

    printf("[3] the same %d tokens in reverse order\n", N);
    BF Xr(X.size()), Gr(G.size());
    for (int j = 0; j < N; j++) { std::copy_n(&X[(size_t)j * D], D, &Xr[(size_t)(N - 1 - j) * D]); std::copy_n(&G[(size_t)j * D], D, &Gr[(size_t)(N - 1 - j) * D]); }
    const Step R = lora_step(Xr, Gr, W, Wt, Ap, Bp, D, N, RP, ls, rng);
    int dy = 0, ddx = 0;
    for (int j = 0; j < N; j++) { dy += diff(&R.Y[(size_t)(N - 1 - j) * D], &F.Y[(size_t)j * D], D); ddx += diff(&R.dX[(size_t)(N - 1 - j) * D], &F.dX[(size_t)j * D], D); }
    printf("    differ from the forward order: dA %d  dB %d  (of %d each)  Y %d  dX %d\n",
           diff(R.dA.data(), F.dA.data(), (size_t)r * D), diff(R.dBt.data(), F.dBt.data(), (size_t)r * D), r * D, dy, ddx);

    printf("[4] baseline (BF16 tensor cores, FP32 sums) for dB^T = G^T H, the token sum\n");
    const BF Gt = transpose(G, N, D), Ht = transpose(tobf(F.H), N, RP), Grt = transpose(Gr, N, D), Hrt = transpose(tobf(R.H), N, RP);
    const FV b1 = bfmm(Gt, Ht, D, RP, N, 1), b4 = bfmm(Gt, Ht, D, RP, N, 4), br = bfmm(Grt, Hrt, D, RP, N, 1);
    // exmm leaves [RP][D] (row = adapter column), bfmm [RP][D] too: part[z][Np][M] with M = D, Np = RP
    long nbr = 0; double mx = 0, sum = 0; int cnt = 0;
    for (int q = 0; q < r; q++) for (int i = 0; i < D; i++) {
        const size_t t = (size_t)q * D + i; const float c = ldexpf(F.dBt[t], -ls);   // undo s (exact)
        nbr += fbits(b1[t]) != fbits(c); const double u = ulps(b1[t], c); mx = std::max(mx, u); sum += u; cnt++;
    }
    printf("    split-K 1 vs 4: %d differ   forward vs reverse order: %d differ   (of %d)\n",
           diff(b1.data(), b4.data(), (size_t)r * D), diff(b1.data(), br.data(), (size_t)r * D), r * D);
    printf("    split-K 1 vs correctly rounded: %ld differ, mean %.2f ulp, max %.0f ulp\n", nbr, sum / cnt, mx);

    {   // the speed baseline: cuBLAS BF16 tensor cores, FP32 accumulation, the shape of Y (best of 5 after a warm-up)
        const BF a = hcat(W, D, Bp, RP, D), x = hcat(X, D, tobf(F.H, ldexpf(1.f, ls)), RP, N); const int K = D + RP;
        __nv_bfloat16 *da, *dx; float *dy; CK(cudaMalloc(&da, a.size() * 2)); CK(cudaMalloc(&dx, x.size() * 2)); CK(cudaMalloc(&dy, (size_t)D * N * 4));
        CK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        cublasHandle_t h; cublasCreate(&h); const float one = 1, zero = 0; float best = 1e30f;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int it = 0; it < 6; it++) {
            cudaEventRecord(e0);
            cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, D, N, K, &one, da, CUDA_R_16BF, K, dx, CUDA_R_16BF, K, &zero, dy, CUDA_R_32F, D, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); float ms; cudaEventElapsedTime(&ms, e0, e1); if (it) best = std::min(best, ms);
        }
        printf("[5] cuBLAS BF16 (FP32 sums) for Y, %d x %d x %d: %.1f ms = %.1f TFLOPS\n", D, N, K, best, 2.0 * D * N * K / best / 1e9);
        cublasDestroy(h); for (void *q : {(void *)da, (void *)dx, (void *)dy}) CK(cudaFree(q));
    }
    printf("reference: %lld sampled outputs against the 640-bit fixed-point sum, %lld differ, %lld uncertified\n", g_checked, g_fail, g_nan);
    return g_fail || g_nan ? 2 : 0;
}
