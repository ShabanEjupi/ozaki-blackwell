// jais_lora.cu -- one LoRA training step of Jais-3-375B-A23B-FP8 (61 layers, 192 FP8 experts) streamed from disk on one
// laptop GPU: rank-r adapters on q_proj and v_proj of every layer, B = 0 and A ~ U(-1/sqrt(H), 1/sqrt(H)) (PEFT's init),
// next-token cross-entropy over one sequence. Every linear map of the forward AND of the backward is one product on the
// INT8 tensor cores with every output correctly rounded (bgemm.cu), so the step is a function of its inputs alone.
//  * Forward: pass() of jais_batch.cu as it is (bit for bit the token-by-token model), saving the residual stream that
//    enters every layer (HSAVE); then the final norm, lm_head with FP32 logits, the loss.
//  * Backward, from the top, every layer read from disk once more and recomputed from its saved input with the forward's
//    own functions, then:
//    - dX = dY W for each map: dY rounded once to BF16 (the gradient format of BF16 training), W transposed on the GPU,
//      one certified product. An FP8 expert weight times its 128x128 block scale is exact in FP32 (4 x 8 significant
//      bits) and splits exactly into hi + lo, two BF16 numbers, so [dY | dY] [W_hi ; W_lo] is the exact product with
//      the FP8 weights and the gradient is never quantized to FP8. Maps that share an input go in one product (q, k, v:
//      K = 8192; an expert's gate and up: K = 4 x 1792), one rounding for the sum.
//    - MoE: the routing weights w_e = 2.5 s_e / sum s (sigmoids of the chosen 8) pass their gradient through the sigmoid
//      to the router's logits and its transposed product; the top-8 choice is a constant, as in every MoE backward.
//    - attention in FP64 on the host (probabilities and their gradient, per sequence: the batch never enters), RoPE's
//      inverse rotation, RMSNorm and SwiGLU in FP64; every sum there runs in a fixed order.
//    - LoRA: G = dL/d(q_proj output) before RoPE, and dL/d(v_proj output); H = BF16(x A^T) by the same products;
//      dB = s G^T H as one certified product over the tokens (K = tokens), so no bit of dB depends on the token order or
//      on how rows are cut into products. dA = s (G B)^T x is exactly 0 at B = 0, the first step of every LoRA run.
// Output: the loss, per layer |g| (the residual stream's gradient) and |dB_q|, |dB_v|, and digests of every dB bit and of
// g at the embeddings: the same digests with OZ_NB (rows per product) and OMP_NUM_THREADS changed are the reproducibility
// proof. OZ_NL=<n>: the model truncated to its first n layers (every code path at a fraction of the reads).
// OZ_EPS=<e>: forward only, the embeddings move to BF16(h +- e d), d a seeded +-1 direction; the step itself prints
// <g_0, D+ - D->, D+- the moves as rounded, so (L+ - L-) of the two forward runs checks the backward (OZ_FDEPS=<e>).
// nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jlora llm/jais_lora.cu -lgomp
// out_jlora <model dir> <token ids, one sequence> [tokens N] [rank r] [alpha] [dir for dB_q, dB_v]
#define JAIS_BATCH_LIB
#include "jais_batch.cu"
#include <functional>

static uint64_t sm64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31);
}
static double EPS; static uint64_t DSEED = 0xD1CEULL;
static void perturb(std::vector<float> &h) {   // h -> BF16(h + EPS d), d_i = +-1
    for (size_t i = 0; i < h.size(); i++) h[i] = bf((double)h[i] + EPS * ((sm64(i ^ DSEED) & 1) ? 1.0 : -1.0));
}

// ---- transposed weights for dX = dY W -------------------------------------------------------------------------------------
__global__ void tr_bf16(const uint16_t *s, uint16_t *d, int M, int K, int ld, int off) {   // d[k][off + m] = s[m][k]
    __shared__ uint16_t t[32][33];
    int k0 = blockIdx.x * 32, m0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) { int m = m0 + i, k = k0 + threadIdx.x; if (m < M && k < K) t[i][threadIdx.x] = s[(size_t)m * K + k]; }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) { int k = k0 + i, m = m0 + threadIdx.x; if (m < M && k < K) d[(size_t)k * ld + off + m] = t[threadIdx.x][i]; }
}
__device__ float e4m3f(uint8_t b) {   // E4M3FN: (8 + m) 2^(e - 10), subnormal m 2^-9 (weights hold no NaN)
    int e = (b >> 3) & 15, m = b & 7; float v = e ? ldexpf((float)(8 + m), e - 10) : ldexpf((float)m, -9); return (b & 128) ? -v : v;
}
// w = e4m3(s[m][k]) scale[m/128][k/128], exact in FP32; d[k][off + m] = BF16(w), d[k][off + M + m] = w - BF16(w) (exact)
__global__ void tr_fp8_hilo(const uint8_t *s, const uint16_t *sc, uint16_t *d, int M, int K, int ld, int off, int *bad) {
    __shared__ float t[32][33];
    int k0 = blockIdx.x * 32, m0 = blockIdx.y * 32, KB = K / 128;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) { int m = m0 + i, k = k0 + threadIdx.x;
        if (m < M && k < K) t[i][threadIdx.x] = e4m3f(s[(size_t)m * K + k]) * __uint_as_float((unsigned)sc[(m / 128) * KB + k / 128] << 16); }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) { int k = k0 + i, m = m0 + threadIdx.x; if (m < M && k < K) {
        float w = t[threadIdx.x][i]; __nv_bfloat16 hi = __float2bfloat16_rn(w); float lo = w - __bfloat162float(hi);
        __nv_bfloat16 lb = __float2bfloat16_rn(lo); if (__bfloat162float(lb) != lo) atomicAdd(bad, 1);
        d[(size_t)k * ld + off + m] = *(uint16_t *)&hi; d[(size_t)k * ld + off + M + m] = *(uint16_t *)&lb; } }
}
static void *dWt, *dSm; static size_t WT_MAX, SM_MAX; static int *dtbad;
static DevW devw(const void *p, int M, int K) { DevW W{}; W.w = p; W.fmt = F_BF16; W.M = M; W.K = K; W.Gm = 1; W.Ge = 0; W.Gf = 1.f; return W; }
// [W_1^T | W_2^T | ...] for maps that share their input dimension, an FP8 one as hi | lo, zeros up to a multiple of 128
static DevW transposed(const std::vector<DevW> &Ws, int &kpad) {
    int K = Ws[0].K, Kt = 0; for (auto &W : Ws) Kt += (W.fmt == F_E4M3B ? 2 : 1) * W.M;
    kpad = (Kt + 127) / 128 * 128; size_t need = (size_t)K * kpad * 2;
    if (need > WT_MAX) { fprintf(stderr, "transposed %d x %d exceeds its buffer\n", K, kpad); exit(1); }
    CK(cudaMemset(dWt, 0, need));
    int off = 0;
    for (auto &W : Ws) {
        dim3 g((W.K + 31) / 32, (W.M + 31) / 32), b(32, 8);
        if (W.fmt == F_E4M3B) { tr_fp8_hilo<<<g, b>>>((const uint8_t *)W.w, (const uint16_t *)W.sw, (uint16_t *)dWt, W.M, W.K, kpad, off, dtbad); off += 2 * W.M; }
        else { tr_bf16<<<g, b>>>((const uint16_t *)W.w, (uint16_t *)dWt, W.M, W.K, kpad, off); off += W.M; }
    }
    CK(cudaGetLastError());
    return devw(dWt, K, kpad);
}
// dX += [dY_1 | dY_2 | ...] [W_1 ; W_2 ; ...] (the forward was Y_i = X W_i^T): every output one correctly rounded FP32 of
// the exact sum, added to the FP64 dX
static void dgrad(const std::vector<DevW> &Ws, const std::vector<const std::vector<double> *> &dY, size_t N, std::vector<double> &dX) {
    int kpad; DevW Wt = transposed(Ws, kpad);
    std::vector<float> X(N * (size_t)kpad, 0.f), Y(N * (size_t)Wt.M);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < N; r++) { size_t off = 0; float *x = X.data() + r * kpad;
        for (size_t i = 0; i < Ws.size(); i++) { int M = Ws[i].M; const double *g = dY[i]->data() + r * M;
            for (int rep = 0; rep < (Ws[i].fmt == F_E4M3B ? 2 : 1); rep++, off += M) for (int m = 0; m < M; m++) x[off + m] = bf(g[m]); } }
    bmm(Wt, X.data(), N, 24, Y.data());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < Y.size(); i++) dX[i] += Y[i];
}
static DevW up(const std::vector<float> &m, int M, int K, void *buf = nullptr) {   // a host matrix, rounded to BF16, as a BF16 map
    if (!buf && (size_t)M * K * 2 > SM_MAX) { fprintf(stderr, "%d x %d exceeds the small buffer\n", M, K); exit(1); }
    std::vector<uint16_t> b(m.size()); for (size_t i = 0; i < m.size(); i++) { float f = bf(m[i]); uint32_t u; memcpy(&u, &f, 4); b[i] = u >> 16; }
    CK(cudaMemcpy(buf ? buf : dSm, b.data(), b.size() * 2, cudaMemcpyHostToDevice)); return devw(buf ? buf : dSm, M, K);
}

// ---- the host parts of the backward, FP64 ---------------------------------------------------------------------------------
static void rope_back(double *v, int heads, size_t pos) {   // the transposed rotation of rope_part
    for (int h = 0; h < heads; h++) for (int i = 0; i < J_RHD / 2; i++) {
        double a = (double)pos * inv_freq[i], c = cos(a), s = sin(a), d1 = v[h * D_HD + i], d2 = v[h * D_HD + i + D_HD / 2];
        v[h * D_HD + i] = d1 * c + d2 * s; v[h * D_HD + i + D_HD / 2] = d2 * c - d1 * s;
    }
}
// y = w x r, r = (mean x^2 + eps)^-1/2: g += r w dy - x r^3 (sum_j w_j dy_j x_j) / H
static void norm_back(const std::vector<float> &x, const std::vector<float> &w, const std::vector<double> &dy, std::vector<double> &g) {
    size_t R = x.size() / D_H;
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < R; r++) { const float *xr = x.data() + r * D_H; const double *d = dy.data() + r * D_H; double *gr = g.data() + r * D_H;
        double ss = 0, s = 0; for (int i = 0; i < D_H; i++) { ss += (double)xr[i] * xr[i]; s += (double)w[i] * d[i] * xr[i]; }
        double rr = 1.0 / sqrt(ss / D_H + D_EPS), c = rr * rr * rr * s / D_H;
        for (int i = 0; i < D_H; i++) gr[i] += rr * w[i] * d[i] - xr[i] * c; }
}
// a = silu(g) u: dg = da u silu'(g), du = da silu(g), silu' = s (1 + g (1 - s)) with s the sigmoid
static void swiglu_back(const std::vector<float> &gg, const std::vector<float> &uu, const std::vector<double> &da, std::vector<double> &dg, std::vector<double> &du) {
    dg.resize(gg.size()); du.resize(gg.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < gg.size(); i++) { double g = gg[i], s = 1.0 / (1.0 + exp(-g)); dg[i] = da[i] * uu[i] * s * (1.0 + g * (1.0 - s)); du[i] = da[i] * g * s; }
}
// w_k = c s_k / S over the chosen experts (s the sigmoids of the logits): ds_j = c (dw_j / S - sum_k dw_k s_k / S^2),
// dlogit_j = ds_j s_j (1 - s_j); the other logits get 0
static void gate_back(const float *lg, const int *ex, const double *dw, double *dl) {
    double s[64], S = 0, q = 0;
    for (int k = 0; k < J_TOPK; k++) { s[k] = 1.0 / (1.0 + exp(-(double)lg[ex[k]])); S += s[k]; }
    for (int k = 0; k < J_TOPK; k++) q += dw[k] * s[k];
    for (int k = 0; k < J_TOPK; k++) dl[ex[k]] = J_RSF * (dw[k] / S - q / (S * S)) * s[k] * (1 - s[k]);
}
// causal attention of one sequence (q, k after RoPE): dq, dk, dv from dO; P and dS are kept per head, query, key <= query
static void attend_back(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v, const std::vector<double> &dO, size_t N,
                        std::vector<double> &dq, std::vector<double> &dk, std::vector<double> &dv) {
    const int HD = D_HD, G = D_Q / D_KV; const double sc = 1.0 / sqrt((double)HD);
    const size_t QD = (size_t)D_Q * HD, KVD = (size_t)D_KV * HD, TRI = N * (N + 1) / 2;
    dq.assign(N * QD, 0.0); dk.assign(N * KVD, 0.0); dv.assign(N * KVD, 0.0);
    std::vector<double> P((size_t)D_Q * TRI), dS((size_t)D_Q * TRI);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int hq = 0; hq < D_Q; hq++) for (size_t t = 0; t < N; t++) {
        int g = hq / G; size_t T = t + 1; double *p = P.data() + hq * TRI + t * (t + 1) / 2, *ds = dS.data() + hq * TRI + t * (t + 1) / 2;
        const float *qt = q.data() + t * QD + hq * HD; const double *dot = dO.data() + t * QD + hq * HD; double mx = -1e300;
        for (size_t u = 0; u < T; u++) { const float *ku = k.data() + u * KVD + g * HD; double a = 0; for (int d = 0; d < HD; d++) a += (double)qt[d] * ku[d];
            p[u] = a * sc; mx = std::max(mx, p[u]); }
        double den = 0; for (size_t u = 0; u < T; u++) { p[u] = exp(p[u] - mx); den += p[u]; }
        double sdp = 0;
        for (size_t u = 0; u < T; u++) { p[u] /= den; const float *vu = v.data() + u * KVD + g * HD; double a = 0; for (int d = 0; d < HD; d++) a += dot[d] * vu[d];
            ds[u] = a; sdp += p[u] * a; }
        double *dqt = dq.data() + t * QD + hq * HD;
        for (size_t u = 0; u < T; u++) { ds[u] = p[u] * (ds[u] - sdp); const float *ku = k.data() + u * KVD + g * HD; for (int d = 0; d < HD; d++) dqt[d] += ds[u] * sc * ku[d]; }
    }
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int g = 0; g < D_KV; g++) for (size_t u = 0; u < N; u++) {
        double *dku = dk.data() + u * KVD + g * HD, *dvu = dv.data() + u * KVD + g * HD;
        for (int hq = g * G; hq < (g + 1) * G; hq++) for (size_t t = u; t < N; t++) {
            double p = P[hq * TRI + t * (t + 1) / 2 + u], ds = dS[hq * TRI + t * (t + 1) / 2 + u];
            const float *qt = q.data() + t * QD + hq * HD; const double *dot = dO.data() + t * QD + hq * HD;
            for (int d = 0; d < HD; d++) { dku[d] += ds * sc * qt[d]; dvu[d] += p * dot[d]; } }
    }
}

// ---- LoRA -----------------------------------------------------------------------------------------------------------------
static int RK = 16; static double SC = 2.0;   // rank, alpha / rank (a power of two: the scaling is exact)
static size_t SEQ;   // tokens per sequence: row r is position r % SEQ of sequence r / SEQ
// A and B in FP32 (the master copy), used as BF16 in every product; m, v: Adam's moments of Bq, Bv, Aq, Av
struct Lora { std::vector<float> Aq, Av, dBq, dBv, Bq, Bv, dAq, dAv; std::vector<double> m[4], v[4]; };
static std::vector<Lora> LO; static void *dLA[2];
static std::vector<float> lora_h(const std::vector<float> &A, const std::vector<float> &x) { return bmm(up(A, RK, D_H), x, 8); }
// dB = s G^T H: the tokens are the product's K (zeros up to a multiple of 128)
static std::vector<float> lora_db(const std::vector<double> &G, int M, const std::vector<float> &H, size_t N) {
    size_t NP = (N + 127) / 128 * 128; std::vector<float> Ht((size_t)RK * NP, 0.f), Gt((size_t)M * NP, 0.f), Y((size_t)M * RK);
    for (size_t n = 0; n < N; n++) for (int j = 0; j < RK; j++) Ht[j * NP + n] = H[n * RK + j];
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) for (size_t n = 0; n < N; n++) Gt[m * NP + n] = bf(G[n * M + m]);
    bmm(up(Ht, RK, NP), Gt.data(), M, 24, Y.data());
    for (auto &y : Y) y = (float)(y * SC);
    return Y;
}
// the adapted map Y = BF16(x W^T + s H B^T) as ONE product, [x | s H | 0] [W | B | 0]^T (s H is exact: s a power of two);
// B = 0: the base map alone, bit for bit the model
static std::vector<float> lora_proj(int L, int which, const DevW &W, const std::vector<float> &x) {
    const auto &B = which ? LO[L].Bv : LO[L].Bq;
    if (B.empty()) return bmm(W, x);
    size_t N = x.size() / D_H; const int M = W.M, KP = D_H + 128;
    auto H = lora_h(which ? LO[L].Av : LO[L].Aq, x);
    CK(cudaMemset(dWt, 0, (size_t)M * KP * 2));
    CK(cudaMemcpy2D(dWt, KP * 2, W.w, D_H * 2, D_H * 2, M, cudaMemcpyDeviceToDevice));
    std::vector<uint16_t> b((size_t)M * 128, 0);
    for (int m = 0; m < M; m++) for (int j = 0; j < RK; j++) { float f = bf(B[m * RK + j]); uint32_t u; memcpy(&u, &f, 4); b[m * 128 + j] = u >> 16; }
    CK(cudaMemcpy2D((uint16_t *)dWt + D_H, KP * 2, b.data(), 256, 256, M, cudaMemcpyHostToDevice));
    std::vector<float> X(N * KP, 0.f), Y(N * M);
    #pragma omp parallel for schedule(static)
    for (size_t n = 0; n < N; n++) { std::copy(x.begin() + n * D_H, x.begin() + (n + 1) * D_H, X.begin() + n * KP);
        for (int j = 0; j < RK; j++) X[n * KP + D_H + j] = (float)(SC * H[n * RK + j]); }
    bmm(devw(dWt, M, KP), X.data(), N, 8, Y.data());
    return Y;
}
// U = s G B [N][r] (FP32 outputs of one product, times s), the adapter's gradient path
static std::vector<double> lora_u(const std::vector<double> &G, int M, const std::vector<float> &B, size_t N) {
    std::vector<float> Bt((size_t)RK * M), Gb(G.size()), Y(N * RK);
    for (int m = 0; m < M; m++) for (int j = 0; j < RK; j++) Bt[j * M + m] = B[m * RK + j];
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < G.size(); i++) Gb[i] = bf(G[i]);
    bmm(up(Bt, RK, M), Gb.data(), N, 24, Y.data());
    std::vector<double> U(Y.size()); for (size_t i = 0; i < Y.size(); i++) U[i] = SC * Y[i];
    return U;
}
// dA = U^T x [r][H], U already scaled: one product over the tokens, U in BF16 as every gradient that enters a product
static std::vector<float> lora_da(const std::vector<double> &U, const std::vector<float> &x, size_t N) {
    size_t NP = (N + 127) / 128 * 128; std::vector<float> Ut((size_t)RK * NP, 0.f), Xt((size_t)D_H * NP, 0.f), Y((size_t)D_H * RK), dA((size_t)RK * D_H);
    for (size_t n = 0; n < N; n++) for (int j = 0; j < RK; j++) Ut[j * NP + n] = bf(U[n * RK + j]);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < D_H; k++) for (size_t n = 0; n < N; n++) Xt[k * NP + n] = x[n * D_H + k];
    bmm(up(Ut, RK, NP), Xt.data(), D_H, 24, Y.data());
    for (int k = 0; k < D_H; k++) for (int j = 0; j < RK; j++) dA[j * D_H + k] = Y[k * RK + j];
    return dA;
}
// AdamW (b1 0.9, b2 0.999, eps 1e-8, no decay) on p from its gradient; returns sum g dp, the update's first-order change
static double adamw(std::vector<float> &p, const std::vector<float> &g, std::vector<double> &m, std::vector<double> &v, int t, double lr) {
    if (p.empty()) p.assign(g.size(), 0.f);
    m.resize(g.size(), 0.0); v.resize(g.size(), 0.0);
    const double b1 = 0.9, b2 = 0.999, c1 = 1 - pow(b1, t), c2 = 1 - pow(b2, t); double ch = 0;
    for (size_t i = 0; i < g.size(); i++) { m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * (double)g[i] * g[i];
        float np = (float)(p[i] - lr * (m[i] / c1) / (sqrt(v[i] / c2) + 1e-8)); ch += (double)g[i] * ((double)bf(np) - bf(p[i])); p[i] = np; }
    return ch;
}

// ---- the backward of one layer --------------------------------------------------------------------------------------------
static void *dAt[4], *dMl[3], *dEw[3], *dEs[3], *dSh[3], *dRt;
static double t_back_att, t_back_mlp;
static DevW fp8w_to(const std::string &name, int M, int K, int i) {
    DevW W = load_w(name + ".weight", F_E4M3B, M, K, dEw[i]); to_dev(name + ".weight_scale", dEs[i]); W.sw = (uint8_t *)dEs[i]; return W;
}
static double secs(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }
static void moe_back(const std::string &P, int L, const std::vector<float> &x2, const std::vector<double> &g, size_t N, std::vector<double> &dx2) {
    DevW Wr = load_w(P + "mlp.gate.weight", F_BF16, J_E, D_H, dRt); auto lg = bmm(Wr, x2);
    std::vector<std::vector<int>> ex(N); std::vector<std::vector<double>> wt(N);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < N; r++) route(std::vector<float>(lg.begin() + r * J_E, lg.begin() + (r + 1) * J_E), BIAS[L], ex[r], wt[r]);
    std::vector<std::vector<std::pair<size_t, int>>> by(J_E);   // expert -> (row, slot)
    for (size_t r = 0; r < N; r++) for (int i = 0; i < J_TOPK; i++) by[ex[r][i]].push_back({r, i});
    std::vector<double> dw(N * J_TOPK, 0.0);   // dL/d(routing weight)
    for (int e = 0; e < J_E; e++) {   // ascending id: each row's sum of its experts' dx in one order
        if (by[e].empty()) continue;
        std::string EP = P + "mlp.experts." + std::to_string(e) + ".";
        size_t n = by[e].size(); std::vector<float> xe(n * D_H);
        for (size_t i = 0; i < n; i++) std::copy(x2.begin() + by[e][i].first * D_H, x2.begin() + (by[e][i].first + 1) * D_H, xe.begin() + i * D_H);
        DevW Wg = fp8w_to(EP + "gate_proj", J_MFF, D_H, 0), Wu = fp8w_to(EP + "up_proj", J_MFF, D_H, 1), Wd = fp8w_to(EP + "down_proj", D_H, J_MFF, 2);
        auto gg = bmm(Wg, xe), uu = bmm(Wu, xe), y = bmm(Wd, swiglu_rows(gg, uu));
        std::vector<double> dy(n * D_H);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) { size_t r = by[e][i].first; int s = by[e][i].second; double w = wt[r][s], d = 0;
            const double *gr = g.data() + r * D_H; const float *yi = y.data() + i * D_H;
            for (int c = 0; c < D_H; c++) { dy[i * D_H + c] = w * gr[c]; d += gr[c] * yi[c]; }
            dw[r * J_TOPK + s] = d; }
        std::vector<double> da(n * (size_t)J_MFF, 0.0), dg, du, dxe(n * D_H, 0.0);
        dgrad({Wd}, {&dy}, n, da); swiglu_back(gg, uu, da, dg, du); dgrad({Wg, Wu}, {&dg, &du}, n, dxe);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) { double *d = dx2.data() + by[e][i].first * D_H; const double *s = dxe.data() + i * D_H; for (int c = 0; c < D_H; c++) d[c] += s[c]; }
    }
    DevW Sg = load_w(P + "mlp.shared_experts.gate_proj.weight", F_BF16, J_MFF, D_H, dSh[0]);
    DevW Su = load_w(P + "mlp.shared_experts.up_proj.weight", F_BF16, J_MFF, D_H, dSh[1]);
    DevW Sd = load_w(P + "mlp.shared_experts.down_proj.weight", F_BF16, D_H, J_MFF, dSh[2]);
    auto gg = bmm(Sg, x2), uu = bmm(Su, x2);
    std::vector<double> da(N * (size_t)J_MFF, 0.0), dg, du;
    dgrad({Sd}, {&g}, N, da); swiglu_back(gg, uu, da, dg, du); dgrad({Sg, Su}, {&dg, &du}, N, dx2);
    std::vector<double> dl(N * (size_t)J_E, 0.0);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < N; r++) gate_back(lg.data() + r * J_E, ex[r].data(), dw.data() + r * J_TOPK, dl.data() + r * J_E);
    dgrad({Wr}, {&dl}, N, dx2);
}
static void back_layer(int L, const std::vector<float> &h, size_t N, std::vector<double> &g, Lora &lo) {
    auto t0 = std::chrono::steady_clock::now();
    std::string P = "model.layers." + std::to_string(L) + ".";
    const size_t QD = (size_t)D_Q * D_HD, KVD = (size_t)D_KV * D_HD;
    // the attention half again: the forward's functions on the forward's values
    auto x1 = norm_rows(h, N1[L]);
    DevW Wq = load_w(P + "self_attn.q_proj.weight", F_BF16, QD, D_H, dAt[0]), Wk = load_w(P + "self_attn.k_proj.weight", F_BF16, KVD, D_H, dAt[1]);
    DevW Wv = load_w(P + "self_attn.v_proj.weight", F_BF16, KVD, D_H, dAt[2]), Wo = load_w(P + "self_attn.o_proj.weight", F_BF16, D_H, QD, dAt[3]);
    auto q = lora_proj(L, 0, Wq, x1); rows_map(q, QD, [&](size_t r, std::vector<float> &v) { rope_part(v, D_Q, r % SEQ); });
    auto k = bmm(Wk, x1); rows_map(k, KVD, [&](size_t r, std::vector<float> &v) { rope_part(v, D_KV, r % SEQ); });
    auto v = lora_proj(L, 1, Wv, x1);
    std::vector<float> o(N * QD);
    #pragma omp parallel for schedule(dynamic)
    for (size_t r = 0; r < N; r++) attend_p(q.data() + r * QD, k.data() + r / SEQ * SEQ * KVD, v.data() + r / SEQ * SEQ * KVD, r % SEQ + 1, D_Q, D_KV, D_HD, o.data() + r * QD, false);
    std::vector<float> hp = h; add_rows(hp, bmm(Wo, o));
    auto x2 = norm_rows(hp, N2[L]);
    // the MLP half and its backward
    auto t1 = std::chrono::steady_clock::now();
    std::vector<double> dx2(N * D_H, 0.0);
    if (!MOE[L]) {
        DevW Wg = load_w(P + "mlp.gate_proj.weight", F_BF16, D_FF, D_H, dMl[0]), Wu = load_w(P + "mlp.up_proj.weight", F_BF16, D_FF, D_H, dMl[1]);
        DevW Wd = load_w(P + "mlp.down_proj.weight", F_BF16, D_H, D_FF, dMl[2]);
        auto gg = bmm(Wg, x2), uu = bmm(Wu, x2);
        std::vector<double> da(N * (size_t)D_FF, 0.0), dg, du;
        dgrad({Wd}, {&g}, N, da); swiglu_back(gg, uu, da, dg, du);
        dgrad({Wg}, {&dg}, N, dx2); dgrad({Wu}, {&du}, N, dx2);   // together K = 2 x 16384, beyond the product's buffers
    } else moe_back(P, L, x2, g, N, dx2);
    std::vector<double> gh = g; norm_back(hp, N2[L], dx2, gh);   // dL/dh' = g + the MLP branch
    double tmlp = secs(t1); t_back_mlp += tmlp;
    // the attention backward
    std::vector<double> dO(N * QD, 0.0), dq, dk, dv; dgrad({Wo}, {&gh}, N, dO);
    dq.assign(N * QD, 0.0); dk.assign(N * KVD, 0.0); dv.assign(N * KVD, 0.0);
    for (size_t j = 0; j < N / SEQ; j++) {   // each sequence attends only to itself
        auto cut = [&](auto &a, size_t w) { return std::vector<typename std::decay_t<decltype(a)>::value_type>(a.begin() + j * SEQ * w, a.begin() + (j + 1) * SEQ * w); };
        std::vector<double> sq, sk, sv; attend_back(cut(q, QD), cut(k, KVD), cut(v, KVD), cut(dO, QD), SEQ, sq, sk, sv);
        std::copy(sq.begin(), sq.end(), dq.begin() + j * SEQ * QD); std::copy(sk.begin(), sk.end(), dk.begin() + j * SEQ * KVD); std::copy(sv.begin(), sv.end(), dv.begin() + j * SEQ * KVD);
    }
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < N; r++) { rope_back(dq.data() + r * QD, D_Q, r % SEQ); rope_back(dk.data() + r * KVD, D_KV, r % SEQ); }
    lo.dBq = lora_db(dq, QD, lora_h(lo.Aq, x1), N); lo.dBv = lora_db(dv, KVD, lora_h(lo.Av, x1), N);
    std::vector<double> dx1(N * D_H, 0.0);
    if (lo.Bq.empty()) dgrad({Wq, Wk, Wv}, {&dq, &dk, &dv}, N, dx1);   // B = 0: G B = 0, no dA and no adapter term in dx1
    else {   // U = s G B; dA = U^T x; the adapters' share of dx1, U A, in the same product as q, k and v
        auto Uq = lora_u(dq, QD, lo.Bq, N), Uv = lora_u(dv, KVD, lo.Bv, N);
        lo.dAq = lora_da(Uq, x1, N); lo.dAv = lora_da(Uv, x1, N);
        DevW Aq = up(lo.Aq, RK, D_H, dLA[0]), Av = up(lo.Av, RK, D_H, dLA[1]);
        dgrad({Wq, Wk, Wv, Aq, Av}, {&dq, &dk, &dv, &Uq, &Uv}, N, dx1);
    }
    norm_back(h, N1[L], dx1, gh);
    g.swap(gh);
    t_back_att += secs(t0) - tmlp;
}

// ---- the loss and the backward of lm_head ---------------------------------------------------------------------------------
// tgt[r]: the token row r predicts, -1 for none (the last position of each sequence)
static double loss_back(const std::vector<float> &hN, const std::vector<int> &tgt, size_t N, std::vector<double> *g) {
    auto hn = norm_rows(hN, NORMF); size_t T = 0; for (int y : tgt) T += y >= 0;
    std::vector<float> lg(N * (size_t)D_V);
    for (int v0 = 0; v0 < D_V; v0 += 16384) {
        DevW Ls = LM; Ls.M = std::min(16384, D_V - v0); Ls.w = (const uint16_t *)LM.w + (size_t)v0 * D_H;
        std::vector<float> y(N * Ls.M); bmm(Ls, hn.data(), N, 24, y.data());
        for (size_t r = 0; r < N; r++) std::copy(y.begin() + r * Ls.M, y.begin() + (r + 1) * Ls.M, lg.begin() + r * D_V + v0);
    }
    std::vector<double> lr(N, 0.0), mx(N), den(N);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < N; r++) { if (tgt[r] < 0) continue; const float *l = lg.data() + r * D_V; double m = -1e300, s = 0;
        for (int v = 0; v < D_V; v++) m = std::max(m, (double)l[v]);
        for (int v = 0; v < D_V; v++) s += exp(l[v] - m);
        mx[r] = m; den[r] = s; lr[r] = m + log(s) - l[tgt[r]]; }
    double loss = 0; for (size_t r = 0; r < N; r++) loss += lr[r]; loss /= T;
    if (!g) return loss;
    std::vector<double> dhn(N * D_H, 0.0);
    for (int v0 = 0; v0 < D_V; v0 += 16384) {   // dL/dlogit = (softmax - onehot) / T, by slices
        DevW Ls = LM; Ls.M = std::min(16384, D_V - v0); Ls.w = (const uint16_t *)LM.w + (size_t)v0 * D_H;
        std::vector<double> d(N * (size_t)Ls.M, 0.0);
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < N; r++) { if (tgt[r] < 0) continue; const float *l = lg.data() + r * D_V + v0; double *dr = d.data() + r * Ls.M;
            for (int v = 0; v < Ls.M; v++) dr[v] = exp(l[v] - mx[r]) / den[r] / T;
            int y = tgt[r] - v0; if (y >= 0 && y < Ls.M) dr[y] -= 1.0 / T; }
        dgrad({Ls}, {&d}, N, dhn);
    }
    norm_back(hN, NORMF, dhn, *g);
    return loss;
}

static double norm2(const std::vector<double> &v) { double s = 0; for (double x : v) s += x * x; return sqrt(s); }
static double norm2f(const std::vector<float> &v) { double s = 0; for (float x : v) s += (double)x * x; return sqrt(s); }
static uint64_t fnvd(uint64_t h, const std::vector<double> &v) {
    for (double x : v) { uint64_t u; memcpy(&u, &x, 8); for (int i = 0; i < 8; i++) { h ^= (u >> (8 * i)) & 255; h *= 1099511628211ULL; } } return h;
}

// ---- --selftest <model dir>: every hand-written backward against central differences of its own FP64 forward --------------
// (no rounding anywhere; inputs on a 2^-12 grid so x +- 2^-12 is exact in float)
static double rnd(uint64_t &s) { s = sm64(s); return (double)(s >> 11) * 0x1p-53 * 2 - 1; }
static int back_selftest() {
    uint64_t sd = 7; const double e = 0x1p-12; double all = 0;
    auto grid = [&](std::vector<float> &v) { for (auto &x : v) x = (float)(std::round(rnd(sd) * 4096) / 4096); };
    auto rel = [](double an, double fd) { return fabs(an - fd) / std::max(fabs(fd), 1e-9); };
    auto probe = [&](const char *name, std::vector<std::vector<float> *> X, std::vector<std::vector<double> *> D, std::function<double()> L) {
        double w = 0;
        for (int i = 0; i < 16; i++) for (size_t a = 0; a < X.size(); a++) {
            auto &x = *X[a]; size_t j = sm64(sd++) % x.size(); float x0 = x[j];
            x[j] = x0 + (float)e; double lp = L(); x[j] = x0 - (float)e; double lm = L(); x[j] = x0;
            w = std::max(w, rel((*D[a])[j], (lp - lm) / (2 * e))); }
        printf("selftest %-10s worst relative difference %.2e\n", name, w); all = std::max(all, w);
    };
    const size_t QD = (size_t)D_Q * D_HD, KVD = (size_t)D_KV * D_HD;
    {   // attention: L = sum o . R, o_t = sum_u softmax(q_t . k_u / sqrt(hd)) v_u
        size_t N = 6; std::vector<float> q(N * QD), k(N * KVD), v(N * KVD); std::vector<double> R(N * QD), dq, dk, dv;
        grid(q); grid(k); grid(v); for (auto &x : R) x = rnd(sd);
        attend_back(q, k, v, R, N, dq, dk, dv);
        probe("attention", {&q, &k, &v}, {&dq, &dk, &dv}, [&]() { double s = 0; const int G = D_Q / D_KV; const double sc = 1.0 / sqrt((double)D_HD);
            for (size_t t = 0; t < N; t++) for (int hq = 0; hq < D_Q; hq++) { int g = hq / G; std::vector<double> p(t + 1); double mx = -1e300, den = 0;
                for (size_t u = 0; u <= t; u++) { double a = 0; for (int d = 0; d < D_HD; d++) a += (double)q[t * QD + hq * D_HD + d] * k[u * KVD + g * D_HD + d];
                    p[u] = a * sc; mx = std::max(mx, p[u]); }
                for (auto &x : p) { x = exp(x - mx); den += x; }
                for (int d = 0; d < D_HD; d++) { double o = 0; for (size_t u = 0; u <= t; u++) o += p[u] / den * v[u * KVD + g * D_HD + d]; s += o * R[t * QD + hq * D_HD + d]; } }
            return s; });
    }
    {   // RMSNorm: L = sum (w x r) . R
        std::vector<float> x(3 * D_H), w(D_H); std::vector<double> R(3 * D_H), dx(3 * D_H, 0.0); grid(x); grid(w); for (auto &y : R) y = rnd(sd);
        norm_back(x, w, R, dx);
        probe("rmsnorm", {&x}, {&dx}, [&]() { double s = 0; for (int r = 0; r < 3; r++) { double ss = 0; for (int i = 0; i < D_H; i++) ss += (double)x[r * D_H + i] * x[r * D_H + i];
            double rr = 1.0 / sqrt(ss / D_H + D_EPS); for (int i = 0; i < D_H; i++) s += (double)w[i] * x[r * D_H + i] * rr * R[r * D_H + i]; } return s; });
    }
    {   // SwiGLU: L = sum silu(g) u R
        std::vector<float> g(4096), u(4096); std::vector<double> R(4096), dg, du; grid(g); grid(u); for (auto &y : R) y = rnd(sd);
        for (auto &y : g) y *= 8;   // the sigmoid's whole range
        swiglu_back(g, u, R, dg, du);
        probe("swiglu", {&g, &u}, {&dg, &du}, [&]() { double s = 0; for (size_t i = 0; i < g.size(); i++) s += (double)g[i] / (1.0 + exp(-(double)g[i])) * u[i] * R[i]; return s; });
    }
    {   // RoPE at position 777: L = sum rope(x) . R
        const int pos = 777; std::vector<float> x(QD); std::vector<double> R(QD); grid(x); for (auto &y : R) y = rnd(sd);
        std::vector<double> dx = R; rope_back(dx.data(), D_Q, pos);
        probe("rope", {&x}, {&dx}, [&]() { std::vector<double> y(x.begin(), x.end());
            for (int h = 0; h < D_Q; h++) for (int i = 0; i < J_RHD / 2; i++) { double a = (double)pos * inv_freq[i], c = cos(a), s = sin(a), x1 = y[h * D_HD + i], x2 = y[h * D_HD + i + D_HD / 2];
                y[h * D_HD + i] = x1 * c - x2 * s; y[h * D_HD + i + D_HD / 2] = x2 * c + x1 * s; }
            double s = 0; for (size_t i = 0; i < QD; i++) s += y[i] * R[i]; return s; });
    }
    {   // routing weights: L = sum_k (c s_k / S) R_k over 8 chosen experts, s the sigmoids of the logits
        std::vector<float> lg(J_E); grid(lg); for (auto &y : lg) y *= 4;
        std::vector<int> ex; for (int k = 0; (int)ex.size() < J_TOPK; k++) { int c = sm64(sd++) % J_E; if (std::find(ex.begin(), ex.end(), c) == ex.end()) ex.push_back(c); }
        std::vector<double> R(J_TOPK), dl(J_E, 0.0); for (auto &y : R) y = rnd(sd);
        gate_back(lg.data(), ex.data(), R.data(), dl.data());
        probe("routing", {&lg}, {&dl}, [&]() { double S = 0, s = 0; for (int k = 0; k < J_TOPK; k++) S += 1.0 / (1.0 + exp(-(double)lg[ex[k]]));
            for (int k = 0; k < J_TOPK; k++) s += J_RSF / (1.0 + exp(-(double)lg[ex[k]])) / S * R[k]; return s; });
    }
    printf("selftest: worst %.2e %s\n", all, all < 1e-6 ? "ok" : "FAILED");
    return all < 1e-6 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc > 2 && !strcmp(argv[1], "--selftest")) { DIR = argv[2]; MOE = jais_init(); return back_selftest(); }
    if (argc < 3) { fprintf(stderr, "usage: out_jlora <model dir> <token ids> [tokens N] [rank r] [alpha] [dB dir] | --selftest <model dir>\n"); return 1; }
    DIR = argv[1];
    // the data: a file of token ids (one sequence), or a .u32 file of sequences cut to OZ_CUT (1024) tokens: OZ_BATCH of them
    // from number OZ_FIRST, the first N tokens of each
    std::vector<std::vector<int>> seqs; size_t N = argc > 3 ? atol(argv[3]) : 512; std::string inp = argv[2];
    if (inp.size() > 4 && inp.substr(inp.size() - 4) == ".u32") {
        int B = getenv("OZ_BATCH") ? atoi(getenv("OZ_BATCH")) : 1; size_t first = getenv("OZ_FIRST") ? atol(getenv("OZ_FIRST")) : 0;
        size_t CUT = getenv("OZ_CUT") ? atol(getenv("OZ_CUT")) : 1024; std::vector<uint32_t> b(CUT); N = std::min(N, CUT);
        FILE *f = fopen(inp.c_str(), "rb"); if (!f) { perror(inp.c_str()); return 1; }
        for (int j = 0; j < B; j++) { fseeko(f, (off_t)((first + j) * CUT * 4), SEEK_SET);
            if (fread(b.data(), 4, CUT, f) != CUT) { fprintf(stderr, "%s holds no sequence %zu\n", inp.c_str(), first + j); return 1; }
            seqs.emplace_back(b.begin(), b.begin() + N); }
        fclose(f);
    } else { std::vector<int> toks; std::ifstream f(inp); int v; while (f >> v) toks.push_back(v); N = std::min(toks.size(), N); toks.resize(N); seqs.push_back(toks); }
    SEQ = N; const size_t R = seqs.size() * N;   // rows: sequence after sequence
    std::vector<int> tgt(R, -1); for (size_t j = 0; j < seqs.size(); j++) for (size_t t = 0; t + 1 < N; t++) tgt[j * N + t] = seqs[j][t + 1];
    RK = argc > 4 ? atoi(argv[4]) : 16; double alpha = argc > 5 ? atof(argv[5]) : 32; SC = alpha / RK;
    std::string out = argc > 6 ? argv[6] : "";
    EPS = getenv("OZ_EPS") ? atof(getenv("OZ_EPS")) : 0; if (getenv("OZ_DSEED")) DSEED = strtoull(getenv("OZ_DSEED"), 0, 0); std::vector<double> FDEPS;
    if (const char *f = getenv("OZ_FDEPS")) { std::stringstream ss(f); std::string t; while (std::getline(ss, t, ',')) FDEPS.push_back(atof(t.c_str())); }
    MOE = jais_init(); NL = MOE.size();
    if (getenv("OZ_NL")) { NL = std::min(NL, atoi(getenv("OZ_NL"))); MOE.resize(NL); }
    jb_init();
    WT_MAX = (size_t)std::max(D_FF, 16384) * D_H * 2; SM_MAX = (size_t)64 << 20;
    CK(cudaMalloc(&dWt, WT_MAX)); CK(cudaMalloc(&dSm, SM_MAX)); CK(cudaMalloc(&dtbad, 4)); CK(cudaMemset(dtbad, 0, 4));
    const size_t QD = (size_t)D_Q * D_HD, KVD = (size_t)D_KV * D_HD;
    CK(cudaMalloc(&dAt[0], QD * D_H * 2)); CK(cudaMalloc(&dAt[1], KVD * D_H * 2)); CK(cudaMalloc(&dAt[2], KVD * D_H * 2)); CK(cudaMalloc(&dAt[3], QD * D_H * 2));
    for (int i = 0; i < 3; i++) { CK(cudaMalloc(&dMl[i], (size_t)D_FF * D_H * 2)); CK(cudaMalloc(&dEw[i], (size_t)J_MFF * D_H)); CK(cudaMalloc(&dEs[i], (size_t)1 << 20));
        CK(cudaMalloc(&dSh[i], (size_t)J_MFF * D_H * 2)); }
    CK(cudaMalloc(&dRt, (size_t)J_E * D_H * 2));
    LO.assign(NL, Lora()); auto &lo = LO;
    for (int i = 0; i < 2; i++) CK(cudaMalloc(&dLA[i], (size_t)RK * D_H * 2));
    const double bound = 1.0 / sqrt((double)D_H);
    for (int L = 0; L < NL; L++) for (int w = 0; w < 2; w++) {
        auto &A = w ? lo[L].Av : lo[L].Aq; A.resize((size_t)RK * D_H);
        for (size_t i = 0; i < A.size(); i++) A[i] = bf(bound * ((double)(sm64(((uint64_t)L << 40) ^ ((uint64_t)w << 32) ^ i) >> 11) * 0x1p-53 * 2 - 1));
    }
    fprintf(stderr, "jais_lora: %d layers (%d MoE), %zu tokens, rank %d, alpha %g%s\n", NL, (int)std::count(MOE.begin(), MOE.end(), true), N, RK, alpha,
            EPS ? (", forward only, embeddings moved by " + std::to_string(EPS)).c_str() : "");
    // OZ_STEPS steps of AdamW (OZ_LR) on the one sequence: step 1 at B = 0, then the adapters in the forward too
    const int STEPS = getenv("OZ_STEPS") ? atoi(getenv("OZ_STEPS")) : 1; const double LR = getenv("OZ_LR") ? atof(getenv("OZ_LR")) : 1e-4;
    PROJ = lora_proj; double prev = 0, predicted = 0;
    for (int st = 1; st <= STEPS; st++) {
        auto t0 = std::chrono::steady_clock::now(); t_back_mlp = t_back_att = 0;
        std::vector<BSeq> S(seqs.size()); std::vector<std::pair<int, size_t>> rows;
        for (size_t j = 0; j < seqs.size(); j++) { auto &s = S[j]; s.prompt = j; s.toks = seqs[j]; s.plen = N; s.end = N; s.kc.resize(NL); s.vc.resize(NL);
            for (size_t t = 0; t < N; t++) rows.push_back({(int)j, t}); }
        std::vector<std::vector<float>> HS; HSAVE = &HS; if (EPS) HEMB = perturb;
        pass(S, rows, std::vector<bool>(R, false)); HSAVE = nullptr; S.clear();
        std::vector<double> g(R * D_H, 0.0);
        double loss = loss_back(HS[NL], tgt, R, EPS ? nullptr : &g);
        double tf = secs(t0);
        printf("step %d loss %.17g (perplexity %.6g) over %zu predictions; forward %.1f s\n", st, loss, exp(loss), seqs.size() * (N - 1), tf);
        if (st > 1) printf("step %d: the update changed the loss by %.9e, its first-order prediction %.9e\n", st, loss - prev, predicted);
        if (EPS) return 0;
        printf("layer %d |g| %.6e\n", NL, norm2(g));
        double d0 = t_disk, b0 = bytes_disk;
        for (int L = NL - 1; L >= 0; L--) {
            auto tl = std::chrono::steady_clock::now();
            back_layer(L, HS[L], R, g, lo[L]);
            printf("layer %d |g| %.6e |dBq| %.6e |dBv| %.6e |dAq| %.6e |dAv| %.6e  %.1f s\n", L, norm2(g), norm2f(lo[L].dBq), norm2f(lo[L].dBv),
                   norm2f(lo[L].dAq), norm2f(lo[L].dAv), secs(tl)); fflush(stdout);
        }
        int tb; CK(cudaMemcpy(&tb, dtbad, 4, cudaMemcpyDeviceToHost));
        uint64_t dig = 1469598103934665603ULL, dia = dig;
        for (int L = 0; L < NL; L++) { dig = fnv(dig, lo[L].dBq); dig = fnv(dig, lo[L].dBv); dia = fnv(dia, lo[L].dAq); dia = fnv(dia, lo[L].dAv); }
        printf("step %d dB digest %016llx, dA digest %016llx, g_0 digest %016llx; hi/lo splits not exact %d; uncertified %lld\n", st,
               (unsigned long long)dig, (unsigned long long)dia, (unsigned long long)fnvd(1469598103934665603ULL, g), tb, st_bad);
        if (st == 1) {
            std::vector<uint64_t> SEEDS{DSEED};
            if (const char *f = getenv("OZ_DSEEDS")) { SEEDS.clear(); std::stringstream ss(f); std::string t; while (std::getline(ss, t, ',')) SEEDS.push_back(strtoull(t.c_str(), 0, 0)); }
            for (uint64_t sd : SEEDS) for (double e : FDEPS) {   // the predicted L(+e) - L(-e) for the moves the OZ_EPS runs make
                double pr = 0; const auto &h0 = HS[0];
                for (size_t i = 0; i < h0.size(); i++) { double d = (sm64(i ^ sd) & 1) ? 1.0 : -1.0;
                    pr += g[i] * ((double)bf((double)h0[i] + e * d) - (double)bf((double)h0[i] - e * d)); }
                printf("predicted seed %llu L(+%g) - L(-%g) = %.9e\n", (unsigned long long)sd, e, e, pr);
            }
            if (!out.empty()) for (int L = 0; L < NL; L++) for (int w = 0; w < 2; w++) {
                std::string fn = out + "/dB" + (w ? "v_" : "q_") + std::to_string(L) + ".f32"; FILE *f = fopen(fn.c_str(), "wb");
                auto &d = w ? lo[L].dBv : lo[L].dBq; if (f) { fwrite(d.data(), 4, d.size(), f); fclose(f); } }
        }
        printf("step %d time %.1f s: forward %.1f, backward %.1f (MLP/MoE %.1f, attention and LoRA %.1f), backward disk %.1f s for %.1f GB\n", st, secs(t0), tf,
               secs(t0) - tf, t_back_mlp, t_back_att, t_disk - d0, (bytes_disk - b0) / 1e9);
        fflush(stdout);
        if (st == STEPS) break;
        predicted = 0;
        for (int L = 0; L < NL; L++) { auto &l = lo[L];
            predicted += adamw(l.Bq, l.dBq, l.m[0], l.v[0], st, LR) + adamw(l.Bv, l.dBv, l.m[1], l.v[1], st, LR);
            if (!l.dAq.empty()) predicted += adamw(l.Aq, l.dAq, l.m[2], l.v[2], st - 1, LR) + adamw(l.Av, l.dAv, l.m[3], l.v[3], st - 1, LR); }   // A moves from step 2 on
        prev = loss;
    }
    return 0;
}
