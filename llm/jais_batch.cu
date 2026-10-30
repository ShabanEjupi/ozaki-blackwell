// jais_batch.cu -- Jais-3-375B-A23B-FP8 layer by layer over many tokens at once: the model streams from disk once per pass
// (a prefill of every prompt of a chunk, then one pass per generated token), and every linear map of the pass is one batched
// product on the INT8 tensor cores (bgemm.cu), correctly rounded. Correct rounding makes the batching invisible: each
// output is the same float as the per-token GEMV of jais.cu, and the host parts (RMSNorm, RoPE, SwiGLU, routing, FP64
// attention, residual adds) are the same functions on the same values, so the whole run is bit for bit the token-by-token
// one -- --validate proves it on a prompts file: the same step lines and the same transcript digest (every residual stream
// after every layer, every logit) as out_jais mode 0.
//  * A pass takes rows (prompt, position). Per layer: norm -> q/k/v (bgemm BF16) -> RoPE -> the rows' K/V appended to their
//    prompt's cache (FP32 values of BF16 numbers) -> attention over the prefix t + 1 -> o -> residual; dense MLP, or router ->
//    route() -> the chosen experts in ascending id, each read once and applied to all rows that chose it (FP8 bgemm, inputs
//    quantized per group of 128 on the GPU exactly as act_fp8g), weighted sum in FP64 rounded once -> shared expert.
//  * Outputs the tensor cores cannot certify come back as NaN and are recomputed: on the CUDA cores (bf16_fixup), and
//    beyond its list on the host GEMV. Nothing uncertified is ever used.
//  * Chunks: prompts are grouped until OZ_CHUNK tokens (default 24000; the KV cache is 0.5 MB per token); each chunk is a
//    prefill pass plus one pass per new token. OZ_PERPROMPT=1 ends each prompt at its own length + new tokens (a benchmark);
//    by default all run to max length + new tokens, as out_jais does.
//  * OZ_SCORE=id,id,...: for the last position of each prompt also print those logits (multiple choice: the letters).
//  * OZ_GATT=1: attention on the GPU FP64 units (gatt), bit for bit attend_p; OZ_GATT=2 runs both and counts outputs that differ.
// nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jais_batch llm/jais_batch.cu -lgomp
// out_jais_batch <model dir> <prompts file: token ids per line> <new tokens> [--validate]
#define JAIS_LIB
#include "jais.cu"
#define BGEMM_LIB
#include "bgemm.cu"
#include <omp.h>

// OZ_PROF=1: each phase synchronized and timed under its name (prof_dump prints and clears); off, pmark only reads a flag
static const bool PROF = getenv("OZ_PROF"); static std::map<std::string, double> prof;
static void pmark(std::chrono::steady_clock::time_point &t, const char *what) {
    if (!PROF) return;
    CK(cudaDeviceSynchronize()); auto n = std::chrono::steady_clock::now(); prof[what] += std::chrono::duration<double>(n - t).count(); t = n;
}
static void prof_dump(const char *head) {
    if (!PROF) return;
    fprintf(stderr, "prof %s:", head); for (auto &kv : prof) fprintf(stderr, " %s %.2f", kv.first.c_str(), kv.second); fprintf(stderr, "\n"); prof.clear();
}

// ---- the batched product: Y[n][M] = X[n][K] W^T to p bits, every output certified ------------------------------------------
static int8_t *g_wd, *g_xd; static int16_t *g_wl, *g_xl; static uint8_t *g_wn, *g_xn, *g_xq; static uint16_t *g_xb;
static float *g_x, *g_xs, *g_Y; static unsigned *g_list, *g_lcnt; static int *g_nb; static size_t G_NB;
static const unsigned G_CAP = 1u << 22;
static long long st_rows, st_outs, st_outs24, st_fix, st_fix24, st_host, st_fixopen;   // rows, outputs (p 8; p 24), left to the CUDA-core fixup, rows to the host GEMV, fixup outputs it left open
static long long st_fix8;   // FP8 outputs the XF8 tiles left to f8x_fixup
static double t_gemm; static const bool OLDK8 = getenv("OZ_OLDK8");

static void bmm_init(size_t maxMK, size_t maxK, size_t maxM) {
    G_NB = getenv("OZ_NB") ? atol(getenv("OZ_NB")) : 4096;
    CK(cudaMalloc(&g_wd, (size_t)DW24 * maxMK)); CK(cudaMalloc(&g_wl, maxMK / 128 * 2)); CK(cudaMalloc(&g_wn, maxMK / 128));
    CK(cudaMalloc(&g_x, G_NB * maxK * 4)); CK(cudaMalloc(&g_xb, G_NB * maxK * 2)); CK(cudaMalloc(&g_xd, (size_t)DX24 * G_NB * maxK));
    CK(cudaMalloc(&g_xq, G_NB * maxK)); CK(cudaMalloc(&g_xs, G_NB * maxK / 128 * 4)); CK(cudaMalloc(&g_xl, G_NB * maxK / 128 * 2));
    CK(cudaMalloc(&g_xn, G_NB * maxK / 128)); CK(cudaMalloc(&g_Y, G_NB * maxM * 4));
    CK(cudaMalloc(&g_list, (size_t)G_CAP * 4)); CK(cudaMalloc(&g_lcnt, 4)); CK(cudaMalloc(&g_nb, 4));
}
static void bmm(const DevW &W, const float *X, size_t N, int p, float *Y) {
    auto t0 = std::chrono::steady_clock::now(), tp = t0;
    int M = W.M, K = W.K, KB = K / 128; bool fp8 = W.fmt == F_E4M3B;
    auto ph = [&](const char *f8, const char *b8, const char *b24) { pmark(tp, fp8 ? f8 : p == 24 ? b24 : b8); };
    if (fp8) fp8_digits<<<1024, 256>>>((const uint8_t *)W.w, g_wd, (size_t)M * K);
    int dw = W.wd ? W.wd : wdig(p);
    if (!fp8) bf16_digits<<<dig_grid((size_t)M * KB), 256>>>((const uint16_t *)W.w, g_wd, g_wl, g_wn, M, K, dw);
    ph("f8 w-digits", "b8 w-digits", "b24 w-digits");
    std::vector<uint16_t> xb;
    for (size_t n0 = 0; n0 < N; n0 += G_NB) {
        size_t n = std::min(G_NB, N - n0), np = (n + 7) / 8 * 8; const float *x = X + n0 * K;
        size_t left;   // NaNs still in Y after the GPU: the host GEMV's rows (none, as a rule: then no scan)
        if (fp8) {
            CK(cudaMemcpy(g_x, x, n * K * 4, cudaMemcpyHostToDevice));
            if (np > n) CK(cudaMemset(g_x + n * K, 0, (np - n) * K * 4));
            quant_fp8g<<<1024, 256>>>(g_x, g_xq, g_xs, np * KB);
            fp8_digits<<<1024, 256>>>(g_xq, g_xd, np * K); ph("f8 x", "", "");
            CK(cudaMemset(g_nb, 0, 4));
            if (OLDK8) {   // OZ_OLDK8=1: fp8b_gemm, the kernel before tgemm.cu's XF8 (the same outputs)
                fp8b_gemm<<<dim3(M / 128, np / 8), 256>>>(g_wd, (const uint16_t *)W.sw, g_xd, g_xs, g_Y, M, np, K, p, g_nb); ph("f8 gemm", "", "");
                int nb; CK(cudaMemcpy(&nb, g_nb, 4, cudaMemcpyDeviceToHost)); left = nb;   // fp8b_gemm counts every NaN it writes
            } else {
                tgemm_go<3, 3, true, 64, 3, true>(g_wd, (const uint16_t *)W.sw, nullptr, nullptr, g_xd, nullptr, nullptr, g_Y, M, (int)np, K, p, g_nb, g_xs);
                ph("f8 gemm", "", "");
                CK(cudaMemset(g_lcnt, 0, 4)); CK(cudaMemset(g_nb, 0, 4));
                nan_list<<<1024, 256>>>(g_Y, np * M, g_list, g_lcnt, G_CAP);
                f8x_fixup<<<1024, 256>>>((const uint8_t *)W.w, (const uint16_t *)W.sw, g_xq, g_xs, K, g_list, g_lcnt, G_CAP, g_Y, M, p, g_nb);
                unsigned lc; CK(cudaMemcpy(&lc, g_lcnt, 4, cudaMemcpyDeviceToHost)); st_fix8 += lc;
                int nb; CK(cudaMemcpy(&nb, g_nb, 4, cudaMemcpyDeviceToHost)); st_fixopen += nb;
                left = (lc > G_CAP ? lc - G_CAP : 0) + nb; ph("f8 fixup", "", "");   // beyond the list, and what fp8b_gemm would leave open too
            }
        } else {
            xb.assign(np * K, 0);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n * K; i++) { uint32_t u; memcpy(&u, &x[i], 4); xb[i] = u >> 16; }   // BF16 values: exact
            CK(cudaMemcpy(g_xb, xb.data(), np * K * 2, cudaMemcpyHostToDevice)); ph("", "b8 x-host", "b24 x-host");
            bf16_digits<<<dig_grid(np * KB), 256>>>(g_xb, g_xd, g_xl, g_xn, np, K, xdig(p)); ph("", "b8 x-digits", "b24 x-digits");
            bf16_gemm_p(p, dw, g_wd, g_wl, g_wn, g_xd, g_xl, g_xn, g_Y, M, np, K, g_nb); ph("", "b8 gemm", "b24 gemm");
            CK(cudaMemset(g_lcnt, 0, 4)); CK(cudaMemset(g_nb, 0, 4));
            nan_list<<<1024, 256>>>(g_Y, np * M, g_list, g_lcnt, G_CAP);
            bf16_fixup<<<1024, 256>>>((const uint16_t *)W.w, g_xb, g_list, g_lcnt, G_CAP, g_Y, M, K, p, g_nb);
            unsigned lc; CK(cudaMemcpy(&lc, g_lcnt, 4, cudaMemcpyDeviceToHost)); (p == 24 ? st_fix24 : st_fix) += lc;
            int nb; CK(cudaMemcpy(&nb, g_nb, 4, cudaMemcpyDeviceToHost)); st_fixopen += nb;   // the fixup left them NaN: the GEMV below
            left = (lc > G_CAP ? lc - G_CAP : 0) + nb;   // beyond the fixup's list, and what it left open
            ph("", "b8 fixup", "b24 fixup");
        }
        float *y = Y + n0 * M; CK(cudaMemcpy(y, g_Y, n * M * 4, cudaMemcpyDeviceToHost)); ph("f8 y-d2h", "b8 y-d2h", "b24 y-d2h");
        std::vector<char> open(n);   // rows with a NaN left, found in parallel (one thread scanned n x M floats per product)
        if (left) {
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < n; r++) { const float *yr = y + r * M; for (int m = 0; m < M && !open[r]; m++) open[r] = yr[m] != yr[m]; }
        }
        for (size_t r = 0; left && r < n; r++) {   // what is still NaN: the GEMV (beyond the fixup's list; FP8 outputs it could not certify)
            float *yr = y + r * M; if (!open[r]) continue; st_host++;
            std::vector<float> xv(x + r * K, x + (r + 1) * K); auto g = gemv(W, fp8 ? act_fp8g(xv) : act_plain(xv), p, 0);
            for (int m = 0; m < M; m++) if (yr[m] != yr[m]) yr[m] = g[m];
        }
        ph("f8 gemv", "b8 gemv", "b24 gemv");
    }
    st_rows += N; (p == 24 ? st_outs24 : st_outs) += (long long)N * M;
    t_gemm += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
static std::vector<float> bmm(const DevW &W, const std::vector<float> &X, int p = 8) {
    size_t N = X.size() / W.K; std::vector<float> Y(N * W.M); bmm(W, X.data(), N, p, Y.data()); return Y;
}

// ---- prompts and passes ----------------------------------------------------------------------------------------------------
struct BSeq { int prompt, mode = 0; std::vector<int> toks; size_t plen, end; std::vector<std::vector<float>> kc, vc;
              uint64_t dig = 1469598103934665603ULL; };
static int NL; static std::vector<bool> MOE; static std::vector<std::vector<float>> N1, N2, BIAS; static std::vector<float> NORMF;
static DevW LM; static bool VALIDATE; static std::vector<int> SCORE;
// jais_lora.cu: HSAVE collects the residual stream entering every layer and leaving the last; HEMB may change the embeddings
static std::vector<std::vector<float>> *HSAVE; static void (*HEMB)(std::vector<float> &);
// of the first HSAVE_ROWS rows only (rows after them ride along, a held-out batch); EXUSED[L][e]: an expert of layer L that
// one of those rows was routed to (the backward announces the same reads to the read-ahead)
static size_t HSAVE_ROWS = SIZE_MAX; static std::vector<std::vector<char>> *EXUSED;
static std::vector<float> (*PROJ)(int L, int which, const DevW &W, const std::vector<float> &x);   // q (0) and v (1) maps, when set

// the rows' function f(row vector) -> row vector, in parallel over rows: the per-token host code on each row as it is
template <class F> static void rows_map(std::vector<float> &A, int K, F f) {
    size_t R = A.size() / K;
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < R; r++) { std::vector<float> v(A.begin() + r * K, A.begin() + (r + 1) * K); f(r, v); std::copy(v.begin(), v.end(), A.begin() + r * K); }
}
static std::vector<float> norm_rows(const std::vector<float> &h, const std::vector<float> &w) {
    std::vector<float> x = h; rows_map(x, D_H, [&](size_t, std::vector<float> &v) { std::vector<float> y; norm(v, w, y); v = y; }); return x;
}
static std::vector<float> swiglu_rows(const std::vector<float> &g, const std::vector<float> &u) {
    std::vector<float> a(g.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < g.size(); i++) { double gv = g[i]; a[i] = bf(gv / (1.0 + exp(-gv)) * u[i]); }   // swiglu() of jais.cu
    return a;
}
static void add_rows(std::vector<float> &h, const std::vector<float> &y) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < h.size(); i++) h[i] = bf((double)h[i] + y[i]);
}
static DevW fp8w(const std::string &name, int M, int K) {
    DevW W = load_w(name + ".weight", F_E4M3B, M, K); to_dev(name + ".weight_scale", dS); W.sw = (uint8_t *)dS; return W;
}

// ---- attention on the GPU (OZ_GATT=1; OZ_GATT=2 also runs attend_p and compares every output bit for bit) -----------------
// The two long sums of attend_p go to the FP64 units in the host's own order: the scores q.k / sqrt(hd) (a product of two
// floats is exact in double, so the order of the adds is all there is) and a = sum_u p_u v_u (__dmul_rn, __dadd_rn: no
// contraction, as the host build, which has no FMA). The softmax between them (max, exp, the sum) stays on the host, the same
// libm exp on the same doubles, so each output bf(a / den) is the float attend_p gives.
__global__ void att_scores(const float *q, const float *K, const size_t *koff, const size_t *soff, const unsigned *T,
                           int nq, int nkv, int hd, double rs, double *sc) {
    extern __shared__ double qs[];
    int r = blockIdx.x, hq = blockIdx.y, g = hq / (nq / nkv);
    for (int d = threadIdx.x; d < hd; d += blockDim.x) qs[d] = q[((size_t)r * nq + hq) * hd + d];
    __syncthreads();
    const float *Kr = K + koff[r]; double *s = sc + soff[r] + (size_t)hq * T[r];
    for (unsigned u = threadIdx.x; u < T[r]; u += blockDim.x) {
        const float *k = Kr + ((size_t)u * nkv + g) * hd; double a = 0;
        for (int d = 0; d < hd; d++) a = __dadd_rn(a, __dmul_rn(qs[d], (double)k[d]));
        s[u] = __ddiv_rn(a, rs);
    }
}
__global__ void att_values(const double *p, const float *V, const size_t *koff, const size_t *soff, const unsigned *T,
                           int nq, int nkv, int hd, double *out) {
    int r = blockIdx.x, hq = blockIdx.y, g = hq / (nq / nkv), d = threadIdx.x;
    const float *Vr = V + koff[r] + (size_t)g * hd + d; const double *pr = p + soff[r] + (size_t)hq * T[r]; double a = 0;
    for (unsigned u = 0; u < T[r]; u++) a = __dadd_rn(a, __dmul_rn(pr[u], (double)Vr[(size_t)u * nkv * hd]));
    out[((size_t)r * nq + hq) * hd + d] = a;
}
static int GATT; static double t_att; static long long st_attdiff;
static float *ga_q, *ga_k, *ga_v; static double *ga_s, *ga_o, *ga_hs, *ga_ho; static size_t *ga_koff, *ga_soff; static unsigned *ga_T;
static size_t GA_ROWS = 4096, GA_SC = (size_t)48 << 20, GA_KV;
static void gatt_init(size_t maxrows) {
    size_t QD = (size_t)D_Q * D_HD; GA_KV = maxrows * D_KV * D_HD;
    CK(cudaMalloc(&ga_q, GA_ROWS * QD * 4)); CK(cudaMalloc(&ga_k, GA_KV * 4)); CK(cudaMalloc(&ga_v, GA_KV * 4));
    CK(cudaMalloc(&ga_s, GA_SC * 8)); CK(cudaMalloc(&ga_o, GA_ROWS * QD * 8));
    CK(cudaMallocHost(&ga_hs, GA_SC * 8)); CK(cudaMallocHost(&ga_ho, GA_ROWS * QD * 8));
    CK(cudaMalloc(&ga_koff, GA_ROWS * 8)); CK(cudaMalloc(&ga_soff, GA_ROWS * 8)); CK(cudaMalloc(&ga_T, GA_ROWS * 4));
}
// o[r] = attention of row r (prompt rows[r].first, position rows[r].second) over its prompt's cache of layer L
static void gatt(std::vector<BSeq> &S, const std::vector<std::pair<int, size_t>> &rows, int L, const std::vector<float> &q, std::vector<float> &o) {
    auto t0 = std::chrono::steady_clock::now();
    size_t R = rows.size(), QD = (size_t)D_Q * D_HD;
    std::vector<size_t> base(S.size(), SIZE_MAX); size_t kv = 0;   // each prompt's cache once, concatenated
    for (auto &rw : rows) if (base[rw.first] == SIZE_MAX) {
        auto &s = S[rw.first]; base[rw.first] = kv;
        if (kv + s.kc[L].size() > GA_KV) { fprintf(stderr, "gatt: the caches of a pass exceed %zu values\n", GA_KV); exit(1); }
        CK(cudaMemcpy(ga_k + kv, s.kc[L].data(), s.kc[L].size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ga_v + kv, s.vc[L].data(), s.vc[L].size() * 4, cudaMemcpyHostToDevice)); kv += s.kc[L].size();
    }
    double rs = sqrt((double)D_HD);
    for (size_t r0 = 0; r0 < R;) {   // groups of rows whose scores fit GA_SC
        size_t r1 = r0, ns = 0; std::vector<size_t> koff, soff; std::vector<unsigned> T;
        while (r1 < R && r1 - r0 < GA_ROWS && ns + (rows[r1].second + 1) * D_Q <= GA_SC) {
            koff.push_back(base[rows[r1].first]); soff.push_back(ns); T.push_back(rows[r1].second + 1); ns += (rows[r1].second + 1) * D_Q; r1++; }
        if (r1 == r0) { fprintf(stderr, "gatt: one row's scores exceed %zu\n", GA_SC); exit(1); }
        size_t n = r1 - r0;
        CK(cudaMemcpy(ga_q, q.data() + r0 * QD, n * QD * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ga_koff, koff.data(), n * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ga_soff, soff.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ga_T, T.data(), n * 4, cudaMemcpyHostToDevice));
        att_scores<<<dim3(n, D_Q), 128, D_HD * 8>>>(ga_q, ga_k, ga_koff, ga_soff, ga_T, D_Q, D_KV, D_HD, rs, ga_s);
        CK(cudaMemcpy(ga_hs, ga_s, ns * 8, cudaMemcpyDeviceToHost));
        std::vector<double> den(n * D_Q);
        #pragma omp parallel for schedule(dynamic, 16)
        for (size_t i = 0; i < n * D_Q; i++) {   // the softmax of attend_p, on its doubles
            size_t j = i / D_Q; double *v = ga_hs + soff[j] + (i % D_Q) * T[j], mx = -1e300, dn = 0;
            for (unsigned u = 0; u < T[j]; u++) mx = std::max(mx, v[u]);
            for (unsigned u = 0; u < T[j]; u++) { v[u] = exp(v[u] - mx); dn += v[u]; }
            den[i] = dn;
        }
        CK(cudaMemcpy(ga_s, ga_hs, ns * 8, cudaMemcpyHostToDevice));
        att_values<<<dim3(n, D_Q), D_HD>>>(ga_s, ga_v, ga_koff, ga_soff, ga_T, D_Q, D_KV, D_HD, ga_o);
        CK(cudaGetLastError()); CK(cudaMemcpy(ga_ho, ga_o, n * QD * 8, cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n * QD; i++) o[r0 * QD + i] = bf(ga_ho[i] / den[i / D_HD]);
        r0 = r1;
    }
    t_att += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// one pass: rows (sequence, position), in increasing position within a sequence; logits for the rows marked in want
static void pass(std::vector<BSeq> &S, const std::vector<std::pair<int, size_t>> &rows, const std::vector<bool> &want) {
    auto t0 = std::chrono::steady_clock::now(); double d0 = t_disk, b0 = bytes_disk, g0 = t_gemm, a0 = t_att;
    size_t R = rows.size(), QD = (size_t)D_Q * D_HD, KVD = (size_t)D_KV * D_HD;
    auto part0 = [&](int L) {   // read-ahead (OZ_AHEAD): a layer's tensors before routing, in the order they are used
        std::string P = "model.layers." + std::to_string(L) + ".";
        for (const char *n : {"q_proj", "k_proj", "v_proj", "o_proj"}) ahead.push(P + "self_attn." + n + ".weight");
        if (MOE[L]) ahead.push(P + "mlp.gate.weight");
        else for (const char *n : {"gate_proj", "up_proj", "down_proj"}) ahead.push(P + "mlp." + n + ".weight");
    };
    part0(0);
    std::vector<float> h(R * D_H);
    for (size_t r = 0; r < R; r++) { auto e = emb_row("model.embed_tokens.weight", S[rows[r].first].toks[rows[r].second], D_H); std::copy(e.begin(), e.end(), h.begin() + r * D_H); }
    if (HEMB) HEMB(h);
    std::vector<std::vector<float>> hist(VALIDATE ? R : 0);   // every row's residual stream after every layer (the digest)
    for (int L = 0; L < NL; L++) {
        if (HSAVE) HSAVE->emplace_back(h.begin(), h.begin() + std::min(R, HSAVE_ROWS) * D_H);
        std::string P = "model.layers." + std::to_string(L) + ".";
        std::vector<float> x = norm_rows(h, N1[L]);
        if (!MOE[L] && L + 1 < NL) part0(L + 1);
        DevW W = load_w(P + "self_attn.q_proj.weight", F_BF16, QD, D_H); auto q = PROJ ? PROJ(L, 0, W, x) : bmm(W, x);
        rows_map(q, QD, [&](size_t r, std::vector<float> &v) { rope_part(v, D_Q, rows[r].second); });
        W = load_w(P + "self_attn.k_proj.weight", F_BF16, KVD, D_H); auto k = bmm(W, x);
        rows_map(k, KVD, [&](size_t r, std::vector<float> &v) { rope_part(v, D_KV, rows[r].second); });
        W = load_w(P + "self_attn.v_proj.weight", F_BF16, KVD, D_H); auto v = PROJ ? PROJ(L, 1, W, x) : bmm(W, x);
        for (size_t r = 0; r < R; r++) {   // the cache of position t is row t
            auto &s = S[rows[r].first]; size_t t = rows[r].second;
            if (s.kc[L].size() != t * KVD) { fprintf(stderr, "cache of prompt %d layer %d holds %zu rows at position %zu\n", s.prompt, L, s.kc[L].size() / KVD, t); exit(1); }
            s.kc[L].insert(s.kc[L].end(), k.begin() + r * KVD, k.begin() + (r + 1) * KVD);
            s.vc[L].insert(s.vc[L].end(), v.begin() + r * KVD, v.begin() + (r + 1) * KVD);
        }
        std::vector<float> o(R * QD);
        if (GATT) gatt(S, rows, L, q, o);
        if (GATT != 1) {
            auto ta = std::chrono::steady_clock::now(); std::vector<float> oh(R * QD);
            auto att = [&](size_t r, bool par) { auto &s = S[rows[r].first];
                attend_p(q.data() + r * QD, s.kc[L].data(), s.vc[L].data(), rows[r].second + 1, D_Q, D_KV, D_HD, oh.data() + r * QD, par); };
            if (R >= 16) {
                #pragma omp parallel for schedule(dynamic)
                for (size_t r = 0; r < R; r++) att(r, false);
            } else for (size_t r = 0; r < R; r++) att(r, true);
            if (GATT) for (size_t i = 0; i < o.size(); i++) st_attdiff += memcmp(&o[i], &oh[i], 4) != 0;
            o.swap(oh); if (!GATT) t_att += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta).count();
        }
        if (HSAVE) for (auto &s : S) { std::vector<float>().swap(s.kc[L]); std::vector<float>().swap(s.vc[L]); }   // training: K, V come again from the saved input
        W = load_w(P + "self_attn.o_proj.weight", F_BF16, D_H, QD); add_rows(h, bmm(W, o));
        x = norm_rows(h, N2[L]);
        std::vector<float> out;
        if (!MOE[L]) {
            W = load_w(P + "mlp.gate_proj.weight", F_BF16, D_FF, D_H); auto g = bmm(W, x);
            W = load_w(P + "mlp.up_proj.weight", F_BF16, D_FF, D_H); g = swiglu_rows(g, bmm(W, x));
            W = load_w(P + "mlp.down_proj.weight", F_BF16, D_H, D_FF); out = bmm(W, g);
        } else {
            W = load_w(P + "mlp.gate.weight", F_BF16, J_E, D_H); auto lg = bmm(W, x);
            std::vector<std::vector<int>> ex(R); std::vector<std::vector<double>> wt(R);
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < R; r++) route(std::vector<float>(lg.begin() + r * J_E, lg.begin() + (r + 1) * J_E), BIAS[L], ex[r], wt[r]);
            std::vector<std::vector<std::pair<size_t, double>>> by(J_E);   // expert -> (row, weight)
            for (size_t r = 0; r < R; r++) for (int i = 0; i < J_TOPK; i++) by[ex[r][i]].push_back({r, wt[r][i]});
            if (EXUSED) { auto &u = (*EXUSED)[L]; u.assign(J_E, 0);
                for (int e = 0; e < J_E; e++) for (auto &p : by[e]) if (p.first < HSAVE_ROWS) { u[e] = 1; break; } }
            for (int e = 0; e < J_E; e++) if (!by[e].empty()) for (const char *n : {"gate_proj", "up_proj", "down_proj"}) for (const char *s : {".weight", ".weight_scale"})
                ahead.push(P + "mlp.experts." + std::to_string(e) + "." + n + s);
            for (const char *n : {"gate_proj", "up_proj", "down_proj"}) ahead.push(P + "mlp.shared_experts." + n + ".weight");
            if (L + 1 < NL) part0(L + 1);
            std::vector<double> acc(R * D_H, 0.0);
            for (int e = 0; e < J_E; e++) {   // ascending id, as the std::set of jais.cu: the same FP64 sum order per row
                if (by[e].empty()) continue;
                std::string EP = P + "mlp.experts." + std::to_string(e) + ".";
                size_t n = by[e].size(); std::vector<float> xe(n * D_H);
                for (size_t i = 0; i < n; i++) std::copy(x.begin() + by[e][i].first * D_H, x.begin() + (by[e][i].first + 1) * D_H, xe.begin() + i * D_H);
                W = fp8w(EP + "gate_proj", J_MFF, D_H); auto g = bmm(W, xe);
                W = fp8w(EP + "up_proj", J_MFF, D_H); g = swiglu_rows(g, bmm(W, xe));
                W = fp8w(EP + "down_proj", D_H, J_MFF); auto d = bmm(W, g);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) { double w = by[e][i].second; double *a = acc.data() + by[e][i].first * D_H; const float *dd = d.data() + i * D_H;
                    for (int c = 0; c < D_H; c++) a[c] += w * dd[c]; }
            }
            W = load_w(P + "mlp.shared_experts.gate_proj.weight", F_BF16, J_MFF, D_H); auto g = bmm(W, x);
            W = load_w(P + "mlp.shared_experts.up_proj.weight", F_BF16, J_MFF, D_H); g = swiglu_rows(g, bmm(W, x));
            W = load_w(P + "mlp.shared_experts.down_proj.weight", F_BF16, D_H, J_MFF); out = bmm(W, g);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < out.size(); i++) out[i] = bf((double)bf(acc[i]) + out[i]);
        }
        add_rows(h, out);
        if (VALIDATE) for (size_t r = 0; r < R; r++) hist[r].insert(hist[r].end(), h.begin() + r * D_H, h.begin() + (r + 1) * D_H);
    }
    if (HSAVE) HSAVE->push_back(h);
    // logits of the wanted rows: lm_head in slices of 16384 vocabulary rows (the digit planes of a slice fit the GPU)
    std::vector<size_t> wr; for (size_t r = 0; r < R; r++) if (want[r]) wr.push_back(r);
    std::vector<float> hn(wr.size() * D_H), lg(wr.size() * (size_t)D_V);
    for (size_t i = 0; i < wr.size(); i++) std::copy(h.begin() + wr[i] * D_H, h.begin() + (wr[i] + 1) * D_H, hn.begin() + i * D_H);
    hn = norm_rows(hn, NORMF);
    for (int v0 = 0; v0 < D_V && !wr.empty(); v0 += 16384) {
        DevW Ls = LM; Ls.M = std::min(16384, D_V - v0); Ls.w = (const uint16_t *)LM.w + (size_t)v0 * D_H;
        std::vector<float> y(wr.size() * Ls.M); bmm(Ls, hn.data(), wr.size(), 24, y.data());
        for (size_t i = 0; i < wr.size(); i++) std::copy(y.begin() + i * Ls.M, y.begin() + (i + 1) * Ls.M, lg.begin() + i * D_V + v0);
    }
    for (size_t r = 0, i = 0; r < R; r++) {
        auto &s = S[rows[r].first]; size_t t = rows[r].second;
        if (VALIDATE) for (int L = 0; L < NL; L++) s.dig = fnv(s.dig, std::vector<float>(hist[r].begin() + (size_t)L * D_H, hist[r].begin() + (size_t)(L + 1) * D_H));
        if (!want[r]) continue;
        std::vector<float> l(lg.begin() + i * D_V, lg.begin() + (i + 1) * D_V); i++;
        if (VALIDATE) s.dig = fnv(s.dig, l);
        int b1 = pick(l, t, s.prompt, 0, s.toks[t]);
        if (t + 1 == s.plen && !SCORE.empty()) { printf("score prompt %d", s.prompt); for (int c : SCORE) printf(" %d %.6f", c, l[c]); printf("\n"); }
        if (t + 1 >= s.plen && t + 1 >= s.toks.size()) s.toks.push_back(b1);
    }
    fflush(stdout);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "pass %zu rows: %.1f s (disk %.1f s, %.1f GB; products %.1f s; attention %.1f s %s), fixup %lld + FP8 %lld (open %lld), host GEMV rows %lld, finished on the host %lld, uncertified %lld%s\n",
            R, sec, t_disk - d0, (bytes_disk - b0) / 1e9, t_gemm - g0, t_att - a0, GATT == 1 ? "GPU" : GATT ? "GPU + host" : "host",
            st_fix, st_fix8, st_fixopen, st_host, st_sup, st_bad, GATT == 2 ? (", attention outputs differing GPU/host " + std::to_string(st_attdiff)).c_str() : "");
}

// the device buffers, lm_head, the norms and router biases of the NL layers (after jais_init; also jais_lora.cu's)
static void jb_init() {
    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)D_FF * D_H * 2)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, 32768 * 4)); CK(cudaMalloc(&dxe, 32768 * 2)); CK(cudaMalloc(&dy, (size_t)D_V * 4)); CK(cudaMalloc(&dbad, 4));
    void *dLM; CK(cudaMalloc(&dLM, (size_t)D_V * D_H * 2)); LM = load_w("lm_head.weight", F_BF16, D_V, D_H, dLM);
    if (const char *a = getenv("OZ_AHEAD")) { ahead.start((size_t)(atof(a) * 1e9)); fprintf(stderr, "read-ahead ring %.1f GB\n", atof(a)); }
    bmm_init((size_t)std::max(D_FF, 16384) * D_H, std::max(D_FF, D_Q * D_HD), std::max(D_FF, 16384));
    bool whole = idx.count("model.norm.weight");
    NORMF = whole ? vec_bf16("model.norm.weight") : std::vector<float>(D_H, 1.f);
    N1.resize(NL); N2.resize(NL); BIAS.resize(NL);
    for (int L = 0; L < NL; L++) { std::string P = "model.layers." + std::to_string(L) + ".";
        N1[L] = vec_bf16(P + "input_layernorm.weight"); N2[L] = vec_bf16(P + "post_attention_layernorm.weight");
        if (MOE[L]) BIAS[L] = vec_bf16(P + "mlp.gate.bias"); }
}

#ifndef JAIS_BATCH_LIB
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: out_jais_batch <model dir> <prompts> <new tokens> [--validate]\n"); return 1; }
    DIR = argv[1]; int NEW = atoi(argv[3]); VALIDATE = argc > 4 && !strcmp(argv[4], "--validate");
    if (const char *sc = getenv("OZ_SCORE")) { std::stringstream ss(sc); std::string t; while (std::getline(ss, t, ',')) SCORE.push_back(atoi(t.c_str())); }
    MOE = jais_init(); NL = MOE.size();
    std::vector<BSeq> all;
    { std::ifstream f(argv[2]); std::string line; int pi = 0;
      while (std::getline(f, line)) { if (line.empty()) continue; std::stringstream ss(line); BSeq s; int v; while (ss >> v) s.toks.push_back(v);
        s.prompt = pi++; s.plen = s.toks.size(); all.push_back(s); } }
    size_t steps = 0; for (auto &s : all) steps = std::max(steps, s.plen + NEW);
    bool per = getenv("OZ_PERPROMPT") && atoi(getenv("OZ_PERPROMPT")); for (auto &s : all) s.end = per ? s.plen + NEW : steps;
    size_t CHUNK = getenv("OZ_CHUNK") ? atol(getenv("OZ_CHUNK")) : 24000;
    fprintf(stderr, "jais_batch: %d layers (%d MoE), %zu prompts, %d new tokens%s, chunks of %zu tokens%s\n", NL,
            (int)std::count(MOE.begin(), MOE.end(), true), all.size(), NEW, per ? " each" : " (to the longest)", CHUNK, VALIDATE ? ", validate" : "");

    jb_init();
    GATT = getenv("OZ_GATT") ? atoi(getenv("OZ_GATT")) : 0;
    if (GATT) { size_t mr = CHUNK; for (auto &s : all) mr = std::max(mr, s.end); gatt_init(mr); }

    std::vector<BSeq> done;
    for (size_t c0 = 0; c0 < all.size();) {
        size_t c1 = c0, tok = 0; while (c1 < all.size() && (c1 == c0 || tok + all[c1].end <= CHUNK)) tok += all[c1++].end;
        std::vector<BSeq> S(all.begin() + c0, all.begin() + c1);
        for (auto &s : S) { s.kc.resize(NL); s.vc.resize(NL); }
        fprintf(stderr, "chunk: prompts %zu-%zu, up to %zu tokens\n", c0, c1 - 1, tok);
        for (bool first = true;; first = false) {   // the prefill, then one pass per position until every prompt reached its end
            std::vector<std::pair<int, size_t>> rows; std::vector<bool> want;
            for (size_t j = 0; j < S.size(); j++) {
                size_t t0 = S[j].kc[0].size() / ((size_t)D_KV * D_HD), t1 = first ? S[j].plen : t0 + 1;
                for (size_t t = t0; t < std::min(t1, S[j].end); t++) { rows.push_back({(int)j, t}); want.push_back(VALIDATE || t + 1 >= S[j].plen); }
            }
            if (rows.empty()) break;
            pass(S, rows, want);
        }
        for (auto &s : S) { s.kc.clear(); s.vc.clear(); done.push_back(std::move(s)); }
        c0 = c1;
    }
    printf("batched: %lld rows; BF16 outputs %lld, left to the fixup %lld (%.4f%%); FP32 logits %lld, fixup %lld (%.4f%%); host GEMV rows %lld; attention outputs differing GPU/host %lld\n",
           st_rows, st_outs, st_fix, 100.0 * st_fix / std::max(1LL, st_outs), st_outs24, st_fix24, 100.0 * st_fix24 / std::max(1LL, st_outs24), st_host, st_attdiff);
    ahead.finish();
    report(done);
}
#endif
