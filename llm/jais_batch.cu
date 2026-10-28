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
// nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jais_batch llm/jais_batch.cu -lgomp
// out_jais_batch <model dir> <prompts file: token ids per line> <new tokens> [--validate]
#define JAIS_LIB
#include "jais.cu"
#define BGEMM_LIB
#include "bgemm.cu"
#include <omp.h>

// ---- the batched product: Y[n][M] = X[n][K] W^T to p bits, every output certified ------------------------------------------
static int8_t *g_wd, *g_xd; static int16_t *g_wl, *g_xl; static uint8_t *g_wn, *g_xn, *g_xq; static uint16_t *g_xb;
static float *g_x, *g_xs, *g_Y; static unsigned *g_list, *g_lcnt; static int *g_nb; static size_t G_NB;
static const unsigned G_CAP = 1u << 22;
static long long st_rows, st_outs, st_outs24, st_fix, st_fix24, st_host;   // rows, outputs (p 8; p 24), left to the CUDA-core fixup, rows to the host GEMV
static double t_gemm;

static void bmm_init(size_t maxMK, size_t maxK, size_t maxM) {
    G_NB = getenv("OZ_NB") ? atol(getenv("OZ_NB")) : 4096;
    CK(cudaMalloc(&g_wd, (size_t)DW24 * maxMK)); CK(cudaMalloc(&g_wl, maxMK / 128 * 2)); CK(cudaMalloc(&g_wn, maxMK / 128));
    CK(cudaMalloc(&g_x, G_NB * maxK * 4)); CK(cudaMalloc(&g_xb, G_NB * maxK * 2)); CK(cudaMalloc(&g_xd, (size_t)DX24 * G_NB * maxK));
    CK(cudaMalloc(&g_xq, G_NB * maxK)); CK(cudaMalloc(&g_xs, G_NB * maxK / 128 * 4)); CK(cudaMalloc(&g_xl, G_NB * maxK / 128 * 2));
    CK(cudaMalloc(&g_xn, G_NB * maxK / 128)); CK(cudaMalloc(&g_Y, G_NB * maxM * 4));
    CK(cudaMalloc(&g_list, (size_t)G_CAP * 4)); CK(cudaMalloc(&g_lcnt, 4)); CK(cudaMalloc(&g_nb, 4));
}
static void bmm(const DevW &W, const float *X, size_t N, int p, float *Y) {
    auto t0 = std::chrono::steady_clock::now();
    int M = W.M, K = W.K, KB = K / 128; bool fp8 = W.fmt == F_E4M3B;
    if (fp8) fp8_digits<<<1024, 256>>>((const uint8_t *)W.w, g_wd, (size_t)M * K);
    else bf16_digits<<<(unsigned)(((size_t)M * KB + 255) / 256), 256>>>((const uint16_t *)W.w, g_wd, g_wl, g_wn, M, K, wdig(p));
    std::vector<uint16_t> xb;
    for (size_t n0 = 0; n0 < N; n0 += G_NB) {
        size_t n = std::min(G_NB, N - n0), np = (n + 7) / 8 * 8; const float *x = X + n0 * K;
        if (fp8) {
            CK(cudaMemcpy(g_x, x, n * K * 4, cudaMemcpyHostToDevice));
            if (np > n) CK(cudaMemset(g_x + n * K, 0, (np - n) * K * 4));
            quant_fp8g<<<1024, 256>>>(g_x, g_xq, g_xs, np * KB);
            fp8_digits<<<1024, 256>>>(g_xq, g_xd, np * K);
            fp8b_gemm<<<dim3(M / 128, np / 8), 256>>>(g_wd, (const uint16_t *)W.sw, g_xd, g_xs, g_Y, M, np, K, p, g_nb);
        } else {
            xb.assign(np * K, 0);
            for (size_t i = 0; i < n * K; i++) { uint32_t u; memcpy(&u, &x[i], 4); xb[i] = u >> 16; }   // BF16 values: exact
            CK(cudaMemcpy(g_xb, xb.data(), np * K * 2, cudaMemcpyHostToDevice));
            bf16_digits<<<(unsigned)((np * KB + 255) / 256), 256>>>(g_xb, g_xd, g_xl, g_xn, np, K, xdig(p));
            bf16_gemm_p(p, g_wd, g_wl, g_wn, g_xd, g_xl, g_xn, g_Y, M, np, K, g_nb);
            CK(cudaMemset(g_lcnt, 0, 4)); CK(cudaMemset(g_nb, 0, 4));
            nan_list<<<1024, 256>>>(g_Y, np * M, g_list, g_lcnt, G_CAP);
            bf16_fixup<<<1024, 256>>>((const uint16_t *)W.w, g_xb, g_list, g_lcnt, G_CAP, g_Y, M, K, p, g_nb);
            unsigned lc; CK(cudaMemcpy(&lc, g_lcnt, 4, cudaMemcpyDeviceToHost)); (p == 24 ? st_fix24 : st_fix) += lc;
            int nb; CK(cudaMemcpy(&nb, g_nb, 4, cudaMemcpyDeviceToHost)); st_bad += nb;   // the fixup's own (never seen)
        }
        float *y = Y + n0 * M; CK(cudaMemcpy(y, g_Y, n * M * 4, cudaMemcpyDeviceToHost));
        for (size_t r = 0; r < n; r++) {   // what is still NaN: the GEMV (beyond the fixup's list; FP8 outputs it could not certify)
            float *yr = y + r * M; bool any = false; for (int m = 0; m < M && !any; m++) any = yr[m] != yr[m];
            if (!any) continue; st_host++;
            std::vector<float> xv(x + r * K, x + (r + 1) * K); auto g = gemv(W, fp8 ? act_fp8g(xv) : act_plain(xv), p, 0);
            for (int m = 0; m < M; m++) if (yr[m] != yr[m]) yr[m] = g[m];
        }
        if (fp8) CK(cudaMemset(g_nb, 0, 4));
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

// one pass: rows (sequence, position), in increasing position within a sequence; logits for the rows marked in want
static void pass(std::vector<BSeq> &S, const std::vector<std::pair<int, size_t>> &rows, const std::vector<bool> &want) {
    auto t0 = std::chrono::steady_clock::now(); double d0 = t_disk, b0 = bytes_disk, g0 = t_gemm;
    size_t R = rows.size(), QD = (size_t)D_Q * D_HD, KVD = (size_t)D_KV * D_HD;
    std::vector<float> h(R * D_H);
    for (size_t r = 0; r < R; r++) { auto e = emb_row("model.embed_tokens.weight", S[rows[r].first].toks[rows[r].second], D_H); std::copy(e.begin(), e.end(), h.begin() + r * D_H); }
    std::vector<std::vector<float>> hist(VALIDATE ? R : 0);   // every row's residual stream after every layer (the digest)
    for (int L = 0; L < NL; L++) {
        std::string P = "model.layers." + std::to_string(L) + ".";
        std::vector<float> x = norm_rows(h, N1[L]);
        DevW W = load_w(P + "self_attn.q_proj.weight", F_BF16, QD, D_H); auto q = bmm(W, x);
        rows_map(q, QD, [&](size_t r, std::vector<float> &v) { rope_part(v, D_Q, rows[r].second); });
        W = load_w(P + "self_attn.k_proj.weight", F_BF16, KVD, D_H); auto k = bmm(W, x);
        rows_map(k, KVD, [&](size_t r, std::vector<float> &v) { rope_part(v, D_KV, rows[r].second); });
        W = load_w(P + "self_attn.v_proj.weight", F_BF16, KVD, D_H); auto v = bmm(W, x);
        for (size_t r = 0; r < R; r++) {   // the cache of position t is row t
            auto &s = S[rows[r].first]; size_t t = rows[r].second;
            if (s.kc[L].size() != t * KVD) { fprintf(stderr, "cache of prompt %d layer %d holds %zu rows at position %zu\n", s.prompt, L, s.kc[L].size() / KVD, t); exit(1); }
            s.kc[L].insert(s.kc[L].end(), k.begin() + r * KVD, k.begin() + (r + 1) * KVD);
            s.vc[L].insert(s.vc[L].end(), v.begin() + r * KVD, v.begin() + (r + 1) * KVD);
        }
        std::vector<float> o(R * QD);
        auto att = [&](size_t r, bool par) { auto &s = S[rows[r].first];
            attend_p(q.data() + r * QD, s.kc[L].data(), s.vc[L].data(), rows[r].second + 1, D_Q, D_KV, D_HD, o.data() + r * QD, par); };
        if (R >= 16) {
            #pragma omp parallel for schedule(dynamic)
            for (size_t r = 0; r < R; r++) att(r, false);
        } else for (size_t r = 0; r < R; r++) att(r, true);
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
    fprintf(stderr, "pass %zu rows: %.1f s (disk %.1f s, %.1f GB; products %.1f s), fixup %lld, host GEMV rows %lld, uncertified %lld\n",
            R, sec, t_disk - d0, (bytes_disk - b0) / 1e9, t_gemm - g0, st_fix, st_host, st_bad);
}

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

    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)D_FF * D_H * 2)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, 32768 * 4)); CK(cudaMalloc(&dxe, 32768 * 2)); CK(cudaMalloc(&dy, (size_t)D_V * 4)); CK(cudaMalloc(&dbad, 4));
    void *dLM; CK(cudaMalloc(&dLM, (size_t)D_V * D_H * 2)); LM = load_w("lm_head.weight", F_BF16, D_V, D_H, dLM);
    bmm_init((size_t)std::max(D_FF, 16384) * D_H, std::max(D_FF, D_Q * D_HD), std::max(D_FF, 16384));
    bool whole = idx.count("model.norm.weight");
    NORMF = whole ? vec_bf16("model.norm.weight") : std::vector<float>(D_H, 1.f);
    N1.resize(NL); N2.resize(NL); BIAS.resize(NL);
    for (int L = 0; L < NL; L++) { std::string P = "model.layers." + std::to_string(L) + ".";
        N1[L] = vec_bf16(P + "input_layernorm.weight"); N2[L] = vec_bf16(P + "post_attention_layernorm.weight");
        if (MOE[L]) BIAS[L] = vec_bf16(P + "mlp.gate.bias"); }

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
    printf("batched: %lld rows; BF16 outputs %lld, left to the fixup %lld (%.4f%%); FP32 logits %lld, fixup %lld (%.4f%%); host GEMV rows %lld\n",
           st_rows, st_outs, st_fix, 100.0 * st_fix / std::max(1LL, st_outs), st_outs24, st_fix24, 100.0 * st_fix24 / std::max(1LL, st_outs24), st_host);
    report(done);
}
