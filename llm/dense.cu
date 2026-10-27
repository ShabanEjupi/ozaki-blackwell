// dense.cu -- the fully open dense models token by token from disk, every linear map correctly rounded (the GEMV of
// ultra.cu, self-tested against a 2560-bit superaccumulator), against FP32 engines with split-K 1 and 4 on the same host code.
//  * OLMo-2-0325-32B-Instruct: weights Apache-2.0, every training token public (ODC-BY) and indexed by infini-gram
//    (v4_olmo-2-0325-32b-instruct_llama, 4.6T tokens), so each generated span can be looked up (llm/olmo_trace.py).
//    transformers v5.3.0 modeling_olmo2.py: post-norm h += RMSNorm(attn(h)), h += RMSNorm(mlp(h)); q_norm / k_norm over
//    all heads before RoPE; RMSNorm = (w * x/rms) then BF16.
//  * Llama architecture (IFM K2-Think-V2, 72.6B, Apache-2.0, TxT360 ODC-BY): pre-norm, RMSNorm = w * BF16(x/rms), RoPE with
//    YaRN when config.rope_scaling says so (transformers v4.57.1 _compute_yarn_parameters: ramp between beta_fast 32 and
//    beta_slow 1, attention_factor 0.1 ln(factor) + 1 on cos and sin). F32 checkpoints are rounded to BF16 on upload, as
//    an engine loads them for config dtype bfloat16.
// Shapes, theta, eps from config.json. Host parts in FP64, rounded once to BF16 (logits FP32).
// nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_dense llm/dense.cu -lgomp
// out_dense <model dir with index.txt and config.json> <prompts file: token ids per line> <new tokens> [modes, e.g. 0,1,4]
#define ULTRA_LIB
#include "ultra.cu"

static int D_H, D_Q, D_KV, D_HD = 128, D_FF, D_V; static double D_EPS, D_THETA, Y_FACTOR = 1, Y_ORIG = 0; static bool OLMO;
static std::vector<float> inv_freq; static double attn_factor = 1;
static double cfgnum(const std::string &s, const std::string &key, double def) {
    size_t a = s.find("\"" + key + "\""); if (a == std::string::npos) return def;
    a = s.find(':', a); return atof(s.c_str() + a + 1);
}
static void rope_init() {   // inv_freq[i], i < head_dim/2, in FP32 as the reference builds it; YaRN if configured
    int dim = D_HD; inv_freq.resize(dim / 2);
    double lo = 0, hi = 0;
    if (Y_FACTOR > 1) {
        auto corr = [&](double rot) { return dim * log(Y_ORIG / (rot * 2 * M_PI)) / (2 * log(D_THETA)); };
        lo = std::max(floor(corr(32)), 0.0); hi = std::min(ceil(corr(1)), (double)dim - 1); if (lo == hi) hi += 0.001;
        attn_factor = 0.1 * log(Y_FACTOR) + 1.0;
    }
    for (int i = 0; i < dim / 2; i++) {
        float pf = (float)pow(D_THETA, (double)(2 * i) / dim), ext = 1.f / pf;
        if (Y_FACTOR <= 1) { inv_freq[i] = ext; continue; }
        float intp = 1.f / ((float)Y_FACTOR * pf), ramp = std::min(1.f, std::max(0.f, (float)((i - lo) / (hi - lo)))), ef = 1.f - ramp;
        inv_freq[i] = intp * (1.f - ef) + ext * ef;
    }
}
static void rope(std::vector<float> &v, int heads, int pos) {   // rotate-half
    for (int h = 0; h < heads; h++) for (int i = 0; i < D_HD / 2; i++) {
        double a = (double)pos * inv_freq[i], c = cos(a) * attn_factor, s = sin(a) * attn_factor;
        double x1 = v[h * D_HD + i], x2 = v[h * D_HD + i + D_HD / 2];
        v[h * D_HD + i] = bf(x1 * c - x2 * s); v[h * D_HD + i + D_HD / 2] = bf(x2 * c + x1 * s);
    }
}
static void norm(const std::vector<float> &x, const std::vector<float> &w, std::vector<float> &y) {
    double ss = 0; for (float v : x) ss += (double)v * v;
    double r = 1.0 / sqrt(ss / x.size() + D_EPS);
    y.resize(x.size());
    for (size_t i = 0; i < x.size(); i++) y[i] = OLMO ? bf((double)w[i] * ((double)x[i] * r)) : bf((double)w[i] * bf((double)x[i] * r));
}
__global__ void f32_to_bf16(const float *s, uint16_t *d, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        __nv_bfloat16 b = __float2bfloat16_rn(s[i]); d[i] = *(uint16_t *)&b;
    }
}
static void *dW2;   // F32 tensors land in dW2 and are rounded into dW
static DevW load_bf(const std::string &name, int M, int K, void *dst = nullptr) {
    if (idx.at(name).dt == "BF16") return load_w(name, F_BF16, M, K, dst);
    DevW W = load_w(name, F_F32, M, K, dW2);
    void *d = dst ? dst : dW; f32_to_bf16<<<4096, 256>>>((const float *)dW2, (uint16_t *)d, (size_t)M * K); CK(cudaDeviceSynchronize());
    W.w = d; W.fmt = F_BF16; return W;
}
static std::vector<float> vec_any(const std::string &name) {
    if (idx.at(name).dt == "BF16") return vec_bf16(name);
    auto v = vec_f32(name); for (auto &x : v) x = bf(x); return v;
}
static std::vector<float> emb_row(const std::string &name, int tok, int H) {
    const TI e = idx.at(name); TI r = e; size_t es = e.dt == "BF16" ? 2 : 4;
    r.off += (size_t)tok * H * es; r.n = (size_t)H * es; idx["__row"] = r;
    std::vector<float> h(H);
    if (es == 2) { std::vector<uint16_t> row(H); readt("__row", row.data()); for (int i = 0; i < H; i++) h[i] = bf16f(row[i]); }
    else { readt("__row", h.data()); for (auto &x : h) x = bf(x); }
    return h;
}
struct OSeq { int prompt, mode; std::vector<int> toks; size_t plen; std::vector<float> h; std::vector<std::vector<float>> kc, vc; };

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: out_dense <model dir> <prompts> <new tokens> [modes]\n"); return 1; }
    DIR = argv[1]; int NEW = atoi(argv[3]);
    std::vector<int> modes{0, 1, 4};
    if (argc > 4) { modes.clear(); std::stringstream ss(argv[4]); std::string t; while (std::getline(ss, t, ',')) modes.push_back(atoi(t.c_str())); }
    { std::ifstream f(DIR + "/index.txt"); std::string nm, dt, sh, fl; size_t off, n;
      while (f >> nm >> dt >> sh >> fl >> off >> n) idx[nm] = {dt, fl, off, n}; }
    { std::ifstream f(DIR + "/config.json"); std::string s((std::istreambuf_iterator<char>(f)), {});
      D_H = cfgnum(s, "hidden_size", 0); D_Q = cfgnum(s, "num_attention_heads", 0); D_KV = cfgnum(s, "num_key_value_heads", D_Q);
      D_HD = cfgnum(s, "head_dim", D_H / D_Q); D_FF = cfgnum(s, "intermediate_size", 0); D_V = cfgnum(s, "vocab_size", 0);
      D_EPS = cfgnum(s, "rms_norm_eps", 1e-6); D_THETA = cfgnum(s, "rope_theta", 10000);
      if (s.find("\"yarn\"") != std::string::npos) { Y_FACTOR = cfgnum(s, "factor", 1); Y_ORIG = cfgnum(s, "original_max_position_embeddings", 0); } }
    OLMO = idx.count("model.layers.0.self_attn.q_norm.weight");
    rope_init();
    int NL = 0; while (idx.count("model.layers." + std::to_string(NL) + ".self_attn.q_proj.weight")) NL++;
    std::vector<OSeq> seqs;
    { std::ifstream f(argv[2]); std::string line; int pi = 0;
      while (std::getline(f, line)) { if (line.empty()) continue; std::stringstream ss(line); std::vector<int> t; int v; while (ss >> v) t.push_back(v);
        for (int m : modes) { OSeq s; s.prompt = pi; s.mode = m; s.toks = t; s.plen = t.size(); s.kc.resize(NL); s.vc.resize(NL); seqs.push_back(s); } pi++; } }
    size_t steps = 0; for (auto &s : seqs) steps = std::max(steps, s.plen + NEW);
    fprintf(stderr, "%s: %d layers, H %d, heads %d/%d, FF %d, vocab %d, theta %g, yarn %g/%g (attn %.6f), %zu sequences, %zu steps\n",
            OLMO ? "olmo2" : "llama", NL, D_H, D_Q, D_KV, D_FF, D_V, D_THETA, Y_FACTOR, Y_ORIG, attn_factor, seqs.size(), steps);

    size_t WMAX = (size_t)D_FF * D_H * 2;
    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, WMAX)); CK(cudaMalloc(&dW2, WMAX * 2)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, 32768 * 4)); CK(cudaMalloc(&dxe, 32768 * 2)); CK(cudaMalloc(&dy, (size_t)D_V * 4)); CK(cudaMalloc(&dbad, 4));
    void *dLM; CK(cudaMalloc(&dLM, (size_t)D_V * D_H * 2));
    DevW LM{};
    { size_t chunk = WMAX / ((size_t)D_H * 2), rows = 0; LM.fmt = F_BF16; LM.M = D_V; LM.K = D_H; LM.Gm = 1; LM.Ge = 0; LM.Gf = 1.f; LM.w = dLM;
      if (idx.at("lm_head.weight").dt == "BF16") LM = load_w("lm_head.weight", F_BF16, D_V, D_H, dLM);
      else for (; rows < (size_t)D_V; rows += chunk) {   // F32 lm_head, rounded to BF16 in slices that fit dW2
          size_t r = std::min(chunk, (size_t)D_V - rows); TI t = idx.at("lm_head.weight"); t.off += rows * D_H * 4; t.n = r * D_H * 4;
          idx["__lm"] = t; load_w("__lm", F_F32, r, D_H, dW2);
          f32_to_bf16<<<4096, 256>>>((const float *)dW2, (uint16_t *)dLM + rows * D_H, r * D_H); CK(cudaDeviceSynchronize());
      } }
    auto normf = vec_any("model.norm.weight");
    std::vector<std::vector<float>> qn(NL), kn(NL), n1(NL), n2(NL);
    for (int L = 0; L < NL; L++) {
        std::string P = "model.layers." + std::to_string(L) + ".";
        if (OLMO) { qn[L] = vec_any(P + "self_attn.q_norm.weight"); kn[L] = vec_any(P + "self_attn.k_norm.weight");
                    n1[L] = vec_any(P + "post_attention_layernorm.weight"); n2[L] = vec_any(P + "post_feedforward_layernorm.weight"); }
        else { n1[L] = vec_any(P + "input_layernorm.weight"); n2[L] = vec_any(P + "post_attention_layernorm.weight"); }
    }
    for (size_t t = 0; t < steps; t++) {
        auto t0 = std::chrono::steady_clock::now(); double d0 = t_disk, b0 = bytes_disk;
        for (auto &s : seqs) s.h = emb_row("model.embed_tokens.weight", s.toks[t], D_H);
        for (int L = 0; L < NL; L++) {
            std::string P = "model.layers." + std::to_string(L) + ".";
            size_t NS = seqs.size();
            std::vector<std::vector<float>> x(NS), q(NS), o(NS, std::vector<float>(D_Q * D_HD)), tmp(NS);
            for (size_t j = 0; j < NS; j++) { if (OLMO) x[j] = seqs[j].h; else norm(seqs[j].h, n1[L], x[j]); }
            DevW W = load_bf(P + "self_attn.q_proj.weight", D_Q * D_HD, D_H);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                if (OLMO) norm(y, qn[L], q[j]); else q[j] = y; rope(q[j], D_Q, t); }
            W = load_bf(P + "self_attn.k_proj.weight", D_KV * D_HD, D_H);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0); std::vector<float> k;
                if (OLMO) norm(y, kn[L], k); else k = y; rope(k, D_KV, t); seqs[j].kc[L].insert(seqs[j].kc[L].end(), k.begin(), k.end()); }
            W = load_bf(P + "self_attn.v_proj.weight", D_KV * D_HD, D_H);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0); seqs[j].vc[L].insert(seqs[j].vc[L].end(), y.begin(), y.end()); }
            for (size_t j = 0; j < NS; j++) {
                auto &K = seqs[j].kc[L], &V = seqs[j].vc[L]; size_t T = K.size() / ((size_t)D_KV * D_HD);
                #pragma omp parallel for schedule(static)
                for (int hq = 0; hq < D_Q; hq++) {
                    int g = hq / (D_Q / D_KV); std::vector<double> sc(T); double mx = -1e300;
                    for (size_t u = 0; u < T; u++) { double a = 0; for (int d = 0; d < D_HD; d++) a += (double)q[j][hq * D_HD + d] * K[(u * D_KV + g) * D_HD + d];
                        sc[u] = a / sqrt((double)D_HD); mx = std::max(mx, sc[u]); }
                    double den = 0; for (auto &v : sc) { v = exp(v - mx); den += v; }
                    for (int d = 0; d < D_HD; d++) { double a = 0; for (size_t u = 0; u < T; u++) a += sc[u] * V[(u * D_KV + g) * D_HD + d]; o[j][hq * D_HD + d] = bf(a / den); }
                }
            }
            W = load_bf(P + "self_attn.o_proj.weight", D_H, D_Q * D_HD);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(o[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                if (OLMO) { std::vector<float> n; norm(y, n1[L], n); y = n; }
                for (int i = 0; i < D_H; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + y[i]); }
            for (size_t j = 0; j < NS; j++) { if (OLMO) x[j] = seqs[j].h; else norm(seqs[j].h, n2[L], x[j]); }
            W = load_bf(P + "mlp.gate_proj.weight", D_FF, D_H);
            for (size_t j = 0; j < NS; j++) tmp[j] = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0);
            W = load_bf(P + "mlp.up_proj.weight", D_FF, D_H);
            for (size_t j = 0; j < NS; j++) { auto u = gemv(W, act_plain(x[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                for (int i = 0; i < D_FF; i++) { double gv = tmp[j][i]; tmp[j][i] = bf(gv / (1.0 + exp(-gv)) * u[i]); } }
            W = load_bf(P + "mlp.down_proj.weight", D_H, D_FF);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(tmp[j]), 8, seqs[j].mode, seqs[j].mode == 0);
                if (OLMO) { std::vector<float> n; norm(y, n2[L], n); y = n; }
                for (int i = 0; i < D_H; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + y[i]); }
        }
        for (auto &s : seqs) {
            std::vector<float> hn; norm(s.h, normf, hn);
            auto lg = gemv(LM, act_plain(hn), 24, s.mode, s.mode == 0);
            int b1 = 0; for (int v = 1; v < D_V; v++) if (lg[v] > lg[b1]) b1 = v;
            int b2 = b1 ? 0 : 1; for (int v = 0; v < D_V; v++) if (v != b1 && lg[v] > lg[b2]) b2 = v;
            if (t + 1 >= s.plen && t + 1 >= s.toks.size()) s.toks.push_back(b1);
            printf("step %zu prompt %d mode %d token %d next %d top1 %.6f top2 %d %.6f\n", t, s.prompt, s.mode, s.toks[t], b1, lg[b1], b2, lg[b2]);
        }
        fflush(stdout);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "step %zu: %.1f s (disk %.1f s, %.1f GB), uncertified so far %lld\n", t, sec, t_disk - d0, (bytes_disk - b0) / 1e9, st_bad);
    }
    const char *fn[4] = {"BF16", "FP32", "FP8", "NVFP4"};
    for (int f = 0; f < 4; f++) if (st_out[f])
        printf("shadow %s outputs %lld engine_s1_differs %lld (%.2f%%) engine_s4_differs %lld (%.2f%%)\n", fn[f], st_out[f],
               st_d1[f], 100.0 * st_d1[f] / st_out[f], st_d4[f], 100.0 * st_d4[f] / st_out[f]);
    printf("uncertified %lld\n", st_bad);
    for (auto &s : seqs) { printf("seq prompt %d mode %d:", s.prompt, s.mode); for (int v : s.toks) printf(" %d", v); printf("\n"); }
}
