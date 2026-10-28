// jais.cu -- Jais-3-375B-A23B-FP8 (IFM and Inception, 29-09-2026, Apache-2.0; K2HorizonForCausalLM, modeling_k2_horizon.py
// of the checkpoint, transformers 5.13) token by token from disk, every linear map correctly rounded (the GEMV of ultra.cu,
// self-tested against a 2560-bit superaccumulator), against FP32 engines with split-K 1 and 4 on the same host code.
//  * 61 layers, H 6144, attention 48 query / 8 KV heads of 128, no q/k norm, no attention gate. RoPE only on rope_head_dim
//    = 64 of the 128: split_to_interleaved, rotate the first 64, interleaved_to_split is, in the checkpoint's halves layout,
//    a rotation of the pairs (i, i + 64) for i < 32 by pos * theta^(-2i/64), theta 1e7; the pairs i >= 32 pass unrotated.
//  * Layers 0-2 a dense SwiGLU MLP (16384, BF16). Layers 3-60 MoE: the router is a BF16 linear without its bias (BF16
//    logits), sigmoid in FP32; the top 8 are chosen on sigmoid + bias, weighted by their sigmoids normalized, times 2.5,
//    cast to BF16. 192 routed SwiGLU experts of 1792 in FP8 E4M3 with a BF16 scale per 128x128 block, their inputs
//    quantized to FP8 per group of 128 at run time (act_fp8g, as vLLM does it); one shared expert in BF16.
//    out = BF16(BF16(sum_e w_e y_e) + shared(x)): the routed sum is one accumulation (FP64, rounded once), the add of the
//    shared expert is the model's own BF16 add.
//  * RMSNorm = BF16(w * x/rms), eps 1e-6; lm_head BF16 with FP32 logits.
// Host parts in FP64, rounded once to BF16, as in ultra.cu and dense.cu. OZ_AHEAD=<GB> reads ahead as in ultra.cu.
// nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jais llm/jais.cu -lgomp
// out_jais <model dir with index.txt and config.json> <prompts file: token ids per line> <new tokens> [modes, e.g. 0,1,4]
#pragma once
#define DENSE_LIB
#include "dense.cu"

static int J_MFF, J_E, J_TOPK, J_RHD; static double J_RSF;
static void rope_part(std::vector<float> &v, int heads, int pos) {   // the pairs (i, i + D_HD/2), i < J_RHD/2
    for (int h = 0; h < heads; h++) for (int i = 0; i < J_RHD / 2; i++) {
        double a = (double)pos * inv_freq[i], c = cos(a), s = sin(a);
        double x1 = v[h * D_HD + i], x2 = v[h * D_HD + i + D_HD / 2];
        v[h * D_HD + i] = bf(x1 * c - x2 * s); v[h * D_HD + i + D_HD / 2] = bf(x2 * c + x1 * s);
    }
}
static std::vector<float> swiglu(const std::vector<float> &g, const std::vector<float> &u) {
    std::vector<float> a(g.size()); for (size_t i = 0; i < g.size(); i++) { double gv = g[i]; a[i] = bf(gv / (1.0 + exp(-gv)) * u[i]); }
    return a;
}

// --check <model dir> <tensor without .weight> <M> <K> <out dir>: one FP8 block-scaled map on a fixed wide-range input,
// exact to BF16 and to FP32; x and both outputs as float32 files for llm/jais_check.py, which shares no code with this.
static int check(int argc, char **argv) {
    DIR = argv[2]; std::string T = argv[3], out = argv[6]; int M = atoi(argv[4]), K = atoi(argv[5]);
    read_index();
    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)M * K)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, K * 4)); CK(cudaMalloc(&dxe, K * 2)); CK(cudaMalloc(&dy, M * 4)); CK(cudaMalloc(&dbad, 4));
    std::vector<float> x(K); for (int k = 0; k < K; k++) x[k] = bf(sin(0.37 * k + 1.0) * exp2(k % 29 - 14));   // tiny groups too
    DevW W = load_w(T + ".weight", F_E4M3B, M, K); to_dev(T + ".weight_scale", dS); W.sw = (uint8_t *)dS;
    Act a = act_fp8g(x); auto y8 = gemv(W, a, 8, 0), y24 = gemv(W, a, 24, 0);
    auto save = [&](const char *n, const std::vector<float> &v) { FILE *f = fopen((out + "/" + n).c_str(), "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f); };
    save("x.f32", x); save("y8.f32", y8); save("y24.f32", y24);
    FILE *f = fopen((out + "/meta.txt").c_str(), "w"); fprintf(f, "%s %d %d %lld\n", T.c_str(), M, K, st_bad); fclose(f);
    fprintf(stderr, "%s: %d x %d, uncertified %lld\n", T.c_str(), M, K, st_bad);
    return 0;
}

// DIR/index.txt and DIR/config.json -> shapes, RoPE; the layers present (OZ_LAYERS=n: the first n only) and which are MoE
static std::vector<bool> jais_init() {
    read_index();
    { std::ifstream f(DIR + "/config.json"); std::string s((std::istreambuf_iterator<char>(f)), {});
      D_H = cfgnum(s, "hidden_size", 0); D_Q = cfgnum(s, "num_attention_heads", 0); D_KV = cfgnum(s, "num_key_value_heads", D_Q);
      D_HD = cfgnum(s, "head_dim", D_H / D_Q); D_FF = cfgnum(s, "intermediate_size", 0); D_V = cfgnum(s, "vocab_size", 0);
      D_EPS = cfgnum(s, "rms_norm_eps", 1e-6); D_THETA = cfgnum(s, "rope_theta", 10000);
      J_MFF = cfgnum(s, "moe_intermediate_size", 0); J_E = cfgnum(s, "num_experts", 0); J_TOPK = cfgnum(s, "num_experts_per_tok", 0);
      J_RHD = cfgnum(s, "rope_head_dim", D_HD); J_RSF = cfgnum(s, "router_scaling_factor", 1); }
    NORM_ONCE = true; rope_init(J_RHD);
    int NL = 0; while (idx.count("model.layers." + std::to_string(NL) + ".self_attn.q_proj.weight")) NL++;
    if (const char *n = getenv("OZ_LAYERS")) NL = std::min(NL, atoi(n));   // the first n layers only (smoke tests)
    std::vector<bool> moe(NL); for (int L = 0; L < NL; L++) moe[L] = idx.count("model.layers." + std::to_string(L) + ".mlp.gate.weight");
    return moe;
}
// the top J_TOPK of sigmoid + bias (lowest id on a tie), weights BF16(J_RSF sigmoid / sum of the chosen sigmoids)
static void route(const std::vector<float> &lg, const std::vector<float> &bias, std::vector<int> &ex, std::vector<double> &wt) {
    std::vector<float> sg(J_E); std::vector<std::pair<float, int>> c(J_E);
    for (int e = 0; e < J_E; e++) { sg[e] = (float)(1.0 / (1.0 + exp(-(double)lg[e]))); c[e] = {sg[e] + bias[e], e}; }
    std::partial_sort(c.begin(), c.begin() + J_TOPK, c.end(), [](auto &a, auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    double sum = 0; for (int k = 0; k < J_TOPK; k++) sum += sg[c[k].second];
    ex.clear(); wt.clear();
    for (int k = 0; k < J_TOPK; k++) { ex.push_back(c[k].second); wt.push_back(bf(J_RSF * sg[c[k].second] / sum)); }
}

#ifndef JAIS_LIB
int main(int argc, char **argv) {
    if (argc > 6 && !strcmp(argv[1], "--check")) return check(argc, argv);
    if (argc < 4) { fprintf(stderr, "usage: out_jais <model dir> <prompts> <new tokens> [modes]\n"); return 1; }
    DIR = argv[1]; int NEW = atoi(argv[3]);
    std::vector<int> modes{0, 1, 4};
    if (argc > 4) { modes.clear(); std::stringstream ss(argv[4]); std::string t; while (std::getline(ss, t, ',')) modes.push_back(atoi(t.c_str())); }
    std::vector<bool> moe = jais_init(); int NL = moe.size();
    std::vector<OSeq> seqs;
    { std::ifstream f(argv[2]); std::string line; int pi = 0;
      while (std::getline(f, line)) { if (line.empty()) continue; std::stringstream ss(line); std::vector<int> t; int v; while (ss >> v) t.push_back(v);
        for (int m : modes) { OSeq s; s.prompt = pi; s.mode = m; s.toks = t; s.plen = t.size(); s.kc.resize(NL); s.vc.resize(NL); seqs.push_back(s); } pi++; } }
    size_t steps = 0; for (auto &s : seqs) steps = std::max(steps, s.plen + NEW);
    fprintf(stderr, "jais: %d layers (%d MoE), H %d, heads %d/%d, rope %d of %d, FF %d, experts %d top %d x %g of %d, vocab %d, theta %g, %zu sequences, %zu steps\n",
            NL, (int)std::count(moe.begin(), moe.end(), true), D_H, D_Q, D_KV, J_RHD, D_HD, D_FF, J_E, J_TOPK, J_RSF, J_MFF, D_V, D_THETA, seqs.size(), steps);

    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)D_FF * D_H * 2)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, 32768 * 4)); CK(cudaMalloc(&dxe, 32768 * 2)); CK(cudaMalloc(&dy, (size_t)D_V * 4)); CK(cudaMalloc(&dbad, 4));
    void *dLM; CK(cudaMalloc(&dLM, (size_t)D_V * D_H * 2));
    DevW LM = load_w("lm_head.weight", F_BF16, D_V, D_H, dLM);
    // a partly downloaded checkpoint runs as the truncated network of the layers present (a smoke test of every code path)
    bool whole = idx.count("model.norm.weight"); if (!whole) fprintf(stderr, "truncated checkpoint: final norm weight 1\n");
    auto normf = whole ? vec_bf16("model.norm.weight") : std::vector<float>(D_H, 1.f);
    std::vector<std::vector<float>> n1(NL), n2(NL), bias(NL);
    for (int L = 0; L < NL; L++) {
        std::string P = "model.layers." + std::to_string(L) + ".";
        n1[L] = vec_bf16(P + "input_layernorm.weight"); n2[L] = vec_bf16(P + "post_attention_layernorm.weight");
        if (moe[L]) bias[L] = vec_bf16(P + "mlp.gate.bias");
    }
    fprintf(stderr, "resident: lm_head + small tensors (%.1f GB read)\n", bytes_disk / 1e9);

    auto part0 = [&](int L) {   // a layer's tensors used before routing: all of them for a dense layer
        std::string P = "model.layers." + std::to_string(L) + ".";
        for (const char *n : {"q_proj", "k_proj", "v_proj", "o_proj"}) ahead.push(P + "self_attn." + n + ".weight");
        if (moe[L]) ahead.push(P + "mlp.gate.weight");
        else for (const char *n : {"gate_proj", "up_proj", "down_proj"}) ahead.push(P + "mlp." + n + ".weight");
    };
    auto next0 = [&](int L, size_t t) { if (L + 1 < NL) part0(L + 1); else if (t + 1 < steps) part0(0); };
    if (const char *a = getenv("OZ_AHEAD")) { ahead.start((size_t)(atof(a) * 1e9)); part0(0); fprintf(stderr, "read-ahead ring %.1f GB\n", atof(a)); }

    for (size_t t = 0; t < steps; t++) {
        auto t0 = std::chrono::steady_clock::now(); double d0 = t_disk, b0 = bytes_disk;
        for (auto &s : seqs) s.h = emb_row("model.embed_tokens.weight", s.toks[t], D_H);
        for (int L = 0; L < NL; L++) {
            std::string P = "model.layers." + std::to_string(L) + ".";
            size_t NS = seqs.size();
            auto mm = [&](const DevW &W, const Act &a, size_t j) { return gemv(W, a, 8, seqs[j].mode, seqs[j].mode == 0); };
            if (!moe[L]) next0(L, t);
            std::vector<std::vector<float>> x(NS), q(NS), o(NS, std::vector<float>(D_Q * D_HD)), out(NS);
            for (size_t j = 0; j < NS; j++) norm(seqs[j].h, n1[L], x[j]);
            DevW W = load_w(P + "self_attn.q_proj.weight", F_BF16, D_Q * D_HD, D_H);
            for (size_t j = 0; j < NS; j++) { q[j] = mm(W, act_plain(x[j]), j); rope_part(q[j], D_Q, t); }
            W = load_w(P + "self_attn.k_proj.weight", F_BF16, D_KV * D_HD, D_H);
            for (size_t j = 0; j < NS; j++) { auto k = mm(W, act_plain(x[j]), j); rope_part(k, D_KV, t); seqs[j].kc[L].insert(seqs[j].kc[L].end(), k.begin(), k.end()); }
            W = load_w(P + "self_attn.v_proj.weight", F_BF16, D_KV * D_HD, D_H);
            for (size_t j = 0; j < NS; j++) { auto v = mm(W, act_plain(x[j]), j); seqs[j].vc[L].insert(seqs[j].vc[L].end(), v.begin(), v.end()); }
            for (size_t j = 0; j < NS; j++) attend(q[j], seqs[j].kc[L], seqs[j].vc[L], D_Q, D_KV, D_HD, o[j]);
            W = load_w(P + "self_attn.o_proj.weight", F_BF16, D_H, D_Q * D_HD);
            for (size_t j = 0; j < NS; j++) { auto y = mm(W, act_plain(o[j]), j); for (int i = 0; i < D_H; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + y[i]); }
            for (size_t j = 0; j < NS; j++) norm(seqs[j].h, n2[L], x[j]);
            if (!moe[L]) {
                std::vector<std::vector<float>> g(NS);
                W = load_w(P + "mlp.gate_proj.weight", F_BF16, D_FF, D_H);
                for (size_t j = 0; j < NS; j++) g[j] = mm(W, act_plain(x[j]), j);
                W = load_w(P + "mlp.up_proj.weight", F_BF16, D_FF, D_H);
                for (size_t j = 0; j < NS; j++) g[j] = swiglu(g[j], mm(W, act_plain(x[j]), j));
                W = load_w(P + "mlp.down_proj.weight", F_BF16, D_H, D_FF);
                for (size_t j = 0; j < NS; j++) out[j] = mm(W, act_plain(g[j]), j);
            } else {
                W = load_w(P + "mlp.gate.weight", F_BF16, J_E, D_H);
                std::vector<std::vector<int>> ex(NS); std::vector<std::vector<double>> wt(NS);
                for (size_t j = 0; j < NS; j++) route(mm(W, act_plain(x[j]), j), bias[L], ex[j], wt[j]);
                std::set<int> uni; for (auto &v : ex) uni.insert(v.begin(), v.end());
                for (int e : uni) for (const char *n : {"gate_proj", "up_proj", "down_proj"}) for (const char *s : {".weight", ".weight_scale"})
                    ahead.push(P + "mlp.experts." + std::to_string(e) + "." + n + s);
                for (const char *n : {"gate_proj", "up_proj", "down_proj"}) ahead.push(P + "mlp.shared_experts." + n + ".weight");
                next0(L, t);
                std::vector<Act> ax(NS); for (size_t j = 0; j < NS; j++) ax[j] = act_fp8g(x[j]);
                std::vector<std::vector<double>> acc(NS, std::vector<double>(D_H, 0.0));
                auto fp8w = [&](const std::string &name, int M, int K) {
                    DevW Wf = load_w(name + ".weight", F_E4M3B, M, K); to_dev(name + ".weight_scale", dS); Wf.sw = (uint8_t *)dS; return Wf; };
                for (int e : uni) {   // each chosen expert read once, applied to every sequence that chose it
                    std::string EP = P + "mlp.experts." + std::to_string(e) + ".";
                    std::vector<size_t> js; for (size_t j = 0; j < NS; j++) if (std::count(ex[j].begin(), ex[j].end(), e)) js.push_back(j);
                    std::vector<std::vector<float>> g(NS);
                    W = fp8w(EP + "gate_proj", J_MFF, D_H); for (size_t j : js) g[j] = mm(W, ax[j], j);
                    W = fp8w(EP + "up_proj", J_MFF, D_H); for (size_t j : js) g[j] = swiglu(g[j], mm(W, ax[j], j));
                    W = fp8w(EP + "down_proj", D_H, J_MFF);
                    for (size_t j : js) {
                        auto d = mm(W, act_fp8g(g[j]), j);
                        double w = wt[j][std::find(ex[j].begin(), ex[j].end(), e) - ex[j].begin()];
                        for (int i = 0; i < D_H; i++) acc[j][i] += w * d[i];   // FP64 in expert order, rounded once below
                    }
                }
                std::vector<std::vector<float>> g(NS);
                W = load_w(P + "mlp.shared_experts.gate_proj.weight", F_BF16, J_MFF, D_H);
                for (size_t j = 0; j < NS; j++) g[j] = mm(W, act_plain(x[j]), j);
                W = load_w(P + "mlp.shared_experts.up_proj.weight", F_BF16, J_MFF, D_H);
                for (size_t j = 0; j < NS; j++) g[j] = swiglu(g[j], mm(W, act_plain(x[j]), j));
                W = load_w(P + "mlp.shared_experts.down_proj.weight", F_BF16, D_H, J_MFF);
                for (size_t j = 0; j < NS; j++) { auto d = mm(W, act_plain(g[j]), j); out[j].resize(D_H);
                    for (int i = 0; i < D_H; i++) out[j][i] = bf((double)bf(acc[j][i]) + d[i]); }
            }
            for (size_t j = 0; j < NS; j++) for (int i = 0; i < D_H; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + out[j][i]);
            for (auto &s : seqs) s.dig = fnv(s.dig, s.h);   // FNV-1a of every residual stream after every layer
        }
        for (auto &s : seqs) {
            std::vector<float> hn; norm(s.h, normf, hn);
            auto lg = gemv(LM, act_plain(hn), 24, s.mode, s.mode == 0); s.dig = fnv(s.dig, lg);
            int b1 = pick(lg, t, s.prompt, s.mode, s.toks[t]);
            if (t + 1 >= s.plen && t + 1 >= s.toks.size()) s.toks.push_back(b1);
        }
        fflush(stdout);
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "step %zu: %.1f s (disk %.1f s, %.1f GB, %.2f GB/s), uncertified so far %lld\n", t, sec, t_disk - d0,
                (bytes_disk - b0) / 1e9, (bytes_disk - b0) / 1e9 / std::max(1e-9, t_disk - d0), st_bad);
    }
    ahead.finish();
    report(seqs);
}
#endif
