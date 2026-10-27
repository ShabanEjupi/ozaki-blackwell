// olmo.cu -- OLMo-2-0325-32B-Instruct, the model with full access: weights Apache-2.0, every training token public
// (ODC-BY) and indexed by infini-gram (v4_olmo-2-0325-32b-instruct_llama, 4.6T tokens), so each generated span can be
// looked up in the data the model learned from (llm/olmo_trace.py). Token by token from disk with every linear map
// correctly rounded (the GEMV of ultra.cu, self-tested against a 2560-bit superaccumulator), against FP32 engines with
// split-K 1 and 4 on the same host code.
// Architecture as transformers v5.3.0 modeling_olmo2.py: post-norm blocks h += RMSNorm(attn(h)), h += RMSNorm(mlp(h));
// q_norm / k_norm over all heads after the projections, before RoPE (theta 500000, rotate-half); GQA 40 / 8 heads of 128;
// SwiGLU 27648; final RMSNorm, lm_head. Host parts in FP64, rounded once to BF16 (logits FP32).
// nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_olmo llm/olmo.cu -lgomp
// out_olmo <model dir with index.txt> <prompts file: token ids per line> <new tokens> [modes, e.g. 0,1,4]
#define ULTRA_LIB
#include "ultra.cu"

constexpr int OH = 5120, OQ = 40, OKV = 8, OD = 128, OFF = 27648, OV = 100352;
static void onorm(const std::vector<float> &x, const std::vector<float> &w, std::vector<float> &y) {   // (w * x/rms) -> BF16
    double ss = 0; for (float v : x) ss += (double)v * v;
    double r = 1.0 / sqrt(ss / x.size() + 1e-6);
    y.resize(x.size()); for (size_t i = 0; i < x.size(); i++) y[i] = bf((double)w[i] * ((double)x[i] * r));
}
static void rope(std::vector<float> &v, int heads, int pos) {   // rotate-half, inv_freq in FP32 as the reference builds it
    for (int h = 0; h < heads; h++) for (int i = 0; i < OD / 2; i++) {
        float inv = (float)(1.0 / pow(500000.0, (double)(2 * i) / OD));
        double a = (double)pos * inv, c = cos(a), s = sin(a), x1 = v[h * OD + i], x2 = v[h * OD + i + OD / 2];
        v[h * OD + i] = bf(x1 * c - x2 * s); v[h * OD + i + OD / 2] = bf(x2 * c + x1 * s);
    }
}
struct OSeq { int prompt, mode; std::vector<int> toks; size_t plen; std::vector<float> h; std::vector<std::vector<float>> kc, vc; };

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: out_olmo <model dir> <prompts> <new tokens> [modes]\n"); return 1; }
    DIR = argv[1]; int NEW = atoi(argv[3]);
    std::vector<int> modes{0, 1, 4};
    if (argc > 4) { modes.clear(); std::stringstream ss(argv[4]); std::string t; while (std::getline(ss, t, ',')) modes.push_back(atoi(t.c_str())); }
    { std::ifstream f(DIR + "/index.txt"); std::string nm, dt, sh, fl; size_t off, n;
      while (f >> nm >> dt >> sh >> fl >> off >> n) idx[nm] = {dt, fl, off, n}; }
    int NL = 0; while (idx.count("model.layers." + std::to_string(NL) + ".self_attn.q_proj.weight")) NL++;
    std::vector<OSeq> seqs;
    { std::ifstream f(argv[2]); std::string line; int pi = 0;
      while (std::getline(f, line)) { if (line.empty()) continue; std::stringstream ss(line); std::vector<int> t; int v; while (ss >> v) t.push_back(v);
        for (int m : modes) { OSeq s; s.prompt = pi; s.mode = m; s.toks = t; s.plen = t.size(); s.kc.resize(NL); s.vc.resize(NL); seqs.push_back(s); } pi++; } }
    size_t steps = 0; for (auto &s : seqs) steps = std::max(steps, s.plen + NEW);
    fprintf(stderr, "%zu tensors, %d layers, %zu sequences, %zu steps\n", idx.size(), NL, seqs.size(), steps);

    CK(cudaMallocHost(&pin, PIN)); CK(cudaMalloc(&dW, (size_t)300 << 20)); CK(cudaMalloc(&dS, (size_t)8 << 20));
    CK(cudaMalloc(&dxm, 32768 * 4)); CK(cudaMalloc(&dxe, 32768 * 2)); CK(cudaMalloc(&dy, VOC * 4)); CK(cudaMalloc(&dbad, 4));
    void *dLM; CK(cudaMalloc(&dLM, (size_t)OV * OH * 2));
    DevW LM = load_w("lm_head.weight", F_BF16, OV, OH, dLM);
    auto normf = vec_bf16("model.norm.weight");
    std::vector<std::vector<float>> qn(NL), kn(NL), pa(NL), pf(NL);
    for (int L = 0; L < NL; L++) {
        std::string P = "model.layers." + std::to_string(L) + ".";
        qn[L] = vec_bf16(P + "self_attn.q_norm.weight"); kn[L] = vec_bf16(P + "self_attn.k_norm.weight");
        pa[L] = vec_bf16(P + "post_attention_layernorm.weight"); pf[L] = vec_bf16(P + "post_feedforward_layernorm.weight");
    }
    for (size_t t = 0; t < steps; t++) {
        auto t0 = std::chrono::steady_clock::now(); double d0 = t_disk, b0 = bytes_disk;
        for (auto &s : seqs) {
            std::vector<uint16_t> row(OH); const TI e = idx.at("model.embed_tokens.weight"); TI r = e;
            r.off += (size_t)s.toks[t] * OH * 2; r.n = (size_t)OH * 2; idx["__row"] = r; readt("__row", row.data());
            s.h.resize(OH); for (int i = 0; i < OH; i++) s.h[i] = bf16f(row[i]);
        }
        for (int L = 0; L < NL; L++) {
            std::string P = "model.layers." + std::to_string(L) + ".";
            size_t NS = seqs.size();
            std::vector<std::vector<float>> q(NS), o(NS, std::vector<float>(OQ * OD)), tmp(NS);
            DevW W = load_w(P + "self_attn.q_proj.weight", F_BF16, OQ * OD, OH);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(seqs[j].h), 8, seqs[j].mode, seqs[j].mode == 0); onorm(y, qn[L], q[j]); rope(q[j], OQ, t); }
            W = load_w(P + "self_attn.k_proj.weight", F_BF16, OKV * OD, OH);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(seqs[j].h), 8, seqs[j].mode, seqs[j].mode == 0); std::vector<float> k; onorm(y, kn[L], k); rope(k, OKV, t);
                seqs[j].kc[L].insert(seqs[j].kc[L].end(), k.begin(), k.end()); }
            W = load_w(P + "self_attn.v_proj.weight", F_BF16, OKV * OD, OH);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(seqs[j].h), 8, seqs[j].mode, seqs[j].mode == 0); seqs[j].vc[L].insert(seqs[j].vc[L].end(), y.begin(), y.end()); }
            for (size_t j = 0; j < NS; j++) {
                auto &K = seqs[j].kc[L], &V = seqs[j].vc[L]; size_t T = K.size() / (OKV * OD);
                #pragma omp parallel for schedule(static)
                for (int hq = 0; hq < OQ; hq++) {
                    int g = hq / (OQ / OKV); std::vector<double> sc(T); double mx = -1e300;
                    for (size_t u = 0; u < T; u++) { double a = 0; for (int d = 0; d < OD; d++) a += (double)q[j][hq * OD + d] * K[(u * OKV + g) * OD + d];
                        sc[u] = a / sqrt((double)OD); mx = std::max(mx, sc[u]); }
                    double den = 0; for (auto &v : sc) { v = exp(v - mx); den += v; }
                    for (int d = 0; d < OD; d++) { double a = 0; for (size_t u = 0; u < T; u++) a += sc[u] * V[(u * OKV + g) * OD + d]; o[j][hq * OD + d] = bf(a / den); }
                }
            }
            W = load_w(P + "self_attn.o_proj.weight", F_BF16, OH, OQ * OD);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(o[j]), 8, seqs[j].mode, seqs[j].mode == 0); std::vector<float> n; onorm(y, pa[L], n);
                for (int i = 0; i < OH; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + n[i]); }
            W = load_w(P + "mlp.gate_proj.weight", F_BF16, OFF, OH);
            for (size_t j = 0; j < NS; j++) tmp[j] = gemv(W, act_plain(seqs[j].h), 8, seqs[j].mode, seqs[j].mode == 0);
            W = load_w(P + "mlp.up_proj.weight", F_BF16, OFF, OH);
            for (size_t j = 0; j < NS; j++) { auto u = gemv(W, act_plain(seqs[j].h), 8, seqs[j].mode, seqs[j].mode == 0);
                for (int i = 0; i < OFF; i++) { double gv = tmp[j][i]; tmp[j][i] = bf(gv / (1.0 + exp(-gv)) * u[i]); } }
            W = load_w(P + "mlp.down_proj.weight", F_BF16, OH, OFF);
            for (size_t j = 0; j < NS; j++) { auto y = gemv(W, act_plain(tmp[j]), 8, seqs[j].mode, seqs[j].mode == 0); std::vector<float> n; onorm(y, pf[L], n);
                for (int i = 0; i < OH; i++) seqs[j].h[i] = bf((double)seqs[j].h[i] + n[i]); }
        }
        for (auto &s : seqs) {
            std::vector<float> hn; onorm(s.h, normf, hn);
            auto lg = gemv(LM, act_plain(hn), 24, s.mode, s.mode == 0);
            int b1 = 0; for (int v = 1; v < OV; v++) if (lg[v] > lg[b1]) b1 = v;
            int b2 = b1 ? 0 : 1; for (int v = 0; v < OV; v++) if (v != b1 && lg[v] > lg[b2]) b2 = v;
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
