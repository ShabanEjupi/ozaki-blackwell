// moe.cu -- one whole MoE block of Kimi K3 (layer 5, real weights), computed exactly and computed as engines do.
// Block (modeling_kimi_linear.py, KimiSparseMoeBlock): router logits = x Wg^T (FP32) -> sigmoid + bias -> top 16 of 896,
// weights renormalized; latent = down(x) (7168 -> 3584); per expert e: w2(situ(w1 l, w3 l)) (MXFP4, 3584 -> 3072 -> 3584);
// y = sum_k weight_k out_k; RMSNorm (eps 1e-5) * weight; up (3584 -> 7168); + shared(x) = down(situ(gate x, up x)) (BF16,
// 7168 -> 6144 -> 7168). situ(g, u) = 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25).
// Exact path: every matrix product correctly rounded (llm/exact.cuh) to BF16 (the router to FP32); the weighted sum and the
// RMSNorm in FP64 in a fixed order, rounded once; the activations in FP64, rounded once. Nothing depends on the batch.
// Engine path: BF16 tensor cores, FP32 accumulation, split-K s, results cast to BF16; float reductions and activations.
// Experts are fetched on demand: exit code 3 + "NEED e1 e2 ..." when the routing wants experts not on disk.
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_moe llm/moe.cu ; out_moe <dir> <tokens> [seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include "exact.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

typedef std::vector<uint16_t> BF;   // BF16 bits
static float b2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2b(double d) {   // round to nearest even BF16 (through float: exact when d already is a float)
    float f = (float)d; uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0;
    return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
template <class T> static std::vector<T> slurp(const std::string &p, size_t n) {
    std::vector<T> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f || fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}

// scratch for one product (tokens <= 128)
static const int NP = 128, KMAX = 7168, MMAX = 7168, ZMAX = 144;
static __nv_bfloat16 *gX; static ex::Digits gXd; static int32_t *gP; static float *gY, *gPart; static int *gStat, *gBad;

struct Lin { int M, K; ex::Digits d; __nv_bfloat16 *wb; };
static Lin lin_bf16(const BF &w, int M, int K) {
    Lin L{M, K, {}, nullptr};
    CK(cudaMalloc(&L.wb, (size_t)M * K * 2)); CK(cudaMemcpy(L.wb, w.data(), (size_t)M * K * 2, cudaMemcpyHostToDevice));
    L.d.S = 4; L.d.rows = M; L.d.K = K;
    CK(cudaMalloc(&L.d.dig, (size_t)2 * 4 * M * K)); CK(cudaMalloc(&L.d.B, ex::TMAX * M * 4));
    CK(cudaMemset(gStat, 0, 12));
    ex::split_rows<<<M, 256>>>(L.wb, L.d.dig, L.d.B, gStat, M, K, 4, 2);   // weights: 2 tiers (more = left over: checked)
    int st[3]; CK(cudaMemcpy(st, gStat, 12, cudaMemcpyDeviceToHost));
    if (st[0] > 2 || st[1]) { fprintf(stderr, "weight tiers %d, left %d\n", st[0], st[1]); exit(1); }
    L.d.T = st[0];
    return L;
}
static Lin lin_mx(const std::vector<uint8_t> &pk, const std::vector<uint8_t> &sc, int M, int K) {
    Lin L{M, K, {}, nullptr};
    const int G = K / 32;
    int A = 1;
    for (int i = 0; i < M; i++) { int a = 255, b = 0; for (int g = 0; g < G; g++) { a = std::min(a, (int)sc[(size_t)i * G + g]); b = std::max(b, (int)sc[(size_t)i * G + g]); } A = std::max(A, (b - a) / 4 + 1); }
    uint8_t *dpk, *dsc; CK(cudaMalloc(&dpk, pk.size())); CK(cudaMalloc(&dsc, sc.size()));
    CK(cudaMemcpy(dpk, pk.data(), pk.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsc, sc.data(), sc.size(), cudaMemcpyHostToDevice));
    L.d.S = 1; L.d.T = A; L.d.rows = M; L.d.K = K;
    CK(cudaMalloc(&L.d.dig, (size_t)A * M * K)); CK(cudaMalloc(&L.d.B, A * M * 4));
    ex::mx_digits<<<M, 256>>>(dpk, dsc, L.d.dig, L.d.B, M, K, A);
    BF wb((size_t)M * K);   // the engine's BF16 copy: E2M1 * 2^(s - 127) is exact in BF16
    static const float e2m1[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (size_t t = 0; t < wb.size(); t++) {
        size_t i = t / K, k = t % K; uint8_t b = pk[t / 2]; int nb = k & 1 ? b >> 4 : b & 15;
        wb[t] = f2b((nb & 8 ? -1 : 1) * ldexp((double)e2m1[nb & 7], (int)sc[i * G + k / 32] - 127));
    }
    CK(cudaMalloc(&L.wb, wb.size() * 2)); CK(cudaMemcpy(L.wb, wb.data(), wb.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaFree(dpk)); CK(cudaFree(dsc));
    return L;
}
static void lin_free(Lin &L) { cudaFree(L.d.dig); cudaFree(L.d.B); cudaFree(L.wb); }

// Y [N][M] = X W^T: exact (split = 0, rounded to p bits) or the engine's (split-K s, FP32)
static std::vector<float> prod(const Lin &W, const BF &X, int N, int split, int p = 8) {
    if (N > NP || W.K > KMAX || W.M > MMAX) { fprintf(stderr, "prod: size\n"); exit(1); }
    CK(cudaMemset(gX, 0, (size_t)NP * W.K * 2)); CK(cudaMemcpy(gX, X.data(), (size_t)N * W.K * 2, cudaMemcpyHostToDevice));
    if (!split) {
        ex::Digits Xd = gXd; Xd.rows = NP; Xd.K = W.K; Xd.S = 4;
        CK(cudaMemset(gStat, 0, 12));
        ex::split_rows<<<NP, 256>>>(gX, Xd.dig, Xd.B, gStat, NP, W.K, 4);
        int st[3]; CK(cudaMemcpy(st, gStat, 12, cudaMemcpyDeviceToHost));
        Xd.T = std::max(st[0], 1);
        const int Z = W.d.T * W.d.S * Xd.T * Xd.S;
        if (Z > ZMAX || st[1]) { fprintf(stderr, "prod: Z %d left %d\n", Z, st[1]); exit(1); }
        ex::gemm_pairs<<<dim3(NP / oz::BN, W.M / oz::BM, Z), oz::THREADS>>>(W.d, Xd, gP, Xd.T);
        ex::combine<<<dim3((W.M + 255) / 256, NP), 256>>>(gP, W.d, Xd, W.d.T, Xd.T, p, gY, W.M, gBad);
    } else {
        ex::bf16_gemm<<<dim3(NP / ex::BN, W.M / ex::BM, split), ex::THREADS>>>(W.wb, gX, gPart, W.M, NP, W.K, split);
        ex::reduce_k<<<512, 256>>>(gPart, gY, (size_t)NP * W.M, split);
    }
    CK(cudaGetLastError());
    std::vector<float> Y((size_t)N * W.M);
    CK(cudaMemcpy(Y.data(), gY, Y.size() * 4, cudaMemcpyDeviceToHost));
    return Y;
}
static BF tobf(const std::vector<float> &v) { BF o(v.size()); for (size_t i = 0; i < v.size(); i++) o[i] = f2b(v[i]); return o; }

static double situ(double g, double u) { return 4.0 * tanh(g / 4.0) / (1.0 + exp(-g)) * 25.0 * tanh(u / 25.0); }
static float situf(float g, float u) { return 4.f * tanhf(g / 4.f) * (1.f / (1.f + expf(-g))) * (25.f * tanhf(u / 25.f)); }

struct Route { std::vector<int> e; std::vector<float> w; };   // per token: experts (by score), weights
static std::vector<Route> route(const std::vector<float> &lg, const std::vector<float> &bias, int N) {
    std::vector<Route> R(N);
    for (int j = 0; j < N; j++) {
        std::vector<std::pair<float, int>> c(896); std::vector<float> s(896);
        for (int e = 0; e < 896; e++) { s[e] = 1.f / (1.f + expf(-lg[(size_t)j * 896 + e])); c[e] = {s[e] + bias[e], e}; }
        std::partial_sort(c.begin(), c.begin() + 16, c.end(), [](auto &a, auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        float den = 0; for (int t = 0; t < 16; t++) den += s[c[t].second];
        for (int t = 0; t < 16; t++) { R[j].e.push_back(c[t].second); R[j].w.push_back(s[c[t].second] / (den + 1e-20f)); }
    }
    return R;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dir> <tokens> [seed]\n", argv[0]); return 1; }
    const std::string dir = argv[1], pre = dir + "/language_model.model.layers.5.";
    const int T = atoi(argv[2]), seed = argc > 3 ? atoi(argv[3]) : 1, H = 7168, HL = 3584, FI = 3072, FS = 6144;
    CK(cudaMalloc(&gX, (size_t)NP * KMAX * 2)); CK(cudaMalloc(&gXd.dig, (size_t)ex::TMAX * 4 * NP * KMAX)); CK(cudaMalloc(&gXd.B, ex::TMAX * NP * 4));
    CK(cudaMalloc(&gP, (size_t)ZMAX * MMAX * NP * 4)); CK(cudaMalloc(&gY, (size_t)NP * MMAX * 4)); CK(cudaMalloc(&gPart, (size_t)8 * NP * MMAX * 4));
    CK(cudaMalloc(&gStat, 12)); CK(cudaMalloc(&gBad, 4)); CK(cudaMemset(gBad, 0, 4));

    // input: Student-t hidden states through RMSNorm with the real post_attention_layernorm weight
    auto lnw = slurp<uint16_t>(pre + "post_attention_layernorm.weight.bin", H);
    std::mt19937_64 rng(seed); std::student_t_distribution<double> st(3.0);
    BF x((size_t)T * H);
    for (int j = 0; j < T; j++) {
        std::vector<double> h(H); double ss = 0;
        for (int k = 0; k < H; k++) { h[k] = st(rng); ss += h[k] * h[k]; }
        const double r = 1.0 / sqrt(ss / H + 1e-5);
        for (int k = 0; k < H; k++) x[(size_t)j * H + k] = f2b(b2f(lnw[k]) * b2f(f2b(h[k] * r)));
    }
    auto bias = slurp<float>(pre + "block_sparse_moe.gate.e_score_correction_bias.bin", 896);
    Lin gate = lin_bf16(slurp<uint16_t>(pre + "block_sparse_moe.gate.weight.bin", (size_t)896 * H), 896, H);

    // routing for every variant first: which experts are needed
    const int VAR[3] = {0, 1, 4};   // exact, engine split-K 1, engine split-K 4
    std::vector<Route> R[3];
    for (int v = 0; v < 3; v++) R[v] = route(prod(gate, x, T, VAR[v], 24), bias, T);
    std::set<int> need;
    for (int v = 0; v < 3; v++) for (auto &r : R[v]) need.insert(r.e.begin(), r.e.end());
    std::string miss;
    for (int e : need) {
        struct stat sb;
        if (stat((pre + "block_sparse_moe.experts." + std::to_string(e) + ".w2.weight_scale.bin").c_str(), &sb)) miss += " " + std::to_string(e);
    }
    if (!miss.empty()) { printf("NEED%s\n", miss.c_str()); return 3; }
    int flips = 0; for (int j = 0; j < T; j++) { auto a = R[1][j].e, b = R[2][j].e; std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); flips += a != b; }
    printf("layer 5 MoE block, %d tokens, %zu distinct experts; tokens whose expert set differs between split-K 1 and 4: %d\n", T, need.size(), flips);

    Lin down = lin_bf16(slurp<uint16_t>(pre + "block_sparse_moe.routed_expert_down_proj.weight.bin", (size_t)HL * H), HL, H);
    Lin up = lin_bf16(slurp<uint16_t>(pre + "block_sparse_moe.routed_expert_up_proj.weight.bin", (size_t)H * HL), H, HL);
    Lin sg = lin_bf16(slurp<uint16_t>(pre + "block_sparse_moe.shared_experts.gate_proj.weight.bin", (size_t)FS * H), FS, H);
    Lin su = lin_bf16(slurp<uint16_t>(pre + "block_sparse_moe.shared_experts.up_proj.weight.bin", (size_t)FS * H), FS, H);
    Lin sd = lin_bf16(slurp<uint16_t>(pre + "block_sparse_moe.shared_experts.down_proj.weight.bin", (size_t)H * FS), H, FS);
    auto nw = slurp<uint16_t>(pre + "block_sparse_moe.routed_expert_norm.weight.bin", HL);

    // tok: the tokens of this run (a subset of 0..T-1: the batch), out[v][j] = the block's output for token j
    auto run = [&](int v, const std::vector<int> &tok) {
        const int n = tok.size(), s = VAR[v];
        BF xb((size_t)n * H); for (int a = 0; a < n; a++) std::copy(x.begin() + (size_t)tok[a] * H, x.begin() + (size_t)(tok[a] + 1) * H, xb.begin() + (size_t)a * H);
        std::vector<Route> Rv = route(prod(gate, xb, n, s, 24), bias, n);
        BF lat = tobf(prod(down, xb, n, s));
        std::vector<double> acc((size_t)n * HL, 0.0); std::vector<float> accf((size_t)n * HL, 0.f);
        std::map<int, std::vector<std::pair<int, float>>> by;   // expert -> (row in batch, weight)
        for (int a = 0; a < n; a++) for (int k = 0; k < 16; k++) by[Rv[a].e[k]].push_back({a, Rv[a].w[k]});
        // exact: per token, the 16 terms added in expert order (fixed); engine: float, in routing order like the reference
        std::vector<std::vector<std::pair<int, std::vector<uint16_t>>>> outs(n);
        for (auto &[e, list] : by) {
            const std::string ep = pre + "block_sparse_moe.experts." + std::to_string(e) + ".";
            Lin w1 = lin_mx(slurp<uint8_t>(ep + "w1.weight_packed.bin", (size_t)FI * HL / 2), slurp<uint8_t>(ep + "w1.weight_scale.bin", (size_t)FI * HL / 32), FI, HL);
            Lin w3 = lin_mx(slurp<uint8_t>(ep + "w3.weight_packed.bin", (size_t)FI * HL / 2), slurp<uint8_t>(ep + "w3.weight_scale.bin", (size_t)FI * HL / 32), FI, HL);
            Lin w2 = lin_mx(slurp<uint8_t>(ep + "w2.weight_packed.bin", (size_t)HL * FI / 2), slurp<uint8_t>(ep + "w2.weight_scale.bin", (size_t)HL * FI / 32), HL, FI);
            const int m = list.size();
            BF le((size_t)m * HL); for (int a = 0; a < m; a++) std::copy(lat.begin() + (size_t)list[a].first * HL, lat.begin() + (size_t)(list[a].first + 1) * HL, le.begin() + (size_t)a * HL);
            BF g = tobf(prod(w1, le, m, s)), u = tobf(prod(w3, le, m, s)), act((size_t)m * FI);
            for (size_t t = 0; t < act.size(); t++) act[t] = s ? f2b(situf(b2f(g[t]), b2f(u[t]))) : f2b(situ(b2f(g[t]), b2f(u[t])));
            BF o = tobf(prod(w2, act, m, s));
            for (int a = 0; a < m; a++) {
                const int r = list[a].first; const float w = list[a].second;
                for (int c = 0; c < HL; c++) {
                    if (s) accf[(size_t)r * HL + c] += w * b2f(o[(size_t)a * HL + c]);
                    else acc[(size_t)r * HL + c] += (double)w * b2f(o[(size_t)a * HL + c]);   // w (24 bits) x bf16 (8): exact in FP64
                }
            }
            lin_free(w1); lin_free(w3); lin_free(w2);
        }
        BF y((size_t)n * HL);
        for (int a = 0; a < n; a++) {   // bf16(sum), RMSNorm, * weight
            double ss = 0; float ssf = 0;
            BF ys(HL);
            for (int c = 0; c < HL; c++) { ys[c] = f2b(s ? accf[(size_t)a * HL + c] : acc[(size_t)a * HL + c]); const double vv = b2f(ys[c]); ss += vv * vv; ssf += (float)(vv * vv); }
            const double r = s ? (double)(1.f / sqrtf(ssf / HL + 1e-5f)) : 1.0 / sqrt(ss / HL + 1e-5);
            for (int c = 0; c < HL; c++) y[(size_t)a * HL + c] = f2b((double)b2f(nw[c]) * b2f(f2b(b2f(ys[c]) * r)));
        }
        BF yu = tobf(prod(up, y, n, s));
        BF g = tobf(prod(sg, xb, n, s)), u = tobf(prod(su, xb, n, s)), act((size_t)n * FS);
        for (size_t t = 0; t < act.size(); t++) act[t] = s ? f2b(situf(b2f(g[t]), b2f(u[t]))) : f2b(situ(b2f(g[t]), b2f(u[t])));
        BF so = tobf(prod(sd, act, n, s)), out((size_t)n * H);
        for (size_t t = 0; t < out.size(); t++) out[t] = f2b((double)b2f(yu[t]) + b2f(so[t]));
        return out;
    };
    std::vector<int> all(T); for (int j = 0; j < T; j++) all[j] = j;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<BF> full(3);
    for (int v = 0; v < 3; v++) {
        cudaEventRecord(e0);
        full[v] = run(v, all);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("  %s: whole block for %d tokens in %.0f ms\n", v ? (v == 1 ? "engine split-K 1" : "engine split-K 4") : "exact", T, ms);
    }
    int bad; CK(cudaMemcpy(&bad, gBad, 4, cudaMemcpyDeviceToHost));
    auto diff = [&](const BF &a, const BF &b) { size_t d = 0; for (size_t t = 0; t < a.size(); t++) d += a[t] != b[t]; return d; };
    const size_t tot = (size_t)T * H;
    printf("  exact products not certified: %d\n", bad);
    printf("  engine split-K 1 vs split-K 4: %zu of %zu outputs differ (%.1f %%)\n", diff(full[1], full[2]), tot, 100.0 * diff(full[1], full[2]) / tot);
    printf("  engine split-K 1 vs exact: %zu of %zu differ (%.1f %%)\n", diff(full[1], full[0]), tot, 100.0 * diff(full[1], full[0]) / tot);
    // batch invariance: every token alone through the exact block must give the same bits as in the batch
    size_t inv = 0;
    for (int j = 0; j < T; j++) {
        BF o = run(0, {j});
        for (int c = 0; c < H; c++) inv += o[c] != full[0][(size_t)j * H + c];
    }
    printf("  exact, each token alone vs in the batch of %d: %zu outputs differ\n", T, inv);
    return 0;
}
