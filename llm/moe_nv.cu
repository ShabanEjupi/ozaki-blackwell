// moe_nv.cu -- one whole routed MoE block of Nemotron 3 Ultra, real weights, exact versus as an engine computes it.
// Exact path: router logits correctly rounded (the FP32 gate as three exact BF16 matrices, llm/router_nv.cu), top-22 by
// sigmoid + e_score_correction_bias, weights normalized and times routed_scaling_factor 5; fc1_latent_proj correctly rounded
// to BF16 (llm/exact.cuh); every chosen expert up_proj -> relu^2 -> down_proj with correctly rounded NVFP4 W4A4 products
// (llm/nvexact.cu); the weighted sum in FP64 in expert order, rounded once to BF16; fc2_latent_proj correctly rounded to BF16.
// Engine path: router in FP32 FMA, latent projections on BF16 tensor cores with FP32 accumulation, experts on the native
// NVFP4 tensor-core instruction, weighted sum in FP32, every product with split-K s (1 or 4).
// Not covered: the shared expert (FP8 W8A8) runs beside this block and is added to its output; sigmoid (FP32 expf, the
// same code on both paths).
// nvcc -O3 -arch=sm_120a -std=c++17 -o out_moe_nv llm/moe_nv.cu ;  out_moe_nv /opt/ozaki/nv/L49 49 <N tokens> [seed]
#include "exact.cuh"
#define NV_NO_MAIN
#include "nvexact.cu"
#include <algorithm>
#include <map>
#include <set>

constexpr int E = 512, H = 8192, LAT = 2048, FF = 5120, TOPK = 22, NP = 128, S = 4;
static std::string D, PRE;   // weights folder, "backbone.layers.<L>.mixer."

// W [M][K] BF16 held as digits once; run(X [NP][K]) -> Y [NP][M] correctly rounded to p bits (24 FP32, 8 BF16)
struct ExactGemm {
    ex::Digits W{}; int M = 0, K = 0;
    void init(const __nv_bfloat16 *hW, int M_, int K_) {
        M = M_; K = K_; W.S = S; W.rows = M; W.K = K;
        __nv_bfloat16 *dW; int *stat, sw[3];
        CK(cudaMalloc(&dW, (size_t)M * K * 2)); CK(cudaMalloc(&stat, 12)); CK(cudaMemset(stat, 0, 12));
        CK(cudaMemcpy(dW, hW, (size_t)M * K * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&W.dig, (size_t)ex::TMAX * S * M * K)); CK(cudaMalloc(&W.B, ex::TMAX * M * 4));
        ex::split_rows<<<M, 256>>>(dW, W.dig, W.B, stat, M, K, S);
        CK(cudaMemcpy(sw, stat, 12, cudaMemcpyDeviceToHost)); W.T = sw[0];
        if (sw[1]) { fprintf(stderr, "weights beyond %d tiers\n", ex::TMAX); exit(1); }
        CK(cudaFree(dW)); CK(cudaFree(stat));
    }
    int run(const __nv_bfloat16 *dX, int p, float *dY) {
        ex::Digits X{}; X.S = S; X.rows = NP; X.K = K;
        int *stat, sx[3], *nbad, nb; int32_t *P;
        CK(cudaMalloc(&stat, 12)); CK(cudaMalloc(&nbad, 4)); CK(cudaMemset(stat, 0, 12)); CK(cudaMemset(nbad, 0, 4));
        CK(cudaMalloc(&X.dig, (size_t)ex::TMAX * S * NP * K)); CK(cudaMalloc(&X.B, ex::TMAX * NP * 4));
        ex::split_rows<<<NP, 256>>>(dX, X.dig, X.B, stat, NP, K, S);
        CK(cudaMemcpy(sx, stat, 12, cudaMemcpyDeviceToHost)); X.T = sx[0];
        if (sx[1]) { fprintf(stderr, "activations beyond %d tiers\n", ex::TMAX); exit(1); }
        int Z = W.T * S * X.T * S;
        CK(cudaMalloc(&P, (size_t)Z * M * NP * 4));
        ex::gemm_pairs<<<dim3(NP / oz::BN, M / oz::BM, Z), oz::THREADS>>>(W, X, P, X.T);
        ex::combine<<<dim3((M + 255) / 256, NP), 256>>>(P, W, X, W.T, X.T, p, dY, M, nbad);
        CK(cudaMemcpy(&nb, nbad, 4, cudaMemcpyDeviceToHost));
        for (void *q : {(void *)stat, (void *)nbad, (void *)X.dig, (void *)X.B, (void *)P}) CK(cudaFree(q));
        return nb;
    }
};
// engine: BF16 tensor cores, FP32 accumulation, split-K s
static void engine_gemm(const __nv_bfloat16 *dW, const __nv_bfloat16 *dX, int M, int K, int s, float *dY) {
    float *part; CK(cudaMalloc(&part, (size_t)s * NP * M * 4));
    ex::bf16_gemm<<<dim3(NP / ex::BN, M / ex::BM, s), ex::THREADS>>>(dW, dX, part, M, NP, K, s);
    ex::reduce_k<<<512, 256>>>(part, dY, (size_t)NP * M, s);
    CK(cudaDeviceSynchronize()); CK(cudaFree(part));
}
__global__ void fp32_router(const float *W, const float *X, float *Y, int s) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= (size_t)E * NP) return;
    int e = o % E, j = o / E, per = H / s;
    float tot = 0.f;
    for (int z = 0; z < s; z++) {
        float acc = 0.f;
        for (int k = z * per; k < (z + 1) * per; k++) acc = fmaf(W[(size_t)e * H + k], X[(size_t)j * H + k], acc);
        tot += acc;
    }
    Y[o] = tot;
}
__global__ void to_bf16(const float *Y, __nv_bfloat16 *X, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) X[i] = __float2bfloat16(Y[i]);
}

// one routed expert, NVFP4 W4A4, on all NP tokens: exact (nv_exact2) or native (split s)
struct Proj { uint8_t *pk, *sw, *swT; int8_t *qw; float gW, gA; int M, K; };
static Proj load_proj(const std::string &pre, int M, int K) {
    Proj p{}; p.M = M; p.K = K; int B = K / 16;
    auto pk = load(D + "/" + pre + ".weight.bin", (size_t)M * K / 2), sw = load(D + "/" + pre + ".weight_scale.bin", (size_t)M * B);
    p.gW = loadf(D + "/" + pre + ".weight_scale_2.bin"); p.gA = loadf(D + "/" + pre + ".input_scale.bin");
    CK(cudaMalloc(&p.pk, pk.size())); CK(cudaMalloc(&p.sw, sw.size())); CK(cudaMalloc(&p.swT, sw.size())); CK(cudaMalloc(&p.qw, (size_t)M * K));
    CK(cudaMemcpy(p.pk, pk.data(), pk.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(p.sw, sw.data(), sw.size(), cudaMemcpyHostToDevice));
    unpack_w<<<1024, 256>>>(p.pk, p.qw, pk.size()); transpose_u8<<<1024, 256>>>(p.sw, p.swT, M, B);
    return p;
}
static void run_proj(const Proj &p, const __nv_bfloat16 *dX, bool exact, int s, float *dY) {
    int B = p.K / 16; int8_t *qa; uint8_t *sa, *saT, *pa;
    CK(cudaMalloc(&qa, (size_t)NP * p.K)); CK(cudaMalloc(&sa, (size_t)NP * B)); CK(cudaMalloc(&saT, (size_t)NP * B)); CK(cudaMalloc(&pa, (size_t)NP * p.K / 2));
    quant_x<<<1024, 256>>>(dX, qa, sa, p.gA, (size_t)NP * B);
    if (exact) {
        int xW, xA; float fW = frexpf(p.gW, &xW), fA = frexpf(p.gA, &xA);
        long long Gm = (long long)ldexpf(fW, 24) * (long long)ldexpf(fA, 24);
        transpose_u8<<<1024, 256>>>(sa, saT, NP, B);
        nv_exact2<<<dim3(p.M / TBM, NP / TBN), 256>>>(p.qw, p.swT, qa, saT, Gm, xW + xA - 48, dY, p.M, NP, p.K, NP);
    } else {
        pack_x<<<1024, 256>>>(qa, pa, (size_t)NP * p.K / 2);
        nv_native<<<dim3((p.M + 127) / 128, NP / 32), 256>>>(p.pk, p.sw, pa, sa, p.gW * p.gA, dY, p.M, NP, p.K, s);
    }
    CK(cudaDeviceSynchronize());
    for (void *q : {(void *)qa, (void *)sa, (void *)saT, (void *)pa}) CK(cudaFree(q));
}
static std::map<int, std::pair<Proj, Proj>> experts;
static void need(const std::set<int> &want) {
    std::string re;
    for (int e : want) if (!experts.count(e)) {
        bool have = true;   // both weight files at full size (a scale file alone may belong to a fetch still running)
        for (const char *w : {".up_proj.weight.bin", ".down_proj.weight.bin"}) {
            FILE *f = fopen((D + "/" + PRE + "experts." + std::to_string(e) + w).c_str(), "rb");
            if (!f) { have = false; break; }
            fseek(f, 0, SEEK_END); have = have && ftell(f) == (long)FF * LAT / 2; fclose(f);
        }
        if (have) continue;
        re += (re.empty() ? "" : "|") + std::to_string(e);
    }
    if (!re.empty()) {
        int L = atoi(PRE.c_str() + strlen("backbone.layers."));
        char shard[64]; snprintf(shard, sizeof shard, "model-%05d-of-00113.safetensors", L + 1);
        std::string cmd = "HF_REPO=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 python3 -I llm/k3_fetch.py " + std::string(shard) +
                          " '^backbone\\.layers\\." + std::to_string(L) + "\\.mixer\\.experts\\.(" + re + ")\\.' " + D + " > /dev/null";
        if (system(cmd.c_str())) { fprintf(stderr, "fetch failed: %s\n", cmd.c_str()); exit(1); }
    }
    for (int e : want) if (!experts.count(e)) {
        std::string p = PRE + "experts." + std::to_string(e);
        experts[e] = {load_proj(p + ".up_proj", FF, LAT), load_proj(p + ".down_proj", LAT, FF)};
    }
}
// expert outputs on all NP tokens (FP32 [NP][LAT]), cached per (expert, path)
static std::map<std::pair<int, int>, std::vector<float>> yexp;
static const std::vector<float> &expert_out(int e, const __nv_bfloat16 *dLat, int mode) {   // mode 0 exact, s > 0 native split s
    auto key = std::make_pair(e, mode);
    if (yexp.count(key)) return yexp[key];
    float *y1, *y2; __nv_bfloat16 *h; CK(cudaMalloc(&y1, (size_t)NP * FF * 4)); CK(cudaMalloc(&y2, (size_t)NP * LAT * 4)); CK(cudaMalloc(&h, (size_t)NP * FF * 2));
    run_proj(experts[e].first, dLat, mode == 0, mode, y1);
    relu2<<<1024, 256>>>(y1, h, (size_t)NP * FF);
    run_proj(experts[e].second, h, mode == 0, mode, y2);
    std::vector<float> v((size_t)NP * LAT); CK(cudaMemcpy(v.data(), y2, v.size() * 4, cudaMemcpyDeviceToHost));
    for (void *q : {(void *)y1, (void *)y2, (void *)h}) CK(cudaFree(q));
    return yexp[key] = v;
}
struct Route { std::vector<int> e; std::vector<double> w; };
static std::vector<Route> route(const std::vector<float> &lg, const std::vector<float> &bias, int N) {
    std::vector<Route> out(N);
    std::vector<std::pair<float, int>> c(E);
    for (int j = 0; j < N; j++) {
        std::vector<float> sg(E);
        for (int e = 0; e < E; e++) { sg[e] = 1.f / (1.f + expf(-lg[(size_t)j * E + e])); c[e] = {sg[e] + bias[e], e}; }
        std::partial_sort(c.begin(), c.begin() + TOPK, c.end(), [](auto &a, auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        double sum = 0; for (int t = 0; t < TOPK; t++) sum += sg[c[t].second];
        for (int t = 0; t < TOPK; t++) out[j].e.push_back(c[t].second);
        std::sort(out[j].e.begin(), out[j].e.end());
        for (int e : out[j].e) out[j].w.push_back(5.0 * sg[e] / sum);
    }
    return out;
}

// the block for tokens x [N][H] (padded to NP): exact (s = 0) or engine (split s); returns BF16 output [N][H] and routes
struct Gates { std::vector<float> Wg, bias; ExactGemm g3, fc1, fc2; __nv_bfloat16 *dfc1, *dfc2; float *dWg; };
static std::vector<__nv_bfloat16> block(Gates &G, const std::vector<__nv_bfloat16> &x, int N, int s, std::vector<Route> &R, int &nbad) {
    nbad = 0;
    std::vector<__nv_bfloat16> xp((size_t)NP * H, __float2bfloat16(0.f)); std::copy(x.begin(), x.begin() + (size_t)N * H, xp.begin());
    __nv_bfloat16 *dx; float *lg, *ylat, *yout; __nv_bfloat16 *dlat, *dsum;
    CK(cudaMalloc(&dx, xp.size() * 2)); CK(cudaMalloc(&lg, (size_t)NP * E * 4)); CK(cudaMalloc(&ylat, (size_t)NP * LAT * 4));
    CK(cudaMalloc(&yout, (size_t)NP * H * 4)); CK(cudaMalloc(&dlat, (size_t)NP * LAT * 2)); CK(cudaMalloc(&dsum, (size_t)NP * LAT * 2));
    CK(cudaMemcpy(dx, xp.data(), xp.size() * 2, cudaMemcpyHostToDevice));
    std::vector<float> hl((size_t)NP * E);
    if (s == 0) {   // [x x x] against the three exact BF16 parts of the FP32 gate
        std::vector<__nv_bfloat16> x3((size_t)NP * 3 * H);
        for (int j = 0; j < NP; j++) for (int c = 0; c < 3; c++) std::copy(xp.begin() + (size_t)j * H, xp.begin() + (size_t)(j + 1) * H, x3.begin() + (size_t)j * 3 * H + (size_t)c * H);
        __nv_bfloat16 *dx3; CK(cudaMalloc(&dx3, x3.size() * 2)); CK(cudaMemcpy(dx3, x3.data(), x3.size() * 2, cudaMemcpyHostToDevice));
        nbad += G.g3.run(dx3, 24, lg); CK(cudaFree(dx3));
        nbad += G.fc1.run(dx, 8, ylat);
    } else {
        std::vector<float> xf((size_t)NP * H); for (size_t i = 0; i < xf.size(); i++) xf[i] = __bfloat162float(xp[i]);
        float *dxf; CK(cudaMalloc(&dxf, xf.size() * 4)); CK(cudaMemcpy(dxf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice));
        fp32_router<<<((size_t)E * NP + 255) / 256, 256>>>(G.dWg, dxf, lg, s); CK(cudaDeviceSynchronize()); CK(cudaFree(dxf));
        engine_gemm(G.dfc1, dx, LAT, H, s, ylat);
    }
    CK(cudaMemcpy(hl.data(), lg, hl.size() * 4, cudaMemcpyDeviceToHost));
    R = route(hl, G.bias, N);
    to_bf16<<<512, 256>>>(ylat, dlat, (size_t)NP * LAT);
    std::set<int> want; for (auto &r : R) want.insert(r.e.begin(), r.e.end());
    need(want);
    std::vector<__nv_bfloat16> sum((size_t)NP * LAT, __float2bfloat16(0.f));
    for (int j = 0; j < N; j++)
        for (int k = 0; k < LAT; k++) {
            if (s == 0) { double a = 0; for (size_t t = 0; t < R[j].e.size(); t++) a += R[j].w[t] * expert_out(R[j].e[t], dlat, 0)[(size_t)j * LAT + k];
                          sum[(size_t)j * LAT + k] = __double2bfloat16(a); }
            else { float a = 0; for (size_t t = 0; t < R[j].e.size(); t++) a += (float)R[j].w[t] * expert_out(R[j].e[t], dlat, s)[(size_t)j * LAT + k];
                   sum[(size_t)j * LAT + k] = __float2bfloat16(a); }
        }
    CK(cudaMemcpy(dsum, sum.data(), sum.size() * 2, cudaMemcpyHostToDevice));
    if (s == 0) nbad += G.fc2.run(dsum, 8, yout); else engine_gemm(G.dfc2, dsum, H, LAT, s, yout);
    std::vector<float> yo((size_t)N * H); CK(cudaMemcpy(yo.data(), yout, yo.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<__nv_bfloat16> out(yo.size()); for (size_t i = 0; i < yo.size(); i++) out[i] = __float2bfloat16(yo[i]);
    for (void *q : {(void *)dx, (void *)lg, (void *)ylat, (void *)yout, (void *)dlat, (void *)dsum}) CK(cudaFree(q));
    yexp.clear();   // expert outputs depend on the batch's latent
    return out;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: out_moe_nv <dir> <layer> <N> [seed]\n"); return 1; }
    D = argv[1]; PRE = std::string("backbone.layers.") + argv[2] + ".mixer.";
    int N = atoi(argv[3]), seed = argc > 4 ? atoi(argv[4]) : 1;
    if (N > NP) { fprintf(stderr, "N <= %d\n", NP); return 1; }
    Gates G;
    auto ld16 = [](const std::string &p, size_t n) { auto v = load(p, n * 2); std::vector<__nv_bfloat16> o(n); memcpy(o.data(), v.data(), n * 2); return o; };
    auto Wb = load(D + "/" + PRE + "gate.weight.bin", (size_t)E * H * 4); G.Wg.resize((size_t)E * H); memcpy(G.Wg.data(), Wb.data(), Wb.size());
    auto bb = load(D + "/" + PRE + "gate.e_score_correction_bias.bin", E * 4); G.bias.resize(E); memcpy(G.bias.data(), bb.data(), bb.size());
    std::vector<__nv_bfloat16> W3((size_t)E * 3 * H);
    for (int e = 0; e < E; e++) for (int k = 0; k < H; k++) {
        float w = G.Wg[(size_t)e * H + k]; __nv_bfloat16 a = __float2bfloat16(w); float r1 = w - __bfloat162float(a);
        __nv_bfloat16 b = __float2bfloat16(r1); __nv_bfloat16 c = __float2bfloat16(r1 - __bfloat162float(b));
        W3[(size_t)e * 3 * H + k] = a; W3[(size_t)e * 3 * H + H + k] = b; W3[(size_t)e * 3 * H + 2 * H + k] = c;
    }
    G.g3.init(W3.data(), E, 3 * H);
    auto fc1 = ld16(D + "/" + PRE + "fc1_latent_proj.weight.bin", (size_t)LAT * H), fc2 = ld16(D + "/" + PRE + "fc2_latent_proj.weight.bin", (size_t)H * LAT);
    G.fc1.init(fc1.data(), LAT, H); G.fc2.init(fc2.data(), H, LAT);
    CK(cudaMalloc(&G.dfc1, fc1.size() * 2)); CK(cudaMalloc(&G.dfc2, fc2.size() * 2)); CK(cudaMalloc(&G.dWg, G.Wg.size() * 4));
    CK(cudaMemcpy(G.dfc1, fc1.data(), fc1.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(G.dfc2, fc2.data(), fc2.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(G.dWg, G.Wg.data(), G.Wg.size() * 4, cudaMemcpyHostToDevice));
    // tokens: Student-t hidden states through RMSNorm with the layer's norm weight (the block's input)
    auto lnw = ld16(D + "/backbone.layers." + std::string(argv[2]) + ".norm.weight.bin", H);
    std::mt19937_64 rng(seed); std::student_t_distribution<float> st(3.f);
    std::vector<__nv_bfloat16> x((size_t)N * H); std::vector<float> h(H);
    for (int j = 0; j < N; j++) {
        double ss = 0; for (int k = 0; k < H; k++) { h[k] = st(rng); ss += (double)h[k] * h[k]; }
        float r = 1.f / sqrtf((float)(ss / H) + 1e-5f);
        for (int k = 0; k < H; k++) x[(size_t)j * H + k] = __float2bfloat16(__bfloat162float(lnw[k]) * __bfloat162float(__float2bfloat16(h[k] * r)));
    }
    std::vector<Route> Rx, R1, R4, Ra; int nb, nb1;
    auto ox = block(G, x, N, 0, Rx, nb);
    auto o1 = block(G, x, N, 1, R1, nb1), o4 = block(G, x, N, 4, R4, nb1);
    size_t alone = 0;
    for (int j = 0; j < N; j++) {   // each token alone through the exact block
        std::vector<__nv_bfloat16> xj(x.begin() + (size_t)j * H, x.begin() + (size_t)(j + 1) * H);
        int nbj; auto oj = block(G, xj, 1, 0, Ra, nbj); nb += nbj;
        for (int k = 0; k < H; k++) alone += memcmp(&oj[k], &ox[(size_t)j * H + k], 2) != 0;
    }
    auto diff = [&](const std::vector<__nv_bfloat16> &a, const std::vector<__nv_bfloat16> &b) { size_t d = 0; for (size_t i = 0; i < a.size(); i++) d += memcmp(&a[i], &b[i], 2) != 0; return d; };
    auto moved = [&](const std::vector<Route> &a, const std::vector<Route> &b) { int m = 0; for (int j = 0; j < N; j++) m += a[j].e != b[j].e; return m; };
    size_t n = (size_t)N * H;
    printf("Nemotron 3 Ultra layer %s routed MoE block, %d tokens (seed %d), %zu experts used, real weights\n", argv[2], N, seed, experts.size());
    printf("  exact: products not certified %d; each token alone vs in the batch of %d: %zu of %zu BF16 outputs differ\n", nb, N, alone, n);
    printf("  engine split-K 1 vs 4: %zu of %zu outputs differ (%.1f %%), %d tokens routed differently\n", diff(o1, o4), n, 100.0 * diff(o1, o4) / n, moved(R1, R4));
    printf("  engine split-K 1 vs exact: %zu differ (%.1f %%), %d tokens routed differently\n", diff(o1, ox), 100.0 * diff(o1, ox) / n, moved(R1, Rx));
    return nb || alone ? 2 : 0;
}
