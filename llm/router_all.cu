// router_all.cu -- the routers of all 92 MoE layers of Kimi K3 (real gates, /opt/ozaki/k3/gates/L<l>, llm/k3_gates.txt):
// for the same tokens, how many go to a different expert set in at least one layer when only the rounding of the FP32
// logits changes (BF16 tensor cores, FP32 accumulation, split-K 1 vs 4; and split-K 1 vs the correctly rounded logits).
// Tokens: Student-t hidden states through each layer's real post_attention_layernorm weight (the gate's actual input
// kind; the hidden states themselves are synthetic). Same routing code as router.cu.
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_router_all llm/router_all.cu ; out_router_all <gates dir> <layers file> <N> [seed]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include "exact.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

template <class T> static std::vector<T> slurp(const std::string &p, size_t n) {
    std::vector<T> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f || fread(v.data(), sizeof(T), n, f) != n) { fprintf(stderr, "cannot read %s\n", p.c_str()); exit(1); }
    fclose(f);
    return v;
}
static std::vector<std::vector<int>> route(const std::vector<float> &lg, const std::vector<float> &bias, int N, int E, int k) {
    std::vector<std::vector<int>> out(N);
    std::vector<std::pair<float, int>> c(E);
    for (int j = 0; j < N; j++) {
        for (int e = 0; e < E; e++) c[e] = {1.f / (1.f + expf(-lg[(size_t)j * E + e])) + bias[e], e};
        std::partial_sort(c.begin(), c.begin() + k, c.end(), [](auto &a, auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        for (int t = 0; t < k; t++) out[j].push_back(c[t].second);
        std::sort(out[j].begin(), out[j].end());
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <gates dir> <layers file> <N> [seed]\n", argv[0]); return 1; }
    const int N = atoi(argv[3]), seed = argc > 4 ? atoi(argv[4]) : 1, E = 896, H = 7168, TOPK = 16, S = 4, Np = (N + 127) / 128 * 128;
    std::vector<int> layers; { FILE *f = fopen(argv[2], "r"); int l; char s[256]; while (f && fscanf(f, "%d %255s", &l, s) == 2) layers.push_back(l); if (f) fclose(f); }
    auto bf = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
    // the hidden states (before the norm) are the same for every layer
    std::mt19937_64 rng(seed); std::student_t_distribution<float> st(3.f);
    std::vector<float> h((size_t)N * H), rr(N);
    for (int j = 0; j < N; j++) { double ss = 0; for (int k = 0; k < H; k++) { h[(size_t)j * H + k] = st(rng); ss += (double)h[(size_t)j * H + k] * h[(size_t)j * H + k]; } rr[j] = 1.f / sqrtf((float)(ss / H) + 1e-5f); }

    __nv_bfloat16 *dW, *dX; int *stat, *nbad; int32_t *P; float *Y, *part;
    ex::Digits W{}, Xd{};
    CK(cudaMalloc(&dW, (size_t)E * H * 2)); CK(cudaMalloc(&dX, (size_t)Np * H * 2)); CK(cudaMalloc(&stat, 12)); CK(cudaMalloc(&nbad, 4));
    W.S = Xd.S = S; W.rows = E; Xd.rows = Np; W.K = Xd.K = H;
    CK(cudaMalloc(&W.dig, (size_t)ex::TMAX * S * E * H)); CK(cudaMalloc(&W.B, ex::TMAX * E * 4));
    CK(cudaMalloc(&Xd.dig, (size_t)ex::TMAX * S * Np * H)); CK(cudaMalloc(&Xd.B, ex::TMAX * Np * 4));
    CK(cudaMalloc(&P, (size_t)ex::TMAX * S * ex::TMAX * S * E * Np * 4 / 4)); CK(cudaMalloc(&Y, (size_t)Np * E * 4)); CK(cudaMalloc(&part, (size_t)4 * Np * E * 4));
    CK(cudaMemset(nbad, 0, 4));
    std::vector<int> any14(N, 0), anyEx(N, 0);
    long sw14 = 0, swEx = 0;
    std::vector<__nv_bfloat16> X((size_t)Np * H, __float2bfloat16(0.f));
    for (int L : layers) {
        const std::string pre = std::string(argv[1]) + "/L" + std::to_string(L) + "/language_model.model.layers." + std::to_string(L) + ".";
        auto Wg = slurp<uint16_t>(pre + "block_sparse_moe.gate.weight.bin", (size_t)E * H);
        auto bias = slurp<float>(pre + "block_sparse_moe.gate.e_score_correction_bias.bin", E);
        auto lnw = slurp<uint16_t>(pre + "post_attention_layernorm.weight.bin", H);
        for (int j = 0; j < N; j++) for (int k = 0; k < H; k++)
            X[(size_t)j * H + k] = __float2bfloat16(bf(lnw[k]) * __bfloat162float(__float2bfloat16(h[(size_t)j * H + k] * rr[j])));
        CK(cudaMemcpy(dW, Wg.data(), (size_t)E * H * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dX, X.data(), (size_t)Np * H * 2, cudaMemcpyHostToDevice));
        int sw[3], sx[3];
        CK(cudaMemset(stat, 0, 12)); ex::split_rows<<<E, 256>>>(dW, W.dig, W.B, stat, E, H, S); CK(cudaMemcpy(sw, stat, 12, cudaMemcpyDeviceToHost));
        CK(cudaMemset(stat, 0, 12)); ex::split_rows<<<Np, 256>>>(dX, Xd.dig, Xd.B, stat, Np, H, S); CK(cudaMemcpy(sx, stat, 12, cudaMemcpyDeviceToHost));
        W.T = sw[0]; Xd.T = sx[0];
        const int Z = W.T * S * Xd.T * S;
        if (Z > ex::TMAX * S * ex::TMAX * S / 4 || sw[1] || sx[1]) { fprintf(stderr, "layer %d: Z %d, left %d %d\n", L, Z, sw[1], sx[1]); return 1; }
        ex::gemm_pairs<<<dim3(Np / oz::BN, E / oz::BM, Z), oz::THREADS>>>(W, Xd, P, Xd.T);
        ex::combine<<<dim3((E + 255) / 256, Np), 256>>>(P, W, Xd, W.T, Xd.T, 24, Y, E, nbad);
        std::vector<float> lg((size_t)N * E);
        CK(cudaMemcpy(lg.data(), Y, lg.size() * 4, cudaMemcpyDeviceToHost));
        auto rex = route(lg, bias, N, E, TOPK);
        std::vector<std::vector<std::vector<int>>> rb;
        for (int s : {1, 4}) {
            ex::bf16_gemm<<<dim3(Np / ex::BN, E / ex::BM, s), ex::THREADS>>>(dW, dX, part, E, Np, H, s);
            ex::reduce_k<<<512, 256>>>(part, Y, (size_t)Np * E, s);
            CK(cudaMemcpy(lg.data(), Y, lg.size() * 4, cudaMemcpyDeviceToHost));
            rb.push_back(route(lg, bias, N, E, TOPK));
        }
        int f14 = 0, fex = 0;
        for (int j = 0; j < N; j++) {
            if (rb[0][j] != rb[1][j]) { f14++; any14[j] = 1; }
            if (rb[0][j] != rex[j]) { fex++; anyEx[j] = 1; }
        }
        sw14 += f14; swEx += fex;
        printf("layer %2d: tiers %d/%d, tokens moved split-K 1 vs 4: %d, split-K 1 vs exact: %d\n", L, W.T, Xd.T, f14, fex);
        fflush(stdout);
    }
    int nb; CK(cudaMemcpy(&nb, nbad, 4, cudaMemcpyDeviceToHost));
    int a14 = 0, aEx = 0; for (int j = 0; j < N; j++) { a14 += any14[j]; aEx += anyEx[j]; }
    printf("%zu layers, %d tokens: (token, layer) swaps split-K 1 vs 4: %ld, vs exact: %ld; tokens with a different expert in at least one layer: "
           "%d (%.2f %%) split-K 1 vs 4, %d (%.2f %%) split-K 1 vs exact; exact logits not certified: %d\n",
           layers.size(), N, sw14, swEx, a14, 100.0 * a14 / N, aEx, 100.0 * aEx / N, nb);
    return 0;
}
