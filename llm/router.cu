// router.cu -- which experts a token goes to, on the real router of Kimi K3 layer 5, with exact and with ordinary logits.
// The router of a MoE layer computes logits = x Wg^T in FP32 (Wg: 896 x 7168 BF16), scores = sigmoid(logits) + a learned
// bias, and sends the token to the 16 best of 896 experts. When the 16th and 17th scores are closer than the rounding
// noise of the logits, the split of K that an engine picks for a given batch decides which expert runs -- and with it
// the whole output of the layer. Here the logits are computed (a) correctly rounded (llm/exact.cuh: BF16 x BF16 on INT8
// tensor cores) and (b) as engines do (BF16 tensor cores, FP32 accumulation, split-K 1, 2, 4, 8); the routing step
// itself is the same CPU code for all.
// Tokens: Student-t hidden states (3 degrees of freedom) through RMSNorm with the layer's real post_attention_layernorm
// weight -- the actual input of the gate. Weights from /opt/ozaki/k3/L5 (llm/k3_fetch.py).
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_router llm/router.cu ;  out_router <dir> <N tokens> [seed]
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

// the 16 experts of every token, as sorted index lists, from FP32 logits [N][E]
static std::vector<std::vector<int>> route(const std::vector<float> &lg, const std::vector<float> &bias, int N, int E, int k,
                                           std::vector<float> *gap = nullptr) {
    std::vector<std::vector<int>> out(N);
    std::vector<std::pair<float, int>> c(E);
    for (int j = 0; j < N; j++) {
        for (int e = 0; e < E; e++) c[e] = {1.f / (1.f + expf(-lg[(size_t)j * E + e])) + bias[e], e};
        std::partial_sort(c.begin(), c.begin() + k + 1, c.end(), [](auto &a, auto &b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
        if (gap) gap->push_back(c[k - 1].first - c[k].first);
        for (int t = 0; t < k; t++) out[j].push_back(c[t].second);
        std::sort(out[j].begin(), out[j].end());
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dir> <N> [seed]\n", argv[0]); return 1; }
    const std::string dir = std::string(argv[1]) + "/language_model.model.layers.5.";
    const int N = atoi(argv[2]), seed = argc > 3 ? atoi(argv[3]) : 1, E = 896, H = 7168, TOPK = 16, S = 4;
    const int Np = (N + 127) / 128 * 128;
    auto Wg = slurp<uint16_t>(dir + "block_sparse_moe.gate.weight.bin", (size_t)E * H);
    auto bias = slurp<float>(dir + "block_sparse_moe.gate.e_score_correction_bias.bin", E);
    auto lnw16 = slurp<uint16_t>(dir + "post_attention_layernorm.weight.bin", H);
    auto bf = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };

    std::mt19937_64 rng(seed);
    std::student_t_distribution<float> st(3.f);
    std::vector<__nv_bfloat16> X((size_t)Np * H, __float2bfloat16(0.f));
    std::vector<float> h(H);
    for (int j = 0; j < N; j++) {
        double ss = 0;
        for (int k = 0; k < H; k++) { h[k] = st(rng); ss += (double)h[k] * h[k]; }
        float r = 1.f / sqrtf((float)(ss / H) + 1e-6f);
        for (int k = 0; k < H; k++) X[(size_t)j * H + k] = __float2bfloat16(bf(lnw16[k]) * __bfloat162float(__float2bfloat16(h[k] * r)));
    }

    // exact logits
    ex::Digits W{}, Xd{};
    __nv_bfloat16 *dW, *dX; int *stat, *nbad; int32_t *P; float *Y, *part;
    CK(cudaMalloc(&dW, (size_t)E * H * 2)); CK(cudaMalloc(&dX, (size_t)Np * H * 2)); CK(cudaMalloc(&stat, 12)); CK(cudaMalloc(&nbad, 4));
    CK(cudaMemcpy(dW, Wg.data(), (size_t)E * H * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X.data(), (size_t)Np * H * 2, cudaMemcpyHostToDevice));
    W.S = Xd.S = S; W.rows = E; Xd.rows = Np; W.K = Xd.K = H;
    CK(cudaMalloc(&W.dig, (size_t)ex::TMAX * S * E * H)); CK(cudaMalloc(&W.B, ex::TMAX * E * 4));
    CK(cudaMalloc(&Xd.dig, (size_t)ex::TMAX * S * Np * H)); CK(cudaMalloc(&Xd.B, ex::TMAX * Np * 4));
    int sw[3], sx[3];
    CK(cudaMemset(stat, 0, 12)); ex::split_rows<<<E, 256>>>(dW, W.dig, W.B, stat, E, H, S); CK(cudaMemcpy(sw, stat, 12, cudaMemcpyDeviceToHost));
    CK(cudaMemset(stat, 0, 12)); ex::split_rows<<<Np, 256>>>(dX, Xd.dig, Xd.B, stat, Np, H, S); CK(cudaMemcpy(sx, stat, 12, cudaMemcpyDeviceToHost));
    W.T = sw[0]; Xd.T = sx[0];
    const int Z = W.T * S * Xd.T * S;
    printf("router of layer 5: %d experts x %d, N %d tokens; digits %d; gate weight %d tier(s), tokens %d tier(s): %d INT8 GEMMs; left over %d + %d\n",
           E, H, N, S, W.T, Xd.T, Z, sw[1], sx[1]);
    CK(cudaMalloc(&P, (size_t)Z * E * Np * 4)); CK(cudaMalloc(&Y, (size_t)Np * E * 4)); CK(cudaMalloc(&part, (size_t)8 * Np * E * 4));
    CK(cudaMemset(nbad, 0, 4));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    ex::gemm_pairs<<<dim3(Np / oz::BN, E / oz::BM, Z), oz::THREADS>>>(W, Xd, P, Xd.T);
    ex::combine<<<dim3((E + 255) / 256, Np), 256>>>(P, W, Xd, W.T, Xd.T, 24, Y, E, nbad);
    cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    int nb; CK(cudaMemcpy(&nb, nbad, 4, cudaMemcpyDeviceToHost));
    std::vector<float> ex_lg((size_t)N * E);
    CK(cudaMemcpy(ex_lg.data(), Y, ex_lg.size() * 4, cudaMemcpyDeviceToHost));
    printf("exact logits: %.2f ms, %d of %zu not certified\n", ms, nb, (size_t)N * E);

    std::vector<float> gap;
    auto r_ex = route(ex_lg, bias, N, E, TOPK, &gap);
    std::vector<float> g2 = gap; std::sort(g2.begin(), g2.end());
    printf("gap between the 16th and 17th score (exact): median %.3g, 1st percentile %.3g, smallest %.3g\n",
           g2[N / 2], g2[N / 100], g2[0]);

    std::vector<std::vector<std::vector<int>>> r_b;
    std::vector<std::vector<float>> lg_b;
    for (int s : {1, 2, 4, 8}) {
        ex::bf16_gemm<<<dim3(Np / ex::BN, E / ex::BM, s), ex::THREADS>>>(dW, dX, part, E, Np, H, s);
        ex::reduce_k<<<512, 256>>>(part, Y, (size_t)Np * E, s);
        CK(cudaDeviceSynchronize());
        std::vector<float> lg((size_t)N * E);
        CK(cudaMemcpy(lg.data(), Y, lg.size() * 4, cudaMemcpyDeviceToHost));
        size_t neq = 0;
        for (size_t t = 0; t < lg.size(); t++) neq += memcmp(&lg[t], &ex_lg[t], 4) != 0;
        auto r = route(lg, bias, N, E, TOPK);
        int moved = 0, experts = 0;
        for (int j = 0; j < N; j++) if (r[j] != r_ex[j]) {
            moved++;
            std::vector<int> d;
            std::set_symmetric_difference(r[j].begin(), r[j].end(), r_ex[j].begin(), r_ex[j].end(), std::back_inserter(d));
            experts += d.size() / 2;
        }
        printf("  BF16 tensor cores, FP32 accumulation, split-K %d: %.1f %% of logits not correctly rounded; "
               "%d of %d tokens (%.2f %%) routed to a different expert set than with exact logits (%d experts swapped)\n",
               s, 100.0 * neq / lg.size(), moved, N, 100.0 * moved / N, experts);
        r_b.push_back(r);
    }
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++) {
            int moved = 0;
            for (int j = 0; j < N; j++) moved += r_b[a][j] != r_b[b][j];
            printf("  split-K %d vs %d: %d tokens (%.2f %%) go to different experts\n", 1 << a, 1 << b, moved, 100.0 * moved / N);
        }
    return 0;
}
