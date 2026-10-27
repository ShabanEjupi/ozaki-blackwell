// router_nv.cu -- which experts a token goes to on the real router of Nemotron 3 Ultra (gate FP32 512 x 8192, sigmoid +
// e_score_correction_bias, top-22 of 512), with exact and with ordinary FP32 logits. The FP32 gate is split exactly into three
// BF16 matrices, so x W^T is one correctly rounded BF16 x BF16 product (llm/exact.cuh) with K = 3 x 8192. Tokens: Student-t
// hidden states (3 degrees of freedom) through RMSNorm with the layer's real norm weight -- the actual input of the gate.
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_router_nv llm/router_nv.cu ;  out_router_nv /opt/ozaki/nv/L49 49 <N> [seed]
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

// FP32 router logits as an engine computes them: x and the gate in FP32, FMA in k order, split-K in s slices reduced in order
__global__ void fp32_router(const float *W, const float *X, float *Y, int E, int N, int H, int s) {
    size_t o = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (o >= (size_t)E * N) return;
    int e = o % E, j = o / E, per = H / s;
    float tot = 0.f;
    for (int z = 0; z < s; z++) {
        float acc = 0.f;
        for (int k = z * per; k < (z + 1) * per; k++) acc = fmaf(W[(size_t)e * H + k], X[(size_t)j * H + k], acc);
        tot += acc;
    }
    Y[o] = tot;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dir> <layer> <N> [seed]\n", argv[0]); return 1; }
    const std::string dir = std::string(argv[1]) + "/backbone.layers." + argv[2] + ".";
    const int N = atoi(argv[3]), seed = argc > 4 ? atoi(argv[4]) : 1, E = 512, H = 8192, H3 = 3 * H, TOPK = 22, S = 4;
    const int Np = (N + 127) / 128 * 128;
    auto Wg = slurp<float>(dir + "mixer.gate.weight.bin", (size_t)E * H);
    auto bias = slurp<float>(dir + "mixer.gate.e_score_correction_bias.bin", E);
    auto lnw16 = slurp<uint16_t>(dir + "norm.weight.bin", H);
    auto bf = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; };
    // the FP32 gate as three BF16 matrices whose sum is exact: w = w1 + w2 + w3; then x w^T = [x x x] [w1 w2 w3]^T, one exact
    // BF16 x BF16 product with K = 3 H
    std::vector<__nv_bfloat16> W3((size_t)E * H3);
    size_t inexact = 0;
    for (int e = 0; e < E; e++)
        for (int k = 0; k < H; k++) {
            float w = Wg[(size_t)e * H + k];
            __nv_bfloat16 a = __float2bfloat16(w); float r1 = w - __bfloat162float(a);
            __nv_bfloat16 b = __float2bfloat16(r1); float r2 = r1 - __bfloat162float(b);
            __nv_bfloat16 c = __float2bfloat16(r2);
            inexact += (double)__bfloat162float(a) + (double)__bfloat162float(b) + (double)__bfloat162float(c) != (double)w;
            W3[(size_t)e * H3 + k] = a; W3[(size_t)e * H3 + H + k] = b; W3[(size_t)e * H3 + 2 * H + k] = c;
        }
    std::mt19937_64 rng(seed);
    std::student_t_distribution<float> st(3.f);
    std::vector<__nv_bfloat16> X3((size_t)Np * H3, __float2bfloat16(0.f));
    std::vector<float> Xf((size_t)Np * H, 0.f), h(H);
    for (int j = 0; j < N; j++) {
        double ss = 0;
        for (int k = 0; k < H; k++) { h[k] = st(rng); ss += (double)h[k] * h[k]; }
        float r = 1.f / sqrtf((float)(ss / H) + 1e-5f);
        for (int k = 0; k < H; k++) {
            __nv_bfloat16 x = __float2bfloat16(bf(lnw16[k]) * __bfloat162float(__float2bfloat16(h[k] * r)));
            for (int c = 0; c < 3; c++) X3[(size_t)j * H3 + c * H + k] = x;
            Xf[(size_t)j * H + k] = __bfloat162float(x);
        }
    }
    printf("Nemotron 3 Ultra router, layer %s: %d experts x %d (FP32 gate as 3 exact BF16 parts: %zu of %zu weights not exact), N %d, top-%d\n",
           argv[2], E, H, inexact, (size_t)E * H, N, TOPK);

    ex::Digits W{}, Xd{};
    __nv_bfloat16 *dW, *dX; int *stat, *nbad; int32_t *P; float *Y;
    CK(cudaMalloc(&dW, (size_t)E * H3 * 2)); CK(cudaMalloc(&dX, (size_t)Np * H3 * 2)); CK(cudaMalloc(&stat, 12)); CK(cudaMalloc(&nbad, 4));
    CK(cudaMemcpy(dW, W3.data(), (size_t)E * H3 * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX, X3.data(), (size_t)Np * H3 * 2, cudaMemcpyHostToDevice));
    W.S = Xd.S = S; W.rows = E; Xd.rows = Np; W.K = Xd.K = H3;
    CK(cudaMalloc(&W.dig, (size_t)ex::TMAX * S * E * H3)); CK(cudaMalloc(&W.B, ex::TMAX * E * 4));
    CK(cudaMalloc(&Xd.dig, (size_t)ex::TMAX * S * Np * H3)); CK(cudaMalloc(&Xd.B, ex::TMAX * Np * 4));
    int sw[3], sx[3];
    CK(cudaMemset(stat, 0, 12)); ex::split_rows<<<E, 256>>>(dW, W.dig, W.B, stat, E, H3, S); CK(cudaMemcpy(sw, stat, 12, cudaMemcpyDeviceToHost));
    CK(cudaMemset(stat, 0, 12)); ex::split_rows<<<Np, 256>>>(dX, Xd.dig, Xd.B, stat, Np, H3, S); CK(cudaMemcpy(sx, stat, 12, cudaMemcpyDeviceToHost));
    W.T = sw[0]; Xd.T = sx[0];
    const int Z = W.T * S * Xd.T * S;
    printf("digits %d; gate %d tier(s), tokens %d tier(s): %d INT8 GEMMs; elements beyond the tiers %d + %d\n", S, W.T, Xd.T, Z, sw[1], sx[1]);
    CK(cudaMalloc(&P, (size_t)Z * E * Np * 4)); CK(cudaMalloc(&Y, (size_t)Np * E * 4));
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
    printf("gap between the 22nd and 23rd score (exact): median %.3g, 1st percentile %.3g, smallest %.3g\n", g2[N / 2], g2[N / 100], g2[0]);

    float *dWf, *dXf, *Yf; CK(cudaMalloc(&dWf, (size_t)E * H * 4)); CK(cudaMalloc(&dXf, (size_t)Np * H * 4)); CK(cudaMalloc(&Yf, (size_t)Np * E * 4));
    CK(cudaMemcpy(dWf, Wg.data(), (size_t)E * H * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dXf, Xf.data(), (size_t)Np * H * 4, cudaMemcpyHostToDevice));
    std::vector<std::vector<std::vector<int>>> r_b;
    for (int s : {1, 2, 4, 8}) {
        fp32_router<<<((size_t)E * N + 255) / 256, 256>>>(dWf, dXf, Yf, E, N, H, s);
        CK(cudaDeviceSynchronize());
        std::vector<float> lg((size_t)N * E);
        CK(cudaMemcpy(lg.data(), Yf, lg.size() * 4, cudaMemcpyDeviceToHost));
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
        printf("  FP32 FMA, split-K %d: %.1f %% of logits not correctly rounded; %d of %d tokens (%.3f %%) routed to a different "
               "expert set than with exact logits (%d experts swapped)\n", s, 100.0 * neq / lg.size(), moved, N, 100.0 * moved / N, experts);
        r_b.push_back(r);
    }
    for (int a = 0; a < 4; a++)
        for (int b = a + 1; b < 4; b++) {
            int moved = 0;
            for (int j = 0; j < N; j++) moved += r_b[a][j] != r_b[b][j];
            printf("  split-K %d vs %d: %d tokens (%.3f %%) go to different experts\n", 1 << a, 1 << b, moved, 100.0 * moved / N);
        }
    return 0;
}
