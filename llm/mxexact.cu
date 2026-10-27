// mxexact.cu -- correctly rounded MXFP4 x BF16 matrix products on INT8 tensor cores, checked on the real weights of
// Kimi K3 (2.78 T parameters, the largest open-weights model; routed experts are MXFP4: E2M1 values, one E8M0 scale per 32).
//
// Why it is exact. A weight is w = q 2^(s-128) with q = 2 * E2M1 in {0, +-1, 2, 3, 4, 6, 8, 12} and s the group's scale byte.
// A BF16 activation is x = m 2^e with |m| < 256. Inside one group g of 32 (= the K depth of one mma.m16n8k32) the activations
// of a token j are aligned to the group's largest exponent E: A = m 2^(D - (E - e)) is an integer of 21 bits when
// E - e <= D = 13, split into three int8 digits (7 + 7 + 7 bits, the top one signed). So the group's contribution
//     sum_k w_ik x_kj = 2^(s_ig - 128 + E_gj - D) * P,  P = sum_k q_ik A_kj = P0 + 2^7 P1 + 2^14 P2  (|P| < 2^29.6, exact in int32)
// comes out of three int8 tensor-core products. Elements more than 2^13 below their group's maximum go to a second tier
// (its own E), and so on. The groups are added in a 128-bit fixed-point accumulator anchored at the top exponent of the
// row and column; a term that falls below its 89-bit window is floored and counted, so the exact sum is known to lie in
// [acc, acc + n) ulps of the window. Rounding both ends to FP32 gives the same float almost always: then that float is
// the correctly rounded exact product (certified). Otherwise the output is flagged.
// A correctly rounded result does not depend on the summation order, the tiling, the batch size or the machine.
//
// The baseline is what inference engines do: dequantize to BF16 (exact) and use BF16 tensor cores with FP32
// accumulation, split-K chosen by shape. Its bits change with the split.
//
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_mx llm/mxexact.cu
// out_mx <dir> <tensor prefix> <N> [seed]   e.g. /opt/ozaki/k3/L5 language_model.model.layers.5.block_sparse_moe.experts.0.w1 4096
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int D = 13, LMAX = 4, WIN = 89, EMPTY = -30000;
constexpr int BM = 64, BN = 64, THREADS = 256, RS = 48;   // tile; shared rows of 48 bytes: the 8 rows of a fragment hit 8 bank groups

// ---- preprocessing ------------------------------------------------------------------------------------------------
// packed [M][K/2] (low nibble = even k) -> q [M][K] int8; Smax[i] = max_g s_ig; Wb [M][K] = the same weights in BF16 (exact)
__global__ void unpack_w(const uint8_t *pk, const uint8_t *sc, int8_t *q, __nv_bfloat16 *wb, int M, int K) {
    const int8_t mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    size_t n = (size_t)M * K / 2;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
        size_t i = t / (K / 2), k = 2 * (t % (K / 2));
        uint8_t b = pk[t];
        for (int h = 0; h < 2; h++) {
            int nb = h ? b >> 4 : b & 15, v = mag[nb & 7] * (nb & 8 ? -1 : 1);
            q[i * K + k + h] = (int8_t)v;
            wb[i * K + k + h] = __float2bfloat16_rn(ldexpf((float)v, (int)sc[i * (K / 32) + (k + h) / 32] - 128));
        }
    }
}
__global__ void row_smax(const uint8_t *sc, int *smax, int M, int G) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    int m = 0;
    for (int g = 0; g < G; g++) m = max(m, (int)sc[(size_t)i * G + g]);
    smax[i] = m;
}

// one thread per (token j, group g): tiers of aligned 21-bit integers, three int8 digits each.
// X [Np][K] bf16 -> xs [LMAX][3][Np][K] int8, Et [LMAX][Np][G] (EMPTY if the tier has nothing), Emax[j] (+ 100000 offset)
__global__ void split_x(const __nv_bfloat16 *X, int8_t *xs, int *Et, int *Emax, int *stat, int Np, int K) {
    const int G = K / 32;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Np * G) return;
    int j = t / G, g = t % G;
    int m[32], e[32];
    unsigned left = 0;
    for (int k = 0; k < 32; k++) {
        uint16_t b = __bfloat16_as_ushort(X[(size_t)j * K + g * 32 + k]);
        int ex = (b >> 7) & 255, man = b & 127;
        if (ex == 255) atomicAdd(&stat[2], 1);   // inf / nan: not a number to round
        m[k] = ex ? (man | 128) : man;
        e[k] = ex ? ex - 134 : -133;
        if (b & 0x8000) m[k] = -m[k];
        if (m[k]) left |= 1u << k;
    }
    for (int l = 0; l < LMAX; l++) {
        int E = EMPTY;
        for (int k = 0; k < 32; k++) if (left >> k & 1) E = max(E, e[k]);
        Et[((size_t)l * Np + j) * G + g] = E;
        if (l == 0 && E != EMPTY) atomicMax(&Emax[j], E + 100000);
        if (E != EMPTY) atomicMax(&stat[0], l + 1);
        for (int k = 0; k < 32; k++) {
            int A = 0;
            if ((left >> k & 1) && E - e[k] <= D) { A = m[k] * (1 << (D - (E - e[k]))); left &= ~(1u << k); }
            size_t o = (size_t)j * K + g * 32 + k, pl = (size_t)Np * K;
            xs[(3 * l + 0) * pl + o] = (int8_t)(A & 127);
            xs[(3 * l + 1) * pl + o] = (int8_t)((A >> 7) & 127);
            xs[(3 * l + 2) * pl + o] = (int8_t)(A >> 14);
        }
    }
    if (left) atomicAdd(&stat[1], __popc(left));   // below LMAX tiers: would be lost (counted, must be 0)
}

// ---- 128-bit rounding -----------------------------------------------------------------------------------------------
// a * 2^X rounded to nearest even FP32 (subnormals and overflow included)
__device__ __host__ inline float round_i128(__int128 a, int X) {
    if (a == 0) return 0.f;
    bool neg = a < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-a) : (unsigned __int128)a;
    uint64_t hi = (uint64_t)(u >> 64), lo = (uint64_t)u;
#ifdef __CUDA_ARCH__
    int nb = hi ? 128 - __clzll(hi) : 64 - __clzll(lo);
#else
    int nb = hi ? 128 - __builtin_clzll(hi) : 64 - __builtin_clzll(lo);
#endif
    int Ev = nb - 1 + X, L = Ev - 23 > -149 ? Ev - 23 : -149, r = L - X;
    float f;
    if (r <= 0) f = ldexpf((float)(uint64_t)u, X);
    else if (r > 128) f = 0.f;
    else {
        unsigned __int128 q = r == 128 ? 0 : u >> r, rem = u - (r == 128 ? 0 : q << r), half = (unsigned __int128)1 << (r - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        f = ldexpf((float)(uint64_t)q, L);
    }
    return neg ? -f : f;
}

// ---- the exact product ----------------------------------------------------------------------------------------------
// Y [Np][M] fp32 = round(W X^T) exactly; nbad counts outputs whose certificate failed (flag[j*M+i] = 1).
__global__ void __launch_bounds__(THREADS) mx_exact(const int8_t *__restrict__ Wq, const uint8_t *__restrict__ sc,
        const int *__restrict__ smax, const int8_t *__restrict__ xs, const int *__restrict__ Et, const int *__restrict__ Emax,
        float *Y, uint8_t *flag, int *nbad, int M, int Np, int K, int L) {
    __shared__ __align__(16) int8_t sW[BM * RS], sX[LMAX * 3][BN * RS];
    __shared__ int sS[BM], sE[LMAX][BN];
    const int G = K / 32, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wm = (warp >> 2) * 32, wn = (warp & 3) * 16, i0 = blockIdx.y * BM, j0 = blockIdx.x * BN;
    const size_t pl = (size_t)Np * K;
    // this thread's rows (2 m16 tiles x {gid, gid+8}) and columns (2 n8 tiles x {2 tig, 2 tig + 1})
    int rTop[4], cTop[4];
    for (int a = 0; a < 4; a++) {
        rTop[a] = smax[i0 + wm + (a >> 1) * 16 + (a & 1) * 8 + gid];
        cTop[a] = Emax[j0 + wn + (a >> 1) * 8 + 2 * tig + (a & 1)] - 100000;
    }
    __int128 acc[2][2][4] = {};
    int lost[2][2][4] = {};
    for (int g = 0; g < G; g++) {
        for (int c = tid; c < BM * 2; c += THREADS) {   // W: 64 rows x 32 bytes
            int r = c >> 1, h = c & 1;
            *(int4 *)&sW[r * RS + h * 16] = *(const int4 *)&Wq[(size_t)(i0 + r) * K + g * 32 + h * 16];
        }
        for (int c = tid; c < L * 3 * BN * 2; c += THREADS) {
            int p = c / (BN * 2), r = (c >> 1) % BN, h = c & 1;
            *(int4 *)&sX[p][r * RS + h * 16] = *(const int4 *)&xs[p * pl + (size_t)(j0 + r) * K + g * 32 + h * 16];
        }
        if (tid < BM) sS[tid] = sc[(size_t)(i0 + tid) * G + g];
        for (int c = tid; c < L * BN; c += THREADS) sE[c / BN][c % BN] = Et[((size_t)(c / BN) * Np + j0 + c % BN) * G + g];
        __syncthreads();
        uint32_t a[2][4];
        for (int mt = 0; mt < 2; mt++) {
            const int8_t *p = sW + (wm + mt * 16 + gid) * RS + tig * 4;
            a[mt][0] = *(const uint32_t *)p; a[mt][1] = *(const uint32_t *)(p + 8 * RS);
            a[mt][2] = *(const uint32_t *)(p + 16); a[mt][3] = *(const uint32_t *)(p + 8 * RS + 16);
        }
        for (int l = 0; l < L; l++) {
            int P[3][2][2][4] = {};
            for (int d = 0; d < 3; d++)
                for (int nt = 0; nt < 2; nt++) {
                    const int8_t *p = sX[3 * l + d] + (wn + nt * 8 + gid) * RS + tig * 4;
                    uint32_t b0 = *(const uint32_t *)p, b1 = *(const uint32_t *)(p + 16);
                    for (int mt = 0; mt < 2; mt++)
                        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                     : "+r"(P[d][mt][nt][0]), "+r"(P[d][mt][nt][1]), "+r"(P[d][mt][nt][2]), "+r"(P[d][mt][nt][3])
                                     : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b0), "r"(b1));
                }
            for (int mt = 0; mt < 2; mt++)
                for (int nt = 0; nt < 2; nt++)
                    for (int v = 0; v < 4; v++) {
                        int pg = P[0][mt][nt][v] + P[1][mt][nt][v] * 128 + P[2][mt][nt][v] * 16384;
                        if (!pg) continue;
                        int ra = mt * 2 + (v >> 1), ca = nt * 2 + (v & 1);
                        int sh = (sS[wm + mt * 16 + (v >> 1) * 8 + gid] - rTop[ra]) + (sE[l][wn + nt * 8 + 2 * tig + (v & 1)] - cTop[ca]) + WIN;
                        if (sh >= 0) acc[mt][nt][v] += (__int128)pg << sh;
                        else { acc[mt][nt][v] += (__int128)(pg >> min(-sh, 31)); lost[mt][nt][v]++; }
                    }
        }
        __syncthreads();
    }
    for (int mt = 0; mt < 2; mt++)
        for (int nt = 0; nt < 2; nt++)
            for (int v = 0; v < 4; v++) {
                int ra = mt * 2 + (v >> 1), ca = nt * 2 + (v & 1);
                int i = i0 + wm + mt * 16 + (v >> 1) * 8 + gid, j = j0 + wn + nt * 8 + 2 * tig + (v & 1);
                int X = rTop[ra] + cTop[ca] - WIN - 128 - D;
                float f = round_i128(acc[mt][nt][v], X);
                bool ok = !lost[mt][nt][v] || round_i128(acc[mt][nt][v] + lost[mt][nt][v], X) == f;
                Y[(size_t)j * M + i] = f;
                flag[(size_t)j * M + i] = !ok;
                if (!ok) atomicAdd(nbad, 1);
            }
}

// ---- baseline: BF16 tensor cores, FP32 accumulation, split-K ---------------------------------------------------------
// part [S][Np][M]: slice z of K; reduce adds the slices in order z = 0..S-1 (what cuBLAS / CUTLASS split-K does)
__global__ void __launch_bounds__(THREADS) bf16_gemm(const __nv_bfloat16 *__restrict__ Wb, const __nv_bfloat16 *__restrict__ X,
        float *part, int M, int Np, int K, int S) {
    constexpr int RB = 40;   // 32 bf16 + 8 pad = 80 bytes: conflict-free fragment rows
    __shared__ __align__(16) __nv_bfloat16 sW[BM * RB], sX[BN * RB];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int wm = (warp >> 2) * 32, wn = (warp & 3) * 16, i0 = blockIdx.y * BM, j0 = blockIdx.x * BN, z = blockIdx.z;
    const int kb = z * (K / S), ke = kb + K / S;
    float acc[2][2][4] = {};
    for (int k0 = kb; k0 < ke; k0 += 32) {
        for (int c = tid; c < BM * 4; c += THREADS) {
            int r = c >> 2, h = c & 3;
            *(int4 *)&sW[r * RB + h * 8] = *(const int4 *)&Wb[(size_t)(i0 + r) * K + k0 + h * 8];
            *(int4 *)&sX[r * RB + h * 8] = *(const int4 *)&X[(size_t)(j0 + r) * K + k0 + h * 8];
        }
        __syncthreads();
        for (int ks = 0; ks < 32; ks += 16) {
            uint32_t a[2][4], b[2][2];
            for (int mt = 0; mt < 2; mt++) {
                const __nv_bfloat16 *p = sW + (wm + mt * 16 + gid) * RB + ks + 2 * tig;
                a[mt][0] = *(const uint32_t *)p; a[mt][1] = *(const uint32_t *)(p + 8 * RB);
                a[mt][2] = *(const uint32_t *)(p + 8); a[mt][3] = *(const uint32_t *)(p + 8 * RB + 8);
            }
            for (int nt = 0; nt < 2; nt++) {
                const __nv_bfloat16 *p = sX + (wn + nt * 8 + gid) * RB + ks + 2 * tig;
                b[nt][0] = *(const uint32_t *)p; b[nt][1] = *(const uint32_t *)(p + 8);
            }
            for (int mt = 0; mt < 2; mt++)
                for (int nt = 0; nt < 2; nt++)
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                                 : "+f"(acc[mt][nt][0]), "+f"(acc[mt][nt][1]), "+f"(acc[mt][nt][2]), "+f"(acc[mt][nt][3])
                                 : "r"(a[mt][0]), "r"(a[mt][1]), "r"(a[mt][2]), "r"(a[mt][3]), "r"(b[nt][0]), "r"(b[nt][1]));
        }
        __syncthreads();
    }
    for (int mt = 0; mt < 2; mt++)
        for (int nt = 0; nt < 2; nt++)
            for (int v = 0; v < 4; v++) {
                int i = i0 + wm + mt * 16 + (v >> 1) * 8 + gid, j = j0 + wn + nt * 8 + 2 * tig + (v & 1);
                part[((size_t)z * Np + j) * M + i] = acc[mt][nt][v];
            }
}
__global__ void reduce_k(const float *part, float *Y, size_t n, int S) {
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
        float s = part[t];
        for (int z = 1; z < S; z++) s += part[z * n + t];
        Y[t] = s;
    }
}

// ---- host -----------------------------------------------------------------------------------------------------------
static std::vector<uint8_t> slurp(const std::string &p) {
    FILE *f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (fread(v.data(), 1, n, f) != (size_t)n) exit(1);
    fclose(f);
    return v;
}
static void dump(const std::string &p, const void *d, size_t n) { FILE *f = fopen(p.c_str(), "wb"); fwrite(d, 1, n, f); fclose(f); }

struct Dev {
    int M, K, G, Np;
    int8_t *Wq, *xs; uint8_t *sc, *flag; int *smax, *Et, *Emax, *stat, *nbad; float *Y, *part; __nv_bfloat16 *Wb, *X;
};

static float ms_since(cudaEvent_t a, cudaEvent_t b) { float t; cudaEventElapsedTime(&t, a, b); return t; }

// the exact product for the first N tokens; returns L (tiers) and fills stat
static int run_exact(Dev &d, int N, int *stat_h, int *nbad_h) {
    int Np = (N + BN - 1) / BN * BN;
    CK(cudaMemset(d.stat, 0, 3 * sizeof(int))); CK(cudaMemset(d.nbad, 0, sizeof(int)));
    CK(cudaMemset(d.Emax, 0, Np * sizeof(int)));   // 0 = -100000: an all-zero token
    split_x<<<(Np * d.G + 127) / 128, 128>>>(d.X, d.xs, d.Et, d.Emax, d.stat, Np, d.K);
    CK(cudaMemcpy(stat_h, d.stat, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    int L = std::max(stat_h[0], 1);
    mx_exact<<<dim3(Np / BN, d.M / BM), THREADS>>>(d.Wq, d.sc, d.smax, d.xs, d.Et, d.Emax, d.Y, d.flag, d.nbad, d.M, Np, d.K, L);
    CK(cudaGetLastError());
    CK(cudaMemcpy(nbad_h, d.nbad, sizeof(int), cudaMemcpyDeviceToHost));
    return L;
}
static void run_bf16(Dev &d, int N, int S, float *Y) {
    int Np = (N + BN - 1) / BN * BN;
    bf16_gemm<<<dim3(Np / BN, d.M / BM, S), THREADS>>>(d.Wb, d.X, d.part, d.M, Np, d.K, S);
    reduce_k<<<512, 256>>>(d.part, Y, (size_t)Np * d.M, S);
    CK(cudaGetLastError());
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <dir> <tensor prefix> <N> [seed] [outdir]\n", argv[0]); return 1; }
    std::string dir = argv[1], pre = argv[2];
    int N = atoi(argv[3]), seed = argc > 4 ? atoi(argv[4]) : 1;
    std::string out = argc > 5 ? argv[5] : "";
    // shapes from the file sizes: packed [M][K/2], scales [M][K/32]
    auto pk = slurp(dir + "/" + pre + ".weight_packed.bin"), sc = slurp(dir + "/" + pre + ".weight_scale.bin");
    long MK2 = pk.size(), MG = sc.size();   // MK2 / MG = 16
    if (MK2 != 16 * MG) { fprintf(stderr, "bad shapes\n"); return 1; }
    // M from meta: 3072 x 3584 (w1, w3) or 3584 x 3072 (w2)
    int M = pre.find(".w2") != std::string::npos ? 3584 : 3072, K = (int)(2 * MK2 / M), G = K / 32;
    int Np = (N + BN - 1) / BN * BN;
    // activations: heavy-tailed (Student t, 3 degrees of freedom: outliers as in real hidden states), times the real
    // routed_expert_norm weight when the input is the 3584-wide latent
    std::vector<float> nw(K, 1.f);
    {
        std::string np = dir + "/" + pre.substr(0, pre.find("experts.")) + "routed_expert_norm.weight.bin";
        FILE *f = fopen(np.c_str(), "rb");
        if (f && K == 3584) {
            std::vector<uint16_t> b(K);
            if (fread(b.data(), 2, K, f) == (size_t)K) for (int k = 0; k < K; k++) { uint32_t u = (uint32_t)b[k] << 16; memcpy(&nw[k], &u, 4); }
            printf("activations scaled by the real routed_expert_norm weight (|w| in [%.3g, %.3g])\n",
                   *std::min_element(nw.begin(), nw.end(), [](float a, float b) { return fabsf(a) < fabsf(b); }),
                   *std::max_element(nw.begin(), nw.end(), [](float a, float b) { return fabsf(a) < fabsf(b); }));
        }
        if (f) fclose(f);
    }
    std::mt19937_64 rng(seed);
    std::student_t_distribution<float> st(3.f);
    std::vector<__nv_bfloat16> Xh((size_t)Np * K, __float2bfloat16(0.f));
    for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) Xh[(size_t)j * K + k] = __float2bfloat16(st(rng) * nw[k]);

    Dev d; d.M = M; d.K = K; d.G = G; d.Np = Np;
    CK(cudaMalloc(&d.Wq, (size_t)M * K)); CK(cudaMalloc(&d.Wb, (size_t)M * K * 2)); CK(cudaMalloc(&d.sc, MG));
    CK(cudaMalloc(&d.smax, M * 4)); CK(cudaMalloc(&d.X, (size_t)Np * K * 2)); CK(cudaMalloc(&d.xs, (size_t)LMAX * 3 * Np * K));
    CK(cudaMalloc(&d.Et, (size_t)LMAX * Np * G * 4)); CK(cudaMalloc(&d.Emax, Np * 4)); CK(cudaMalloc(&d.stat, 12)); CK(cudaMalloc(&d.nbad, 4));
    CK(cudaMalloc(&d.Y, (size_t)Np * M * 4)); CK(cudaMalloc(&d.flag, (size_t)Np * M)); CK(cudaMalloc(&d.part, (size_t)8 * Np * M * 4));
    float *Yb; CK(cudaMalloc(&Yb, (size_t)Np * M * 4));
    // poison: a fresh cudaMalloc is zero, which hides tiles that are never written
    CK(cudaMemset(d.Y, 0x5a, (size_t)Np * M * 4)); CK(cudaMemset(d.xs, 0x5a, (size_t)LMAX * 3 * Np * K));
    uint8_t *pkd; CK(cudaMalloc(&pkd, MK2));
    CK(cudaMemcpy(pkd, pk.data(), MK2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.sc, sc.data(), MG, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.X, Xh.data(), (size_t)Np * K * 2, cudaMemcpyHostToDevice));
    unpack_w<<<1024, 256>>>(pkd, d.sc, d.Wq, d.Wb, M, K);
    row_smax<<<(M + 127) / 128, 128>>>(d.sc, d.smax, M, G);
    CK(cudaDeviceSynchronize());
    {
        int lo = 255, hi = 0, spread = 0;
        for (int i = 0; i < M; i++) {
            int a = 255, b = 0;
            for (int g = 0; g < G; g++) { a = std::min(a, (int)sc[(size_t)i * G + g]); b = std::max(b, (int)sc[(size_t)i * G + g]); }
            lo = std::min(lo, a); hi = std::max(hi, b); spread = std::max(spread, b - a);
        }
        printf("%s: M %d K %d, scale bytes %d..%d (2^%d..2^%d), widest row spread %d binades, N %d tokens\n",
               pre.c_str(), M, K, lo, hi, lo - 127, hi - 127, spread, N);
    }

    int stat[3], nbad;
    int L = run_exact(d, N, stat, &nbad);
    CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)Np * M), B((size_t)Np * M);
    std::vector<uint8_t> fl((size_t)Np * M);
    CK(cudaMemcpy(Y.data(), d.Y, Y.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(fl.data(), d.flag, fl.size(), cudaMemcpyDeviceToHost));
    printf("exact: %d tier(s), %d elements beyond %d tiers, %d inf/nan, %d of %zu outputs not certified\n", L, stat[1], LMAX, stat[2],
           nbad, (size_t)N * M);

    // the same products for smaller batches (the first n tokens): a correctly rounded result must not move
    for (int n : {1, 3, 17, 64, 200}) {
        if (n > N) continue;
        int st2[3], nb2;
        run_exact(d, n, st2, &nb2);
        std::vector<float> Z((size_t)n * M);
        CK(cudaMemcpy(Z.data(), d.Y, Z.size() * 4, cudaMemcpyDeviceToHost));
        size_t diff = 0;
        for (size_t t = 0; t < Z.size(); t++) diff += memcmp(&Z[t], &Y[t], 4) != 0;
        printf("  exact, batch %4d: %zu of %zu outputs differ from batch %d\n", n, diff, Z.size(), N);
    }
    // baseline: split-K 1, 2, 4, 8 (engines pick it from the shape) -- how many bits move, and against the exact
    std::vector<std::vector<float>> base;
    for (int S : {1, 2, 4, 8}) {
        run_bf16(d, N, S, Yb);
        CK(cudaMemcpy(B.data(), Yb, B.size() * 4, cudaMemcpyDeviceToHost));
        size_t neq = 0; double maxu = 0;
        for (size_t t = 0; t < (size_t)N * M; t++) {
            if (memcmp(&B[t], &Y[t], 4)) neq++;
            int32_t a, b; memcpy(&a, &B[t], 4); memcpy(&b, &Y[t], 4);
            if ((a < 0) == (b < 0)) maxu = std::max(maxu, fabs((double)a - b));
        }
        printf("  BF16 tensor cores, FP32 accumulation, split-K %d: %zu of %zu outputs (%.1f %%) not correctly rounded, worst %.0f ulp\n",
               S, neq, (size_t)N * M, 100.0 * neq / ((double)N * M), maxu);
        base.push_back(std::vector<float>(B.begin(), B.begin() + (size_t)N * M));
    }
    {
        size_t diff = 0;
        for (size_t t = 0; t < base[0].size(); t++) diff += memcmp(&base[0][t], &base[2][t], 4) != 0;
        printf("  BF16 split-K 1 vs 4: %zu outputs differ\n", diff);
    }
    if (!out.empty()) {
        dump(out + "/X.bf16", Xh.data(), (size_t)N * K * 2);
        dump(out + "/Y.f32", Y.data(), (size_t)N * M * 4);
        dump(out + "/Yb1.f32", base[0].data(), base[0].size() * 4);
        dump(out + "/Yb4.f32", base[2].data(), base[2].size() * 4);
        FILE *f = fopen((out + "/meta.txt").c_str(), "w"); fprintf(f, "%d %d %d %s\n", M, K, N, pre.c_str()); fclose(f);
    }

    // timing
    cudaEvent_t e0, e1, e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
    for (int n : {1, 16, 128, 1024, 4096}) {
        if (n > N) continue;
        int npd = (n + BN - 1) / BN * BN, st2[3], nb2;
        run_exact(d, n, st2, &nb2);
        const int R = n <= 128 ? 50 : 10;
        cudaEventRecord(e0);
        for (int r = 0; r < R; r++) split_x<<<(npd * G + 127) / 128, 128>>>(d.X, d.xs, d.Et, d.Emax, d.stat, npd, K);
        cudaEventRecord(e1);
        for (int r = 0; r < R; r++)
            mx_exact<<<dim3(npd / BN, M / BM), THREADS>>>(d.Wq, d.sc, d.smax, d.xs, d.Et, d.Emax, d.Y, d.flag, d.nbad, M, npd, K, std::max(st2[0], 1));
        cudaEventRecord(e2); CK(cudaEventSynchronize(e2));
        float ts = ms_since(e0, e1) / R, tg = ms_since(e1, e2) / R;
        cudaEventRecord(e0);
        for (int r = 0; r < R; r++) run_bf16(d, n, 1, Yb);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float tb = ms_since(e0, e1) / R;
        double fl2 = 2.0 * M * K * n;
        printf("  N %5d: exact %.3f ms (split %.3f + products %.3f) = %.1f TOPS; BF16 baseline %.3f ms = %.1f TFLOPS\n",
               n, ts + tg, ts, tg, fl2 / (ts + tg) / 1e9, tb, fl2 / tb / 1e9);
    }
    return 0;
}
