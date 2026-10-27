// mxexact.cu -- correctly rounded MXFP4 x BF16 matrix products on INT8 tensor cores, checked on the real weights of
// Kimi K3 (2.78 T parameters, the largest open-weights model; routed experts are MXFP4: E2M1 values, one E8M0 scale per 32).
//
// Why it is exact. A weight is w = q 2^(s-128) with q = 2 * E2M1 in {0, +-1, 2, 3, 4, 6, 8, 12} and s the scale byte of its
// group of 32. Within a row the scales span few binades (K3: at most 2), so q' = q 2^((s - smin) mod 4) is still an int8
// (|q'| <= 96) and the row keeps one scale per band of 4 binades (K3: one band). A BF16 activation is x = m 2^e, |m| < 256.
// Each token is aligned to its largest exponent E: A = m 2^(Dc - (E - e)) is an integer of 7S bits when E - e <= Dc = 7S - 8,
// cut into S int8 digits (the top one signed); smaller elements form the next tier with their own E. Then
//     y_ij = sum_{band a, tier l, digit d} 2^(smin_i + 4a - 128 + E_jl - Dc + 7d) * P_ald(i, j),   P_ald = W^(a) X^(l,d)
// and every P_ald is an ordinary INT8 GEMM whose int32 sums cannot overflow (|P| <= K * 96 * 128 < 2^26.4 for K <= 7168).
// The few terms of one output are added in a 128-bit integer anchored at the top exponent and rounded once to FP32
// (a term below the 94-bit window is floored and counted; the output is certified when both ends of the interval round to
// the same float, which K3 data always gave). A correctly rounded result does not depend on the summation order, the
// tiling, the batch size or the machine. Version 1 (commit 9b77801) aligned per group of 32 and paid a 128-bit
// shift-add per group: right, but 13x slower than BF16.
//
// The baseline is what inference engines do: dequantize to BF16 (exact) and use BF16 tensor cores with FP32
// accumulation, split-K chosen by shape. Its bits change with the split.
//
// nvcc -O3 -arch=sm_120 -std=c++17 -o out_mx llm/mxexact.cu
// out_mx <dir> <tensor prefix> <N> [digits S] [seed] [outdir]
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
#include "../src/gemm_i8.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int LMAX = 6, SMAX = 4, WIN = 94, EMPTY = -30000;
constexpr int BM = 64, BN = 64, THREADS = 256, NPAD = 128;

// ---- preprocessing ------------------------------------------------------------------------------------------------
// packed [M][K/2] (low nibble = even k), scales [M][K/32] -> Wt [A][M][K] int8 (q' of band a, 0 elsewhere), smin [M];
// Wb [M][K] = the same weights in BF16 (exact) for the baseline
__global__ void unpack_w(const uint8_t *pk, const uint8_t *sc, const int *smin, int8_t *Wt, __nv_bfloat16 *wb, int M, int K, int A) {
    const int8_t mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    size_t n = (size_t)M * K / 2, MK = (size_t)M * K;
    for (size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x; t < n; t += (size_t)gridDim.x * blockDim.x) {
        size_t i = t / (K / 2), k = 2 * (t % (K / 2));
        uint8_t b = pk[t];
        int s = sc[i * (K / 32) + k / 32], r = s - smin[i];
        for (int h = 0; h < 2; h++) {
            int nb = h ? b >> 4 : b & 15, v = mag[nb & 7] * (nb & 8 ? -1 : 1);
            for (int a = 0; a < A; a++) Wt[a * MK + i * K + k + h] = a == r / 4 ? (int8_t)(v << (r % 4)) : 0;
            wb[i * K + k + h] = __float2bfloat16_rn(ldexpf((float)v, s - 128));
        }
    }
}

// one block per token j: tiers over the whole row. X [Np][K] bf16 -> xs [LMAX][S][Np][K] int8 (tier l, digit d),
// Ec [LMAX][Np] = the tier's exponent E (EMPTY when unused); stat = {tiers used, elements beyond LMAX, inf/nan}
__global__ void split_x(const __nv_bfloat16 *X, int8_t *xs, int *Ec, int *stat, int Np, int K, int S) {
    const int j = blockIdx.x, Dc = 7 * S - 8;
    const size_t pl = (size_t)Np * K;
    __shared__ int red[32];
    int done = 0, lmax = 0;
    for (int l = 0; l < LMAX; l++) {
        int E = EMPTY;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            uint16_t b = __bfloat16_as_ushort(X[(size_t)j * K + k]);
            int ex = (b >> 7) & 255, e = ex ? ex - 134 : -133;
            if ((b & 0x7FFF) && ex != 255) {
                // already placed in a tier above?  (it was placed iff its gap to that tier's E was <= Dc)
                bool placed = false;
                for (int u = 0; u < l; u++) { int Eu = Ec[(size_t)u * Np + j]; if (Eu != EMPTY && Eu - e <= Dc && e <= Eu) { placed = true; break; } }
                if (!placed) E = max(E, e);
            }
        }
        for (int o = 16; o; o >>= 1) E = max(E, __shfl_xor_sync(~0u, E, o));
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = E;
        __syncthreads();
        if (threadIdx.x < 32) {
            int v = threadIdx.x < blockDim.x / 32 ? red[threadIdx.x] : EMPTY;
            for (int o = 16; o; o >>= 1) v = max(v, __shfl_xor_sync(~0u, v, o));
            if (threadIdx.x == 0) red[0] = v;
        }
        __syncthreads();
        E = red[0];
        __syncthreads();
        if (threadIdx.x == 0) Ec[(size_t)l * Np + j] = E;
        if (E != EMPTY) lmax = l + 1;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            uint16_t b = __bfloat16_as_ushort(X[(size_t)j * K + k]);
            int ex = (b >> 7) & 255, man = b & 127, m = ex ? (man | 128) : man, e = ex ? ex - 134 : -133, Av = 0;
            if (b & 0x8000) m = -m;
            if (E != EMPTY && m && ex != 255 && e <= E && E - e <= Dc) {
                bool above = false;   // belongs to an earlier tier
                for (int u = 0; u < l; u++) { int Eu = Ec[(size_t)u * Np + j]; if (Eu != EMPTY && Eu - e <= Dc && e <= Eu) { above = true; break; } }
                if (!above) { Av = m * (1 << (Dc - (E - e))); done++; }
            }
            for (int d = 0; d < S; d++) xs[((size_t)l * S + d) * pl + (size_t)j * K + k] = (int8_t)(d < S - 1 ? (Av >> (7 * d)) & 127 : Av >> (7 * d));
        }
        __syncthreads();   // Ec[l] visible before the next tier reads it
    }
    // count: nonzero finite elements that found no tier
    int nz = 0, bad = 0;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        uint16_t b = __bfloat16_as_ushort(X[(size_t)j * K + k]);
        if (((b >> 7) & 255) == 255) bad++;
        else if (b & 0x7FFF) nz++;
    }
    if (nz - done) atomicAdd(&stat[1], nz - done);
    if (bad) atomicAdd(&stat[2], bad);
    if (threadIdx.x == 0) atomicMax(&stat[0], lmax);
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
// P [A*L*S][M][Np] int32: z = (a * L + l) * S + d is W^(a) X^(l,d); plain INT8 GEMMs (src/gemm_i8.cuh, 157 TOPS)
__global__ void __launch_bounds__(oz::THREADS) gemm_parts(const int8_t *__restrict__ Wt, const int8_t *__restrict__ xs,
                                                          int32_t *__restrict__ P, int M, int Np, int K, int L, int S) {
    const int z = blockIdx.z, a = z / (L * S), ld = z % (L * S);
    const int8_t *A = Wt + (size_t)a * M * K, *Bt = xs + (size_t)ld * Np * K;   // xs is [LMAX][S]: (l, d) = l * S + d
    int32_t *C = P + (size_t)z * M * Np;
    oz::gemm_i8_tile(A, Bt, K, [&](int row, int col, int c00, int c01, int c10, int c11) {
        *(int2 *)&C[(size_t)row * Np + col] = make_int2(c00, c01);
        *(int2 *)&C[(size_t)(row + 8) * Np + col] = make_int2(c10, c11);
    });
}

// Y [Np][M] = the correctly rounded sum of the A*L*S terms of each output; flag + nbad when the certificate fails
__global__ void combine(const int32_t *__restrict__ P, const int *__restrict__ smin, const int *__restrict__ Ec, float *Y,
                        uint8_t *flag, int *nbad, int M, int Np, int A, int L, int S) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (i >= M) return;
    const int Dc = 7 * S - 8, E0 = Ec[j];
    float f = 0.f;
    bool ok = true;
    if (E0 != EMPTY) {
        const int top = smin[i] + 4 * (A - 1) - 128 + E0 - Dc + 7 * (S - 1), base = top - WIN;
        __int128 acc = 0;
        int lost = 0;
        for (int a = 0; a < A; a++)
            for (int l = 0; l < L; l++) {
                const int El = Ec[(size_t)l * Np + j];
                if (El == EMPTY) continue;
                for (int d = 0; d < S; d++) {
                    int p = P[((size_t)((a * L + l) * S + d) * M + i) * Np + j];
                    if (!p) continue;
                    int sh = smin[i] + 4 * a - 128 + El - Dc + 7 * d - base;
                    if (sh >= 0) acc += (__int128)p << sh;
                    else { acc += (__int128)(p >> min(-sh, 31)); lost++; }
                }
            }
        f = round_i128(acc, base);
        ok = !lost || round_i128(acc + lost, base) == f;
    }
    Y[(size_t)j * M + i] = f;
    flag[(size_t)j * M + i] = !ok;
    if (!ok) atomicAdd(nbad, 1);
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

// ---- decode: a few tokens, weights read once as packed 4-bit ---------------------------------------------------------
// Token generation reads every weight once per token, so it is bound by memory (or disk), not by arithmetic. This kernel
// unpacks the nibbles in registers (q' = E2M1 * 2 << (s - smin), one band), keeps the L*S int32 sums of all tiers and
// digits per output, splits K over the 8 warps of a block (integer partial sums: the order cannot matter) and rounds once.
// Block: 16 rows x 8 tokens; Np = 8. Result is bit-identical to the prefill path by construction (both are correctly rounded).
template <int L, int S>
__global__ void __launch_bounds__(256) mx_decode(const uint8_t *__restrict__ pk, const uint8_t *__restrict__ sc,
        const int *__restrict__ smin, const int8_t *__restrict__ xs, const int *__restrict__ Ec, float *Y, int M, int K) {
    constexpr int Np = 8;
    __shared__ int red[L * S][16][8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3, i0 = blockIdx.x * 16;
    const int G = K / 32, g0 = warp * G / 8, g1 = (warp + 1) * G / 8;
    for (int t = tid; t < L * S * 128; t += 256) (&red[0][0][0])[t] = 0;
    int acc[L * S][4] = {};
    const uint8_t *p0 = pk + (size_t)(i0 + gid) * (K / 2), *p1 = p0 + 8 * (K / 2);
    const int sm0 = smin[i0 + gid], sm1 = smin[i0 + gid + 8];
    const uint32_t LUT = 0xC8643210u;   // 2 * E2M1 magnitudes 0 1 2 3 4 6 8 12, one per nibble
    auto unpack4 = [&](uint32_t b2, int sh) {   // 2 packed bytes = 4 weights, low nibble first -> 4 int8
        uint32_t r = 0;
#pragma unroll
        for (int h = 0; h < 4; h++) {
            uint32_t nb = (b2 >> (4 * h)) & 15, v = ((LUT >> (4 * (nb & 7))) & 15) << sh;
            r |= ((nb & 8 ? (0u - v) : v) & 255u) << (8 * h);
        }
        return r;
    };
    for (int g = g0; g < g1; g++) {
        const int k0 = g * 32 + tig * 4;
        const int s0 = sc[(size_t)(i0 + gid) * G + g] - sm0, s1 = sc[(size_t)(i0 + gid + 8) * G + g] - sm1;
        uint32_t a0 = unpack4(*(const uint16_t *)(p0 + k0 / 2), s0), a1 = unpack4(*(const uint16_t *)(p1 + k0 / 2), s1);
        uint32_t a2 = unpack4(*(const uint16_t *)(p0 + k0 / 2 + 8), s0), a3 = unpack4(*(const uint16_t *)(p1 + k0 / 2 + 8), s1);
#pragma unroll
        for (int z = 0; z < L * S; z++) {
            const int8_t *xb = xs + ((size_t)z * Np + gid) * K + k0;
            uint32_t b0 = *(const uint32_t *)xb, b1 = *(const uint32_t *)(xb + 16);
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                         : "+r"(acc[z][0]), "+r"(acc[z][1]), "+r"(acc[z][2]), "+r"(acc[z][3])
                         : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    __syncthreads();
#pragma unroll
    for (int z = 0; z < L * S; z++)
#pragma unroll
        for (int v = 0; v < 4; v++) atomicAdd(&red[z][gid + (v >> 1) * 8][2 * tig + (v & 1)], acc[z][v]);
    __syncthreads();
    if (tid < 128) {
        const int r = tid >> 3, j = tid & 7, i = i0 + r, E0 = Ec[j], Dc = 7 * S - 8;
        float f = 0.f;
        if (E0 != EMPTY) {
            const int base = smin[i] - 128 + E0 - Dc + 7 * (S - 1) - WIN;
            __int128 a = 0;
            int lost = 0;
#pragma unroll
            for (int l = 0; l < L; l++) {
                const int El = Ec[l * Np + j];
                if (El == EMPTY) continue;
#pragma unroll
                for (int d = 0; d < S; d++) {
                    int p = red[l * S + d][r][j], sh = smin[i] - 128 + El - Dc + 7 * d - base;
                    if (!p) continue;
                    if (sh >= 0) a += (__int128)p << sh;
                    else { a += (__int128)(p >> min(-sh, 31)); lost++; }
                }
            }
            f = round_i128(a, base);
            if (lost && round_i128(a + lost, base) != f) f = __int_as_float(0x7fc00000);   // uncertified: NaN, never seen
        }
        Y[(size_t)j * M + i] = f;
    }
}

// what engines do for decode: dequantize the nibbles in registers, FP32 multiply-add, warp reduction. One warp per row.
__global__ void fp4_gemv(const uint8_t *__restrict__ pk, const uint8_t *__restrict__ sc, const __nv_bfloat16 *__restrict__ X,
                         float *Y, int M, int K, int N) {
    const int lane = threadIdx.x & 31, i = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (i >= M) return;
    const float val[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float acc[8] = {};
    for (int k2 = lane; k2 < K / 2; k2 += 32) {   // one packed byte = two weights per lane and step
        uint8_t b = pk[(size_t)i * (K / 2) + k2];
        float sc2 = ldexpf(1.f, (int)sc[(size_t)i * (K / 32) + 2 * k2 / 32] - 127);
        float w0 = (b & 8 ? -val[b & 7] : val[b & 7]) * sc2, w1 = (b & 128 ? -val[(b >> 4) & 7] : val[(b >> 4) & 7]) * sc2;
        for (int j = 0; j < N; j++)
            acc[j] = fmaf(w1, __bfloat162float(X[(size_t)j * K + 2 * k2 + 1]), fmaf(w0, __bfloat162float(X[(size_t)j * K + 2 * k2]), acc[j]));
    }
    for (int j = 0; j < N; j++) {
        float v = acc[j];
        for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
        if (!lane) Y[(size_t)j * M + i] = v;
    }
}

typedef void (*DecodeFn)(const uint8_t *, const uint8_t *, const int *, const int8_t *, const int *, float *, int, int);
static DecodeFn decode_fn(int L, int S) {
#define DF(l) (S == 3 ? (DecodeFn)mx_decode<l, 3> : (DecodeFn)mx_decode<l, 4>)
    switch (L) { case 1: return DF(1); case 2: return DF(2); case 3: return DF(3); case 4: return DF(4); default: return nullptr; }
#undef DF
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
    int M, K, G, Np, A, S;
    int8_t *Wt, *xs; uint8_t *sc, *flag; int *smin, *Ec, *stat, *nbad; int32_t *P; float *Y, *part; __nv_bfloat16 *Wb, *X;
};

static float ms_since(cudaEvent_t a, cudaEvent_t b) { float t; cudaEventElapsedTime(&t, a, b); return t; }

// the exact product for the first N tokens (stat: tiers, lost elements, inf/nan; returns the tiers used)
static int run_split(Dev &d, int N, int *stat_h) {
    int Np = (N + NPAD - 1) / NPAD * NPAD;
    CK(cudaMemset(d.stat, 0, 3 * sizeof(int)));
    split_x<<<Np, 256>>>(d.X, d.xs, d.Ec, d.stat, Np, d.K, d.S);
    CK(cudaMemcpy(stat_h, d.stat, 3 * sizeof(int), cudaMemcpyDeviceToHost));
    return std::max(stat_h[0], 1);
}
static void run_products(Dev &d, int N, int L) {
    int Np = (N + NPAD - 1) / NPAD * NPAD;
    CK(cudaMemset(d.nbad, 0, sizeof(int)));
    gemm_parts<<<dim3(Np / oz::BN, d.M / oz::BM, d.A * L * d.S), oz::THREADS>>>(d.Wt, d.xs, d.P, d.M, Np, d.K, L, d.S);
    combine<<<dim3((d.M + 255) / 256, Np), 256>>>(d.P, d.smin, d.Ec, d.Y, d.flag, d.nbad, d.M, Np, d.A, L, d.S);
    CK(cudaGetLastError());
}
static int run_exact(Dev &d, int N, int *stat_h, int *nbad_h) {
    int L = run_split(d, N, stat_h);
    run_products(d, N, L);
    CK(cudaMemcpy(nbad_h, d.nbad, sizeof(int), cudaMemcpyDeviceToHost));
    return L;
}
static void run_bf16(Dev &d, int N, int S, float *Y) {
    int Np = (N + NPAD - 1) / NPAD * NPAD;
    bf16_gemm<<<dim3(Np / BN, d.M / BM, S), THREADS>>>(d.Wb, d.X, d.part, d.M, Np, d.K, S);
    reduce_k<<<512, 256>>>(d.part, Y, (size_t)Np * d.M, S);
    CK(cudaGetLastError());
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <dir> <tensor prefix> <N> [digits] [seed] [outdir]\n", argv[0]); return 1; }
    std::string dir = argv[1], pre = argv[2];
    int N = atoi(argv[3]), S = argc > 4 ? atoi(argv[4]) : 3, seed = argc > 5 ? atoi(argv[5]) : 1;
    std::string out = argc > 6 ? argv[6] : "";
    if (S < 2 || S > SMAX) { fprintf(stderr, "digits 2..%d\n", SMAX); return 1; }
    auto pk = slurp(dir + "/" + pre + ".weight_packed.bin"), sc = slurp(dir + "/" + pre + ".weight_scale.bin");
    long MK2 = pk.size(), MG = sc.size();
    if (MK2 != 16 * MG) { fprintf(stderr, "bad shapes\n"); return 1; }
    int M = pre.find(".w2") != std::string::npos ? 3584 : 3072, K = (int)(2 * MK2 / M), G = K / 32;
    int Np = (N + NPAD - 1) / NPAD * NPAD;
    // activations: heavy-tailed (Student t, 3 degrees of freedom: outliers as in real hidden states), times the real
    // routed_expert_norm weight when the input is the 3584-wide latent
    std::vector<float> nw(K, 1.f);
    {
        std::string np = dir + "/" + pre.substr(0, pre.find("experts.")) + "routed_expert_norm.weight.bin";
        FILE *f = fopen(np.c_str(), "rb");
        if (f && K == 3584) {
            std::vector<uint16_t> b(K);
            if (fread(b.data(), 2, K, f) == (size_t)K) for (int k = 0; k < K; k++) { uint32_t u = (uint32_t)b[k] << 16; memcpy(&nw[k], &u, 4); }
            printf("activations scaled by the real routed_expert_norm weight\n");
        }
        if (f) fclose(f);
    }
    std::mt19937_64 rng(seed);
    std::student_t_distribution<float> st(3.f);
    std::vector<__nv_bfloat16> Xh((size_t)Np * K, __float2bfloat16(0.f));
    for (int j = 0; j < N; j++) for (int k = 0; k < K; k++) Xh[(size_t)j * K + k] = __float2bfloat16(st(rng) * nw[k]);

    // scales: smin per row, bands of 4 binades
    std::vector<int> smin(M);
    int A = 1, lo = 255, hi = 0, spread = 0;
    for (int i = 0; i < M; i++) {
        int a = 255, b = 0;
        for (int g = 0; g < G; g++) { a = std::min(a, (int)sc[(size_t)i * G + g]); b = std::max(b, (int)sc[(size_t)i * G + g]); }
        smin[i] = a; lo = std::min(lo, a); hi = std::max(hi, b); spread = std::max(spread, b - a); A = std::max(A, (b - a) / 4 + 1);
    }
    printf("%s: M %d K %d, scales 2^%d..2^%d, widest row %d binades -> %d band(s); N %d tokens, %d digits (Dc %d)\n",
           pre.c_str(), M, K, lo - 127, hi - 127, spread, A, N, S, 7 * S - 8);

    Dev d; d.M = M; d.K = K; d.G = G; d.Np = Np; d.A = A; d.S = S;
    CK(cudaMalloc(&d.Wt, (size_t)A * M * K)); CK(cudaMalloc(&d.Wb, (size_t)M * K * 2)); CK(cudaMalloc(&d.sc, MG));
    CK(cudaMalloc(&d.smin, M * 4)); CK(cudaMalloc(&d.X, (size_t)Np * K * 2)); CK(cudaMalloc(&d.xs, (size_t)LMAX * S * Np * K));
    CK(cudaMalloc(&d.Ec, (size_t)LMAX * Np * 4)); CK(cudaMalloc(&d.stat, 12)); CK(cudaMalloc(&d.nbad, 4));
    CK(cudaMalloc(&d.P, (size_t)A * LMAX * S * M * Np * 4));
    CK(cudaMalloc(&d.Y, (size_t)Np * M * 4)); CK(cudaMalloc(&d.flag, (size_t)Np * M)); CK(cudaMalloc(&d.part, (size_t)8 * Np * M * 4));
    float *Yb; CK(cudaMalloc(&Yb, (size_t)Np * M * 4));
    // poison: a fresh cudaMalloc is zero, which hides tiles that are never written
    CK(cudaMemset(d.Y, 0x5a, (size_t)Np * M * 4)); CK(cudaMemset(d.P, 0x5a, (size_t)A * LMAX * S * M * Np * 4));
    uint8_t *pkd; CK(cudaMalloc(&pkd, MK2));
    CK(cudaMemcpy(pkd, pk.data(), MK2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(d.sc, sc.data(), MG, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.smin, smin.data(), M * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.X, Xh.data(), (size_t)Np * K * 2, cudaMemcpyHostToDevice));
    unpack_w<<<1024, 256>>>(pkd, d.sc, d.smin, d.Wt, d.Wb, M, K, A);
    CK(cudaDeviceSynchronize());

    int stat[3], nbad;
    int L = run_exact(d, N, stat, &nbad);
    CK(cudaDeviceSynchronize());
    std::vector<float> Y((size_t)Np * M), B((size_t)Np * M);
    CK(cudaMemcpy(Y.data(), d.Y, Y.size() * 4, cudaMemcpyDeviceToHost));
    printf("exact: %d tier(s), %d elements found no tier, %d inf/nan, %d of %zu outputs not certified\n", L, stat[1], stat[2],
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
    for (int s : {1, 2, 4, 8}) {
        run_bf16(d, N, s, Yb);
        CK(cudaMemcpy(B.data(), Yb, B.size() * 4, cudaMemcpyDeviceToHost));
        size_t neq = 0;
        double rel = 0;
        for (size_t t = 0; t < (size_t)N * M; t++) {
            if (memcmp(&B[t], &Y[t], 4)) neq++;
            if (Y[t] != 0) rel = std::max(rel, fabs((double)B[t] - Y[t]) / fabs((double)Y[t]));
        }
        printf("  BF16 tensor cores, FP32 accumulation, split-K %d: %zu of %zu outputs (%.1f %%) not correctly rounded, worst relative error %.2g\n",
               s, neq, (size_t)N * M, 100.0 * neq / ((double)N * M), rel);
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

    // decode: 1..8 tokens through the packed-4-bit kernel; must equal the prefill result bit for bit
    if (A == 1) {
        __nv_bfloat16 *Xd; float *Yd; int8_t *xd; int *Ed;
        CK(cudaMalloc(&Xd, 8 * K * 2)); CK(cudaMalloc(&Yd, 8 * M * 4)); CK(cudaMalloc(&xd, (size_t)LMAX * S * 8 * K)); CK(cudaMalloc(&Ed, LMAX * 8 * 4));
        cudaEvent_t e0, e1, e2; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
        std::vector<float> Z(8 * M);
        for (int n : {1, 2, 4, 8}) {
            CK(cudaMemset(Xd, 0, 8 * K * 2)); CK(cudaMemcpy(Xd, d.X, (size_t)n * K * 2, cudaMemcpyDeviceToDevice));
            CK(cudaMemset(d.stat, 0, 12));
            split_x<<<8, 256>>>(Xd, xd, Ed, d.stat, 8, K, S);
            int st2[3]; CK(cudaMemcpy(st2, d.stat, 12, cudaMemcpyDeviceToHost));
            DecodeFn fn = decode_fn(std::max(st2[0], 1), S);
            if (!fn) { printf("  decode: %d tiers, no instance\n", st2[0]); continue; }
            CK(cudaMemset(Yd, 0x5a, 8 * M * 4));
            fn<<<M / 16, 256>>>(pkd, d.sc, d.smin, xd, Ed, Yd, M, K);
            CK(cudaGetLastError());
            CK(cudaMemcpy(Z.data(), Yd, (size_t)n * M * 4, cudaMemcpyDeviceToHost));
            size_t diff = 0, nb = 0;
            for (size_t t = 0; t < (size_t)n * M; t++) diff += memcmp(&Z[t], &Y[t], 4) != 0;
            fp4_gemv<<<M / 8, 256>>>(pkd, d.sc, Xd, Yb, M, K, n);
            CK(cudaMemcpy(Z.data(), Yb, (size_t)n * M * 4, cudaMemcpyDeviceToHost));
            for (size_t t = 0; t < (size_t)n * M; t++) nb += memcmp(&Z[t], &Y[t], 4) != 0;
            const int R = 200;
            cudaEventRecord(e0);
            for (int r = 0; r < R; r++) { split_x<<<8, 256>>>(Xd, xd, Ed, d.stat, 8, K, S); fn<<<M / 16, 256>>>(pkd, d.sc, d.smin, xd, Ed, Yd, M, K); }
            cudaEventRecord(e1);
            for (int r = 0; r < R; r++) fp4_gemv<<<M / 8, 256>>>(pkd, d.sc, Xd, Yb, M, K, n);
            cudaEventRecord(e2); CK(cudaEventSynchronize(e2));
            float te = ms_since(e0, e1) / R, tb = ms_since(e1, e2) / R;
            double by = MK2 + MG;
            printf("  decode %d token(s): exact %.4f ms (%.0f GB/s of packed weights), %zu outputs differ from prefill; "
                   "FP4 GEMV FP32 %.4f ms (%.0f GB/s), %zu of %zu not correctly rounded\n",
                   n, te, by / te / 1e6, diff, tb, by / tb / 1e6, nb, (size_t)n * M);
        }
    }

    // timing
    cudaEvent_t e0, e1, e2, e3; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);
    for (int n : {128, 1024, 4096}) {
        if (n > N) continue;
        int st2[3], nb2, npd = (n + NPAD - 1) / NPAD * NPAD;
        int Ln = run_exact(d, n, st2, &nb2);
        const int R = n <= 128 ? 50 : 10;
        cudaEventRecord(e0);
        for (int r = 0; r < R; r++) split_x<<<npd, 256>>>(d.X, d.xs, d.Ec, d.stat, npd, K, S);
        cudaEventRecord(e1);
        for (int r = 0; r < R; r++) gemm_parts<<<dim3(npd / oz::BN, M / oz::BM, A * Ln * S), oz::THREADS>>>(d.Wt, d.xs, d.P, M, npd, K, Ln, S);
        cudaEventRecord(e2);
        for (int r = 0; r < R; r++) combine<<<dim3((M + 255) / 256, npd), 256>>>(d.P, d.smin, d.Ec, d.Y, d.flag, d.nbad, M, npd, A, Ln, S);
        cudaEventRecord(e3); CK(cudaEventSynchronize(e3));
        float ts = ms_since(e0, e1) / R, tg = ms_since(e1, e2) / R, tc = ms_since(e2, e3) / R;
        cudaEventRecord(e0);
        for (int r = 0; r < R; r++) run_bf16(d, n, 1, Yb);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float tb = ms_since(e0, e1) / R;
        double fl2 = 2.0 * M * K * n;
        printf("  N %5d: exact %.3f ms (split %.3f, %d INT8 GEMMs %.3f = %.0f TOPS each, combine %.3f) = %.1f TFLOPS-equivalent; "
               "BF16 baseline %.3f ms = %.1f TFLOPS\n", n, ts + tg + tc, ts, A * Ln * S, tg, fl2 * A * Ln * S / tg / 1e9, tc,
               fl2 / (ts + tg + tc) / 1e9, tb, fl2 / tb / 1e9);
    }
    return 0;
}
