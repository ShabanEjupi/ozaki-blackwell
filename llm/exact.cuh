// exact.cuh -- correctly rounded products of narrow floating-point matrices on INT8 tensor cores, general form.
// A matrix is held as integer digits: value(r, k) = sum_t sum_d 2^(B[t][r] + 7 d) * dig[t][d][r][k], with T tiers per row
// (exponent windows of Dc = 7S - 8 bits below the tier's top), S int8 digits (low ones in [0, 127], the top one signed).
// BF16 rows are split here (split_rows); MXFP4 weights are one digit per band (mxexact.cu builds them).
// The product of two such matrices is a sum over (tier, digit) pairs of plain INT8 GEMMs, |P| <= K 2^14 < 2^27 for
// K <= 8192, added per output in a 128-bit integer and rounded once (to FP32 or BF16, nearest even). A term below the
// window is floored and counted; the output is certified when both ends of the interval round alike (always so far).
#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include "../src/gemm_i8.cuh"

namespace ex {

constexpr int TMAX = 6, WIN = 88, EMPTY = -30000;

struct Digits {
    int8_t *dig;   // [T][S][rows][K]
    int *B;        // [T][rows]: base exponent of the tier (EMPTY when the row has nothing there)
    int T, S, rows, K;
    __host__ __device__ size_t plane() const { return (size_t)rows * K; }
};

// one block per row: tiers over the whole row of BF16 values. stat = {tiers used (max), elements left over, inf/nan}
__global__ void split_rows(const __nv_bfloat16 *X, int8_t *dig, int *Bt, int *stat, int rows, int K, int S, int tmax = TMAX) {
    const int j = blockIdx.x, Dc = 7 * S - 8;
    const size_t pl = (size_t)rows * K;
    __shared__ int red[32], Es[TMAX];
    int done = 0, lmax = 0;
    for (int l = 0; l < tmax; l++) {
        int E = EMPTY;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            uint16_t b = __bfloat16_as_ushort(X[(size_t)j * K + k]);
            int ex = (b >> 7) & 255, e = ex ? ex - 134 : -133;
            if (!(b & 0x7FFF) || ex == 255) continue;
            bool placed = false;
            for (int u = 0; u < l; u++) if (Es[u] != EMPTY && e <= Es[u] && Es[u] - e <= Dc) { placed = true; break; }
            if (!placed) E = max(E, e);
        }
        for (int o = 16; o; o >>= 1) E = max(E, __shfl_xor_sync(~0u, E, o));
        if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = E;
        __syncthreads();
        if (threadIdx.x == 0) {
            int v = EMPTY;
            for (int w = 0; w < (int)blockDim.x / 32; w++) v = max(v, red[w]);
            Es[l] = v;
            Bt[(size_t)l * rows + j] = v == EMPTY ? EMPTY : v - Dc;
        }
        __syncthreads();
        E = Es[l];
        if (E != EMPTY) lmax = l + 1;
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            uint16_t b = __bfloat16_as_ushort(X[(size_t)j * K + k]);
            int ex = (b >> 7) & 255, man = b & 127, m = ex ? (man | 128) : man, e = ex ? ex - 134 : -133, Av = 0;
            if (b & 0x8000) m = -m;
            if (E != EMPTY && m && ex != 255 && e <= E && E - e <= Dc) {
                bool above = false;
                for (int u = 0; u < l; u++) if (Es[u] != EMPTY && e <= Es[u] && Es[u] - e <= Dc) { above = true; break; }
                if (!above) { Av = m * (1 << (Dc - (E - e))); done++; }
            }
            for (int d = 0; d < S; d++) dig[((size_t)l * S + d) * pl + (size_t)j * K + k] = (int8_t)(d < S - 1 ? (Av >> (7 * d)) & 127 : Av >> (7 * d));
        }
        __syncthreads();
    }
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

// a * 2^X rounded to nearest even with p significand bits (24: FP32, 8: BF16), subnormals and overflow included
__device__ __host__ inline double round_i128(__int128 a, int X, int p) {
    if (a == 0) return 0.0;
    bool neg = a < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-a) : (unsigned __int128)a;
    uint64_t hi = (uint64_t)(u >> 64), lo = (uint64_t)u;
#ifdef __CUDA_ARCH__
    int nb = hi ? 128 - __clzll(hi) : 64 - __clzll(lo);
#else
    int nb = hi ? 128 - __builtin_clzll(hi) : 64 - __builtin_clzll(lo);
#endif
    const int emin = -149 + (24 - p);   // the subnormal ulp of the format: FP32 2^-149, BF16 2^-133
    int Ev = nb - 1 + X, L = Ev - (p - 1) > emin ? Ev - (p - 1) : emin, r = L - X;
    double f;
    if (r <= 0) f = ldexp((double)(uint64_t)u, X);
    else if (r > 128) f = 0.0;
    else {
        unsigned __int128 q = r == 128 ? 0 : u >> r, rem = u - (r == 128 ? 0 : q << r), half = (unsigned __int128)1 << (r - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        f = ldexp((double)(uint64_t)q, L);
    }
    const double big = p == 24 ? 0x1.fffffep127 : 0x1.fep127;
    if (f > big) f = INFINITY;
    return neg ? -f : f;
}

// MXFP4 weights as Digits with one digit: packed [M][K/2] (low nibble = even k) and E8M0 scales [M][K/32] ->
// dig[a][0][i][k] = 2 E2M1 << ((s - smin_i) mod 4) when (s - smin_i) / 4 == a, else 0; B[a][i] = smin_i + 4 a - 128.
// One block per row. A (bands) from the host: (max row spread) / 4 + 1.
__global__ void mx_digits(const uint8_t *pk, const uint8_t *sc, int8_t *dig, int *B, int M, int K, int A) {
    const int i = blockIdx.x, G = K / 32;
    __shared__ int smin;
    if (threadIdx.x == 0) { int m = 255; for (int g = 0; g < G; g++) m = min(m, (int)sc[(size_t)i * G + g]); smin = m; }
    __syncthreads();
    if (threadIdx.x < A) B[(size_t)threadIdx.x * M + i] = smin + 4 * threadIdx.x - 128;
    const uint32_t LUT = 0xC8643210u;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const uint8_t b = pk[(size_t)i * (K / 2) + k / 2];
        const int nb = k & 1 ? b >> 4 : b & 15, r = sc[(size_t)i * G + k / 32] - smin;
        const int v = (int)((LUT >> (4 * (nb & 7))) & 15) << (r % 4);
        for (int a = 0; a < A; a++) dig[((size_t)a * M + i) * K + k] = a == r / 4 ? (int8_t)(nb & 8 ? -v : v) : 0;
    }
}

// P[z][M][Np], z = ((tw Sw + dw) Tx + tx) Sx + dx: the INT8 GEMM of one digit plane of W with one of X
__global__ void __launch_bounds__(oz::THREADS) gemm_pairs(Digits W, Digits X, int32_t *__restrict__ P, int Tx) {
    const int z = blockIdx.z, zx = z % (Tx * X.S), zw = z / (Tx * X.S);
    const int8_t *A = W.dig + (size_t)zw * W.plane(), *Bt = X.dig + (size_t)zx * X.plane();
    int32_t *C = P + (size_t)z * W.rows * X.rows;
    const int Np = X.rows;
    oz::gemm_i8_tile(A, Bt, W.K, [&](int row, int col, int c00, int c01, int c10, int c11) {
        *(int2 *)&C[(size_t)row * Np + col] = make_int2(c00, c01);
        *(int2 *)&C[(size_t)(row + 8) * Np + col] = make_int2(c10, c11);
    });
}

// Y[j][i] (ldy apart) = the correctly rounded sum of the Tw Sw Tx Sx terms (p = 24: float, 8: bf16 stored as float)
__global__ void combine(const int32_t *__restrict__ P, Digits W, Digits X, int Tw, int Tx, int p, float *Y, int ldy, int *nbad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y, M = W.rows, Np = X.rows;
    if (i >= M) return;
    int tw_top = EMPTY, tx_top = EMPTY;
    for (int t = 0; t < Tw; t++) tw_top = max(tw_top, W.B[(size_t)t * M + i]);
    for (int t = 0; t < Tx; t++) tx_top = max(tx_top, X.B[(size_t)t * Np + j]);
    float f = 0.f;
    if (tw_top != EMPTY && tx_top != EMPTY) {
        const int base = tw_top + tx_top + 7 * (W.S - 1 + X.S - 1) - WIN;
        __int128 acc = 0;
        int lost = 0;
        for (int tw = 0; tw < Tw; tw++) {
            const int bw = W.B[(size_t)tw * M + i];
            if (bw == EMPTY) continue;
            for (int dw = 0; dw < W.S; dw++)
                for (int tx = 0; tx < Tx; tx++) {
                    const int bx = X.B[(size_t)tx * Np + j];
                    if (bx == EMPTY) continue;
                    for (int dx = 0; dx < X.S; dx++) {
                        const int z = ((tw * W.S + dw) * Tx + tx) * X.S + dx;
                        const int v = P[((size_t)z * M + i) * Np + j];
                        if (!v) continue;
                        const int sh = bw + bx + 7 * (dw + dx) - base;
                        if (sh >= 0) acc += (__int128)v << sh;
                        else { acc += (__int128)(v >> min(-sh, 31)); lost++; }
                    }
                }
        }
        f = (float)round_i128(acc, base, p);
        if (lost && (float)round_i128(acc + lost, base, p) != f) { atomicAdd(nbad, 1); f = __int_as_float(0x7fc00000); }
    }
    Y[(size_t)j * ldy + i] = f;
}

constexpr int BM = 64, BN = 64, THREADS = 256;
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

}  // namespace ex
