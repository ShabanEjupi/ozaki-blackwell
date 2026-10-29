// strassen_mod.cuh -- C = A B mod m on the INT8 tensor cores with 7^L block products instead of 8^L, for ANY modulus m <= 256
// (every Ozaki-II modulus, the composites and 256 included: Strassen needs a ring, not a field). A sum of residues reduced
// mod m is again a residue, so the operand sums stay int8 at every level and the result is bit-identical to the classical
// product mod m: over Z/m Strassen has no stability question. B is given transposed (Bt, K contiguous), as in gemm_i8.cuh.
// Two levels run as ONE bilinear algorithm with 49 products whose coefficients are Kronecker products (U x U, V x V, W x W):
// each of the 16 sub-blocks is read once, the 49 operands written once, the 49 products are one batched launch, and C's 16
// sub-blocks are written once from them, never read. L = 4 is two levels over two.
#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "gemm_i8.cuh"

namespace oz {

// v - m rint(v / m) is v mod m in [-m/2, m/2] for |v| < 2^21: there the float quotient is off by under 1/(4m), and an odd m
// keeps v/m at least 1/(2m) from a tie. m = 256 is exact and its tie gives +128, which is -128 as int8: the same residue.
struct Mod { int m; float inv; };
inline Mod mod_of(int m) { return {m, 1.0f / (float)m}; }
__device__ __forceinline__ int fold(int v, Mod q) { return v - q.m * __float2int_rn((float)v * q.inv); }
// any int32 (a leaf's dot product): the top 20 bits first, then the low 12 on top of that residue (both steps below 2^20)
__device__ __forceinline__ int8_t red(int x, Mod q) { return (int8_t)fold(fold(x >> 12, q) * 4096 + (x & 4095), q); }

struct Term { int n, idx[16], sgn[16]; };   // a linear combination of sub-blocks (A, Bt) or products (C)
struct Tab { Term a[49], b[49], c[16]; };
// Strassen's M1..M7 (quadrant index 2 r + c: 11, 12, 21, 22), two levels deep. Sub-block (R, C) of the 4 x 4 grid is
// quadrant (R/2, C/2) of the outer level and (R%2, C%2) of the inner one.
__host__ __device__ constexpr Tab two_tab() {
    const int U[7][4] = {{1, 0, 0, 1}, {0, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {-1, 0, 1, 0}, {0, 1, 0, -1}};
    const int V[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 0}, {0, 1, 0, -1}, {-1, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {0, 0, 1, 1}};
    const int W[4][7] = {{1, 0, 0, 1, -1, 0, 1}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0, 0}, {1, -1, 1, 0, 0, 1, 0}};
    Tab t{};
    for (int k1 = 0; k1 < 7; k1++) for (int k2 = 0; k2 < 7; k2++) {
        const int K = 7 * k1 + k2;
        for (int R = 0; R < 4; R++) for (int C = 0; C < 4; C++) {
            const int q1 = 2 * (R / 2) + C / 2, q2 = 2 * (R % 2) + C % 2, X = 4 * R + C;
            if (int s = U[k1][q1] * U[k2][q2]) { t.a[K].idx[t.a[K].n] = X; t.a[K].sgn[t.a[K].n++] = s; }
            if (int s = V[k1][q1] * V[k2][q2]) { t.b[K].idx[t.b[K].n] = 4 * C + R; t.b[K].sgn[t.b[K].n++] = s; }   // B (R, C) = Bt (C, R)
            if (int s = W[q1][k1] * W[q2][k2]) { t.c[X].idx[t.c[X].n] = K; t.c[X].sgn[t.c[X].n++] = s; }
        }
    }
    return t;
}

__device__ __forceinline__ int lane(int w, int b) { return (int8_t)(w >> (8 * b)); }
// Every table entry is a template argument, so the table is evaluated by the compiler only: no runtime index into the 16
// sub-blocks or 49 products, which then stay in registers (runtime indices put them in local memory: 2-3x slower)
template <int W, int K, int E> __device__ __forceinline__ int tterm(const int *x, int b) {   // W: 0 A, 1 Bt, 2 C
    constexpr Term T = W == 0 ? two_tab().a[K] : W == 1 ? two_tab().b[K] : two_tab().c[K];
    if constexpr (E < T.n) return T.sgn[E] * lane(x[T.idx[E]], b); else return 0;
}
template <int W, int K, int... E> __device__ __forceinline__ int tsum(const int *x, int b, std::integer_sequence<int, E...>) {
    return (0 + ... + tterm<W, K, E>(x, b));
}
template <int W, int K> __device__ __forceinline__ int tpack(const int *x, Mod q) {   // 4 lanes of combination K, mod m, packed
    int o = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) o |= (fold(tsum<W, K>(x, b, std::make_integer_sequence<int, 16>{}), q) & 0xff) << (8 * b);
    return o;
}
template <int W, int... K> __device__ __forceinline__ void tops(const int (*x)[16], int K0, int cnt, int8_t *O, size_t ss, size_t pos, Mod q, std::integer_sequence<int, K...>) {
    ((K >= K0 && K < K0 + cnt ? (void)(*(int4 *)&O[(size_t)(K - K0) * ss + pos] = make_int4(tpack<W, K>(x[0], q), tpack<W, K>(x[1], q), tpack<W, K>(x[2], q), tpack<W, K>(x[3], q))) : (void)0), ...);
}
template <int... T> __device__ __forceinline__ void touts(const int (*m)[49], int8_t *C, int s, int ldc, size_t r, size_t c, Mod q, std::integer_sequence<int, T...>) {
    ((*(int2 *)&C[((T / 4) * (size_t)s + r) * ldc + (T % 4) * s + c] = make_int2(tpack<2, T>(m[0], q), tpack<2, T>(m[1], q))), ...);
}

// O[K - K0] (s x s, packed) = the sub-blocks of X that combination K names, summed mod m; 16 bytes per thread.
// use: the sub-blocks the chunk K0 .. K0 + cnt - 1 names (bit t), the only ones read: a chunk of the 49 needs 4-16 of the 16
template <bool isB>
__global__ void two_ops(const int8_t *X, int ld, int s, int8_t *O, int K0, int cnt, unsigned use, Mod q) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = s / 16;
    if (i >= (size_t)s * w) return;
    const size_t r = i / w, c = (i % w) * 16;
    int x[4][16];
#pragma unroll
    for (int t = 0; t < 16; t++) {
        const int4 v = use >> t & 1 ? *(const int4 *)&X[((t / 4) * (size_t)s + r) * ld + (t % 4) * s + c] : make_int4(0, 0, 0, 0);
        x[0][t] = v.x; x[1][t] = v.y; x[2][t] = v.z; x[3][t] = v.w;
    }
    tops<isB ? 1 : 0>(x, K0, cnt, O, (size_t)s * s, r * s + c, q, std::make_integer_sequence<int, 49>{});
}

// C's 16 sub-blocks = their combinations of the 49 products, mod m (written, never read); 8 bytes per thread
__global__ void two_out(const int8_t *M, int s, int8_t *C, int ldc, Mod q) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = s / 8;
    if (i >= (size_t)s * w) return;
    const size_t r = i / w, c = (i % w) * 8;
    int m[2][49];
#pragma unroll
    for (int K = 0; K < 49; K++) { const int2 v = *(const int2 *)&M[(size_t)K * s * s + r * s + c]; m[0][K] = v.x; m[1][K] = v.y; }
    touts(m, C, s, ldc, r, c, q, std::make_integer_sequence<int, 16>{});
}

// leaf: C (h x h residues, row stride ldc) = A Bt^T mod m; g = block band (rasterization, gemm_i8_tile)
template <int WN, int ST>
__global__ void __launch_bounds__(THREADS, 1) leaf(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int g, Mod q) {
    gemm_i8_tile<WN, ST>(A, Bt, h, [&](int r, int c, int c00, int c01, int c10, int c11) {
        *(char2 *)&C[(size_t)r * ldc + c] = make_char2(red(c00, q), red(c01, q));
        *(char2 *)&C[(size_t)(r + 8) * ldc + c] = make_char2(red(c10, q), red(c11, q));
    }, lda, ldb, g);
}
// z products in one launch: blockIdx.z picks the product, operands and results packed s x s
template <int WN, int ST>
__global__ void __launch_bounds__(THREADS, 1) leafz(const int8_t *A, const int8_t *Bt, int8_t *C, int s, int g, Mod q) {
    const size_t o = (size_t)blockIdx.z * s * s;
    gemm_i8_tile<WN, ST>(A + o, Bt + o, s, [&](int r, int c, int c00, int c01, int c10, int c11) {
        *(char2 *)&C[o + (size_t)r * s + c] = make_char2(red(c00, q), red(c01, q));
        *(char2 *)&C[o + (size_t)(r + 8) * s + c] = make_char2(red(c10, q), red(c11, q));
    }, s, s, g);
}

// the sub-blocks of A (isB = 0) or Bt (1) that combinations K0 .. K0 + cnt - 1 read
inline unsigned two_use(bool isB, int K0, int cnt) {
    static constexpr Tab tab = two_tab(); unsigned m = 0;
    for (int K = K0; K < K0 + cnt; K++) { const Term &T = isB ? tab.b[K] : tab.a[K]; for (int e = 0; e < T.n; e++) m |= 1u << T.idx[e]; }
    return m;
}

// C = A B mod m for h x h with L (even) Strassen levels, two per pass. Buffers per depth t (nodes of h / 4^t): the 49
// products in full, the operands in chunks of cap (depth 0 may be chunked to fit the card; deeper depths hold all 49).
struct StrassenMod {
    int tile = 1, G = 8;   // leaf tile 0, 1, 2 = 32x3 (128 x 128), 64x3, 64x4 (128 x 256); G = block band
    struct Buf { int8_t *SA = nullptr, *SB = nullptr, *M = nullptr; int cap = 0; };
    std::vector<Buf> tb;
    std::vector<int8_t *> owned;

    static void grant() {
        auto g = [](auto f, int bytes) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes); };
        g(leaf<32, 3>, gemm_smem<32, 3>()); g(leaf<64, 3>, gemm_smem<64, 3>()); g(leaf<64, 4>, gemm_smem<64, 4>());
        g(leafz<32, 3>, gemm_smem<32, 3>()); g(leafz<64, 3>, gemm_smem<64, 3>()); g(leafz<64, 4>, gemm_smem<64, 4>());
    }
    // bytes of the buffers for n x n with L levels and cap operands at depth 0
    static size_t bytes(int n, int L, int cap) {
        size_t b = 0;
        for (int t = 0; t < L / 2; t++) { const size_t ss = (size_t)(n >> (2 * t + 2)) * (n >> (2 * t + 2)); b += (2 * (t ? 49 : cap) + 49) * ss; }
        return b;
    }
    // the buffers, carved from arena when given (it must hold bytes(n, L, cap)), else allocated here
    void plan(int n, int L, int cap, int8_t *arena = nullptr) {
        release(); tb.assign(L / 2, Buf{});
        for (int t = 0; t < L / 2; t++) {
            const size_t ss = (size_t)(n >> (2 * t + 2)) * (n >> (2 * t + 2)); Buf &b = tb[t]; b.cap = t ? 49 : cap;
            int8_t **p[3] = {&b.SA, &b.SB, &b.M}; const size_t sz[3] = {b.cap * ss, b.cap * ss, 49 * ss};
            for (int k = 0; k < 3; k++)
                if (arena) { *p[k] = arena; arena += sz[k]; }
                else { cudaMalloc(p[k], sz[k]); owned.push_back(*p[k]); }
        }
    }
    void release() { for (auto *p : owned) cudaFree(p); owned.clear(); tb.clear(); }

    template <int WN, int ST> void leaf_go(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int z, Mod q, cudaStream_t st) {
        if (z) leafz<WN, ST><<<dim3(h / (4 * WN), h / BM, z), THREADS, gemm_smem<WN, ST>(), st>>>(A, Bt, C, h, G, q);
        else leaf<WN, ST><<<dim3(h / (4 * WN), h / BM), THREADS, gemm_smem<WN, ST>(), st>>>(A, lda, Bt, ldb, C, ldc, h, G, q);
    }
    // C = A B mod m (h x h); z > 0: z products packed h x h (A, Bt, C contiguous, the ld arguments unused)
    void leaf_run(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int z, Mod q, cudaStream_t st = 0) {
        if (tile == 1) leaf_go<64, 3>(A, lda, Bt, ldb, C, ldc, h, z, q, st);
        else if (tile == 2) leaf_go<64, 4>(A, lda, Bt, ldb, C, ldc, h, z, q, st);
        else leaf_go<32, 3>(A, lda, Bt, ldb, C, ldc, h, z, q, st);
    }
    // prof (optional): ms in operand sums, products, output sums of the L = 2 nodes, accumulated
    void two_rec(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L, int t, Mod q, float *prof = nullptr) {
        const int s = h / 4; const size_t ss = (size_t)s * s;
        const unsigned g = (unsigned)((ss / 16 + 255) / 256), go = (unsigned)((ss / 8 + 255) / 256);
        const Buf &b = tb[t];
        cudaEvent_t e[4]; const bool pr = prof && L == 2;
        if (pr) for (auto &x : e) cudaEventCreate(&x);
        for (int K0 = 0; K0 < 49; K0 += b.cap) {
            const int cnt = 49 - K0 < b.cap ? 49 - K0 : b.cap;
            if (pr) cudaEventRecord(e[0]);
            two_ops<false><<<g, 256>>>(A, lda, s, b.SA, K0, cnt, two_use(false, K0, cnt), q);
            two_ops<true><<<g, 256>>>(Bt, ldb, s, b.SB, K0, cnt, two_use(true, K0, cnt), q);
            if (pr) cudaEventRecord(e[1]);
            if (L == 2) leaf_run(b.SA, 0, b.SB, 0, b.M + K0 * ss, 0, s, cnt, q);
            else for (int z = 0; z < cnt; z++) two_rec(b.SA + z * ss, s, b.SB + z * ss, s, b.M + (K0 + z) * ss, s, s, L - 2, t + 1, q, prof);
            if (pr) { float a, c; cudaEventRecord(e[2]); cudaEventSynchronize(e[2]); cudaEventElapsedTime(&a, e[0], e[1]); cudaEventElapsedTime(&c, e[1], e[2]); prof[0] += a; prof[1] += c; }
        }
        if (pr) cudaEventRecord(e[2]);
        two_out<<<go, 256>>>(b.M, s, C, ldc, q);
        if (pr) { float c; cudaEventRecord(e[3]); cudaEventSynchronize(e[3]); cudaEventElapsedTime(&c, e[2], e[3]); prof[2] += c; for (auto &x : e) cudaEventDestroy(x); }
    }
    // C = A B mod m, h x h, L in {0, 2, 4}: L = 0 is the classical product (the reference every L must equal bit for bit)
    void mul(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L, Mod q, float *prof = nullptr) {
        if (!L) leaf_run(A, lda, Bt, ldb, C, ldc, h, 0, q);
        else two_rec(A, lda, Bt, ldb, C, ldc, h, L, 0, q, prof);
    }
};

}  // namespace oz
