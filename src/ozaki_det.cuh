// ozaki_det.cuh -- exact integer determinant: LU modulo thousands of primes < 65280 at once, every O(n^3) step an INT8
// tensor-core GEMM. Nothing big ever enters or leaves the GPU: the input is int8, the output one residue per prime, and the
// only big-number step is one CRT of a scalar on the host -- the n^2 P^2 conversions that capped the big-integer GEMM at
// ~131 000 bits do not exist here, so the result can be as long as the primes allow.
//
// No pivoting: a prime whose leading minors vanish (probability ~n/p) is marked bad and dropped from the CRT; spares cover it.
// Recursive LU (Toledo): factor the left half, U of the right half by recursive forward substitution, one GEMM for the
// trailing part, factor the right half. Leaves are 64-column panels whose 64 x 64 diagonal blocks are inverted in shared
// memory. Each level moves ~n^2/4 residues through memory, against n^2 B/128 for a blocked LU with 64-wide updates.
//
// A product X W mod p of residues: X = Xlo + 256 Xhi (balanced bytes), V = 256 W mod p, then
// X W = [Xlo Xhi] [Wlo; Vlo] + 256 [Xlo Xhi] [Whi; Vhi] (mod p): two INT8 GEMMs of depth 2K, reduced once in the epilogue.
#pragma once
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "gemm_i8.cuh"

namespace oz {

constexpr int DB = 64;   // leaf panel / diagonal block

__device__ __forceinline__ int bal(int x, int p) { return x > (p >> 1) ? x - p : x; }
__device__ __forceinline__ uint32_t mulm(uint32_t a, uint32_t b, uint32_t p) { return a * b % p; }   // a, b < p < 2^16
__device__ uint32_t powm(uint32_t a, uint32_t e, uint32_t p) {
    uint32_t r = 1;
    for (; e; e >>= 1, a = mulm(a, a, p)) if (e & 1) r = mulm(r, a, p);
    return r;
}

// R[z] = [A I] mod p_z (n x ld, the identity only when ld = 2n), residues in [0, p)
__global__ void det_init(const int8_t *__restrict__ A, int32_t *__restrict__ R, int n, int ld, const uint32_t *__restrict__ P) {
    const int p = P[blockIdx.y];
    R += (size_t)blockIdx.y * n * ld;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)n * ld; i += (size_t)gridDim.x * blockDim.x) {
        int r = i / ld, c = i % ld, a = c < n ? A[(size_t)r * n + c] % p : c - n == r;
        R[i] = a < 0 ? a + p : a;
    }
}

__device__ __forceinline__ void split(int b, int8_t &lo, int8_t &hi) { lo = (int8_t)b; hi = (int8_t)((b - lo) >> 8); }

// out[r][k], out[r][K + k] = lo, hi byte of bal(X[r][k]); rows >= M are zero (padding to the 128-row tile). 4 k per thread.
__global__ void pack_a(const int32_t *__restrict__ X, size_t zx, int ld, int M, int K, int Mpad, int8_t *__restrict__ out,
                       const uint32_t *__restrict__ P) {
    const int z = blockIdx.y, p = P[z], K4 = K / 4;
    X += z * zx; out += (size_t)z * Mpad * 2 * K;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Mpad * K4; i += gridDim.x * blockDim.x) {
        int r = i / K4, k = (i % K4) * 4;
        int4 x = r < M ? *(const int4 *)&X[(size_t)r * ld + k] : make_int4(0, 0, 0, 0);
        char4 lo, hi;
        split(bal(x.x, p), (int8_t &)lo.x, (int8_t &)hi.x); split(bal(x.y, p), (int8_t &)lo.y, (int8_t &)hi.y);
        split(bal(x.z, p), (int8_t &)lo.z, (int8_t &)hi.z); split(bal(x.w, p), (int8_t &)lo.w, (int8_t &)hi.w);
        *(char4 *)&out[(size_t)r * 2 * K + k] = lo; *(char4 *)&out[(size_t)r * 2 * K + K + k] = hi;
    }
}

// W is K x N (row-major, ld); Bt1[c] = [lo(W[:,c]) lo(V[:,c])], Bt2[c] = [hi(W[:,c]) hi(V[:,c])], V = 256 W mod p.
// One 64 x 64 tile per block, transposed through shared memory: reads along c, writes 16-byte runs along k.
__global__ void __launch_bounds__(256) pack_b(const int32_t *__restrict__ W, size_t zw, int ld, int K, int N, int Npad,
                                              int8_t *__restrict__ B1, int8_t *__restrict__ B2, const uint32_t *__restrict__ P) {
    __shared__ __align__(16) int8_t s[4][64][80];   // wl, wh, vl, vh as [c][k]; rows of 80 bytes: 4-way bank conflicts, not 16
    const int z = blockIdx.z, p = P[z], c0 = blockIdx.x * 64, k0 = blockIdx.y * 64, tid = threadIdx.x;
    W += z * zw; B1 += (size_t)z * Npad * 2 * K; B2 += (size_t)z * Npad * 2 * K;
    for (int e = tid; e < 64 * 64; e += 256) {
        int k = e / 64, c = e % 64, w = c0 + c < N ? W[(size_t)(k0 + k) * ld + c0 + c] : 0;
        int b = bal(w, p), v = bal((int)(((uint32_t)w << 8) % p), p);
        split(b, s[0][c][k], s[1][c][k]); split(v, s[2][c][k], s[3][c][k]);
    }
    __syncthreads();
    for (int e = tid; e < 4 * 64 * 4; e += 256) {   // (plane, c, 16-byte chunk)
        int q = e & 3, c = (e >> 2) & 63, pl = e >> 8;
        int8_t *dst = (pl & 1 ? B2 : B1) + (size_t)(c0 + c) * 2 * K + (pl & 2 ? K : 0) + k0 + q * 16;
        *(int4 *)dst = *(const int4 *)&s[pl][c][q * 16];
    }
}

// SUB: R -= X W (mod p); else R = X W (mod p). R points at the M x N output corner of prime 0, zr apart per prime.
template <bool SUB>
__global__ void __launch_bounds__(THREADS) modgemm(const int8_t *__restrict__ Ap, const int8_t *__restrict__ B1,
                                                    const int8_t *__restrict__ B2, int32_t *R, size_t zr, int ld, int M, int N,
                                                    int K2, int Mpad, int Npad, const uint32_t *__restrict__ P) {
    const int z = blockIdx.z, p = P[z];
    Ap += (size_t)z * Mpad * K2; B1 += (size_t)z * Npad * K2; B2 += (size_t)z * Npad * K2; R += z * zr;
    // pass 0: R = R - G1 (SUB) or G1; |G1| <= K2 128^2 <= 2^26 (K <= 2048), so int32 holds it next to a residue.
    // pass 1: R = (R -+ 256 (G2 mod p)) mod p, in [0, p); every thread rereads exactly the elements it wrote in pass 0.
    // One call site in a loop: two calls would each get their own 48 KB of shared tiles.
    auto fin = [&](int32_t *r, int g, int pass) {
        if (pass == 0) { *r = SUB ? *r - g : g; return; }
        int v = SUB ? *r - 256 * (g % p) : *r + 256 * (g % p);
        v %= p; *r = v < 0 ? v + p : v;
    };
#pragma unroll 1
    for (int pass = 0; pass < 2; pass++) {
        if (pass) __syncthreads();   // the second tile loop refills shared buffers other warps may still read
        gemm_i8_tile(Ap, pass ? B2 : B1, K2, [&](int row, int col, int c00, int c01, int c10, int c11) {
            if (col >= N) return;
            if (row < M) { int32_t *r = &R[(size_t)row * ld + col]; fin(r, c00, pass); fin(r + 1, c01, pass); }
            if (row + 8 < M) { int32_t *r = &R[(size_t)(row + 8) * ld + col]; fin(r, c10, pass); fin(r + 1, c11, pass); }
        });
    }
}

// The 64 x 64 diagonal block at (j, j): LU without pivoting in shared memory, pivots into det[z], then L^-1 (unit lower)
// and U^-1 to Linv / Uinv (64 x 64 each, zl apart per prime). A zero pivot marks the prime bad.
__global__ void __launch_bounds__(256) diag_inv(const int32_t *__restrict__ R, size_t zr, int ld, int32_t *__restrict__ Linv, size_t zl,
                                                int32_t *__restrict__ Uinv, uint32_t *det, int *bad, const uint32_t *__restrict__ P) {
    __shared__ uint32_t S[DB][DB + 1], inv[DB];
    const int z = blockIdx.x, tid = threadIdx.x;
    const uint32_t p = P[z];
    R += z * zr; Linv += z * zl; Uinv += z * zl;
    for (int i = tid; i < DB * DB; i += 256) S[i / DB][i % DB] = R[(size_t)(i / DB) * ld + i % DB];
    __syncthreads();
    for (int t = 0; t < DB; t++) {
        if (tid == 0) {
            uint32_t pv = S[t][t];
            if (pv == 0) { bad[z] = 1; pv = 1; }
            det[z] = mulm(det[z], pv, p); inv[t] = powm(pv, p - 2, p);
        }
        __syncthreads();
        for (int i = t + 1 + tid; i < DB; i += 256) S[i][t] = mulm(S[i][t], inv[t], p);
        __syncthreads();
        for (int e = tid; e < (DB - 1 - t) * (DB - 1 - t); e += 256) {
            int i = t + 1 + e / (DB - 1 - t), c = t + 1 + e % (DB - 1 - t);
            S[i][c] = (S[i][c] + p - mulm(S[i][t], S[t][c], p)) % p;
        }
        __syncthreads();
    }
    if (tid < DB) {   // column c of L^-1: x_c = 1, x_i = -sum_{c<=s<i} L[i][s] x_s
        const int c = tid; uint32_t x[DB];
        for (int i = 0; i < DB; i++) {
            if (i < c) x[i] = 0;
            else if (i == c) x[i] = 1;
            else { uint64_t s = 0; for (int q = c; q < i; q++) s += (uint64_t)S[i][q] * x[q]; x[i] = (uint32_t)((p - s % p) % p); }
        }
        for (int i = 0; i < DB; i++) Linv[i * DB + c] = x[i];
    } else if (tid < 2 * DB) {   // column c of U^-1: y_c = 1/U_cc, y_i = -(sum_{i<s<=c} U[i][s] y_s) / U_ii, i = c-1 .. 0
        const int c = tid - DB; uint32_t y[DB];
        for (int i = DB - 1; i >= 0; i--) {
            if (i > c) y[i] = 0;
            else if (i == c) y[i] = inv[c];
            else { uint64_t s = 0; for (int q = i + 1; q <= c; q++) s += (uint64_t)S[i][q] * y[q]; y[i] = mulm((uint32_t)((p - s % p) % p), inv[i], p); }
        }
        for (int i = 0; i < DB; i++) Uinv[i * DB + c] = y[i];
    }
}

// Workspace for one batch of Pb primes: R is n x ld per prime (ld = n for the determinant, 2n for [A I] -> A^-1)
struct DetWork {
    int n, ld, Pb;
    int32_t *R, *Linv, *Uinv;   // R: Pb n ld; Linv, Uinv: Pb (n/64) 64^2, one per diagonal block
    const int32_t *preU = nullptr, *preL = nullptr;   // unit upper / lower n x n, entries in [0, 127]: det(U A L) = det A
    int8_t *Ap, *B1, *B2;
    uint32_t *P, *det; int *bad;
    void alloc(int n_, int Pb_, int ld_ = 0) {
        n = n_; ld = ld_ ? ld_ : n_; Pb = Pb_;
        size_t pk = (size_t)Pb * (ld + BM) * 2 * std::max(n, DB);   // M or N padded, times 2K with K <= n (the preconditioner)
        cudaMalloc(&R, (size_t)Pb * n * ld * 4); cudaMalloc(&Linv, (size_t)Pb * n * DB * 4); cudaMalloc(&Uinv, (size_t)Pb * n * DB * 4);
        cudaMalloc(&Ap, pk); cudaMalloc(&B1, pk); cudaMalloc(&B2, pk);
        cudaMalloc(&P, Pb * 4); cudaMalloc(&det, Pb * 4); cudaMalloc(&bad, Pb * 4);
    }
    void free_() { cudaFree(R); cudaFree(Linv); cudaFree(Uinv); cudaFree(Ap); cudaFree(B1); cudaFree(B2); cudaFree(P); cudaFree(det); cudaFree(bad); }
    int32_t *at(int r, int c) const { return R + (size_t)r * ld + c; }
    size_t zr() const { return (size_t)n * ld; }
};

inline int rup(int x, int m) { return (x + m - 1) / m * m; }

// OZ_PROF=1: synchronous timing of the three kinds of work (diagonal inverses, byte packing, GEMMs), seconds
struct DetProf {
    bool on = getenv("OZ_PROF") != nullptr; double t[3] = {}; std::chrono::steady_clock::time_point s;
    void begin() { if (on) { cudaDeviceSynchronize(); s = std::chrono::steady_clock::now(); } }
    void end(int k) { if (on) { cudaDeviceSynchronize(); t[k] += std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count(); } }
};
inline DetProf det_prof;

// The M x N corner of R at (r, c) (=|-=) X W, X: M x K at (X, ldx, zx), W: K x N at (W, ldw, zw); K % 64 == 0
template <bool SUB>
inline void mod_xw(DetWork &w, const int32_t *X, size_t zx, int ldx, const int32_t *W, size_t zw, int ldw, int M, int N, int K,
                   int r, int c) {
    if (M <= 0 || N <= 0) return;
    const int Mpad = rup(M, BM), Npad = rup(N, BN);
    det_prof.begin();
    pack_a<<<dim3(std::min(2048, Mpad * K / 1024 + 1), w.Pb), 256>>>(X, zx, ldx, M, K, Mpad, w.Ap, w.P);
    pack_b<<<dim3(Npad / 64, K / 64, w.Pb), 256>>>(W, zw, ldw, K, N, Npad, w.B1, w.B2, w.P);
    det_prof.end(1); det_prof.begin();
    modgemm<SUB><<<dim3(Npad / BN, Mpad / BM, w.Pb), THREADS>>>(w.Ap, w.B1, w.B2, w.at(r, c), w.zr(), w.ld, M, N, 2 * K, Mpad,
                                                                  Npad, w.P);
    det_prof.end(2);
}
// the same with both factors inside R
template <bool SUB>
inline void mod_rr(DetWork &w, int xr, int xc, int wr, int wc, int M, int N, int K, int r, int c) {
    mod_xw<SUB>(w, w.at(xr, xc), w.zr(), w.ld, w.at(wr, wc), w.zr(), w.ld, M, N, K, r, c);
}

inline int half(int wd) { return wd / (2 * DB) * DB; }   // the left part of a split, a multiple of 64

// rows [j, j + h) of columns [c, c + N) := L^-1 (them), L the unit lower triangle of the diagonal block at j
inline void det_trsm(DetWork &w, int j, int h, int c, int N) {
    if (h == DB) { mod_xw<false>(w, w.Linv + (size_t)j * DB, (size_t)w.n * DB, DB, w.at(j, c), w.zr(), w.ld, DB, N, DB, j, c); return; }
    const int h1 = half(h);
    det_trsm(w, j, h1, c, N);
    mod_rr<true>(w, j + h1, j, j, c, h - h1, N, h1, j + h1, c);
    det_trsm(w, j + h1, h - h1, c, N);
}

// rows [j, j + h) of columns [c, c + N) := U^-1 (them), U the upper triangle of the diagonal block at j
inline void det_trsm_up(DetWork &w, int j, int h, int c, int N) {
    if (h == DB) { mod_xw<false>(w, w.Uinv + (size_t)j * DB, (size_t)w.n * DB, DB, w.at(j, c), w.zr(), w.ld, DB, N, DB, j, c); return; }
    const int h1 = half(h);
    det_trsm_up(w, j + h1, h - h1, c, N);
    mod_rr<true>(w, j, j + h1, j + h1, c, h1, N, h - h1, j, c);
    det_trsm_up(w, j, h1, c, N);
}

// LU of columns [j, j + wd) over rows [j, n): L below the diagonal blocks, U above, pivots into det
inline void det_lu(DetWork &w, int j, int wd) {
    const int n = w.n;
    if (wd == DB) {
        int32_t *Ui = w.Uinv + (size_t)j * DB;
        det_prof.begin();
        diag_inv<<<w.Pb, 256>>>(w.at(j, j), w.zr(), w.ld, w.Linv + (size_t)j * DB, (size_t)n * DB, Ui, w.det, w.bad, w.P);
        det_prof.end(0);
        mod_xw<false>(w, w.at(j + DB, j), w.zr(), w.ld, Ui, (size_t)n * DB, DB, n - j - DB, DB, DB, j + DB, j);   // L21 = A21 U^-1
        return;
    }
    const int h = half(wd), c = j + h;
    det_lu(w, j, h);
    det_trsm(w, j, h, c, wd - h);
    mod_rr<true>(w, c, j, j, c, n - c, wd - h, h, c, c);
    det_lu(w, c, wd - h);
}

// det(A) mod each of the Pb primes (n % 64 == 0); results in w.det, bad primes flagged in w.bad.
// With ld = 2n, R[:, n:2n] ends as A^-1 mod p.
inline void det_mod_batch(DetWork &w, const int8_t *dA) {
    const int n = w.n;
    det_init<<<dim3(256, w.Pb), 256>>>(dA, w.R, n, w.ld, w.P);
    // A zero leading minor over Z (a structured matrix) fails every prime without pivoting; U A L has generic rank profile
    // with high probability (Kaltofen-Saunders). The entries are the same small integers for every prime, so zw = 0.
    if (w.preU) {
        mod_xw<false>(w, w.at(0, 0), w.zr(), w.ld, w.preL, 0, n, n, n, n, 0, 0);
        mod_xw<false>(w, w.preU, 0, n, w.at(0, 0), w.zr(), w.ld, n, n, n, 0, 0);
    }
    det_lu(w, 0, n);
    if (w.ld == 2 * n) {
        det_trsm(w, 0, n, n, n); det_trsm_up(w, 0, n, n, n);
        if (w.preU) {   // (U A L)^-1 = L^-1 A^-1 U^-1, so A^-1 = L (U A L)^-1 U
            mod_xw<false>(w, w.at(0, n), w.zr(), w.ld, w.preU, 0, n, n, n, n, 0, n);
            mod_xw<false>(w, w.preL, 0, n, w.at(0, n), w.zr(), w.ld, n, n, n, 0, n);
        }
    }
}

// Dixon p-adic lifting with A^-1 mod p: digit x_i = A^-1 r_i mod p, then r_{i+1} = (r_i - A x_i) / p exactly.
// |r| stays below n 127 + |b|, so every step is two matrix-vector products of small numbers (memory-bound: A^-1 as 16-bit
// residues and A as int8 are read once per step), and after m steps x = sum x_i p^i = A^-1 b mod p^m.
__global__ void to_u16(const int32_t *__restrict__ R, int ld, int c0, int n, uint16_t *__restrict__ out) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)n * n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = (uint16_t)R[(i / n) * ld + c0 + i % n];
}
// one warp per row: x[i] = sum_k Ai[i][k] rm[k] mod p (each product < 2^32, the sum of 2^16 of them < 2^48)
__global__ void lift_x(const uint16_t *__restrict__ Ai, const uint32_t *__restrict__ rm, uint16_t *__restrict__ x, int n, uint32_t p) {
    const int lane = threadIdx.x & 31, i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= n) return;
    const uint16_t *a = Ai + (size_t)i * n; uint64_t s = 0;
    for (int k = lane * 8; k < n; k += 256) {
        uint4 v = *(const uint4 *)&a[k]; uint4 r0 = *(const uint4 *)&rm[k], r1 = *(const uint4 *)&rm[k + 4];
        s += (uint64_t)(v.x & 0xffff) * r0.x + (uint64_t)(v.x >> 16) * r0.y + (uint64_t)(v.y & 0xffff) * r0.z + (uint64_t)(v.y >> 16) * r0.w
           + (uint64_t)(v.z & 0xffff) * r1.x + (uint64_t)(v.z >> 16) * r1.y + (uint64_t)(v.w & 0xffff) * r1.z + (uint64_t)(v.w >> 16) * r1.w;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) x[i] = (uint16_t)(s % p);
}
// one warp per row: r[i] = (r[i] - sum_k A[i][k] x[k]) / p, rm[i] = r[i] mod p; a remainder != 0 sets *err
__global__ void lift_r(const int8_t *__restrict__ A, const uint16_t *__restrict__ x, int32_t *__restrict__ r, uint32_t *__restrict__ rm,
                       int n, int p, int *err) {
    const int lane = threadIdx.x & 31, i = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (i >= n) return;
    const int8_t *a = A + (size_t)i * n; long long s = 0;
    for (int k = lane * 16; k < n; k += 512) {
        int4 v = *(const int4 *)&a[k]; const int8_t *b = (const int8_t *)&v;
        uint4 x0 = *(const uint4 *)&x[k], x1 = *(const uint4 *)&x[k + 8];
        const uint16_t *y0 = (const uint16_t *)&x0, *y1 = (const uint16_t *)&x1;
        int t = 0;   // 16 products below 2^7 2^16 each: < 2^27
#pragma unroll
        for (int q = 0; q < 8; q++) t += b[q] * (int)y0[q] + b[8 + q] * (int)y1[q];
        s += t;
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) {
        long long t = r[i] - s, q = t / p;
        if (q * p != t) *err = 1;
        r[i] = (int32_t)q; int m = (int)(q % p); rm[i] = m < 0 ? m + p : m;
    }
}

// Base-p digits to integers for all n entries at once, as ONE INT8 GEMM: x_j = sum_i d_ij p^i. Digit d = lo + 256 (h + 128),
// lo, h signed bytes, so x_j = [lo | h]_j [P; 256 P] + 32768 sum_i p^i, with P[i] = the balanced bytes of p^i (256 P is the
// same bytes one position up). Accumulators: K2 2^14 < 2^31 for K2 < 131072.
__global__ void digits_to_a(const uint16_t *__restrict__ X, int m, int mpad, int n, int8_t *__restrict__ out) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * mpad; e += (size_t)gridDim.x * blockDim.x) {
        int j = e / mpad, i = e % mpad, d = i < m ? X[(size_t)i * n + j] : 0;
        int8_t lo = (int8_t)d; int h = i < m ? (d - lo) / 256 - 128 : 0;
        out[(size_t)j * 2 * mpad + i] = lo; out[(size_t)j * 2 * mpad + mpad + i] = (int8_t)h;
    }
}
// one thread per row: C[j][0..N) (int32 coefficients of 256^l) -> bytes + a final signed carry
__global__ void carry_rows(const int32_t *__restrict__ C, int N, int rows, uint8_t *__restrict__ out, long long *__restrict__ top) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= rows) return;
    long long c = 0;
    for (int l = 0; l < N; l++) {
        c += C[(size_t)j * N + l];
        out[(size_t)j * N + l] = (uint8_t)(c & 255); c >>= 8;   // arithmetic shift: floor division, so c & 255 is the digit
    }
    top[j] = c;
}

}  // namespace oz
