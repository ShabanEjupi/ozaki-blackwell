// ozaki_det.cuh -- exact integer determinant: LU modulo thousands of primes < 65280 at once, every O(n^3) step an INT8
// tensor-core GEMM. Nothing big ever enters or leaves the GPU: the input is int8, the output one residue per prime, and the
// only big-number step is one CRT of a scalar on the host -- the n^2 P^2 conversions that capped the big-integer GEMM at
// ~131 000 bits do not exist here, so the result can be as long as the primes allow.
//
// No pivoting: a prime whose leading minors vanish (probability ~n/p) is marked bad and dropped from the CRT; spares cover it.
// Blocked right-looking LU, inner blocks of 64 (their 64x64 diagonal blocks inverted in shared memory), outer panels of
// B = 512 so the big trailing update runs with K = 512 (the K = 64 update alone would be bound by memory, ~21 ops/byte).
//
// A product X W mod p of residues: X = Xlo + 256 Xhi (balanced bytes), V = 256 W mod p, then
// X W = [Xlo Xhi] [Wlo; Vlo] + 256 [Xlo Xhi] [Whi; Vhi] (mod p): two INT8 GEMMs of depth 2K, reduced once in the epilogue.
#pragma once
#include <cstdint>
#include <algorithm>
#include "gemm_i8.cuh"

namespace oz {

constexpr int DB = 64;    // inner block
constexpr int DBO = 512;  // outer panel

__device__ __forceinline__ int bal(int x, int p) { return x > (p >> 1) ? x - p : x; }
__device__ __forceinline__ uint32_t mulm(uint32_t a, uint32_t b, uint32_t p) { return (uint32_t)(((uint64_t)a * b) % p); }
__device__ uint32_t powm(uint32_t a, uint32_t e, uint32_t p) {
    uint32_t r = 1;
    for (; e; e >>= 1, a = mulm(a, a, p)) if (e & 1) r = mulm(r, a, p);
    return r;
}

// R[z] = A mod p_z, residues in [0, p)
__global__ void det_init(const int8_t *__restrict__ A, int32_t *__restrict__ R, size_t nn, const uint32_t *__restrict__ P) {
    const int p = P[blockIdx.y];
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nn; i += (size_t)gridDim.x * blockDim.x) {
        int a = A[i] % p; R[blockIdx.y * nn + i] = a < 0 ? a + p : a;
    }
}

// out[r][k], out[r][K + k] = lo, hi byte of bal(X[r][k]); rows >= M are zero (padding to the 128-row tile)
__global__ void pack_a(const int32_t *__restrict__ X, size_t zx, int ld, int M, int K, int Mpad, int8_t *__restrict__ out,
                       const uint32_t *__restrict__ P) {
    const int z = blockIdx.y, p = P[z];
    X += z * zx; out += (size_t)z * Mpad * 2 * K;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Mpad * K; i += gridDim.x * blockDim.x) {
        int r = i / K, k = i % K, b = r < M ? bal(X[(size_t)r * ld + k], p) : 0;
        int8_t lo = (int8_t)b;
        out[(size_t)r * 2 * K + k] = lo; out[(size_t)r * 2 * K + K + k] = (int8_t)((b - lo) >> 8);
    }
}

// W is K x N (row-major, ld); Bt1[c] = [lo(W[:,c]) lo(V[:,c])], Bt2[c] = [hi(W[:,c]) hi(V[:,c])], V = 256 W mod p
__global__ void pack_b(const int32_t *__restrict__ W, size_t zw, int ld, int K, int N, int Npad, int8_t *__restrict__ B1,
                       int8_t *__restrict__ B2, const uint32_t *__restrict__ P) {
    const int z = blockIdx.y, p = P[z];
    W += z * zw; B1 += (size_t)z * Npad * 2 * K; B2 += (size_t)z * Npad * 2 * K;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Npad * K; i += gridDim.x * blockDim.x) {
        int k = i / Npad, c = i % Npad;   // c fastest: the reads of W are coalesced
        int w = c < N ? W[(size_t)k * ld + c] : 0, b = bal(w, p), v = bal((int)(((uint32_t)w << 8) % p), p);
        int8_t wl = (int8_t)b, vl = (int8_t)v;
        size_t o = (size_t)c * 2 * K + k;
        B1[o] = wl; B1[o + K] = vl; B2[o] = (int8_t)((b - wl) >> 8); B2[o + K] = (int8_t)((v - vl) >> 8);
    }
}

// SUB: R -= X W (mod p); else R = X W (mod p). R points at the M x N output corner of prime 0, zr apart per prime.
template <bool SUB>
__global__ void __launch_bounds__(THREADS) modgemm(const int8_t *__restrict__ Ap, const int8_t *__restrict__ B1,
                                                    const int8_t *__restrict__ B2, int32_t *R, size_t zr, int ld, int M, int N,
                                                    int K2, int Mpad, int Npad, const uint32_t *__restrict__ P) {
    const int z = blockIdx.z, p = P[z];
    Ap += (size_t)z * Mpad * K2; B1 += (size_t)z * Npad * K2; B2 += (size_t)z * Npad * K2; R += z * zr;
    // pass 0: R = R - G1 (SUB) or G1; |G1| <= K2 128^2 <= 2^24, so int32 holds it next to a residue.
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
// and U^-1 to Linv / Uinv (64 x 64 each, per prime). A zero pivot marks the prime bad.
__global__ void __launch_bounds__(256) diag_inv(const int32_t *__restrict__ R, size_t zr, int ld, int32_t *__restrict__ Linv, size_t zl,
                                                int32_t *__restrict__ Uinv, uint32_t *det, int *bad, const uint32_t *__restrict__ P) {
    __shared__ uint32_t S[DB][DB + 1], inv[DB];
    const int z = blockIdx.x, tid = threadIdx.x;
    const uint32_t p = P[z];
    R += z * zr; Linv += z * zl; Uinv += (size_t)z * DB * DB;
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

// Workspace for one batch of Pb primes, n x n
struct DetWork {
    int n, Pb;
    int32_t *R, *Linv, *Uinv;   // R: Pb n^2; Linv: Pb (DBO/DB) 64^2 (one per inner block of the outer panel); Uinv: Pb 64^2
    int8_t *Ap, *B1, *B2;
    uint32_t *P, *det; int *bad;
    void alloc(int n_, int Pb_) {
        n = n_; Pb = Pb_;
        size_t pk = (size_t)Pb * (n + BM) * 2 * DBO;
        cudaMalloc(&R, (size_t)Pb * n * n * 4); cudaMalloc(&Linv, (size_t)Pb * (DBO / DB) * DB * DB * 4);
        cudaMalloc(&Uinv, (size_t)Pb * DB * DB * 4);
        cudaMalloc(&Ap, pk); cudaMalloc(&B1, pk); cudaMalloc(&B2, pk);
        cudaMalloc(&P, Pb * 4); cudaMalloc(&det, Pb * 4); cudaMalloc(&bad, Pb * 4);
    }
    void free_() { cudaFree(R); cudaFree(Linv); cudaFree(Uinv); cudaFree(Ap); cudaFree(B1); cudaFree(B2); cudaFree(P); cudaFree(det); cudaFree(bad); }
};

inline int rup(int x, int m) { return (x + m - 1) / m * m; }

// out(M x N corner of R at row r, col c) (=|-=) X W, X: M x K at (X, ldx, zx), W: K x N at (W, ldw, zw)
template <bool SUB>
inline void mod_xw(DetWork &w, const int32_t *X, size_t zx, int ldx, const int32_t *W, size_t zw, int ldw, int M, int N, int K,
                   int r, int c) {
    if (M <= 0 || N <= 0) return;
    const int Mpad = rup(M, BM), Npad = rup(N, BN);
    pack_a<<<dim3(std::min(1024, Mpad * K / 256 + 1), w.Pb), 256>>>(X, zx, ldx, M, K, Mpad, w.Ap, w.P);
    pack_b<<<dim3(std::min(1024, Npad * K / 256 + 1), w.Pb), 256>>>(W, zw, ldw, K, N, Npad, w.B1, w.B2, w.P);
    modgemm<SUB><<<dim3(Npad / BN, Mpad / BM, w.Pb), THREADS>>>(w.Ap, w.B1, w.B2, w.R + (size_t)r * w.n + c, (size_t)w.n * w.n,
                                                                  w.n, M, N, 2 * K, Mpad, Npad, w.P);
}

// det(A) mod each of the Pb primes (n % 64 == 0); results in w.det, bad primes flagged in w.bad
inline void det_mod_batch(DetWork &w, const int8_t *dA) {
    const int n = w.n; const size_t nn = (size_t)n * n, z64 = (size_t)DB * DB, zl = (DBO / DB) * z64;
    int32_t *R = w.R;
    det_init<<<dim3(256, w.Pb), 256>>>(dA, R, nn, w.P);
    for (int j0 = 0; j0 < n; j0 += DBO) {
        const int ce = std::min(n, j0 + DBO);   // the outer panel is columns [j0, ce)
        for (int j = j0; j < ce; j += DB) {
            const int k = (j - j0) / DB, e = j + DB;
            int32_t *Lk = w.Linv + k * z64;
            diag_inv<<<w.Pb, 256>>>(R + (size_t)j * n + j, nn, n, Lk, zl, w.Uinv, w.det, w.bad, w.P);
            mod_xw<false>(w, R + (size_t)e * n + j, nn, n, w.Uinv, z64, DB, n - e, DB, DB, e, j);        // L21 = A21 U^-1
            mod_xw<false>(w, Lk, zl, DB, R + (size_t)j * n + e, nn, n, DB, ce - e, DB, j, e);            // U12 = L^-1 A12 (panel)
            mod_xw<true>(w, R + (size_t)e * n + j, nn, n, R + (size_t)j * n + e, nn, n, n - e, ce - e, DB, e, e);
        }
        if (ce == n) break;
        for (int j = j0; j < ce; j += DB) {   // U of the panel rows right of the panel: block forward substitution
            const int k = (j - j0) / DB, e = j + DB;
            mod_xw<false>(w, w.Linv + k * z64, zl, DB, R + (size_t)j * n + ce, nn, n, DB, n - ce, DB, j, ce);
            mod_xw<true>(w, R + (size_t)e * n + j, nn, n, R + (size_t)j * n + ce, nn, n, ce - e, n - ce, DB, e, ce);
        }
        mod_xw<true>(w, R + (size_t)ce * n + j0, nn, n, R + (size_t)j0 * n + ce, nn, n, n - ce, n - ce, ce - j0, ce, ce);
    }
}

}  // namespace oz
