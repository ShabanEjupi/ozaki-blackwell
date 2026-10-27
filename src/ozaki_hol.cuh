// ozaki_hol.cuh -- high-order lifting (Storjohann 2005, section 8, "sparse inverse expansion"; row form) on INT8 tensor
// cores: y = b A^-1 mod X^N for an n x n matrix of L-byte entries, every O(n^3) step a modular or exact INT8 GEMM.
// Dixon (ozaki_det.cuh lift_*) gains one X-digit per step and reads A and A^-1 each time: memory-bound, ~L^2 for L-byte
// entries. Here the residue R_i = (I - A C_i) / X^(m_i) is squared, so the precision doubles per level: ~log2 N levels
// of n^3 products instead of N matrix-vector steps.
//
// X = p_0 ... p_{s-1} (distinct 16-bit primes, B0 = A^-1 mod X from the LU of [A I] modulo each), X >= 6.25 n^2 |A|.
//   R_0 = (I - A B0) / X;  level i: Rb = R_i^2,  M_i = B0 Rb mod X in [0, X),  R_{i+1} = (Rb - A M_i) / X  (exact)
//   |R_i| <= 1.25 n |A|.  With C_0 = B0, C_{i+1} = C_i (I + X^m R_i) + X^(2m) M_i, m_{i+1} = 2 m_i + 1, m_0 = 1:
//   A^-1 = C_i + X^(m_i) A^-1 R_i, so c_i = b C_i = b A^-1 mod X^(m_i).
// The residue chain runs in RNS: R_i modulo the s primes of X and w more primes q_r (prod q > 2 |R|); M_i comes from its
// s residues by Garner; R_{i+1} mod q_r = (Rb - A M_i) X^-1 mod q_r, its exact value by Garner over the q_r. 2 (s + w)
// modular GEMMs (ozaki_det.cuh modgemm) per level, no big numbers anywhere.
// Every matrix is kept TRANSPOSED (Z_i = R_i^T, B0^T, A^T, M_i^T), so that the vector-matrix products below read
// K-contiguous rows: Z_{i+1} = (Z_i^2 - M_i^T A^T) / X, M_i^T = Z_i^2 B0^T.
// The vector chain: c_i is stored as balanced digits in a PERIODIC mixed radix (position e has radix p_{e mod s}), so
// X^m is a shift by s m positions and the digits of M_i from Garner are digits of this radix already. c_i R_i is one
// INT8 GEMM: rows = the digits of c_i (lo | hi bytes, K = 2n), columns = the byte planes of Z_i ([Z_v | Z_{v-1}]),
// so column v collects 256^v; the planes are combined per (position, entry) in parallel, and the carries run down the
// positions, one thread per entry.
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include "ozaki_det.cuh"

namespace oz {

constexpr int HOL_SMAX = 24, HOL_WMAX = 20, HOL_BMAX = 2 * HOL_WMAX;   // primes of X, primes q, bytes below prod q

struct HolConst {
    int s, w;
    uint32_t p[HOL_SMAX], q[HOL_WMAX];
    uint32_t pinv[HOL_SMAX][HOL_SMAX], qinv[HOL_WMAX][HOL_WMAX];   // [u][t] = p_u^-1 mod p_t (q likewise)
    uint32_t pq[HOL_SMAX][HOL_WMAX], qp[HOL_WMAX][HOL_SMAX];       // p_t mod q_r, q_r mod p_t
    uint32_t xinv[HOL_WMAX], piq_p[HOL_SMAX];                      // X^-1 mod q_r, prod q mod p_t
    unsigned long long pm[HOL_SMAX];                               // floor((2^64 - 1) / p_t)
    uint32_t pmi[HOL_SMAX];                                        // floor(2^32 / p_t): Barrett
    uint8_t piq[HOL_BMAX];                                         // prod q, little-endian bytes
};
__constant__ HolConst hc;
// x mod m for any x < 2^32, mi = floor(2^32 / m): the quotient estimate is short by at most 1
__device__ __forceinline__ uint32_t redm(uint32_t x, uint32_t m, uint32_t mi) { const uint32_t r = x - __umulhi(x, mi) * m; return r >= m ? r - m : r; }

// The residue chain over 8-bit moduli (n % 128 == 0 and 2 zbits + log2 n + 10 bits within the 53 odd moduli < 256):
// a 16-bit modular product costs 4 INT8 products and two passes, an 8-bit one 1 product -- 2.8x less per bit of
// modulus at n = 4096 (bench/rchain_bench.cu). Zb = Z^2 is computed EXACTLY in an RNS of W odd moduli r_u < 256
// (prod > 2^(2 zbits + log2 n + 10)), then reduced to the p_t by an approximate-CRT base extension (alpha from a float
// sum, |Zb| <= 2^-9 prod); the first WQ moduli serve as the q of Z (prod > 2^(zbits + 1)), so A M is taken modulo those
// alone. Only M = B0 Zb mod X keeps 16-bit products: B0 = A^-1 mod p_t needs LU without pivoting, which small primes fail.
constexpr int H8_MAX = 56, H8_QMAX = 24;
struct Hol8Const {
    int W, WQ, nby;                     // moduli, how many are q, bytes of prod q
    uint32_t r[H8_MAX], rmi[H8_MAX], cinv[H8_MAX];   // moduli, floor(2^32 / r_u), (R / r_u)^-1 mod r_u (R = all W)
    float rinv[H8_MAX];
    uint16_t ginv[H8_QMAX][H8_QMAX];    // [u][v] = r_u^-1 mod r_v (Garner among the q)
    uint16_t qr[H8_QMAX][H8_MAX];       // [v][u] = r_v mod r_u
    uint16_t piq_r[H8_MAX], xinv[H8_QMAX];   // prod q mod r_u, X^-1 mod q_r
    uint16_t Cp[H8_MAX][HOL_SMAX];      // (R / r_u) mod p_t
    uint16_t Rp[HOL_SMAX];              // R mod p_t
    uint16_t pq8[HOL_SMAX][H8_QMAX];    // p_t mod q_r
    uint8_t piq[HOL_BMAX];              // prod q, bytes
    uint16_t qcinv[H8_QMAX];            // (Q / q_r)^-1 mod q_r
    uint8_t Qr[H8_QMAX][H8_MAX];        // (Q / q_r) mod r_u
    uint8_t Qby[H8_QMAX][H8_QMAX + 2];  // bytes of Q / q_r
};
__constant__ Hol8Const hc8;
__device__ __forceinline__ int8_t bal8(uint32_t v, uint32_t m) { return (int8_t)(v > m / 2 ? (int)v - (int)m : (int)v); }

// dst[z][c][r] = src[z][r][c]: n x n, src rows ld apart and zs per z, dst compact; 32 x 32 tiles, block (32, 8)
__global__ void transpose_i32(const int32_t *__restrict__ src, int ld, size_t zs, int32_t *__restrict__ dst, int n) {
    __shared__ int32_t t[32][33];
    src += blockIdx.z * zs; dst += blockIdx.z * (size_t)n * n;
    const int r0 = blockIdx.y * 32, c0 = blockIdx.x * 32;
    for (int y = threadIdx.y; y < 32; y += blockDim.y) t[y][threadIdx.x] = src[(size_t)(r0 + y) * ld + c0 + threadIdx.x];
    __syncthreads();
    for (int y = threadIdx.y; y < 32; y += blockDim.y) dst[(size_t)(c0 + y) * n + r0 + threadIdx.x] = t[threadIdx.x][y];
}

// One warp per row j of M^T (s residue matrices, n^2 apart): Garner -> the mixed-radix digits y_t of each entry of M in
// [0, X); Mq[r] = M^T mod q_r (unless null; Mq8: modulo the 8-bit q of hc8, balanced int8); bm[t][j] = sum_k b[k] y_t(j, k),
// i.e. digit t of (b M)[j], unnormalized.
template <int SB>   // SB >= s: the arrays stay in registers
__global__ void __launch_bounds__(256) hol_garner(const int32_t *__restrict__ Mt, const int32_t *__restrict__ b, int n,
                                                 int32_t *__restrict__ Mq, long long *__restrict__ bm, int8_t *__restrict__ Mq8 = nullptr) {
    const int lane = threadIdx.x & 31, j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= n) return;
    const int s = hc.s, w = hc.w; const size_t nn = (size_t)n * n;
    long long acc[SB];
#pragma unroll
    for (int t = 0; t < SB; t++) acc[t] = 0;
    for (int k = lane; k < n; k += 32) {
        const size_t i = (size_t)j * n + k; const int bk = b[k];
        uint32_t y[SB];
#pragma unroll
        for (int t = 0; t < SB; t++) if (t < s) {
            const uint32_t p = hc.p[t]; uint32_t v = (uint32_t)Mt[t * nn + i];
#pragma unroll
            for (int u = 0; u < t; u++) { v = v + 2 * p - y[u]; v = v >= p ? v - p : v; v = v >= p ? v - p : v; v = redm(v * hc.pinv[u][t], p, hc.pmi[t]); }   // y[u] < p_u < 2p
            y[t] = v; acc[t] += (long long)bk * (int)v;
        }
        if (Mq)
#pragma unroll
            for (int r = 0; r < HOL_WMAX; r++) if (r < w) {   // Horner from the top digit; m p + y < 65279^2 + 65279 < 2^32
                const uint32_t q = hc.q[r]; uint32_t m = 0;
#pragma unroll
                for (int t = SB - 1; t >= 0; t--) if (t < s) m = (m * hc.pq[t][r] + y[t]) % q;
                Mq[r * nn + i] = m;
            }
        if (Mq8)
#pragma unroll
            for (int r = 0; r < H8_QMAX; r++) if (r < hc8.WQ) {   // m q + y < 2^16 + 2^16
                const uint32_t q = hc8.r[r]; uint32_t m = 0;
#pragma unroll
                for (int t = SB - 1; t >= 0; t--) if (t < s) m = redm(m * hc8.pq8[t][r] + y[t], q, hc8.rmi[r]);
                Mq8[r * nn + i] = bal8(m, q);
            }
    }
#pragma unroll
    for (int t = 0; t < SB; t++) if (t < s) {
        long long a = acc[t];
        for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) bm[(size_t)t * n + j] = a;
    }
}

// Z_next = (Zb - AM) / X from residues mod q_r (ident: Zb = I). Out: Z_next modulo the q_r and the p_t (Zres: s + w
// matrices, the p_t first) and its U balanced byte planes into Bt, the B operand of the vector chain: rows (v, j), K = 2n,
// row (v, j) = [Z_v[j][:] | Z_{v-1}[j][:]] (plane v's second half and plane U's first half stay zero). |Z| < 2^zbits is
// checked: a wrong residue anywhere gives a value of the size of prod q, far above.
__global__ void __launch_bounds__(256) hol_next(const int32_t *__restrict__ Zbq, const int32_t *__restrict__ AMq, int n, int ident,
                                               int32_t *__restrict__ Zres, int8_t *__restrict__ Bt, int U, int zbits, int *err) {
    const size_t nn = (size_t)n * n, pl = (size_t)n * 2 * n;
    const int s = hc.s, w = hc.w;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nn; i += (size_t)gridDim.x * blockDim.x) {
        const int j = (int)(i / n), k = (int)(i % n);
        uint32_t y[HOL_WMAX];
#pragma unroll
        for (int r = 0; r < HOL_WMAX; r++) if (r < w) {
            const uint32_t q = hc.q[r], zb = ident ? (j == k) : (uint32_t)Zbq[r * nn + i];
            uint32_t v = (zb + q - (uint32_t)AMq[r * nn + i]) % q * hc.xinv[r] % q;
            Zres[(s + r) * nn + i] = v;
#pragma unroll
            for (int u = 0; u < r; u++) v = (v + q - y[u] % q) % q * hc.qinv[u][r] % q;
            y[r] = v;
        }
        // symmetric: negative iff the value exceeds (prod q - 1) / 2, whose mixed-radix digits are (q_r - 1) / 2
        bool neg = false, dec = false;
#pragma unroll
        for (int r = HOL_WMAX - 1; r >= 0; r--) if (r < w && !dec) {
            const uint32_t h = (hc.q[r] - 1) / 2;
            if (y[r] != h) { neg = y[r] > h; dec = true; }
        }
#pragma unroll
        for (int t = 0; t < HOL_SMAX; t++) if (t < s) {
            const uint32_t p = hc.p[t]; uint32_t m = 0;
#pragma unroll
            for (int r = HOL_WMAX - 1; r >= 0; r--) if (r < w) m = (m * hc.qp[r][t] + y[r]) % p;
            if (neg) m = (m + p - hc.piq_p[t]) % p;
            Zres[t * nn + i] = m;
        }
        uint32_t d[HOL_BMAX];   // the value in [0, prod q) as bytes, Horner from the top digit; then |Z| = prod q - it
#pragma unroll
        for (int u = 0; u < HOL_BMAX; u++) d[u] = 0;
#pragma unroll
        for (int r = HOL_WMAX - 1; r >= 0; r--) if (r < w) {
            uint32_t c = y[r]; const uint32_t q = hc.q[r];
#pragma unroll
            for (int u = 0; u < HOL_BMAX; u++) { const uint32_t t = d[u] * q + c; d[u] = t & 255; c = t >> 8; }
        }
        if (neg) {
            int bo = 0;
#pragma unroll
            for (int u = 0; u < HOL_BMAX; u++) { const int t = (int)hc.piq[u] - (int)d[u] - bo; bo = t < 0; d[u] = (uint32_t)(t + 256 * bo); }
        }
        bool big = false;
#pragma unroll
        for (int u = 0; u < HOL_BMAX; u++) { const int sh = zbits - 8 * u; big |= sh <= 0 ? d[u] != 0 : sh < 8 && (d[u] >> sh); }
        if (big) *err = 1;
        int c = 0; int8_t *o = Bt + (size_t)j * 2 * n + k;
#pragma unroll
        for (int u = 0; u < HOL_BMAX; u++) if (u < U) {   // balanced bytes in [-128, 127] of +-|Z|
            const int t = (neg ? -(int)d[u] : (int)d[u]) + c, z = ((t + 128) & 255) - 128;
            c = (t - z) >> 8;
            o[u * pl] = (int8_t)z; o[(u + 1) * pl + n] = (int8_t)z;
        }
    }
}

// rows [e0, e0 + rows) of the digit array c (int16, row = position, n entries) -> A operand [lo | hi], K = 2n; up to rpad zero
__global__ void hol_pack(const int16_t *__restrict__ c, int n, int e0, int rows, int rpad, int8_t *__restrict__ out) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)rpad * n; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / n), k = (int)(i % n); int8_t lo = 0, hi = 0;
        if (r < rows) split(c[(size_t)(e0 + r) * n + k], lo, hi);
        out[(size_t)r * 2 * n + k] = lo; out[(size_t)r * 2 * n + n + k] = hi;
    }
}

// a signed integer of NL 32-bit limbs, two's complement, little-endian
template <int NL> struct Wide {
    uint32_t l[NL];
    __device__ __forceinline__ void add(long long v) {
        const uint32_t ext = v < 0 ? ~0u : 0u; uint64_t c = 0;
#pragma unroll
        for (int k = 0; k < NL; k++) {
            const uint32_t a = k == 0 ? (uint32_t)v : k == 1 ? (uint32_t)((unsigned long long)v >> 32) : ext;
            const uint64_t t = (uint64_t)l[k] + a + c; l[k] = (uint32_t)t; c = t >> 32;
        }
    }
    __device__ __forceinline__ void add(const Wide &o) {
        uint64_t c = 0;
#pragma unroll
        for (int k = 0; k < NL; k++) { const uint64_t t = (uint64_t)l[k] + o.l[k] + c; l[k] = (uint32_t)t; c = t >> 32; }
    }
    __device__ __forceinline__ void shl8() {
#pragma unroll
        for (int k = NL - 1; k > 0; k--) l[k] = l[k] << 8 | l[k - 1] >> 24;
        l[0] <<= 8;
    }
    // this = floor(this / p), returns the remainder in [0, p); m = floor((2^64 - 1) / p). A negative x is divided as
    // ~x = -x - 1 >= 0: x = -(q + 1) p + (p - 1 - r), and -(q + 1) = ~q.
    __device__ __forceinline__ uint32_t divp(uint32_t p, unsigned long long m) {
        const uint32_t f = (int32_t)l[NL - 1] < 0 ? ~0u : 0u;
        uint32_t rem = 0;
#pragma unroll
        for (int k = NL - 1; k >= 0; k--) {
            const uint64_t cur = (uint64_t)rem << 32 | (l[k] ^ f);   // < p 2^32: the quotient digit fits 32 bits
            uint64_t q = __umul64hi(cur, m); uint32_t r = (uint32_t)(cur - q * p);   // q is exact or one short
            if (r >= p) { q++; r -= p; }
            l[k] = (uint32_t)q ^ f; rem = r;
        }
        return f ? p - 1 - rem : rem;
    }
    __device__ __forceinline__ bool zero() const {
        uint32_t o = 0;
#pragma unroll
        for (int k = 0; k < NL; k++) o |= l[k];
        return !o;
    }
};

// G (rows x (U + 1) n int32, row = a position, column v n + j) -> sum_v 256^v G[r][v n + j] as NL limbs, in place over
// the first NL planes (each thread reads its U + 1 values before writing)
template <int NL>
__global__ void hol_combine(int32_t *__restrict__ G, int rows, int n, int U) {
    const size_t ld = (size_t)(U + 1) * n;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)rows * n; i += (size_t)gridDim.x * blockDim.x) {
        int32_t *g = G + (i / n) * ld + i % n;
        Wide<NL> a;
#pragma unroll
        for (int k = 0; k < NL; k++) a.l[k] = 0;
        for (int v = U; v >= 0; v--) { a.shl8(); a.add((long long)g[(size_t)v * n]); }
#pragma unroll
        for (int k = 0; k < NL; k++) g[(size_t)k * n] = (int32_t)a.l[k];
    }
}

// One thread per entry j: positions [p0, end) of cnew = cold + G rows (positions [g0, g1), combined) + bm (positions
// [b0, b0 + s)) + the carry in, normalized to balanced digits of radix p_{pos mod s}; past p1 only while a carry is left.
// The carry goes back to `carry` for the next chunk; top = max(top, the last nonzero position + 1).
template <int NL>
__global__ void hol_carry(const int16_t *__restrict__ cold, int16_t *__restrict__ cnew, int n, const int32_t *__restrict__ G,
                          int U, int g0, int g1, const long long *__restrict__ bm, int b0, int p0, int p1, int end,
                          uint32_t *__restrict__ carry, int first, int *top) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const int s = hc.s; const size_t ldg = (size_t)(U + 1) * n;
    Wide<NL> t;
#pragma unroll
    for (int k = 0; k < NL; k++) t.l[k] = first ? 0u : carry[(size_t)k * n + j];
    int hi = 0;
    for (int pos = p0; pos < end; pos++) {
        if (pos >= p1 && t.zero()) break;
        if (pos >= g0 && pos < g1) {
            Wide<NL> a; const int32_t *g = G + (size_t)(pos - g0) * ldg + j;
#pragma unroll
            for (int k = 0; k < NL; k++) a.l[k] = (uint32_t)g[(size_t)k * n];
            t.add(a);
        }
        t.add((long long)cold[(size_t)pos * n + j]);
        if (bm && pos >= b0 && pos < b0 + s) t.add(bm[(size_t)(pos - b0) * n + j]);
        const int e = pos % s; const uint32_t p = hc.p[e];
        const uint32_t r = t.divp(p, hc.pm[e]);
        int d = (int)r;
        if (r > p / 2) { d -= (int)p; t.add(1ll); }
        cnew[(size_t)pos * n + j] = (int16_t)d;
        if (d) hi = pos + 1;
    }
#pragma unroll
    for (int k = 0; k < NL; k++) carry[(size_t)k * n + j] = t.l[k];
    if (hi) atomicMax(top, hi);
}

// out[z] = (A_z Bt_z^T) mod r_{z0+z} in [0, r): n x n, operands int8 rows K = n contiguous, nn apart per z
__global__ void __launch_bounds__(THREADS) hol_mm8(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt,
                                                   uint8_t *__restrict__ out, int n, int z0) {
    const int z = blockIdx.z, m = (int)hc8.r[z0 + z]; const size_t nn = (size_t)n * n;
    A += z * nn; Bt += z * nn; out += z * nn;
    gemm_i8_tile(A, Bt, n, [&](int row, int col, int c00, int c01, int c10, int c11) {
        auto f = [m](int v) { v %= m; return (uint8_t)(v < 0 ? v + m : v); };
        *(uchar2 *)&out[(size_t)row * n + col] = make_uchar2(f(c00), f(c01));
        *(uchar2 *)&out[(size_t)(row + 8) * n + col] = make_uchar2(f(c10), f(c11));
    });
}

// A8[r] = A mod q_r, balanced int8 (the B operand of M^T A^T: its rows are the rows of A), from the L byte planes
__global__ void hol_amod8(const int8_t *__restrict__ A, int n, int L, int8_t *__restrict__ A8) {
    const size_t nn = (size_t)n * n; const int r = blockIdx.y; const int q = (int)hc8.r[r];
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nn; i += (size_t)gridDim.x * blockDim.x) {
        int a = 0;
        for (int j = L - 1; j >= 0; j--) { a = (a * 256 + A[j * nn + i]) % q; if (a < 0) a += q; }
        A8[r * nn + i] = bal8(a, q);
    }
}

// Zb from its W residues (uint8, nn apart) -> Zb mod p_t, balanced and split into the [lo | hi] rows of modgemm's A
// operand (K = n, s primes Mpad 2n apart). x = sum xi_u R / r_u - alpha R, xi_u = x_u (R / r_u)^-1 mod r_u,
// alpha = round(sum xi_u / r_u): exact while |x| <= 2^-9 R (the float sum errs by < 2^-17).
template <int SB>
__global__ void __launch_bounds__(256) hol_bext(const uint8_t *__restrict__ Zb, int n, int8_t *__restrict__ Ap, int Mpad, int cmp = 0) {
    const size_t nn = (size_t)n * n; const int W = hc8.W, s = hc.s;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nn; i += (size_t)gridDim.x * blockDim.x) {
        float a = 0; uint32_t acc[SB];
#pragma unroll
        for (int t = 0; t < SB; t++) acc[t] = 0;
        for (int u = 0; u < W; u++) {
            const uint32_t m = hc8.r[u], xi = redm(Zb[u * nn + i] * hc8.cinv[u], m, hc8.rmi[u]);
            a += (float)xi * hc8.rinv[u];
#pragma unroll
            for (int t = 0; t < SB; t++) if (t < s) acc[t] += xi * hc8.Cp[u][t];   // < 2^24 each, W < 64
        }
        const int al = __float2int_rn(a);
        const int j = (int)(i / n), k = (int)(i % n);
#pragma unroll
        for (int t = 0; t < SB; t++) if (t < s) {
            const int p = (int)hc.p[t];
            const int v = (int)redm(acc[t] + 64u * p - (uint32_t)al * hc8.Rp[t], p, hc.pmi[t]);   // al Rp < 64 p, acc < 2^30
            if (cmp) { Ap[(size_t)t * Mpad * n + (size_t)j * n + k] = (int8_t)bal(v, p); continue; }   // p < 256: one byte, rows of n
            int8_t lo, hi; split(bal(v, p), lo, hi);
            int8_t *o = Ap + (size_t)t * Mpad * 2 * n + (size_t)j * 2 * n + k;
            o[0] = lo; o[n] = hi;
        }
    }
}

// X of primes < 256 (det.cu OZ_HOLX8, B0 from the LU with pivoting): B0 mod p_t as balanced int8 rows, from the [A I]
// workspace (columns n..2n), and M^T = Zb B0^T mod p_t as ONE INT8 GEMM of depth n per prime (|sum| <= n 127^2 < 2^31)
// instead of modgemm's two of depth 2n.
__global__ void hol_b8(const int32_t *__restrict__ R, int ld, size_t zr, int8_t *__restrict__ B8, int n) {
    const size_t nn = (size_t)n * n;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < nn * hc.s; e += (size_t)gridDim.x * blockDim.x) {
        const int z = (int)(e / nn); const size_t i = e % nn;
        B8[e] = (int8_t)bal(R[z * zr + (i / n) * ld + n + i % n], (int)hc.p[z]);
    }
}
__global__ void __launch_bounds__(THREADS) hol_mmp(const int8_t *__restrict__ Ap, const int8_t *__restrict__ B8, int32_t *__restrict__ Mres,
                                                   int n, int Mpad) {
    const int z = blockIdx.z, p = (int)hc.p[z];
    const size_t nn = (size_t)n * n;
    int32_t *out = Mres + z * nn;
    gemm_i8_tile(Ap + (size_t)z * Mpad * n, B8 + z * nn, n, [&](int row, int col, int c00, int c01, int c10, int c11) {
        auto f = [&](int v) { v %= p; return v < 0 ? v + p : v; };
        if (row < n) { out[(size_t)row * n + col] = f(c00); out[(size_t)row * n + col + 1] = f(c01); }
        if (row + 8 < n) { out[(size_t)(row + 8) * n + col] = f(c10); out[(size_t)(row + 8) * n + col + 1] = f(c11); }
    });
}

// Z_next = (Zb - AM) / X from the residues v_r mod the WQ q_r (ident: Zb = I), by approximate CRT (|Z| < Q / 4, Q = prod q):
// Z = sum xi_r Q / q_r - alpha Q, xi_r = v_r (Q / q_r)^-1 mod q_r, alpha = round(sum xi_r / q_r). Out: Z mod every modulus
// r_u (balanced int8, Z8[u] row-major) and Z's U balanced byte planes into Bt (as hol_next), the bytes accumulated
// signed from the bytes of Q / q_r. A digit at or above plane U flags an error: a wrong residue gives a value of size ~Q.
template <int QB>
__global__ void __launch_bounds__(256) hol_next8(const uint8_t *__restrict__ Zb8, const uint8_t *__restrict__ AM8, int n, int ident,
                                                int8_t *__restrict__ Z8, int8_t *__restrict__ Bt, int U, int *err) {
    const size_t nn = (size_t)n * n, pl = (size_t)n * 2 * n;
    const int W = hc8.W, WQ = hc8.WQ, nby = hc8.nby;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nn; i += (size_t)gridDim.x * blockDim.x) {
        const int j = (int)(i / n), k = (int)(i % n);
        uint32_t xi[QB]; float a = 0;
#pragma unroll
        for (int r = 0; r < QB; r++) {
            xi[r] = 0;
            if (r < WQ) {
                const uint32_t q = hc8.r[r], qm = hc8.rmi[r], zb = ident ? (j == k) : Zb8[r * nn + i];
                const uint32_t v = redm((zb + q - AM8[r * nn + i]) * hc8.xinv[r], q, qm);
                Z8[r * nn + i] = bal8(v, q);
                xi[r] = redm(v * hc8.qcinv[r], q, qm);
                a += (float)xi[r] * hc8.rinv[r];
            }
        }
        const int al = __float2int_rn(a);
        for (int u = WQ; u < W; u++) {   // sum < 24 * 255^2 < 2^21; al (Q mod r_u) < 24 * 256 < 256 r_u
            const uint32_t m = hc8.r[u]; uint32_t acc = 256 * m - (uint32_t)al * hc8.piq_r[u];
#pragma unroll
            for (int r = 0; r < QB; r++) acc += xi[r] * hc8.Qr[r][u];
            Z8[u * nn + i] = bal8(redm(acc, m, hc8.rmi[u]), m);
        }
        int c = 0, bad = 0; int8_t *o = Bt + (size_t)j * 2 * n + k;
#pragma unroll
        for (int b = 0; b < QB + 2; b++) if (b < nby) {
            int t = c - al * (int)hc8.piq[b];
#pragma unroll
            for (int r = 0; r < QB; r++) t += (int)xi[r] * (int)hc8.Qby[r][b];
            const int z = ((t + 128) & 255) - 128;
            c = (t - z) >> 8;
            if (b < U) { o[b * pl] = (int8_t)z; o[(b + 1) * pl + n] = (int8_t)z; }
            else bad |= z;
        }
        if (bad | c) *err = 1;
    }
}

// dst[z] = src[z]^T, int8 n x n, nn apart; 64 x 64 tiles, block 256
__global__ void __launch_bounds__(256) transpose_i8(const int8_t *__restrict__ src, int8_t *__restrict__ dst, int n) {
    __shared__ int8_t t[64][68];
    const size_t nn = (size_t)n * n; src += blockIdx.z * nn; dst += blockIdx.z * nn;
    const int r0 = blockIdx.y * 64, c0 = blockIdx.x * 64, tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    for (int y = ty; y < 64; y += 16) *(char4 *)&t[y][tx * 4] = *(const char4 *)&src[(size_t)(r0 + y) * n + c0 + tx * 4];
    __syncthreads();
    for (int y = ty; y < 64; y += 16)
        *(char4 *)&dst[(size_t)(c0 + y) * n + r0 + tx * 4] = make_char4(t[tx * 4][y], t[tx * 4 + 1][y], t[tx * 4 + 2][y], t[tx * 4 + 3][y]);
}

struct HolStats { double init = 0, rchain = 0, cchain = 0; int levels = 0, gemms = 0, gemms8 = 0, W = 0, WQ = 0; double ph[8] = {}; };

inline uint32_t hol_invm(uint64_t a, uint64_t p) { uint64_t r = 1, e = p - 2; for (a %= p; e; e >>= 1, a = a * a % p) if (e & 1) r = r * a % p; return (uint32_t)r; }
inline uint32_t hol_inv(int64_t a, int64_t m) {   // any modulus coprime to a (the 8-bit set has prime powers)
    int64_t g = m, x = 0, x1 = 1, b = ((a % m) + m) % m;
    while (b) { const int64_t t = g / b; g -= t * b; std::swap(g, b); x -= t * x1; std::swap(x, x1); }
    return (uint32_t)(((x % m) + m) % m);
}

// The 8-bit moduli for 2 zbits + log2 n + 10 bits of Zb and zbits + 2 of Z, or false: odd primes 17..251 and 243 169 125
// 121 49, largest first (53 moduli, ~360 bits)
inline bool hol8_setup(int n, int zbits, int lgn, const uint32_t *p, int s, HolStats &st) {
    if (n % 128) return false;
    std::vector<uint32_t> m;
    for (uint32_t v = 251; v >= 17; v -= 2) { bool pr = true; for (uint32_t d = 3; d * d <= v; d += 2) if (v % d == 0) { pr = false; break; } if (pr) m.push_back(v); }
    for (uint32_t v : {243u, 169u, 125u, 121u, 49u}) m.push_back(v);
    std::sort(m.rbegin(), m.rend());
    int W = 0, WQ = 0; double lg = 0;
    for (; W < (int)m.size() && lg < 2.0 * zbits + lgn + 10; W++) { lg += log2((double)m[W]); if (!WQ && lg >= zbits + 2) WQ = W + 1; }
    if (lg < 2.0 * zbits + lgn + 10 || W > H8_MAX || WQ > H8_QMAX || WQ == 0) return false;
    Hol8Const h{}; h.W = W; h.WQ = WQ;
    for (int u = 0; u < W; u++) { h.r[u] = m[u]; h.rinv[u] = 1.0f / (float)m[u]; h.rmi[u] = (uint32_t)((1ull << 32) / m[u]); }
    for (int u = 0; u < W; u++) {
        uint64_t x = 1; for (int v = 0; v < W; v++) if (v != u) x = x * m[v] % m[u];
        h.cinv[u] = hol_inv((int64_t)x, m[u]);
        for (int t = 0; t < s; t++) { uint64_t y = 1; for (int v = 0; v < W; v++) if (v != u) y = y * m[v] % p[t]; h.Cp[u][t] = (uint16_t)y; }
    }
    for (int t = 0; t < s; t++) { uint64_t y = 1; for (int v = 0; v < W; v++) y = y * m[v] % p[t]; h.Rp[t] = (uint16_t)y; }
    for (int v = 0; v < WQ; v++) for (int u = 0; u < WQ; u++) if (u < v) h.ginv[u][v] = (uint16_t)hol_inv(m[u], m[v]);
    for (int v = 0; v < WQ; v++) for (int u = 0; u < W; u++) h.qr[v][u] = (uint16_t)(m[v] % m[u]);
    for (int u = 0; u < W; u++) { uint64_t x = 1; for (int v = 0; v < WQ; v++) x = x * m[v] % m[u]; h.piq_r[u] = (uint16_t)x; }
    for (int r = 0; r < WQ; r++) { uint64_t x = 1; for (int t = 0; t < s; t++) x = x * p[t] % m[r]; h.xinv[r] = (uint16_t)hol_inv((int64_t)x, m[r]); }
    for (int t = 0; t < s; t++) for (int r = 0; r < WQ; r++) h.pq8[t][r] = (uint16_t)(p[t] % m[r]);
    { uint32_t by[HOL_BMAX] = {1}; double lq = 0;
      for (int r = 0; r < WQ; r++) { lq += log2((double)m[r]); uint32_t c = 0; for (int u = 0; u < HOL_BMAX; u++) { uint32_t v = by[u] * m[r] + c; by[u] = v & 255; c = v >> 8; } }
      for (int u = 0; u < HOL_BMAX; u++) h.piq[u] = (uint8_t)by[u];
      h.nby = std::min(HOL_BMAX, (int)ceil(lq / 8) + 1); }
    if (h.nby > WQ + 2) return false;
    for (int r = 0; r < WQ; r++) {   // Q / q_r: its residues and bytes
        uint32_t by[H8_QMAX + 2] = {1};
        for (int v = 0; v < WQ; v++) if (v != r) { uint32_t c = 0; for (int b = 0; b < WQ + 2; b++) { uint32_t t = by[b] * m[v] + c; by[b] = t & 255; c = t >> 8; } }
        for (int b = 0; b < WQ + 2; b++) h.Qby[r][b] = (uint8_t)by[b];
        uint64_t x = 1; for (int v = 0; v < WQ; v++) if (v != r) x = x * m[v] % m[r];
        h.qcinv[r] = (uint16_t)hol_inv((int64_t)x, m[r]);
        for (int u = 0; u < W; u++) { uint64_t y = 1; for (int v = 0; v < WQ; v++) if (v != r) y = y * m[v] % m[u]; h.Qr[r][u] = (uint8_t)y; }
    }
    cudaMemcpyToSymbol(hc8, &h, sizeof h);
    st.W = W; st.WQ = WQ;
    return true;
}

// y = b A^-1 mod X^N, X = p_0 ... p_{s-1}. lu: the LU workspace of [A I] modulo the s primes (A^-1 mod p_t in columns
// n..2n), freed here. dA: L planes of signed bytes. q: w primes, none among the p, prod q > 2^(zbits + 1), |R_i| < 2^zbits.
// Returns the balanced digits of entries 0..nout-1 of y: digit e (radix p_{e mod s}) of entry j at out[e * nout + j], s N digits.
inline std::vector<int16_t> hol_solve(DetWork &lu, const int8_t *dA, int L, const uint32_t *p, int s, const uint32_t *q,
                                     int w, const int32_t *b, int N, int zbits, HolStats &st, int *err_out, int nout = 4) {
    using clk = std::chrono::steady_clock;
    auto now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = lu.n, sN = s * N;
    int U = (zbits + 2 + 7) / 8;
    if ((U + 1) * n % BN) U++;   // n % 128 == 64: one more (zero) plane, so the vector GEMM has whole 128-column tiles
    const size_t nn = (size_t)n * n;
    // the carry needs |c digit| |Z| n < 2^(15 + log2 n + zbits), a sign bit and room for the carry itself
    int lgn = 0; while ((1 << lgn) < n) lgn++;
    const int nlb = 15 + lgn + zbits + 3, NL = nlb <= 128 ? 4 : nlb <= 192 ? 6 : nlb <= 256 ? 8 : nlb <= 320 ? 10 : 12;
    if (s > HOL_SMAX || w > HOL_WMAX || nlb > 384 || 2 * w > HOL_BMAX || NL > U + 1) { fprintf(stderr, "HOL: s %d w %d bits %d out of range\n", s, w, nlb); exit(1); }
    double t0 = now();

    HolConst h{}; h.s = s; h.w = w;
    for (int t = 0; t < s; t++) { h.p[t] = p[t]; h.pm[t] = ~0ull / p[t]; h.pmi[t] = (uint32_t)((1ull << 32) / p[t]); }
    for (int r = 0; r < w; r++) h.q[r] = q[r];
    for (int t = 0; t < s; t++) for (int u = 0; u < t; u++) h.pinv[u][t] = hol_invm(p[u], p[t]);
    for (int r = 0; r < w; r++) for (int u = 0; u < r; u++) h.qinv[u][r] = hol_invm(q[u], q[r]);
    for (int t = 0; t < s; t++) for (int r = 0; r < w; r++) { h.pq[t][r] = p[t] % q[r]; h.qp[r][t] = q[r] % p[t]; }
    for (int r = 0; r < w; r++) { uint64_t x = 1; for (int t = 0; t < s; t++) x = x * p[t] % q[r]; h.xinv[r] = hol_invm(x, q[r]); }
    for (int t = 0; t < s; t++) { uint64_t x = 1; for (int r = 0; r < w; r++) x = x * q[r] % p[t]; h.piq_p[t] = (uint32_t)x; }
    { uint32_t by[HOL_BMAX] = {1};   // prod q in bytes
      for (int r = 0; r < w; r++) { uint32_t c = 0; for (int u = 0; u < HOL_BMAX; u++) { uint32_t v = by[u] * q[r] + c; by[u] = v & 255; c = v >> 8; } }
      for (int u = 0; u < HOL_BMAX; u++) h.piq[u] = (uint8_t)by[u]; }
    cudaMemcpyToSymbol(hc, &h, sizeof h);

    // r8: the residue chain over 8-bit moduli (hol8_setup); OZ_HOL16=1 keeps the 16-bit chain
    const bool r8 = !getenv("OZ_HOL16") && hol8_setup(n, zbits, lgn, p, s, st);
    const int wo = r8 ? 0 : w, W8 = st.W, WQ8 = st.WQ;   // wo: the 16-bit q of the old chain

    // buffers: residue matrices n^2 int32 each
    DetWork hw; hw.alloc(n, s + wo);   // its R (s + w matrices) holds Zb = Z^2; its packing buffers serve every product
    { std::vector<uint32_t> pqv(p, p + s); pqv.insert(pqv.end(), q, q + wo); cudaMemcpy(hw.P, pqv.data(), (s + wo) * 4, cudaMemcpyHostToDevice); }
    int32_t *B0T, *AT, *Zres, *Mres, *Mq, *AMq, *db; int8_t *Bt, *Apk; int16_t *cb[2]; int32_t *G; long long *bm;
    int8_t *Z8 = nullptr, *Z8T = nullptr, *Mq8 = nullptr, *A8 = nullptr; uint8_t *Zb8 = nullptr, *AM8 = nullptr;
    uint32_t *carry; int *dtop, *derr;
    const int Ec = std::max(128, std::min(4096, (int)((256ull << 20) / ((size_t)(U + 1) * n * 4)) / 128 * 128));   // chunk rows
    cudaMalloc(&B0T, s * nn * 4); cudaMalloc(&AT, wo * nn * 4); cudaMalloc(&Zres, (r8 ? 0 : s + w) * nn * 4);
    cudaMalloc(&Mres, s * nn * 4); cudaMalloc(&Mq, wo * nn * 4); cudaMalloc(&AMq, wo * nn * 4);
    if (r8) {
        cudaMalloc(&Z8, W8 * nn); cudaMalloc(&Z8T, W8 * nn); cudaMalloc(&Zb8, W8 * nn);
        cudaMalloc(&Mq8, WQ8 * nn); cudaMalloc(&AM8, WQ8 * nn); cudaMalloc(&A8, WQ8 * nn);
        if (getenv("OZ_POISON")) for (void *v : {(void *)Z8, (void *)Z8T, (void *)Zb8, (void *)Mq8, (void *)AM8, (void *)A8}) cudaMemset(v, 0x5a, W8 * nn > WQ8 * nn && (v == Z8 || v == Z8T || v == Zb8) ? W8 * nn : WQ8 * nn);
        if (getenv("OZ_POISON")) cudaMemset(hw.Ap, 0x5a, (size_t)s * (n + BM) * 2 * n);
    }
    cudaMalloc(&Bt, (size_t)(U + 1) * n * 2 * n); cudaMemset(Bt, 0, (size_t)(U + 1) * n * 2 * n);
    for (auto &c : cb) { cudaMalloc(&c, (size_t)sN * n * 2); cudaMemset(c, 0, (size_t)sN * n * 2); }
    cudaMalloc(&G, (size_t)Ec * (U + 1) * n * 4); cudaMalloc(&Apk, (size_t)Ec * 2 * n);
    cudaMalloc(&bm, (size_t)s * n * 8); cudaMalloc(&db, n * 4); cudaMalloc(&carry, (size_t)NL * n * 4);
    cudaMalloc(&dtop, 4); cudaMalloc(&derr, 4); cudaMemset(derr, 0, 4);
    cudaMemcpy(db, b, n * 4, cudaMemcpyHostToDevice);

    // B0^T = (A^-1 mod p_t)^T from the LU, A^T mod q_r through Zres (r8: A mod the 8-bit q)
    const dim3 tg(n / 32, n / 32, s), tb(32, 8);
    transpose_i32<<<tg, tb>>>(lu.R + n, lu.ld, lu.zr(), B0T, n);
    // X of primes < 256 (det.cu OZ_HOLX8): M^T = Zb B0^T as one INT8 GEMM per prime, B0 kept as balanced bytes
    bool x8 = r8; for (int t = 0; t < s; t++) x8 = x8 && p[t] < 256;
    int8_t *B8 = nullptr;
    if (x8) { cudaMalloc(&B8, s * nn); hol_b8<<<4096, 256>>>(lu.R, lu.ld, lu.zr(), B8, n); }
    cudaStreamSynchronize(cudaStreamPerThread); lu.free_();
    if (r8) hol_amod8<<<dim3(256, WQ8), 256>>>(dA, n, L, A8);
    else {
        det_init<<<dim3(256, w), 256>>>(dA, Zres, n, n, hw.P + s, L);
        transpose_i32<<<dim3(n / 32, n / 32, w), tb>>>(Zres, n, nn, AT, n);
    }
    // the right factors that never change, packed once: B0^T (M^T = Zb B0^T) and A^T ((A M)^T = M^T A^T)
    const int Mpad = rup(n, BM), Npad = rup(n, BN); const size_t pk = (size_t)Npad * 2 * n;
    int8_t *PB[4]; cudaMalloc(&PB[0], s * pk); cudaMalloc(&PB[1], s * pk); cudaMalloc(&PB[2], wo * pk); cudaMalloc(&PB[3], wo * pk);
    pack_b<<<dim3(Npad / 64, n / 64, s), 256>>>(B0T, nn, n, n, n, Npad, PB[0], PB[1], hw.P);
    if (!r8) pack_b<<<dim3(Npad / 64, n / 64, w), 256>>>(AT, nn, n, n, n, Npad, PB[2], PB[3], hw.P + s);
    cudaStreamSynchronize(cudaStreamPerThread); cudaFree(AT); AT = nullptr;

    // out = X W modulo nz primes from z0 (X, W: residue matrices nn apart per prime)
    auto mgemm = [&](int32_t *out, int z0, int nz, const int32_t *X, const int32_t *W) {
        DetWork v = hw; v.R = out; v.P = hw.P + z0; v.Pb = nz;
        mod_xw<false>(v, X, nn, n, W, nn, n, n, n, n, 0, 0);
        st.gemms += nz;
    };
    auto mgemm_pre = [&](int32_t *out, int z0, int nz, const int32_t *X, int8_t *const *B) {   // W packed in B[0], B[1]
        const uint32_t *P = hw.P + z0;
        det_prof.begin();
        pack_a<<<dim3(std::min(2048, Mpad * n / 1024 + 1), nz), 256>>>(X, nn, n, n, n, Mpad, hw.Ap, P);
        det_prof.end(1); det_prof.begin();
        modgemm<0><<<dim3(Npad / BN, Mpad / BM, nz), THREADS>>>(hw.Ap, B[0], B[1], out, nn, n, n, n, 2 * n, Mpad, Npad, P);
        det_prof.end(2);
        st.gemms += nz;
    };
    const int gb = (int)std::min<size_t>(8192, (nn + 255) / 256);
#define OZ_SB(K, ...) do { if (s <= 8) K<8>__VA_ARGS__; else if (s <= 16) K<16>__VA_ARGS__; else K<24>__VA_ARGS__; } while (0)
    double tph = 0;   // OZ_PROF: synchronous phase times of the 8-bit chain into st.ph
    auto ph = [&](int k) { if (!det_prof.on) return; cudaDeviceSynchronize(); const double t = now(); if (k >= 0) st.ph[k] += t - tph; tph = t; };
    auto next = [&](const int32_t *Zb, int ident) {   // Z_next from Zb and M^T (in Mq): A M, then the division by X
        mgemm_pre(AMq, s, w, Mq, PB + 2);
        hol_next<<<gb, 256>>>(Zb, AMq, n, ident, Zres, Bt, U, zbits, derr);
    };
    auto next8 = [&](int ident) {   // the same over the 8-bit q: (A M)^T = M^T A^T mod q, Z_next mod every modulus, Z^T
        ph(-1);
        hol_mm8<<<dim3(n / BN, n / BM, WQ8), THREADS>>>(Mq8, A8, AM8, n, 0); st.gemms8 += WQ8;
        ph(4);
        if (WQ8 <= 12) hol_next8<12><<<gb, 256>>>(Zb8, AM8, n, ident, Z8, Bt, U, derr);
        else hol_next8<24><<<gb, 256>>>(Zb8, AM8, n, ident, Z8, Bt, U, derr);
        ph(5);
        transpose_i8<<<dim3(n / 64, n / 64, W8), 256>>>(Z8, Z8T, n);
        ph(6);
    };
    int top[2] = {0, 0}, cur = 0;
    auto carry_k = [&](const int16_t *co, int16_t *cn, const int32_t *g, int g0, int g1, const long long *bmp, int b0,
                       int p0, int p1, int end, int first) {
#define OZ_HC(K) case K: hol_carry<K><<<(n + 63) / 64, 64>>>(co, cn, n, g, U, g0, g1, bmp, b0, p0, p1, end, carry, first, dtop); break;
        switch (NL) { OZ_HC(4) OZ_HC(6) OZ_HC(8) OZ_HC(10) OZ_HC(12) }
#undef OZ_HC
    };
    auto combine_k = [&](int rows) {
        const int gr = (int)std::min<size_t>(8192, ((size_t)rows * n + 255) / 256);
#define OZ_HB(K) case K: hol_combine<K><<<gr, 256>>>(G, rows, n, U); break;
        switch (NL) { OZ_HB(4) OZ_HB(6) OZ_HB(8) OZ_HB(10) OZ_HB(12) }
#undef OZ_HB
    };
    auto set_top = [&](int v) { cudaMemcpy(dtop, &v, 4, cudaMemcpyHostToDevice); };
    auto get_top = [&] { int v; cudaMemcpy(&v, dtop, 4, cudaMemcpyDeviceToHost); return v; };

    // level -1: c_0 = b B0 (digits of B0 by Garner), Z_0 = (I - B0^T A^T) / X
    int I = 0; while ((2ll << I) - 1 < N) I++;   // m_I = 2^(I+1) - 1 >= N
    st.levels = I;
    OZ_SB(hol_garner, <<<(n + 7) / 8, 256>>>(B0T, db, n, I > 0 && !r8 ? Mq : nullptr, bm, I > 0 && r8 ? Mq8 : nullptr));
    set_top(0);
    carry_k(cb[1], cb[0], nullptr, 0, 0, bm, 0, 0, std::min(s, sN), sN, 1);
    top[0] = get_top();
    if (I > 0) { if (r8) next8(1); else next(nullptr, 1); }
    cudaStreamSynchronize(cudaStreamPerThread); st.init = now() - t0;

    for (int i = 0; i < I; i++) {
        const long long m = (2ll << i) - 1;
        const bool needM = 2 * m < N, needZ = i + 1 < I;
        double ta = now();
        int32_t *Zb = hw.R;
        if (r8 && (needM || needZ)) {   // Zb exact over the W moduli, to the p_t by base extension, then M^T = Zb B0^T mod p_t
            ph(-1);
            hol_mm8<<<dim3(n / BN, n / BM, W8), THREADS>>>(Z8, Z8T, Zb8, n, 0); st.gemms8 += W8;
            ph(0);
            OZ_SB(hol_bext, <<<gb, 256>>>(Zb8, n, hw.Ap, Mpad, x8));
            ph(1);
            if (x8) { hol_mmp<<<dim3(n / BN, Mpad / BM, s), THREADS>>>(hw.Ap, B8, Mres, n, Mpad); st.gemms8 += s; }
            else { modgemm<0><<<dim3(Npad / BN, Mpad / BM, s), THREADS>>>(hw.Ap, PB[0], PB[1], Mres, nn, n, n, n, 2 * n, Mpad, Npad, hw.P); st.gemms += s; }
            ph(2);
            OZ_SB(hol_garner, <<<(n + 7) / 8, 256>>>(Mres, db, n, nullptr, bm, needZ ? Mq8 : nullptr));
            ph(3);
        } else if (needM || needZ) {
            mgemm(Zb, 0, needZ ? s + w : s, Zres, Zres);
            mgemm_pre(Mres, 0, s, Zb, PB);
            OZ_SB(hol_garner, <<<(n + 7) / 8, 256>>>(Mres, db, n, needZ ? Mq : nullptr, bm));
        }
        cudaStreamSynchronize(cudaStreamPerThread); st.rchain += now() - ta; ta = now();
        // c_{i+1} = c_i + X^m (c_i mod X^(N - m)) R_i + X^(2m) b M_i, mod X^N
        const int16_t *co = cb[cur]; int16_t *cn = cb[cur ^ 1];
        const int sm = (int)(s * m), E = std::min(top[cur], (int)(s * (N - m))), b0 = needM ? (int)(2 * m * s) : -1;
        const int hi = std::min(sN, std::max({sm + E, top[cur], top[cur ^ 1], needM ? b0 + s : 0}));
        cudaMemcpyAsync(cn, co, (size_t)sm * n * 2, cudaMemcpyDeviceToDevice);
        set_top(std::min(top[cur], sm));
        const long long *bmp = needM ? bm : nullptr;
        if (E == 0) carry_k(co, cn, nullptr, 0, 0, bmp, b0, sm, hi, sN, 1);
        for (int e0 = 0; e0 < E; e0 += Ec) {
            const int rows = std::min(Ec, E - e0), rpad = rup(rows, BM), last = e0 + Ec >= E;
            hol_pack<<<2048, 256>>>(co, n, e0, rows, rpad, Apk);
            gemm_i8<<<dim3((U + 1) * n / BN, rpad / BM, 1), THREADS>>>(Apk, Bt, G, rpad, (U + 1) * n, 2 * n);
            combine_k(rows);
            carry_k(co, cn, G, sm + e0, sm + e0 + rows, bmp, b0, sm + e0, last ? hi : sm + e0 + rows, last ? sN : sm + e0 + rows, e0 == 0);
        }
        top[cur ^ 1] = get_top(); cur ^= 1;
        st.cchain += now() - ta; ta = now();
        if (needZ) { if (r8) next8(0); else next(Zb + s * nn, 0); }   // Zb mod q_r: after the s matrices mod p_t
        cudaStreamSynchronize(cudaStreamPerThread); st.rchain += now() - ta;
    }

#undef OZ_SB
    std::vector<int16_t> out((size_t)sN * nout);   // entries 0 .. nout - 1 of every position
    cudaMemcpy2D(out.data(), (size_t)nout * 2, cb[cur], (size_t)n * 2, (size_t)nout * 2, sN, cudaMemcpyDeviceToHost);
    cudaMemcpy(err_out, derr, 4, cudaMemcpyDeviceToHost);
    hw.free_();
    for (void *v : {(void *)B8, (void *)B0T, (void *)PB[0], (void *)PB[1], (void *)PB[2], (void *)PB[3], (void *)Zres, (void *)Mres, (void *)Mq, (void *)AMq, (void *)Bt, (void *)cb[0],
                    (void *)cb[1], (void *)G, (void *)Apk, (void *)bm, (void *)db, (void *)carry, (void *)dtop, (void *)derr, (void *)Z8, (void *)Z8T, (void *)Zb8, (void *)Mq8, (void *)AM8, (void *)A8})
        cudaFree(v);
    return out;
}

}  // namespace oz
