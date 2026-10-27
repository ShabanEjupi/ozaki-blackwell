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
#include "ozaki_det.cuh"

namespace oz {

constexpr int HOL_SMAX = 16, HOL_WMAX = 12, HOL_BMAX = 2 * HOL_WMAX;   // primes of X, primes q, bytes below prod q

struct HolConst {
    int s, w;
    uint32_t p[HOL_SMAX], q[HOL_WMAX];
    uint32_t pinv[HOL_SMAX][HOL_SMAX], qinv[HOL_WMAX][HOL_WMAX];   // [u][t] = p_u^-1 mod p_t (q likewise)
    uint32_t pq[HOL_SMAX][HOL_WMAX], qp[HOL_WMAX][HOL_SMAX];       // p_t mod q_r, q_r mod p_t
    uint32_t xinv[HOL_WMAX], piq_p[HOL_SMAX];                      // X^-1 mod q_r, prod q mod p_t
    unsigned long long pm[HOL_SMAX];                               // floor((2^64 - 1) / p_t)
    uint8_t piq[HOL_BMAX];                                         // prod q, little-endian bytes
};
__constant__ HolConst hc;

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
// [0, X); Mq[r] = M^T mod q_r (unless null); bm[t][j] = sum_k b[k] y_t(j, k), i.e. digit t of (b M)[j], unnormalized.
__global__ void __launch_bounds__(256) hol_garner(const int32_t *__restrict__ Mt, const int32_t *__restrict__ b, int n,
                                                 int32_t *__restrict__ Mq, long long *__restrict__ bm) {
    const int lane = threadIdx.x & 31, j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= n) return;
    const int s = hc.s, w = hc.w; const size_t nn = (size_t)n * n;
    long long acc[HOL_SMAX];
#pragma unroll
    for (int t = 0; t < HOL_SMAX; t++) acc[t] = 0;
    for (int k = lane; k < n; k += 32) {
        const size_t i = (size_t)j * n + k; const int bk = b[k];
        uint32_t y[HOL_SMAX];
#pragma unroll
        for (int t = 0; t < HOL_SMAX; t++) if (t < s) {
            const uint32_t p = hc.p[t]; uint32_t v = (uint32_t)Mt[t * nn + i];
#pragma unroll
            for (int u = 0; u < t; u++) v = (v + p - y[u] % p) % p * hc.pinv[u][t] % p;
            y[t] = v; acc[t] += (long long)bk * (int)v;
        }
        if (Mq)
#pragma unroll
            for (int r = 0; r < HOL_WMAX; r++) if (r < w) {   // Horner from the top digit; m p + y < 65279^2 + 65279 < 2^32
                const uint32_t q = hc.q[r]; uint32_t m = 0;
#pragma unroll
                for (int t = HOL_SMAX - 1; t >= 0; t--) if (t < s) m = (m * hc.pq[t][r] + y[t]) % q;
                Mq[r * nn + i] = m;
            }
    }
#pragma unroll
    for (int t = 0; t < HOL_SMAX; t++) if (t < s) {
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

struct HolStats { double init = 0, rchain = 0, cchain = 0; int levels = 0, gemms = 0; };

inline uint32_t hol_invm(uint64_t a, uint64_t p) { uint64_t r = 1, e = p - 2; for (a %= p; e; e >>= 1, a = a * a % p) if (e & 1) r = r * a % p; return (uint32_t)r; }

// y = b A^-1 mod X^N, X = p_0 ... p_{s-1}. lu: the LU workspace of [A I] modulo the s primes (A^-1 mod p_t in columns
// n..2n), freed here. dA: L planes of signed bytes. q: w primes, none among the p, prod q > 2^(zbits + 1), |R_i| < 2^zbits.
// Returns the balanced digits of entries 0..3 of y: digit e (radix p_{e mod s}) of entry j at out[e * 4 + j], s N digits.
inline std::vector<int16_t> hol_solve(DetWork &lu, const int8_t *dA, int L, const uint32_t *p, int s, const uint32_t *q,
                                     int w, const int32_t *b, int N, int zbits, HolStats &st, int *err_out) {
    using clk = std::chrono::steady_clock;
    auto now = [] { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); };
    const int n = lu.n, U = (zbits + 2 + 7) / 8, sN = s * N;
    const size_t nn = (size_t)n * n;
    // the carry needs |c digit| |Z| n < 2^(15 + log2 n + zbits), a sign bit and room for the carry itself
    int lgn = 0; while ((1 << lgn) < n) lgn++;
    const int nlb = 15 + lgn + zbits + 3, NL = nlb <= 128 ? 4 : nlb <= 192 ? 6 : 8;
    if (s > HOL_SMAX || w > HOL_WMAX || nlb > 256 || 2 * w > HOL_BMAX || NL > U + 1) { fprintf(stderr, "HOL: s %d w %d bits %d out of range\n", s, w, nlb); exit(1); }
    double t0 = now();

    HolConst h{}; h.s = s; h.w = w;
    for (int t = 0; t < s; t++) { h.p[t] = p[t]; h.pm[t] = ~0ull / p[t]; }
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

    // buffers: residue matrices n^2 int32 each
    DetWork hw; hw.alloc(n, s + w);   // its R (s + w matrices) holds Zb = Z^2; its packing buffers serve every product
    { std::vector<uint32_t> pqv(p, p + s); pqv.insert(pqv.end(), q, q + w); cudaMemcpy(hw.P, pqv.data(), (s + w) * 4, cudaMemcpyHostToDevice); }
    int32_t *B0T, *AT, *Zres, *Mres, *Mq, *AMq, *db; int8_t *Bt, *Apk; int16_t *cb[2]; int32_t *G; long long *bm;
    uint32_t *carry; int *dtop, *derr;
    const int Ec = std::max(128, std::min(4096, (int)((256ull << 20) / ((size_t)(U + 1) * n * 4)) / 128 * 128));   // chunk rows
    cudaMalloc(&B0T, s * nn * 4); cudaMalloc(&AT, w * nn * 4); cudaMalloc(&Zres, (s + w) * nn * 4);
    cudaMalloc(&Mres, s * nn * 4); cudaMalloc(&Mq, w * nn * 4); cudaMalloc(&AMq, w * nn * 4);
    cudaMalloc(&Bt, (size_t)(U + 1) * n * 2 * n); cudaMemset(Bt, 0, (size_t)(U + 1) * n * 2 * n);
    for (auto &c : cb) { cudaMalloc(&c, (size_t)sN * n * 2); cudaMemset(c, 0, (size_t)sN * n * 2); }
    cudaMalloc(&G, (size_t)Ec * (U + 1) * n * 4); cudaMalloc(&Apk, (size_t)Ec * 2 * n);
    cudaMalloc(&bm, (size_t)s * n * 8); cudaMalloc(&db, n * 4); cudaMalloc(&carry, (size_t)NL * n * 4);
    cudaMalloc(&dtop, 4); cudaMalloc(&derr, 4); cudaMemset(derr, 0, 4);
    cudaMemcpy(db, b, n * 4, cudaMemcpyHostToDevice);

    // B0^T = (A^-1 mod p_t)^T from the LU, A^T mod q_r through Zres
    const dim3 tg(n / 32, n / 32, s), tb(32, 8);
    transpose_i32<<<tg, tb>>>(lu.R + n, lu.ld, lu.zr(), B0T, n);
    cudaDeviceSynchronize(); lu.free_();
    det_init<<<dim3(256, w), 256>>>(dA, Zres, n, n, hw.P + s, L);
    transpose_i32<<<dim3(n / 32, n / 32, w), tb>>>(Zres, n, nn, AT, n);

    // out = X W modulo nz primes from z0 (X, W: residue matrices nn apart per prime)
    auto mgemm = [&](int32_t *out, int z0, int nz, const int32_t *X, const int32_t *W) {
        DetWork v = hw; v.R = out; v.P = hw.P + z0; v.Pb = nz;
        mod_xw<false>(v, X, nn, n, W, nn, n, n, n, n, 0, 0);
        st.gemms += nz;
    };
    const int gb = (int)std::min<size_t>(8192, (nn + 255) / 256);
    auto next = [&](const int32_t *Zb, int ident) {   // Z_next from Zb and M^T (in Mq): A M, then the division by X
        mgemm(AMq, s, w, Mq, AT);
        hol_next<<<gb, 256>>>(Zb, AMq, n, ident, Zres, Bt, U, zbits, derr);
    };
    int top[2] = {0, 0}, cur = 0;
    auto carry_k = [&](const int16_t *co, int16_t *cn, const int32_t *g, int g0, int g1, const long long *bmp, int b0,
                       int p0, int p1, int end, int first) {
#define OZ_HC(K) case K: hol_carry<K><<<(n + 63) / 64, 64>>>(co, cn, n, g, U, g0, g1, bmp, b0, p0, p1, end, carry, first, dtop); break;
        switch (NL) { OZ_HC(4) OZ_HC(6) OZ_HC(8) }
#undef OZ_HC
    };
    auto combine_k = [&](int rows) {
        const int gr = (int)std::min<size_t>(8192, ((size_t)rows * n + 255) / 256);
        switch (NL) { case 4: hol_combine<4><<<gr, 256>>>(G, rows, n, U); break;
                      case 6: hol_combine<6><<<gr, 256>>>(G, rows, n, U); break;
                      default: hol_combine<8><<<gr, 256>>>(G, rows, n, U); }
    };
    auto set_top = [&](int v) { cudaMemcpy(dtop, &v, 4, cudaMemcpyHostToDevice); };
    auto get_top = [&] { int v; cudaMemcpy(&v, dtop, 4, cudaMemcpyDeviceToHost); return v; };

    // level -1: c_0 = b B0 (digits of B0 by Garner), Z_0 = (I - B0^T A^T) / X
    int I = 0; while ((2ll << I) - 1 < N) I++;   // m_I = 2^(I+1) - 1 >= N
    st.levels = I;
    hol_garner<<<(n + 7) / 8, 256>>>(B0T, db, n, I > 0 ? Mq : nullptr, bm);
    set_top(0);
    carry_k(cb[1], cb[0], nullptr, 0, 0, bm, 0, 0, std::min(s, sN), sN, 1);
    top[0] = get_top();
    if (I > 0) next(nullptr, 1);
    cudaDeviceSynchronize(); st.init = now() - t0;

    for (int i = 0; i < I; i++) {
        const long long m = (2ll << i) - 1;
        const bool needM = 2 * m < N, needZ = i + 1 < I;
        double ta = now();
        int32_t *Zb = hw.R;
        if (needM || needZ) {
            mgemm(Zb, 0, needZ ? s + w : s, Zres, Zres);
            mgemm(Mres, 0, s, Zb, B0T);
            hol_garner<<<(n + 7) / 8, 256>>>(Mres, db, n, needZ ? Mq : nullptr, bm);
        }
        cudaDeviceSynchronize(); st.rchain += now() - ta; ta = now();
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
        if (needZ) next(Zb + s * nn, 0);   // Zb mod q_r: after the s matrices mod p_t
        cudaDeviceSynchronize(); st.rchain += now() - ta;
    }

    std::vector<int16_t> out((size_t)sN * 4);
    cudaMemcpy2D(out.data(), 8, cb[cur], (size_t)n * 2, 8, sN, cudaMemcpyDeviceToHost);
    cudaMemcpy(err_out, derr, 4, cudaMemcpyDeviceToHost);
    hw.free_();
    for (void *v : {(void *)B0T, (void *)AT, (void *)Zres, (void *)Mres, (void *)Mq, (void *)AMq, (void *)Bt, (void *)cb[0],
                    (void *)cb[1], (void *)G, (void *)Apk, (void *)bm, (void *)db, (void *)carry, (void *)dtop, (void *)derr})
        cudaFree(v);
    return out;
}

}  // namespace oz
