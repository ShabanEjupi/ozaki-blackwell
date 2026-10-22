// ozaki_td.cuh -- triple-double GEMM (~159-bit significand) from INT8 tensor cores: precision as a software choice, on a
// GPU that has 53-bit FP64 and nothing wider. The Ozaki-II pipeline of ozaki_dd.cuh with up to 49 moduli (product ~2^342),
// inputs and output as (hi, mid, lo) triples.
//   residues: x * 2^s from the three parts; each integral part split exactly into 53-bit int64 pieces, then mod m;
//   CRT:      the direct sum over 12 32-bit limbs on the integer units (as crt_dd), then three doubles peeled off the
//             integer one by one -- no floating-point arithmetic on the 342-bit value at all.
#pragma once
#include "ozaki_dd.cuh"

namespace oz {

constexpr int TDL = 12;   // limbs of the CRT sum: M < 2^343 and S < 49 * 256 * M < 2^357 < 2^384
__constant__ unsigned c_W3[MAXM][TDL], c_M3[TDL];
__constant__ long long c_F3[MAXM];
inline void crt_consts_td(int nm) {
    static int done = -1;
    if (done == nm) return;
    done = nm;
    unsigned W[MAXM][TDL] = {}, M[TDL] = {}; long long F[MAXM] = {};
    auto mul_small = [](unsigned *a, unsigned m) {
        unsigned long long c = 0;
        for (int k = 0; k < TDL; k++) { c += (unsigned long long)a[k] * m; a[k] = (unsigned)c; c >>= 32; }
    };
    M[0] = 1; for (int l = 0; l < nm; l++) mul_small(M, MODULI[l]);
    for (int l = 0; l < nm; l++) {
        int r = 1; W[l][0] = 1;
        for (int t = 0; t < nm; t++) if (t != l) { mul_small(W[l], MODULI[t]); r = r * (MODULI[t] % MODULI[l]) % MODULI[l]; }
        int w = inv_mod(r, MODULI[l]);
        mul_small(W[l], w);
        F[l] = llround(ldexp((double)w / MODULI[l], 40));
    }
    cudaMemcpyToSymbol(c_W3, W, sizeof W); cudaMemcpyToSymbol(c_M3, M, sizeof M); cudaMemcpyToSymbol(c_F3, F, sizeof F);
}

// an integral double, |v| < 2^160, as v = p2 2^106 + p1 2^53 + p0 -- every step exact (the pieces are bits of v's mantissa)
__device__ __forceinline__ void split3(double v, long long &p2, long long &p1, long long &p0) {
    double t2 = trunc(ldexp(v, -106)), r = v - ldexp(t2, 106);
    double t1 = trunc(ldexp(r, -53)), r0 = r - ldexp(t1, 53);
    p2 = (long long)t2; p1 = (long long)t1; p0 = (long long)r0;
}

template <int NM>
__global__ void residues_td(const double *Xh, const double *Xm, const double *Xl, int len, long long sRow, long long sCol,
                            const int *s, int8_t *R, int ld, size_t plane) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (k >= len) return;
    size_t at = r * sRow + k * sCol;
    double a = ldexp(Xh[at], s[r]), b = ldexp(Xm[at], s[r]), c = ldexp(Xl[at], s[r]);
    double ai = trunc(a), bi = trunc(b), ci = trunc(c);
    double f = rint((a - ai) + (b - bi) + (c - ci));   // |f| <= 3
    long long u[9];
    split3(ai, u[0], u[1], u[2]); split3(bi, u[3], u[4], u[5]); split3(ci, u[6], u[7], u[8]);
    each<0, NM>([&](auto L) {
        constexpr int m = modulus(decltype(L)::value), p53 = pow2mod(53, m), p106 = pow2mod(106, m);
        int v = (balc<m>(u[0]) + balc<m>(u[3]) + balc<m>(u[6])) * p106 + (balc<m>(u[1]) + balc<m>(u[4]) + balc<m>(u[7])) * p53 +
                balc<m>(u[2]) + balc<m>(u[5]) + balc<m>(u[8]) + (int)f;   // |.| < 3 * 128 * 256 + 3 * 128 + 3
        R[decltype(L)::value * plane + (size_t)r * ld + k] = (int8_t)balc<m>(v);
    });
}

// 256-bit unsigned in four 64-bit words, X3 the most significant; only compile-time word indices (registers, not stack)
struct U256 { unsigned long long w0, w1, w2, w3; };
__device__ __forceinline__ unsigned long long word(const U256 &x, int i) {
    return i == 3 ? x.w3 : i == 2 ? x.w2 : i == 1 ? x.w1 : i == 0 ? x.w0 : 0ull;
}
__device__ __forceinline__ int clz256(const U256 &x) {
    return x.w3 ? __clzll(x.w3) : x.w2 ? 64 + __clzll(x.w2) : x.w1 ? 128 + __clzll(x.w1) : x.w0 ? 192 + __clzll(x.w0) : 256;
}
__device__ __forceinline__ unsigned long long top64(const U256 &x, int lz) {   // bits [255 - lz - 63, 255 - lz], zeros below 0
    int ws = lz >> 6, bs = lz & 63;
    unsigned long long hi = word(x, 3 - ws), lo = word(x, 2 - ws);
    return bs ? hi << bs | lo >> (64 - bs) : hi;
}
__device__ __forceinline__ void sub_at(U256 &x, unsigned long long d, int sh) {   // x -= d * 2^sh, sh >= -63, no underflow
    U256 y{0, 0, 0, 0};
    if (sh < 0) y.w0 = d >> -sh;
    else {
        int ws = sh >> 6, bs = sh & 63;
        unsigned long long lo = d << bs, hi = bs ? d >> (64 - bs) : 0;
        if (ws == 0) { y.w0 = lo; y.w1 = hi; } else if (ws == 1) { y.w1 = lo; y.w2 = hi; } else if (ws == 2) { y.w2 = lo; y.w3 = hi; } else y.w3 = lo;
    }
    unsigned long long b = x.w0 < y.w0; x.w0 -= y.w0;
    unsigned long long t = x.w1 - y.w1 - b; b = (x.w1 < y.w1) || (x.w1 - y.w1 < b); x.w1 = t;
    t = x.w2 - y.w2 - b; b = (x.w2 < y.w2) || (x.w2 - y.w2 < b); x.w2 = t;
    x.w3 = x.w3 - y.w3 - b;
}

template <int NM>
__global__ void crt_td(const int32_t *Cl, size_t plane, int M, int N, int ldc, const int *sa, const int *sb,
                       double *Ch, double *Cm, double *Cl_, int ldo) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= N) return;
    unsigned long long acc[TDL] = {}; long long qf = 0;
    each<0, NM>([&](auto L) {
        constexpr int l = decltype(L)::value, m = modulus(l);
        int r = Cl[l * plane + (size_t)i * ldc + j] % m;
        if (r < 0) r += m;
        qf += (long long)r * c_F3[l];
#pragma unroll
        for (int k = 0; k < TDL; k++) acc[k] += (unsigned long long)r * c_W3[l][k];
    });
    long long q = (qf + (1ll << 39)) >> 40, c = 0;
    unsigned v[TDL];
#pragma unroll
    for (int k = 0; k < TDL; k++) { c += (long long)acc[k] - q * (long long)c_M3[k]; v[k] = (unsigned)c; c >>= 32; }
    const bool neg = (int)v[TDL - 1] < 0;
    if (neg) {
        unsigned long long b = 1;
#pragma unroll
        for (int k = 0; k < TDL; k++) { b += (unsigned)~v[k]; v[k] = (unsigned)b; b >>= 32; }
    }
    int top = -1;
#pragma unroll
    for (int k = 0; k < TDL; k++) if (v[k]) top = k;
    size_t o = (size_t)i * ldo + j;
    if (top < 0) { Ch[o] = 0; Cm[o] = 0; Cl_[o] = 0; return; }
    unsigned w[8] = {};   // limbs top..top-7 as a 256-bit frame whose lowest bit is 2^(32 (top - 7))
#pragma unroll
    for (int k = 0; k < TDL; k++)
        if (k == top) {
#pragma unroll
            for (int t = 0; t < 8; t++) w[7 - t] = k - t >= 0 ? v[k - t >= 0 ? k - t : 0] : 0;
        }
    U256 x{(unsigned long long)w[1] << 32 | w[0], (unsigned long long)w[3] << 32 | w[2],
           (unsigned long long)w[5] << 32 | w[4], (unsigned long long)w[7] << 32 | w[6]};
    const int e0 = 32 * (top - 7) - (sa[i] + sb[j]);
    double out[3];
#pragma unroll
    for (int p = 0; p < 3; p++) {   // peel: rd(V), rd(V - c0), rn(V - c0 - c1), each exact from the integer
        int lz = clz256(x);
        if (lz == 256) { out[p] = 0; continue; }
        unsigned long long T = top64(x, lz);
        double d = p < 2 ? __ull2double_rd(T) : __ull2double_rn(T);
        out[p] = ldexp(d, e0 + 192 - lz);
        if (p < 2) sub_at(x, (unsigned long long)d, 192 - lz);
    }
    dd a = quick_two_sum(out[0], out[1]), b = quick_two_sum(a.lo, out[2]);   // rd pieces -> canonical: |mid| <= ulp(hi)/2
    a = quick_two_sum(a.hi, b.hi);
    Ch[o] = neg ? -a.hi : a.hi; Cm[o] = neg ? -a.lo : a.lo; Cl_[o] = neg ? -b.lo : b.lo;
}

// A = Ah + Am + Al (M x K row-major), B likewise (K x N), C = Ch + Cm + Cl (M x N); p from plan(..., nm <= 49, 160)
template <int NM> void tdgemm_t(Plan &p, const double *Ah, const double *Am, const double *Al, const double *Bh, const double *Bm,
                                const double *Bl, double *Ch, double *Cm, double *Cl) {
    crt_consts_td(p.nm);
    scales_dd<<<p.M, 256>>>(Ah, p.K, 1, p.K, p.bits, p.sa);
    scales_dd<<<p.N, 256>>>(Bh, 1, p.N, p.K, p.bits, p.sb);
    residues_td<NM><<<dim3((p.K + 255) / 256, p.M), 256>>>(Ah, Am, Al, p.K, p.K, 1, p.sa, p.Ar, p.Kp, (size_t)p.Mp * p.Kp);
    residues_td<NM><<<dim3((p.K + 255) / 256, p.N), 256>>>(Bh, Bm, Bl, p.K, 1, p.N, p.sb, p.Br, p.Kp, (size_t)p.Np * p.Kp);
    gemm_i8<<<dim3(p.Np / BN, p.Mp / BM, NM), THREADS>>>(p.Ar, p.Br, p.Cl, p.Mp, p.Np, p.Kp);
    crt_td<NM><<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.M, p.N, p.Np, p.sa, p.sb, Ch, Cm, Cl, p.N);
}

}  // namespace oz
