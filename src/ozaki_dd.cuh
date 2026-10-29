// ozaki_dd.cuh -- double-double GEMM (~106-bit significand) from INT8 tensor cores: a precision no GPU has in hardware.
// The same Ozaki-II pipeline as ozaki2.cuh with up to 32 moduli (product ~2^240), inputs and output as (hi, lo) pairs.
//   residues: x * 2^s split into integer and fraction parts of hi and lo; fmod is exact in IEEE, so
//             round(x * 2^s) mod m = fmod(int(hi)) + fmod(int(lo)) + rint(frac(hi) + frac(lo))  (mod m), no int128;
//   CRT:      the direct sum over 8 32-bit limbs on the integer units (see crt_dd), then to double-double.
#pragma once
#include "ozaki2.cuh"

namespace oz {

__host__ __device__ constexpr int pow2mod(int e, int m) { int r = 1 % m; for (int i = 0; i < e; i++) r = r * 2 % m; return r; }

struct dd { double hi, lo; };
__device__ __forceinline__ dd two_sum(double a, double b) { double s = a + b, v = s - a; return {s, (a - (s - v)) + (b - v)}; }
__device__ __forceinline__ dd quick_two_sum(double a, double b) { double s = a + b; return {s, b - (s - a)}; }
__device__ __forceinline__ dd dd_mul_add(dd y, double m, double x) {   // y * m + x
    double p = y.hi * m, pe = fma(y.hi, m, -p);   // exact product of the high part
    dd s = two_sum(p, x);
    return quick_two_sum(s.hi, s.lo + pe + y.lo * m);
}

// one block per row (A) or column (B), from the hi parts
__global__ void scales_dd(const double *Xh, long long sRow, long long sCol, int len, int bits, int *s) {
    int r = blockIdx.x;
    double mx = 0;
    for (int k = threadIdx.x; k < len; k += blockDim.x) mx = fmax(mx, fabs(Xh[r * sRow + k * sCol]));
    for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
    __shared__ double w[32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) mx = fmax(mx, w[i]);
        s[r] = mx == 0 ? 0 : bits - 1 - ilogb(mx);
    }
}

// An int64 cut once into 16-bit chunks, so that its residue mod each m is a few 32-bit multiply-adds and one int32
// reduction; reducing int64 pieces per modulus made the td residues cost as much as the 49 INT8 GEMMs (22-10: 57 vs 56 ms).
struct C16 { int c3, c2, c1, c0; };   // S = c3 2^48 + c2 2^32 + c1 2^16 + c0, c3 signed, the rest in [0, 2^16); |S| < 2^56
__device__ __forceinline__ C16 cut16(long long S) { return {(int)(S >> 48), (int)((S >> 32) & 0xffff), (int)((S >> 16) & 0xffff), (int)(S & 0xffff)}; }
template <int m> __device__ __forceinline__ int res16(C16 c) {   // |sum| < 2^26
    constexpr int p16 = pow2mod(16, m), p32 = pow2mod(32, m), p48 = pow2mod(48, m);
    return balc<m>(c.c3 * p48 + c.c2 * p32 + c.c1 * p16 + c.c0);
}

template <int NM>
__global__ void residues_dd(const double *Xh, const double *Xl, int len, long long sRow, long long sCol, const int *s,
                            int8_t *R, int ld, size_t plane) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (k >= len) return;
    size_t at = r * sRow + k * sCol;
    double a = ldexp(Xh[at], s[r]), b = ldexp(Xl[at], s[r]);
    double ai = trunc(a), bi = trunc(b);
    double f = rint((a - ai) + (b - bi));   // |f| <= 2
    // ai = u * 2^53 + w exactly (u, w in int64), then everything mod m with compile-time m.
    // fmod on a 2^107 value loops bit by bit (15-10: residues were most of the 211 ms).
    long long u = (long long)trunc(ldexp(ai, -53)), w = (long long)(ai - ldexp((double)u, 53)), bb = (long long)bi;
    const C16 c1 = cut16(u), c0 = cut16(w + bb + (long long)f);
    each<0, NM>([&](auto L) {
        constexpr int m = modulus(decltype(L)::value), p53 = pow2mod(53, m);
        int v = res16<m>(c1) * p53 + res16<m>(c0);   // |.| < 128 * 256 + 128
        R[decltype(L)::value * plane + (size_t)r * ld + k] = (int8_t)balc<m>(v);
    });
}

// CRT by the direct sum, on the integer units. Garner digits + Horner in double-double was 41.5 of the
// 47 ms of a 4096^2 x 256 dd GEMM (22-10): 31 dd steps of FP64, which runs at 1/64 rate on this chip, plus 496 modular steps.
//   x = sum_l r_l W_l - q M,  W_l = (M/m_l) ((M/m_l)^-1 mod m_l),  r_l in [0, m_l),  q = round(sum_l r_l w_l / m_l)
// The sum is 8 limbs of 32 bits held in 64-bit accumulators (no carries until the end: 32 terms < 2^45); q comes from
// 40-bit fixed-point fractions w_l / m_l; then 5 limbs from the top, normalized, become a double-double.
__constant__ unsigned c_W[MAXM][8], c_M[8];
__constant__ long long c_F[MAXM];
inline void crt_consts(int nm) {
    static int done = -1;
    if (done == nm) return;
    done = nm;
    unsigned W[MAXM][8] = {}, M[8] = {}; long long F[MAXM] = {};
    auto mul_small = [](unsigned *a, unsigned m) {
        unsigned long long c = 0;
        for (int k = 0; k < 8; k++) { c += (unsigned long long)a[k] * m; a[k] = (unsigned)c; c >>= 32; }
    };
    M[0] = 1; for (int l = 0; l < nm; l++) mul_small(M, MODULI[l]);
    for (int l = 0; l < nm; l++) {
        int r = 1; W[l][0] = 1;
        for (int t = 0; t < nm; t++) if (t != l) { mul_small(W[l], MODULI[t]); r = r * (MODULI[t] % MODULI[l]) % MODULI[l]; }
        int w = inv_mod(r, MODULI[l]);
        mul_small(W[l], w);
        F[l] = llround(ldexp((double)w / MODULI[l], 40));
    }
    cudaMemcpyToSymbol(c_W, W, sizeof W); cudaMemcpyToSymbol(c_M, M, sizeof M); cudaMemcpyToSymbol(c_F, F, sizeof F);
}

template <int NM>
__global__ void crt_dd(const int32_t *Cl, size_t plane, int M, int N, int ldc, const int *sa, const int *sb, double *Ch, double *Cl_, int ldo,
                       int mode) {   // mode 0: C = AB, 1: C -= AB, 2: C = -AB
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= N) return;
    unsigned long long acc[8] = {}; long long qf = 0;
    each<0, NM>([&](auto L) {
        constexpr int l = decltype(L)::value, m = modulus(l);
        int r = Cl[l * plane + (size_t)i * ldc + j] % m;
        if (r < 0) r += m;
        qf += (long long)r * c_F[l];
#pragma unroll
        for (int k = 0; k < 8; k++) acc[k] += (unsigned long long)r * c_W[l][k];
    });
    long long q = (qf + (1ll << 39)) >> 40, c = 0;
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) { c += (long long)acc[k] - q * (long long)c_M[k]; v[k] = (unsigned)c; c >>= 32; }   // two's complement
    const bool neg = (int)v[7] < 0;
    if (neg) {
        unsigned long long b = 1;
#pragma unroll
        for (int k = 0; k < 8; k++) { b += (unsigned)~v[k]; v[k] = (unsigned)b; b >>= 32; }
    }
    int top = -1;
#pragma unroll
    for (int k = 0; k < 8; k++) if (v[k]) top = k;
    size_t o = (size_t)i * ldo + j;
    if (top < 0) { if (mode != 1) { Ch[o] = 0; Cl_[o] = 0; } return; }
    unsigned t4 = 0, t3 = 0, t2 = 0, t1 = 0, t0 = 0;   // limbs top..top-4, by compile-time index only
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (k == top) { t4 = v[k]; t3 = k >= 1 ? v[k - 1] : 0; t2 = k >= 2 ? v[k - 2] : 0; t1 = k >= 3 ? v[k - 3] : 0; t0 = k >= 4 ? v[k - 4] : 0; }
    int z = __clz(t4);
    unsigned long long H = (unsigned long long)t4 << 32 | t3, Lw = (unsigned long long)t2 << 32 | t1;
    if (z) { H = H << z | Lw >> (64 - z); Lw = Lw << z | t0 >> (32 - z); }
    double dh = __ull2double_rd(H);                    // 53 bits of H, rounded down: the rest is < 2^11 and exact
    double lo = fma((double)(H - (unsigned long long)dh), 18446744073709551616.0, (double)Lw);
    int e = 32 * (top - 3) - z - (sa[i] + sb[j]);
    dd y = quick_two_sum(ldexp(dh, e + 64), ldexp(lo, e));
    if (neg != (mode != 0)) y = {-y.hi, -y.lo};
    if (mode == 1) {   // C + (-AB), accurate double-double addition
        dd s = two_sum(Ch[o], y.hi), t = two_sum(Cl_[o], y.lo);
        s = quick_two_sum(s.hi, s.lo + t.hi);
        y = quick_two_sum(s.hi, s.lo + t.lo);
    }
    Ch[o] = y.hi; Cl_[o] = y.lo;
}


// A = Ah + Al (M x K row-major), B = Bh + Bl (K x N row-major), C = Ch + Cl (M x N); p from plan(..., nm, 106).
// Leading dimensions lda, ldb, ldo (0: K, N, N) let A, B and C be blocks of larger matrices; mode as in crt_dd.
template <int NM> void ddgemm_t(Plan &p, const double *Ah, const double *Al, const double *Bh, const double *Bl, double *Ch, double *Cl,
                                int ldo = 0, int lda = 0, int ldb = 0, int mode = 0) {
    crt_consts(p.nm);
    lda = lda ? lda : p.K; ldb = ldb ? ldb : p.N;
    scales_dd<<<p.M, 256>>>(Ah, lda, 1, p.K, p.bits, p.sa);
    scales_dd<<<p.N, 256>>>(Bh, 1, ldb, p.K, p.bits, p.sb);
    residues_dd<NM><<<dim3((p.K + 255) / 256, p.M), 256>>>(Ah, Al, p.K, lda, 1, p.sa, p.Ar, p.Kp, (size_t)p.Mp * p.Kp);
    residues_dd<NM><<<dim3((p.K + 255) / 256, p.N), 256>>>(Bh, Bl, p.K, 1, ldb, p.sb, p.Br, p.Kp, (size_t)p.Np * p.Kp);
    gemm_i8_run(p.Ar, p.Br, p.Cl, p.Mp, p.Np, p.Kp, NM);   // all moduli in one launch
    crt_dd<NM><<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.M, p.N, p.Np, p.sa, p.sb, Ch, Cl, ldo ? ldo : p.N, mode);
}

}  // namespace oz
