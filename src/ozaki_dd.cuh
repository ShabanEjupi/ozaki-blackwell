// ozaki_dd.cuh -- double-double GEMM (~106-bit significand) from INT8 tensor cores: a precision no GPU has in hardware.
// The same Ozaki-II pipeline as ozaki2.cuh with up to 32 moduli (product ~2^240), inputs and output as (hi, lo) pairs.
//   residues: x * 2^s split into integer and fraction parts of hi and lo; fmod is exact in IEEE, so
//             round(x * 2^s) mod m = fmod(int(hi)) + fmod(int(lo)) + rint(frac(hi) + frac(lo))  (mod m), no int128;
//   CRT:      Garner digits as before, Horner in double-double arithmetic.
#pragma once
#include "ozaki2.cuh"

namespace oz {

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

template <int NM>
__global__ void residues_dd(const double *Xh, const double *Xl, int len, long long sRow, long long sCol, const int *s,
                            int8_t *R, int ld, size_t plane) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (k >= len) return;
    size_t at = r * sRow + k * sCol;
    double a = ldexp(Xh[at], s[r]), b = ldexp(Xl[at], s[r]);
    double ai = trunc(a), bi = trunc(b);
    double f = rint((a - ai) + (b - bi));   // |f| <= 2
    each<0, NM>([&](auto L) {
        constexpr int m = modulus(decltype(L)::value);
        int v = (int)fmod(ai, (double)m) + (int)fmod(bi, (double)m) + (int)f;   // each |.| < m
        R[decltype(L)::value * plane + (size_t)r * ld + k] = (int8_t)balc<m>(v);
    });
}

template <int NM>
__global__ void crt_dd(const int32_t *Cl, size_t plane, int M, int N, int ldc, const int *sa, const int *sb, double *Ch, double *Cl_) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= N) return;
    int x[NM];
    each<0, NM>([&](auto L) {
        constexpr int l = decltype(L)::value, m = modulus(l);
        int v = balc<m>(Cl[l * plane + (size_t)i * ldc + j]);
        each<0, l>([&](auto T) { v = balc<m>((v - x[decltype(T)::value]) * c_inv[decltype(T)::value][l]); });
        x[l] = v;
    });
    dd y{(double)x[NM - 1], 0};
#pragma unroll
    for (int l = NM - 2; l >= 0; l--) y = dd_mul_add(y, (double)modulus(l), (double)x[l]);
    int e = -(sa[i] + sb[j]);
    Ch[(size_t)i * N + j] = ldexp(y.hi, e); Cl_[(size_t)i * N + j] = ldexp(y.lo, e);
}

// A = Ah + Al (M x K row-major), B = Bh + Bl (K x N row-major), C = Ch + Cl (M x N); p from plan(..., nm, 106)
template <int NM> void ddgemm_t(Plan &p, const double *Ah, const double *Al, const double *Bh, const double *Bl, double *Ch, double *Cl) {
    scales_dd<<<p.M, 256>>>(Ah, p.K, 1, p.K, p.bits, p.sa);
    scales_dd<<<p.N, 256>>>(Bh, 1, p.N, p.K, p.bits, p.sb);
    residues_dd<NM><<<dim3((p.K + 255) / 256, p.M), 256>>>(Ah, Al, p.K, p.K, 1, p.sa, p.Ar, p.Kp, (size_t)p.Mp * p.Kp);
    residues_dd<NM><<<dim3((p.K + 255) / 256, p.N), 256>>>(Bh, Bl, p.K, 1, p.N, p.sb, p.Br, p.Kp, (size_t)p.Np * p.Kp);
    for (int l = 0; l < NM; l++)
        gemm_i8<<<dim3(p.Np / BN, p.Mp / BM), THREADS>>>(p.Ar + (size_t)l * p.Mp * p.Kp, p.Br + (size_t)l * p.Np * p.Kp,
                                                         p.Cl + (size_t)l * p.Mp * p.Np, p.Mp, p.Np, p.Kp);
    crt_dd<NM><<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.M, p.N, p.Np, p.sa, p.sb, Ch, Cl);
}

}  // namespace oz
