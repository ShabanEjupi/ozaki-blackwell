// ozaki_mp.cuh -- arbitrary-precision GEMM (Q doubles per number, ~53 Q bits) from INT8 tensor cores.
// The 49 pairwise-coprime moduli <= 256 are every one there is (every prime < 256 is used, alone or inside 255, 253, 247,
// 217, 256): their product, ~2^342, caps one Ozaki-II GEMM at ~160-bit inputs -- triple-double. Beyond that the inputs
// are sliced: x 2^s = sum_i X_i 2^(160 i), balanced 160-bit slices, and C = sum_(i,j) (X_i Y_j) 2^(160 (i+j)), each slice
// product one exact 49-modulus Ozaki-II GEMM. Pairs too small to reach the Q doubles of the result are skipped (for Q = 4:
// 3 of 4). The slice products are added exactly into a wide integer per element, peeled into Q doubles at the end.
#pragma once
#include "ozaki_td.cuh"

namespace oz {

constexpr int MPW = 160;   // bits per slice: |X_i Y_j| K < 2^318 2^12 < M/2 ~ 2^341

template <int Q> __host__ __device__ constexpr int mp_slices() { return (53 * Q + 2 + MPW - 1) / MPW; }
template <int Q> __host__ __device__ constexpr bool mp_keep(int i, int j) { return MPW * (2 * (mp_slices<Q>() - 1) - i - j) <= 53 * Q + 16; }
template <int Q> struct MpPtr { double *p[Q]; };

// v += t 2^(32 li) for a signed t (|t| < 2^120), v two's complement over NL limbs
template <int NL> __device__ __forceinline__ void add_at(unsigned *v, __int128 t, int li) {
    long long c = 0;
    for (int k = li; k < NL; k++) {
        c += (long long)v[k] + (long long)(unsigned)(t & 0xffffffffu);
        v[k] = (unsigned)c; c >>= 32; t >>= 32;
    }
}

// residue planes R[(slice NM + l) plane + r ld + k] of the slices of round(x 2^s[r]), x = sum of the Q component planes
template <int Q, int NM>
__global__ void residues_mp(MpPtr<Q> X, int len, long long sRow, long long sCol, const int *s, int8_t *R, int ld, size_t plane) {
    constexpr int S = mp_slices<Q>(), NL = 5 * S;
    int k = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (k >= len) return;
    size_t at = r * sRow + k * sCol;
    unsigned v[NL] = {};
    double fr = 0;
#pragma unroll
    for (int q = 0; q < Q; q++) {   // each part's integer part added exactly at its own offset, the fractions summed
        double a = ldexp(X.p[q][at], s[r]), ai = trunc(a);
        fr += a - ai;
        if (ai == 0) continue;
        int e = ilogb(ai) - 52;
        if (e <= 0) add_at<NL>(v, (__int128)(long long)ai, 0);
        else add_at<NL>(v, (__int128)(long long)ldexp(ai, -e) << (e & 31), e >> 5);
    }
    add_at<NL>(v, (__int128)(long long)rint(fr), 0);
    // balanced slices: X_i = u_i - b_i 2^160 + c_i; chunks of 16 bits of u_i
    int ch[S][10], bo[S], ci[S];
    int carry = 0;
#pragma unroll
    for (int i = 0; i < S; i++) {
#pragma unroll
        for (int j = 0; j < 5; j++) { unsigned w = v[5 * i + j]; ch[i][2 * j] = w & 0xffff; ch[i][2 * j + 1] = w >> 16; }
        ci[i] = carry;
        bo[i] = (v[5 * i + 4] >> 31) & 1;   // top bit: for i < S-1 it is the balance borrow, for the top slice the sign
        carry = bo[i];
    }
    each<0, NM>([&](auto L) {
        constexpr int l = decltype(L)::value, m = modulus(l), p160 = pow2mod(160, m);
        each<0, S>([&](auto I) {   // compile-time slice and chunk indices: registers, and constant 2^(16 j) mod m
            constexpr int i = decltype(I)::value;
            int a = 0;
            each<0, 10>([&](auto J) { constexpr int j = decltype(J)::value, pj = pow2mod(16 * j, m); a += ch[i][j] * pj; });   // < 2^28
            R[((size_t)i * NM + l) * plane + (size_t)r * ld + k] = (int8_t)balc<m>(balc<m>(a) - bo[i] * p160 + ci[i]);
        });
    });
}

// the slice product of one pair, exactly, added into ACC (NA limbs, limb-major) at limb offset li
template <int NM, int NA>
__global__ void crt_mp_acc(const int32_t *Cl, size_t plane, int M, int N, int ldc, unsigned *ACC, int li) {
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
    long long q = (qf + (1ll << 39)) >> 40, c = 0, d = 0;
    size_t o = (size_t)i * N + j, MN = (size_t)M * N;
#pragma unroll
    for (int k = 0; k < TDL; k++) {   // V = S - q M, added limb by limb, sign-extended up to NA
        c += (long long)acc[k] - q * (long long)c_M3[k];
        d += (long long)ACC[(li + k) * MN + o] + (long long)(unsigned)c;
        ACC[(li + k) * MN + o] = (unsigned)d; d >>= 32; c >>= 32;
    }
    for (int k = li + TDL; k < NA; k++) {   // c is 0 or -1 here
        d += (long long)ACC[k * MN + o] + (long long)(unsigned)c;
        ACC[k * MN + o] = (unsigned)d; d >>= 32;
    }
}

// ACC (two's complement, NA limbs) * 2^-(sa + sb) -> Q doubles, each peeled exactly off the integer
template <int Q, int NA>
__global__ void mp_out(const unsigned *ACC, int M, int N, const int *sa, const int *sb, MpPtr<Q> C, int ldo) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= N) return;
    size_t o = (size_t)i * N + j, MN = (size_t)M * N;
    unsigned v[NA];
    for (int k = 0; k < NA; k++) v[k] = ACC[k * MN + o];
    const bool neg = (int)v[NA - 1] < 0;
    if (neg) { unsigned long long b = 1; for (int k = 0; k < NA; k++) { b += (unsigned)~v[k]; v[k] = (unsigned)b; b >>= 32; } }
    const int e0 = -(sa[i] + sb[j]);
    double out[Q];
    for (int p = 0; p < Q; p++) {
        int t = NA - 1;
        while (t >= 0 && !v[t]) t--;
        if (t < 0) { out[p] = 0; continue; }
        unsigned w2 = v[t], w1 = t >= 1 ? v[t - 1] : 0, w0 = t >= 2 ? v[t - 2] : 0;
        int lz = __clz(w2);
        unsigned long long T = ((unsigned long long)w2 << 32 | w1) << lz | (lz ? (unsigned long long)w0 >> (32 - lz) : 0);
        double d = p < Q - 1 ? __ull2double_rd(T) : __ull2double_rn(T);
        int pos = 32 * (t - 1) - lz;   // T's lowest bit
        out[p] = ldexp(d, pos + e0);
        if (p == Q - 1) break;
        unsigned long long D = (unsigned long long)d;   // v -= D 2^pos (D <= T: no borrow out of the top)
        __int128 sub = pos >= 0 ? (__int128)D << (pos & 31) : (__int128)(D >> -pos);
        int li = pos >= 0 ? pos >> 5 : 0;
        long long c = 0;
        for (int k = li; k < NA && (sub != 0 || c != 0); k++) {
            c += (long long)v[k] - (long long)(unsigned)(sub & 0xffffffffu);
            v[k] = (unsigned)c; c >>= 32; sub >>= 32;
        }
    }
    for (int p = 0; p < Q; p++) C.p[p][(size_t)i * ldo + j] = neg ? -out[p] : out[p];
}

// C = A B with Q doubles per number. p from plan(M, N, K, 49, 160) with p.Ar, p.Br holding S planes sets (mp_plan).
template <int Q> struct MpPlan { Plan p; unsigned *acc; };
template <int Q> MpPlan<Q> mp_plan(int M, int N, int K) {
    constexpr int S = mp_slices<Q>(), NA = 5 * (2 * S - 2) + TDL + 1;
    MpPlan<Q> m; m.p = plan(M, N, K, 49, 160);
    cudaFree(m.p.Ar); cudaFree(m.p.Br);
    cudaMalloc(&m.p.Ar, (size_t)S * 49 * m.p.Mp * m.p.Kp); cudaMalloc(&m.p.Br, (size_t)S * 49 * m.p.Np * m.p.Kp);
    cudaMemset(m.p.Ar, 0, (size_t)S * 49 * m.p.Mp * m.p.Kp); cudaMemset(m.p.Br, 0, (size_t)S * 49 * m.p.Np * m.p.Kp);
    cudaMalloc(&m.acc, (size_t)NA * M * N * 4);
    return m;
}
template <int Q> void mp_release(MpPlan<Q> &m) { release(m.p); cudaFree(m.acc); }
template <int Q> int mp_pairs() { int c = 0; for (int i = 0; i < mp_slices<Q>(); i++) for (int j = 0; j < mp_slices<Q>(); j++) c += mp_keep<Q>(i, j); return c; }

template <int Q> void mpgemm(MpPlan<Q> &m, MpPtr<Q> A, MpPtr<Q> B, MpPtr<Q> C) {
    constexpr int S = mp_slices<Q>(), NA = 5 * (2 * S - 2) + TDL + 1;
    Plan &p = m.p;
    crt_consts_td(49);
    p.bits = MPW * S - 2;   // the integer of a row's largest element stays below 2^(160 S - 2)
    scales_dd<<<p.M, 256>>>(A.p[0], p.K, 1, p.K, p.bits, p.sa);
    scales_dd<<<p.N, 256>>>(B.p[0], 1, p.N, p.K, p.bits, p.sb);
    size_t pa = (size_t)p.Mp * p.Kp, pb = (size_t)p.Np * p.Kp;
    residues_mp<Q, 49><<<dim3((p.K + 255) / 256, p.M), 256>>>(A, p.K, p.K, 1, p.sa, p.Ar, p.Kp, pa);
    residues_mp<Q, 49><<<dim3((p.K + 255) / 256, p.N), 256>>>(B, p.K, 1, p.N, p.sb, p.Br, p.Kp, pb);
    cudaMemsetAsync(m.acc, 0, (size_t)NA * p.M * p.N * 4);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            if (!mp_keep<Q>(i, j)) continue;
            gemm_i8<<<dim3(p.Np / BN, p.Mp / BM, 49), THREADS>>>(p.Ar + i * 49 * pa, p.Br + j * 49 * pb, p.Cl, p.Mp, p.Np, p.Kp);
            crt_mp_acc<49, NA><<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.M, p.N, p.Np, m.acc, 5 * (i + j));
        }
    mp_out<Q, NA><<<dim3((p.N + 255) / 256, p.M), 256>>>(m.acc, p.M, p.N, p.sa, p.sb, C, p.N);
}

}  // namespace oz
