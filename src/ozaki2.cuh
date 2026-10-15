// ozaki2.cuh -- C = A * B in FP64 from INT8 tensor cores (Ozaki scheme II: Uchino, Ozaki, Imamura 2025).
//   1. scale: row i of A by 2^sa[i], column j of B by 2^sb[j], rounded to integers below 2^bits;
//   2. residues: every integer mod each of nm pairwise coprime moduli <= 256, balanced, so they fit int8;
//   3. one exact INT8 GEMM per modulus (int32 never overflows while K < 2^31 / 128^2 = 131072);
//   4. Chinese remainder theorem (Garner, balanced mixed radix), Horner in FP64, undo the scales.
// The integer product is exact while sum_k |A'||B'| < P/2, P = product of the moduli: bits = (log2 P - 2 - log2 K) / 2.
#pragma once
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include "gemm_i8.cuh"

namespace oz {

constexpr int MAXM = 16;
// pairwise coprime, largest first: 2^8, 3*5*17, 11*23, 251, 13*19, then primes, 7*31
constexpr int MODULI[MAXM] = {256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193};
__constant__ int c_mod[MAXM];
__constant__ int c_inv[MAXM][MAXM];   // c_inv[j][i] = m_j^-1 mod m_i (j < i)

__device__ __forceinline__ int bal(long long x, int m) {   // x mod m into [-m/2, m/2)
    int r = (int)(x % m);
    if (r < 0) r += m;
    return r >= (m + 1) / 2 ? r - m : r;
}

// one block per row (A) or column (B): the exponent that brings the largest |x| just under 2^bits
__global__ void scales(const double *X, int rows, int len, long long sRow, long long sCol, int bits, int *s) {
    int r = blockIdx.x;
    double mx = 0;
    for (int k = threadIdx.x; k < len; k += blockDim.x) mx = fmax(mx, fabs(X[r * sRow + k * sCol]));
    for (int o = 16; o; o >>= 1) mx = fmax(mx, __shfl_xor_sync(~0u, mx, o));
    __shared__ double w[32];
    if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < (int)(blockDim.x >> 5); i++) mx = fmax(mx, w[i]);
        s[r] = mx == 0 ? 0 : bits - 1 - ilogb(mx);   // mx * 2^s < 2^bits
    }
}

// residue planes R[l][r][k] (K contiguous, padded to ld) of round(X[r,k] * 2^s[r]); X read with strides
__global__ void residues(const double *X, int rows, int len, long long sRow, long long sCol, const int *s, int nm,
                         int8_t *R, int ld, size_t plane) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (k >= len) return;
    long long v = llrint(ldexp(X[r * sRow + k * sCol], s[r]));
    for (int l = 0; l < nm; l++) R[l * plane + (size_t)r * ld + k] = (int8_t)bal(v, c_mod[l]);
}

// Garner: balanced mixed-radix digits, then Horner from the top digit in FP64, then undo the two scales
__global__ void crt(const int32_t *Cl, size_t plane, int nm, int M, int N, int ldc, const int *sa, const int *sb, double *C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= N) return;
    int x[MAXM];
    for (int l = 0; l < nm; l++) {
        int v = bal(Cl[l * plane + (size_t)i * ldc + j], c_mod[l]);
        for (int t = 0; t < l; t++) v = bal((long long)(v - x[t]) * c_inv[t][l], c_mod[l]);
        x[l] = v;
    }
    double y = x[nm - 1];
    for (int l = nm - 2; l >= 0; l--) y = fma(y, (double)c_mod[l], (double)x[l]);
    C[(size_t)i * N + j] = ldexp(y, -(sa[i] + sb[j]));
}

static int inv_mod(int a, int m) { for (int x = 1; x < m; x++) if (a * x % m == 1) return x; return 0; }

struct Plan {
    int M, N, K, Mp, Np, Kp, nm, bits;
    int8_t *Ar, *Br; int32_t *Cl; int *sa, *sb;
};

// A: M x K row-major, B: K x N row-major, C: M x N row-major, all device FP64
inline Plan plan(int M, int N, int K, int nm) {
    Plan p{M, N, K, (M + BM - 1) / BM * BM, (N + BN - 1) / BN * BN, (K + BK - 1) / BK * BK, nm, 0};
    double log2P = 0;
    for (int l = 0; l < nm; l++) log2P += log2((double)MODULI[l]);
    p.bits = (int)floor((log2P - 2 - log2((double)K)) / 2);
    if (p.bits > 62) p.bits = 62;   // llrint into int64
    int mods[MAXM], inv[MAXM][MAXM] = {};
    for (int l = 0; l < MAXM; l++) mods[l] = MODULI[l];
    for (int t = 0; t < MAXM; t++) for (int l = t + 1; l < MAXM; l++) inv[t][l] = inv_mod(MODULI[t] % MODULI[l], MODULI[l]);
    cudaMemcpyToSymbol(c_mod, mods, sizeof mods); cudaMemcpyToSymbol(c_inv, inv, sizeof inv);
    cudaMalloc(&p.Ar, (size_t)nm * p.Mp * p.Kp); cudaMalloc(&p.Br, (size_t)nm * p.Np * p.Kp);
    cudaMalloc(&p.Cl, (size_t)nm * p.Mp * p.Np * 4);
    cudaMemset(p.Ar, 0, (size_t)nm * p.Mp * p.Kp); cudaMemset(p.Br, 0, (size_t)nm * p.Np * p.Kp);   // padding stays 0
    cudaMalloc(&p.sa, p.M * 4); cudaMalloc(&p.sb, p.N * 4);
    return p;
}
inline void release(Plan &p) { cudaFree(p.Ar); cudaFree(p.Br); cudaFree(p.Cl); cudaFree(p.sa); cudaFree(p.sb); }

inline void dgemm(Plan &p, const double *A, const double *B, double *C) {
    scales<<<p.M, 256>>>(A, p.M, p.K, p.K, 1, p.bits, p.sa);
    scales<<<p.N, 256>>>(B, p.N, p.K, 1, p.N, p.bits, p.sb);   // column j of B: element k at k*N + j
    residues<<<dim3((p.K + 255) / 256, p.M), 256>>>(A, p.M, p.K, p.K, 1, p.sa, p.nm, p.Ar, p.Kp, (size_t)p.Mp * p.Kp);
    residues<<<dim3((p.K + 255) / 256, p.N), 256>>>(B, p.N, p.K, 1, p.N, p.sb, p.nm, p.Br, p.Kp, (size_t)p.Np * p.Kp);
    for (int l = 0; l < p.nm; l++)
        gemm_i8<<<dim3(p.Np / BN, p.Mp / BM), THREADS>>>(p.Ar + (size_t)l * p.Mp * p.Kp, p.Br + (size_t)l * p.Np * p.Kp,
                                                         p.Cl + (size_t)l * p.Mp * p.Np, p.Mp, p.Np, p.Kp);
    crt<<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.nm, p.M, p.N, p.Np, p.sa, p.sb, C);
}

}  // namespace oz
