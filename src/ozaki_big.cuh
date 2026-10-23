// ozaki_big.cuh -- exact big-integer GEMM C = A B (entries of thousands of bits) on INT8 tensor cores, linear in the bits.
// The 49 moduli <= 256 cap Ozaki-II at ~160-bit inputs and slicing beyond that costs ~S^2 GEMMs. Here the moduli are
// primes < 65280 (thousands of them, ~70 000 bits of M): a balanced residue r (|r| < 32640) is split into two bytes,
// r = 256 h + l, and A B mod p is four INT8 GEMMs (hh, hl, lh, ll). The number of primes grows linearly with the bits,
// so the GEMM work does too, while a CPU multiplying the entries pays M(P) per product (Karatsuba / Toom).
// The two conversions are INT8 GEMMs as well (conversions as matrix products, as in Doliskani-Giorgi-Lebreton-Schost):
//   residues: digits x_j (balanced bytes) times c_(l,j) = 2^(8 j) mod p_l (split in two bytes) -> r_l
//   CRT:      t_l = r_l (M/p_l)^-1 mod p_l (two bytes) times the balanced bytes of M/p_l -> the digits of
//             x + q M, q = round(sum t_l / p_l) from doubles; the digits are carried and q M taken off per element.
// Entries are two's complement integers of NL 32-bit limbs, |x| < 2^(32 NL - 9). Output: NA limbs per element.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <gmp.h>
#include "gemm_i8.cuh"

namespace oz {

struct BigPlan {
    int M, N, K, NL, L, NA, Kr, Kc, D;   // Kr: 4 NL digits padded; Kc = 2L; D: output digits padded
    int Mp, Np, Kp;
    std::vector<int> p;
    int *dp, *dinv; double *drp;
    int8_t *Bres, *Wc;                   // residue weights [2L][Kr], CRT weights [D][Kc]
    unsigned char *dMb;                  // the bytes of M, D of them
    int8_t *Xd, *Ar, *Br, *T;            // digits, residue planes [2][L][Mp Kp] / [2][L][Np Kp], T [M N][Kc]
    int32_t *Cg;                         // scratch for every GEMM output
    size_t cg_elems;
    int LC;                              // primes per main-GEMM chunk
};

// 64 elements per block, 32 limbs at a time through shared memory: limbs read and digit rows written contiguously
__global__ void __launch_bounds__(64) big_digits(const unsigned *X, int NL, int Kr, int rows, int cols, bool trans, int8_t *Xd) {
    __shared__ unsigned sx[64][33];
    __shared__ __align__(16) int8_t sd[64][128];
    const size_t n = (size_t)rows * cols, e0 = (size_t)blockIdx.x * 64;
    const int t = threadIdx.x;
    int carry = 0;
    for (int k0 = 0; k0 < NL; k0 += 32) {
        const int kc = min(32, NL - k0);
        for (int x = t; x < 64 * 32; x += 64) {   // row x / 32, limb x % 32: a row's limbs are adjacent in X
            int r = x >> 5, k = x & 31; size_t e = e0 + r;
            if (e < n && k < kc) { size_t src = trans ? (e % cols) * rows + e / cols : e; sx[r][k] = X[src * NL + k0 + k]; }
        }
        __syncthreads();
        for (int k = 0; k < kc; k++) {
            unsigned w = sx[t][k];
            for (int b = 0; b < 4; b++) {
                int v = (int)((w >> (8 * b)) & 255) + carry;
                carry = v >= 128; if (carry) v -= 256;   // the final carry equals the sign bit for |x| < 2^(32 NL - 9)
                sd[t][4 * k + b] = (int8_t)v;
            }
        }
        __syncthreads();
        for (int x = t; x < 64 * 8; x += 64) {   // 64 rows x 8 int4 = the 128 digit bytes of this limb chunk
            int r = x >> 3, c = x & 7; size_t e = e0 + r;
            if (e < n && 16 * c < 4 * kc) *(int4 *)&Xd[e * Kr + 4 * k0 + 16 * c] = *(const int4 *)&sd[r][16 * c];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ int bal(long long c, int p) { int r = (int)(c % p); if (r < 0) r += p; return r > p / 2 ? r - p : r; }

// R [2L][ldr] (lo rows, then hi rows: the residue GEMM runs transposed so that one prime's row is contiguous)
// -> balanced residues split into byte planes, element r0 + e = (i, k) at i Kp + k
__global__ void big_res_out(const int32_t *R, size_t ldr, size_t r0, size_t rows, int L, const int *p, int cols, int Kp,
                            size_t plane, int8_t *P) {
    size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; int l = blockIdx.y;
    if (e >= rows) return;
    int r = bal((long long)R[(L + l) * ldr + e] * 256 + R[l * ldr + e], p[l]);
    int lo = ((r + 128) & 255) - 128, hi = (r - lo) >> 8;
    size_t g = r0 + e, pos = g / cols * Kp + g % cols;
    P[(size_t)l * plane + pos] = (int8_t)lo;
    P[((size_t)L + l) * plane + pos] = (int8_t)hi;
}

// the four products of a chunk of primes -> t_l = (A B mod p_l) (M/p_l)^-1 mod p_l, two bytes, into T [M N][Kc]
__global__ void big_t(const int32_t *C, int LC, int lc, size_t pc, int l0, int L, int M, int N, int Np, const int *p, const int *inv, int8_t *T, int Kc) {
    __shared__ int8_t sl[64][65], sh[64][65];   // 64 columns x up to 64 primes: read along j, written along l
    const int i = blockIdx.y, j0 = blockIdx.x * 64;
    const size_t s = (size_t)LC * pc;
    for (int x = threadIdx.x; x < 64 * lc; x += blockDim.x) {
        int jl = x & 63, lz = x >> 6, j = j0 + jl, l = l0 + lz;
        if (j >= N) continue;
        size_t o = lz * pc + (size_t)i * Np + j;
        long long hh = C[o], hl = C[s + o], lh = C[2 * s + o], ll = C[3 * s + o];
        int c = bal((hh << 16) + ((hl + lh) << 8) + ll, p[l]);
        int t = bal((long long)c * inv[l], p[l]);
        int lo = ((t + 128) & 255) - 128;
        sl[jl][lz] = (int8_t)lo; sh[jl][lz] = (int8_t)((t - lo) >> 8);
    }
    __syncthreads();
    for (int x = threadIdx.x; x < 64 * lc; x += blockDim.x) {
        int lz = x % lc, jl = x / lc, j = j0 + jl;
        if (j >= N) continue;
        size_t e = (size_t)i * N + j;
        T[e * Kc + l0 + lz] = sl[jl][lz]; T[e * Kc + L + l0 + lz] = sh[jl][lz];
    }
}

// G [rows][D] digits of sum t_l M/p_l -> x = that - q M, NA limbs per element
__global__ void big_out(const int32_t *G, const int8_t *T, size_t e0, size_t rows, int L, int Kc, int D, const double *rp,
                        const unsigned char *Mb, int NA, unsigned *X) {
    size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (e >= rows) return;
    const int8_t *t = T + (e0 + e) * Kc;
    double qf = 0;
    for (int l = 0; l < L; l++) qf += (256 * t[L + l] + t[l]) * rp[l];
    long long q = llrint(qf), c = 0;
    const int32_t *g = G + e * D;
    unsigned w = 0;
    for (int d = 0; d < 4 * NA; d++) {
        c += (long long)g[d] - q * Mb[d];
        w |= (unsigned)(c & 255) << (8 * (d & 3));
        c >>= 8;
        if ((d & 3) == 3) { X[(e0 + e) * NA + d / 4] = w; w = 0; }
    }
}

inline BigPlan big_plan(int M, int N, int K, int NL) {
    BigPlan b{}; b.M = M; b.N = N; b.K = K; b.NL = NL;
    b.Mp = (M + BM - 1) / BM * BM; b.Np = (N + BN - 1) / BN * BN; b.Kp = (K + BK - 1) / BK * BK;
    // primes from 65279 down until M > 2^(2 (32 NL - 9) + log2 K + 8); |hi| <= 128 needs p/2 + 128 < 32768
    double need = 2.0 * (32 * NL - 9) + log2((double)K) + 8, have = 0;
    std::vector<char> comp(65280, 0);
    for (int i = 2; i * i < 65280; i++) if (!comp[i]) for (int j = i * i; j < 65280; j += i) comp[j] = 1;
    for (int x = 65279; x > 255 && (have < need || b.p.size() % 64); x--) if (!comp[x]) { b.p.push_back(x); have += log2((double)x); }
    if (have < need) { fprintf(stderr, "big: %d-bit entries need more primes than there are below 65280\n", 32 * NL); abort(); }
    b.L = (int)b.p.size(); b.Kc = 2 * b.L; b.Kr = (4 * NL + BK - 1) / BK * BK;
    mpz_t Mz, Ml; mpz_inits(Mz, Ml, NULL); mpz_set_ui(Mz, 1);
    for (int x : b.p) mpz_mul_ui(Mz, Mz, x);
    b.NA = (int)((mpz_sizeinbase(Mz, 2) + 2 + 31) / 32);
    b.D = (4 * b.NA + 1 + BN - 1) / BN * BN;
    std::vector<int> inv(b.L); std::vector<double> rp(b.L);
    std::vector<int8_t> wc((size_t)b.D * b.Kc, 0), br((size_t)b.Kc * b.Kr, 0);
    std::vector<unsigned char> mb(b.D, 0);
    { size_t cnt; mpz_export(mb.data(), &cnt, -1, 1, 0, 0, Mz); }
    for (int l = 0; l < b.L; l++) {
        int p = b.p[l];
        mpz_divexact_ui(Ml, Mz, p);
        long long a = mpz_fdiv_ui(Ml, p), x = 1, m = p;   // (M/p)^-1 mod p
        { long long r0 = a, r1 = m, s0 = 1, s1 = 0; while (r1) { long long qq = r0 / r1, t = r0 - qq * r1; r0 = r1; r1 = t; t = s0 - qq * s1; s0 = s1; s1 = t; } x = (s0 % p + p) % p; }
        inv[l] = (int)x; rp[l] = 1.0 / p;
        std::vector<unsigned char> u(4 * b.NA + 8, 0); size_t cnt; mpz_export(u.data(), &cnt, -1, 1, 0, 0, Ml);
        int carry = 0;
        for (int d = 0; d < 4 * b.NA; d++) {   // balanced bytes of M/p into column l (lo t-byte) and l + L shifted by one (hi)
            int v = u[d] + carry; carry = v >= 128; if (carry) v -= 256;
            wc[(size_t)d * b.Kc + l] = (int8_t)v;
            if (d + 1 < b.D) wc[(size_t)(d + 1) * b.Kc + b.L + l] = (int8_t)v;
        }
        long long c = 1;
        for (int j = 0; j < 4 * NL; j++) {   // 2^(8 j) mod p, balanced, two bytes
            int r = (int)c > p / 2 ? (int)c - p : (int)c, lo = ((r + 128) & 255) - 128, hi = (r - lo) >> 8;
            br[(size_t)l * b.Kr + j] = (int8_t)lo; br[(size_t)(b.L + l) * b.Kr + j] = (int8_t)hi;
            c = c * 256 % p;
        }
    }
    mpz_clears(Mz, Ml, NULL);
    cudaMalloc(&b.dp, b.L * 4); cudaMalloc(&b.dinv, b.L * 4); cudaMalloc(&b.drp, b.L * 8);
    cudaMemcpy(b.dp, b.p.data(), b.L * 4, cudaMemcpyHostToDevice); cudaMemcpy(b.dinv, inv.data(), b.L * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b.drp, rp.data(), b.L * 8, cudaMemcpyHostToDevice);
    cudaMalloc(&b.Wc, wc.size()); cudaMemcpy(b.Wc, wc.data(), wc.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&b.Bres, br.size()); cudaMemcpy(b.Bres, br.data(), br.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&b.dMb, b.D); cudaMemcpy(b.dMb, mb.data(), b.D, cudaMemcpyHostToDevice);
    size_t rowsX = std::max((size_t)M * K, (size_t)K * N), rowsXp = (rowsX + BM - 1) / BM * BM;
    cudaMalloc(&b.Xd, rowsXp * b.Kr); cudaMemset(b.Xd, 0, rowsXp * b.Kr);
    size_t pa = (size_t)b.Mp * b.Kp, pb = (size_t)b.Np * b.Kp;
    cudaMalloc(&b.Ar, 2 * b.L * pa); cudaMemset(b.Ar, 0, 2 * b.L * pa);
    cudaMalloc(&b.Br, 2 * b.L * pb); cudaMemset(b.Br, 0, 2 * b.L * pb);
    size_t MNp = ((size_t)M * N + BM - 1) / BM * BM;
    cudaMalloc(&b.T, MNp * b.Kc); cudaMemset(b.T, 0, MNp * b.Kc);
    b.cg_elems = (size_t)640 << 20 >> 2;   // 640 MB of int32 scratch
    b.LC = std::max(1, std::min(std::min(b.L, 64), (int)(b.cg_elems / (4 * (size_t)b.Mp * b.Np))));
    cudaMalloc(&b.Cg, b.cg_elems * 4);
    return b;
}

inline void big_release(BigPlan &b) {
    for (void *q : {(void *)b.dp, (void *)b.dinv, (void *)b.drp, (void *)b.Wc, (void *)b.Bres, (void *)b.dMb, (void *)b.Xd,
                    (void *)b.Ar, (void *)b.Br, (void *)b.T, (void *)b.Cg}) cudaFree(q);
}

// residues of a matrix of rows x cols big entries (row-major, NL limbs each) into byte planes
inline void big_residues(BigPlan &b, const unsigned *X, int rows, int cols, bool trans, int8_t *P, size_t plane) {
    size_t n = (size_t)rows * cols;
    big_digits<<<(n + 63) / 64, 64>>>(X, b.NL, b.Kr, rows, cols, trans, b.Xd);
    size_t rc = std::max<size_t>(BN, b.cg_elems / (2 * (size_t)b.L) / BN * BN);
    for (size_t r0 = 0; r0 < n; r0 += rc) {
        size_t m = std::min(rc, n - r0), mp = (m + BN - 1) / BN * BN;
        gemm_i8<<<dim3(mp / BN, b.Kc / BM, 1), THREADS>>>(b.Bres, b.Xd + r0 * b.Kr, b.Cg, b.Kc, (int)mp, b.Kr);
        big_res_out<<<dim3((m + 255) / 256, b.L), 256>>>(b.Cg, mp, r0, m, b.L, b.dp, cols, b.Kp, plane, P);
    }
}

// C = A B exactly: A M x K, B K x N, NL limbs per entry in; NA limbs per entry out (two's complement)
inline void biggemm(BigPlan &b, const unsigned *A, const unsigned *B, unsigned *C) {
    static const bool prof = getenv("OZ_PROF");
    cudaEvent_t ev[5]; if (prof) for (auto &x : ev) { cudaEventCreate(&x); } if (prof) cudaEventRecord(ev[0]);
    size_t pa = (size_t)b.Mp * b.Kp, pb = (size_t)b.Np * b.Kp, pc = (size_t)b.Mp * b.Np;
    big_residues(b, A, b.M, b.K, false, b.Ar, pa);
    big_residues(b, B, b.N, b.K, true, b.Br, pb);   // B^T: N x K
    if (prof) cudaEventRecord(ev[1]);
    for (int l0 = 0; l0 < b.L; l0 += b.LC) {
        int lc = std::min(b.LC, b.L - l0);
        const int prs[4][2] = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};   // hh, hl, lh, ll
        for (int q = 0; q < 4; q++)
            gemm_i8<<<dim3(b.Np / BN, b.Mp / BM, lc), THREADS>>>(b.Ar + ((size_t)prs[q][0] * b.L + l0) * pa,
                                                                 b.Br + ((size_t)prs[q][1] * b.L + l0) * pb, b.Cg + (size_t)q * b.LC * pc, b.Mp, b.Np, b.Kp);
        big_t<<<dim3((b.N + 63) / 64, b.M), 256>>>(b.Cg, b.LC, lc, pc, l0, b.L, b.M, b.N, b.Np, b.dp, b.dinv, b.T, b.Kc);
    }
    if (prof) cudaEventRecord(ev[2]);
    size_t n = (size_t)b.M * b.N, rc = std::max<size_t>(BM, b.cg_elems / b.D / BM * BM);
    for (size_t e0 = 0; e0 < n; e0 += rc) {
        size_t m = std::min(rc, n - e0), mp = (m + BM - 1) / BM * BM;
        gemm_i8<<<dim3(b.D / BN, mp / BM, 1), THREADS>>>(b.T + e0 * b.Kc, b.Wc, b.Cg, (int)mp, b.D, b.Kc);
        big_out<<<(m + 255) / 256, 256>>>(b.Cg, b.T, e0, m, b.L, b.Kc, b.D, b.drp, b.dMb, b.NA, C);
    }
    if (prof) {
        cudaEventRecord(ev[3]); cudaEventSynchronize(ev[3]); float t[3];
        for (int i = 0; i < 3; i++) cudaEventElapsedTime(&t[i], ev[i], ev[i + 1]);
        fprintf(stderr, "  phases: residues %.1f ms, products + t %.1f ms, CRT %.1f ms\n", t[0], t[1], t[2]);
    }
}

}  // namespace oz
