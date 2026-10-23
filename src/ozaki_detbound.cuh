// ozaki_detbound.cuh -- a proved upper bound on log2 |det A| that is tight to a few bits, so det / d needs a handful of
// primes instead of the ~0.72 n bits of slack Hadamard's bound leaves on a random matrix.
//
// FP32 LU with partial pivoting (cuSOLVER) gives PA ~ LU. Round X_L = 2^s L^-1 (unit diagonal -> exactly 2^s) and
// X_U = 2^s U^-1 diag(u) to integers: both triangular, so det X_L = 2^(s n) and det X_U = prod of its diagonal, exactly.
// B = X_L (PA) X_U is computed EXACTLY (modular INT8 GEMMs, CRT to 128 bits), then Hadamard on B:
//     |det A| = |det B| / (2^(s n) |det X_U|) <= prod_i ||b_i|| / (2^(s n) prod_j |X_U[j][j]|).
// Floating point only decides how close B is to 2^(2s) diag(u), i.e. how tight the bound is -- never whether it holds.
#pragma once
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "ozaki_det.cuh"

namespace oz {

__global__ void bound_build_x(const float *__restrict__ LU, const float *__restrict__ Li, const float *__restrict__ Ui, int n, int s,
                              long long *__restrict__ XL, long long *__restrict__ XU) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * n; e += (size_t)gridDim.x * blockDim.x) {
        const int i = e / n, k = e % n;   // row-major (i, k); the float matrices are column-major
        XL[e] = k < i ? llrint(ldexp((double)Li[i + (size_t)k * n], s)) : k == i ? 1ll << s : 0;
        XU[e] = i <= k ? llrint(ldexp((double)Ui[i + (size_t)k * n] * LU[k + (size_t)k * n], s)) : 0;
    }
}
// rows of X_L and columns of X_U: their L1 norms (rounded up) bound |B_ij| <= |X_L[i]|_1 127 |X_U[:,j]|_1
__global__ void bound_l1(const long long *__restrict__ XL, const long long *__restrict__ XU, int n, double *rl, double *cu,
                         long long *du) {
    const int lane = threadIdx.x & 31, w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w < n) {   // a warp per row of X_L (coalesced)
        double a = 0;
        for (int k = lane; k <= w; k += 32) a = __dadd_ru(a, __ll2double_ru(llabs(XL[(size_t)w * n + k])));
        for (int o = 16; o; o >>= 1) a = __dadd_ru(a, __shfl_xor_sync(0xffffffffu, a, o));
        if (lane == 0) rl[w] = a;
    }
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) {   // a thread per column of X_U (rows read across threads: coalesced)
        double b = 0;
        for (int k = 0; k <= t; k++) b = __dadd_ru(b, __ll2double_ru(llabs(XU[(size_t)k * n + t])));
        cu[t] = b; du[t] = XU[(size_t)t * n + t];
    }
}
__global__ void bound_prep(const int8_t *__restrict__ A, int n, float *__restrict__ LU, float *__restrict__ Li, float *__restrict__ Ui) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * n; e += (size_t)gridDim.x * blockDim.x) {
        const int i = e % n, k = e / n;   // column-major (i, k)
        LU[e] = A[(size_t)i * n + k]; Li[e] = Ui[e] = i == k;
    }
}
__global__ void bound_rows_perm(const int8_t *__restrict__ A, const int *__restrict__ ord, int n, int8_t *__restrict__ PA) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * n; e += (size_t)gridDim.x * blockDim.x)
        PA[e] = A[(size_t)ord[e / n] * n + e % n];
}

struct Garner { int P; uint32_t q[7]; uint32_t inv[7][7]; unsigned long long pre_lo[8], pre_hi[8]; };   // pre = prod_{t<k} q_t

// one block per row i: B_ij from its residues (Garner, 128-bit, symmetric), sum_j B_ij^2 rounded up into sq[i]
__global__ void __launch_bounds__(256) bound_rows(const uint16_t *__restrict__ R, size_t zr, int n, Garner g, double *sq) {
    const int i = blockIdx.x;
    const unsigned __int128 M = (unsigned __int128)g.pre_hi[g.P] << 64 | g.pre_lo[g.P];
    double acc = 0;
    for (int j = threadIdx.x; j < n; j += 256) {
        uint32_t v[7]; unsigned __int128 x = 0;
        for (int k = 0; k < g.P; k++) {
            uint32_t qk = g.q[k], t = (uint32_t)R[k * zr + (size_t)i * n + j];
            for (int u = 0; u < k; u++) t = (uint32_t)((uint64_t)(t + qk - v[u] % qk) % qk * g.inv[u][k] % qk);
            v[k] = t;
            x += (unsigned __int128)t * ((unsigned __int128)g.pre_hi[k] << 64 | g.pre_lo[k]);
        }
        if (x > M / 2) x = M - x;   // |B_ij|
        double d = __dadd_ru(__dmul_ru(__ull2double_ru((unsigned long long)(x >> 64)), 18446744073709551616.0),
                             __ull2double_ru((unsigned long long)x));
        acc = __dadd_ru(acc, __dmul_ru(d, d));
    }
    __shared__ double red[256];
    red[threadIdx.x] = acc; __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = __dadd_ru(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    if (threadIdx.x == 0) sq[i] = red[0];
}

// A: n x n int8 on the host. Returns a proved upper bound on log2 |det A|, or +inf when FP32 LU finds A singular
// (the caller then keeps Hadamard). *bits_B: the exact size of B's entries, *primes: how many primes that took.
// dA: A on the device. Every launch goes to this host thread's default stream (build with --default-stream per-thread),
// so the bound can run on its own thread while the lifting runs on another.
inline double det_log2_bound(int n, const int8_t *dA, int s = 20, int *primes = nullptr) {
    const bool prof = getenv("OZ_PROF"); auto T = std::chrono::steady_clock::now();
    auto lap = [&](const char *what) {
        if (!prof) return;
        cudaStreamSynchronize(cudaStreamPerThread); auto t = std::chrono::steady_clock::now();
        printf("  bound %-10s %.3f s\n", what, std::chrono::duration<double>(t - T).count()); T = t;
    };
    cusolverDnHandle_t sh; cublasHandle_t bh; cusolverDnCreate(&sh); cublasCreate(&bh);
    cusolverDnSetStream(sh, cudaStreamPerThread); cublasSetStream(bh, cudaStreamPerThread);
    const size_t nn = (size_t)n * n;
    float *dLU, *dLi, *dUi, *dwork; int *dpiv, *dinfo, lw = 0;
    cudaMalloc(&dLU, nn * 4); cudaMalloc(&dLi, nn * 4); cudaMalloc(&dUi, nn * 4);
    cudaMalloc(&dpiv, n * 4); cudaMalloc(&dinfo, 4);
    bound_prep<<<2048, 256>>>(dA, n, dLU, dLi, dUi);
    cusolverDnSgetrf_bufferSize(sh, n, n, dLU, n, &lw); cudaMalloc(&dwork, (size_t)lw * 4);
    lap("setup");
    cusolverDnSgetrf(sh, n, n, dLU, n, dwork, dpiv, dinfo);
    lap("sgetrf");
    int info; cudaMemcpy(&info, dinfo, 4, cudaMemcpyDeviceToHost);
    double ub = INFINITY;
    if (info == 0) {
        const float one = 1.f;
        cublasStrsm(bh, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, n, &one, dLU, n, dLi, n);
        cublasStrsm(bh, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &one, dLU, n, dUi, n);
        long long *XL, *XU; double *drl, *dcu; long long *ddu; cudaMalloc(&ddu, n * 8);
        cudaMalloc(&XL, (size_t)n * n * 8); cudaMalloc(&XU, (size_t)n * n * 8); cudaMalloc(&drl, n * 8); cudaMalloc(&dcu, n * 8);
        bound_build_x<<<2048, 256>>>(dLU, dLi, dUi, n, s, XL, XU);
        bound_l1<<<(n + 7) / 8, 256>>>(XL, XU, n, drl, dcu, ddu);
        lap("trsm+x");
        cudaFree(dLi); cudaFree(dUi); cudaFree(dwork); cudaFree(dLU); dLU = nullptr;
        std::vector<double> rl(n), cu(n); std::vector<long long> du(n);
        cudaMemcpy(rl.data(), drl, n * 8, cudaMemcpyDeviceToHost); cudaMemcpy(cu.data(), dcu, n * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(du.data(), ddu, n * 8, cudaMemcpyDeviceToHost); cudaFree(ddu);
        double mr = 0, mc = 0; for (int i = 0; i < n; i++) { mr = std::max(mr, rl[i]); mc = std::max(mc, cu[i]); }
        lap("l1+diag");
        const double bb = log2(mr) + log2(127.0) + log2(mc) + 2;   // |B_ij| < 2^bb, and the sign
        // PA: LAPACK's ipiv is a sequence of row swaps (1-based)
        std::vector<int> piv(n), ord(n); cudaMemcpy(piv.data(), dpiv, n * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) ord[i] = i;
        for (int i = 0; i < n; i++) std::swap(ord[i], ord[piv[i] - 1]);
        int P = 0; double pb = 0; Garner g{}; g.pre_lo[0] = 1;
        for (uint32_t q = 65279; pb < bb + 1 && P < 7; q--) {
            bool pr = true; for (uint32_t t = 2; t * t <= q && pr; t++) pr = q % t;
            if (pr) { g.q[P++] = q; pb += log2((double)q); }
        }
        if (pb >= bb + 1) {
            g.P = P;
            for (int k = 0; k < P; k++) {
                unsigned __int128 pre = (unsigned __int128)g.pre_hi[k] << 64 | g.pre_lo[k], nx = pre * g.q[k];
                g.pre_lo[k + 1] = (unsigned long long)nx; g.pre_hi[k + 1] = (unsigned long long)(nx >> 64);
                for (int u = 0; u < k; u++) {   // q_u^-1 mod q_k
                    uint64_t r = 1, a = g.q[u] % g.q[k], e = g.q[k] - 2;
                    for (; e; e >>= 1, a = a * a % g.q[k]) if (e & 1) r = r * a % g.q[k];
                    g.inv[u][k] = (uint32_t)r;
                }
            }
            lap("PA host");
            int8_t *dPA; int *dord; cudaMalloc(&dPA, nn); cudaMalloc(&dord, n * 4);
            cudaMemcpy(dord, ord.data(), n * 4, cudaMemcpyHostToDevice);
            bound_rows_perm<<<2048, 256>>>(dA, dord, n, dPA); cudaFree(dord);
            // one prime at a time (n = 16384 fits in 12 GB): B mod q kept as 16-bit residues
            uint16_t *Bres; cudaMalloc(&Bres, nn * 2 * P);
            DetWork w; w.alloc(n, 1);
            for (int k = 0; k < P; k++) {
                cudaMemcpy(w.P, &g.q[k], 4, cudaMemcpyHostToDevice);
                det_init<<<dim3(256, 1), 256>>>(dPA, w.R, n, n, w.P);
                mod_xw<false, long long, int32_t>(w, XL, 0, n, w.R, w.zr(), n, n, n, n, 0, 0);   // T = X_L PA
                mod_xw<false, int32_t, long long>(w, w.R, w.zr(), n, XU, 0, n, n, n, n, 0, 0);   // B = T X_U
                to_u16<<<2048, 256>>>(w.R, n, 0, n, Bres + k * nn);
            }
            w.free_();
            lap("modgemm");
            double *dsq; cudaMalloc(&dsq, n * 8);
            bound_rows<<<n, 256>>>(Bres, nn, n, g, dsq); cudaFree(Bres);
            std::vector<double> sq(n); cudaMemcpy(sq.data(), dsq, n * 8, cudaMemcpyDeviceToHost);
            lap("rows");
            double lg = 0;
            for (int i = 0; i < n; i++) lg += 0.5 * log2(sq[i]);
            for (int j = 0; j < n; j++) lg -= log2((double)llabs(du[j]));
            lg -= (double)s * n;
            ub = lg + 1 + 1e-9 * n;   // rounding of the logs (relative 2^-52 each) is far inside one bit
            cudaFree(dPA); cudaFree(dsq);
            if (primes) *primes = P;
        }
        cudaFree(XL); cudaFree(XU); cudaFree(drl); cudaFree(dcu);
    }
    cudaFree(dLU); cudaFree(dpiv); cudaFree(dinfo);
    cusolverDnDestroy(sh); cublasDestroy(bh);
    return ub;
}

}  // namespace oz
