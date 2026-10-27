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
// L > 1 (entries of L byte planes): the float LU only needs an approximation of A, here A / 256^(L-1) rounded -- the
// bound stays proved, since B below is computed exactly from the integer A
__global__ void bound_prep(const int8_t *__restrict__ A, int n, float *__restrict__ LU, float *__restrict__ Li, float *__restrict__ Ui,
                           int L = 1) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * n; e += (size_t)gridDim.x * blockDim.x) {
        const int i = e % n, k = e / n;   // column-major (i, k)
        double a = 0; for (int j = 0; j < L; j++) a = a / 256 + A[(size_t)j * n * n + (size_t)i * n + k];
        LU[e] = (float)a; if (Li) Li[e] = Ui[e] = i == k;
    }
}
__global__ void bound_rows_perm(const int8_t *__restrict__ A, const int *__restrict__ ord, int n, int8_t *__restrict__ PA, int L = 1) {
    const size_t nn = (size_t)n * n;
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < L * nn; e += (size_t)gridDim.x * blockDim.x) {
        const size_t f = e % nn; PA[e] = A[e - f + (size_t)ord[f / n] * n + f % n];
    }
}

constexpr int BOUND_PMAX = 16;   // 16-bit primes: |B_ij| up to ~255 bits (L-byte entries add 8 (L - 1) bits to int8's ~110)
struct Garner { int P; uint32_t q[BOUND_PMAX]; uint32_t inv[BOUND_PMAX][BOUND_PMAX]; };

// the primes below 65280 whose product exceeds 2^(bb + 1), with Garner's constants; 0 when BOUND_PMAX are not enough
inline int garner_for(double bb, Garner &g) {
    int P = 0; double pb = 0; g = Garner{};
    for (uint32_t q = 65279; pb < bb + 1 && P < BOUND_PMAX; q--) {
        bool pr = true; for (uint32_t t = 2; t * t <= q && pr; t++) pr = q % t;
        if (pr) { g.q[P++] = q; pb += log2((double)q); }
    }
    if (pb < bb + 1) return 0;
    g.P = P;
    for (int k = 0; k < P; k++) {        for (int u = 0; u < k; u++) {   // q_u^-1 mod q_k
            uint64_t r = 1, a = g.q[u] % g.q[k], e = g.q[k] - 2;
            for (; e; e >>= 1, a = a * a % g.q[k]) if (e & 1) r = r * a % g.q[k];
            g.inv[u][k] = (uint32_t)r;
        }
    }
    return P;
}

// column-major n x w panel of the identity's columns [j0, j0 + w)
__global__ void bound_ident(float *__restrict__ I, int n, int j0, int w) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * w; e += (size_t)gridDim.x * blockDim.x)
        I[e] = e % n == j0 + e / n;
}
// the columns [j0, j0 + w) of X_L (upper = 0) or X_U (upper = 1) from the matching panel of L^-1 or U^-1 (column-major),
// row-major n x w as in bound_build_x
__global__ void bound_x_panel(const float *__restrict__ LU, const float *__restrict__ Pn, int n, int j0, int w, int s, int upper,
                              long long *__restrict__ X) {
    for (size_t e = blockIdx.x * (size_t)blockDim.x + threadIdx.x; e < (size_t)n * w; e += (size_t)gridDim.x * blockDim.x) {
        const int i = e / w, c = e % w, k = j0 + c;
        const double v = Pn[i + (size_t)c * n];
        X[e] = upper ? (i <= k ? llrint(ldexp(v * LU[k + (size_t)k * n], s)) : 0) : (k < i ? llrint(ldexp(v, s)) : k == i ? 1ll << s : 0);
    }
}

// one block per row i: B_ij from its residues (Garner, 128-bit, symmetric), sum_j B_ij^2 rounded up into sq[i]
__global__ void __launch_bounds__(256) bound_rows(const uint16_t *__restrict__ R, size_t zr, int n, Garner g, double *sq) {
    // B_ij by Garner as mixed-radix digits y_k. Symmetric: negative iff the digits exceed those of (M - 1) / 2, which are
    // (q_k - 1) / 2; then |B_ij| = (M - 1 - x) + 1, whose digits are q_k - 1 - y_k. |B_ij| rounded up by Horner in double
    // with upward rounding: no wide integers, any number of primes.
    const int i = blockIdx.x;
    double acc = 0;
    for (int j = threadIdx.x; j < n; j += 256) {
        uint32_t v[BOUND_PMAX];
        for (int k = 0; k < g.P; k++) {
            uint32_t qk = g.q[k], t = (uint32_t)R[k * zr + (size_t)i * n + j];
            for (int u = 0; u < k; u++) t = (uint32_t)((uint64_t)(t + qk - v[u] % qk) % qk * g.inv[u][k] % qk);
            v[k] = t;
        }
        bool neg = false;
        for (int k = g.P - 1; k >= 0; k--) { const uint32_t h = (g.q[k] - 1) / 2; if (v[k] != h) { neg = v[k] > h; break; } }
        double d = 0;
        for (int k = g.P - 1; k >= 0; k--) d = __dadd_ru(__dmul_ru(d, (double)g.q[k]), (double)(neg ? g.q[k] - 1 - v[k] : v[k]));
        if (neg) d = __dadd_ru(d, 1.0);
        acc = __dadd_ru(acc, __dmul_ru(d, d));
    }
    __shared__ double red[256];
    red[threadIdx.x] = acc; __syncthreads();
    for (int o = 128; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = __dadd_ru(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
    if (threadIdx.x == 0) sq[i] = red[0];
}

// x rounded up to a double
inline double ru_double(unsigned __int128 x) { double d = (double)x; return (unsigned __int128)d < x ? nextafter(d, INFINITY) : d; }

// The same bound for n where nothing n x n in float or int64 fits next to another on 12 GB (n = 32768: 4 GB per float
// matrix, 8.6 GB per int64 one). X_L and X_U are built in column panels (trsm against identity panels) and kept on the
// host; per prime q, T = X_L PA mod q in row panels (T int32 on the GPU), then B = T X_U in column panels to the host as
// 16-bit residues; then Garner and the row sums in row panels. The L1 norms are summed exactly (128-bit) on the host.
inline double det_log2_bound_panels(int n, const int8_t *dA, int s, int *primes) {
    const bool prof = getenv("OZ_PROF"); auto T0 = std::chrono::steady_clock::now();
    auto lap = [&](const char *what) {
        if (!prof) return;
        cudaStreamSynchronize(cudaStreamPerThread); auto t = std::chrono::steady_clock::now();
        printf("  bound %-10s %.3f s\n", what, std::chrono::duration<double>(t - T0).count()); T0 = t;
    };
    const int wb = std::min(n, 2048);
    const size_t nn = (size_t)n * n;
    cusolverDnHandle_t sh; cublasHandle_t bh; cusolverDnCreate(&sh); cublasCreate(&bh);
    cusolverDnSetStream(sh, cudaStreamPerThread); cublasSetStream(bh, cudaStreamPerThread);
    float *dLU, *dwork, *dPn; int *dpiv, *dinfo, lw = 0; long long *dX;
    cudaMalloc(&dLU, nn * 4); cudaMalloc(&dpiv, n * 4); cudaMalloc(&dinfo, 4);
    bound_prep<<<2048, 256>>>(dA, n, dLU, nullptr, nullptr);
    cusolverDnSgetrf_bufferSize(sh, n, n, dLU, n, &lw); cudaMalloc(&dwork, (size_t)lw * 4);
    cusolverDnSgetrf(sh, n, n, dLU, n, dwork, dpiv, dinfo);
    int info; cudaMemcpy(&info, dinfo, 4, cudaMemcpyDeviceToHost);
    cudaFree(dwork); cudaFree(dinfo);
    lap("sgetrf");
    if (info) { cudaFree(dLU); cudaFree(dpiv); cusolverDnDestroy(sh); cublasDestroy(bh); return INFINITY; }
    std::vector<long long> XL(nn), XU(nn);
    cudaMalloc(&dPn, (size_t)n * wb * 4); cudaMalloc(&dX, (size_t)n * wb * 8);
    const float one = 1.f;
    for (int j0 = 0; j0 < n; j0 += wb) {
        const int w = std::min(wb, n - j0);
        for (int up = 0; up < 2; up++) {
            bound_ident<<<2048, 256>>>(dPn, n, j0, w);
            cublasStrsm(bh, CUBLAS_SIDE_LEFT, up ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                        up ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT, n, w, &one, dLU, n, dPn, n);
            bound_x_panel<<<2048, 256>>>(dLU, dPn, n, j0, w, s, up, dX);
            cudaMemcpy2D((up ? XU : XL).data() + j0, (size_t)n * 8, dX, (size_t)w * 8, (size_t)w * 8, n, cudaMemcpyDeviceToHost);
        }
    }
    cudaFree(dPn); cudaFree(dX); cudaFree(dLU); cusolverDnDestroy(sh); cublasDestroy(bh);
    lap("trsm+x");
    double mr = 0, mc = 0;
    std::vector<long long> du(n);
#pragma omp parallel for reduction(max : mr)
    for (int i = 0; i < n; i++) {
        unsigned __int128 a = 0;
        for (int k = 0; k <= i; k++) a += (unsigned long long)llabs(XL[(size_t)i * n + k]);
        mr = std::max(mr, ru_double(a));
    }
#pragma omp parallel for reduction(max : mc)
    for (int j0 = 0; j0 < n; j0 += 64) {
        unsigned __int128 b[64] = {};
        for (int i = 0; i < std::min(n, j0 + 64); i++)
            for (int c = 0; c < 64; c++) b[c] += (unsigned long long)llabs(XU[(size_t)i * n + j0 + c]);
        for (int c = 0; c < 64; c++) { mc = std::max(mc, ru_double(b[c])); du[j0 + c] = XU[(size_t)(j0 + c) * n + j0 + c]; }
    }
    lap("l1+diag");
    const double bb = log2(mr) + log2(127.0) + log2(mc) + 2;
    Garner g; const int P = garner_for(bb, g);
    if (!P) { cudaFree(dpiv); return INFINITY; }
    std::vector<int> piv(n), ord(n); cudaMemcpy(piv.data(), dpiv, n * 4, cudaMemcpyDeviceToHost); cudaFree(dpiv);
    for (int i = 0; i < n; i++) ord[i] = i;
    for (int i = 0; i < n; i++) std::swap(ord[i], ord[piv[i] - 1]);
    int8_t *dPA; int *dord; cudaMalloc(&dPA, nn); cudaMalloc(&dord, n * 4);
    cudaMemcpy(dord, ord.data(), n * 4, cudaMemcpyHostToDevice);
    bound_rows_perm<<<2048, 256>>>(dA, dord, n, dPA); cudaFree(dord);
    std::vector<uint16_t> Bh(nn * P);
    {
        DetWork wt, wp; wt.alloc(n, 1); wp.alloc(n, 1, wb);   // T = X_L PA mod q (n x n); one column panel of B (n x wb)
        long long *dXp; uint16_t *dBu; cudaMalloc(&dXp, (size_t)n * wb * 8); cudaMalloc(&dBu, (size_t)n * wb * 2);
        for (int k = 0; k < P; k++) {
            cudaMemcpy(wt.P, &g.q[k], 4, cudaMemcpyHostToDevice); cudaMemcpy(wp.P, &g.q[k], 4, cudaMemcpyHostToDevice);
            for (int i0 = 0; i0 < n; i0 += wb) {
                const int h = std::min(wb, n - i0);
                cudaMemcpy(dXp, XL.data() + (size_t)i0 * n, (size_t)h * n * 8, cudaMemcpyHostToDevice);
                mod_xw<false, long long, int8_t>(wt, dXp, 0, n, dPA, 0, n, h, n, n, i0, 0);
            }
            for (int j0 = 0; j0 < n; j0 += wb) {
                const int w = std::min(wb, n - j0);
                cudaMemcpy2D(dXp, (size_t)w * 8, XU.data() + j0, (size_t)n * 8, (size_t)w * 8, n, cudaMemcpyHostToDevice);
                mod_xw<false, int32_t, long long>(wp, wt.R, wt.zr(), n, dXp, 0, w, n, w, n, 0, 0);
                to_u16_rc<<<2048, 256>>>(wp.R, wp.ld, 0, n, w, dBu);
                cudaMemcpy2D(Bh.data() + k * nn + j0, (size_t)n * 2, dBu, (size_t)w * 2, (size_t)w * 2, n, cudaMemcpyDeviceToHost);
            }
        }
        wt.free_(); wp.free_(); cudaFree(dXp); cudaFree(dBu); cudaFree(dPA);
    }
    if (cudaError_t e = cudaGetLastError()) { fprintf(stderr, "bound: %s\n", cudaGetErrorString(e)); exit(1); }
    lap("modgemm");
    uint16_t *dR; double *dsq; cudaMalloc(&dR, (size_t)P * wb * n * 2); cudaMalloc(&dsq, n * 8);
    for (int i0 = 0; i0 < n; i0 += wb) {
        const int h = std::min(wb, n - i0);
        for (int k = 0; k < P; k++)
            cudaMemcpy(dR + (size_t)k * wb * n, Bh.data() + k * nn + (size_t)i0 * n, (size_t)h * n * 2, cudaMemcpyHostToDevice);
        bound_rows<<<h, 256>>>(dR, (size_t)wb * n, n, g, dsq + i0);
    }
    std::vector<double> sq(n); cudaMemcpy(sq.data(), dsq, n * 8, cudaMemcpyDeviceToHost); cudaFree(dR); cudaFree(dsq);
    lap("rows");
    double lg = 0;
    for (int i = 0; i < n; i++) lg += 0.5 * log2(sq[i]);
    for (int j = 0; j < n; j++) lg -= log2((double)llabs(du[j]));
    lg -= (double)s * n;
    if (primes) *primes = P;
    return lg + 1 + 1e-9 * n;
}

// A: n x n int8 on the host. Returns a proved upper bound on log2 |det A|, or +inf when FP32 LU finds A singular
// (the caller then keeps Hadamard). *bits_B: the exact size of B's entries, *primes: how many primes that took.
// dA: A on the device. Every launch goes to this host thread's default stream (build with --default-stream per-thread),
// so the bound can run on its own thread while the lifting runs on another.
// L > 1: entries of L byte planes (n <= 16384)
inline double det_log2_bound(int n, const int8_t *dA, int s = 20, int *primes = nullptr, int L = 1) {
    if (L > 1 && n > 16384) return INFINITY;
    if (n > 16384 || getenv("OZ_PANEL") || getenv("OZ_KCAP")) return det_log2_bound_panels(n, dA, s, primes);   // in place with K = n below
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
    bound_prep<<<2048, 256>>>(dA, n, dLU, dLi, dUi, L);
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
        const double bb = log2(mr) + log2(127.0 * ((pow(256.0, L) - 1) / 255)) + log2(mc) + 2;   // |B_ij| < 2^bb, and the sign
        // PA: LAPACK's ipiv is a sequence of row swaps (1-based)
        std::vector<int> piv(n), ord(n); cudaMemcpy(piv.data(), dpiv, n * 4, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; i++) ord[i] = i;
        for (int i = 0; i < n; i++) std::swap(ord[i], ord[piv[i] - 1]);
        Garner g{}; const int P = garner_for(bb, g);
        if (P) {
            lap("PA host");
            int8_t *dPA; int *dord; cudaMalloc(&dPA, L * nn); cudaMalloc(&dord, n * 4);
            cudaMemcpy(dord, ord.data(), n * 4, cudaMemcpyHostToDevice);
            bound_rows_perm<<<2048, 256>>>(dA, dord, n, dPA, L); cudaFree(dord);
            // one prime at a time (n = 16384 fits in 12 GB): B mod q kept as 16-bit residues
            uint16_t *Bres; cudaMalloc(&Bres, nn * 2 * P);
            DetWork w; w.alloc(n, 1);
            for (int k = 0; k < P; k++) {
                cudaMemcpy(w.P, &g.q[k], 4, cudaMemcpyHostToDevice);
                det_init<<<dim3(256, 1), 256>>>(dPA, w.R, n, n, w.P, L);
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
