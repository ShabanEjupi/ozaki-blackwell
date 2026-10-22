// solve_td.cu -- Ax = b at kappa(A) = 1e36 in triple-double (~159 bits) on the GPU, where FP64 and double-double have no
// correct digit (kappa * u = 1e20 and 1e4). The structure of solve.cu with three parts per number:
//   A = Q1 diag(sigma) Q2 built with MPFR on the host (DST-I from a table of 2n+2 sines, sigma from 1 to 1e-36) and the
//   product as a triple-double GEMM on the INT8 tensor cores; b = A x_true summed exactly in MPFR.
//   LU: recursive panel (16-column cooperative leaves, the row of a thread in registers), halves joined by td GEMMs,
//   U12 through the assembled inverse of L11, trailing updates as td GEMMs subtracting in place; left-looking solves
//   with inverted 32 x 32 diagonal blocks; one refinement step with the original A in triple-double.
//   Reference: LU with partial pivoting in MPFR at 160 bits on every CPU core (what one would run today), and FP64 LU.
// Build: nvcc -O3 -arch=sm_120 -std=c++17 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off -o out_soltd bench/solve_td.cu -lmpfr -lgmp -lgomp
// Run:   ./out_soltd n [nb [kappa_exponent [seed]]]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <mpfr.h>
#include <omp.h>
#include <cooperative_groups.h>
#include "../src/ozaki_td.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---- triple-double arithmetic, host and device; "sloppy" in the third part (errors ~2^-155 relative), renormalized
struct T { double x, y, z; };
#define HD static inline __host__ __device__
HD void tsum(double a, double b, double &s, double &e) { s = a + b; double v = s - a; e = (a - (s - v)) + (b - v); }
HD void qsum(double a, double b, double &s, double &e) { s = a + b; e = b - (s - a); }
HD T tneg(T a) { return {-a.x, -a.y, -a.z}; }
HD T tadd(T a, T b) {
    double s0, e0, s1, e1, t, f, h, m, l;
    tsum(a.x, b.x, s0, e0); tsum(a.y, b.y, s1, e1);
    tsum(s1, e0, t, f);
    double s2 = a.z + b.z + e1 + f;
    tsum(s0, t, h, m); tsum(m, s2, m, l); qsum(h, m, h, m);
    return {h, m, l};
}
HD T tsub(T a, T b) { return tadd(a, tneg(b)); }
HD T tmul(T a, T b) {
    double p0 = a.x * b.x, q0 = fma(a.x, b.x, -p0), p1 = a.x * b.y, q1 = fma(a.x, b.y, -p1), p2 = a.y * b.x, q2 = fma(a.y, b.x, -p2);
    double u, f, g, h, m, l;
    tsum(p1, p2, u, f); tsum(u, q0, u, g);
    double w = a.x * b.z + a.y * b.y + a.z * b.x + q1 + q2 + f + g;
    qsum(p0, u, h, m); tsum(m, w, m, l); qsum(h, m, h, m);
    return {h, m, l};
}
HD T tdiv(T a, T b) {
    double q0 = a.x / b.x; T r = tsub(a, tmul(b, {q0, 0, 0}));
    double q1 = r.x / b.x; r = tsub(r, tmul(b, {q1, 0, 0}));
    double q2 = r.x / b.x;
    return tadd(tadd({q0, 0, 0}, {q1, 0, 0}), {q2, 0, 0});
}
// a matrix in three planes, row-major
struct M3 { double *x, *y, *z; };
#define AT(i, j) ((size_t)(i) * n + (j))
__device__ __forceinline__ T ld3(const M3 &a, size_t k) { return {a.x[k], a.y[k], a.z[k]}; }
__device__ __forceinline__ void st3(const M3 &a, size_t k, T v) { a.x[k] = v.x; a.y[k] = v.y; a.z[k] = v.z; }

// ---- panel leaf: columns j0..j0+wl (wl <= 16), rows j0..n-1; a thread per row holding the row, one grid sync per column
constexpr int LW = 16;
__global__ void __launch_bounds__(256) k_leaf(M3 A, int n, int j0, int wl, int *piv, double *bv, int *bi, double *cx, double *cy, double *cz) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    __shared__ double sv[8], ux[LW], uy[LW], uz[LW], rx, ry, rz; __shared__ int si[8], sp;
    const int G = gridDim.x, lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    int row = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = row < n;
    bool act = valid;
    double h[LW], m[LW], l[LW];
#pragma unroll
    for (int c = 0; c < LW; c++) {
        bool on = act && c < wl;
        h[c] = on ? A.x[AT(row, j0 + c)] : 0; m[c] = on ? A.y[AT(row, j0 + c)] : 0; l[c] = on ? A.z[AT(row, j0 + c)] : 0;
    }
    for (int j = 0; j < wl; j++) {   // not unrolled (grid.sync); the row is indexed by compile-time c only
        const int cj = j0 + j, par = j & 1;
        double hj = 0;
#pragma unroll
        for (int c = 0; c < LW; c++) if (c == j) hj = h[c];
        double v = act ? fabs(hj) : -1; int ix = act ? row : 0x7fffffff;
        for (int o = 16; o; o >>= 1) {
            double x = __shfl_xor_sync(~0u, v, o); int y = __shfl_xor_sync(~0u, ix, o);
            if (x > v || (x == v && y < ix)) { v = x; ix = y; }
        }
        if (!lane) { sv[wib] = v; si[wib] = ix; }
        __syncthreads();
        if (!threadIdx.x) {
            int b = 0;
            for (int w = 1; w < (int)(blockDim.x >> 5); w++) if (sv[w] > sv[b] || (sv[w] == sv[b] && si[w] < si[b])) b = w;
            bv[par * G + blockIdx.x] = sv[b]; bi[par * G + blockIdx.x] = si[b]; sp = si[b];
        }
        __syncthreads();
        if (act && row == sp) {
            size_t q = ((size_t)par * G + blockIdx.x) * LW;
#pragma unroll
            for (int c = 0; c < LW; c++) { cx[q + c] = h[c]; cy[q + c] = m[c]; cz[q + c] = l[c]; }
        }
        grid.sync();
        if (wib == 0) {   // every block finds the winner itself
            double bvv = -1; int bii = 0x7fffffff, bb = 0;
            for (int b = lane; b < G; b += 32) {
                double x = bv[par * G + b]; int y = bi[par * G + b];
                if (x > bvv || (x == bvv && y < bii)) { bvv = x; bii = y; bb = b; }
            }
            for (int o = 16; o; o >>= 1) {
                double x = __shfl_xor_sync(~0u, bvv, o); int y = __shfl_xor_sync(~0u, bii, o), z = __shfl_xor_sync(~0u, bb, o);
                if (x > bvv || (x == bvv && y < bii)) { bvv = x; bii = y; bb = z; }
            }
            if (lane < LW) { size_t q = ((size_t)par * G + bb) * LW + lane; ux[lane] = cx[q]; uy[lane] = cy[q]; uz[lane] = cz[q]; }
            __syncwarp();
            if (!lane) { sp = bii; T r = tdiv({1, 0, 0}, {ux[j], uy[j], uz[j]}); rx = r.x; ry = r.y; rz = r.z; }
        }
        __syncthreads();
        const int p = sp;
        if (blockIdx.x == 0 && threadIdx.x == 0) piv[cj] = p;
        if (act) {   // rename instead of swapping: the owner of row cj becomes row p, the pivot row's owner retires as cj
            if (row == cj) { if (p == cj) act = false; else row = p; }
            else if (row == p) { act = false; row = cj; }
        }
        if (act) {
            T mu{0, 0, 0};
#pragma unroll
            for (int c = 0; c < LW; c++)
                if (c == j) {   // a / d as a * (1/d) plus one Newton step
                    T a{h[c], m[c], l[c]}, ri{rx, ry, rz};
                    mu = tmul(a, ri); mu = tadd(mu, tmul(tsub(a, tmul(mu, {ux[c], uy[c], uz[c]})), ri));
                    h[c] = mu.x; m[c] = mu.y; l[c] = mu.z;
                } else if (c > j) {
                    T x = tsub({h[c], m[c], l[c]}, tmul(mu, {ux[c], uy[c], uz[c]}));
                    h[c] = x.x; m[c] = x.y; l[c] = x.z;
                }
        }
        __syncthreads();
    }
    if (valid)
#pragma unroll
        for (int c = 0; c < LW; c++) if (c < wl) { A.x[AT(row, j0 + c)] = h[c]; A.y[AT(row, j0 + c)] = m[c]; A.z[AT(row, j0 + c)] = l[c]; }
}
__global__ void k_laswp(M3 A, int n, int j0, int w, const int *piv) {   // the leaf's swaps on every other column
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n || (c >= j0 && c < j0 + w)) return;
    for (int j = j0; j < j0 + w; j++) {
        int p = piv[j];
        if (p == j) continue;
        T a = ld3(A, AT(j, c)), b = ld3(A, AT(p, c));
        st3(A, AT(j, c), b); st3(A, AT(p, c), a);
    }
}
// inverse of the unit lower w x w block at (k0, k0) into I (leading dimension ld): block c solves for column c
__global__ void k_inv(M3 A, int n, int k0, int w, M3 I, int ld) {
    __shared__ double xx[64], xy[64], xz[64];
    int c = blockIdx.x, r = threadIdx.x;
    xx[r] = r == c; xy[r] = 0; xz[r] = 0;
    __syncthreads();
    for (int t = c; t < w - 1; t++) {
        if (r > t) { T v = tsub({xx[r], xy[r], xz[r]}, tmul(ld3(A, AT(k0 + r, k0 + t)), {xx[t], xy[t], xz[t]})); xx[r] = v.x; xy[r] = v.y; xz[r] = v.z; }
        __syncthreads();
    }
    size_t q = (size_t)r * ld + c;
    I.x[q] = xx[r]; I.y[q] = xy[r]; I.z[q] = xz[r];
}

// ---- triple-double GEMM on blocks in place: C (mode 0) = A B, (1) -= A B, (2) = -A B; C wider than the residue planes in chunks
struct Mm { oz::Plan p; M3 s; double log2P; size_t clcap; int ncap; };
static Mm mm_new(int n, int nb, size_t scratch) {
    Mm s;
    s.log2P = 0; for (int l = 0; l < 49; l++) s.log2P += log2((double)oz::MODULI[l]);
    CK(cudaMalloc(&s.s.x, scratch * 8)); CK(cudaMalloc(&s.s.y, scratch * 8)); CK(cudaMalloc(&s.s.z, scratch * 8));
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    int np = (n + oz::BN - 1) / oz::BN * oz::BN, kp = (nb + oz::BK - 1) / oz::BK * oz::BK;
    size_t left = fr - std::min(fr, ((size_t)49 * 2 * np * kp + ((size_t)768 << 20)));
    s.ncap = (int)std::min<size_t>(np, left / ((size_t)49 * 4 * np) / oz::BN * oz::BN);
    if (s.ncap < oz::BN) { fprintf(stderr, "mm: no room for the residue planes (%zu MB free)\n", fr >> 20); exit(1); }
    s.p = oz::plan(n, s.ncap, nb, 49, 160);
    s.clcap = (size_t)np * s.ncap;
    return s;
}
static void mm_free(Mm &s) { oz::release(s.p); cudaFree(s.s.x); cudaFree(s.s.y); cudaFree(s.s.z); }
static void mm(Mm &s, int M, int N, int K, M3 a, size_t ao, int lda, M3 b, size_t bo, int ldb, M3 c, size_t co, int ldo, int mode) {
    oz::Plan &p = s.p;
    p.M = M; p.K = K;
    p.Mp = (M + oz::BM - 1) / oz::BM * oz::BM; p.Kp = (K + oz::BK - 1) / oz::BK * oz::BK;
    p.bits = std::min(160, (int)floor((s.log2P - 2 - log2((double)K)) / 2));
    int nc = (int)std::min<size_t>(s.ncap, s.clcap / p.Mp / oz::BN * oz::BN);
    for (int c0 = 0; c0 < N; c0 += nc) {
        int w = std::min(nc, N - c0);
        p.N = w; p.Np = (w + oz::BN - 1) / oz::BN * oz::BN;
        if (K % oz::BK) { CK(cudaMemsetAsync(p.Ar, 0, (size_t)49 * p.Mp * p.Kp)); CK(cudaMemsetAsync(p.Br, 0, (size_t)49 * p.Np * p.Kp)); }
        oz::tdgemm_t<49>(p, a.x + ao, a.y + ao, a.z + ao, b.x + bo + c0, b.y + bo + c0, b.z + bo + c0,
                         c.x + co + c0, c.y + co + c0, c.z + co + c0, ldo, lda, ldb, mode);
    }
}

// ---- recursive panel: left half, U12 = inv(L11) A12, A22 -= L21 U12, right half; leaves inv of its unit lower block in r.I
struct Rec { int n, nb, sms; int *piv; double *bv, *cx, *cy, *cz; int *bi; M3 I; Mm *s; double tleaf; };
static void rec(Rec &r, M3 A, int k0, int j0, int w) {
    int n = r.n, nb = r.nb;
    size_t o = (size_t)(j0 - k0) * nb + (j0 - k0);
    if (w <= LW) {
        static const bool prof = getenv("OZ_PROF");
        double t0 = prof ? (cudaDeviceSynchronize(), now()) : 0;
        void *args[] = {&A, &n, &j0, &w, &r.piv, &r.bv, &r.bi, &r.cx, &r.cy, &r.cz};
        int T_ = std::min(256, std::max(32, (n - j0 + 32 * r.sms - 1) / (32 * r.sms) * 32));
        CK(cudaLaunchCooperativeKernel((void *)k_leaf, (n - j0 + T_ - 1) / T_, T_, args));
        if (prof) { cudaDeviceSynchronize(); r.tleaf += now() - t0; }
        k_laswp<<<(n + 255) / 256, 256>>>(A, n, j0, w, r.piv);
        M3 Io{r.I.x + o, r.I.y + o, r.I.z + o};
        k_inv<<<w, w>>>(A, n, j0, w, Io, nb);
        return;
    }
    int w1 = std::max(LW, w / (2 * LW) * LW), w2 = w - w1, c1 = j0 + w1;
    Mm &s = *r.s;
    size_t ia = o, ic = o + (size_t)w1 * nb + w1, ib = o + (size_t)w1 * nb, u = (size_t)j0 * n + c1, bl = (size_t)c1 * n + j0;
    rec(r, A, k0, j0, w1);
    mm(s, w1, w2, w1, r.I, ia, nb, A, u, n, A, u, n, 0);                              // U12 in place
    if (c1 < n) mm(s, n - c1, w2, w1, A, bl, n, A, u, n, A, (size_t)c1 * n + c1, n, 1);   // A22 -= L21 U12
    rec(r, A, k0, c1, w2);
    mm(s, w2, w1, w1, A, bl, n, r.I, ia, nb, s.s, 0, w1, 0);                          // B inv(A), B as swapped by the right half
    mm(s, w2, w1, w2, r.I, ic, nb, s.s, 0, w1, r.I, ib, nb, 2);                        // -inv(C) B inv(A)
}

// ---- solves
__global__ void k_permidx(int n, const int *piv, int *P) {
    extern __shared__ int ix[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) ix[i] = i;
    __syncthreads();
    if (!threadIdx.x) for (int j = 0; j < n; j++) { int p = piv[j], t = ix[j]; ix[j] = ix[p]; ix[p] = t; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) P[i] = ix[i];
}
__global__ void k_gather(int n, const int *P, M3 x, M3 y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st3(y, i, ld3(x, P[i]));
}
__global__ void k_dot(M3 A, int n, int r0, int c0, int c1, M3 X) {   // x_i -= A(i, c0:c1) x(c0:c1), a block per row
    __shared__ double wx[32], wy[32], wz[32];
    int i = r0 + blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    T s{0, 0, 0};
    for (int t = c0 + threadIdx.x; t < c1; t += blockDim.x) s = tadd(s, tmul(ld3(A, AT(i, t)), ld3(X, t)));
    for (int o = 16; o; o >>= 1) s = tadd(s, {__shfl_down_sync(~0u, s.x, o), __shfl_down_sync(~0u, s.y, o), __shfl_down_sync(~0u, s.z, o)});
    if (!lane) { wx[w] = s.x; wy[w] = s.y; wz[w] = s.z; }
    __syncthreads();
    if (!threadIdx.x) {
        for (int k = 1; k < (int)(blockDim.x >> 5); k++) s = tadd(s, {wx[k], wy[k], wz[k]});
        st3(X, i, tsub(ld3(X, i), s));
    }
}
constexpr int SB = 32;   // diagonal blocks of the solves, inverted once
__global__ void k_invdiag(M3 A, int n, bool upper, M3 D) {
    __shared__ double xx[SB], xy[SB], xz[SB];
    int b0 = blockIdx.x * SB, bb = min(SB, n - b0), c = blockIdx.y, r = threadIdx.x;
    if (c >= bb) return;
    if (r < bb) { xx[r] = r == c; xy[r] = 0; xz[r] = 0; }
    __syncthreads();
    for (int s = 0; s < bb; s++) {
        int t = upper ? bb - 1 - s : s;
        if (upper && r == t) { T v = tdiv({xx[t], xy[t], xz[t]}, ld3(A, AT(b0 + t, b0 + t))); xx[t] = v.x; xy[t] = v.y; xz[t] = v.z; }
        __syncthreads();
        if (r < bb && (upper ? r < t : r > t)) {
            T v = tsub({xx[r], xy[r], xz[r]}, tmul(ld3(A, AT(b0 + r, b0 + t)), {xx[t], xy[t], xz[t]})); xx[r] = v.x; xy[r] = v.y; xz[r] = v.z;
        }
        __syncthreads();
    }
    if (r < bb) st3(D, (size_t)(b0 + r) * SB + c, {xx[r], xy[r], xz[r]});
}
__global__ void k_dmv(int b0, int bb, M3 D, M3 X) {   // x_b = inv(T_b) x_b, a warp per row
    __shared__ double sx[SB], sy[SB], sz[SB];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (threadIdx.x < bb) { sx[threadIdx.x] = X.x[b0 + threadIdx.x]; sy[threadIdx.x] = X.y[b0 + threadIdx.x]; sz[threadIdx.x] = X.z[b0 + threadIdx.x]; }
    __syncthreads();
    for (int i = w; i < bb; i += nw) {
        T s{0, 0, 0};
        if (lane < bb) s = tmul(ld3(D, (size_t)(b0 + i) * SB + lane), {sx[lane], sy[lane], sz[lane]});
        for (int o = 16; o; o >>= 1) s = tadd(s, {__shfl_down_sync(~0u, s.x, o), __shfl_down_sync(~0u, s.y, o), __shfl_down_sync(~0u, s.z, o)});
        if (!lane) st3(X, b0 + i, s);
    }
}
__global__ void k_resid(M3 A, int n, M3 B, M3 X) {   // b_i -= A(i, :) x, a warp per row
    int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (i >= n) return;
    T s{0, 0, 0};
    for (int t = lane; t < n; t += 32) s = tadd(s, tmul(ld3(A, AT(i, t)), ld3(X, t)));
    for (int o = 16; o; o >>= 1) s = tadd(s, {__shfl_down_sync(~0u, s.x, o), __shfl_down_sync(~0u, s.y, o), __shfl_down_sync(~0u, s.z, o)});
    if (!lane) st3(B, i, tsub(ld3(B, i), s));
}
__global__ void k_addv(int n, M3 D, M3 X) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) st3(X, i, tadd(ld3(X, i), ld3(D, i)));
}

static M3 dalloc(size_t k) { M3 a; CK(cudaMalloc(&a.x, k * 8)); CK(cudaMalloc(&a.y, k * 8)); CK(cudaMalloc(&a.z, k * 8)); return a; }
static void dfree(M3 a) { cudaFree(a.x); cudaFree(a.y); cudaFree(a.z); }
static void up(M3 d, const std::vector<double> *h, size_t k) { for (int c = 0; c < 3; c++) CK(cudaMemcpy(c == 0 ? d.x : c == 1 ? d.y : d.z, h[c].data(), k * 8, cudaMemcpyHostToDevice)); }
static void down(std::vector<double> *h, M3 d, size_t k) { for (int c = 0; c < 3; c++) CK(cudaMemcpy(h[c].data(), c == 0 ? d.x : c == 1 ? d.y : d.z, k * 8, cudaMemcpyDeviceToHost)); }

// x = A^-1 b on the GPU in triple-double; seconds from A, b on the host to x on the host
static double solve_gpu(int n, int nb, const std::vector<double> *Ah, const std::vector<double> *bh, std::vector<double> *xh) {
    size_t nn = (size_t)n * n;
    M3 A = dalloc(nn), A0 = dalloc(nn), X = dalloc(n), B0 = dalloc(n), Tm = dalloc(n), I = dalloc((size_t)nb * nb);
    M3 DL = dalloc((size_t)n * SB), DU = dalloc((size_t)n * SB);
    int *piv, *P; CK(cudaMalloc(&piv, n * 4)); CK(cudaMalloc(&P, n * 4));
    Mm s = mm_new(n, nb, (size_t)nb * nb);
    cudaDeviceProp dp; CK(cudaGetDeviceProperties(&dp, 0));
    int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_leaf, 256, 0));
    Rec r{n, nb, dp.multiProcessorCount, piv, nullptr, nullptr, nullptr, nullptr, nullptr, I, &s, 0};
    int G = dp.multiProcessorCount * occ;
    if ((long)G * 256 < n) { fprintf(stderr, "leaf: %d x 256 threads < n\n", G); exit(1); }
    CK(cudaMalloc(&r.bv, G * 2 * 8)); CK(cudaMalloc(&r.bi, G * 2 * 4));
    CK(cudaMalloc(&r.cx, (size_t)G * 2 * LW * 8)); CK(cudaMalloc(&r.cy, (size_t)G * 2 * LW * 8)); CK(cudaMalloc(&r.cz, (size_t)G * 2 * LW * 8));
    if (n * 4 > 48 * 1024) CK(cudaFuncSetAttribute(k_permidx, cudaFuncAttributeMaxDynamicSharedMemorySize, n * 4));
    static const bool prof = getenv("OZ_PROF"); double ph[4] = {}, tp;
#define PH(k) if (prof) { cudaDeviceSynchronize(); double t_ = now(); ph[k] += t_ - tp; tp = t_; }
    CK(cudaDeviceSynchronize());
    double t0 = now(); tp = t0;
    up(A, Ah, nn); up(X, bh, n);
    for (int c = 0; c < 3; c++) {
        CK(cudaMemcpyAsync(c == 0 ? A0.x : c == 1 ? A0.y : A0.z, c == 0 ? A.x : c == 1 ? A.y : A.z, nn * 8, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(c == 0 ? B0.x : c == 1 ? B0.y : B0.z, c == 0 ? X.x : c == 1 ? X.y : X.z, n * 8, cudaMemcpyDeviceToDevice));
    }
    PH(0);
    for (int k0 = 0; k0 < n; k0 += nb) {
        int kb = std::min(nb, n - k0), e = k0 + kb, m = n - e;
        for (double *q : {I.x, I.y, I.z}) CK(cudaMemsetAsync(q, 0, (size_t)nb * nb * 8));
        rec(r, A, k0, k0, kb);
        PH(1);
        if (!m) break;
        mm(s, kb, m, kb, I, 0, nb, A, (size_t)k0 * n + e, n, A, (size_t)k0 * n + e, n, 0);
        mm(s, m, m, kb, A, (size_t)e * n + k0, n, A, (size_t)k0 * n + e, n, A, (size_t)e * n + e, n, 1);
        CK(cudaGetLastError());
        PH(2);
    }
    k_permidx<<<1, 1024, n * 4>>>(n, piv, P);
    k_invdiag<<<dim3((n + SB - 1) / SB, SB), SB>>>(A, n, false, DL);
    k_invdiag<<<dim3((n + SB - 1) / SB, SB), SB>>>(A, n, true, DU);
    auto trisolve = [&](M3 Y) {
        k_gather<<<(n + 255) / 256, 256>>>(n, P, Y, Tm);
        for (int c = 0; c < 3; c++) CK(cudaMemcpyAsync(c == 0 ? Y.x : c == 1 ? Y.y : Y.z, c == 0 ? Tm.x : c == 1 ? Tm.y : Tm.z, n * 8, cudaMemcpyDeviceToDevice));
        for (int b0 = 0; b0 < n; b0 += SB) {   // L y = P b
            int bb = std::min(SB, n - b0);
            if (b0) k_dot<<<bb, 256>>>(A, n, b0, 0, b0, Y);
            k_dmv<<<1, 1024>>>(b0, bb, DL, Y);
        }
        for (int b0 = (n - 1) / SB * SB; b0 >= 0; b0 -= SB) {   // U x = y
            int bb = std::min(SB, n - b0);
            if (b0 + bb < n) k_dot<<<bb, 256>>>(A, n, b0, b0 + bb, n, Y);
            k_dmv<<<1, 1024>>>(b0, bb, DU, Y);
        }
    };
    trisolve(X);
    PH(3);
    if (!getenv("OZ_NOREF")) {   // one refinement step: r = b - A x with the original A, x += LU \ r
        k_resid<<<(n + 7) / 8, 256>>>(A0, n, B0, X);
        trisolve(B0);
        k_addv<<<(n + 255) / 256, 256>>>(n, B0, X);
    }
    down(xh, X, n);
    double t = now() - t0;
    PH(3);
    if (prof) fprintf(stderr, "phases: upload %.3f  panels %.3f (leaves %.3f)  U12+trailing %.3f  solves %.3f s\n", ph[0], ph[1], r.tleaf, ph[2], ph[3]);
    CK(cudaGetLastError());
    mm_free(s);
    for (M3 q : {A, A0, X, B0, Tm, I, DL, DU}) dfree(q);
    for (void *q : {(void *)piv, (void *)P, (void *)r.bv, (void *)r.bi, (void *)r.cx, (void *)r.cy, (void *)r.cz}) cudaFree(q);
    return t;
}

// ---- host: the matrix in MPFR, references
static void split(mpfr_t v, double &a, double &b, double &c, mpfr_t t) {   // v -> three doubles, t scratch
    a = mpfr_get_d(v, MPFR_RNDN); mpfr_sub_d(t, v, a, MPFR_RNDN);
    b = mpfr_get_d(t, MPFR_RNDN); mpfr_sub_d(t, t, b, MPFR_RNDN);
    c = mpfr_get_d(t, MPFR_RNDN);
}
static void set3(mpfr_t v, double a, double b, double c) { mpfr_set_d(v, a, MPFR_RNDN); mpfr_add_d(v, v, b, MPFR_RNDN); mpfr_add_d(v, v, c, MPFR_RNDN); }

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1024, nb = argc > 2 ? atoi(argv[2]) : 256;
    double ek = argc > 3 ? atof(argv[3]) : 36;
    std::mt19937_64 g(argc > 4 ? atoll(argv[4]) : 7);
    int thr = omp_get_max_threads();
    printf("Ax = b in triple-double, n=%d, kappa_2(A) = 1e%.0f by construction, nb=%d, %d CPU threads\n", n, ek, nb, thr);
    size_t nn = (size_t)n * n;
    // Q1 = DST-I: sqrt(2/(n+1)) sin(pi (i+1)(j+1)/(n+1)), from a table of the 2n+2 sines; sigma_i = 10^(-ek i/(n-1))
    std::vector<double> S[3], sg[3], Q[3], Bm[3], Q2[3], A[3];
    for (int c = 0; c < 3; c++) { S[c].resize(2 * n + 2); sg[c].resize(n); Q[c].resize(nn); Bm[c].resize(nn); Q2[c].resize(nn); A[c].resize(nn); }
#pragma omp parallel
    {
        mpfr_t v, t, pi, sc; mpfr_inits2(400, v, t, pi, sc, (mpfr_ptr)0);
        mpfr_const_pi(pi, MPFR_RNDN); mpfr_set_ui(sc, 2, MPFR_RNDN); mpfr_div_ui(sc, sc, n + 1, MPFR_RNDN); mpfr_sqrt(sc, sc, MPFR_RNDN);
#pragma omp for
        for (int k = 0; k < 2 * n + 2; k++) {
            mpfr_mul_ui(v, pi, k, MPFR_RNDN); mpfr_div_ui(v, v, n + 1, MPFR_RNDN); mpfr_sin(v, v, MPFR_RNDN); mpfr_mul(v, v, sc, MPFR_RNDN);
            split(v, S[0][k], S[1][k], S[2][k], t);
        }
#pragma omp for
        for (int i = 0; i < n; i++) {   // 10^(-ek i / (n - 1))
            mpfr_set_d(t, -ek, MPFR_RNDN); mpfr_mul_si(t, t, i, MPFR_RNDN); mpfr_div_si(t, t, n - 1, MPFR_RNDN);
            mpfr_exp10(v, t, MPFR_RNDN);
            split(v, sg[0][i], sg[1][i], sg[2][i], t);
        }
        mpfr_clears(v, t, pi, sc, (mpfr_ptr)0);
    }
    std::vector<int> perm(n); for (int i = 0; i < n; i++) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), g);
    std::vector<double> sgn(n); for (auto &x : sgn) x = (g() & 1) ? 1 : -1;
#pragma omp parallel for
    for (int r = 0; r < n; r++)
        for (int i = 0; i < n; i++) {
            int k = (int)(((long)(r + 1) * (i + 1)) % (2 * n + 2));
            T q{S[0][k], S[1][k], S[2][k]}, b = tmul(q, {sg[0][i], sg[1][i], sg[2][i]});
            Bm[0][(size_t)r * n + i] = b.x; Bm[1][(size_t)r * n + i] = b.y; Bm[2][(size_t)r * n + i] = b.z;
            int k2 = (int)(((long)(perm[i] + 1) * (r + 1)) % (2 * n + 2));   // Q2[i][r] = sign_i Q1[perm i][r]
            Q2[0][(size_t)i * n + r] = sgn[i] * S[0][k2]; Q2[1][(size_t)i * n + r] = sgn[i] * S[1][k2]; Q2[2][(size_t)i * n + r] = sgn[i] * S[2][k2];
        }
    {   // A = B Q2 on the GPU (triple-double GEMM, row chunks)
        int R = std::min(n, 1024);
        M3 a = dalloc((size_t)R * n), b = dalloc(nn), c = dalloc((size_t)R * n);
        up(b, Q2, nn);
        oz::Plan p = oz::plan(R, n, n, 49, 160);
        for (int r0 = 0; r0 < n; r0 += R) {
            int rr = std::min(R, n - r0);
            for (int k = 0; k < 3; k++) CK(cudaMemcpy(k == 0 ? a.x : k == 1 ? a.y : a.z, Bm[k].data() + (size_t)r0 * n, (size_t)rr * n * 8, cudaMemcpyHostToDevice));
            p.M = rr; p.Mp = (rr + oz::BM - 1) / oz::BM * oz::BM;
            oz::tdgemm_t<49>(p, a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
            CK(cudaGetLastError());
            for (int k = 0; k < 3; k++) CK(cudaMemcpy(A[k].data() + (size_t)r0 * n, k == 0 ? c.x : k == 1 ? c.y : c.z, (size_t)rr * n * 8, cudaMemcpyDeviceToHost));
        }
        oz::release(p); dfree(a); dfree(b); dfree(c);
    }
    for (int c = 0; c < 3; c++) { std::vector<double>().swap(Q[c]); std::vector<double>().swap(Bm[c]); std::vector<double>().swap(Q2[c]); }
    // x_true in [-1, 1]; b = A x_true summed exactly (600 bits) and rounded to triple-double
    std::vector<double> xt(n), b[3];
    std::uniform_real_distribution<double> u(-1, 1);
    for (auto &v : xt) v = u(g);
    for (int c = 0; c < 3; c++) b[c].resize(n);
#pragma omp parallel
    {
        mpfr_t s, a, t; mpfr_inits2(600, s, a, t, (mpfr_ptr)0);
#pragma omp for
        for (int i = 0; i < n; i++) {
            mpfr_set_zero(s, 1);
            for (int j = 0; j < n; j++) { set3(a, A[0][(size_t)i * n + j], A[1][(size_t)i * n + j], A[2][(size_t)i * n + j]); mpfr_mul_d(a, a, xt[j], MPFR_RNDN); mpfr_add(s, s, a, MPFR_RNDN); }
            split(s, b[0][i], b[1][i], b[2][i], t);
        }
        mpfr_clears(s, a, t, (mpfr_ptr)0);
    }
    auto err = [&](const std::vector<double> *x) {   // max |x - x_true| / max |x_true|, the parts summed in order
        double w = 0, m = 0;
        for (int i = 0; i < n; i++) { w = fmax(w, fabs(((x[0][i] - xt[i]) + x[1][i]) + x[2][i])); m = fmax(m, fabs(xt[i])); }
        return w / m;
    };
    printf("%-46s %9s %15s\n", "solver", "seconds", "err vs x_true");
    std::vector<double> x[3]; for (int c = 0; c < 3; c++) x[c].assign(n, 0);
    double tg = solve_gpu(n, nb, A, b, x);   // warm-up
    tg = solve_gpu(n, nb, A, b, x);
    printf("%-46s %9.3f %15.3g\n", "triple-double LU, all on GPU", tg, err(x));
    if (getenv("OZ_GPU")) return 0;
    {   // FP64 LU
        std::vector<double> F(A[0]), y(b[0]); std::vector<int> pv(n);
        double t0 = now();
        for (int j = 0; j < n; j++) {
            int p = j; for (int i = j + 1; i < n; i++) if (fabs(F[(size_t)i * n + j]) > fabs(F[(size_t)p * n + j])) p = i;
            pv[j] = p; if (p != j) for (int c = 0; c < n; c++) std::swap(F[(size_t)j * n + c], F[(size_t)p * n + c]);
#pragma omp parallel for
            for (int i = j + 1; i < n; i++) { double l = F[(size_t)i * n + j] /= F[(size_t)j * n + j]; for (int c = j + 1; c < n; c++) F[(size_t)i * n + c] -= l * F[(size_t)j * n + c]; }
        }
        for (int j = 0; j < n; j++) std::swap(y[j], y[pv[j]]);
        for (int i = 0; i < n; i++) for (int t = 0; t < i; t++) y[i] -= F[(size_t)i * n + t] * y[t];
        for (int i = n - 1; i >= 0; i--) { for (int t = i + 1; t < n; t++) y[i] -= F[(size_t)i * n + t] * y[t]; y[i] /= F[(size_t)i * n + i]; }
        std::vector<double> z[3] = {y, std::vector<double>(n, 0), std::vector<double>(n, 0)};
        printf("%-46s %9.2f %15.3g\n", "FP64 LU (CPU)", now() - t0, err(z));
    }
    if (!getenv("OZ_NOCPUTD")) {   // triple-double LU on the CPU (the same arithmetic as the GPU kernels), rows in parallel, SIMD over columns
        std::vector<double> F[3] = {A[0], A[1], A[2]}, y[3] = {b[0], b[1], b[2]}; std::vector<int> pv(n);
        double t0 = now();
        for (int j = 0; j < n; j++) {
            int p = j; for (int i = j + 1; i < n; i++) if (fabs(F[0][(size_t)i * n + j]) > fabs(F[0][(size_t)p * n + j])) p = i;
            pv[j] = p;
            if (p != j) for (int c = 0; c < 3; c++) std::swap_ranges(F[c].begin() + (size_t)j * n, F[c].begin() + (size_t)j * n + n, F[c].begin() + (size_t)p * n);
            T d{F[0][(size_t)j * n + j], F[1][(size_t)j * n + j], F[2][(size_t)j * n + j]};
            const double *ux = &F[0][(size_t)j * n], *uy = &F[1][(size_t)j * n], *uz = &F[2][(size_t)j * n];
#pragma omp parallel for schedule(static)
            for (int i = j + 1; i < n; i++) {
                double *rx = &F[0][(size_t)i * n], *ry = &F[1][(size_t)i * n], *rz = &F[2][(size_t)i * n];
                T l = tdiv({rx[j], ry[j], rz[j]}, d);
                rx[j] = l.x; ry[j] = l.y; rz[j] = l.z;
#pragma omp simd
                for (int c = j + 1; c < n; c++) { T v = tsub({rx[c], ry[c], rz[c]}, tmul(l, {ux[c], uy[c], uz[c]})); rx[c] = v.x; ry[c] = v.y; rz[c] = v.z; }
            }
        }
        auto at = [&](int i, int k) { size_t q = (size_t)i * n + k; return T{F[0][q], F[1][q], F[2][q]}; };
        std::vector<T> z(n); for (int i = 0; i < n; i++) z[i] = {y[0][i], y[1][i], y[2][i]};
        for (int j = 0; j < n; j++) std::swap(z[j], z[pv[j]]);
        for (int i = 0; i < n; i++) for (int k = 0; k < i; k++) z[i] = tsub(z[i], tmul(at(i, k), z[k]));
        for (int i = n - 1; i >= 0; i--) { for (int k = i + 1; k < n; k++) z[i] = tsub(z[i], tmul(at(i, k), z[k])); z[i] = tdiv(z[i], at(i, i)); }
        double tc = now() - t0;
        std::vector<double> zz[3]; for (int c = 0; c < 3; c++) zz[c].resize(n);
        for (int i = 0; i < n; i++) { zz[0][i] = z[i].x; zz[1][i] = z[i].y; zz[2][i] = z[i].z; }
        char lab[64]; snprintf(lab, sizeof lab, "triple-double LU on the CPU, %d threads", thr);
        printf("%-46s %9.2f %15.3g\n", lab, tc, err(zz));
        printf("%-46s %9.0fx\n", "  speedup of the GPU", tc / tg);
    }
    if (getenv("OZ_NOMPFR")) return 0;
    {   // MPFR LU with partial pivoting at 160 bits, rows in parallel
        int prec = getenv("OZ_PREC") ? atoi(getenv("OZ_PREC")) : 160;
        std::vector<__mpfr_struct> F(nn), y(n);   // mpfr_t is an array type
        for (size_t k = 0; k < nn; k++) { mpfr_init2(&F[k], prec); set3(&F[k], A[0][k], A[1][k], A[2][k]); }
        for (int i = 0; i < n; i++) { mpfr_init2(&y[i], prec); set3(&y[i], b[0][i], b[1][i], b[2][i]); }
        std::vector<int> pv(n);
        double t0 = now();
        for (int j = 0; j < n; j++) {
            int p = j; for (int i = j + 1; i < n; i++) if (mpfr_cmpabs(&F[(size_t)i * n + j], &F[(size_t)p * n + j]) > 0) p = i;
            pv[j] = p; if (p != j) for (int c = 0; c < n; c++) mpfr_swap(&F[(size_t)j * n + c], &F[(size_t)p * n + c]);
#pragma omp parallel
            {
                mpfr_t l; mpfr_init2(l, prec);
#pragma omp for schedule(static)
                for (int i = j + 1; i < n; i++) {
                    mpfr_div(&F[(size_t)i * n + j], &F[(size_t)i * n + j], &F[(size_t)j * n + j], MPFR_RNDN);
                    mpfr_neg(l, &F[(size_t)i * n + j], MPFR_RNDN);
                    for (int c = j + 1; c < n; c++) mpfr_fma(&F[(size_t)i * n + c], l, &F[(size_t)j * n + c], &F[(size_t)i * n + c], MPFR_RNDN);
                }
                mpfr_clear(l);
            }
        }
        mpfr_t t; mpfr_init2(t, prec);
        for (int j = 0; j < n; j++) mpfr_swap(&y[j], &y[pv[j]]);
        for (int i = 0; i < n; i++) for (int k = 0; k < i; k++) { mpfr_mul(t, &F[(size_t)i * n + k], &y[k], MPFR_RNDN); mpfr_sub(&y[i], &y[i], t, MPFR_RNDN); }
        for (int i = n - 1; i >= 0; i--) {
            for (int k = i + 1; k < n; k++) { mpfr_mul(t, &F[(size_t)i * n + k], &y[k], MPFR_RNDN); mpfr_sub(&y[i], &y[i], t, MPFR_RNDN); }
            mpfr_div(&y[i], &y[i], &F[(size_t)i * n + i], MPFR_RNDN);
        }
        double tq = now() - t0;
        std::vector<double> z[3]; for (int c = 0; c < 3; c++) z[c].resize(n);
        mpfr_t s2; mpfr_init2(s2, prec);
        for (int i = 0; i < n; i++) split(&y[i], z[0][i], z[1][i], z[2][i], s2);
        char lab[64]; snprintf(lab, sizeof lab, "MPFR LU, %d bits, %d threads (reference)", prec, thr);
        printf("%-46s %9.2f %15.3g\n", lab, tq, err(z));
        printf("%-46s %9.0fx\n", "  speedup of the GPU", tq / tg);
        for (size_t k = 0; k < nn; k++) mpfr_clear(&F[k]);
        for (int i = 0; i < n; i++) mpfr_clear(&y[i]);
        mpfr_clears(t, s2, (mpfr_ptr)0);
    }
}
