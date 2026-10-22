// solve.cu -- a real job for the double-double GEMM: Ax = b with kappa(A) = 1e20, where FP64 has no correct digit.
// A = Q1 diag(sigma) Q2: Q1 the orthogonal DST-I, Q2 its rows permuted and sign-flipped, sigma from 1 down to 10^-e.
// Not fixable by row/column scaling: the small singular values are spread over every entry.
// One matrix, four solvers:
//   FP64 LU on the CPU; the same + iterative refinement with double-double residuals (the textbook rescue, needs kappa*u < 1);
//   blocked double-double LU: panels and U12 on the CPU, the trailing updates (2/3 n^3 of the work) on the INT8 tensor cores;
//   __float128 LU on every CPU core: the reference, and what one would run today to get this accuracy.
// Build: g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o &&
//        nvcc -O3 -arch=sm_120 -std=c++17 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off -o out_solve bench/solve.cu out_ref.o -lquadmath -lgomp
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <omp.h>
#include <cooperative_groups.h>
#include "../src/ozaki_dd.cuh"
void quad_dst(int n, double *Qh, double *Ql);
void quad_sigma(int n, double e, double *h, double *l);
double quad_solve(int n, const double *Ah, const double *Al, const double *bh, const double *bl, double *xh, double *xl);

// host double-double (Dekker / Bailey)
struct D { double hi, lo; };
static inline __host__ __device__ D two_sum(double a, double b) { double s = a + b, v = s - a; return {s, (a - (s - v)) + (b - v)}; }
static inline __host__ __device__ D qts(double a, double b) { double s = a + b; return {s, b - (s - a)}; }
static inline __host__ __device__ D add(D a, D b) {
    D s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
    s = qts(s.hi, s.lo + t.hi);
    return qts(s.hi, s.lo + t.lo);
}
static inline __host__ __device__ D sub(D a, D b) { return add(a, {-b.hi, -b.lo}); }
static inline __host__ __device__ D mul(D a, D b) { double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p); return qts(p, e + (a.hi * b.lo + a.lo * b.hi)); }
static inline __host__ __device__ D dvd(D a, D b) {
    double q1 = a.hi / b.hi; D r = sub(a, mul(b, {q1, 0}));
    double q2 = r.hi / b.hi; r = sub(r, mul(b, {q2, 0}));
    return add(qts(q1, q2), {r.hi / b.hi, 0});
}

#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// C = A * B, all double-double on the GPU (building the matrix, and the hybrid solver's updates), in row chunks of A so that
// n = 8192 fits: the residue planes of a whole 8192^2 C alone are 8.6 GB.
struct Dev { double *p[6] = {}; size_t ca = 0, cb = 0, cc = 0; };
static void gpu_dd(Dev &g, int M, int N, int K, const double *Ah, const double *Al, const double *Bh, const double *Bl,
                   double *Ch, double *Cl, double &tg) {
    int R = (size_t)M * N <= (size_t)4096 * 4096 ? M : std::max(128, (int)((size_t)4096 * 4096 / N) / 128 * 128);
    size_t na = (size_t)R * K, nbb = (size_t)K * N, nc = (size_t)R * N;
    if (na > g.ca) { for (int i : {0, 1}) { cudaFree(g.p[i]); CK(cudaMalloc(&g.p[i], na * 8)); } g.ca = na; }
    if (nbb > g.cb) { for (int i : {2, 3}) { cudaFree(g.p[i]); CK(cudaMalloc(&g.p[i], nbb * 8)); } g.cb = nbb; }
    if (nc > g.cc) { for (int i : {4, 5}) { cudaFree(g.p[i]); CK(cudaMalloc(&g.p[i], nc * 8)); } g.cc = nc; }
    double t0 = now();
    CK(cudaMemcpy(g.p[2], Bh, nbb * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(g.p[3], Bl, nbb * 8, cudaMemcpyHostToDevice));
    oz::Plan p = oz::plan(R, N, K, 32, 106);
    for (int r0 = 0; r0 < M; r0 += R) {
        int rr = std::min(R, M - r0);
        CK(cudaMemcpy(g.p[0], Ah + (size_t)r0 * K, (size_t)rr * K * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(g.p[1], Al + (size_t)r0 * K, (size_t)rr * K * 8, cudaMemcpyHostToDevice));
        p.M = rr; p.Mp = (rr + oz::BM - 1) / oz::BM * oz::BM;
        oz::ddgemm_t<32>(p, g.p[0], g.p[1], g.p[2], g.p[3], g.p[4], g.p[5]);
        CK(cudaGetLastError());
        CK(cudaMemcpy(Ch + (size_t)r0 * N, g.p[4], (size_t)rr * N * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(Cl + (size_t)r0 * N, g.p[5], (size_t)rr * N * 8, cudaMemcpyDeviceToHost));
    }
    oz::release(p);
    tg += now() - t0;
}

// blocked right-looking LU with partial pivoting in double-double, row-major in place (H + L); piv[j] = row swapped with j
static void lu_dd(int n, int nb, std::vector<double> &H, std::vector<double> &L, std::vector<int> &piv, double &tg) {
    auto at = [n](int i, int j) { return (size_t)i * n + j; };
    auto get = [&](int i, int j) { return D{H[at(i, j)], L[at(i, j)]}; };
    auto put = [&](int i, int j, D v) { H[at(i, j)] = v.hi; L[at(i, j)] = v.lo; };
    Dev g;
    std::vector<double> Lh, Ll, Uh, Ul, Ph, Pl;
    for (int k0 = 0; k0 < n; k0 += nb) {
        int kb = std::min(nb, n - k0), e = k0 + kb, m = n - e;
        for (int j = k0; j < e; j++) {   // panel
            int p = j;
            for (int i = j + 1; i < n; i++) if (fabs(H[at(i, j)]) > fabs(H[at(p, j)])) p = i;
            piv[j] = p;
            if (p != j) for (int c = 0; c < n; c++) { std::swap(H[at(j, c)], H[at(p, c)]); std::swap(L[at(j, c)], L[at(p, c)]); }
            D d = get(j, j);
#pragma omp parallel for schedule(static)
            for (int i = j + 1; i < n; i++) {
                D l = dvd(get(i, j), d);
                put(i, j, l);
                for (int c = j + 1; c < e; c++) put(i, c, sub(get(i, c), mul(l, get(j, c))));
            }
        }
        if (!m) break;
#pragma omp parallel for schedule(static)
        for (int c = e; c < n; c++)   // U12 = L11^-1 A12
            for (int r = k0 + 1; r < e; r++) {
                D s = get(r, c);
                for (int t = k0; t < r; t++) s = sub(s, mul(get(r, t), get(t, c)));
                put(r, c, s);
            }
        Lh.resize((size_t)m * kb); Ll.resize(Lh.size()); Uh.resize((size_t)kb * m); Ul.resize(Uh.size());
        Ph.resize((size_t)m * m); Pl.resize(Ph.size());
        for (int i = 0; i < m; i++) for (int t = 0; t < kb; t++) { Lh[(size_t)i * kb + t] = H[at(e + i, k0 + t)]; Ll[(size_t)i * kb + t] = L[at(e + i, k0 + t)]; }
        for (int t = 0; t < kb; t++) for (int c = 0; c < m; c++) { Uh[(size_t)t * m + c] = H[at(k0 + t, e + c)]; Ul[(size_t)t * m + c] = L[at(k0 + t, e + c)]; }
        gpu_dd(g, m, m, kb, Lh.data(), Ll.data(), Uh.data(), Ul.data(), Ph.data(), Pl.data(), tg);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < m; i++)
            for (int c = 0; c < m; c++) put(e + i, e + c, sub(get(e + i, e + c), D{Ph[(size_t)i * m + c], Pl[(size_t)i * m + c]}));
    }
    for (auto x : g.p) cudaFree(x);
}

// The same blocked LU with A resident on the GPU: panel, U12 and the solves as double-double kernels on the FP64 units,
// the trailing update on the INT8 tensor cores. Nothing but A, b in and x out crosses PCIe.
#define AT(i, j) ((size_t)(i) * n + (j))
__global__ void k_pivot(const double *H, int n, int j, int *piv) {   // one block: first row of max |a_ij|, i >= j
    __shared__ double bv[1024]; __shared__ int bi[1024];
    double v = -1; int ix = j;
    for (int i = j + threadIdx.x; i < n; i += blockDim.x) { double a = fabs(H[AT(i, j)]); if (a > v) { v = a; ix = i; } }
    bv[threadIdx.x] = v; bi[threadIdx.x] = ix; __syncthreads();
    for (int s = blockDim.x / 2; s; s >>= 1) {
        if (threadIdx.x < s) {
            double w = bv[threadIdx.x + s]; int k = bi[threadIdx.x + s];
            if (w > bv[threadIdx.x] || (w == bv[threadIdx.x] && k < bi[threadIdx.x])) { bv[threadIdx.x] = w; bi[threadIdx.x] = k; }
        }
        __syncthreads();
    }
    if (!threadIdx.x) piv[j] = bi[0];
}
__global__ void k_swap(double *H, double *L, int n, int j, const int *piv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x, p = piv[j];
    if (c >= n || p == j) return;
    double t = H[AT(j, c)]; H[AT(j, c)] = H[AT(p, c)]; H[AT(p, c)] = t;
    t = L[AT(j, c)]; L[AT(j, c)] = L[AT(p, c)]; L[AT(p, c)] = t;
}
__global__ void k_col(double *H, double *L, int n, int j) {   // l_ij = a_ij / a_jj
    int i = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    D l = dvd({H[AT(i, j)], L[AT(i, j)]}, {H[AT(j, j)], L[AT(j, j)]});
    H[AT(i, j)] = l.hi; L[AT(i, j)] = l.lo;
}
__global__ void k_rank1(double *H, double *L, int n, int j, int e) {   // a_ic -= l_ij u_jc inside the panel
    int c = j + 1 + blockIdx.x * blockDim.x + threadIdx.x, i = j + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= e || i >= n) return;
    D v = sub({H[AT(i, c)], L[AT(i, c)]}, mul({H[AT(i, j)], L[AT(i, j)]}, {H[AT(j, c)], L[AT(j, c)]}));
    H[AT(i, c)] = v.hi; L[AT(i, c)] = v.lo;
}
// L11^-1 (unit lower, kb x kb, row-major in Ih/Il): one block per column c of I, one thread per row, x in shared memory.
// Done once per panel, it turns U12 = L11^-1 A12 into a double-double GEMM on the INT8 tensor cores.
__global__ void k_inv(const double *H, const double *L, int n, int k0, int kb, double *Ih, double *Il, int ld) {
    __shared__ double xh[1024], xl[1024];
    int c = blockIdx.x, r = threadIdx.x;
    xh[r] = r == c; xl[r] = 0;
    __syncthreads();
    for (int t = c; t < kb - 1; t++) {
        if (r > t) { D v = sub({xh[r], xl[r]}, mul({H[AT(k0 + r, k0 + t)], L[AT(k0 + r, k0 + t)]}, {xh[t], xl[t]})); xh[r] = v.hi; xl[r] = v.lo; }
        __syncthreads();
    }
    Ih[(size_t)r * ld + c] = xh[r]; Il[(size_t)r * ld + c] = xl[r];
}
__global__ void k_trsm(double *H, double *L, int n, int k0, int e) {   // U12 = L11^-1 A12, one thread per column
    int c = e + blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int r = k0 + 1; r < e; r++) {
        D s{H[AT(r, c)], L[AT(r, c)]};
        for (int t = k0; t < r; t++) s = sub(s, mul({H[AT(r, t)], L[AT(r, t)]}, {H[AT(t, c)], L[AT(t, c)]}));
        H[AT(r, c)] = s.hi; L[AT(r, c)] = s.lo;
    }
}
__global__ void k_sub(double *H, double *L, int n, int e, int m, const double *Ph, const double *Pl) {   // A22 -= L21 U12
    int c = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (c >= m) return;
    size_t a = AT(e + i, e + c), q = (size_t)i * m + c;
    D v = sub({H[a], L[a]}, {Ph[q], Pl[q]});
    H[a] = v.hi; L[a] = v.lo;
}
__global__ void k_solve(const double *H, const double *L, int n, const int *piv, double *xh, double *xl) {   // one block
    if (!threadIdx.x)
        for (int j = 0; j < n; j++) { int p = piv[j]; double t = xh[j]; xh[j] = xh[p]; xh[p] = t; t = xl[j]; xl[j] = xl[p]; xl[p] = t; }
    __syncthreads();
    for (int t = 0; t < n; t++) {   // L y = P b, unit diagonal
        D y{xh[t], xl[t]};
        for (int i = t + 1 + threadIdx.x; i < n; i += blockDim.x) {
            D v = sub({xh[i], xl[i]}, mul({H[AT(i, t)], L[AT(i, t)]}, y)); xh[i] = v.hi; xl[i] = v.lo;
        }
        __syncthreads();
    }
    for (int t = n - 1; t >= 0; t--) {   // U x = y
        if (!threadIdx.x) { D v = dvd({xh[t], xl[t]}, {H[AT(t, t)], L[AT(t, t)]}); xh[t] = v.hi; xl[t] = v.lo; }
        __syncthreads();
        D y{xh[t], xl[t]};
        for (int i = threadIdx.x; i < t; i += blockDim.x) {
            D v = sub({xh[i], xl[i]}, mul({H[AT(i, t)], L[AT(i, t)]}, y)); xh[i] = v.hi; xl[i] = v.lo;
        }
        __syncthreads();
    }
}
// The blocked solve: the diagonal nb x nb block in shared memory (one block), the rest as a GEMV, one warp per row
// reading its row of the factor in order. k_solve reads columns with stride n from one SM and takes 0.13 s at n=4096.
__global__ void k_perm(int n, const int *piv, double *xh, double *xl) {
    for (int j = 0; j < n; j++) { int p = piv[j]; double t = xh[j]; xh[j] = xh[p]; xh[p] = t; t = xl[j]; xl[j] = xl[p]; xl[p] = t; }
}
// Left-looking: before block [r0, r0+bb) is solved, each of its rows takes its whole dot product with the solved part
// (columns c0..c1), a block of threads per row. The right-looking k_gemv paid a double-double shuffle reduction per row
// per block -- n^2/64 reductions, 73 ms at n = 8192 -- where this pays n.
__global__ void k_gemvl(const double *H, const double *L, int n, int r0, int c0, int c1, double *Xh, double *Xl) {
    __shared__ double wh[32], wl[32];
    int i = r0 + blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    D s{0, 0};
    for (int t = c0 + threadIdx.x; t < c1; t += blockDim.x) s = add(s, mul({H[AT(i, t)], L[AT(i, t)]}, {Xh[t], Xl[t]}));
    for (int o = 16; o; o >>= 1) s = add(s, {__shfl_down_sync(~0u, s.hi, o), __shfl_down_sync(~0u, s.lo, o)});
    if (!lane) { wh[w] = s.hi; wl[w] = s.lo; }
    __syncthreads();
    if (!threadIdx.x) {
        for (int k = 1; k < (int)(blockDim.x >> 5); k++) s = add(s, {wh[k], wl[k]});
        D v = sub({Xh[i], Xl[i]}, s); Xh[i] = v.hi; Xl[i] = v.lo;
    }
}
// The diagonal blocks of the solves as explicit inverses, all built at once after the factorization: CUDA block
// (b, c) solves T_b x = e_c by substitution (T_b = unit lower or upper diagonal block b), so the per-solve work is a
// parallel mat-vec. k_diag did the substitution inside every solve: 64 dependent dd steps, 180 us a block (n = 8192:
// 46 ms a solve). The refinement step absorbs what an explicit inverse costs in accuracy.
__global__ void k_invdiag(const double *H, const double *L, int n, int sb, bool upper, double *Dh, double *Dl) {
    __shared__ double xh[64], xl[64];
    int b0 = blockIdx.x * sb, bb = min(sb, n - b0), c = blockIdx.y, r = threadIdx.x;
    if (c >= bb) return;
    if (r < bb) { xh[r] = r == c; xl[r] = 0; }
    __syncthreads();
    for (int s = 0; s < bb; s++) {
        int t = upper ? bb - 1 - s : s;
        if (upper && r == t) { D v = dvd({xh[t], xl[t]}, {H[AT(b0 + t, b0 + t)], L[AT(b0 + t, b0 + t)]}); xh[t] = v.hi; xl[t] = v.lo; }
        __syncthreads();
        if (r < bb && (upper ? r < t : r > t)) {
            D v = sub({xh[r], xl[r]}, mul({H[AT(b0 + r, b0 + t)], L[AT(b0 + r, b0 + t)]}, {xh[t], xl[t]})); xh[r] = v.hi; xl[r] = v.lo;
        }
        __syncthreads();
    }
    if (r < bb) { Dh[(size_t)(b0 + r) * sb + c] = xh[r]; Dl[(size_t)(b0 + r) * sb + c] = xl[r]; }
}
__global__ void k_dmv(int b0, int bb, int sb, const double *Dh, const double *Dl, double *Xh, double *Xl) {   // x_b = inv(T_b) x_b
    __shared__ double sh[64], sl[64];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (threadIdx.x < bb) { sh[threadIdx.x] = Xh[b0 + threadIdx.x]; sl[threadIdx.x] = Xl[b0 + threadIdx.x]; }
    __syncthreads();
    for (int i = w; i < bb; i += nw) {
        D s{0, 0};
        for (int t = lane; t < bb; t += 32) s = add(s, mul({Dh[(size_t)(b0 + i) * sb + t], Dl[(size_t)(b0 + i) * sb + t]}, {sh[t], sl[t]}));
        for (int o = 16; o; o >>= 1) s = add(s, {__shfl_down_sync(~0u, s.hi, o), __shfl_down_sync(~0u, s.lo, o)});
        if (!lane) { Xh[b0 + i] = s.hi; Xl[b0 + i] = s.lo; }
    }
}
// The row interchanges as one permutation, built once per factorization (the swaps one by one in shared memory), then a
// parallel gather per solve. k_perm did the n swaps on the vector in global memory from one thread: 8 ms at n = 8192.
__global__ void k_permidx(int n, const int *piv, int *P) {
    extern __shared__ int ix[];
    for (int i = threadIdx.x; i < n; i += blockDim.x) ix[i] = i;
    __syncthreads();
    if (!threadIdx.x) for (int j = 0; j < n; j++) { int p = piv[j], t = ix[j]; ix[j] = ix[p]; ix[p] = t; }
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) P[i] = ix[i];
}
__global__ void k_gather(int n, const int *P, const double *Xh, const double *Xl, double *Yh, double *Yl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { Yh[i] = Xh[P[i]]; Yl[i] = Xl[P[i]]; }
}
// The diagonal block of a triangular solve, staged in shared memory (rows padded by one against bank conflicts): read
// from global, each step was a column with stride n, one L2 trip per step.
__global__ void k_diag(const double *H, const double *L, int n, int b0, int bb, bool upper, double *Xh, double *Xl) {
    extern __shared__ double sm[];
    const int ld = bb + 1;
    double *ah = sm, *al = sm + bb * ld, *xh = al + bb * ld, *xl = xh + bb;
    int r = threadIdx.x;
    for (int k = r; k < bb * bb; k += bb) { int i = k / bb, c = k % bb; ah[i * ld + c] = H[AT(b0 + i, b0 + c)]; al[i * ld + c] = L[AT(b0 + i, b0 + c)]; }
    xh[r] = Xh[b0 + r]; xl[r] = Xl[b0 + r];
    __syncthreads();
    for (int s = 0; s < bb; s++) {
        int t = upper ? bb - 1 - s : s;
        if (upper && r == t) { D v = dvd({xh[t], xl[t]}, {ah[t * ld + t], al[t * ld + t]}); xh[t] = v.hi; xl[t] = v.lo; }
        __syncthreads();
        if (upper ? r < t : r > t) {
            D v = sub({xh[r], xl[r]}, mul({ah[r * ld + t], al[r * ld + t]}, {xh[t], xl[t]})); xh[r] = v.hi; xl[r] = v.lo;
        }
        __syncthreads();
    }
    Xh[b0 + r] = xh[r]; Xl[b0 + r] = xl[r];
}
__global__ void k_gemv(const double *H, const double *L, int n, int r0, int r1, int b0, int bb, double *Xh, double *Xl) {
    int i = r0 + blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;   // x_i -= A(i, b0:b0+bb) x(b0:)
    if (i >= r1) return;
    D s{0, 0};
    for (int t = lane; t < bb; t += 32) s = add(s, mul({H[AT(i, b0 + t)], L[AT(i, b0 + t)]}, {Xh[b0 + t], Xl[b0 + t]}));
    for (int o = 16; o; o >>= 1) s = add(s, {__shfl_down_sync(~0u, s.hi, o), __shfl_down_sync(~0u, s.lo, o)});
    if (!lane) { D v = sub({Xh[i], Xl[i]}, s); Xh[i] = v.hi; Xl[i] = v.lo; }
}

// The recursive panel. The column-at-a-time panel above costs 4 launches per column plus a rank-1 update on the FP64 units
// that grows with nb (0.33 s of 0.70 at n=4096, nb=256; 0.88 s at nb=1024). Here the panel is split in halves down to
// 32-column leaves; the halves are joined by double-double GEMMs on the INT8 tensor cores, so nb can grow to 1024 and the
// trailing updates get cheaper too (the Ozaki GEMM costs ~2.5 ns per output element, nearly independent of K).
// Leaf: columns j0..j0+wl (wl <= 32), rows j0..n-1, one cooperative launch, one grid sync per column. A thread owns one row
// and keeps its 32 entries in registers for the whole leaf: the first version (a warp per row, a lane per column) spent
// 56 us per column on FP64 issue -- every lane ran the dd division and the idle lanes left of the pivot still paid.
// No row moves in memory: on a swap the owner of row j takes the name p (its content is what belongs at p), the owner of
// the pivot row takes the name j and retires; every thread writes its registers to the row it is named after at the end.
// Each block publishes its best pivot candidate with the whole row, double-buffered by column parity.
#ifdef LEAFPROF
__device__ unsigned long long g_lp[8];
#define LP(k) if (!blockIdx.x && !threadIdx.x) { long long t_ = clock64(); g_lp[k] += t_ - lp0; lp0 = t_; }
#else
#define LP(k)
#endif
__global__ void __launch_bounds__(256) k_leaf(double *H, double *L, int n, int j0, int wl, int *piv, double *bv, int *bi,
                                              double *ch, double *cl) {
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    __shared__ double sv[8], uh[32], ul[32], rih, ril; __shared__ int si[8], sp;
    const int G = gridDim.x, lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    int row = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = row < n;
    bool act = valid;
    double h[32], l[32];
#pragma unroll
    for (int c = 0; c < 32; c++) { h[c] = act && c < wl ? H[AT(row, j0 + c)] : 0; l[c] = act && c < wl ? L[AT(row, j0 + c)] : 0; }
#ifdef LEAFPROF
    long long lp0 = clock64();
#endif
    for (int j = 0; j < wl; j++) {   // not unrolled (grid.sync); h, l are indexed by compile-time c only, or they go to the stack
        const int cj = j0 + j, par = j & 1;
        double hj = 0;
#pragma unroll
        for (int c = 0; c < 32; c++) if (c == j) hj = h[c];
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
            double *qh = ch + ((size_t)par * G + blockIdx.x) * 32, *ql = cl + ((size_t)par * G + blockIdx.x) * 32;
#pragma unroll
            for (int c = 0; c < 32; c++) { qh[c] = h[c]; ql[c] = l[c]; }
        }
        LP(0)
        grid.sync();
        LP(1)
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
            uh[lane] = ch[((size_t)par * G + bb) * 32 + lane]; ul[lane] = cl[((size_t)par * G + bb) * 32 + lane];
            __syncwarp();
            if (!lane) { sp = bii; D r = dvd({1, 0}, {uh[j], ul[j]}); rih = r.hi; ril = r.lo; }
        }
        __syncthreads();
        LP(2)
        const int p = sp;
        if (blockIdx.x == 0 && threadIdx.x == 0) piv[cj] = p;
        if (act) {
            if (row == cj) { if (p == cj) act = false; else row = p; }
            else if (row == p) { act = false; row = cj; }
        }
        if (act) {
            D m{0, 0};
#pragma unroll
            for (int c = 0; c < 32; c++)
                if (c == j) {   // a / d as a * (1/d) plus one Newton step: dd division accuracy for 2 multiplies
                    D a{h[c], l[c]}, ri{rih, ril};
                    m = mul(a, ri); m = add(m, mul(sub(a, mul(m, {uh[c], ul[c]})), ri)); h[c] = m.hi; l[c] = m.lo;
                }
                else if (c > j) { D x = sub({h[c], l[c]}, mul(m, {uh[c], ul[c]})); h[c] = x.hi; l[c] = x.lo; }
        }
        __syncthreads();   // uh, sp are rewritten by the next column
        LP(3)
    }
    if (valid)
#pragma unroll
        for (int c = 0; c < 32; c++) if (c < wl) { H[AT(row, j0 + c)] = h[c]; L[AT(row, j0 + c)] = l[c]; }
}
__global__ void k_laswp(double *H, double *L, int n, int j0, int w, const int *piv) {   // the leaf's swaps on all other columns
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n || (c >= j0 && c < j0 + w)) return;
    for (int j = j0; j < j0 + w; j++) {
        int p = piv[j];
        if (p == j) continue;
        double t = H[AT(j, c)]; H[AT(j, c)] = H[AT(p, c)]; H[AT(p, c)] = t;
        t = L[AT(j, c)]; L[AT(j, c)] = L[AT(p, c)]; L[AT(p, c)] = t;
    }
}
#undef AT

// C = A B in double-double on the tensor cores, A (M x K) and B (K x N) anywhere with leading dimensions; C contiguous in s.Ch, s.Cl.
// One set of buffers for every shape: the plan is re-dimensioned per call, the K padding cleared when K is not a tile multiple.
// The 32 int32 residue planes of C are the big buffer (8.6 GB for n = 8192); they get what the card has left, and a C wider
// than that is done in column chunks.
struct Mm { oz::Plan p; double *Ch, *Cl; double log2P; size_t clcap; int ncap; };
static Mm mm_new(int n, int nb, size_t scratch) {   // scratch: elements of s.Ch, s.Cl
    Mm s;
    s.log2P = 0; for (int l = 0; l < 32; l++) s.log2P += log2((double)oz::MODULI[l]);
    CK(cudaMalloc(&s.Ch, scratch * 8)); CK(cudaMalloc(&s.Cl, scratch * 8));
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    int np = (n + oz::BN - 1) / oz::BN * oz::BN, kp = (nb + oz::BK - 1) / oz::BK * oz::BK;
    size_t left = fr - std::min(fr, ((size_t)32 * 2 * np * kp + ((size_t)768 << 20)));   // A, B residues + 768 MB slack
    s.ncap = (int)std::min<size_t>(np, left / ((size_t)32 * 4 * np) / oz::BN * oz::BN);
    if (s.ncap < oz::BN) { fprintf(stderr, "mm: no room for the residue planes (%zu MB free)\n", fr >> 20); exit(1); }
    s.p = oz::plan(n, s.ncap, nb, 32, 106);
    s.clcap = (size_t)np * s.ncap;
    if (getenv("OZ_PROF")) fprintf(stderr, "mm: residue planes for %d x %d (%zu MB)\n", np, s.ncap, s.clcap * 128 >> 20);
    return s;
}
static void mm_free(Mm &s) { oz::release(s.p); for (double *x : {s.Ch, s.Cl}) cudaFree(x); }
// C (mode 0) = A B, (1) -= A B, (2) = -A B in double-double on the tensor cores; A, B, C are blocks of larger matrices
// (leading dimensions lda, ldb, ldo), read and written in place -- C defaults to the scratch s.Ch, s.Cl with ldo = N.
static void mm(Mm &s, int M, int N, int K, const double *Ah, const double *Al, int lda, const double *Bh, const double *Bl, int ldb,
               double *Ch = nullptr, double *Cl = nullptr, int ldo = 0, int mode = 0) {
    if (!Ch) { Ch = s.Ch; Cl = s.Cl; ldo = N; }
    oz::Plan &p = s.p;
    p.M = M; p.K = K;
    p.Mp = (M + oz::BM - 1) / oz::BM * oz::BM; p.Kp = (K + oz::BK - 1) / oz::BK * oz::BK;
    p.bits = std::min(106, (int)floor((s.log2P - 2 - log2((double)K)) / 2));
    int nc = (int)std::min<size_t>(s.ncap, s.clcap / p.Mp / oz::BN * oz::BN);
    for (int c0 = 0; c0 < N; c0 += nc) {
        int w = std::min(nc, N - c0);
        p.N = w; p.Np = (w + oz::BN - 1) / oz::BN * oz::BN;
        if (K % oz::BK) { CK(cudaMemsetAsync(p.Ar, 0, (size_t)32 * p.Mp * p.Kp)); CK(cudaMemsetAsync(p.Br, 0, (size_t)32 * p.Np * p.Kp)); }
        oz::ddgemm_t<32>(p, Ah, Al, Bh + c0, Bl + c0, Ch + c0, Cl + c0, ldo, lda, ldb, mode);
    }
}

__global__ void k_resid(const double *H, const double *L, int n, double *Bh, double *Bl, const double *Xh, const double *Xl) {
    int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;   // b_i -= A(i, :) x, a warp per row
    if (i >= n) return;
    D s{0, 0};
    for (int t = lane; t < n; t += 32) s = add(s, mul({H[(size_t)i * n + t], L[(size_t)i * n + t]}, {Xh[t], Xl[t]}));
    for (int o = 16; o; o >>= 1) s = add(s, {__shfl_down_sync(~0u, s.hi, o), __shfl_down_sync(~0u, s.lo, o)});
    if (!lane) { D v = sub({Bh[i], Bl[i]}, s); Bh[i] = v.hi; Bl[i] = v.lo; }
}
__global__ void k_addv(int n, const double *Dh, const double *Dl, double *Xh, double *Xl) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { D v = add({Xh[i], Xl[i]}, {Dh[i], Dl[i]}); Xh[i] = v.hi; Xl[i] = v.lo; }
}

// Recursive LU of the panel A(j0.., j0..j0+w): left half, U12 = L11^-1 A12, A22 -= L21 U12, right half. It also leaves the
// inverse of its unit lower w x w block in r.Ih/Il (leading dimension r.nb, at offset j0 - k0 on the diagonal), assembled as
// inv([A 0; B C]) = [inv(A) 0; -inv(C) B inv(A) inv(C)]: the caller's U12 needs it, and k_inv on 1024 rows was 25 ms a panel.
struct Rec { int n, nb, G, T, sms; double tleaf; int *piv; double *bv, *ch, *cl, *Ih, *Il; int *bi; Mm *s; };
static void rec(Rec &r, double *H, double *L, int k0, int j0, int w) {
    int n = r.n, nb = r.nb;
    size_t o = (size_t)(j0 - k0) * nb + (j0 - k0);   // this block's diagonal place in the inverse
    if (w <= 32) {
        void *args[] = {&H, &L, &n, &j0, &w, &r.piv, &r.bv, &r.bi, &r.ch, &r.cl};
        static const bool prof = getenv("OZ_PROF");
        double t0 = prof ? (cudaDeviceSynchronize(), now()) : 0;
        int T = std::min(256, std::max(32, (n - j0 + 32 * r.sms - 1) / (32 * r.sms) * 32));   // a thread per row, the rows over every SM
        CK(cudaLaunchCooperativeKernel((void *)k_leaf, (n - j0 + T - 1) / T, T, args));
        if (prof) { cudaDeviceSynchronize(); r.tleaf += now() - t0; }
        k_laswp<<<(n + 255) / 256, 256>>>(H, L, n, j0, w, r.piv);
        k_inv<<<w, w>>>(H, L, n, j0, w, r.Ih + o, r.Il + o, nb);
        return;
    }
    int w1 = std::max(32, w / 64 * 32), w2 = w - w1, c1 = j0 + w1;
    Mm &s = *r.s;
    double *Iah = r.Ih + o, *Ial = r.Il + o, *Ich = Iah + (size_t)w1 * nb + w1, *Icl = Ial + (size_t)w1 * nb + w1;
    double *Uh = H + (size_t)j0 * n + c1, *Ul = L + (size_t)j0 * n + c1, *Bh = H + (size_t)c1 * n + j0, *Bl = L + (size_t)c1 * n + j0;
    rec(r, H, L, k0, j0, w1);
    mm(s, w1, w2, w1, Iah, Ial, nb, Uh, Ul, n, Uh, Ul, n);   // U12 in place
    if (c1 < n) mm(s, n - c1, w2, w1, Bh, Bl, n, Uh, Ul, n, H + (size_t)c1 * n + c1, L + (size_t)c1 * n + c1, n, 1);   // A22 -= L21 U12
    rec(r, H, L, k0, c1, w2);
    mm(s, w2, w1, w1, Bh, Bl, n, Iah, Ial, nb);   // B inv(A), B as permuted by the right half's swaps
    mm(s, w2, w1, w2, Ich, Icl, nb, s.Ch, s.Cl, w1, Iah + (size_t)w1 * nb, Ial + (size_t)w1 * nb, nb, 2);   // -inv(C) B inv(A)
}

// x = A^-1 b, everything on the GPU; returns seconds from A on the host to x on the host, tg = the INT8 GEMM part
static double solve_gpu(int n, int nb, const double *Ah, const double *Al, const double *bh, const double *bl,
                        double *xh, double *xl, double &tg) {
    static const bool flat = getenv("OZ_FLAT");           // OZ_FLAT=1: the column-at-a-time panel (22-10 baseline)
    double *H, *L, *Lh = 0, *Ll = 0, *Uh = 0, *Ul = 0, *Ph = 0, *Pl = 0, *Xh, *Xl; int *piv;
    size_t nn = (size_t)n * n, mk = (size_t)n * nb, mm2 = (size_t)(n - nb) * (n - nb);
    CK(cudaMalloc(&H, nn * 8)); CK(cudaMalloc(&L, nn * 8));
    CK(cudaMalloc(&Xh, n * 8)); CK(cudaMalloc(&Xl, n * 8)); CK(cudaMalloc(&piv, n * 4));
    oz::Plan p{}, q{}; Mm s{}; Rec r{};
    double *Ih, *Il; CK(cudaMalloc(&Ih, (size_t)nb * nb * 8)); CK(cudaMalloc(&Il, (size_t)nb * nb * 8));
    static const bool refine = !getenv("OZ_NOREF");   // OZ_NOREF=1: the LU solve alone
    double *A0h = 0, *A0l = 0, *B0h = 0, *B0l = 0;
    if (refine) { CK(cudaMalloc(&A0h, nn * 8)); CK(cudaMalloc(&A0l, nn * 8)); CK(cudaMalloc(&B0h, n * 8)); CK(cudaMalloc(&B0l, n * 8)); }
    if (flat) {
        CK(cudaMalloc(&Lh, mk * 8)); CK(cudaMalloc(&Ll, mk * 8));
        CK(cudaMalloc(&Uh, mk * 8)); CK(cudaMalloc(&Ul, mk * 8)); CK(cudaMalloc(&Ph, mm2 * 8)); CK(cudaMalloc(&Pl, mm2 * 8));
        p = oz::plan(n - nb, n - nb, nb, 32, 106);   // buffers for the largest update, reused shrinking
        q = oz::plan(nb, n - nb, nb, 32, 106);       // U12 = L11^-1 A12
    } else {
        s = mm_new(n, nb, (size_t)nb * nb);   // the scratch only holds B inv(A) of the inverse assembly
        cudaDeviceProp dp; CK(cudaGetDeviceProperties(&dp, 0));
        r.T = getenv("OZ_T") ? atoi(getenv("OZ_T")) : 256;
        int occ; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_leaf, r.T, 0));
        r.n = n; r.G = getenv("OZ_G") ? atoi(getenv("OZ_G")) : dp.multiProcessorCount * occ;
        if ((long)r.G * r.T < n) { fprintf(stderr, "leaf: %d x %d threads < n=%d rows\n", r.G, r.T, n); exit(1); }
        if (getenv("OZ_PROF")) fprintf(stderr, "leaf grid %d x %d (%d SMs, occupancy %d)\n", r.G, r.T, dp.multiProcessorCount, occ);
        r.sms = dp.multiProcessorCount; r.nb = nb;
        r.piv = piv; r.Ih = Ih; r.Il = Il; r.s = &s;
        CK(cudaMalloc(&r.bv, r.G * 2 * 8)); CK(cudaMalloc(&r.bi, r.G * 2 * 4));
        CK(cudaMalloc(&r.ch, (size_t)r.G * 64 * 8)); CK(cudaMalloc(&r.cl, (size_t)r.G * 64 * 8));
    }
    static const bool trsm_ref = getenv("OZ_TRSM");       // OZ_TRSM=1: the column-by-column substitution instead
    cudaEvent_t g0, g1; cudaEventCreate(&g0); cudaEventCreate(&g1);
    CK(cudaDeviceSynchronize());
    double t0 = now();
    CK(cudaMemcpy(H, Ah, nn * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(L, Al, nn * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Xh, bh, n * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Xl, bl, n * 8, cudaMemcpyHostToDevice));
    if (refine) {
        CK(cudaMemcpyAsync(A0h, H, nn * 8, cudaMemcpyDeviceToDevice)); CK(cudaMemcpyAsync(A0l, L, nn * 8, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpyAsync(B0h, Xh, n * 8, cudaMemcpyDeviceToDevice)); CK(cudaMemcpyAsync(B0l, Xl, n * 8, cudaMemcpyDeviceToDevice));
    }
    float gms = 0;
    // OZ_PROF=1: seconds per phase, with a sync at every boundary -- only for finding where the time goes
    static const bool prof = getenv("OZ_PROF"); double ph[5] = {}, tp = now();
#define PH(k) if (prof) { cudaDeviceSynchronize(); double t_ = now(); ph[k] += t_ - tp; tp = t_; }
    for (int k0 = 0; k0 < n; k0 += nb) {
        int kb = std::min(nb, n - k0), e = k0 + kb, m = n - e;
        PH(0);
        if (!flat) {   // recursive panel, then U12 and the trailing update through the shared GEMM buffers
            CK(cudaMemsetAsync(Ih, 0, (size_t)nb * nb * 8)); CK(cudaMemsetAsync(Il, 0, (size_t)nb * nb * 8));   // inverse: upper part 0
            rec(r, H, L, k0, k0, kb);
            PH(1);
            if (!m) break;
            double *Uh = H + (size_t)k0 * n + e, *Ul = L + (size_t)k0 * n + e;
            mm(s, kb, m, kb, Ih, Il, nb, Uh, Ul, n, Uh, Ul, n);
            PH(2);
            cudaEventRecord(g0);
            mm(s, m, m, kb, H + (size_t)e * n + k0, L + (size_t)e * n + k0, n, Uh, Ul, n, H + (size_t)e * n + e, L + (size_t)e * n + e, n, 1);
            cudaEventRecord(g1); cudaEventSynchronize(g1);
            float ms; cudaEventElapsedTime(&ms, g0, g1); gms += ms;
            CK(cudaGetLastError());
            continue;
        }
        for (int j = k0; j < e; j++) {
            k_pivot<<<1, 1024>>>(H, n, j, piv);
            k_swap<<<(n + 255) / 256, 256>>>(H, L, n, j, piv);
            if (j + 1 < n) k_col<<<(n - j - 1 + 255) / 256, 256>>>(H, L, n, j);
            if (j + 1 < e) k_rank1<<<dim3((e - j - 1 + 31) / 32, (n - j - 1 + 7) / 8), dim3(32, 8)>>>(H, L, n, j, e);
        }
        PH(1);
        if (!m) break;
        if (trsm_ref) {
            k_trsm<<<(m + 127) / 128, 128>>>(H, L, n, k0, e);
            CK(cudaMemcpy2D(Uh, m * 8, H + (size_t)k0 * n + e, n * 8, m * 8, kb, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy2D(Ul, m * 8, L + (size_t)k0 * n + e, n * 8, m * 8, kb, cudaMemcpyDeviceToDevice));
        } else {   // U12 = L11^-1 A12 on the tensor cores, written back into A for the final solve
            k_inv<<<kb, kb>>>(H, L, n, k0, kb, Ih, Il, kb);
            CK(cudaMemcpy2D(Ph, m * 8, H + (size_t)k0 * n + e, n * 8, m * 8, kb, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy2D(Pl, m * 8, L + (size_t)k0 * n + e, n * 8, m * 8, kb, cudaMemcpyDeviceToDevice));
            q.N = m; q.Np = (m + oz::BN - 1) / oz::BN * oz::BN;
            oz::ddgemm_t<32>(q, Ih, Il, Ph, Pl, Uh, Ul);
            CK(cudaMemcpy2D(H + (size_t)k0 * n + e, n * 8, Uh, m * 8, m * 8, kb, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy2D(L + (size_t)k0 * n + e, n * 8, Ul, m * 8, m * 8, kb, cudaMemcpyDeviceToDevice));
        }
        PH(2);
        CK(cudaMemcpy2D(Lh, kb * 8, H + (size_t)e * n + k0, n * 8, kb * 8, m, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy2D(Ll, kb * 8, L + (size_t)e * n + k0, n * 8, kb * 8, m, cudaMemcpyDeviceToDevice));
        p.M = p.N = m; p.Mp = (m + oz::BM - 1) / oz::BM * oz::BM; p.Np = (m + oz::BN - 1) / oz::BN * oz::BN;
        cudaEventRecord(g0);
        oz::ddgemm_t<32>(p, Lh, Ll, Uh, Ul, Ph, Pl);
        cudaEventRecord(g1); cudaEventSynchronize(g1);
        float ms; cudaEventElapsedTime(&ms, g0, g1); gms += ms;
        k_sub<<<dim3((m + 255) / 256, m), 256>>>(H, L, n, e, m, Ph, Pl);
        CK(cudaGetLastError());
    }
    PH(0);
    static const int sb = getenv("OZ_SB") ? atoi(getenv("OZ_SB")) : std::min(nb, 64);   // block of the triangular solves
    const size_t dsm = (size_t)(2 * sb * (sb + 1) + 2 * sb) * 8;
    CK(cudaFuncSetAttribute(k_diag, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)dsm));
    int *Pm; double *Th, *Tl; CK(cudaMalloc(&Pm, n * 4)); CK(cudaMalloc(&Th, n * 8)); CK(cudaMalloc(&Tl, n * 8));
    k_permidx<<<1, 1024, n * 4>>>(n, piv, Pm);
    double *DLh, *DLl, *DUh, *DUl;
    for (double **x : {&DLh, &DLl, &DUh, &DUl}) CK(cudaMalloc(x, (size_t)n * sb * 8));
    if (sb > 64) { fprintf(stderr, "OZ_SB <= 64\n"); exit(1); }
    k_invdiag<<<dim3((n + sb - 1) / sb, sb), sb>>>(H, L, n, sb, false, DLh, DLl);
    k_invdiag<<<dim3((n + sb - 1) / sb, sb), sb>>>(H, L, n, sb, true, DUh, DUl);
    static double tsv[3] = {};   // OZ_PROF: gather, dot products, diagonal blocks
    static const bool prof2 = getenv("OZ_PROF") && atoi(getenv("OZ_PROF")) == 2;   // OZ_PROF=2: also sync around every solve kernel
    std::vector<std::pair<int, cudaEvent_t>> evs;   // GPU timestamps around each kernel: a host sync costs tens of us on WSL
    auto tk = [&](int k, auto f) {
        if (!prof2) { f(); return; }
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a); f(); cudaEventRecord(b); evs.push_back({k, a}); evs.push_back({k, b});
    };
    auto trisolve = [&](double *Yh, double *Yl) {
        if (getenv("OZ_SOLVE")) { k_solve<<<1, 1024>>>(H, L, n, piv, Yh, Yl); return; }   // OZ_SOLVE=1: the one-block column solve
        tk(0, [&] {
            k_gather<<<(n + 255) / 256, 256>>>(n, Pm, Yh, Yl, Th, Tl);
            CK(cudaMemcpyAsync(Yh, Th, n * 8, cudaMemcpyDeviceToDevice)); CK(cudaMemcpyAsync(Yl, Tl, n * 8, cudaMemcpyDeviceToDevice));
        });
        for (int b0 = 0; b0 < n; b0 += sb) {   // L y = P b
            int bb = std::min(sb, n - b0);
            if (b0) tk(1, [&] { k_gemvl<<<bb, 256>>>(H, L, n, b0, 0, b0, Yh, Yl); });
            tk(2, [&] { k_dmv<<<1, 1024>>>(b0, bb, sb, DLh, DLl, Yh, Yl); });
        }
        for (int b0 = (n - 1) / sb * sb; b0 >= 0; b0 -= sb) {   // U x = y
            int bb = std::min(sb, n - b0);
            if (b0 + bb < n) tk(1, [&] { k_gemvl<<<bb, 256>>>(H, L, n, b0, b0 + bb, n, Yh, Yl); });
            tk(2, [&] { k_dmv<<<1, 1024>>>(b0, bb, sb, DUh, DUl, Yh, Yl); });
        }
    };
    if (getenv("OZ_SOLVEBENCH")) {   // 10 solves back to back, one sync
        cudaDeviceSynchronize(); double t_ = now();
        for (int q_ = 0; q_ < 10; q_++) trisolve(B0h, B0l);
        cudaDeviceSynchronize(); fprintf(stderr, "  one solve: %.2f ms\n", (now() - t_) * 100);
        for (int q_ = 0; q_ < 3; q_++) { cudaDeviceSynchronize(); t_ = now(); k_gemvl<<<64, 256>>>(H, L, n, 4096, 0, 4096, B0h, B0l); cudaDeviceSynchronize(); fprintf(stderr, "  one k_gemvl 64 x 4096: %.1f us\n", (now() - t_) * 1e6); }
        for (int q_ = 0; q_ < 3; q_++) { cudaDeviceSynchronize(); t_ = now(); k_dmv<<<1, 1024>>>(4096, 64, sb, DLh, DLl, B0h, B0l); cudaDeviceSynchronize(); fprintf(stderr, "  one k_dmv: %.1f us\n", (now() - t_) * 1e6); }
        for (int q_ = 0; q_ < 3; q_++) { cudaDeviceSynchronize(); t_ = now(); k_addv<<<1, 32>>>(0, B0h, B0l, Xh, Xl); cudaDeviceSynchronize(); fprintf(stderr, "  empty-ish kernel + sync: %.1f us\n", (now() - t_) * 1e6); }
    }
    trisolve(Xh, Xl);
    PH(3);
    static const int steps = getenv("OZ_REF") ? atoi(getenv("OZ_REF")) : 1;
    for (int it = 0; refine && it < steps; it++) {   // refinement: r = b - A x with the original A in double-double, x += LU \ r
        if (it) { CK(cudaMemcpyAsync(B0h, bh, n * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpyAsync(B0l, bl, n * 8, cudaMemcpyHostToDevice)); }
        k_resid<<<(n + 7) / 8, 256>>>(A0h, A0l, n, B0h, B0l, Xh, Xl);
        trisolve(B0h, B0l);
        k_addv<<<(n + 255) / 256, 256>>>(n, B0h, B0l, Xh, Xl);
    }
    PH(4);
    if (prof) fprintf(stderr, "phases: copies+gemm+sub %.3f  panel %.3f  trsm %.3f  solve %.3f  refine %.3f s\n", ph[0], ph[1], ph[2], ph[3], ph[4]);
    if (prof && !flat) fprintf(stderr, "  of the panel, leaves %.3f s\n", r.tleaf);
    cudaDeviceSynchronize();
    for (size_t q_ = 0; q_ + 1 < evs.size(); q_ += 2) { float ms; cudaEventElapsedTime(&ms, evs[q_].second, evs[q_ + 1].second); tsv[evs[q_].first] += ms / 1e3; }
    if (prof2) fprintf(stderr, "  of the solves: gather %.3f  dot products %.3f  diagonal blocks %.3f s\n", tsv[0], tsv[1], tsv[2]);
#ifdef LEAFPROF
    { unsigned long long c[8]; cudaMemcpyFromSymbol(c, g_lp, sizeof c); fprintf(stderr, "  leaf block 0, ms at 3.09 GHz: publish %.1f  grid.sync %.1f  pick+recip %.1f  update %.1f\n", c[0] / 3.09e6, c[1] / 3.09e6, c[2] / 3.09e6, c[3] / 3.09e6); }
#endif
    CK(cudaMemcpy(xh, Xh, n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(xl, Xl, n * 8, cudaMemcpyDeviceToHost));
    double t = now() - t0;
    tg = gms / 1e3;
    if (flat) { p.M = p.N = n - nb; oz::release(p); q.N = n - nb; oz::release(q); }
    else { mm_free(s); for (void *x : {(void *)r.bv, (void *)r.bi, (void *)r.ch, (void *)r.cl}) cudaFree(x); }
    cudaFree(Ih); cudaFree(Il);
    for (double *q : {H, L, Lh, Ll, Uh, Ul, Ph, Pl, Xh, Xl, A0h, A0l, B0h, B0l, Th, Tl}) cudaFree(q);
    cudaFree(piv);
    cudaFree(Pm);
    for (double *x : {DLh, DLl, DUh, DUl}) cudaFree(x);
    return t;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048, nb = argc > 2 ? atoi(argv[2]) : 256;
    double ek = argc > 3 ? atof(argv[3]) : 20;
    int thr = omp_get_max_threads();
    if (getenv("OZ_MMBENCH")) {   // cost of one double-double GEMM by shape: n x n output, K from the list
        Mm s = mm_new(n, 1024, (size_t)n * n); double *x[4]; for (auto &p : x) CK(cudaMalloc(&p, (size_t)n * 1024 * 8));
        {   // random double-double entries of mixed sign and magnitude
            std::vector<double> h((size_t)n * 1024); std::mt19937_64 g(1); std::uniform_real_distribution<double> u(-1, 1);
            for (int t = 0; t < 4; t++) { for (auto &v : h) v = ldexp(u(g), t % 2 ? -53 : (int)(g() % 8)); CK(cudaMemcpy(x[t], h.data(), h.size() * 8, cudaMemcpyHostToDevice)); }
        }
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        if (atoi(getenv("OZ_MMBENCH")) == 2) {   // the kernels of one n x n x 256 GEMM, one by one
            oz::Plan &p = s.p; mm(s, n, s.ncap, 256, x[0], x[1], 256, x[2], x[3], s.ncap); int N = s.ncap;
            auto T = [&](const char *w, auto f) { f(); cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b); printf("%-12s %.3f ms\n", w, ms); };
            T("scales A", [&] { oz::scales_dd<<<p.M, 256>>>(x[0], p.K, 1, p.K, p.bits, p.sa); });
            T("residues A", [&] { oz::residues_dd<32><<<dim3((p.K + 255) / 256, p.M), 256>>>(x[0], x[1], p.K, p.K, 1, p.sa, p.Ar, p.Kp, (size_t)p.Mp * p.Kp); });
            T("scales B", [&] { oz::scales_dd<<<p.N, 256>>>(x[2], 1, p.N, p.K, p.bits, p.sb); });
            T("residues B", [&] { oz::residues_dd<32><<<dim3((p.K + 255) / 256, p.N), 256>>>(x[2], x[3], p.K, 1, p.N, p.sb, p.Br, p.Kp, (size_t)p.Np * p.Kp); });
            T("gemm_i8 x32", [&] { oz::gemm_i8<<<dim3(p.Np / oz::BN, p.Mp / oz::BM, 32), oz::THREADS>>>(p.Ar, p.Br, p.Cl, p.Mp, p.Np, p.Kp); });
            T("crt_dd", [&] { oz::crt_dd<32><<<dim3((p.N + 255) / 256, p.M), 256>>>(p.Cl, (size_t)p.Mp * p.Np, p.M, p.N, p.Np, p.sa, p.sb, s.Ch, s.Cl, N); });
            {   // C[0][0..2] against a long-double sum of the inputs
                std::vector<double> a(3), b(3), ah(p.K), al(p.K), bh((size_t)p.K * N), bl((size_t)p.K * N);
                cudaMemcpy(a.data(), s.Ch, 24, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), s.Cl, 24, cudaMemcpyDeviceToHost);
                cudaMemcpy(ah.data(), x[0], p.K * 8, cudaMemcpyDeviceToHost); cudaMemcpy(al.data(), x[1], p.K * 8, cudaMemcpyDeviceToHost);
                cudaMemcpy(bh.data(), x[2], bh.size() * 8, cudaMemcpyDeviceToHost); cudaMemcpy(bl.data(), x[3], bl.size() * 8, cudaMemcpyDeviceToHost);
                for (int k = 0; k < 3; k++) {
                    long double r = 0; for (int t = 0; t < p.K; t++) r += ((long double)ah[t] + al[t]) * ((long double)bh[(size_t)t * N + k] + bl[(size_t)t * N + k]);
                    printf("C[0][%d] = %.17g + %.3g, long double ref - hi = %.3Lg\n", k, a[k], b[k], r - (long double)a[k]);
                }
            }
            return 0;
        }
        for (int K : {32, 64, 128, 256, 512, 1024}) for (int N : {n, 512, 64}) {
            mm(s, n, N, K, x[0], x[1], K, x[2], x[3], N); cudaEventRecord(a);
            for (int r = 0; r < 5; r++) mm(s, n, N, K, x[0], x[1], K, x[2], x[3], N);
            cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
            printf("M=%d N=%d K=%d: %.3f ms, %.2f ns per output\n", n, N, K, ms / 5, ms / 5 * 1e6 / ((double)n * N));
        }
        return 0;
    }
    printf("Ax = b, n=%d, kappa_2(A) = 1e%.0f by construction (A = Q1 diag(sigma) Q2), nb=%d, %d CPU threads\n", n, ek, nb, thr);

    // A = (Q1 diag(sigma)) * Q2, in double-double, the product on the GPU
    std::vector<double> Qh((size_t)n * n), Ql(Qh.size()), sh(n), sl(n);
    quad_dst(n, Qh.data(), Ql.data()); quad_sigma(n, ek, sh.data(), sl.data());
    std::mt19937_64 g(argc > 4 ? atoll(argv[4]) : 7);   // seed: matrix permutation, signs and x_true
    std::vector<int> perm(n); for (int i = 0; i < n; i++) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), g);
    std::vector<double> Bh(Qh.size()), Bl(Qh.size()), Q2h(Qh.size()), Q2l(Qh.size()), Ah(Qh.size()), Al(Qh.size());
    for (int r = 0; r < n; r++)
        for (int i = 0; i < n; i++) {
            D v = mul({Qh[(size_t)r * n + i], Ql[(size_t)r * n + i]}, {sh[i], sl[i]});
            Bh[(size_t)r * n + i] = v.hi; Bl[(size_t)r * n + i] = v.lo;
            double s = (g() & 1) ? 1 : -1;   // row i of Q2 = +-row perm[i] of Q1: still orthogonal
            Q2h[(size_t)i * n + r] = s * Qh[(size_t)perm[i] * n + r]; Q2l[(size_t)i * n + r] = s * Ql[(size_t)perm[i] * n + r];
        }
    { Dev d; double t = 0; gpu_dd(d, n, n, n, Bh.data(), Bl.data(), Q2h.data(), Q2l.data(), Ah.data(), Al.data(), t); for (auto x : d.p) cudaFree(x); }

    // x_true in [-1, 1]; b = A x_true in double-double (b is then the data; the quad solve of (A, b) is the reference)
    std::uniform_real_distribution<double> u(-1, 1);
    std::vector<double> xt(n), bh(n), bl(n);
    for (auto &v : xt) v = u(g);
    for (int i = 0; i < n; i++) {
        D s{0, 0};
        for (int j = 0; j < n; j++) s = add(s, mul({Ah[(size_t)i * n + j], Al[(size_t)i * n + j]}, {xt[j], 0}));
        bh[i] = s.hi; bl[i] = s.lo;
    }

    if (getenv("OZ_PROF") || getenv("OZ_GPU")) {   // only the GPU solver, twice (the first run pays the module load); OZ_GPU=1: no phase syncs
        std::vector<double> xh(n), xl(n); double tg = 0;
        for (int r = 0; r < 2; r++)
            printf("all on GPU: %.3f s (INT8 GEMM %.3f s)\n", solve_gpu(n, nb, Ah.data(), Al.data(), bh.data(), bl.data(), xh.data(), xl.data(), tg), tg);
        return 0;
    }
    std::vector<double> rh(n), rl(n);
    double tq = quad_solve(n, Ah.data(), Al.data(), bh.data(), bl.data(), rh.data(), rl.data());
    double nr = 0, nt = 0; for (int i = 0; i < n; i++) { nr = fmax(nr, fabs(rh[i])); nt = fmax(nt, fabs(xt[i])); }
    auto err_t = [&](const double *h, const double *l) { double w = 0; for (int i = 0; i < n; i++) w = fmax(w, fabs((h[i] - xt[i]) + (l ? l[i] : 0))); return w / nt; };
    auto err_q = [&](const double *h, const double *l) { double w = 0; for (int i = 0; i < n; i++) w = fmax(w, fabs((h[i] - rh[i]) + ((l ? l[i] : 0) - rl[i]))); return w / nr; };
    printf("%-44s %9s %15s %15s\n", "solver", "seconds", "err vs x_true", "err vs quad");

    static const bool skip = getenv("OZ_SKIPCPU");   // OZ_SKIPCPU=1: only the GPU solver against __float128 (large n)
    if (!skip) {
        std::vector<double> A(Ah); std::vector<int> piv(n);
        double t0 = now();
        for (int j = 0; j < n; j++) {
            int p = j;
            for (int i = j + 1; i < n; i++) if (fabs(A[(size_t)i * n + j]) > fabs(A[(size_t)p * n + j])) p = i;
            piv[j] = p;
            if (p != j) for (int c = 0; c < n; c++) std::swap(A[(size_t)j * n + c], A[(size_t)p * n + c]);
            double d = A[(size_t)j * n + j];
#pragma omp parallel for schedule(static)
            for (int i = j + 1; i < n; i++) {
                double l = A[(size_t)i * n + j] / d;
                A[(size_t)i * n + j] = l;
                for (int c = j + 1; c < n; c++) A[(size_t)i * n + c] -= l * A[(size_t)j * n + c];
            }
        }
        auto solve = [&](std::vector<double> &x) {
            for (int j = 0; j < n; j++) std::swap(x[j], x[piv[j]]);
            for (int i = 0; i < n; i++) for (int t = 0; t < i; t++) x[i] -= A[(size_t)i * n + t] * x[t];
            for (int i = n - 1; i >= 0; i--) { double s = x[i]; for (int t = i + 1; t < n; t++) s -= A[(size_t)i * n + t] * x[t]; x[i] = s / A[(size_t)i * n + i]; }
        };
        std::vector<double> xh(bh), xl(n, 0.0);
        solve(xh);
        double t64 = now() - t0;
        printf("%-44s %9.2f %15.3g %15.3g\n", "FP64 LU (CPU)", t64, err_t(xh.data(), nullptr), err_q(xh.data(), nullptr));
        double best = 1e300; int bi = 0;
        for (int it = 1; it <= 10; it++) {
            std::vector<double> r(n);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                D s{bh[i], bl[i]};
                for (int j = 0; j < n; j++) s = sub(s, mul({Ah[(size_t)i * n + j], Al[(size_t)i * n + j]}, {xh[j], xl[j]}));
                r[i] = s.hi;
            }
            solve(r);
            for (int i = 0; i < n; i++) { D v = add({xh[i], xl[i]}, {r[i], 0}); xh[i] = v.hi; xl[i] = v.lo; }
            double e = err_q(xh.data(), xl.data());
            if (e < best) { best = e; bi = it; }
        }
        char lab[64]; snprintf(lab, sizeof lab, "  + 10 refinement steps, dd residual (best@%d)", bi);
        printf("%-44s %9s %15.3g %15.3g\n", lab, "", err_t(xh.data(), xl.data()), best);
    }

    // double-double LU, trailing updates on the INT8 tensor cores
    if (!skip) {
        std::vector<double> H(Ah), L(Al); std::vector<int> piv(n);
        double tg = 0, t0 = now();
        lu_dd(n, nb, H, L, piv, tg);
        std::vector<D> x(n);
        for (int i = 0; i < n; i++) x[i] = {bh[i], bl[i]};
        for (int j = 0; j < n; j++) std::swap(x[j], x[piv[j]]);
        for (int i = 0; i < n; i++) for (int t = 0; t < i; t++) x[i] = sub(x[i], mul({H[(size_t)i * n + t], L[(size_t)i * n + t]}, x[t]));
        for (int i = n - 1; i >= 0; i--) {
            D s = x[i];
            for (int t = i + 1; t < n; t++) s = sub(s, mul({H[(size_t)i * n + t], L[(size_t)i * n + t]}, x[t]));
            x[i] = dvd(s, {H[(size_t)i * n + i], L[(size_t)i * n + i]});
        }
        double tdd = now() - t0;
        std::vector<double> xh(n), xl(n); for (int i = 0; i < n; i++) { xh[i] = x[i].hi; xl[i] = x[i].lo; }
        char lab[64]; snprintf(lab, sizeof lab, "double-double LU, INT8 TC (GPU part %.2f s)", tg);
        printf("%-44s %9.2f %15.3g %15.3g\n", lab, tdd, err_t(xh.data(), xl.data()), err_q(xh.data(), xl.data()));
    }
    {
        std::vector<double> xh(n), xl(n); double tg = 0;
        solve_gpu(n, nb, Ah.data(), Al.data(), bh.data(), bl.data(), xh.data(), xl.data(), tg);   // warm-up: clocks idle during the CPU solvers
        double tdd = solve_gpu(n, nb, Ah.data(), Al.data(), bh.data(), bl.data(), xh.data(), xl.data(), tg);
        char lab[64]; snprintf(lab, sizeof lab, "double-double LU, all on GPU (INT8 GEMM %.3f s)", tg);
        printf("%-44s %9.3f %15.3g %15.3g\n", lab, tdd, err_t(xh.data(), xl.data()), err_q(xh.data(), xl.data()));
        printf("%-44s %9.0fx\n", "  speedup vs __float128 LU", tq / tdd);
    }
    printf("%-44s %9.2f %15.3g %15s\n", "__float128 LU (CPU, reference)", tq, err_t(rh.data(), rl.data()), "0");
}
