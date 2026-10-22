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

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// C = A * B, all n-dim double-double on the GPU (one-off, for building the matrix)
struct Dev { double *p[6] = {}; size_t cap = 0; };
static void gpu_dd(Dev &g, int M, int N, int K, const double *Ah, const double *Al, const double *Bh, const double *Bl,
                   double *Ch, double *Cl, double &tg) {
    size_t need = std::max({(size_t)M * K, (size_t)K * N, (size_t)M * N});
    if (need > g.cap) { for (auto &x : g.p) { cudaFree(x); cudaMalloc(&x, need * 8); } g.cap = need; }
    double t0 = now();
    cudaMemcpy(g.p[0], Ah, (size_t)M * K * 8, cudaMemcpyHostToDevice); cudaMemcpy(g.p[1], Al, (size_t)M * K * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(g.p[2], Bh, (size_t)K * N * 8, cudaMemcpyHostToDevice); cudaMemcpy(g.p[3], Bl, (size_t)K * N * 8, cudaMemcpyHostToDevice);
    oz::Plan p = oz::plan(M, N, K, 32, 106);
    oz::ddgemm_t<32>(p, g.p[0], g.p[1], g.p[2], g.p[3], g.p[4], g.p[5]);
    cudaError_t e = cudaGetLastError(); if (e) { fprintf(stderr, "ddgemm: %s\n", cudaGetErrorString(e)); exit(1); }
    cudaMemcpy(Ch, g.p[4], (size_t)M * N * 8, cudaMemcpyDeviceToHost); cudaMemcpy(Cl, g.p[5], (size_t)M * N * 8, cudaMemcpyDeviceToHost);
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
#define CK(x) do { cudaError_t e_ = (x); if (e_) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
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
__global__ void k_inv(const double *H, const double *L, int n, int k0, int kb, double *Ih, double *Il) {
    __shared__ double xh[1024], xl[1024];
    int c = blockIdx.x, r = threadIdx.x;
    xh[r] = r == c; xl[r] = 0;
    __syncthreads();
    for (int t = c; t < kb - 1; t++) {
        if (r > t) { D v = sub({xh[r], xl[r]}, mul({H[AT(k0 + r, k0 + t)], L[AT(k0 + r, k0 + t)]}, {xh[t], xl[t]})); xh[r] = v.hi; xl[r] = v.lo; }
        __syncthreads();
    }
    Ih[(size_t)r * kb + c] = xh[r]; Il[(size_t)r * kb + c] = xl[r];
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
__global__ void k_diag(const double *H, const double *L, int n, int b0, int bb, bool upper, double *Xh, double *Xl) {
    __shared__ double xh[1024], xl[1024];
    int r = threadIdx.x;
    xh[r] = Xh[b0 + r]; xl[r] = Xl[b0 + r];
    __syncthreads();
    for (int s = 0; s < bb; s++) {
        int t = upper ? bb - 1 - s : s;
        if (upper && r == t) { D v = dvd({xh[t], xl[t]}, {H[AT(b0 + t, b0 + t)], L[AT(b0 + t, b0 + t)]}); xh[t] = v.hi; xl[t] = v.lo; }
        __syncthreads();
        if (upper ? r < t : r > t) {
            D v = sub({xh[r], xl[r]}, mul({H[AT(b0 + r, b0 + t)], L[AT(b0 + r, b0 + t)]}, {xh[t], xl[t]})); xh[r] = v.hi; xl[r] = v.lo;
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
#undef AT

// x = A^-1 b, everything on the GPU; returns seconds from A on the host to x on the host, tg = the INT8 GEMM part
static double solve_gpu(int n, int nb, const double *Ah, const double *Al, const double *bh, const double *bl,
                        double *xh, double *xl, double &tg) {
    double *H, *L, *Lh, *Ll, *Uh, *Ul, *Ph, *Pl, *Xh, *Xl; int *piv;
    size_t nn = (size_t)n * n, mk = (size_t)n * nb, mm = (size_t)(n - nb) * (n - nb);
    CK(cudaMalloc(&H, nn * 8)); CK(cudaMalloc(&L, nn * 8)); CK(cudaMalloc(&Lh, mk * 8)); CK(cudaMalloc(&Ll, mk * 8));
    CK(cudaMalloc(&Uh, mk * 8)); CK(cudaMalloc(&Ul, mk * 8)); CK(cudaMalloc(&Ph, mm * 8)); CK(cudaMalloc(&Pl, mm * 8));
    CK(cudaMalloc(&Xh, n * 8)); CK(cudaMalloc(&Xl, n * 8)); CK(cudaMalloc(&piv, n * 4));
    oz::Plan p = oz::plan(n - nb, n - nb, nb, 32, 106);   // buffers for the largest update, reused shrinking
    oz::Plan q = oz::plan(nb, n - nb, nb, 32, 106);       // U12 = L11^-1 A12
    double *Ih, *Il; CK(cudaMalloc(&Ih, (size_t)nb * nb * 8)); CK(cudaMalloc(&Il, (size_t)nb * nb * 8));
    static const bool trsm_ref = getenv("OZ_TRSM");       // OZ_TRSM=1: the column-by-column substitution instead
    cudaEvent_t g0, g1; cudaEventCreate(&g0); cudaEventCreate(&g1);
    CK(cudaDeviceSynchronize());
    double t0 = now();
    CK(cudaMemcpy(H, Ah, nn * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(L, Al, nn * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Xh, bh, n * 8, cudaMemcpyHostToDevice)); CK(cudaMemcpy(Xl, bl, n * 8, cudaMemcpyHostToDevice));
    float gms = 0;
    // OZ_PROF=1: seconds per phase, with a sync at every boundary -- only for finding where the time goes
    static const bool prof = getenv("OZ_PROF"); double ph[5] = {}, tp = now();
#define PH(k) if (prof) { cudaDeviceSynchronize(); double t_ = now(); ph[k] += t_ - tp; tp = t_; }
    for (int k0 = 0; k0 < n; k0 += nb) {
        int kb = std::min(nb, n - k0), e = k0 + kb, m = n - e;
        PH(0);
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
            k_inv<<<kb, kb>>>(H, L, n, k0, kb, Ih, Il);
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
    if (getenv("OZ_SOLVE")) k_solve<<<1, 1024>>>(H, L, n, piv, Xh, Xl);   // OZ_SOLVE=1: the one-block column solve
    else {
        k_perm<<<1, 1>>>(n, piv, Xh, Xl);
        for (int b0 = 0; b0 < n; b0 += nb) {   // L y = P b
            int bb = std::min(nb, n - b0);
            k_diag<<<1, bb>>>(H, L, n, b0, bb, false, Xh, Xl);
            if (b0 + bb < n) k_gemv<<<(n - b0 - bb + 7) / 8, 256>>>(H, L, n, b0 + bb, n, b0, bb, Xh, Xl);
        }
        for (int b0 = (n - 1) / nb * nb; b0 >= 0; b0 -= nb) {   // U x = y
            int bb = std::min(nb, n - b0);
            k_diag<<<1, bb>>>(H, L, n, b0, bb, true, Xh, Xl);
            if (b0) k_gemv<<<(b0 + 7) / 8, 256>>>(H, L, n, 0, b0, b0, bb, Xh, Xl);
        }
    }
    PH(3);
    if (prof) fprintf(stderr, "phases: copies+gemm+sub %.3f  panel %.3f  trsm %.3f  solve %.3f s\n", ph[0], ph[1], ph[2], ph[3]);
    CK(cudaMemcpy(xh, Xh, n * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(xl, Xl, n * 8, cudaMemcpyDeviceToHost));
    double t = now() - t0;
    tg = gms / 1e3;
    p.M = p.N = n - nb; oz::release(p); q.N = n - nb; oz::release(q); cudaFree(Ih); cudaFree(Il);
    for (double *q : {H, L, Lh, Ll, Uh, Ul, Ph, Pl, Xh, Xl}) cudaFree(q);
    cudaFree(piv);
    return t;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048, nb = argc > 2 ? atoi(argv[2]) : 256;
    double ek = argc > 3 ? atof(argv[3]) : 20;
    int thr = omp_get_max_threads();
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

    if (getenv("OZ_PROF")) {   // only the GPU solver, twice (the first run pays the module load)
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

    // FP64: LU of A rounded to double, then refinement with residuals b - A x in double-double
    {
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
    {
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
        double tdd = solve_gpu(n, nb, Ah.data(), Al.data(), bh.data(), bl.data(), xh.data(), xl.data(), tg);
        char lab[64]; snprintf(lab, sizeof lab, "double-double LU, all on GPU (INT8 GEMM %.3f s)", tg);
        printf("%-44s %9.3f %15.3g %15.3g\n", lab, tdd, err_t(xh.data(), xl.data()), err_q(xh.data(), xl.data()));
        printf("%-44s %9.0fx\n", "  speedup vs __float128 LU", tq / tdd);
    }
    printf("%-44s %9.2f %15.3g %15s\n", "__float128 LU (CPU, reference)", tq, err_t(rh.data(), rl.data()), "0");
}
