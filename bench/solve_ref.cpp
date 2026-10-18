// solve_ref.cpp -- the __float128 side of bench/solve.cu: exact pieces of the test matrix, and the reference solve.
// A separate translation unit so g++, not nvcc's front end, sees __float128.
#include <quadmath.h>
#include <omp.h>
#include <vector>
typedef __float128 Q;
static void split(Q v, double &h, double &l) { h = (double)v; l = (double)(v - (Q)h); }

// orthogonal DST-I, Q_ij = sqrt(2/(n+1)) sin(pi (i+1)(j+1) / (n+1)), as hi + lo
void quad_dst(int n, double *Qh, double *Ql) {
    Q c = sqrtq((Q)2 / (n + 1));
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            long long k = (long long)(i + 1) * (j + 1) % (2 * (n + 1));   // exact argument reduction, sin is 2(n+1)-periodic in k
            split(c * sinq(M_PIq * k / (n + 1)), Qh[(size_t)i * n + j], Ql[(size_t)i * n + j]);
        }
}

// sigma_i = 10^(-e i / (n-1)), i = 0..n-1
void quad_sigma(int n, double e, double *h, double *l) {
    for (int i = 0; i < n; i++) split(powq(10, -(Q)e * i / (n - 1)), h[i], l[i]);
}

// LU with partial pivoting, every operation in __float128, on all cores; returns seconds
double quad_solve(int n, const double *Ah, const double *Al, const double *bh, const double *bl, double *xh, double *xl) {
    std::vector<Q> A((size_t)n * n), b(n);
    for (size_t t = 0; t < A.size(); t++) A[t] = (Q)Ah[t] + Al[t];
    for (int i = 0; i < n; i++) b[i] = (Q)bh[i] + bl[i];
    double t0 = omp_get_wtime();
    for (int j = 0; j < n; j++) {
        int p = j;
        for (int i = j + 1; i < n; i++) if (fabsq(A[(size_t)i * n + j]) > fabsq(A[(size_t)p * n + j])) p = i;
        if (p != j) { for (int c = 0; c < n; c++) std::swap(A[(size_t)j * n + c], A[(size_t)p * n + c]); std::swap(b[j], b[p]); }
        Q d = A[(size_t)j * n + j];
#pragma omp parallel for schedule(static)
        for (int i = j + 1; i < n; i++) {
            Q l = A[(size_t)i * n + j] / d;
            A[(size_t)i * n + j] = l;
            for (int c = j + 1; c < n; c++) A[(size_t)i * n + c] -= l * A[(size_t)j * n + c];
        }
    }
    for (int i = 0; i < n; i++) for (int t = 0; t < i; t++) b[i] -= A[(size_t)i * n + t] * b[t];
    for (int i = n - 1; i >= 0; i--) {
        Q s = b[i];
        for (int t = i + 1; t < n; t++) s -= A[(size_t)i * n + t] * b[t];
        b[i] = s / A[(size_t)i * n + i];
    }
    double sec = omp_get_wtime() - t0;
    for (int i = 0; i < n; i++) split(b[i], xh[i], xl[i]);
    return sec;
}
