// solve.cu -- a real job for the double-double GEMM: Ax = b with kappa(A) = 1e20, where FP64 has no correct digit.
// A = Q1 diag(sigma) Q2: Q1 the orthogonal DST-I, Q2 its rows permuted and sign-flipped, sigma from 1 down to 10^-e.
// Not fixable by row/column scaling: the small singular values are spread over every entry.
// One matrix, four solvers:
//   FP64 LU on the CPU; the same + iterative refinement with double-double residuals (the textbook rescue, needs kappa*u < 1);
//   blocked double-double LU: panels and U12 on the CPU, the trailing updates (2/3 n^3 of the work) on the INT8 tensor cores;
//   __float128 LU on every CPU core: the reference, and what one would run today to get this accuracy.
// Build: g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o &&
//        nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp,-ffp-contract=off -o out_solve bench/solve.cu out_ref.o -lquadmath -lgomp
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
static inline D two_sum(double a, double b) { double s = a + b, v = s - a; return {s, (a - (s - v)) + (b - v)}; }
static inline D qts(double a, double b) { double s = a + b; return {s, b - (s - a)}; }
static inline D add(D a, D b) {
    D s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
    s = qts(s.hi, s.lo + t.hi);
    return qts(s.hi, s.lo + t.lo);
}
static inline D sub(D a, D b) { return add(a, {-b.hi, -b.lo}); }
static inline D mul(D a, D b) { double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p); return qts(p, e + (a.hi * b.lo + a.lo * b.hi)); }
static inline D dvd(D a, D b) {
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

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 2048, nb = argc > 2 ? atoi(argv[2]) : 256;
    double ek = argc > 3 ? atof(argv[3]) : 20;
    int thr = omp_get_max_threads();
    printf("Ax = b, n=%d, kappa_2(A) = 1e%.0f by construction (A = Q1 diag(sigma) Q2), nb=%d, %d CPU threads\n", n, ek, nb, thr);

    // A = (Q1 diag(sigma)) * Q2, in double-double, the product on the GPU
    std::vector<double> Qh((size_t)n * n), Ql(Qh.size()), sh(n), sl(n);
    quad_dst(n, Qh.data(), Ql.data()); quad_sigma(n, ek, sh.data(), sl.data());
    std::mt19937_64 g(7);
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
    printf("%-44s %9.2f %15.3g %15s\n", "__float128 LU (CPU, reference)", tq, err_t(rh.data(), rl.data()), "0");
}
