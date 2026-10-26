// det_flint.cpp -- FLINT fmpz_mat_det of the same int8 matrices, in its own g++ TU (nvcc does not take FLINT's headers)
#include <cstdint>
#include <chrono>
#include <string>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
// returns seconds; *bits = bit length of |det|, *hex = det in hex (for the bit-exact check).
// L > 1: entry = sum_j A[j][i][k] 256^j, L planes of n x n signed bytes (the large-entry matrices of det.cu)
double flint_det(int n, const int8_t *A, int threads, long *bits, std::string *hex, int L) {
    flint_set_num_threads(threads);
    fmpz_mat_t X; fmpz_mat_init(X, n, n);
    for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) {
        fmpz *e = fmpz_mat_entry(X, i, k);
        for (int j = L - 1; j >= 0; j--) { fmpz_mul_2exp(e, e, 8); fmpz_add_si(e, e, A[(size_t)j * n * n + (size_t)i * n + k]); }
    }
    fmpz_t d; fmpz_init(d);
    auto t0 = std::chrono::steady_clock::now();
    fmpz_mat_det(d, X);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    *bits = fmpz_bits(d);
    char *c = fmpz_get_str(nullptr, 16, d); *hex = c; flint_free(c);
    fmpz_clear(d); fmpz_mat_clear(X);
    return s;
}
#ifdef DET_FLINT_MAIN
#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc, char **argv) {
    int n = atoi(argv[1]), th = argc > 2 ? atoi(argv[2]) : 24;
    std::vector<int8_t> A((size_t)n * n); uint64_t s = 12345;
    for (auto &a : A) { s = s * 6364136223846793005ull + 1442695040888963407ull; a = (int8_t)((s >> 33) % 255 - 127); }
    long bits; std::string hex; double t = flint_det(n, A.data(), th, &bits, &hex, 1);
    printf("n=%d threads=%d  FLINT fmpz_mat_det %.2f s  |det| %ld bits  head %.16s\n", n, th, t, bits, hex.c_str());
}
#endif

// FLINT fmpz_mat_solve_dixon_den on the same A and b; returns seconds, *ok = 1 when num[j] / D == X[j] / den for every j
#include <gmp.h>
double flint_solve(int n, const int8_t *A, const int32_t *b, int threads, const __mpz_struct *num, const mpz_t D, int *ok) {
    flint_set_num_threads(threads);
    fmpz_mat_t X, Y, B; fmpz_mat_init(X, n, n); fmpz_mat_init(Y, n, 1); fmpz_mat_init(B, n, 1);
    for (int i = 0; i < n; i++) {
        fmpz_set_si(fmpz_mat_entry(B, i, 0), b[i]);
        for (int k = 0; k < n; k++) fmpz_set_si(fmpz_mat_entry(X, i, k), A[(size_t)i * n + k]);
    }
    fmpz_t den, l, r, f; fmpz_init(den); fmpz_init(l); fmpz_init(r); fmpz_init(f);
    auto t0 = std::chrono::steady_clock::now();
    fmpz_mat_solve_dixon_den(Y, den, X, B);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    *ok = 1;
    for (int j = 0; j < n && *ok; j++) {   // num_j den == Y_j D
        fmpz_set_mpz(f, &num[j]); fmpz_mul(l, f, den);
        fmpz_set_mpz(f, D); fmpz_mul(r, fmpz_mat_entry(Y, j, 0), f);
        *ok = fmpz_equal(l, r);
    }
    fmpz_clear(den); fmpz_clear(l); fmpz_clear(r); fmpz_clear(f);
    fmpz_mat_clear(X); fmpz_mat_clear(Y); fmpz_mat_clear(B);
    return s;
}
