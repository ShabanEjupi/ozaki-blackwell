// big_flint.cpp -- FLINT fmpz_mat_mul on the same big-integer matrices, in its own g++ TU (nvcc does not take FLINT's headers)
#include <vector>
#include <chrono>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
double flint_gemm(int n, int NL, const unsigned *A, const unsigned *B, int threads) {
    flint_set_num_threads(threads);
    fmpz_mat_t X, Y, Z; fmpz_mat_init(X, n, n); fmpz_mat_init(Y, n, n); fmpz_mat_init(Z, n, n);
    std::vector<ulong> w((NL + 1) / 2);
    auto set = [&](fmpz *z, const unsigned *v) {   // 32-bit two's complement limbs -> 64-bit, sign-extended
        for (int t = 0; t < (NL + 1) / 2; t++) {
            ulong lo = v[2 * t], hi = 2 * t + 1 < NL ? v[2 * t + 1] : ((int)v[NL - 1] < 0 ? 0xffffffffu : 0);
            w[t] = hi << 32 | lo;
        }
        fmpz_set_signed_ui_array(z, w.data(), (NL + 1) / 2);
    };
    for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) {
        set(fmpz_mat_entry(X, i, k), &A[((size_t)i * n + k) * NL]);
        set(fmpz_mat_entry(Y, i, k), &B[((size_t)i * n + k) * NL]);
    }
    auto t0 = std::chrono::steady_clock::now();
    fmpz_mat_mul(Z, X, Y);
    double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fmpz_mat_clear(X); fmpz_mat_clear(Y); fmpz_mat_clear(Z);
    return t;
}
