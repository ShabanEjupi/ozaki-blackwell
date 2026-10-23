// det_flint.cpp -- FLINT fmpz_mat_det of the same int8 matrices, in its own g++ TU (nvcc does not take FLINT's headers)
#include <cstdint>
#include <chrono>
#include <string>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
// returns seconds; *bits = bit length of |det|, *hex = det in hex (for the bit-exact check)
double flint_det(int n, const int8_t *A, int threads, long *bits, std::string *hex) {
    flint_set_num_threads(threads);
    fmpz_mat_t X; fmpz_mat_init(X, n, n);
    for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) fmpz_set_si(fmpz_mat_entry(X, i, k), A[(size_t)i * n + k]);
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
    long bits; std::string hex; double t = flint_det(n, A.data(), th, &bits, &hex);
    printf("n=%d threads=%d  FLINT fmpz_mat_det %.2f s  |det| %ld bits  head %.16s\n", n, th, t, bits, hex.c_str());
}
#endif
