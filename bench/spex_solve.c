// spex_solve.c -- the same integer system (bench/mm2oz.py .int) solved by SPEX Left LU (SuiteSparse 7.6, exact, sparse,
// CPU, GMP): the state of the art for exact sparse solving. Prints its time, the denominator and the digest
// sum_j x_j r^j mod 2^61 - 1 (r = 1234567, as bench/det.cu OZ_FILE) so the two solutions compare without printing them.
// gcc -O2 -o out_spex bench/spex_solve.c -I/usr/include/suitesparse -lspex -lgmp -lm
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>
#include "SPEX.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "r"); long n, nz;
    if (!f || fscanf(f, "%ld %ld", &n, &nz) != 2) return 1;
    SPEX_initialize();
    SPEX_options *opt; SPEX_create_default_options(&opt);
    SPEX_matrix *T, *A, *b, *x;
    SPEX_matrix_allocate(&T, SPEX_TRIPLET, SPEX_MPZ, n, n, nz, false, true, opt);
    char buf[1 << 16];
    for (long k = 0; k < nz; k++) {
        long i, j; if (fscanf(f, "%ld %ld %65535s", &i, &j, buf) != 3) return 1;
        T->i[k] = i - 1; T->j[k] = j - 1; mpz_set_str(T->x.mpz[k], buf, 10);
    }
    T->nz = nz;
    SPEX_matrix_allocate(&b, SPEX_DENSE, SPEX_MPZ, n, 1, n, false, true, opt);
    for (long k = 0; k < n; k++) { if (fscanf(f, "%65535s", buf) != 1) return 1; mpz_set_str(b->x.mpz[k], buf, 10); }
    fclose(f);
    SPEX_matrix_copy(&A, SPEX_CSC, SPEX_MPZ, T, opt);
    const double t0 = now();
    SPEX_info ok = SPEX_Left_LU_backslash(&x, SPEX_MPQ, A, b, opt);
    const double t = now() - t0;
    if (ok != SPEX_OK) { printf("SPEX: status %d after %.2f s\n", ok, t); return 1; }
    const unsigned __int128 P = (1ull << 61) - 1; unsigned __int128 h = 0, rp = 1;
    mpz_t D, r, nu; mpz_inits(D, r, nu, NULL); mpz_set_ui(D, 1);
    double tot = 0;
    for (long j = 0; j < n; j++) {
        mpz_lcm(D, D, mpq_denref(x->x.mpq[j])); tot += mpz_sizeinbase(mpq_numref(x->x.mpq[j]), 2);
        // x_j mod P = num * den^-1
        mpz_set_ui(r, (unsigned long)P); mpz_invert(nu, mpq_denref(x->x.mpq[j]), r);
        mpz_mul(nu, nu, mpq_numref(x->x.mpq[j])); mpz_fdiv_r(nu, nu, r);
        h = (h + (unsigned __int128)mpz_get_ui(nu) * rp) % P; rp = rp * 1234567 % P;
    }
    printf("SPEX Left LU: n %ld nnz %ld  %.3f s  D %zu bits, numerators %.1f Mbit (own denominators), digest %016llx\n",
           n, nz, t, mpz_sizeinbase(D, 2), tot / 1e6, (unsigned long long)h);
    return 0;
}
