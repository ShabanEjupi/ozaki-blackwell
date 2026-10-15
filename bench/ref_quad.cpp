// ref_quad.cpp -- the reference for the double-double GEMM: every product and sum in __float128 (113-bit significand).
// A separate translation unit so g++, not nvcc's front end, sees __float128.
#include <quadmath.h>
#include <cmath>
void ref_quad(const double *Ah, const double *Al, const double *Bh, const double *Bl, int M, int N, int K,
              double *Rh, double *Rl, double *absAB) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            __float128 s = 0, a = 0;
            for (int k = 0; k < K; k++) {
                __float128 p = ((__float128)Ah[i * K + k] + Al[i * K + k]) * ((__float128)Bh[k * N + j] + Bl[k * N + j]);
                s += p; a += fabsq(p);
            }
            Rh[i * N + j] = (double)s; Rl[i * N + j] = (double)(s - (__float128)Rh[i * N + j]); absAB[i * N + j] = (double)a;
        }
}
