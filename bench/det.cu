// det.cu -- exact determinant of an n x n int8 matrix on INT8 tensor cores (src/ozaki_det.cuh) against FLINT fmpz_mat_det.
// g++ -O3 -c bench/det_flint.cpp -o out_dflint.o && nvcc -O3 -arch=sm_120 -std=c++17 -o out_det bench/det.cu out_dflint.o -lflint -lgmp
// ./out_det n [primes per batch=32] [flint: 1 compare, 0 skip]
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <gmp.h>
#include "../src/ozaki_det.cuh"
double flint_det(int n, const int8_t *A, int threads, long *bits, std::string *hex);

int main(int argc, char **argv) {
    const int n = atoi(argv[1]), Pb = argc > 2 ? atoi(argv[2]) : 32, cmp = argc > 3 ? atoi(argv[3]) : 1;
    std::vector<int8_t> A((size_t)n * n); uint64_t s = 12345;   // the same LCG as det_flint.cpp's main
    for (auto &a : A) { s = s * 6364136223846793005ull + 1442695040888963407ull; a = (int8_t)((s >> 33) % 255 - 127); }

    double hb = 0;   // Hadamard: log2 |det| <= sum_i log2 |row_i|
    for (int i = 0; i < n; i++) { double r = 0; for (int k = 0; k < n; k++) r += (double)A[(size_t)i * n + k] * A[(size_t)i * n + k]; hb += 0.5 * log2(r); }
    std::vector<char> comp(65280, 0); std::vector<uint32_t> pr;   // primes downwards from 65279
    for (int i = 2; i < 65280; i++) if (!comp[i]) for (int k = 2 * i; k < 65280; k += i) comp[k] = 1;
    double lb = 0; const double need = hb + 2;   // the sign, and a bit of slack
    for (int i = 65279; i > 2 && lb < need * 1.02 + 16 * 16; i--) if (!comp[i]) { pr.push_back(i); lb += log2((double)i); }
    if (lb < need) { printf("n=%d needs %.0f bits, the primes < 65280 give %.0f\n", n, need, lb); return 1; }
    while (pr.size() % Pb) pr.pop_back();

    int8_t *dA; cudaMalloc(&dA, A.size()); cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    oz::DetWork w; w.alloc(n, Pb);
    std::vector<uint32_t> det(pr.size()); std::vector<int> bad(pr.size());
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (size_t b = 0; b < pr.size(); b += Pb) {
        std::vector<uint32_t> one(Pb, 1); std::vector<int> zero(Pb, 0);
        cudaMemcpy(w.P, &pr[b], Pb * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(w.det, one.data(), Pb * 4, cudaMemcpyHostToDevice); cudaMemcpy(w.bad, zero.data(), Pb * 4, cudaMemcpyHostToDevice);
        oz::det_mod_batch(w, dA);
        cudaMemcpy(&det[b], w.det, Pb * 4, cudaMemcpyDeviceToHost); cudaMemcpy(&bad[b], w.bad, Pb * 4, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA: %s\n", cudaGetErrorString(err)); return 1; }

    // CRT over the good primes: x += M ((r - x) M^-1 mod p), symmetric at the end
    auto t0 = std::chrono::steady_clock::now();
    mpz_t x, M, t; mpz_init_set_ui(x, 0); mpz_init_set_ui(M, 1); mpz_init(t);
    int good = 0; double gb = 0;
    for (size_t i = 0; i < pr.size(); i++) {
        if (bad[i]) continue;
        uint32_t p = pr[i], xm = mpz_fdiv_ui(x, p), mm = mpz_fdiv_ui(M, p);
        uint64_t inv = 1, a = mm, e = p - 2;
        for (; e; e >>= 1, a = a * a % p) if (e & 1) inv = inv * a % p;
        uint64_t k = (uint64_t)(det[i] + p - xm) % p * inv % p;
        mpz_addmul_ui(x, M, k); mpz_mul_ui(M, M, p); good++; gb += log2((double)p);
    }
    mpz_fdiv_q_2exp(t, M, 1); if (mpz_cmp(x, t) > 0) mpz_sub(x, x, M);
    double crt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("n=%d  Hadamard %.0f bits  primes %zu (%d good, %.0f bits)  GPU %.3f s  CRT %.3f s  |det| %zu bits\n", n, hb,
           pr.size(), good, gb, ms / 1e3, crt, mpz_sizeinbase(x, 2));
    if (gb < need) { printf("NOT ENOUGH GOOD PRIMES\n"); return 1; }
    if (cmp) {
        long fb; std::string fh; double ft = flint_det(n, A.data(), 24, &fb, &fh);
        char *gh = mpz_get_str(nullptr, 16, x);
        printf("  FLINT fmpz_mat_det (24 thr) %.2f s  -> %s  (%.1fx)\n", ft, fh == gh ? "BIT-EXACT" : "MISMATCH", ft / (ms / 1e3 + crt));
        free(gh);
    }
    return 0;
}
