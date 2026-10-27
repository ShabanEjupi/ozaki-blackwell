// lu8_test.cu -- A^-1 modulo primes below 256 with partial pivoting (src/ozaki_det.cuh panel_lu_piv): the B0 that an 8-bit
// X of the HOL residue chain needs. Random int8 n x n, s primes; checks (A mod p) A^-1 = I on random rows, against the
// 16-bit LU without pivoting for time.  nvcc -O3 -arch=sm_120 -std=c++17 -o out_lu8 bench/lu8_test.cu ; out_lu8 n [s]
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../src/ozaki_det.cuh"

int main(int argc, char **argv) {
    const int n = atoi(argv[1]), s = argc > 2 ? atoi(argv[2]) : 12;
    std::vector<int8_t> A((size_t)n * n); uint64_t q = 12345;
    for (auto &a : A) { q = q * 6364136223846793005ull + 1442695040888963407ull; a = (int8_t)((q >> 33) % 255 - 127); }
    int8_t *dA; cudaMalloc(&dA, A.size()); cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    std::vector<uint32_t> p8;
    for (uint32_t c = 255; p8.size() < (size_t)s && c > 2; c--) { bool pr = true; for (uint32_t d = 2; d * d <= c; d++) if (c % d == 0) pr = false; if (pr) p8.push_back(c); }
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    for (int mode = 0; mode < 2; mode++) {   // 0: s primes < 256 with pivoting; 1: s/2 primes < 2^16 without (today's B0)
        const int sz = mode ? s / 2 : s;
        std::vector<uint32_t> P(sz), one(sz, 1); std::vector<int> zero(sz, 0), bad(sz);
        for (int t = 0; t < sz; t++) P[t] = mode ? 65521 - 2 * t * 6 : p8[t];
        if (mode) { int k = 0; for (uint32_t c = 65521; k < sz; c -= 2) { bool pr = true; for (uint32_t d = 3; d * d <= c; d += 2) if (c % d == 0) pr = false; if (pr) P[k++] = c; } }
        oz::DetWork w; w.alloc(n, sz, 2 * n); w.piv = mode == 0;
        cudaMemcpy(w.P, P.data(), sz * 4, cudaMemcpyHostToDevice); cudaMemcpy(w.det, one.data(), sz * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(w.bad, zero.data(), sz * 4, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize(); const double t0 = now();
        oz::det_mod_batch(w, dA);
        cudaDeviceSynchronize(); const double t1 = now();
        cudaMemcpy(bad.data(), w.bad, sz * 4, cudaMemcpyDeviceToHost);
        int nbad = 0; for (int b : bad) nbad += b;
        // check: row i of (A mod p) times A^-1 = e_i, 3 rows per prime, the whole row
        int wrong = 0;
        std::vector<int32_t> Ai((size_t)n * n);
        for (int t = 0; t < sz; t++) {
            if (bad[t]) continue;
            cudaMemcpy2D(Ai.data(), n * 4, w.R + (size_t)t * w.zr() + n, (size_t)w.ld * 4, n * 4, n, cudaMemcpyDeviceToHost);
            for (int rr = 0; rr < 3; rr++) {
                const int i = (rr * 7919 + t * 104729) % n;
                for (int c = 0; c < n; c++) {
                    long long acc = 0;
                    for (int k = 0; k < n; k++) acc += (long long)(((A[(size_t)i * n + k] % (int)P[t]) + P[t]) % P[t]) * Ai[(size_t)k * n + c];
                    if (acc % P[t] != (c == i)) wrong++;
                }
            }
        }
        printf("n %d, %d primes %s: %.3f s, %d singular, %d wrong entries in %d checked rows\n", n, sz,
               mode ? "< 2^16, no pivoting" : "< 256, partial pivoting", t1 - t0, nbad, wrong, 3 * (sz - nbad));
        w.free_();
    }
    return 0;
}
