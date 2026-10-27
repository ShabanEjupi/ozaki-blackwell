// rchain_bench.cu -- what one modular product of the HOL residue chain costs, on random data (the power budget of the
// laptop depends on how many bits toggle, so memset-1 inputs flatter the GEMM): plain INT8 GEMM, the 16-bit modgemm
// (two GEMMs of depth 2K), and an 8-bit-prime product (one GEMM of depth K, balanced residues as int8, reduced in the
// epilogue). nvcc -O3 -arch=sm_120 -std=c++17 -o out_rcb bench/rchain_bench.cu
#include <cstdio>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "../src/ozaki_det.cuh"

namespace oz {
// out[z] = (A_z B_z) mod p_z in [0, p): A_z rows K-contiguous int8 (balanced residues), Bt_z likewise
__global__ void __launch_bounds__(THREADS) modgemm8(const int8_t *__restrict__ A, const int8_t *__restrict__ Bt, int32_t *out,
                                                    int M, int N, int K, const uint32_t *__restrict__ P) {
    const int z = blockIdx.z, p = P[z];
    A += (size_t)z * M * K; Bt += (size_t)z * N * K; out += (size_t)z * M * N;
    gemm_i8_tile(A, Bt, K, [&](int row, int col, int c00, int c01, int c10, int c11) {
        auto f = [&](int v) { v %= p; return v < 0 ? v + p : v; };
        out[(size_t)row * N + col] = f(c00); out[(size_t)row * N + col + 1] = f(c01);
        out[(size_t)(row + 8) * N + col] = f(c10); out[(size_t)(row + 8) * N + col + 1] = f(c11);
    });
}
}  // namespace oz

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 4096, nz = argc > 2 ? atoi(argv[2]) : 6, reps = 5;
    const size_t nn = (size_t)n * n;
    std::mt19937 g(1);
    std::vector<int8_t> h(nn * nz * 2);
    for (auto &x : h) x = (int8_t)((int)(g() % 255) - 127);
    int8_t *A, *B; int32_t *C, *R; uint32_t *P;
    cudaMalloc(&A, nn * nz * 2); cudaMalloc(&B, nn * nz * 2); cudaMalloc(&C, nn * nz * 4); cudaMalloc(&R, nn * nz * 4); cudaMalloc(&P, 64 * 4);
    cudaMemcpy(A, h.data(), h.size(), cudaMemcpyHostToDevice); cudaMemcpy(B, h.data(), h.size(), cudaMemcpyHostToDevice);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    auto timeit = [&](const char *name, double macs, auto f) {
        f(); cudaDeviceSynchronize();
        cudaEventRecord(t0); for (int r = 0; r < reps; r++) f(); cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
        printf("%-44s %8.3f ms  %6.1f TOPS  %s\n", name, ms, 2 * macs / (ms * 1e-3) / 1e12, cudaGetErrorString(cudaGetLastError()));
        return ms;
    };
    using namespace oz;
    const double u = (double)n * n * n;
    timeit("gemm_i8 K=n, random bytes", u, [&] { gemm_i8<<<dim3(n / BN, n / BM), THREADS>>>(A, B, C, n, n, n); });
    timeit("gemm_i8 K=2n, random bytes", 2 * u, [&] { gemm_i8<<<dim3(n / BN, n / BM), THREADS>>>(A, B, C, n, n, 2 * n); });
    // 16-bit residues: nz primes, A packed [lo hi], B packed [lo(W) lo(V)] / [hi(W) hi(V)]: 4 n^3 MACs per prime
    std::vector<uint32_t> p16, p8;
    for (uint32_t q = 65279; p16.size() < 64; q -= 2) { bool pr = true; for (uint32_t d = 3; d * d <= q; d += 2) if (q % d == 0) { pr = false; break; } if (pr) p16.push_back(q); }
    for (uint32_t q = 251; p8.size() < 40; q -= 2) { bool pr = true; for (uint32_t d = 3; d * d <= q; d += 2) if (q % d == 0) { pr = false; break; } if (pr) p8.push_back(q); }
    cudaMemcpy(P, p16.data(), 64 * 4, cudaMemcpyHostToDevice);
    DetWork w; w.alloc(n, nz);
    { std::vector<int32_t> r(nn * nz); for (size_t i = 0; i < r.size(); i++) r[i] = g() % 65000; cudaMemcpy(R, r.data(), r.size() * 4, cudaMemcpyHostToDevice); }
    cudaMemcpy(w.P, p16.data(), nz * 4, cudaMemcpyHostToDevice);
    const int Mpad = rup(n, BM), Npad = rup(n, BN);
    pack_b<<<dim3(Npad / 64, n / 64, nz), 256>>>(R, nn, n, n, n, Npad, w.B1, w.B2, w.P);
    char nm[64];
    snprintf(nm, 64, "pack_a 16-bit x %d", nz);
    timeit(nm, 0, [&] { pack_a<<<dim3(std::min(2048, Mpad * n / 1024 + 1), nz), 256>>>(R, nn, n, n, n, Mpad, w.Ap, w.P); });
    snprintf(nm, 64, "modgemm<0> 16-bit x %d (4n^3 each)", nz);
    double m16 = timeit(nm, 4 * u * nz, [&] { modgemm<0><<<dim3(Npad / BN, Mpad / BM, nz), THREADS>>>(w.Ap, w.B1, w.B2, C, nn, n, n, n, 2 * n, Mpad, Npad, w.P); });
    cudaMemcpy(P, p8.data(), 40 * 4, cudaMemcpyHostToDevice);
    snprintf(nm, 64, "modgemm8 8-bit x %d (n^3 each)", 2 * nz);
    double m8 = timeit(nm, u * 2 * nz, [&] { modgemm8<<<dim3(n / BN, n / BM, 2 * nz), THREADS>>>(A, B, C, n, n, n, P); });
    printf("per bit of modulus: 16-bit %.3f ms, 8-bit %.3f ms  (ratio %.2f)\n", m16 / (16.0 * nz), m8 / (7.9 * 2 * nz), m16 / 16.0 / (m8 / 7.9 / 2));
    return 0;
}
