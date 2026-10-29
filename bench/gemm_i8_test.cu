// gemm_i8_test.cu -- the INT8 GEMM is exact (every element equal to a CPU reference), then how fast it is.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../src/gemm_i8.cuh"
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static bool check(int M, int N, int K) {
    std::vector<int8_t> A((size_t)M * K), Bt((size_t)N * K);
    srand(M * 7 + N * 3 + K);
    for (auto &x : A) x = (int8_t)(rand() % 255 - 127);
    for (auto &x : Bt) x = (int8_t)(rand() % 255 - 127);
    int8_t *dA, *dB; int32_t *dC;
    CK(cudaMalloc(&dA, A.size())); CK(cudaMalloc(&dB, Bt.size())); CK(cudaMalloc(&dC, (size_t)M * N * 4));
    CK(cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, Bt.data(), Bt.size(), cudaMemcpyHostToDevice));
    oz::gemm_i8_run(dA, dB, dC, M, N, K);
    CK(cudaGetLastError());
    std::vector<int32_t> C((size_t)M * N);
    CK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int32_t s = 0;
            for (int k = 0; k < K; k++) s += A[(size_t)i * K + k] * Bt[(size_t)j * K + k];
            if (s != C[(size_t)i * N + j] && bad++ < 3) printf("  C[%d,%d] = %d, want %d\n", i, j, C[(size_t)i * N + j], s);
        }
    printf("exact %dx%dx%d: %s (%ld wrong)\n", M, N, K, bad ? "FAIL" : "ok", bad);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad == 0;
}

int main() {
    // the last one is large enough (M N >= 2^26) for gemm_i8_run to take the 128 x 256 tile
    bool ok = check(128, 128, 64) & check(256, 384, 512) & check(512, 256, 1024) & check(8192, 8192, 128);
    if (!ok) return 1;
    for (int n : {2048, 4096, 8192}) {
        int8_t *dA, *dB; int32_t *dC;
        CK(cudaMalloc(&dA, (size_t)n * n)); CK(cudaMalloc(&dB, (size_t)n * n)); CK(cudaMalloc(&dC, (size_t)n * n * 4));
        CK(cudaMemset(dA, 1, (size_t)n * n)); CK(cudaMemset(dB, 1, (size_t)n * n));
        oz::gemm_i8_run(dA, dB, dC, n, n, n);
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int reps = n >= 8192 ? 3 : 10;
        cudaEventRecord(t0);
        for (int r = 0; r < reps; r++) oz::gemm_i8_run(dA, dB, dC, n, n, n);
        cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
        printf("n=%5d  %7.2f ms  %6.1f TOPS\n", n, ms, 2.0 * n * n * n / (ms * 1e-3) / 1e12);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    return 0;
}
