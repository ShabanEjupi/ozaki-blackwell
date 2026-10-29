// strassen_p.cu -- Strassen over Z/p on the INT8 tensor cores: C = A B mod p with 7^L instead of 8^L block products.
// Ozaki-II multiplies residues mod p < 256. A sum of two residues, reduced mod p, is again a residue, so Strassen's operand
// sums stay int8 at every level (over the integers each level would add a bit and leave int8 after the first). Every L is
// checked bit for bit against the classical product (L = 0); the effective rate is 2 n^3 / time, the classical op count.
// Random residues (not constant fill): the GEMM is power-bound on this laptop, and constant data draws less power.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "../src/gemm_i8.cuh"
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int P = 251, HP = (P - 1) / 2;   // the largest prime below 256; residues in [-125, 125]

__device__ __forceinline__ int8_t modp(int x) {
    int r = x % P;
    return (int8_t)(r > HP ? r - P : r < -HP ? r + P : r);
}

// leaf: C (h x h residues, row stride ldc) = A B mod p; B is given transposed (Bt, K contiguous)
__global__ void __launch_bounds__(oz::THREADS) leaf(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h) {
    oz::gemm_i8_tile(A, Bt, h, [&](int r, int c, int c00, int c01, int c10, int c11) {
        *(char2 *)&C[(size_t)r * ldc + c] = make_char2(modp(c00), modp(c01));
        *(char2 *)&C[(size_t)(r + 8) * ldc + c] = make_char2(modp(c10), modp(c11));
    }, lda, ldb);
}

// Z = X + s Y mod p (h x h, 16 bytes per thread)
__global__ void comb(const int8_t *X, int ldx, const int8_t *Y, int ldy, int s, int8_t *Z, int ldz, int h) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = h / 16;
    if (i >= (size_t)h * w) return;
    size_t r = i / w, c = (i % w) * 16;
    int4 x = *(const int4 *)&X[r * ldx + c], y = *(const int4 *)&Y[r * ldy + c], z;
    const int8_t *px = (const int8_t *)&x, *py = (const int8_t *)&y; int8_t *pz = (int8_t *)&z;
#pragma unroll
    for (int k = 0; k < 16; k++) { int v = px[k] + s * py[k]; pz[k] = (int8_t)(v > HP ? v - P : v < -HP ? v + P : v); }
    *(int4 *)&Z[r * ldz + c] = z;
}

// C1 = [C1 +] s1 M, C2 = [C2 +] s2 M (mod p); "set" overwrites instead of adding (first touch of a quadrant)
__global__ void acc(const int8_t *M, int ldm, int8_t *C1, int s1, bool set1, int8_t *C2, int s2, bool set2, int ldc, int h) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = h / 16;
    if (i >= (size_t)h * w) return;
    size_t r = i / w, c = (i % w) * 16;
    int4 m = *(const int4 *)&M[r * ldm + c];
    const int8_t *pm = (const int8_t *)&m;
    int8_t *D[2] = {C1, C2}; int S[2] = {s1, s2}; bool Z[2] = {set1, set2};
#pragma unroll
    for (int d = 0; d < 2; d++) {
        if (!D[d]) continue;
        int4 o = Z[d] ? make_int4(0, 0, 0, 0) : *(int4 *)&D[d][r * ldc + c];
        int8_t *po = (int8_t *)&o;
#pragma unroll
        for (int k = 0; k < 16; k++) { int v = po[k] + S[d] * pm[k]; po[k] = (int8_t)(v > HP ? v - P : v < -HP ? v + P : v); }
        *(int4 *)&D[d][r * ldc + c] = o;
    }
}

static unsigned grid16(int h) { return (unsigned)(((size_t)h * (h / 16) + 255) / 256); }

struct Ws { int8_t *S, *T, *M; };   // one set per recursion depth, (h/2)^2 each

// C = A B mod p, h x h, L Strassen levels (B transposed: block B_ij of B is block (j, i) of Bt)
static void strassen(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L,
                     std::vector<Ws> &ws, int d) {
    if (L == 0) { leaf<<<dim3(h / oz::BN, h / oz::BM), oz::THREADS>>>(A, lda, Bt, ldb, C, ldc, h); return; }
    const int q = h / 2;
    const int8_t *A11 = A, *A12 = A + q, *A21 = A + (size_t)q * lda, *A22 = A21 + q;
    const int8_t *B11 = Bt, *B12 = Bt + (size_t)q * ldb, *B21 = Bt + q, *B22 = B12 + q;
    int8_t *C11 = C, *C12 = C + q, *C21 = C + (size_t)q * ldc, *C22 = C21 + q;
    int8_t *S = ws[d].S, *T = ws[d].T, *M = ws[d].M;
    const unsigned g = grid16(q);
    auto sum = [&](const int8_t *X, int ldx, const int8_t *Y, int ldy, int s, int8_t *Z) { comb<<<g, 256>>>(X, ldx, Y, ldy, s, Z, q, q); };
    auto prod = [&](const int8_t *X, int ldx, const int8_t *Y, int ldy) { strassen(X, ldx, Y, ldy, M, q, q, L - 1, ws, d + 1); };
    auto put = [&](int8_t *D1, int s1, bool z1, int8_t *D2, int s2, bool z2) { acc<<<g, 256>>>(M, q, D1, s1, z1, D2, s2, z2, ldc, q); };
    sum(A11, lda, A22, lda, 1, S); sum(B11, ldb, B22, ldb, 1, T); prod(S, q, T, q); put(C11, 1, true, C22, 1, true);    // M1
    sum(A21, lda, A22, lda, 1, S); prod(S, q, B11, ldb); put(C21, 1, true, C22, -1, false);                              // M2
    sum(B12, ldb, B22, ldb, -1, T); prod(A11, lda, T, q); put(C12, 1, true, C22, 1, false);                              // M3
    sum(B21, ldb, B11, ldb, -1, T); prod(A22, lda, T, q); put(C11, 1, false, C21, 1, false);                             // M4
    sum(A11, lda, A12, lda, 1, S); prod(S, q, B22, ldb); put(C11, -1, false, C12, 1, false);                             // M5
    sum(A21, lda, A11, lda, -1, S); sum(B11, ldb, B12, ldb, 1, T); prod(S, q, T, q); put(C22, 1, false, nullptr, 0, false);  // M6
    sum(A12, lda, A22, lda, -1, S); sum(B21, ldb, B22, ldb, 1, T); prod(S, q, T, q); put(C11, 1, false, nullptr, 0, false);  // M7
}

__global__ void fill(int8_t *X, size_t n, unsigned seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    X[i] = (int8_t)((int)(x % P) - HP);
}

__global__ void diff(const int8_t *X, const int8_t *Y, size_t n, unsigned long long *bad) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && X[i] != Y[i]) atomicAdd(bad, 1ull);
}

int main(int argc, char **argv) {
    const int maxL = argc > 1 ? atoi(argv[1]) : 4;
    for (int n : {8192, 16384, 32768}) {
        const size_t nn = (size_t)n * n;
        int8_t *A, *Bt, *C0, *C;
        CK(cudaMalloc(&A, nn)); CK(cudaMalloc(&Bt, nn)); CK(cudaMalloc(&C0, nn)); CK(cudaMalloc(&C, nn));
        fill<<<(nn + 255) / 256, 256>>>(A, nn, 12345u); fill<<<(nn + 255) / 256, 256>>>(Bt, nn, 67890u);
        std::vector<Ws> ws;
        for (int h = n / 2, d = 0; d < maxL; d++, h /= 2) {
            Ws w; size_t s = (size_t)h * h;
            CK(cudaMalloc(&w.S, s)); CK(cudaMalloc(&w.T, s)); CK(cudaMalloc(&w.M, s)); ws.push_back(w);
        }
        unsigned long long *bad; CK(cudaMalloc(&bad, 8));
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int L = 0; L <= maxL && (n >> L) >= 1024; L++) {
            int8_t *out = L ? C : C0;
            strassen(A, n, Bt, n, out, n, n, L, ws, 0);   // warm-up (and the L = 0 reference)
            const int reps = n >= 32768 ? 2 : 3;
            cudaEventRecord(t0);
            for (int r = 0; r < reps; r++) strassen(A, n, Bt, n, out, n, n, L, ws, 0);
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
            float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
            // the same number of leaf GEMMs alone, back to back: how much of the time is tensor-core work
            const int h = n >> L; size_t leaves = 1; for (int l = 0; l < L; l++) leaves *= 7;
            cudaEventRecord(t0);
            for (size_t j = 0; j < leaves; j++) leaf<<<dim3(h / oz::BN, h / oz::BM), oz::THREADS>>>(A, n, Bt, n, C, n, h);
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
            float gm; cudaEventElapsedTime(&gm, t0, t1);
            unsigned long long nb = 0;
            if (!L) {   // the reference itself, independently: 64 entries on the CPU from their row of A and row of Bt
                std::vector<int8_t> ra(n), rb(n); int8_t got; int wrong = 0; srand(n);
                for (int e = 0; e < 64; e++) {
                    size_t r = rand() % n, c = rand() % n;
                    CK(cudaMemcpy(ra.data(), A + r * n, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rb.data(), Bt + c * n, n, cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(&got, C0 + r * n + c, 1, cudaMemcpyDeviceToHost));
                    long long s = 0; for (int k = 0; k < n; k++) s += ra[k] * rb[k];
                    int m = (int)(s % P); m = m > HP ? m - P : m < -HP ? m + P : m;
                    wrong += m != got;
                }
                printf("n=%5d reference: %d of 64 CPU-checked entries wrong\n", n, wrong);
                if (wrong) return 1;
            }
            if (L) {
                strassen(A, n, Bt, n, C, n, n, L, ws, 0);
                CK(cudaMemset(bad, 0, 8)); diff<<<(nn + 255) / 256, 256>>>(C0, C, nn, bad); CK(cudaMemcpy(&nb, bad, 8, cudaMemcpyDeviceToHost));
                if (L == 1) {   // mutant: one byte changed must be counted, or "differing 0" means nothing
                    int8_t v; CK(cudaMemcpy(&v, C + nn / 3, 1, cudaMemcpyDeviceToHost)); int8_t w = v == HP ? -HP : v + 1;
                    CK(cudaMemcpy(C + nn / 3, &w, 1, cudaMemcpyHostToDevice));
                    unsigned long long mb = 0; CK(cudaMemset(bad, 0, 8)); diff<<<(nn + 255) / 256, 256>>>(C0, C, nn, bad);
                    CK(cudaMemcpy(&mb, bad, 8, cudaMemcpyDeviceToHost));
                    if (mb != nb + 1) { printf("mutant not caught (%llu)\n", mb); return 1; }
                }
            }
            CK(cudaGetLastError());
            printf("n=%5d L=%d  %8.2f ms  effective %6.1f TOPS  (leaf GEMMs alone %8.2f ms = %5.1f TOPS on %zu x %d^3)  differing %llu\n",
                   n, L, ms, 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12, gm, 2.0 * leaves * h * h * (double)h / (gm * 1e-3) / 1e12,
                   leaves, h, nb);
            fflush(stdout);
        }
        for (auto &w : ws) { cudaFree(w.S); cudaFree(w.T); cudaFree(w.M); }
        cudaFree(A); cudaFree(Bt); cudaFree(C0); cudaFree(C); cudaFree(bad);
    }
    return 0;
}
