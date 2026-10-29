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
static int G = 8;   // block rasterization band (argv[2]; 0 = row-major, as every other caller)

__global__ void __launch_bounds__(oz::THREADS) leaf(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int g) {
    oz::gemm_i8_tile(A, Bt, h, [&](int r, int c, int c00, int c01, int c10, int c11) {
        *(char2 *)&C[(size_t)r * ldc + c] = make_char2(modp(c00), modp(c01));
        *(char2 *)&C[(size_t)(r + 8) * ldc + c] = make_char2(modp(c10), modp(c11));
    }, lda, ldb, g);
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

struct Ws { int8_t *S, *T, *M; };   // (h/2)^2 each; two sets per recursion depth, used alternately when pipelined

// Pipelined schedule: operand sums on a high-priority stream run while the tensor cores are busy with the previous
// product, so the memory-bound additions hide behind the compute-bound GEMMs. Products k alternate between the two
// workspace sets of their depth (j = k & 1); the only hazards are a set's S/T rewritten while leaves still read them
// (wait mm[d][j]) and its M rewritten before put() has read it (wait ac[d][j]). Off: one set, legacy stream, as before.
struct Pipe { bool on = false; cudaStream_t add = 0, mm = 0, ac = 0; std::vector<cudaEvent_t> ev; };
static Pipe pp;
static cudaEvent_t &evt(int d, int j, int kind) { return pp.ev[(d * 2 + j) * 3 + kind]; }   // kind: 0 sums, 1 mm, 2 put

// Two Strassen levels in one pass. Level by level, a node of size h moves ~39 h^2 bytes through DRAM (sums written, read
// back, summed again; M written, read, added into C, which is read and written again). Two levels are one bilinear
// algorithm with 49 products whose coefficients are Kronecker products (U x U, V x V, W x W): each of the 16 sub-blocks
// is read once and each of the 49 leaf operands written once (A: h^2 + 49/16 h^2), the 49 products run as ONE batched
// launch, and C's 16 sub-blocks are written once from them, never read (~8 h^2 in all). Sub-block (R, c) of a 4 x 4 grid
// is quadrant (R/2, c/2) of the outer level and (R%2, c%2) of the inner one.
struct Term { int8_t n, idx[16], sgn[16]; };          // a linear combination of sub-blocks (A, Bt) or products (C)
__constant__ Term cA[49], cB[49], cC[16];
struct Two { bool on = false; int maxh = 0; int8_t *SA = nullptr, *SB = nullptr, *M = nullptr; };
static Two two;

static void two_tables() {   // M1..M7 exactly as in strassen() below; quadrant index 2 r + c (11, 12, 21, 22)
    const int U[7][4] = {{1, 0, 0, 1}, {0, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {-1, 0, 1, 0}, {0, 1, 0, -1}};
    const int V[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 0}, {0, 1, 0, -1}, {-1, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {0, 0, 1, 1}};
    const int W[4][7] = {{1, 0, 0, 1, -1, 0, 1}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0, 0}, {1, -1, 1, 0, 0, 1, 0}};
    Term a[49] = {}, b[49] = {}, c[16] = {};
    for (int k1 = 0; k1 < 7; k1++) for (int k2 = 0; k2 < 7; k2++) {
        const int K = 7 * k1 + k2;
        for (int R = 0; R < 4; R++) for (int C = 0; C < 4; C++) {
            const int q1 = 2 * (R / 2) + C / 2, q2 = 2 * (R % 2) + C % 2;
            if (int s = U[k1][q1] * U[k2][q2]) { a[K].idx[a[K].n] = 4 * R + C; a[K].sgn[a[K].n++] = s; }
            if (int s = V[k1][q1] * V[k2][q2]) { b[K].idx[b[K].n] = 4 * C + R; b[K].sgn[b[K].n++] = s; }   // B (R, C) = Bt (C, R)
            if (int s = W[q1][k1] * W[q2][k2]) { c[4 * R + C].idx[c[4 * R + C].n] = K; c[4 * R + C].sgn[c[4 * R + C].n++] = s; }
        }
    }
    CK(cudaMemcpyToSymbol(cA, a, sizeof a)); CK(cudaMemcpyToSymbol(cB, b, sizeof b)); CK(cudaMemcpyToSymbol(cC, c, sizeof c));
}

__device__ __forceinline__ int lane(int w, int b) { return (int8_t)(w >> (8 * b)); }

// O[K] (s x s, packed) = sum of the sub-blocks of X that combination K names, mod p; one thread per 4 bytes
template <bool isB>
__global__ void two_ops(const int8_t *X, int ld, int s, int8_t *O) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = s / 4;
    if (i >= (size_t)s * w) return;
    const size_t r = i / w, c = (i % w) * 4;
    int x[16];
#pragma unroll
    for (int t = 0; t < 16; t++) x[t] = *(const int *)&X[((t / 4) * (size_t)s + r) * ld + (t % 4) * s + c];
    const Term *T = isB ? cB : cA;
#pragma unroll 7
    for (int K = 0; K < 49; K++) {
        int v[4] = {};
        for (int e = 0; e < T[K].n; e++)
#pragma unroll
            for (int b = 0; b < 4; b++) v[b] += T[K].sgn[e] * lane(x[T[K].idx[e]], b);
        int o = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) o |= (modp(v[b]) & 0xff) << (8 * b);
        *(int *)&O[(size_t)K * s * s + r * s + c] = o;
    }
}

// C's 16 sub-blocks = their combinations of the 49 products, mod p (written, never read)
__global__ void two_out(const int8_t *M, int s, int8_t *C, int ldc) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = s / 4;
    if (i >= (size_t)s * w) return;
    const size_t r = i / w, c = (i % w) * 4;
    int m[49];
#pragma unroll
    for (int K = 0; K < 49; K++) m[K] = *(const int *)&M[(size_t)K * s * s + r * s + c];
#pragma unroll 4
    for (int t = 0; t < 16; t++) {
        int v[4] = {};
        for (int e = 0; e < cC[t].n; e++)
#pragma unroll
            for (int b = 0; b < 4; b++) v[b] += cC[t].sgn[e] * lane(m[cC[t].idx[e]], b);
        int o = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) o |= (modp(v[b]) & 0xff) << (8 * b);
        *(int *)&C[((t / 4) * (size_t)s + r) * ldc + (t % 4) * s + c] = o;
    }
}

// the 49 leaf products in one launch: blockIdx.z picks the product, operands and results packed s x s
__global__ void __launch_bounds__(oz::THREADS) leafz(const int8_t *A, const int8_t *Bt, int8_t *C, int s, int g) {
    const size_t o = (size_t)blockIdx.z * s * s;
    oz::gemm_i8_tile(A + o, Bt + o, s, [&](int r, int c, int c00, int c01, int c10, int c11) {
        *(char2 *)&C[o + (size_t)r * s + c] = make_char2(modp(c00), modp(c01));
        *(char2 *)&C[o + (size_t)(r + 8) * s + c] = make_char2(modp(c10), modp(c11));
    }, s, s, g);
}

static void two_level(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h) {
    const int s = h / 4; const unsigned g = (unsigned)(((size_t)s * (s / 4) + 255) / 256);
    two_ops<false><<<g, 256>>>(A, lda, s, two.SA);
    two_ops<true><<<g, 256>>>(Bt, ldb, s, two.SB);
    leafz<<<dim3(s / oz::BN, s / oz::BM, 49), oz::THREADS>>>(two.SA, two.SB, two.M, s, G);
    two_out<<<g, 256>>>(two.M, s, C, ldc);
}

// C = A B mod p, h x h, L Strassen levels (B transposed: block B_ij of B is block (j, i) of Bt)
static void strassen(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L,
                     std::vector<Ws> &ws, int d) {
    if (L == 0) { leaf<<<dim3(h / oz::BN, h / oz::BM), oz::THREADS, 0, pp.mm>>>(A, lda, Bt, ldb, C, ldc, h, G); return; }
    if (L == 2 && two.on && h <= two.maxh) { two_level(A, lda, Bt, ldb, C, ldc, h); return; }
    const int q = h / 2;
    const int8_t *A11 = A, *A12 = A + q, *A21 = A + (size_t)q * lda, *A22 = A21 + q;
    const int8_t *B11 = Bt, *B12 = Bt + (size_t)q * ldb, *B21 = Bt + q, *B22 = B12 + q;
    int8_t *C11 = C, *C12 = C + q, *C21 = C + (size_t)q * ldc, *C22 = C21 + q;
    int8_t *S, *T, *M; int j = 0;
    const unsigned g = grid16(q);
    auto next = [&](int k) {   // product k takes set k & 1; its S/T may be rewritten once the leaves of product k - 2 are done
        j = pp.on ? k & 1 : 0; S = ws[2 * d + j].S; T = ws[2 * d + j].T; M = ws[2 * d + j].M;
        if (pp.on) cudaStreamWaitEvent(pp.add, evt(d, j, 1));
    };
    auto sum = [&](const int8_t *X, int ldx, const int8_t *Y, int ldy, int s, int8_t *Z) { comb<<<g, 256, 0, pp.add>>>(X, ldx, Y, ldy, s, Z, q, q); };
    auto prod = [&](const int8_t *X, int ldx, const int8_t *Y, int ldy) {
        if (pp.on) { cudaEventRecord(evt(d, j, 0), pp.add); cudaStreamWaitEvent(pp.mm, evt(d, j, 0)); cudaStreamWaitEvent(pp.mm, evt(d, j, 2)); }
        strassen(X, ldx, Y, ldy, M, q, q, L - 1, ws, d + 1);
        if (pp.on) cudaEventRecord(evt(d, j, 1), pp.mm);
    };
    auto put = [&](int8_t *D1, int s1, bool z1, int8_t *D2, int s2, bool z2) {
        if (pp.on) cudaStreamWaitEvent(pp.ac, evt(d, j, 1));
        acc<<<g, 256, 0, pp.ac>>>(M, q, D1, s1, z1, D2, s2, z2, ldc, q);
        if (pp.on) cudaEventRecord(evt(d, j, 2), pp.ac);
    };
    next(0); sum(A11, lda, A22, lda, 1, S); sum(B11, ldb, B22, ldb, 1, T); prod(S, q, T, q); put(C11, 1, true, C22, 1, true);    // M1
    next(1); sum(A21, lda, A22, lda, 1, S); prod(S, q, B11, ldb); put(C21, 1, true, C22, -1, false);                              // M2
    next(2); sum(B12, ldb, B22, ldb, -1, T); prod(A11, lda, T, q); put(C12, 1, true, C22, 1, false);                              // M3
    next(3); sum(B21, ldb, B11, ldb, -1, T); prod(A22, lda, T, q); put(C11, 1, false, C21, 1, false);                             // M4
    next(4); sum(A11, lda, A12, lda, 1, S); prod(S, q, B22, ldb); put(C11, -1, false, C12, 1, false);                             // M5
    next(5); sum(A21, lda, A11, lda, -1, S); sum(B11, ldb, B12, ldb, 1, T); prod(S, q, T, q); put(C22, 1, false, nullptr, 0, false);  // M6
    next(6); sum(A12, lda, A22, lda, -1, S); sum(B21, ldb, B22, ldb, 1, T); prod(S, q, T, q); put(C11, 1, false, nullptr, 0, false);  // M7
}


__global__ void fill(int8_t *X, size_t n, unsigned seed, int lo, int hi) {   // uniform in [lo, hi], a subrange of the residues
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    X[i] = (int8_t)(lo + (int)(x % (unsigned)(hi - lo + 1)));
}

__global__ void diff(const int8_t *X, const int8_t *Y, size_t n, unsigned long long *bad) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && X[i] != Y[i]) atomicAdd(bad, 1ull);
}

int main(int argc, char **argv) {
    const int maxL = argc > 1 ? atoi(argv[1]) : 4;
    if (argc > 2) G = atoi(argv[2]);
    // argv[3], argv[4]: operand range (default all residues). The GEMM is power-bound, and fewer toggling bits draw fewer watts
    const int lo = argc > 4 ? atoi(argv[3]) : -HP, hi = argc > 4 ? atoi(argv[4]) : HP;
    if (lo < -HP || hi > HP || lo > hi) { printf("range must lie in [%d, %d]\n", -HP, HP); return 1; }
    printf("block band G=%d, operands in [%d, %d]\n", G, lo, hi);
    int plo, phi; cudaDeviceGetStreamPriorityRange(&plo, &phi);   // phi = the highest priority: sums and puts get SMs first
    CK(cudaStreamCreateWithPriority(&pp.add, cudaStreamDefault, phi)); CK(cudaStreamCreateWithPriority(&pp.ac, cudaStreamDefault, phi));
    CK(cudaStreamCreate(&pp.mm));   // blocking streams: the legacy stream (timing events, diff) waits for all three
    two_tables();
    pp.ev.resize(2 * 3 * (maxL + 1));
    for (auto &ev : pp.ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    for (int n : {8192, 16384, 32768}) {
        const size_t nn = (size_t)n * n;
        int8_t *A, *Bt, *C0, *C;
        CK(cudaMalloc(&A, nn)); CK(cudaMalloc(&Bt, nn)); CK(cudaMalloc(&C0, nn)); CK(cudaMalloc(&C, nn));
        fill<<<(nn + 255) / 256, 256>>>(A, nn, 12345u, lo, hi); fill<<<(nn + 255) / 256, 256>>>(Bt, nn, 67890u, lo, hi);
        std::vector<Ws> ws;
        for (int h = n / 2, d = 0; d < maxL; d++, h /= 2) for (int j = 0; j < 2; j++) {
            Ws w; size_t s = (size_t)h * h;
            CK(cudaMalloc(&w.S, s)); CK(cudaMalloc(&w.T, s)); CK(cudaMalloc(&w.M, s)); ws.push_back(w);
        }
        if (maxL >= 2) {   // one two-level node at a time: 49 operands, 49 products, (h/4)^2 each, for h up to 16384 (12 GB card)
            two.maxh = n < 16384 ? n : 16384; const size_t s = two.maxh / 4, b = 49 * s * s;
            CK(cudaMalloc(&two.SA, b)); CK(cudaMalloc(&two.SB, b)); CK(cudaMalloc(&two.M, b));
        }
        unsigned long long *bad; CK(cudaMalloc(&bad, 8));
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int L = 0; L <= maxL && (n >> L) >= 1024; L++) for (int mode = 0; mode <= (L >= 2 ? 2 : L); mode++) {   // 0 serial, 1 pipelined, 2 two levels per pass
            pp.on = mode == 1; two.on = mode == 2;
            Pipe saved = pp; if (!pp.on) pp.add = pp.mm = pp.ac = 0;
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
            for (size_t j = 0; j < leaves; j++) leaf<<<dim3(h / oz::BN, h / oz::BM), oz::THREADS>>>(A, n, Bt, n, C, n, h, G);
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
            CK(cudaGetLastError()); pp = saved;
            printf("n=%5d L=%d %s %8.2f ms effective %6.1f TOPS  (leaf GEMMs alone %8.2f ms = %5.1f TOPS on %zu x %d^3)  differing %llu\n",
                   n, L, mode == 2 ? "two   " : mode ? "pipe  " : "serial", ms, 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12, gm, 2.0 * leaves * h * h * (double)h / (gm * 1e-3) / 1e12,
                   leaves, h, nb);
            fflush(stdout);
        }
        for (auto &w : ws) { cudaFree(w.S); cudaFree(w.T); cudaFree(w.M); }
        cudaFree(two.SA); cudaFree(two.SB); cudaFree(two.M); two = Two();
        cudaFree(A); cudaFree(Bt); cudaFree(C0); cudaFree(C); cudaFree(bad);
    }
    return 0;
}
