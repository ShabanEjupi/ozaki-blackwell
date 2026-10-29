// strassen_p.cu -- Strassen over Z/p on the INT8 tensor cores: C = A B mod p with 7^L instead of 8^L block products.
// Ozaki-II multiplies residues mod p < 256. A sum of two residues, reduced mod p, is again a residue, so Strassen's operand
// sums stay int8 at every level (over the integers each level would add a bit and leave int8 after the first). Every L is
// checked bit for bit against the classical product (L = 0); the effective rate is 2 n^3 / time, the classical op count.
// Random residues (not constant fill): the GEMM is power-bound on this laptop, and constant data draws less power.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>
#include <cuda_runtime.h>
#include "../src/gemm_i8.cuh"
#define CK(x) do { cudaError_t ck_ = (x); if (ck_) { printf("%s: %s\n", #x, cudaGetErrorString(ck_)); exit(1); } } while (0)

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
struct TwoBuf { int8_t *SA, *SB, *M, *SA2, *SB2, *M2; int cap; };   // set 2: the overlapped schedule below
struct Two { bool on = false, ov = false; std::vector<TwoBuf> tb; cudaStream_t st[3]; cudaEvent_t ev[2][3]; };
static Two two;

struct Term { int n, idx[16], sgn[16]; };          // a linear combination of sub-blocks (A, Bt) or products (C)
struct Tab { Term a[49], b[49], c[16]; };
// M1..M7 exactly as in strassen() below; quadrant index 2 r + c (11, 12, 21, 22)
__host__ __device__ constexpr Tab two_tab() {
    const int U[7][4] = {{1, 0, 0, 1}, {0, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {-1, 0, 1, 0}, {0, 1, 0, -1}};
    const int V[7][4] = {{1, 0, 0, 1}, {1, 0, 0, 0}, {0, 1, 0, -1}, {-1, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {0, 0, 1, 1}};
    const int W[4][7] = {{1, 0, 0, 1, -1, 0, 1}, {0, 0, 1, 0, 1, 0, 0}, {0, 1, 0, 1, 0, 0, 0}, {1, -1, 1, 0, 0, 1, 0}};
    Tab t{};
    for (int k1 = 0; k1 < 7; k1++) for (int k2 = 0; k2 < 7; k2++) {
        const int K = 7 * k1 + k2;
        for (int R = 0; R < 4; R++) for (int C = 0; C < 4; C++) {
            const int q1 = 2 * (R / 2) + C / 2, q2 = 2 * (R % 2) + C % 2, X = 4 * R + C;
            if (int s = U[k1][q1] * U[k2][q2]) { t.a[K].idx[t.a[K].n] = X; t.a[K].sgn[t.a[K].n++] = s; }
            if (int s = V[k1][q1] * V[k2][q2]) { t.b[K].idx[t.b[K].n] = 4 * C + R; t.b[K].sgn[t.b[K].n++] = s; }   // B (R, C) = Bt (C, R)
            if (int s = W[q1][k1] * W[q2][k2]) { t.c[X].idx[t.c[X].n] = K; t.c[X].sgn[t.c[X].n++] = s; }
        }
    }
    return t;
}

__device__ __forceinline__ int lane(int w, int b) { return (int8_t)(w >> (8 * b)); }
// v mod p in [-125, 125] for |v| < 2^22: v - p round(v / p), exact in float there (p odd: no ties)
__device__ __forceinline__ int fold(int v) { return v - P * __float2int_rn((float)v * (1.0f / P)); }

// Every table entry is a template argument, so the table is evaluated by the compiler only: no runtime index into the 16
// sub-blocks or 49 products, which then stay in registers (runtime indices put them in local memory: 2-3x slower)
template <int W, int K, int E> __device__ __forceinline__ int tterm(const int *x, int b) {   // W: 0 A, 1 Bt, 2 C
    constexpr Term T = W == 0 ? two_tab().a[K] : W == 1 ? two_tab().b[K] : two_tab().c[K];
    if constexpr (E < T.n) return T.sgn[E] * lane(x[T.idx[E]], b); else return 0;
}
template <int W, int K, int... E> __device__ __forceinline__ int tsum(const int *x, int b, std::integer_sequence<int, E...>) {
    return (0 + ... + tterm<W, K, E>(x, b));
}
template <int W, int K> __device__ __forceinline__ int tpack(const int *x) {   // 4 lanes of combination K, mod p, packed
    int o = 0;
#pragma unroll
    for (int b = 0; b < 4; b++) o |= (fold(tsum<W, K>(x, b, std::make_integer_sequence<int, 16>{})) & 0xff) << (8 * b);
    return o;
}
template <int W, int... K> __device__ __forceinline__ void tops(const int (*x)[16], int K0, int cnt, int8_t *O, size_t ss, size_t pos, std::integer_sequence<int, K...>) {
    ((K >= K0 && K < K0 + cnt ? (void)(*(int4 *)&O[(size_t)(K - K0) * ss + pos] = make_int4(tpack<W, K>(x[0]), tpack<W, K>(x[1]), tpack<W, K>(x[2]), tpack<W, K>(x[3]))) : (void)0), ...);
}
template <int... T> __device__ __forceinline__ void touts(const int (*m)[49], int8_t *C, int s, int ldc, size_t r, size_t c, std::integer_sequence<int, T...>) {
    ((*(int2 *)&C[((T / 4) * (size_t)s + r) * ldc + (T % 4) * s + c] = make_int2(tpack<2, T>(m[0]), tpack<2, T>(m[1]))), ...);
}

// O[K - K0] (s x s, packed) = the sub-blocks of X that combination K names, summed mod p; 16 bytes per thread
template <bool isB>
__global__ void two_ops(const int8_t *X, int ld, int s, int8_t *O, int K0, int cnt) {   // combinations K0 .. K0 + cnt - 1
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = s / 16;
    if (i >= (size_t)s * w) return;
    const size_t r = i / w, c = (i % w) * 16;
    int x[4][16];
#pragma unroll
    for (int t = 0; t < 16; t++) {
        const int4 v = *(const int4 *)&X[((t / 4) * (size_t)s + r) * ld + (t % 4) * s + c];
        x[0][t] = v.x; x[1][t] = v.y; x[2][t] = v.z; x[3][t] = v.w;
    }
    tops<isB ? 1 : 0>(x, K0, cnt, O, (size_t)s * s, r * s + c, std::make_integer_sequence<int, 49>{});
}

// C's 16 sub-blocks = their combinations of the 49 products, mod p (written, never read); 8 bytes per thread
__global__ void two_out(const int8_t *M, int s, int8_t *C, int ldc) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = s / 8;
    if (i >= (size_t)s * w) return;
    const size_t r = i / w, c = (i % w) * 8;
    int m[2][49];
#pragma unroll
    for (int K = 0; K < 49; K++) { const int2 v = *(const int2 *)&M[(size_t)K * s * s + r * s + c]; m[0][K] = v.x; m[1][K] = v.y; }
    touts(m, C, s, ldc, r, c, std::make_integer_sequence<int, 16>{});
}


// the 49 leaf products in one launch: blockIdx.z picks the product, operands and results packed s x s
__global__ void __launch_bounds__(oz::THREADS) leafz(const int8_t *A, const int8_t *Bt, int8_t *C, int s, int g) {
    const size_t o = (size_t)blockIdx.z * s * s;
    oz::gemm_i8_tile(A + o, Bt + o, s, [&](int r, int c, int c00, int c01, int c10, int c11) {
        *(char2 *)&C[o + (size_t)r * s + c] = make_char2(modp(c00), modp(c01));
        *(char2 *)&C[o + (size_t)(r + 8) * s + c] = make_char2(modp(c10), modp(c11));
    }, s, s, g);
}

// C = A B mod p with L (even) levels, two per pass; the 49 products are leaves (L = 2, one batched launch) or two_rec again
static float prof[3]; static bool profon = false;   // ms in operand sums, products, output sums (depth 0, L = 2)
static cudaEvent_t pe[4];
static void two_rec(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L, int t) {
    const int s = h / 4; const size_t ss = (size_t)s * s; const unsigned g = (unsigned)((ss / 16 + 255) / 256), go = (unsigned)((ss / 8 + 255) / 256);
    const TwoBuf &b = two.tb[t];
    for (int K0 = 0; K0 < 49; K0 += b.cap) {
        const int cnt = 49 - K0 < b.cap ? 49 - K0 : b.cap;
        const bool pr = profon && L == 2;
        if (pr) cudaEventRecord(pe[0]);
        two_ops<false><<<g, 256>>>(A, lda, s, b.SA, K0, cnt);
        two_ops<true><<<g, 256>>>(Bt, ldb, s, b.SB, K0, cnt);
        if (pr) cudaEventRecord(pe[1]);
        if (L == 2) leafz<<<dim3(s / oz::BN, s / oz::BM, cnt), oz::THREADS>>>(b.SA, b.SB, b.M + K0 * ss, s, G);
        else if (L == 4 && two.ov) {   // children are two-level nodes: their passes (DRAM) overlap the products (tensor cores)
            // of their neighbours. Child z uses set z & 1; a set's operands are rewritten once its products are done (ev 1),
            // its products once its output pass is done (ev 2). The legacy stream (parent passes) waits for all three.
            const TwoBuf &c = two.tb[t + 1]; const int sc = s / 4; const size_t ssc = (size_t)sc * sc;
            const unsigned gc = (unsigned)((ssc / 16 + 255) / 256), gco = (unsigned)((ssc / 8 + 255) / 256);
            for (int z = 0; z < cnt; z++) {
                const int j = z & 1; int8_t *SA = j ? c.SA2 : c.SA, *SB = j ? c.SB2 : c.SB, *M = j ? c.M2 : c.M;
                cudaStreamWaitEvent(two.st[0], two.ev[j][1]);
                two_ops<false><<<gc, 256, 0, two.st[0]>>>(b.SA + z * ss, s, sc, SA, 0, 49);
                two_ops<true><<<gc, 256, 0, two.st[0]>>>(b.SB + z * ss, s, sc, SB, 0, 49);
                cudaEventRecord(two.ev[j][0], two.st[0]);
                cudaStreamWaitEvent(two.st[1], two.ev[j][0]); cudaStreamWaitEvent(two.st[1], two.ev[j][2]);
                leafz<<<dim3(sc / oz::BN, sc / oz::BM, 49), oz::THREADS, 0, two.st[1]>>>(SA, SB, M, sc, G);
                cudaEventRecord(two.ev[j][1], two.st[1]);
                cudaStreamWaitEvent(two.st[2], two.ev[j][1]);
                two_out<<<gco, 256, 0, two.st[2]>>>(M, sc, b.M + (K0 + z) * ss, s);
                cudaEventRecord(two.ev[j][2], two.st[2]);
            }
        }
        else for (int z = 0; z < cnt; z++) two_rec(b.SA + z * ss, s, b.SB + z * ss, s, b.M + (K0 + z) * ss, s, s, L - 2, t + 1);
    }
    if (profon && L == 2) cudaEventRecord(pe[2]);
    two_out<<<go, 256>>>(b.M, s, C, ldc);
    if (profon && L == 2) { float a, b2, c2; cudaEventRecord(pe[3]); cudaEventSynchronize(pe[3]); cudaEventElapsedTime(&a, pe[0], pe[1]);
        cudaEventElapsedTime(&b2, pe[1], pe[2]); cudaEventElapsedTime(&c2, pe[2], pe[3]); prof[0] += a; prof[1] += b2; prof[2] += c2; }
}

// C = A B mod p, h x h, L Strassen levels (B transposed: block B_ij of B is block (j, i) of Bt)
static void strassen(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L,
                     std::vector<Ws> &ws, int d) {
    if (L == 0) { leaf<<<dim3(h / oz::BN, h / oz::BM), oz::THREADS, 0, pp.mm>>>(A, lda, Bt, ldb, C, ldc, h, G); return; }
    if (two.on && L % 2 == 0) { two_rec(A, lda, Bt, ldb, C, ldc, h, L, 0); return; }   // odd L: this level as below, then pairs
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

// differing bytes between the device result and the host reference (memcmp per 64 KB, counted only where they differ)
static unsigned long long differing(const int8_t *C, const std::vector<int8_t> &ref, std::vector<int8_t> &host) {
    CK(cudaMemcpy(host.data(), C, host.size(), cudaMemcpyDeviceToHost));
    unsigned long long nb = 0; const size_t B = 1 << 16;
    for (size_t o = 0; o < host.size(); o += B) {
        const size_t m = host.size() - o < B ? host.size() - o : B;
        if (memcmp(&host[o], &ref[o], m)) for (size_t i = o; i < o + m; i++) nb += host[i] != ref[i];
    }
    return nb;
}

int main(int argc, char **argv) {
    // strassen_p [maxL [G [lo hi [modes [n]]]]]: modes = bit mask, 1 serial, 2 pipelined, 4 two levels per pass, 8 the same overlapped
    const int maxL = argc > 1 ? atoi(argv[1]) : 4;
    if (argc > 2) G = atoi(argv[2]);
    // argv[3], argv[4]: operand range (default all residues). The GEMM is power-bound, and fewer toggling bits draw fewer watts
    const int lo = argc > 4 ? atoi(argv[3]) : -HP, hi = argc > 4 ? atoi(argv[4]) : HP;
    if (lo < -HP || hi > HP || lo > hi) { printf("range must lie in [%d, %d]\n", -HP, HP); return 1; }
    const int modes = argc > 5 ? atoi(argv[5]) : 7;
    std::vector<int> sizes = {8192, 16384, 32768};
    if (argc > 6) sizes = {atoi(argv[6])};
    printf("block band G=%d, operands in [%d, %d], modes %d\n", G, lo, hi, modes);
    int plo, phi; cudaDeviceGetStreamPriorityRange(&plo, &phi);   // phi = the highest priority: sums and puts get SMs first
    CK(cudaStreamCreateWithPriority(&pp.add, cudaStreamDefault, phi)); CK(cudaStreamCreateWithPriority(&pp.ac, cudaStreamDefault, phi));
    CK(cudaStreamCreate(&pp.mm));   // blocking streams: the legacy stream (timing events, memcpy) waits for all three
    pp.ev.resize(2 * 3 * (maxL + 1));
    for (auto &ev : pp.ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CK(cudaStreamCreateWithPriority(&two.st[0], cudaStreamDefault, phi)); CK(cudaStreamCreate(&two.st[1]));
    CK(cudaStreamCreateWithPriority(&two.st[2], cudaStreamDefault, phi));
    for (auto &r : two.ev) for (auto &e : r) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    for (int n : sizes) {
        const size_t nn = (size_t)n * n;
        int8_t *A, *Bt, *C;   // the reference C lives on the host: the device memory goes to the two-level buffers
        CK(cudaMalloc(&A, nn)); CK(cudaMalloc(&Bt, nn)); CK(cudaMalloc(&C, nn));
        std::vector<int8_t> ref(nn), host(nn);
        fill<<<(nn + 255) / 256, 256>>>(A, nn, 12345u, lo, hi); fill<<<(nn + 255) / 256, 256>>>(Bt, nn, 67890u, lo, hi);
        std::vector<Ws> ws;   // a level done as below: set 0, and set 1 only when the pipelined schedule runs
        for (int h = n / 2, d = 0; d < maxL; d++, h /= 2) for (int j = 0; j < 2; j++) {
            Ws w = {}; size_t s = (size_t)h * h;
            if (j == 0 || (modes & 2)) { CK(cudaMalloc(&w.S, s)); CK(cudaMalloc(&w.T, s)); CK(cudaMalloc(&w.M, s)); }
            ws.push_back(w);
        }
        if ((modes & 12) && maxL >= 2) {   // depth t holds nodes of n / 4^t: deeper ones in full, depth 0 in chunks that fit
            const int T = maxL / 2; two.tb.assign(T, TwoBuf{});
            for (int t = T - 1; t >= 0; t--) {
                const size_t s = (size_t)(n >> (2 * t + 2)), ss = s * s; int cap = 49;
                if (!t) { size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot)); const size_t keep = 512ull << 20;
                          cap = fr > keep + 49 * ss ? (int)((fr - keep - 49 * ss) / (2 * ss)) : 0; cap = cap > 49 ? 49 : cap; }
                if (cap < 1) { printf("n=%d: no room for the two-level buffers\n", n); return 1; }
                CK(cudaMalloc(&two.tb[t].SA, cap * ss)); CK(cudaMalloc(&two.tb[t].SB, cap * ss)); CK(cudaMalloc(&two.tb[t].M, 49 * ss));
                if (t && (modes & 8)) { CK(cudaMalloc(&two.tb[t].SA2, cap * ss)); CK(cudaMalloc(&two.tb[t].SB2, cap * ss)); CK(cudaMalloc(&two.tb[t].M2, 49 * ss)); }
                two.tb[t].cap = cap;
            }
            printf("n=%5d two-level chunk at depth 0: %d of 49 products\n", n, two.tb[0].cap);
        }
        bool mutant = false;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int L = 0; L <= maxL && (n >> L) >= 1024; L++) for (int mode = 0; mode < 4; mode++) {   // 0 serial, 1 pipelined, 2 two levels per pass, 3 overlapped
            if (L ? !(modes >> mode & 1) || (mode == 1 && L < 1) || (mode >= 2 && L < 2) : mode) continue;
            pp.on = mode == 1; two.on = mode >= 2; two.ov = mode == 3;
            Pipe saved = pp; if (!pp.on) pp.add = pp.mm = pp.ac = 0;
            strassen(A, n, Bt, n, C, n, n, L, ws, 0);   // warm-up (and the L = 0 reference)
            const int reps = n >= 32768 ? 2 : 3;
            cudaEventRecord(t0);
            for (int r = 0; r < reps; r++) strassen(A, n, Bt, n, C, n, n, L, ws, 0);
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1));
            float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
            unsigned long long nb = 0;
            if (!L) {   // the reference itself, independently: 64 entries on the CPU from their row of A and row of Bt
                CK(cudaMemcpy(ref.data(), C, nn, cudaMemcpyDeviceToHost));
                std::vector<int8_t> ra(n), rb(n); int wrong = 0; srand(n);
                for (int e = 0; e < 64; e++) {
                    size_t r = rand() % n, c = rand() % n;
                    CK(cudaMemcpy(ra.data(), A + r * n, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(rb.data(), Bt + c * n, n, cudaMemcpyDeviceToHost));
                    long long s = 0; for (int k = 0; k < n; k++) s += ra[k] * rb[k];
                    int m = (int)(s % P); m = m > HP ? m - P : m < -HP ? m + P : m;
                    wrong += m != ref[r * n + c];
                }
                printf("n=%5d reference: %d of 64 CPU-checked entries wrong\n", n, wrong);
                if (wrong) return 1;
            } else {
                nb = differing(C, ref, host);   // C holds the last timed run
                if (!mutant) {   // once per n, on the first checked product: mutant: one byte changed must be counted, or "differing 0" means nothing
                    int8_t v; CK(cudaMemcpy(&v, C + nn / 3, 1, cudaMemcpyDeviceToHost)); int8_t w = v == HP ? -HP : v + 1;
                    CK(cudaMemcpy(C + nn / 3, &w, 1, cudaMemcpyHostToDevice));
                    if (differing(C, ref, host) != nb + 1) { printf("mutant not caught\n"); return 1; }
                    mutant = true;
                }
            }
            CK(cudaGetLastError()); pp = saved;
            printf("n=%5d L=%d %s %8.2f ms  effective %6.1f TOPS  differing %llu\n",
                   n, L, mode == 3 ? "two-ov" : mode == 2 ? "two   " : mode ? "pipe  " : "serial", ms, 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12, nb);
            if (mode == 2 && getenv("OZ_PROF")) {   // one more run, synchronised per pass: where the time goes (last chunk only when chunked)
                for (auto &e : pe) cudaEventCreate(&e);
                prof[0] = prof[1] = prof[2] = 0; profon = true; two.on = true; strassen(A, n, Bt, n, C, n, n, L, ws, 0); profon = false; two.on = false;
                printf("          two-level passes (L = 2 nodes): operand sums %.2f ms, products %.2f ms, output sums %.2f ms\n", prof[0], prof[1], prof[2]);
            }
            fflush(stdout);
        }
        for (auto &w : ws) { cudaFree(w.S); cudaFree(w.T); cudaFree(w.M); }
        for (auto &b : two.tb) { cudaFree(b.SA); cudaFree(b.SB); cudaFree(b.M); cudaFree(b.SA2); cudaFree(b.SB2); cudaFree(b.M2); }
        two.tb.clear();
        cudaFree(A); cudaFree(Bt); cudaFree(C);
    }
    return 0;
}
