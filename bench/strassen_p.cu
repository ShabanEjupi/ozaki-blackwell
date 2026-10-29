// strassen_p.cu -- Strassen over Z/p on the INT8 tensor cores: C = A B mod p with 7^L instead of 8^L block products.
// Ozaki-II multiplies residues mod p < 256. A sum of two residues, reduced mod p, is again a residue, so Strassen's operand
// sums stay int8 at every level (over the integers each level would add a bit and leave int8 after the first). Every L is
// checked bit for bit against the classical product (L = 0); the effective rate is 2 n^3 / time, the classical op count.
// Random residues (not constant fill): the GEMM is power-bound on this laptop, and constant data draws less power.
// The two-level kernels live in src/strassen_mod.cuh (any modulus; oz_strassen.cu runs them inside Ozaki-II). The schedule
// that overlapped child passes with the products (correct, gains 0: 433.7 vs 433.2 ms) is in commit 07a9bf7.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include "../src/strassen_mod.cuh"
#define CK(x) do { cudaError_t ck_ = (x); if (ck_) { printf("%s: %s\n", #x, cudaGetErrorString(ck_)); exit(1); } } while (0)

constexpr int P = 251, HP = (P - 1) / 2;   // the largest prime below 256; residues in [-125, 125]
static oz::Mod q;                           // = oz::mod_of(P)
static oz::StrassenMod sm;                  // leaf tile, block band, two-level buffers

// Z = X + s Y mod p (h x h, 16 bytes per thread)
__global__ void comb(const int8_t *X, int ldx, const int8_t *Y, int ldy, int s, int8_t *Z, int ldz, int h, oz::Mod q) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x, w = h / 16;
    if (i >= (size_t)h * w) return;
    size_t r = i / w, c = (i % w) * 16;
    int4 x = *(const int4 *)&X[r * ldx + c], y = *(const int4 *)&Y[r * ldy + c], z;
    const int8_t *px = (const int8_t *)&x, *py = (const int8_t *)&y; int8_t *pz = (int8_t *)&z;
#pragma unroll
    for (int k = 0; k < 16; k++) pz[k] = (int8_t)oz::fold(px[k] + s * py[k], q);
    *(int4 *)&Z[r * ldz + c] = z;
}

// C1 = [C1 +] s1 M, C2 = [C2 +] s2 M (mod p); "set" overwrites instead of adding (first touch of a quadrant)
__global__ void acc(const int8_t *M, int ldm, int8_t *C1, int s1, bool set1, int8_t *C2, int s2, bool set2, int ldc, int h, oz::Mod q) {
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
        for (int k = 0; k < 16; k++) po[k] = (int8_t)oz::fold(po[k] + S[d] * pm[k], q);
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
static bool two_on = false;                          // even L: two levels per pass (strassen_mod.cuh)
static float prof[3]; static float *profp = nullptr;  // ms in operand sums, products, output sums of the L = 2 nodes

// C = A B mod p, h x h, L Strassen levels (B transposed: block B_ij of B is block (j, i) of Bt)
static void strassen(const int8_t *A, int lda, const int8_t *Bt, int ldb, int8_t *C, int ldc, int h, int L,
                     std::vector<Ws> &ws, int d) {
    if (L == 0) { sm.leaf_run(A, lda, Bt, ldb, C, ldc, h, 0, q, pp.mm); return; }
    if (two_on && L % 2 == 0) { sm.two_rec(A, lda, Bt, ldb, C, ldc, h, L, 0, q, profp); return; }   // odd L: this level as below, then pairs
    const int q2 = h / 2;
    const int8_t *A11 = A, *A12 = A + q2, *A21 = A + (size_t)q2 * lda, *A22 = A21 + q2;
    const int8_t *B11 = Bt, *B12 = Bt + (size_t)q2 * ldb, *B21 = Bt + q2, *B22 = B12 + q2;
    int8_t *C11 = C, *C12 = C + q2, *C21 = C + (size_t)q2 * ldc, *C22 = C21 + q2;
    int8_t *S, *T, *M; int j = 0;
    const unsigned g = grid16(q2);
    auto next = [&](int k) {   // product k takes set k & 1; its S/T may be rewritten once the leaves of product k - 2 are done
        j = pp.on ? k & 1 : 0; S = ws[2 * d + j].S; T = ws[2 * d + j].T; M = ws[2 * d + j].M;
        if (pp.on) cudaStreamWaitEvent(pp.add, evt(d, j, 1));
    };
    auto sum = [&](const int8_t *X, int ldx, const int8_t *Y, int ldy, int s, int8_t *Z) { comb<<<g, 256, 0, pp.add>>>(X, ldx, Y, ldy, s, Z, q2, q2, q); };
    auto prod = [&](const int8_t *X, int ldx, const int8_t *Y, int ldy) {
        if (pp.on) { cudaEventRecord(evt(d, j, 0), pp.add); cudaStreamWaitEvent(pp.mm, evt(d, j, 0)); cudaStreamWaitEvent(pp.mm, evt(d, j, 2)); }
        strassen(X, ldx, Y, ldy, M, q2, q2, L - 1, ws, d + 1);
        if (pp.on) cudaEventRecord(evt(d, j, 1), pp.mm);
    };
    auto put = [&](int8_t *D1, int s1, bool z1, int8_t *D2, int s2, bool z2) {
        if (pp.on) cudaStreamWaitEvent(pp.ac, evt(d, j, 1));
        acc<<<g, 256, 0, pp.ac>>>(M, q2, D1, s1, z1, D2, s2, z2, ldc, q2, q);
        if (pp.on) cudaEventRecord(evt(d, j, 2), pp.ac);
    };
    next(0); sum(A11, lda, A22, lda, 1, S); sum(B11, ldb, B22, ldb, 1, T); prod(S, q2, T, q2); put(C11, 1, true, C22, 1, true);    // M1
    next(1); sum(A21, lda, A22, lda, 1, S); prod(S, q2, B11, ldb); put(C21, 1, true, C22, -1, false);                               // M2
    next(2); sum(B12, ldb, B22, ldb, -1, T); prod(A11, lda, T, q2); put(C12, 1, true, C22, 1, false);                               // M3
    next(3); sum(B21, ldb, B11, ldb, -1, T); prod(A22, lda, T, q2); put(C11, 1, false, C21, 1, false);                              // M4
    next(4); sum(A11, lda, A12, lda, 1, S); prod(S, q2, B22, ldb); put(C11, -1, false, C12, 1, false);                              // M5
    next(5); sum(A21, lda, A11, lda, -1, S); sum(B11, ldb, B12, ldb, 1, T); prod(S, q2, T, q2); put(C22, 1, false, nullptr, 0, false);  // M6
    next(6); sum(A12, lda, A22, lda, -1, S); sum(B21, ldb, B22, ldb, 1, T); prod(S, q2, T, q2); put(C11, 1, false, nullptr, 0, false);  // M7
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
    // strassen_p [maxL [G [lo hi [modes [n]]]]]: modes = bit mask, 1 serial, 2 pipelined, 4 two levels per pass
    const int maxL = argc > 1 ? atoi(argv[1]) : 4;
    if (argc > 2) sm.G = atoi(argv[2]);
    // argv[3], argv[4]: operand range (default all residues). The GEMM is power-bound, and fewer toggling bits draw fewer watts
    const int lo = argc > 4 ? atoi(argv[3]) : -HP, hi = argc > 4 ? atoi(argv[4]) : HP;
    if (lo < -HP || hi > HP || lo > hi) { printf("range must lie in [%d, %d]\n", -HP, HP); return 1; }
    const int modes = argc > 5 ? atoi(argv[5]) : 7;
    std::vector<int> sizes = {8192, 16384, 32768};
    if (argc > 6) sizes = {atoi(argv[6])};
    q = oz::mod_of(P);
    sm.tile = 0;
    if (const char *e = getenv("OZ_TILE")) sm.tile = !strcmp(e, "64x3") ? 1 : !strcmp(e, "64x4") ? 2 : 0;
    printf("block band G=%d, operands in [%d, %d], modes %d, tile %s\n", sm.G, lo, hi, modes, getenv("OZ_TILE") ? getenv("OZ_TILE") : "32x3");
    int plo, phi; cudaDeviceGetStreamPriorityRange(&plo, &phi);   // phi = the highest priority: sums and puts get SMs first
    CK(cudaStreamCreateWithPriority(&pp.add, cudaStreamDefault, phi)); CK(cudaStreamCreateWithPriority(&pp.ac, cudaStreamDefault, phi));
    CK(cudaStreamCreate(&pp.mm));   // blocking streams: the legacy stream (timing events, memcpy) waits for all three
    oz::StrassenMod::grant();
    pp.ev.resize(2 * 3 * (maxL + 1));
    for (auto &ev : pp.ev) CK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
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
        if ((modes & 4) && maxL >= 2) {   // depth t holds nodes of n / 4^t: deeper ones in full, depth 0 in chunks that fit
            const int L2 = maxL / 2 * 2; const size_t ss = (size_t)(n / 4) * (n / 4), keep = 512ull << 20, need = oz::StrassenMod::bytes(n, L2, 0);
            size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
            int cap = fr > keep + need ? (int)((fr - keep - need) / (2 * ss)) : 0; cap = cap > 49 ? 49 : cap;
            if (cap < 1) { printf("n=%d: no room for the two-level buffers\n", n); return 1; }
            sm.plan(n, L2, cap); CK(cudaGetLastError());
            printf("n=%5d two-level chunk at depth 0: %d of 49 products\n", n, cap);
        }
        bool mutant = false;
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        for (int L = 0; L <= maxL && (n >> L) >= 1024; L++) for (int mode = 0; mode < 3; mode++) {   // 0 serial, 1 pipelined, 2 two levels per pass
            if (L ? !(modes >> mode & 1) || (mode == 2 && L < 2) : mode) continue;
            pp.on = mode == 1; two_on = mode == 2;
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
                   n, L, mode == 2 ? "two   " : mode ? "pipe  " : "serial", ms, 2.0 * n * n * (double)n / (ms * 1e-3) / 1e12, nb);
            if (mode == 2 && getenv("OZ_PROF")) {   // one more run, synchronised per pass: where the time goes (last chunk only when chunked)
                prof[0] = prof[1] = prof[2] = 0; profp = prof; two_on = true; strassen(A, n, Bt, n, C, n, n, L, ws, 0); profp = nullptr; two_on = false;
                printf("          two-level passes (L = 2 nodes): operand sums %.2f ms, products %.2f ms, output sums %.2f ms\n", prof[0], prof[1], prof[2]);
            }
            fflush(stdout);
        }
        for (auto &w : ws) { cudaFree(w.S); cudaFree(w.T); cudaFree(w.M); }
        sm.release();
        cudaFree(A); cudaFree(Bt); cudaFree(C);
    }
    return 0;
}
