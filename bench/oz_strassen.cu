// oz_strassen.cu -- FP64 DGEMM from INT8 tensor cores (Ozaki-II) with every modular product done by Strassen over Z/m.
// C = A Bt^T (both operands K-contiguous, the NT GEMM), n x n x n, all on the 12 GB card:
//   scales, then per group of g moduli: residues of A and Bt -> one product mod m per modulus (L = 0: the classical
//   tensor-core GEMM with the reduction mod m in its epilogue; L = 2: two Strassen levels in one pass, 49 products) ->
//   int8 residue planes of C; then Garner + Horner (crt) from the int8 planes.
// The residues of C are the same for every L (exact arithmetic over Z/m), so the FP64 results must be bit-identical:
// checked on all planes and on C, with a mutant. Accuracy: 1024 sampled entries against a double-double reference,
// componentwise against |A||Bt^T| (the bound a classic FP64 GEMM is judged by), for this and for the GPU's own FP64.
// Energy: NVML (in process, dlopen) over a window of repeated DGEMMs: joules per DGEMM, FP64-equivalent GFLOPS per watt.
// Memory at n = 16384: A, Bt, C 2 GB each + nm int8 planes of C, which first hold A's residues for all moduli (one read of
// A; the result of modulus l overwrites plane l); Bt's planes, two moduli per read, and the Strassen buffers live inside C's
// FP64 buffer until the CRT overwrites it (operand chunks of 23 of 49).
//   oz_strassen [n [nm_list [spread]]]   e.g. oz_strassen 16384 12,13,14 0
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <random>
#include <thread>
#include <vector>
#include "../src/ozaki2.cuh"
#include "../src/strassen_mod.cuh"
#define CK(x) do { cudaError_t ck_ = (x); if (ck_) { printf("%s: %s (line %d)\n", #x, cudaGetErrorString(ck_), __LINE__); exit(1); } } while (0)

// GPU power through NVML, loaded at run time (the WSL toolkit here has no nvml.h): the power samples integrated every 2 ms.
// The driver's energy counter is read too, as a cross-check only: on this laptop it advances in steps of ~1 s of energy,
// so a 1-8 s window can read 20-150 % high (measured 29-10-2026: 296.8 W "average" against 112.3 W sampled)
struct Power {
    void *dev = nullptr; bool ok = false, counter = false;
    int (*pw)(void *, unsigned *) = nullptr; int (*en)(void *, unsigned long long *) = nullptr;
    std::atomic<bool> run{false}; std::thread th; double joules = 0, sampled = 0, counted = 0, secs = 0;   // joules = sampled: the integral of the samples
    Power() {
        void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW);
        if (!h) return;
        auto init = (int (*)())dlsym(h, "nvmlInit_v2");
        auto get = (int (*)(unsigned, void **))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
        pw = (int (*)(void *, unsigned *))dlsym(h, "nvmlDeviceGetPowerUsage");
        en = (int (*)(void *, unsigned long long *))dlsym(h, "nvmlDeviceGetTotalEnergyConsumption");
        ok = init && get && pw && !init() && !get(0, &dev);
        unsigned long long e; counter = ok && en && !en(dev, &e);
    }
    void start() {
        joules = sampled = 0; run = true;
        th = std::thread([this] {
            auto t0 = std::chrono::steady_clock::now(), tp = t0; unsigned long long e0 = 0, e1 = 0;
            if (counter) en(dev, &e0);
            unsigned mw = 0; pw(dev, &mw); double last = mw * 1e-3;
            while (run) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                auto t = std::chrono::steady_clock::now(); pw(dev, &mw); const double p = mw * 1e-3;
                sampled += 0.5 * (p + last) * std::chrono::duration<double>(t - tp).count(); last = p; tp = t;
            }
            secs = std::chrono::duration<double>(tp - t0).count();
            joules = sampled; if (counter) { en(dev, &e1); counted = (e1 - e0) * 1e-3; }
        });
    }
    void stop() { run = false; th.join(); }
};

// normal * 2^(+-spread), from a counter hash (Box-Muller), so n = 16384 fills in milliseconds on the device
__global__ void fill(double *X, size_t n, unsigned long long seed, double spread) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto h = [](unsigned long long x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; };
    const unsigned long long a = h(i * 3 + seed), b = h(i * 3 + 1 + seed), c = h(i * 3 + 2 + seed);
    const double u1 = ((a >> 11) + 0.5) * 0x1p-53, u2 = (b >> 11) * 0x1p-53, u3 = (c >> 11) * 0x1p-53;
    X[i] = sqrt(-2 * log(u1)) * cospi(2 * u2) * exp2(spread * (2 * u3 - 1));
}

// residue planes of moduli L0 .. L0 + G - 1 (compile-time moduli: % is a multiply), row r of X (K contiguous)
// the scale of row r from its 2-norm: |C'_ij| <= ||a'_i|| ||b'_j|| (Cauchy-Schwarz) < P/2 once every row norm, rounding included,
// is below 2^beta, beta = (log2 P - 1) / 2. K leaves the bound, and a row whose mass is spread over K entries keeps
// log2(||a_i|| / max_k |a_ik|) more bits than the max-based scale (Gaussian rows at K = 16384: ~3 bits). Exact for any data.
__global__ void scales_norm(const double *X, int len, int beta, int *s) {
    const double *x = X + (size_t)blockIdx.x * len;
    __shared__ double w[32];
    auto reduce = [&](double v, bool mx) {
        for (int o = 16; o; o >>= 1) { const double u = __shfl_xor_sync(~0u, v, o); v = mx ? fmax(v, u) : v + u; }
        __syncthreads(); if ((threadIdx.x & 31) == 0) w[threadIdx.x >> 5] = v; __syncthreads();
        v = w[0]; for (int i = 1; i < (int)(blockDim.x >> 5); i++) v = mx ? fmax(v, w[i]) : v + w[i];
        return v;
    };
    double mx = 0;
    for (int k = threadIdx.x; k < len; k += blockDim.x) mx = fmax(mx, fabs(x[k]));
    mx = reduce(mx, true);
    const int e = mx == 0 ? 0 : ilogb(mx);
    double ss = 0;   // the squares scaled by 2^-2e: no overflow, no underflow of the entries that matter
    for (int k = threadIdx.x; k < len; k += blockDim.x) { const double v = ldexp(x[k], -e); ss = fma(v, v, ss); }
    ss = reduce(ss, false);
    // 1 + 2^-30 covers the rounding of the sum (K 2^-53) and of sqrt; the rounding of the entries (sqrt(K) / 2) sits in beta
    if (threadIdx.x == 0) s[blockIdx.x] = mx == 0 ? 0 : beta - 1 - ilogb(ldexp(sqrt(ss) * (1 + 0x1p-30), e));
}

template <int L0, int G>
__global__ void residues_g(const double *X, int len, const int *s, int8_t *R, size_t plane) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (k >= len) return;
    const long long v = llrint(ldexp(X[(size_t)r * len + k], s[r]));
    oz::each<L0, L0 + G>([&](auto L) { constexpr int l = decltype(L)::value; R[(l - L0) * plane + (size_t)r * len + k] = (int8_t)oz::balc<oz::modulus(l)>(v); });
}
template <int G, int... I> void residues_at(int l0, const double *X, int n, const int *s, int8_t *R, std::integer_sequence<int, I...>) {
    ((l0 == I ? (void)residues_g<I, G><<<dim3((n + 255) / 256, n), 256>>>(X, n, s, R, (size_t)n * n) : (void)0), ...);
}
static void residues(int l0, int g, const double *X, int n, const int *s, int8_t *R) {
    if (g == 2) residues_at<2>(l0, X, n, s, R, std::make_integer_sequence<int, 15>{});
    else residues_at<1>(l0, X, n, s, R, std::make_integer_sequence<int, 16>{});
}

template <int... I> static void resA_at(int nm, const double *X, int n, const int *s, int8_t *R, std::integer_sequence<int, I...>) {
    ((nm == I ? (void)residues_g<0, I><<<dim3((n + 255) / 256, n), 256>>>(X, n, s, R, (size_t)n * n) : (void)0), ...);
}
static void resA(int nm, const double *X, int n, const int *s, int8_t *R) { resA_at(nm, X, n, s, R, std::integer_sequence<int, 8, 9, 10, 11, 12, 13, 14, 15, 16>{}); }

// Garner as in ozaki2.cuh, then Horner in 128-bit integers: the integer C'_ij exactly, rounded to FP64 once. The FP64 Horner
// rounds nm - 1 times and runs at 1/64 of the FP32 rate (OZ_CRT=fp64: that one)
template <int NM>
__global__ void crt_int(const int8_t *Cl, size_t plane, int n, const int *sa, const int *sb, double *C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (j >= n) return;
    int x[NM];
    oz::each<0, NM>([&](auto L) {
        constexpr int l = decltype(L)::value, m = oz::modulus(l);
        int v = Cl[l * plane + (size_t)i * n + j];   // already balanced
        oz::each<0, l>([&](auto t) { v = oz::balc<m>((v - x[decltype(t)::value]) * oz::c_inv[decltype(t)::value][l]); });
        x[l] = v;
    });
    __int128 y = x[NM - 1];
#pragma unroll
    for (int l = NM - 2; l >= 0; l--) y = y * oz::modulus(l) + x[l];
    const long long hi = (long long)(y >> 64); const unsigned long long lo = (unsigned long long)y;   // |y| < 2^126: hi exact in FP64
    C[(size_t)i * n + j] = ldexp(fma((double)hi, 0x1p64, (double)lo), -(sa[i] + sb[j]));
}

template <int NM> static void crt_run(const int8_t *Cpl, int n, const int *sa, const int *sb, double *C) {
    static const bool fp64 = getenv("OZ_CRT") && !strcmp(getenv("OZ_CRT"), "fp64");
    if (fp64) oz::crt<NM><<<dim3((n + 255) / 256, n), 256>>>(Cpl, (size_t)n * n, n, n, n, sa, sb, C);
    else crt_int<NM><<<dim3((n + 255) / 256, n), 256>>>(Cpl, (size_t)n * n, n, sa, sb, C);
}
static void crt(int nm, const int8_t *Cpl, int n, const int *sa, const int *sb, double *C) {
    switch (nm) {
    case 8: crt_run<8>(Cpl, n, sa, sb, C); break;   case 9: crt_run<9>(Cpl, n, sa, sb, C); break;
    case 10: crt_run<10>(Cpl, n, sa, sb, C); break; case 11: crt_run<11>(Cpl, n, sa, sb, C); break;
    case 12: crt_run<12>(Cpl, n, sa, sb, C); break; case 13: crt_run<13>(Cpl, n, sa, sb, C); break;
    case 14: crt_run<14>(Cpl, n, sa, sb, C); break; case 15: crt_run<15>(Cpl, n, sa, sb, C); break;
    default: crt_run<16>(Cpl, n, sa, sb, C);
    }
}

// the GPU's own FP64, C = A Bt^T: 64 x 64 tiles, 4 x 4 per thread, k in order (the sum of a plain fma loop); compute-bound
// at 1/64 of the FP32 rate, so it runs at the FP64 peak (0.449 TFLOPS) that no FP64 kernel passes
__global__ void dgemm_native(const double *A, const double *Bt, double *C, int n) {
    __shared__ double sA[16][65], sB[16][65];
    const int tx = threadIdx.x, ty = threadIdx.y, i0 = blockIdx.y * 64, j0 = blockIdx.x * 64;
    double c[4][4] = {};
    for (int k0 = 0; k0 < n; k0 += 16) {
        for (int t = ty * 16 + tx; t < 64 * 16; t += 256) {
            const int r = t / 16, k = t % 16;
            sA[k][r] = A[(size_t)(i0 + r) * n + k0 + k]; sB[k][r] = Bt[(size_t)(j0 + r) * n + k0 + k];
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 16; k++) {
            double a[4], b[4];
#pragma unroll
            for (int u = 0; u < 4; u++) { a[u] = sA[k][ty * 4 + u]; b[u] = sB[k][tx * 4 + u]; }
#pragma unroll
            for (int u = 0; u < 4; u++)
#pragma unroll
                for (int v = 0; v < 4; v++) c[u][v] = fma(a[u], b[v], c[u][v]);
        }
        __syncthreads();
    }
    for (int u = 0; u < 4; u++) for (int v = 0; v < 4; v++) C[(size_t)(i0 + ty * 4 + u) * n + j0 + tx * 4 + v] = c[u][v];
}

struct DD { double h, l; };   // double-double: the sampled reference, ~2^-100 relative
static DD dd_add(DD a, double b) { double s = a.h + b, v = s - a.h, e = (a.h - (s - v)) + (b - v); e += a.l; double h = s + e; return {h, e - (h - s)}; }

int main(int argc, char **argv) {
    const int n = argc > 1 ? atoi(argv[1]) : 16384;
    std::vector<int> nms = {12, 13, 14};
    if (argc > 2) { nms.clear(); for (char *p = argv[2]; *p;) { nms.push_back((int)strtol(p, &p, 10)); if (*p == ',') p++; } }
    const double spread = argc > 3 ? atof(argv[3]) : 0;
    const int g = 2;   // residue planes of Bt per pass over Bt; those of A come from one pass, into C's planes
    std::vector<int> Ls = {0, 2};   // OZ_L=2: only the Strassen run (sweeps); OZ_NATIVE=0 skips the 17 s native FP64 run
    if (const char *e = getenv("OZ_L")) Ls = {atoi(e)};
    const bool native = !getenv("OZ_NATIVE") || atoi(getenv("OZ_NATIVE"));
    const bool norm = getenv("OZ_SCALE") && !strcmp(getenv("OZ_SCALE"), "norm");   // else the max-based scale of ozaki2.cuh
    const double window = getenv("OZ_WINDOW") ? atof(getenv("OZ_WINDOW")) : 8;     // seconds of repeated DGEMMs per energy reading
    if (n % 1024) { printf("n must be a multiple of 1024 (two levels of 128 x 256 tiles)\n"); return 1; }
    const size_t nn = (size_t)n * n;
    // the Strassen buffers and the Bt planes live in C's FP64 buffer (8 n^2 bytes) until the CRT overwrites it: operand chunks of
    // cap, the largest that fits (n = 16384: 23 of 49, three chunks)
    int cap = 49;
    while (cap > 1 && g * nn + oz::StrassenMod::bytes(n, 2, cap) > 8 * nn) cap--;
    oz::init_moduli(); oz::StrassenMod::grant();
    oz::StrassenMod sm; sm.tile = 1; sm.G = 8;   // 128 x 256 leaves, 3 stages; block bands of 8
    double *A, *Bt, *C; int8_t *Cpl; int *sa, *sb;
    const int maxnm = *std::max_element(nms.begin(), nms.end());
    CK(cudaMalloc(&A, nn * 8)); CK(cudaMalloc(&Bt, nn * 8)); CK(cudaMalloc(&C, nn * 8)); CK(cudaMalloc(&Cpl, (size_t)maxnm * nn));
    CK(cudaMalloc(&sa, n * 4)); CK(cudaMalloc(&sb, n * 4));
    fill<<<(nn + 255) / 256, 256>>>(A, nn, 1, spread); fill<<<(nn + 255) / 256, 256>>>(Bt, nn, 1ull << 40, spread); CK(cudaDeviceSynchronize());
    int8_t *arena = (int8_t *)C, *Bpl = arena, *tmp = arena + g * nn;   // Bt's planes, then the Strassen buffers (L = 2) or tmp (L = 0)
    sm.plan(n, 2, cap, tmp);
    Power pw;
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    printf("n=%d spread 2^+-%g, NVML %s, %.2f GB free of %.2f, operand chunks of %d\n", n, spread, !pw.ok ? "absent" : pw.counter ? "power samples (energy counter as a cross-check)" : "power samples",
           fr / 1e9, tot / 1e9, cap);

    // the sampled reference: 1024 entries in double-double, and |A||Bt^T| for the componentwise bound
    std::vector<double> hA(nn), hB(nn), hC(nn), hC0(nn);
    CK(cudaMemcpy(hA.data(), A, nn * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hB.data(), Bt, nn * 8, cudaMemcpyDeviceToHost));
    const int S = 1024; std::vector<size_t> si(S), sj(S); std::vector<double> ref(S), mag(S);
    std::mt19937_64 rg(7);
    for (int e = 0; e < S; e++) {
        si[e] = rg() % n; sj[e] = rg() % n; DD s{0, 0}; double m = 0;
        for (int k = 0; k < n; k++) {
            const double a = hA[si[e] * n + k], b = hB[sj[e] * n + k], p = a * b;
            s = dd_add(dd_add(s, p), fma(a, b, -p)); m += fabs(p);
        }
        ref[e] = s.h + s.l; mag[e] = m;
    }
    auto err = [&](const std::vector<double> &X, double &mx, double &mean) {   // in units of 2^-53 of |A||Bt^T|
        mx = 0; mean = 0;
        for (int e = 0; e < S; e++) { const double r = fabs(X[si[e] * n + sj[e]] - ref[e]) / mag[e] / 0x1p-53; mx = fmax(mx, r); mean += r / S; }
    };
    const double flops = 2.0 * n * n * (double)n;
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // the GPU's own FP64 (a plain fma loop: its ceiling is the 0.449 TFLOPS peak, which no FP64 kernel passes)
    if (native) {
        pw.start(); cudaEventRecord(t0); dgemm_native<<<dim3(n / 64, n / 64), dim3(16, 16)>>>(A, Bt, C, n); cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); pw.stop();
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        CK(cudaMemcpy(hC.data(), C, nn * 8, cudaMemcpyDeviceToHost)); double mx, mean; err(hC, mx, mean);
        printf("native FP64   %9.1f ms %7.3f TF  %6.1f W (counter %5.1f) %8.1f J  %6.2f GFLOPS/W  err max %6.2f mean %6.3f\n", ms, flops / ms / 1e9,
               pw.joules / pw.secs, pw.counted / pw.secs, pw.joules, flops / pw.joules / 1e9, mx, mean);
        fflush(stdout);
    }

    for (int nm : nms) {
        int bits = oz::bits_for(nm, n);
        if (norm) { double l2 = 0; for (int l = 0; l < nm; l++) l2 += log2((double)oz::MODULI[l]); bits = (int)floor((l2 - 1.001) / 2); if (bits > 62) bits = 62; }
        std::vector<int8_t> pl0, pl1;
        for (int L : Ls) {
            // one DGEMM; ph (OZ_PROF): ms per phase, every phase synchronised (so their sum is a little above a timed run)
            auto dgemm = [&](float *ph) {
                cudaEvent_t e0 = nullptr, e1 = nullptr; float pr[3] = {};
                if (ph) { cudaEventCreate(&e0); cudaEventCreate(&e1); }
                auto lap = [&](int k, auto f) {
                    if (!ph) { f(); return; }
                    cudaEventRecord(e0); f(); cudaEventRecord(e1); cudaEventSynchronize(e1); float t; cudaEventElapsedTime(&t, e0, e1); ph[k] += t;
                };
                lap(0, [&] { if (norm) { scales_norm<<<n, 256>>>(A, n, bits, sa); scales_norm<<<n, 256>>>(Bt, n, bits, sb); }
                             else { oz::scales<<<n, 256>>>(A, n, n, n, 1, bits, sa); oz::scales<<<n, 256>>>(Bt, n, n, n, 1, bits, sb); } });
                // all nm residue planes of A from one read of A, into C's planes: plane l of A is read by the product of modulus l
                // only, and its result overwrites it (L = 2 writes C after its last operand pass; L = 0 goes through tmp)
                lap(1, [&] { resA(nm, A, n, sa, Cpl); });
                for (int l0 = 0; l0 < nm; l0 += g) {
                    const int gg = nm - l0 < g ? nm - l0 : g;
                    lap(1, [&] { residues(l0, gg, Bt, n, sb, Bpl); });
                    for (int l = l0; l < l0 + gg; l++) lap(3, [&] {
                        int8_t *Al = Cpl + (size_t)l * nn;
                        sm.mul(Al, n, Bpl + (l - l0) * nn, n, L ? Al : tmp, n, n, L, oz::mod_of(oz::MODULI[l]), ph && L ? pr : nullptr);
                        if (!L) CK(cudaMemcpyAsync(Al, tmp, nn, cudaMemcpyDeviceToDevice));
                    });
                }
                lap(5, [&] { crt(nm, Cpl, n, sa, sb, C); });
                if (ph && L) { ph[2] += pr[0]; ph[3] -= pr[0] + pr[2]; ph[4] += pr[2]; }   // the products' phase without their sums
                if (ph) { cudaEventDestroy(e0); cudaEventDestroy(e1); }
            };
            dgemm(nullptr); CK(cudaDeviceSynchronize());   // warm-up
            if (getenv("OZ_PROF")) {   // where the time goes
                float ph[6] = {}; dgemm(ph);
                printf("   phases ms: scales %.1f, residues %.1f, operand sums %.1f, products %.1f, output sums %.1f, CRT %.1f\n", ph[0], ph[1], ph[2], ph[3], ph[4], ph[5]);
            }
            // a window of at least 8 s: the GPU settles at its power cap within ~1 s
            int reps = 0; pw.start(); cudaEventRecord(t0);
            auto w0 = std::chrono::steady_clock::now();
            do { dgemm(nullptr); CK(cudaDeviceSynchronize()); reps++; } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < window);
            cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); pw.stop();
            float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= reps;
            CK(cudaGetLastError());
            // the residue planes of C (before the CRT is checked: identical planes are the claim), then C itself
            std::vector<int8_t> &pl = L ? pl1 : pl0; pl.resize((size_t)nm * nn);
            CK(cudaMemcpy(pl.data(), Cpl, pl.size(), cudaMemcpyDeviceToHost));
            std::vector<double> &hX = L ? hC : hC0;
            CK(cudaMemcpy(hX.data(), C, nn * 8, cudaMemcpyDeviceToHost));
            double mx, mean; err(hX, mx, mean);
            const double J = pw.joules / reps;
            printf("oz%-2d%s L=%d bits %2d %8.1f ms %7.3f TF  %6.1f W (counter %5.1f) %8.2f J  %6.2f GFLOPS/W  err max %6.2f mean %6.3f  (%d runs)\n", nm, norm ? "n" : "", L, bits, ms,
                   flops / ms / 1e9, pw.joules / pw.secs, pw.counted / pw.secs, J, flops / J / 1e9, mx, mean, reps);
            fflush(stdout);
        }
        if (Ls.size() < 2) continue;
        // the claim: every residue plane, and so every FP64 entry, is the same for L = 2 as for the classical product
        size_t dp = 0, dc = 0;
        for (size_t i = 0; i < pl0.size(); i++) dp += pl0[i] != pl1[i];
        for (size_t i = 0; i < nn; i++) dc += memcmp(&hC[i], &hC0[i], 8) != 0;
        pl1[pl1.size() / 3] ^= 1; size_t dm = 0;   // mutant: one flipped bit must be seen, or "0 differing" means nothing
        for (size_t i = 0; i < pl0.size(); i++) dm += pl0[i] != pl1[i];
        printf("oz%-2d L=2 vs L=0: residue bytes differing %zu of %zu, FP64 entries differing %zu, mutant seen %s\n", nm, dp, pl0.size(), dc,
               dm == dp + 1 ? "yes" : "NO");
        if (dp || dc || dm != dp + 1) return 1;
    }
    return 0;
}
