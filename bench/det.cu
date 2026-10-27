// det.cu -- exact determinant of an n x n int8 matrix on INT8 tensor cores (src/ozaki_det.cuh) against FLINT fmpz_mat_det.
// g++ -O3 -c bench/det_flint.cpp -o out_dflint.o && nvcc -O3 -arch=sm_120 -std=c++17 --default-stream per-thread -Xcompiler -fopenmp -o out_det bench/det.cu out_dflint.o -lflint -lgmp -lgomp -lcusolver -lcublas
// ./out_det n [primes per batch=32] [flint: 1 compare, 0 skip] [method: 0 multimodular, 1 divisor by p-adic lifting]
//
// Method 0: det mod every prime up to the Hadamard bound, one CRT.
// Method 1 (the same certificate, far fewer LUs): A^-1 mod one prime, Dixon lifting of x = A^-1 b on the GPU, rational
// reconstruction of a few entries of x gives d = their common denominator, which divides det; then det / d is bounded by
// Hadamard / d (a few thousand bits instead of ~50 000) and needs only that many primes. Deterministic and proved: the
// randomness of b only decides how large d comes out.
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <vector>
#include <gmp.h>
#include "../src/ozaki_det.cuh"
#include "../src/ozaki_detbound.cuh"
#include "../src/ozaki_hol.cuh"
double flint_det(int n, const int8_t *A, int threads, long *bits, std::string *hex, int L = 1);
double flint_solve(int n, const int8_t *A, const int32_t *b, int threads, const __mpz_struct *num, const mpz_t D, int *ok);
bool flint_ratrec(mpz_t num, mpz_t den, const mpz_t X, const mpz_t Q, long nb, long db);   // half-gcd (det_flint.cpp)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// det mod the primes pr (a multiple of Pb of them), bad ones flagged
static const int32_t *gU = nullptr, *gL = nullptr;   // the preconditioner, once it is needed
static int gPlanes = 1;   // OZ_PLANES=L: entries of L bytes, sum_j A[j] 256^j

static void det_mod_primes(int n, int Pb, const int8_t *dA, const std::vector<uint32_t> &pr, std::vector<uint32_t> &det,
                           std::vector<int> &bad) {
    oz::DetWork w; w.alloc(n, Pb);
    w.preU = gU; w.preL = gL; w.L = gPlanes;
    det.resize(pr.size()); bad.resize(pr.size());
    std::vector<uint32_t> one(Pb, 1); std::vector<int> zero(Pb, 0);
    for (size_t b = 0; b < pr.size(); b += Pb) {
        cudaMemcpy(w.P, &pr[b], Pb * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(w.det, one.data(), Pb * 4, cudaMemcpyHostToDevice); cudaMemcpy(w.bad, zero.data(), Pb * 4, cudaMemcpyHostToDevice);
        oz::det_mod_batch(w, dA);
        cudaMemcpy(&det[b], w.det, Pb * 4, cudaMemcpyDeviceToHost); cudaMemcpy(&bad[b], w.bad, Pb * 4, cudaMemcpyDeviceToHost);
    }
    w.free_();
}

static uint64_t invm(uint64_t a, uint64_t p) { uint64_t r = 1, e = p - 2; for (a %= p; e; e >>= 1, a = a * a % p) if (e & 1) r = r * a % p; return r; }

// x = the symmetric CRT of residues v[i] mod pr[i] over the i with use[i]; returns the bits of the modulus
static double crt(mpz_t x, const std::vector<uint32_t> &pr, const std::vector<uint32_t> &v, const std::vector<int> &use) {
    mpz_t M, t; mpz_init_set_ui(M, 1); mpz_init(t); mpz_set_ui(x, 0); double gb = 0;
    for (size_t i = 0; i < pr.size(); i++) {
        if (!use[i]) continue;
        uint32_t p = pr[i], xm = mpz_fdiv_ui(x, p);
        uint64_t k = (uint64_t)(v[i] + p - xm) % p * invm(mpz_fdiv_ui(M, p), p) % p;
        mpz_addmul_ui(x, M, k); mpz_mul_ui(M, M, p); gb += log2((double)p);
    }
    mpz_fdiv_q_2exp(t, M, 1); if (mpz_cmp(x, t) > 0) mpz_sub(x, x, M);
    mpz_clear(M); mpz_clear(t); return gb;
}

// rational reconstruction: num/den = X mod Q with |num| < 2^nb, 0 < den < 2^db (requires 2^(nb+db+1) <= Q); false if none
static bool ratrec(mpz_t num, mpz_t den, const mpz_t X, const mpz_t Q, long nb, long db) {
    mpz_t r0, r1, t0, t1, q, tmp; mpz_inits(r0, r1, t0, t1, q, tmp, nullptr);
    mpz_set(r0, Q); mpz_mod(r1, X, Q); mpz_set_ui(t0, 0); mpz_set_ui(t1, 1);
    while ((long)mpz_sizeinbase(r1, 2) > nb - 1 && mpz_sgn(r1)) {
        mpz_fdiv_qr(q, tmp, r0, r1); mpz_swap(r0, r1); mpz_swap(r1, tmp);   // (r0, r1) = (r1, r0 - q r1)
        mpz_submul(t0, q, t1); mpz_swap(t0, t1);                           // (t0, t1) = (t1, t0 - q t1)
    }
    bool ok = mpz_sgn(t1) && (long)mpz_sizeinbase(t1, 2) <= db - 1;
    if (ok) { mpz_set(num, r1); mpz_set(den, t1); if (mpz_sgn(den) < 0) { mpz_neg(den, den); mpz_neg(num, num); } }
    mpz_clears(r0, r1, t0, t1, q, tmp, nullptr);
    return ok;
}

int main(int argc, char **argv) {
    const int n = atoi(argv[1]), Pb = argc > 2 ? atoi(argv[2]) : 32, cmp = argc > 3 ? atoi(argv[3]) : 1,
              method = argc > 4 ? atoi(argv[4]) : 1, kind = argc > 5 ? atoi(argv[5]) : 0;
    const int L = getenv("OZ_PLANES") ? atoi(getenv("OZ_PLANES")) : 1; gPlanes = L;
    if (L > 1 && method == 2) { printf("OZ_PLANES: methods 0 and 1 (FLINT's solve check takes int8)\n"); return 1; }
    std::vector<int8_t> A((size_t)L * n * n); uint64_t s = 12345;   // the same LCG as det_flint.cpp's main; plane 0 = the int8 matrix
    for (auto &a : A) { s = s * 6364136223846793005ull + 1442695040888963407ull; a = (int8_t)((s >> 33) % 255 - 127); }
    if (kind == 1) for (int j = 0; j < L; j++) A[(size_t)j * n * n] = 0;   // a zero 1 x 1 leading minor (every plane)
    if (kind == 2) for (int j = 0; j < L; j++) for (int k = 0; k < n; k++) A[(size_t)j * n * n + (size_t)(n - 1) * n + k] = A[(size_t)j * n * n + k];   // singular: last row = first row
    // U unit upper, L unit lower, entries in [0, 127] off the diagonal: the same residues for every prime
    auto make_pre = [&] {
        std::vector<int32_t> U((size_t)n * n, 0), L((size_t)n * n, 0); uint64_t q = 999;
        for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) {
            q = q * 6364136223846793005ull + 1442695040888963407ull; int v = (int)((q >> 33) % 128);
            if (i < k) U[(size_t)i * n + k] = v; else if (i > k) L[(size_t)i * n + k] = v; else U[(size_t)i * n + k] = L[(size_t)i * n + k] = 1;
        }
        int32_t *dU, *dL; cudaMalloc(&dU, U.size() * 4); cudaMalloc(&dL, L.size() * 4);
        cudaMemcpy(dU, U.data(), U.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dL, L.data(), L.size() * 4, cudaMemcpyHostToDevice);
        gU = dU; gL = dL;
    };

    double hr = 0, hc = 0, cmin = 1e300, rmin = 1e300;   // Hadamard over rows and over columns: log2 |det| <= min of the two
    std::vector<double> cn(n, 0);
    for (int i = 0; i < n; i++) {
        double r = 0;
        for (int k = 0; k < n; k++) { double a = 0; for (int j = 0; j < L; j++) a += A[(size_t)j * n * n + (size_t)i * n + k] * ldexp(1.0, 8 * (j - L + 1)); r += a * a; cn[k] += a * a; }
        hr += 0.5 * log2(r) + 8.0 * (L - 1);   // entries scaled by 2^-8(L-1) above
        rmin = std::min(rmin, 0.5 * log2(r) + 8.0 * (L - 1));
    }
    for (int k = 0; k < n; k++) { hc += 0.5 * log2(cn[k]) + 8.0 * (L - 1); cmin = std::min(cmin, 0.5 * log2(cn[k]) + 8.0 * (L - 1)); }
    const double hb = std::min(hr, hc);
    std::vector<char> comp(65280, 0); std::vector<uint32_t> all;   // primes downwards from 65279
    for (int i = 2; i < 65280; i++) if (!comp[i]) for (int k = 2 * i; k < 65280; k += i) comp[k] = 1;
    for (int i = 65279; i > 2; i--) if (!comp[i]) all.push_back(i);
    // the first primes from `from` whose log2 sum reaches `bits` with spares for bad ones, rounded up to a batch
    auto take = [&](size_t from, double bits, int batch) {
        std::vector<uint32_t> pr; double lb = 0;
        for (size_t i = from; i < all.size() && (lb < bits * (1.0 + 2.5 * n / 65000.0) + 64 || pr.size() % batch); i++) {
            pr.push_back(all[i]); lb += log2((double)all[i]);
        }
        return pr;
    };

    int8_t *dA; cudaMalloc(&dA, A.size()); cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    mpz_t D; mpz_init(D);
    const double t0 = now();
    double t_inv = 0, t_lift = 0, t_rr = 0, t_mod = 0, t_bound = 0, gb = 0, ub = hb, t_bth = 0; int bprimes = 0;
    std::thread bth;
    long dbits = 0; int good = 0; size_t nprimes = 0, steps = 0;

    if (method == 0) {
        std::vector<uint32_t> pr = take(0, hb + 2, Pb), det; std::vector<int> bad, use;
        det_mod_primes(n, Pb, dA, pr, det, bad);
        int nbad = 0; for (int b : bad) nbad += b;
        if (nbad > (int)pr.size() / 2) {
            make_pre(); det_mod_primes(n, Pb, dA, pr, det, bad);
            nbad = 0; for (int b : bad) nbad += b;
            if (nbad > (int)pr.size() / 2) { printf("n=%d: det = 0 (most preconditioned primes fail; probabilistic)\n", n); return 0; }
        }
        cudaDeviceSynchronize(); t_mod = now() - t0;
        for (int b : bad) { use.push_back(!b); good += !b; }
        gb = crt(D, pr, det, use); nprimes = pr.size();
        if (gb < hb + 2) { printf("NOT ENOUGH GOOD PRIMES\n"); return 1; }
    } else {
        // 0. the proved bound on log2 |det| (step 4 needs it) on its own thread and stream, overlapping steps 1-3
        // (above n = 8192 it waits until the [A I] workspace is freed: both do not fit in 12 GB together)
        auto start_bound = [&] {
            if (method == 1 && !getenv("OZ_HADAMARD") && (n >= 4096 || (gPlanes > 1 && n >= 2048 && n <= 16384) || getenv("OZ_BOUND")))   // large entries: Hadamard leaves ~0.72 n bits, the primes for them cost more
                bth = std::thread([&] { const double tb = now(); ub = std::min(hb, oz::det_log2_bound(n, dA, getenv("OZ_S") ? atoi(getenv("OZ_S")) : 20, &bprimes, gPlanes)); t_bth = now() - tb; });
        };
        if (n <= 8192) start_bound();
        // 1. A^-1 mod p from the LU of [A I]
        // S primes at once (src/ozaki_det.cuh liftm_*) for large entries: S = L / 2 by default (OZ_LIFTS), 1 = one prime
        int S = 1;
        if (L > 1) {   // the largest of 1 2 3 4 6 8 12 16 not above L / 2, or OZ_LIFTS
            for (int c : {2, 3, 4, 6, 8, 12, 16}) if (c <= L / 2) S = c;
            if (getenv("OZ_LIFTS")) S = atoi(getenv("OZ_LIFTS"));
        }
        // OZ_HOL=1: high-order lifting (src/ozaki_hol.cuh) instead of Dixon. X = S primes with X >= 6.25 n^2 |A|
        // (|A| < 2^(8L-1); 1 bit spare for the retries' slightly smaller primes); prod q > 2^(zbits+1), |R| < 2^zbits
        const bool hol = getenv("OZ_HOL") != nullptr;
        int zbits = 0; std::vector<uint32_t> qhol;
        if (hol) {
            const double need = log2(6.25) + 2 * log2((double)n) + 8.0 * L - 1 + 1;
            double lg = 0; S = 0;
            while (lg < need) lg += log2((double)all[S++]);
            zbits = (int)ceil(log2(1.25 * n) + 8.0 * L - 1);
            for (lg = 0; lg < zbits + 2;) { qhol.push_back(all[400 + qhol.size()]); lg += log2((double)qhol.back()); }   // far from the retries' primes
        }
        if (S > 1 && n > 16384) { printf("OZ_LIFTS > 1 above n = 16384: not wired\n"); return 1; }
        oz::DetWork w; w.alloc(n, S, 2 * n); w.L = L;
        uint32_t p = 0; size_t pi = 0; oz::LiftPrimes lp{}; lp.s = S;
        bool singular = false;
        for (;; pi++) {   // 4 plain primes failing = a zero leading minor over Z, or det 0: precondition and try 8 more
            if (pi == 4 && !gU) { make_pre(); w.preU = gU; w.preL = gL; }
            if (pi == 12) { singular = true; break; }
            std::vector<uint32_t> one(S, 1); std::vector<int> zero(S, 0), bads(S);
            for (int t = 0; t < S; t++) lp.p[t] = all[pi * S + t];
            p = lp.p[0];
            cudaMemcpy(w.P, lp.p, S * 4, cudaMemcpyHostToDevice); cudaMemcpy(w.det, one.data(), S * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(w.bad, zero.data(), S * 4, cudaMemcpyHostToDevice);
            oz::det_mod_batch(w, dA);
            cudaMemcpy(bads.data(), w.bad, S * 4, cudaMemcpyDeviceToHost);
            int bad = 0; for (int v : bads) bad |= v;
            if (!bad) break;
        }
        if (singular) {   // Monte Carlo, not proved: U A L had a zero pivot modulo 8 primes
            if (bth.joinable()) bth.join();
            printf("n=%d: det = 0 (8 preconditioned primes all fail; probabilistic)\n", n);
            if (cmp) { long fb; std::string fh; flint_det(n, A.data(), 24, &fb, &fh, L); printf("  FLINT: %s\n", fh.c_str()); }
            return 0;
        }
        std::vector<__mpz_struct> x4(4); for (auto &v : x4) mpz_init(&v);   // 4 entries of the solution mod Q
        mpz_t Q; mpz_init(Q); long nb = 0, db = 0; double t2 = 0;
        if (hol) {   // row form y A = b: the same denominator (s_n of A and of A^T), numerators by Cramer on A^T (rows)
            t_inv = now() - t0;
            std::vector<int32_t> b(n); double bn = 0; uint64_t sb = 777;
            for (int i = 0; i < n; i++) { sb = sb * 6364136223846793005ull + 1442695040888963407ull; b[i] = (int)((sb >> 33) % 255) - 127; bn += (double)b[i] * b[i]; }
            nb = (long)ceil(hr - rmin + 0.5 * log2(bn)) + 2; db = (long)ceil(hb) + 2;
            double lgX = 0; for (int t = 0; t < S; t++) lgX += log2((double)lp.p[t]);
            const int N = (int)ceil((nb + db + 2) / lgX); steps = N;
            const double t1 = now(); int herr = 0; oz::HolStats hs;
            std::vector<int16_t> dig = oz::hol_solve(w, dA, L, lp.p, S, qhol.data(), (int)qhol.size(), b.data(), N, zbits, hs, &herr);
            t_lift = now() - t1;
            if (oz::det_prof.on) printf("  HOL8 phases: Z^2 %.3f  bext %.3f  M %.3f  garner %.3f  AM %.3f  next %.3f  transpose %.3f s\n", hs.ph[0], hs.ph[1], hs.ph[2], hs.ph[3], hs.ph[4], hs.ph[5], hs.ph[6]);
            if (herr) { printf("HOL: a residue outside its bound\n"); return 1; }
            printf("  HOL: X = %d primes, %zu primes q, %d X-digits, %d levels, %d modular GEMMs 16-bit + %d 8-bit (W %d, WQ %d); setup %.3f  residue chain %.3f  vector chain %.3f s\n",
                   S, qhol.size(), N, hs.levels, hs.gemms, hs.gemms8, hs.W, hs.WQ, hs.init, hs.rchain, hs.cchain);
            t2 = now();
            const int sN = S * N;
#pragma omp parallel for
            for (int j = 0; j < 4; j++)   // Horner over the positions from the top: x = x p_(e mod S) + d_e
                for (int e = sN - 1; e >= 0; e--) {
                    mpz_mul_ui(&x4[j], &x4[j], lp.p[e % S]);
                    const int d = dig[(size_t)e * 4 + j];
                    if (d >= 0) mpz_add_ui(&x4[j], &x4[j], d); else mpz_sub_ui(&x4[j], &x4[j], -d);
                }
            mpz_set_ui(Q, 1); for (int t = 0; t < S; t++) mpz_mul_ui(Q, Q, lp.p[t]);
            mpz_pow_ui(Q, Q, N);
            for (auto &v : x4) mpz_mod(&v, &v, Q);
        } else {
        uint16_t *Ai;
        if (n <= 16384) {
            cudaMalloc(&Ai, (size_t)S * n * n * 2);
            for (int t = 0; t < S; t++) oz::to_u16<<<1024, 256>>>(w.R + t * w.zr(), w.ld, n, n, Ai + (size_t)t * n * n);
            cudaDeviceSynchronize(); w.free_();
        } else {   // [A I] (8.6 GB at 32768) and a 16-bit A^-1 beside it do not fit: through the host, 4096 rows at a time
            std::vector<uint16_t> h((size_t)n * n); uint16_t *tmp; cudaMalloc(&tmp, (size_t)4096 * n * 2);
            for (int i0 = 0; i0 < n; i0 += 4096) {
                const int r = std::min(4096, n - i0);
                oz::to_u16_rc<<<1024, 256>>>(w.at(i0, 0), w.ld, n, r, n, tmp);
                cudaMemcpy(h.data() + (size_t)i0 * n, tmp, (size_t)r * n * 2, cudaMemcpyDeviceToHost);
            }
            cudaFree(tmp); w.free_();
            cudaMalloc(&Ai, (size_t)n * n * 2); cudaMemcpy(Ai, h.data(), h.size() * 2, cudaMemcpyHostToDevice);
        }
        t_inv = now() - t0;
        if (n > 8192) start_bound();   // above 16384 the panelled bound (~8 GB) fits beside the lifting (3 GB)

        // 2. lifting: enough digits to reconstruct num / den with |num| <= Hc |b| / min col, den <= H
        std::vector<int32_t> b(n); std::vector<uint32_t> bm(n); double bn = 0; uint64_t sb = 777;
        for (int i = 0; i < n; i++) {
            sb = sb * 6364136223846793005ull + 1442695040888963407ull; b[i] = (int)((sb >> 33) % 255) - 127;
            bm[i] = ((b[i] % (int)p) + p) % p; bn += (double)b[i] * b[i];
        }
        nb = (long)ceil(hc - cmin + 0.5 * log2(bn)) + 2; db = (long)ceil(hb) + 2;
        double lgM = 0;   // log2 of the modulus gained per step
        for (int t = 0; t < S; t++) {
            lgM += log2((double)lp.p[t]);
            for (int u = 0; u < t; u++) lp.inv[u][t] = (uint32_t)invm(lp.p[u], lp.p[t]);
        }
        steps = (size_t)ceil((nb + db + 2) / lgM);
        int32_t *dr; uint32_t *drm; uint16_t *dx; int *derr;
        cudaMalloc(&dr, n * 4); cudaMalloc(&drm, n * 4); cudaMalloc(&dx, (method == 2 ? steps : 1) * n * 2);   // method 1 keeps 4 entries per digit, in dh
        uint16_t *dh; cudaMalloc(&dh, steps * 4 * S * 2); cudaMalloc(&derr, 4);
        cudaMemcpy(dr, b.data(), n * 4, cudaMemcpyHostToDevice); cudaMemcpy(drm, bm.data(), n * 4, cudaMemcpyHostToDevice);
        const int J = L + 2 * S + 2; long long *dr64 = nullptr;   // L > 1: the residual as J base-256 digits per row
        uint32_t *drmS = nullptr; uint16_t *dxS = nullptr;       // S > 1: r mod p_t (S x n), x_i as S base-2^16 digits (n x S)
        if (S > 1) {
            std::vector<uint32_t> rmS((size_t)S * n);
            for (int t = 0; t < S; t++) for (int i = 0; i < n; i++) rmS[(size_t)t * n + i] = ((b[i] % (int)lp.p[t]) + lp.p[t]) % lp.p[t];
            cudaMalloc(&drmS, rmS.size() * 4); cudaMemcpy(drmS, rmS.data(), rmS.size() * 4, cudaMemcpyHostToDevice);
            cudaMalloc(&dxS, (size_t)n * S * 2);
        }
        if (L > 1) {
            std::vector<long long> r64((size_t)n * J, 0); for (int i = 0; i < n; i++) r64[(size_t)i * J] = b[i];
            cudaMalloc(&dr64, r64.size() * 8); cudaMemcpy(dr64, r64.data(), r64.size() * 8, cudaMemcpyHostToDevice);
        }
        cudaMemset(derr, 0, 4);
        const double t1 = now();
        for (size_t i = 0; i < steps; i++) {
            if (S > 1) {
                oz::liftm_step(S, Ai, drmS, lp, dxS, n, dh + i * 4 * S, dA, L, dr64, J, derr);
                continue;
            }
            uint16_t *xi = method == 2 ? dx + i * n : dx;
            oz::lift_x<<<n / 8, 256>>>(Ai, drm, xi, n, p, dh + i * 4);
            if (L > 1) oz::lift_r_planes<<<n / 8, 256>>>(dA, L, xi, dr64, J, drm, n, p, derr);
            else oz::lift_r<<<n / 8, 256>>>(dA, xi, dr, drm, n, p, derr);
        }
        std::vector<uint16_t> X(method == 2 ? steps * n : steps * 4 * S); int err;
        cudaMemcpy(X.data(), method == 2 ? dx : dh, X.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost);
        cudaFree(dh); cudaFree(Ai);
        if (method != 2) cudaFree(dx);   // method 2 converts dx on the GPU (OZ_GPUCONV)
        t_lift = now() - t1;
        if (err) { printf("LIFTING: inexact division\n"); return 1; }

        if (method == 2) {   // the exact rational solution x = num / D of A x = b, every entry
            const double t2 = now();
            int lg = 0; while ((1ull << lg) < steps) lg++;
            std::vector<__mpz_struct> pw(lg + 1);   // p^(2^k)
            for (int k = 0; k <= lg; k++) { mpz_init(&pw[k]); if (k) mpz_mul(&pw[k], &pw[k - 1], &pw[k - 1]); else mpz_set_ui(&pw[k], p); }
            std::vector<__mpz_struct> num(n); for (auto &v : num) mpz_init(&v);
            mpz_t Q; mpz_init(Q); mpz_ui_pow_ui(Q, p, steps);
            // x_j = sum_i X[i][j] p^i by halves (subquadratic), then num_j = D x_j mod Q, symmetric, |num_j| < 2^nb
            std::function<void(mpz_t, int, size_t, int)> conv = [&](mpz_t out, int j, size_t i0, int k) {
                if (i0 >= steps) { mpz_set_ui(out, 0); return; }
                if (k == 0) { mpz_set_ui(out, X[i0 * n + j]); return; }
                mpz_t hi; mpz_init(hi);
                conv(out, j, i0, k - 1); conv(hi, j, i0 + ((size_t)1 << (k - 1)), k - 1);
                mpz_addmul(out, hi, &pw[k - 1]); mpz_clear(hi);
            };
            std::vector<int> fail(n, 0);
            double t_conv = now();
            if (!getenv("OZ_GPUCONV")) {   // Horner by halves on 24 threads: faster than the GEMM below while the powers are built on the host
#pragma omp parallel for schedule(dynamic, 16)
                for (int j = 0; j < n; j++) conv(&num[j], j, 0, lg);
            } else {   // OZ_GPUCONV=1: one INT8 GEMM (src/ozaki_det.cuh digits_to_a): [lo | h] x [P; 256 P], then carries
                const int mpad = oz::rup((int)steps, 64), K2 = 2 * mpad, Mpad = oz::rup(n, oz::BM);
                const int L = (int)(steps * log2((double)p) / 8) + 3, Npad = oz::rup(L + 1, oz::BN);
                std::vector<int8_t> Bt((size_t)Npad * K2, 0);   // Bt[l][i] = byte l of p^i, Bt[l][mpad + i] = byte l - 1
                mpz_t S; mpz_init(S);                           // 32768 sum_i p^i
#pragma omp parallel
                {
                    mpz_t q; mpz_init(q); std::vector<uint8_t> by(L + 8);
#pragma omp for schedule(static)
                    for (size_t i = 0; i < steps; i++) {
                        mpz_ui_pow_ui(q, p, i); size_t cnt = 0;
                        std::fill(by.begin(), by.end(), 0);
                        mpz_export(by.data(), &cnt, -1, 1, 0, 0, q);
                        int c = 0;
                        for (int l = 0; l <= L; l++) {   // balanced bytes: v in [-128, 127], carry up
                            int v = by[l] + c; c = v > 127; if (c) v -= 256;
                            Bt[(size_t)l * K2 + i] = (int8_t)v;
                            if (l + 1 < Npad) Bt[(size_t)(l + 1) * K2 + mpad + i] = (int8_t)v;
                        }
                    }
                    mpz_clear(q);
                }
                for (size_t i = steps; i-- > 0;) { mpz_mul_ui(S, S, p); mpz_add_ui(S, S, 1); }
                mpz_mul_2exp(S, S, 15);
                int8_t *dAp, *dBt; int32_t *dC; uint8_t *dby; long long *dtop;
                cudaMalloc(&dAp, (size_t)Mpad * K2); cudaMalloc(&dBt, Bt.size()); cudaMalloc(&dC, (size_t)Mpad * Npad * 4);
                cudaMalloc(&dby, (size_t)n * Npad); cudaMalloc(&dtop, n * 8);
                cudaMemset(dAp, 0, (size_t)Mpad * K2);
                cudaMemcpy(dBt, Bt.data(), Bt.size(), cudaMemcpyHostToDevice);
                oz::digits_to_a<<<2048, 256>>>(dx, (int)steps, mpad, n, dAp);
                oz::gemm_i8<<<dim3(Npad / oz::BN, Mpad / oz::BM, 1), oz::THREADS>>>(dAp, dBt, dC, Mpad, Npad, K2);
                oz::carry_rows<<<(n + 127) / 128, 128>>>(dC, Npad, n, dby, dtop);
                std::vector<uint8_t> by((size_t)n * Npad); std::vector<long long> top(n);
                cudaMemcpy(by.data(), dby, by.size(), cudaMemcpyDeviceToHost); cudaMemcpy(top.data(), dtop, n * 8, cudaMemcpyDeviceToHost);
                cudaFree(dAp); cudaFree(dBt); cudaFree(dC); cudaFree(dby); cudaFree(dtop);
#pragma omp parallel for
                for (int j = 0; j < n; j++) {
                    mpz_t t; mpz_init(t);
                    mpz_import(&num[j], Npad, -1, 1, 0, 0, &by[(size_t)j * Npad]);
                    mpz_set_si(t, top[j]); mpz_mul_2exp(t, t, 8 * Npad); mpz_add(&num[j], &num[j], t);
                    mpz_add(&num[j], &num[j], S); mpz_clear(t);
                }
                mpz_clear(S);
            }
            t_conv = now() - t_conv;
            if (getenv("OZ_CONVCHECK")) {   // the GEMM conversion against Horner for a few entries
                mpz_t y; mpz_init(y);
                for (int j = 0; j < n; j += n / 4) {
                    mpz_set_ui(y, 0);
                    for (size_t i = steps; i-- > 0;) { mpz_mul_ui(y, y, p); mpz_add_ui(y, y, X[i * n + j]); }
                    printf("  conv check entry %d: %s\n", j, mpz_cmp(y, &num[j]) ? "MISMATCH" : "ok");
                }
                mpz_clear(y);
            }
            mpz_t nm, dn, h; mpz_inits(nm, dn, h, nullptr); mpz_fdiv_q_2exp(h, Q, 1);
            // D from entry 0, then every num_j = D x_j mod Q (symmetric) in parallel; an entry whose result is not below
            // 2^nb needs a larger denominator: reconstruct it, D = lcm, and redo (rare: random b gives D = s_n at once)
            std::vector<__mpz_struct> y(n); for (auto &v : y) mpz_init(&v);
            if (!ratrec(nm, dn, &num[0], Q, nb, db)) { printf("RATREC failed at entry 0\n"); return 1; }
            mpz_set(D, dn);
            for (int pass = 0; pass < 8; pass++) {
                std::vector<int> big(n, 0);
#pragma omp parallel for
                for (int j = 0; j < n; j++) {
                    mpz_mul(&y[j], &num[j], D); mpz_mod(&y[j], &y[j], Q); if (mpz_cmp(&y[j], h) > 0) mpz_sub(&y[j], &y[j], Q);
                    big[j] = (long)mpz_sizeinbase(&y[j], 2) >= nb;
                }
                int again = 0;
                for (int j = 0; j < n; j++) if (big[j]) {   // with the D grown so far it may already fit
                    mpz_mul(nm, &num[j], D); mpz_mod(nm, nm, Q); if (mpz_cmp(nm, h) > 0) mpz_sub(nm, nm, Q);
                    if ((long)mpz_sizeinbase(nm, 2) < nb) continue;
                    if (!ratrec(nm, dn, &num[j], Q, nb, db)) { printf("RATREC failed at entry %d\n", j); return 1; }
                    mpz_lcm(D, D, dn); again = 1;
                }
                if (!again) break;
            }
            for (int j = 0; j < n; j++) mpz_swap(&num[j], &y[j]);
            t_rr = now() - t2;
            size_t tb = 0; for (auto &v : num) tb += mpz_sizeinbase(&v, 2);
            printf("n=%d solve  x = num / D: D %zu bits, numerators %.1f Mbit in all  total %.3f s  [A^-1 %.3f  lift %zu steps %.3f  "
                   "to integers %.3f + D %.3f]\n", n, mpz_sizeinbase(D, 2), tb / 1e6, now() - t0, t_inv, steps, t_lift, t_conv, t_rr - t_conv);
            if (cmp) {
                int ok; double ft = flint_solve(n, A.data(), b.data(), 24, num.data(), D, &ok);
                printf("  FLINT fmpz_mat_solve_dixon_den (24 thr) %.2f s  -> %s  (%.1fx)\n", ft, ok ? "BIT-EXACT" : "MISMATCH", ft / (now() - t0 - ft));
            }
            return 0;
        }

        // 3. a few entries of x back to integers (Horner in p), rational reconstruction, d = lcm of their denominators
        t2 = now();
        mpz_t Mz; mpz_init_set_ui(Mz, 1);   // the modulus of one step: p, or p_0 ... p_{S-1}
        for (int t = 0; t < S; t++) mpz_mul_ui(Mz, Mz, lp.p[t]);
        mpz_pow_ui(Q, Mz, steps);
#pragma omp parallel for
        for (int j = 0; j < 4; j++) {
            mpz_t dg; mpz_init(dg);
            for (size_t i = steps; i-- > 0;) {   // digit i of entry j: S base-2^16 words, little-endian
                mpz_import(dg, S, -1, 2, 0, 0, &X[(i * 4 + j) * S]);
                mpz_mul(&x4[j], &x4[j], Mz); mpz_add(&x4[j], &x4[j], dg);
            }
            mpz_clear(dg);
        }
        mpz_clear(Mz);
        }   // Dixon
        // only the entries whose D x_j mod Q (symmetric) is not below 2^nb need a reconstruction (Euclid on
        // ~2(nb + db)-bit numbers, the costly part): when it is, y / D is THE reconstruction, so den | D already
        mpz_t xj, num, den; mpz_inits(xj, num, den, nullptr); mpz_set_ui(D, 1);
        mpz_t hq; mpz_init(hq); mpz_fdiv_q_2exp(hq, Q, 1);
        for (int j = 0; j < 4; j++) {
            mpz_mul(xj, &x4[j], D); mpz_mod(xj, xj, Q); if (mpz_cmp(xj, hq) > 0) mpz_sub(xj, xj, Q);
            if ((long)mpz_sizeinbase(xj, 2) < nb) continue;
            if (!(getenv("OZ_GMPRR") ? ratrec : flint_ratrec)(num, den, &x4[j], Q, nb, db)) { printf("RATREC failed at entry %d\n", j); return 1; }
            mpz_lcm(D, D, den);
        }
        mpz_clear(hq); for (auto &v : x4) mpz_clear(&v);
        dbits = mpz_sizeinbase(D, 2); t_rr = now() - t2;

        // 4. c = det / d from as many primes as the bound / d needs: |c| < 2^(ub - dbits + 1). ub: the proved bound of
        // src/ozaki_detbound.cuh (a few bits above log2 |det|), or Hadamard (below n = 4096, where cuSOLVER's ~0.3 s setup costs more than the primes it saves; OZ_HADAMARD=1; or FP32 LU finds A singular)
        const double t3 = now();
        if (bth.joinable()) bth.join();
        t_bound = now() - t3;   // the wait for the bound thread; t_bth is its own time
        const double need_c = ub - dbits + 3;
        const int bc = n > 16384 ? 1 : std::min(Pb,(int)ceil(need_c / 15.9) + 2);
        std::vector<uint32_t> pr = take(0, need_c, bc), det; std::vector<int> bad, use;
        det_mod_primes(n, bc, dA, pr, det, bad); cudaDeviceSynchronize();
        for (size_t i = 0; i < pr.size(); i++) {
            uint32_t dm = mpz_fdiv_ui(D, pr[i]);
            use.push_back(!bad[i] && dm); good += use.back();
            if (use.back()) det[i] = (uint64_t)det[i] * invm(dm, pr[i]) % pr[i];
        }
        mpz_t c; mpz_init(c); gb = crt(c, pr, det, use); nprimes = pr.size();
        if (gb < need_c) { printf("NOT ENOUGH GOOD PRIMES\n"); return 1; }
        mpz_mul(D, D, c); t_mod = now() - t3;
        mpz_clears(Q, xj, num, den, c, nullptr);
    }
    const double total = now() - t0;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA: %s\n", cudaGetErrorString(err)); return 1; }
    if (oz::det_prof.on) printf("prof: diag %.3f s  pack %.3f s  gemm %.3f s\n", oz::det_prof.t[0], oz::det_prof.t[1], oz::det_prof.t[2]);
    { char *hx = mpz_get_str(nullptr, 16, D); printf("n=%d method %d  head %.16s", n, method, hx); free(hx); }
    printf("  Hadamard %.0f bits  |det| %zu bits  total %.3f s", hb, mpz_sizeinbase(D, 2), total);
    if (method) printf("  [A^-1 %.3f  lift %zu steps %.3f  ratrec %.3f (d %ld bits)  bound %.0f bits (%d primes) %.3f, waited %.3f  det/d: %zu primes %.3f]",
                       t_inv, steps, t_lift, t_rr, dbits, ub, bprimes, t_bth, t_bound, nprimes, t_mod - t_bound);
    else printf("  [%zu primes, %d good]", nprimes, good);
    printf("\n");
    if (getenv("OZ_CHECK")) {   // where FLINT would take hours: det mod 4 primes used nowhere above, against D mod q
        // ~32 000, far below the ones used; a prime with a zero pivot (probability ~n/p, 0.4 at n = 32768) is replaced by the next
        const int cb_ = n > 16384 ? 1 : 4;   // n x n int32 per prime: 4.3 GB at 32768
        for (size_t at = 3000, ok = 0; ok < 4 && at < 3040; at += cb_) {
            std::vector<uint32_t> cq(all.begin() + at, all.begin() + at + cb_), cd; std::vector<int> cb;
            det_mod_primes(n, cb_, dA, cq, cd, cb);
            for (int k = 0; k < cb_; k++) {
                if (cb[k] || ok >= 4) continue;
                ok++; printf("  check mod %u: %s\n", cq[k], mpz_fdiv_ui(D, cq[k]) == cd[k] ? "ok" : "MISMATCH");
            }
        }
    }
    if (cmp) {
        long fb; std::string fh; double ft = flint_det(n, A.data(), 24, &fb, &fh, L);
        char *gh = mpz_get_str(nullptr, 16, D);
        printf("  FLINT fmpz_mat_det (24 thr) %.2f s  -> %s  (%.1fx)\n", ft, fh == gh ? "BIT-EXACT" : "MISMATCH", ft / total);
        free(gh);
    }
    return 0;
}
