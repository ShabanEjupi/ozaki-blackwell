// det.cu -- exact determinant of an n x n int8 matrix on INT8 tensor cores (src/ozaki_det.cuh) against FLINT fmpz_mat_det.
// g++ -O3 -c bench/det_flint.cpp -o out_dflint.o && nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp -o out_det bench/det.cu out_dflint.o -lflint -lgmp -lgomp
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
#include <vector>
#include <gmp.h>
#include "../src/ozaki_det.cuh"
double flint_det(int n, const int8_t *A, int threads, long *bits, std::string *hex);
double flint_solve(int n, const int8_t *A, const int32_t *b, int threads, const __mpz_struct *num, const mpz_t D, int *ok);

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// det mod the primes pr (a multiple of Pb of them), bad ones flagged
static const int32_t *gU = nullptr, *gL = nullptr;   // the preconditioner, once it is needed

static void det_mod_primes(int n, int Pb, const int8_t *dA, const std::vector<uint32_t> &pr, std::vector<uint32_t> &det,
                           std::vector<int> &bad) {
    oz::DetWork w; w.alloc(n, Pb);
    w.preU = gU; w.preL = gL;
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
    std::vector<int8_t> A((size_t)n * n); uint64_t s = 12345;   // the same LCG as det_flint.cpp's main
    for (auto &a : A) { s = s * 6364136223846793005ull + 1442695040888963407ull; a = (int8_t)((s >> 33) % 255 - 127); }
    if (kind == 1) A[0] = 0;                                                         // a zero 1 x 1 leading minor
    if (kind == 2) for (int k = 0; k < n; k++) A[(size_t)(n - 1) * n + k] = A[k];   // singular: last row = first row
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

    double hr = 0, hc = 0, cmin = 1e300;   // Hadamard over rows and over columns: log2 |det| <= min of the two
    std::vector<double> cn(n, 0);
    for (int i = 0; i < n; i++) {
        double r = 0;
        for (int k = 0; k < n; k++) { double a = A[(size_t)i * n + k]; r += a * a; cn[k] += a * a; }
        hr += 0.5 * log2(r);
    }
    for (int k = 0; k < n; k++) { hc += 0.5 * log2(cn[k]); cmin = std::min(cmin, 0.5 * log2(cn[k])); }
    const double hb = std::min(hr, hc);
    std::vector<char> comp(65280, 0); std::vector<uint32_t> all;   // primes downwards from 65279
    for (int i = 2; i < 65280; i++) if (!comp[i]) for (int k = 2 * i; k < 65280; k += i) comp[k] = 1;
    for (int i = 65279; i > 2; i--) if (!comp[i]) all.push_back(i);
    // the first primes from `from` whose log2 sum reaches `bits` with spares for bad ones, rounded up to a batch
    auto take = [&](size_t from, double bits) {
        std::vector<uint32_t> pr; double lb = 0;
        for (size_t i = from; i < all.size() && (lb < bits * (1.0 + 2.5 * n / 65000.0) + 64 || pr.size() % Pb); i++) {
            pr.push_back(all[i]); lb += log2((double)all[i]);
        }
        return pr;
    };

    int8_t *dA; cudaMalloc(&dA, A.size()); cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    mpz_t D; mpz_init(D);
    const double t0 = now();
    double t_inv = 0, t_lift = 0, t_rr = 0, t_mod = 0, gb = 0;
    long dbits = 0; int good = 0; size_t nprimes = 0, steps = 0;

    if (method == 0) {
        std::vector<uint32_t> pr = take(0, hb + 2), det; std::vector<int> bad, use;
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
        // 1. A^-1 mod p from the LU of [A I]
        oz::DetWork w; w.alloc(n, 1, 2 * n);
        uint32_t p = 0; size_t pi = 0;
        bool singular = false;
        for (;; pi++) {   // 4 plain primes failing = a zero leading minor over Z, or det 0: precondition and try 8 more
            if (pi == 4 && !gU) { make_pre(); w.preU = gU; w.preL = gL; }
            if (pi == 12) { singular = true; break; }
            uint32_t one = 1; int zero = 0; p = all[pi];
            cudaMemcpy(w.P, &p, 4, cudaMemcpyHostToDevice); cudaMemcpy(w.det, &one, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(w.bad, &zero, 4, cudaMemcpyHostToDevice);
            oz::det_mod_batch(w, dA);
            int bad; cudaMemcpy(&bad, w.bad, 4, cudaMemcpyDeviceToHost);
            if (!bad) break;
        }
        if (singular) {   // Monte Carlo, not proved: U A L had a zero pivot modulo 8 primes
            printf("n=%d: det = 0 (8 preconditioned primes all fail; probabilistic)\n", n);
            if (cmp) { long fb; std::string fh; flint_det(n, A.data(), 24, &fb, &fh); printf("  FLINT: %s\n", fh.c_str()); }
            return 0;
        }
        uint16_t *Ai; cudaMalloc(&Ai, (size_t)n * n * 2);
        oz::to_u16<<<1024, 256>>>(w.R, w.ld, n, n, Ai);
        cudaDeviceSynchronize(); w.free_(); t_inv = now() - t0;

        // 2. lifting: enough digits to reconstruct num / den with |num| <= Hc |b| / min col, den <= H
        std::vector<int32_t> b(n); std::vector<uint32_t> bm(n); double bn = 0; uint64_t sb = 777;
        for (int i = 0; i < n; i++) {
            sb = sb * 6364136223846793005ull + 1442695040888963407ull; b[i] = (int)((sb >> 33) % 255) - 127;
            bm[i] = ((b[i] % (int)p) + p) % p; bn += (double)b[i] * b[i];
        }
        const long nb = (long)ceil(hc - cmin + 0.5 * log2(bn)) + 2, db = (long)ceil(hb) + 2;
        steps = (size_t)ceil((nb + db + 2) / log2((double)p));
        int32_t *dr; uint32_t *drm; uint16_t *dx; int *derr;
        cudaMalloc(&dr, n * 4); cudaMalloc(&drm, n * 4); cudaMalloc(&dx, steps * n * 2); cudaMalloc(&derr, 4);
        cudaMemcpy(dr, b.data(), n * 4, cudaMemcpyHostToDevice); cudaMemcpy(drm, bm.data(), n * 4, cudaMemcpyHostToDevice);
        cudaMemset(derr, 0, 4);
        const double t1 = now();
        for (size_t i = 0; i < steps; i++) {
            oz::lift_x<<<n / 8, 256>>>(Ai, drm, dx + i * n, n, p);
            oz::lift_r<<<n / 8, 256>>>(dA, dx + i * n, dr, drm, n, p, derr);
        }
        std::vector<uint16_t> X(steps * n); int err;
        cudaMemcpy(X.data(), dx, X.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(&err, derr, 4, cudaMemcpyDeviceToHost);
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
        const double t2 = now();
        mpz_t Q, xj, num, den; mpz_inits(Q, xj, num, den, nullptr);
        mpz_ui_pow_ui(Q, p, steps); mpz_set_ui(D, 1);
        for (int j = 0; j < 4; j++) {
            mpz_set_ui(xj, 0);
            for (size_t i = steps; i-- > 0;) { mpz_mul_ui(xj, xj, p); mpz_add_ui(xj, xj, X[i * n + j]); }
            if (!ratrec(num, den, xj, Q, nb, db)) { printf("RATREC failed at entry %d\n", j); return 1; }
            mpz_lcm(D, D, den);
        }
        dbits = mpz_sizeinbase(D, 2); t_rr = now() - t2;

        // 4. c = det / d from as many primes as Hadamard / d needs: |c| < 2^(hb - dbits + 1)
        const double t3 = now();
        std::vector<uint32_t> pr = take(0, hb - dbits + 3), det; std::vector<int> bad, use;
        det_mod_primes(n, Pb, dA, pr, det, bad); cudaDeviceSynchronize();
        for (size_t i = 0; i < pr.size(); i++) {
            uint32_t dm = mpz_fdiv_ui(D, pr[i]);
            use.push_back(!bad[i] && dm); good += use.back();
            if (use.back()) det[i] = (uint64_t)det[i] * invm(dm, pr[i]) % pr[i];
        }
        mpz_t c; mpz_init(c); gb = crt(c, pr, det, use); nprimes = pr.size();
        if (gb < hb - dbits + 3) { printf("NOT ENOUGH GOOD PRIMES\n"); return 1; }
        mpz_mul(D, D, c); t_mod = now() - t3;
        mpz_clears(Q, xj, num, den, c, nullptr);
    }
    const double total = now() - t0;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("CUDA: %s\n", cudaGetErrorString(err)); return 1; }
    if (oz::det_prof.on) printf("prof: diag %.3f s  pack %.3f s  gemm %.3f s\n", oz::det_prof.t[0], oz::det_prof.t[1], oz::det_prof.t[2]);
    printf("n=%d method %d  Hadamard %.0f bits  |det| %zu bits  total %.3f s", n, method, hb, mpz_sizeinbase(D, 2), total);
    if (method) printf("  [A^-1 %.3f  lift %zu steps %.3f  ratrec %.3f (d %ld bits)  det/d: %zu primes %.3f]", t_inv, steps,
                       t_lift, t_rr, dbits, nprimes, t_mod);
    else printf("  [%zu primes, %d good]", nprimes, good);
    printf("\n");
    if (cmp) {
        long fb; std::string fh; double ft = flint_det(n, A.data(), 24, &fb, &fh);
        char *gh = mpz_get_str(nullptr, 16, D);
        printf("  FLINT fmpz_mat_det (24 thr) %.2f s  -> %s  (%.1fx)\n", ft, fh == gh ? "BIT-EXACT" : "MISMATCH", ft / total);
        free(gh);
    }
    return 0;
}
