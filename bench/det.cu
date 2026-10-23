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
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <string>
#include <vector>
#include <gmp.h>
#include "../src/ozaki_det.cuh"
double flint_det(int n, const int8_t *A, int threads, long *bits, std::string *hex);

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// det mod the primes pr (a multiple of Pb of them), bad ones flagged
static void det_mod_primes(int n, int Pb, const int8_t *dA, const std::vector<uint32_t> &pr, std::vector<uint32_t> &det,
                           std::vector<int> &bad) {
    oz::DetWork w; w.alloc(n, Pb);
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
              method = argc > 4 ? atoi(argv[4]) : 1;
    std::vector<int8_t> A((size_t)n * n); uint64_t s = 12345;   // the same LCG as det_flint.cpp's main
    for (auto &a : A) { s = s * 6364136223846793005ull + 1442695040888963407ull; a = (int8_t)((s >> 33) % 255 - 127); }

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
        det_mod_primes(n, Pb, dA, pr, det, bad); cudaDeviceSynchronize(); t_mod = now() - t0;
        for (int b : bad) { use.push_back(!b); good += !b; }
        gb = crt(D, pr, det, use); nprimes = pr.size();
        if (gb < hb + 2) { printf("NOT ENOUGH GOOD PRIMES\n"); return 1; }
    } else {
        // 1. A^-1 mod p from the LU of [A I]
        oz::DetWork w; w.alloc(n, 1, 2 * n);
        uint32_t p = 0; size_t pi = 0;
        for (; pi < 8; pi++) {   // 8 primes failing = a leading minor that vanishes over Z (or det 0): needs preconditioning
            uint32_t one = 1; int zero = 0; p = all[pi];
            cudaMemcpy(w.P, &p, 4, cudaMemcpyHostToDevice); cudaMemcpy(w.det, &one, 4, cudaMemcpyHostToDevice);
            cudaMemcpy(w.bad, &zero, 4, cudaMemcpyHostToDevice);
            oz::det_mod_batch(w, dA);
            int bad; cudaMemcpy(&bad, w.bad, 4, cudaMemcpyDeviceToHost);
            if (!bad) break;
        }
        if (pi == 8) { printf("n=%d: no LU without pivoting for 8 primes (singular or a zero leading minor); not handled yet\n", n); return 2; }
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
