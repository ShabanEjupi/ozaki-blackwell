// crt_check.cpp -- the direct-sum CRT of ozaki_dd.cuh on the host, against a known 200-bit integer.
// g++ -O2 bench/crt_check.cpp -o out_crt && ./out_crt
#include <cmath>
#include <cstdio>
#include <random>
static const int MOD[32] = {256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193,
                            191, 181, 179, 173, 167, 163, 157, 151, 149, 139, 137, 131, 127, 113, 109, 107};
static int inv_mod(int a, int m) { for (int x = 1; x < m; x++) if (a * x % m == 1) return x; return 0; }
int main() {
    unsigned W[32][8] = {}, M[8] = {}; long long F[32] = {};
    auto mul_small = [](unsigned *a, unsigned m) { unsigned long long c = 0; for (int k = 0; k < 8; k++) { c += (unsigned long long)a[k] * m; a[k] = (unsigned)c; c >>= 32; } };
    M[0] = 1; for (int l = 0; l < 32; l++) mul_small(M, MOD[l]);
    for (int l = 0; l < 32; l++) {
        int r = 1; W[l][0] = 1;
        for (int t = 0; t < 32; t++) if (t != l) { mul_small(W[l], MOD[t]); r = r * (MOD[t] % MOD[l]) % MOD[l]; }
        int w = inv_mod(r, MOD[l]); mul_small(W[l], w); F[l] = llround(ldexp((double)w / MOD[l], 40));
    }
    printf("M limbs:"); for (int k = 7; k >= 0; k--) printf(" %08x", M[k]); printf("\n");
    std::mt19937_64 g(3); int bad = 0;
    for (int it = 0; it < 100000; it++) {
        unsigned V[8] = {}; int nl = 1 + g() % 6; for (int k = 0; k < nl; k++) V[k] = (unsigned)g();
        bool sgn = g() & 1;
        if (sgn) { unsigned long long b = 1; for (int k = 0; k < 8; k++) { b += (unsigned)~V[k]; V[k] = (unsigned)b; b >>= 32; } }
        int r[32];   // V mod m, V two's complement 256-bit: reduce limbs from the top, then fix the sign by 2^256 mod m
        for (int l = 0; l < 32; l++) {
            long long x = 0; for (int k = 7; k >= 0; k--) x = ((x << 32) + V[k]) % MOD[l];
            if (sgn) { long long p = 1; for (int t = 0; t < 256; t++) p = p * 2 % MOD[l]; x = ((x - p) % MOD[l] + MOD[l]) % MOD[l]; }
            r[l] = (int)x;
        }
        unsigned long long acc[8] = {}; long long qf = 0;
        for (int l = 0; l < 32; l++) { qf += (long long)r[l] * F[l]; for (int k = 0; k < 8; k++) acc[k] += (unsigned long long)r[l] * W[l][k]; }
        long long q = (qf + (1ll << 39)) >> 40, c = 0; unsigned v[8];
        for (int k = 0; k < 8; k++) { c += (long long)acc[k] - q * (long long)M[k]; v[k] = (unsigned)c; c >>= 32; }
        bool same = true; for (int k = 0; k < 8; k++) same &= v[k] == V[k];
        if (!same && bad++ < 3) { printf("it %d sgn %d q %lld\n V:", it, sgn, q); for (int k = 7; k >= 0; k--) printf(" %08x", V[k]); printf("\n v:"); for (int k = 7; k >= 0; k--) printf(" %08x", v[k]); printf("\n"); }
    }
    printf("%d of 100000 wrong\n", bad);
}
