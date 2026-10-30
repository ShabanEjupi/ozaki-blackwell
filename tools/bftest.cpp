// bftest: bf (bits) against bf_slow (frexp/ldexp/nearbyint) -- random doubles over the whole range, ties, and the edges
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>
#include <chrono>
#include <vector>
#define BF_ONLY
static float bf_slow(double v) { if (v == 0 || !std::isfinite(v)) return (float)v; int ex; frexp(v, &ex); int q = std::max(ex - 8, -133); return (float)ldexp(nearbyint(ldexp(v, -q)), q); }
static float bf(double v) { if (!(fabs(v) >= 0x1p-126) || !std::isfinite(v)) return bf_slow(v);
    uint64_t u; memcpy(&u, &v, 8); u += ((1ULL << 44) - 1) + ((u >> 45) & 1); u &= ~((1ULL << 45) - 1); double r; memcpy(&r, &u, 8); return (float)r; }
static bool same(float a, float b) { uint32_t x, y; memcpy(&x, &a, 4); memcpy(&y, &b, 4); return x == y || (a != a && b != b); }
int main() {
    std::mt19937_64 rng(30102026); long long n = 0, bad = 0;
    auto chk = [&](double v) { n++; if (!same(bf(v), bf_slow(v))) { if (bad++ < 5) printf("differ %a: %a vs %a\n", v, (double)bf(v), (double)bf_slow(v)); } };
    for (long long i = 0; i < 200000000; i++) { uint64_t u = rng(); double v; memcpy(&v, &u, 8); chk(v); }              // every bit pattern
    for (long long i = 0; i < 100000000; i++) { double v = ldexp((double)(rng() >> 11), (int)(rng() % 300) - 200 - 53); chk(rng() & 1 ? v : -v); }  // the used range
    for (int e = -140; e < 140; e++) for (int m = 0; m < 512; m++) for (int d = -2; d <= 2; d++) {      // ties and their neighbours
        double t = ldexp(256.0 + m * 0.5, e - 8); uint64_t u; memcpy(&u, &t, 8); u += d; memcpy(&t, &u, 8); chk(t); chk(-t); }
    for (double v : {0x1p-126, 0x1.ffp-127, 0x1.fffffffffffffp-127, 0x1.ff8p+127, 0x1.ff7fffffffffffp+127, 0x1p+128, 1e308, -0.0, (double)INFINITY, -(double)INFINITY, (double)NAN}) chk(v);
    std::vector<double> xs(1 << 24); for (auto &x : xs) x = ldexp((double)(rng() >> 11), (int)(rng() % 40) - 73);
    float s = 0; auto t0 = std::chrono::steady_clock::now(); for (double x : xs) s += bf_slow(x); auto t1 = std::chrono::steady_clock::now(); for (double x : xs) s += bf(x); auto t2 = std::chrono::steady_clock::now();
    printf("%lld values, %lld differ; slow %.2f ns, bits %.2f ns a call (%g)\n", n, bad, std::chrono::duration<double, std::nano>(t1 - t0).count() / xs.size(),
           std::chrono::duration<double, std::nano>(t2 - t1).count() / xs.size(), s);
}
