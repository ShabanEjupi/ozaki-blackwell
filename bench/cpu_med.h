// cpu_med.h -- the CPU baselines' size and repetitions (td_test.cu, mp_test.cu): OZ_CPU_N=<n> measures them at that size
// instead of the test's default (in cache, then scaled by n^3: 03-11-2026, runs at n = 512 varied up to 2.8x), OZ_CPU_REP=<r>
// runs each r times; cpu_median returns the median and lists every run's seconds in `all`.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static int cpu_n(int d) { return getenv("OZ_CPU_N") ? atoi(getenv("OZ_CPU_N")) : d; }
static int cpu_rep() { return getenv("OZ_CPU_REP") ? std::max(1, atoi(getenv("OZ_CPU_REP"))) : 1; }
template <class F> static double cpu_median(F run, std::string &all) {
    std::vector<double> t;
    for (int r = 0; r < cpu_rep(); r++) {
        auto t0 = std::chrono::steady_clock::now(); run();
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    all.clear(); char b[32]; for (double x : t) { snprintf(b, sizeof b, " %.3f", x); all += b; }
    std::sort(t.begin(), t.end());
    return t.size() % 2 ? t[t.size() / 2] : 0.5 * (t[t.size() / 2 - 1] + t[t.size() / 2]);
}
