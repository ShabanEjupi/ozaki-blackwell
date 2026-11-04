"""python -m ozaki_blackwell [n]: self-test against exact rational sums, then the GPU's speed at n (default 4096)."""
import sys
import time
from fractions import Fraction

import numpy as np

import ozaki_blackwell as ob


def exact_err(parts_a, parts_b, parts_c, unit, samples, rng):
    """max over sampled (i, j) of |C - AB| / (|A||B|) in units of `unit`, all in exact rationals"""
    m, n = parts_c[0].shape
    k = parts_a[0].shape[1]
    worst = 0.0
    for _ in range(samples):
        i, j = int(rng.integers(m)), int(rng.integers(n))
        a = [sum(Fraction(float(p[i, t])) for p in parts_a) for t in range(k)]
        b = [sum(Fraction(float(p[t, j])) for p in parts_b) for t in range(k)]
        ref = sum(x * y for x, y in zip(a, b))
        mag = sum(abs(x * y) for x, y in zip(a, b))
        got = sum(Fraction(float(p[i, j])) for p in parts_c)
        worst = max(worst, float(abs(got - ref) / mag / Fraction(unit)) if mag else 0.0)
    return worst


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    d = ob.device()
    print(f"ozaki_blackwell {ob.__version__} on {d['name']} (sm_{d['sm']}, {d['sms']} SMs, L2 {d['l2_kib'] // 1024} MiB)")
    rng = np.random.default_rng(2026)
    m = k = 256
    ok = True

    def rnd(spread, scale=1.0):
        return rng.standard_normal((m, k)) * np.exp2(rng.integers(-spread, spread + 1, (m, k))) * scale

    for spread in (0, 8):
        a, b = rnd(spread), rnd(spread)
        e = exact_err([a], [b], [ob.matmul(a, b)], 2.0 ** -53, 32, rng)
        e_np = exact_err([a], [b], [a @ b], 2.0 ** -53, 32, rng)
        good = e <= 8
        ok &= good
        print(f"  FP64, K={k}, spread 2^+-{spread}: {e:.3g} units of 2^-53 (NumPy's FP64 on the CPU: {e_np:.3g})  {'ok' if good else 'FAIL'}")
    a = (rnd(0), rnd(0, 2.0 ** -53))
    b = (rnd(0), rnd(0, 2.0 ** -53))
    e = exact_err(a, b, ob.matmul_dd(a, b), 2.0 ** -106, 16, rng)
    good = e <= 16
    ok &= good
    print(f"  double-double, K={k}: {e:.3g} units of 2^-106  {'ok' if good else 'FAIL'}")
    a = (rnd(0), rnd(0, 2.0 ** -53), rnd(0, 2.0 ** -106))
    b = (rnd(0), rnd(0, 2.0 ** -53), rnd(0, 2.0 ** -106))
    e = exact_err(a, b, ob.matmul_td(a, b), 2.0 ** -159, 8, rng)
    good = e <= 16
    ok &= good
    print(f"  triple-double, K={k}: {e:.3g} units of 2^-159  {'ok' if good else 'FAIL'}")

    print(f"speed at n={n}, on the device, median of 5 (FP64-equivalent TFLOPS):")
    for kind in ("fp64", "dd", "td"):
        ms, tf = ob.bench(n, kind)
        print(f"  {kind:5s} {ms:9.2f} ms  {tf:6.2f} TFLOPS")
    t = time.perf_counter()
    a, b = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    t = time.perf_counter()
    ob.matmul(a, b)
    host = time.perf_counter() - t
    print(f"  fp64 from NumPy arrays, transfers included: {host * 1e3:.1f} ms")
    ob.release()
    print("self-test", "passed" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
