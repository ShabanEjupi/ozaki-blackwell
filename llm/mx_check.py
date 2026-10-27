#!/usr/bin/env python3
# mx_check.py <weights dir> <run dir> [samples] -- independent check of mxexact: for random outputs (i, j) the exact value
# of sum_k w_ik x_kj is built with Python integers from the raw safetensors bytes (no shared code with the CUDA side), and
# the GPU float must be the nearest FP32 (ties to even): |v - f| <= |v - neighbour| for both neighbours.
import sys, random
from fractions import Fraction
import numpy as np

wdir, rdir = sys.argv[1], sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
M, K, N, pre = open(rdir + "/meta.txt").read().split()
M, K, N, G = int(M), int(K), int(N), int(K) // 32
pk = np.fromfile("%s/%s.weight_packed.bin" % (wdir, pre), dtype=np.uint8).reshape(M, K // 2)
sc = np.fromfile("%s/%s.weight_scale.bin" % (wdir, pre), dtype=np.uint8).reshape(M, G)
X = np.fromfile(rdir + "/X.bf16", dtype=np.uint16).reshape(N, K)
Ys = {name: np.fromfile("%s/%s.f32" % (rdir, name), dtype=np.float32).reshape(N, M) for name in ("Y", "Yb1", "Yb4")}
E2M1 = [Fraction(v) for v in (0, 0.5, 1, 1.5, 2, 3, 4, 6)]

def weight(i, k):                      # w = E2M1 * 2^(s - 127), low nibble = even k
    b = int(pk[i, k // 2]); nib = b & 15 if k % 2 == 0 else b >> 4
    return (-1 if nib & 8 else 1) * E2M1[nib & 7] * Fraction(2) ** (int(sc[i, k // 32]) - 127)

def act(j, k):                         # bf16 -> exact rational
    b = int(X[j, k]); ex, man = (b >> 7) & 255, b & 127
    v = Fraction(man, 128) * Fraction(2) ** -126 if ex == 0 else (1 + Fraction(man, 128)) * Fraction(2) ** (ex - 127)
    return -v if b & 0x8000 else v

def nearest(v, f):                     # is float32 f the round-to-nearest-even of the rational v?
    f = np.float32(f)
    if not np.isfinite(f): return False
    d = abs(v - Fraction(float(f)))
    for to in (np.float32(np.inf), np.float32(-np.inf)):
        n = np.nextafter(f, to, dtype=np.float32)
        if not np.isfinite(n): continue
        dn = abs(v - Fraction(float(n)))
        if dn < d: return False
        if dn == d and (int(f.view(np.uint32)) & 1): return False
    return True

random.seed(7)
bad = {name: 0 for name in Ys}
for s in range(samples):
    i, j = random.randrange(M), random.randrange(N)
    v = sum(weight(i, k) * act(j, k) for k in range(K) if X[j, k] & 0x7FFF)
    for name, Y in Ys.items():
        bad[name] += not nearest(v, Y[j, i])
print("%d random outputs of %s (N %d): %s" % (samples, pre, N,
      ", ".join("%s %d not correctly rounded" % (n, b) for n, b in bad.items())))
