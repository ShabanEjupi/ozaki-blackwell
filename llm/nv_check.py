#!/usr/bin/env python3
# nv_check.py <weights dir> <run dir>/<up_proj|down_proj> [samples] -- independent check of nvexact: for random outputs (i, j)
# the exact value of sum_k w_ik a_jk is built with Python rationals from the raw safetensors bytes of the NVFP4 weight
# (E2M1 nibbles, low nibble = even k; E4M3 block scales; FP32 global scale) and the dumped NVFP4 activations (2 E2M1 as
# int8, E4M3 block scales, the checkpoint's input_scale). No code is shared with the CUDA side; the GPU float must be the
# nearest FP32 (ties to even): |v - f| <= |v - neighbour| for both neighbours.
import sys, random, struct
from fractions import Fraction
import numpy as np

wdir, rdir = sys.argv[1], sys.argv[2]
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 500
M, K, N, pre = open(rdir + "/meta.txt").read().split()
M, K, N, B = int(M), int(K), int(N), int(K) // 16
pk = np.fromfile("%s/%s.weight.bin" % (wdir, pre), dtype=np.uint8).reshape(M, K // 2)
sw = np.fromfile("%s/%s.weight_scale.bin" % (wdir, pre), dtype=np.uint8).reshape(M, B)
gW = Fraction(struct.unpack("<f", open("%s/%s.weight_scale_2.bin" % (wdir, pre), "rb").read())[0])
gA = Fraction(struct.unpack("<f", open("%s/%s.input_scale.bin" % (wdir, pre), "rb").read())[0])
qa = np.fromfile(rdir + "/qa.i8", dtype=np.int8).reshape(N, K)
sa = np.fromfile(rdir + "/sa.e4m3", dtype=np.uint8).reshape(N, B)
Ys = {name: np.fromfile("%s/%s.f32" % (rdir, name), dtype=np.float32).reshape(N, M) for name in ("Y", "Yb1", "Yb4")}
E2M1 = [Fraction(v) for v in ("0", "1/2", "1", "3/2", "2", "3", "4", "6")]

def e4m3(b):                           # OCP FP8 E4M3: bias 7, subnormals 2^-6 * m/8
    e, m = (b >> 3) & 15, b & 7
    v = Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7)
    return -v if b & 0x80 else v

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
    v = Fraction(0)
    for b in range(B):
        sw_b, sa_b, acc = e4m3(int(sw[i, b])), e4m3(int(sa[j, b])), Fraction(0)
        for k in range(16 * b, 16 * b + 16):
            q = int(qa[j, k])
            if q == 0: continue
            byte = int(pk[i, k // 2]); nib = byte & 15 if k % 2 == 0 else byte >> 4
            acc += (-1 if nib & 8 else 1) * E2M1[nib & 7] * Fraction(q, 2)
        v += acc * sw_b * sa_b
    v *= gW * gA
    for name, Y in Ys.items():
        bad[name] += not nearest(v, Y[j, i])
print("%d random outputs of %s (N %d): %s" % (samples, pre, N,
      ", ".join("%s %d not correctly rounded" % (n, b) for n, b in bad.items())))
