#!/usr/bin/env python3
# jais_check.py <model dir> <out dir of out_jais --check> -- the FP8 block-scaled map of jais.cu again on the CPU, sharing
# no code with it: weights read from the shard through index.txt, E4M3 from the OCP definition as (significand on 1/8
# steps, exponent), the BF16 block scale from its bits, the input quantized per group of 128 with numpy float32
# arithmetic (scale = max(amax, 1e-10) / 448, q = nearest-even E4M3 of x / scale, saturating), every product and sum
# as Python integers, rounded to BF16 and FP32 by integer arithmetic. 256 rows spread over all row blocks.
# Two mutants must disagree, or the check has no teeth: the scale of the neighbouring block column, and the input
# quantized as x * (1 / scale) (the reciprocal some kernels use) -- the second counts how much that choice matters.
import sys, math, bisect
import numpy as np

mdir, odir = sys.argv[1], sys.argv[2]
T, M, K, unc = open(odir + "/meta.txt").read().split(); M, K = int(M), int(K)
idx = {}
for line in open(mdir + "/index.txt"):
    nm, dt, sh, fl, off, n = line.split(); idx[nm] = (fl, int(off), int(n))
def tensor(name, dtype):
    fl, off, n = idx[name]
    return np.fromfile("%s/%s" % (mdir, fl), dtype=dtype, count=n // np.dtype(dtype).itemsize, offset=off)
w = tensor(T + ".weight", np.uint8).reshape(M, K)
sc = tensor(T + ".weight_scale", np.uint16).reshape(M // 128, K // 128)
x = np.fromfile(odir + "/x.f32", dtype=np.float32)
y8, y24 = np.fromfile(odir + "/y8.f32", dtype=np.float32), np.fromfile(odir + "/y24.f32", dtype=np.float32)

def e4m3(b):                       # (sign, integer significand on 1/8 steps, exponent): v = sign * sig/8 * 2^ex
    e, m = (b >> 3) & 15, b & 7
    return (-1 if b & 0x80 else 1), (m if e == 0 else 8 + m), (-6 if e == 0 else e - 7)
VALS = sorted((s * 2.0 ** (ex - 3), c) for c in range(0x7F) for s, ex in [e4m3(c)[1:]])   # finite, non-negative
VV = [v for v, c in VALS]
def to_e4m3(t):                    # nearest E4M3 code to t >= 0, ties to the even code, saturating at 448
    if t >= 448.0: return 0x7E
    i = bisect.bisect_left(VV, t)
    if i < len(VV) and VV[i] == t: return VALS[i][1]
    lo, hi = VALS[i - 1], VALS[i]
    dl, dh = t - lo[0], hi[0] - t
    return lo[1] if dl < dh or (dl == dh and lo[1] % 2 == 0) else hi[1]

def quant(x, recip=False):         # per group of 128: list of (integer, exponent) with value = integer * 2^exponent
    out = []
    for g in range(0, K, 128):
        xs = x[g:g + 128]
        s = np.float32(max(np.float32(np.max(np.abs(xs))), np.float32(1e-10))) / np.float32(448)
        r = np.float32(1) / s
        fm, fe = math.frexp(float(s)); sm, se = int(fm * 2 ** 24), fe - 24          # s = sm 2^se exactly
        for v in xs:
            t = np.float32(v) * r if recip else np.float32(v) / s
            c = to_e4m3(abs(float(t)))
            _, sig, ex = e4m3(c)
            out.append(((-1 if t < 0 else 1) * sig * sm, ex - 3 + se))
    return out

def round_p(n, e, p):              # n * 2^e to p significant bits, nearest even, FP32 exponent range
    if n == 0: return 0.0
    sg = -1 if n < 0 else 1; n = abs(n)
    lsb = max(n.bit_length() - 1 + e - (p - 1), -126 - (p - 1)); sh = lsb - e
    if sh <= 0: q = n << -sh
    else:
        q, r = n >> sh, n & ((1 << sh) - 1); half = 1 << (sh - 1)
        if r > half or (r == half and q & 1): q += 1
    return sg * math.ldexp(q, lsb)

def bf16(u):                       # BF16 bits -> (integer significand, exponent)
    ex, f = (u >> 7) & 255, u & 127
    return (128 + f, ex - 134) if ex else (f, -133)
WS = np.array([[bf16(int(u)) for u in row] for row in sc], dtype=np.int64)          # (sig, exp) per block
WE = np.array([[e4m3(int(b)) for b in range(256)]], dtype=np.int64)[0]              # (sign, sig, exp) per code

def row_value(r, a, shift=0):      # exact sum over k as (integer, exponent)
    cw = WE[w[r]]                                                                   # K x 3
    blk = (np.arange(K) // 128 + shift) % (K // 128)
    ss, se = WS[r // 128, blk, 0], WS[r // 128, blk, 1]
    am = np.array([v for v, e in a], dtype=np.int64); ae = np.array([e for v, e in a], dtype=np.int64)
    P = cw[:, 0] * cw[:, 1] * ss * am                                               # |P| < 2^41
    E = (cw[:, 2] - 3) + se + ae
    e0 = int(E.min()); tot = 0
    for e in np.unique(E): tot += int(P[E == e].sum()) << int(e - e0)              # a group sum < 2^54
    return tot, e0

rows = range(0, M, max(1, M // 256))
a, ar = quant(x), quant(x, recip=True)
bad8 = bad24 = mut_scale = mut_recip = 0
for r in rows:
    n, e = row_value(r, a)
    bad8 += round_p(n, e, 8) != float(y8[r]); bad24 += round_p(n, e, 24) != float(y24[r])
    n2, e2 = row_value(r, a, shift=1); mut_scale += round_p(n2, e2, 8) != float(y8[r])
    n3, e3 = row_value(r, ar); mut_recip += round_p(n3, e3, 8) != float(y8[r])
print("%s: %d rows x %d on the CPU (numpy + Python integers): BF16 %d differ, FP32 %d differ (GPU uncertified %s); "
      "mutants: neighbouring scale column %d differ, x * (1/scale) %d differ, input elements quantized differently %d of %d"
      % (T, len(rows), K, bad8, bad24, unc, mut_scale, mut_recip, sum(p != q for p, q in zip(a, ar)), K))
sys.exit(1 if bad8 or bad24 or not mut_scale else 0)
