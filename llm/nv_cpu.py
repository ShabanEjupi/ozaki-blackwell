#!/usr/bin/env python3
# nv_cpu.py <weights dir> <run dir>/<up_proj|down_proj> -- the whole NVFP4 product again on the CPU, every output, and a
# bit-for-bit comparison with the GPU tensor-core result. Shares no code with nvexact.cu: block sums by float64 BLAS (exact:
# |2w 2a| <= 144, 16 terms), scales from the OCP E4M3 definition, the global scales as Python fractions, the sum as Python
# integers, rounding to FP32 by integer arithmetic. Correct rounding makes the answer a property of the model, not of the
# machine: a GPU tensor core and a CPU must print the same bits.
import sys, struct, time
from fractions import Fraction
import numpy as np

wdir, rdir = sys.argv[1], sys.argv[2]
M, K, N, pre = open(rdir + "/meta.txt").read().split()
M, K, N, B = int(M), int(K), int(N), int(K) // 16
t0 = time.time()
pk = np.fromfile("%s/%s.weight.bin" % (wdir, pre), dtype=np.uint8).reshape(M, K // 2)
sw = np.fromfile("%s/%s.weight_scale.bin" % (wdir, pre), dtype=np.uint8).reshape(M, B)
gW = Fraction(struct.unpack("<f", open("%s/%s.weight_scale_2.bin" % (wdir, pre), "rb").read())[0])
gA = Fraction(struct.unpack("<f", open("%s/%s.input_scale.bin" % (wdir, pre), "rb").read())[0])
qa = np.fromfile(rdir + "/qa.i8", dtype=np.int8).reshape(N, K).astype(np.float64)
sa = np.fromfile(rdir + "/sa.e4m3", dtype=np.uint8).reshape(N, B)
Y = np.fromfile(rdir + "/Y.f32", dtype=np.float32).reshape(N, M)

T = np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.float64)          # 2 * E2M1
def nib(n): return np.where(n & 8, -1.0, 1.0) * T[n & 7]
qw = np.empty((M, K)); qw[:, 0::2] = nib(pk & 15); qw[:, 1::2] = nib(pk >> 4)

def e4m3(b):                       # OCP E4M3 as (integer significand on 1/8 steps, exponent): v = sig/8 * 2^ex
    e, m = (b >> 3) & 15, b & 7
    return np.where(e == 0, m, 8 + m).astype(np.int64), np.where(e == 0, -6, e.astype(np.int64) - 7)
sigW, exW = e4m3(sw); sigA, exA = e4m3(sa)
# value = gW gA / 4 * sum_b (sigW sigA / 64) 2^(exW + exA) p_b; exW + exA >= -12, so scale every term by 2^12
acc = np.zeros((M, N), dtype=np.int64)
for b in range(B):
    p = (qw[:, 16 * b:16 * b + 16] @ qa[:, 16 * b:16 * b + 16].T).astype(np.int64)       # exact in float64
    c = sigW[:, b:b + 1] * sigA[:, b][None, :]
    sh = exW[:, b:b + 1] + exA[:, b][None, :] + 12
    acc += (c * p) << sh
G = gW * gA / (4 * 64 * 2 ** 12)                                    # value = acc * G
Gn, Gd = G.numerator, G.denominator                                  # Gd is a power of two
assert Gd & (Gd - 1) == 0
gd = Gd.bit_length() - 1

def f32_bits(a):                   # round a * Gn / 2^gd to nearest-even FP32, return the bit pattern
    if a == 0: return 0
    s = 0x80000000 if a < 0 else 0
    v = abs(a) * Gn                                                    # value = v 2^-gd
    e = v.bit_length() - 1 - gd                                        # 2^e <= value < 2^(e+1)
    q = max(e, -126) - 23                                              # ulp exponent (subnormals below 2^-126)
    sh = -gd - q                                                       # mantissa integer = v 2^(sh)
    if sh >= 0: m = v << sh
    else:
        m, r = v >> -sh, v & ((1 << -sh) - 1); half = 1 << (-sh - 1)
        if r > half or (r == half and m & 1): m += 1
    if m >= 1 << 24: m >>= 1; q += 1                                   # carried into the next binade
    if m < 1 << 23: return s | m                                       # subnormal (q = -149)
    E = q + 23 + 127
    if E >= 255: return s | 0x7F800000
    return s | (E << 23) | (m - (1 << 23))

gpu = Y.view(np.uint32)
bad = 0
for i in range(M):
    col = acc[i]
    for j in range(N):
        if f32_bits(int(col[j])) != int(gpu[j, i]): bad += 1
print("%s: all %d outputs on the CPU (numpy + Python integers), %d differ from the GPU tensor cores (%.0f s)"
      % (pre, M * N, bad, time.time() - t0))
