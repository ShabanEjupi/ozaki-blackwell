# hol_proto.py -- exact check of the planned GPU high-order lifting (row form y = b A^-1 mod X^N), small n, Python ints.
# X = p_1..p_s distinct primes; residue chain in RNS (P and Q); c-chain in periodic mixed radix with balanced digits.
import random, sys
from math import log2, gcd, isqrt

n, L = int(sys.argv[1]), int(sys.argv[2])   # n, entry bytes
rnd = random.Random(12345)
A = [[rnd.randrange(-2**(8*L-1), 2**(8*L-1)) for _ in range(n)] for _ in range(n)]
b = [rnd.randrange(-127, 128) for _ in range(n)]
nA = max(abs(v) for r in A for v in r)

def primes_down(lo=3, hi=65280):
    s = bytearray([1]) * hi; s[0] = s[1] = 0
    for i in range(2, int(hi**0.5) + 1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i in range(hi - 1, lo, -1) if s[i]]
allp = primes_down()

def inv_mod_matrix(M, p):   # Gauss-Jordan with pivoting mod p; None if singular
    m = len(M); W = [[x % p for x in r] + [int(i == j) for j in range(m)] for i, r in enumerate(M)]
    for c in range(m):
        piv = next((r for r in range(c, m) if W[r][c]), None)
        if piv is None: return None
        W[c], W[piv] = W[piv], W[c]; iv = pow(W[c][c], -1, p)
        W[c] = [x * iv % p for x in W[c]]
        for r in range(m):
            if r != c and W[r][c]:
                f = W[r][c]; W[r] = [(x - f * y) % p for x, y in zip(W[r], W[c])]
    return [r[m:] for r in W]

need_X = log2(6.25 * n * n * nA)
P, Ainv, lgX, k = [], [], 0.0, 0
while lgX < need_X + 1:
    p = allp[k]; k += 1; I = inv_mod_matrix(A, p)
    if I is None: continue
    P.append(p); Ainv.append(I); lgX += log2(p)
s = len(P); X = 1
for p in P: X *= p
Rb = 1.25 * n * nA
Q = []; lgQ = 0.0
while lgQ < log2(Rb) + 2: Q.append(allp[k]); lgQ += log2(allp[k]); k += 1
w = len(Q)
print(f"n={n} L={L} s={s} (X {lgX:.0f} bit, need {need_X:.1f}) w={w}")

def garner(res, mods):   # mixed-radix digits y_t in [0, m_t): value = y_0 + m_0 (y_1 + m_1 (...))
    y = list(res)
    for t in range(len(mods)):
        for u in range(t): y[t] = (y[t] - y[u]) * pow(mods[u], -1, mods[t]) % mods[t]
    return y
def from_mr(y, mods):
    v = 0
    for t in reversed(range(len(mods))): v = v * mods[t] + y[t]
    return v
def sym(v, M): v %= M; return v - M if v > M // 2 else v
def matmul_mod(Xm, Ym, p):
    Yt = list(zip(*Ym)); return [[sum(a * c for a, c in zip(r, col)) % p for col in Yt] for r in Xm]
def matmul(Xm, Ym):
    Yt = list(zip(*Ym)); return [[sum(a * c for a, c in zip(r, col)) for col in Yt] for r in Xm]

# B0 = A^-1 mod X in [0, X), as mixed-radix digits; R0 = (I - A B0) / X exactly
B0d = [[garner([Ainv[t][i][j] for t in range(s)], P) for j in range(n)] for i in range(n)]
B0 = [[from_mr(B0d[i][j], P) for j in range(n)] for i in range(n)]
AB = matmul(A, B0)
R = [[(int(i == j) - AB[i][j]) for j in range(n)] for i in range(n)]
assert all(v % X == 0 for r in R for v in r); R = [[v // X for v in r] for r in R]
Rs, Ms = [R], []
mods = P + Q
# the needed X-digits: y mod X^N must give num/den with |num| <= H n|b|... take Hadamard on both sides
H = sum(0.5 * log2(sum(v * v for v in r)) for r in A)
N = int((2 * H + log2(n) + 8 + 10) / lgX) + 1
levels = 0
while (2 ** (levels + 1) - 1) < N: levels += 1
print(f"Hadamard {H:.0f} bit, N = {N} X-digits, levels = {levels}")
mx = max(abs(v) for r in R for v in r) / (n * nA)
for i in range(levels - 1):   # R_1 .. R_{levels-1} and M_0 .. M_{levels-2}, all through residues
    Rres = {m: [[v % m for v in r] for r in R] for m in mods}
    Rbar = {m: matmul_mod(Rres[m], Rres[m], m) for m in mods}
    Mres = [matmul_mod(Ainv[t], Rbar[P[t]], P[t]) for t in range(s)]
    Md = [[garner([Mres[t][i2][j] for t in range(s)], P) for j in range(n)] for i2 in range(n)]
    M = [[from_mr(Md[i2][j], P) for j in range(n)] for i2 in range(n)]
    Rn_res = {}
    for q in Q:
        AM = matmul_mod([[v % q for v in r] for r in A], [[v % q for v in r] for r in M], q)
        xi = pow(X % q, -1, q)
        Rn_res[q] = [[(Rbar[q][a][c] - AM[a][c]) * xi % q for c in range(n)] for a in range(n)]
    Qp = 1
    for q in Q: Qp *= q
    Rn = [[sym(from_mr(garner([Rn_res[q][a][c] for q in Q], Q), Q), Qp) for c in range(n)] for a in range(n)]
    Rbar_exact = matmul(R, R); AMx = matmul(A, M)   # the same quantity computed directly with big integers
    for a in range(n):
        for c in range(n):
            t = Rbar_exact[a][c] - AMx[a][c]; assert t % X == 0 and t // X == Rn[a][c], (i, a, c)
    Ms.append(M); R = Rn; Rs.append(R); mx = max(mx, max(abs(v) for r in R for v in r) / (n * nA))
print(f"residue chain: {levels - 1} levels, RNS == big-integer formula, max |R| / (n |A|) = {mx:.3f} (bound 1.25)")

# c-chain in periodic mixed radix: position e has radix P[e % s]; balanced digits
def to_digits(v, E):
    d = []
    for e in range(E):
        p = P[e % s]; r = sym(v, p); d.append(r); v = (v - r) // p
    return d, v                                   # v = what is left above E digits
def from_digits(d):
    v = 0
    for e in reversed(range(len(d))): v = v * P[e % s] + d[e]
    return v
def normalize(cols):                              # cols[e] = arbitrary integers per position -> balanced digits, carry out
    out, c = [], 0
    for e, g in enumerate(cols):
        t = g + c; p = P[e % s]; r = sym(t, p); out.append(r); c = (t - r) // p
    return out, c
ND = s * N
def row_times(cdig, Rm):                          # digit rows (ND x n) times Rm: per position e the row C_e Rm, then carries
    G = [[sum(cdig[e][kk] * Rm[kk][j] for kk in range(n)) for j in range(n)] for e in range(len(cdig))]
    res = []
    for j in range(n):
        dj, _ = normalize([G[e][j] for e in range(len(G))]); res.append(dj)
    return res                                    # res[j] = digits of column j
c = [sum(b[kk] * B0[kk][j] for kk in range(n)) for j in range(n)]   # c_0 = b B0, exact
XN = X ** N
m = 1
for i in range(levels):
    cR_big = [sum(c[kk] * Rs[i][kk][j] for kk in range(n)) for j in range(n)]
    # the same product through digit rows: c mod X^(N - m) as balanced digits, GEMM, carries
    keep = s * (N - m)
    if keep > 0:
        cd = [to_digits(v, keep)[0] for v in c]
        rows = [[cd[j][e] for j in range(n)] for e in range(keep)]
        res = row_times(rows, Rs[i])
        for j in range(n):
            assert (from_digits(res[j]) - cR_big[j]) % X ** (N - m) == 0, ("digit GEMM", i, j)
    bM = [sum(b[kk] * Ms[i][kk][j] for kk in range(n)) for j in range(n)] if i < len(Ms) else [0] * n
    c = [(c[j] + X ** m * cR_big[j] + X ** (2 * m) * bM[j]) % XN for j in range(n)]
    m = 2 * m + 1
# y = c mod X^N must satisfy y A == b (mod X^N)
yA = [sum(c[kk] * A[kk][j] for kk in range(n)) for j in range(n)]
assert all((yA[j] - b[j]) % XN == 0 for j in range(n)), "y A != b mod X^N"
print(f"c-chain: y A == b mod X^{N} for all {n} entries; digit-row GEMM == big-integer product at every level")

def ratrec(u, Mo):   # u = num/den mod Mo, |num|, den < sqrt(Mo/2)
    r0, r1, t0, t1 = Mo, u % Mo, 0, 1
    bnd = isqrt(Mo // 2)
    while r1 > bnd: qq = r0 // r1; r0, r1 = r1, r0 - qq * r1; t0, t1 = t1, t0 - qq * t1
    return (r1, t1) if t1 > 0 else (-r1, -t1)
den = 1
for j in range(4):
    num, d = ratrec(c[j] * den % XN, XN); den *= d
def bareiss(M):
    M = [r[:] for r in M]; sg, prev = 1, 1
    for kk in range(n - 1):
        if M[kk][kk] == 0:
            sw = next(r for r in range(kk + 1, n) if M[r][kk]); M[kk], M[sw] = M[sw], M[kk]; sg = -sg
        for i2 in range(kk + 1, n):
            for j in range(kk + 1, n): M[i2][j] = (M[i2][j] * M[kk][kk] - M[i2][kk] * M[kk][j]) // prev
        prev = M[kk][kk]
    return sg * M[n - 1][n - 1]
D = bareiss(A)
print(f"denominator d: {den.bit_length()} bit, |det| {abs(D).bit_length()} bit, d | det: {D % den == 0}, det/d = {abs(D) // den}")
