#!/usr/bin/env python3
# mm2oz.py -- a real Matrix Market matrix (NIST / SuiteSparse) -> the exact integer system the GPU solver takes, without
# one rounding: every double is m 2^e exactly, and row i is multiplied by 2^k_i (the largest denominator in the row), so
# (D A) x = D b has integer entries and the SAME solution x. Rows are permuted by FP64 partial pivoting (P D A x = P D b) so
# that the leading minors are nonzero (the GPU LU does not pivot); the permutation does not change x either.
#   out.oz      "OZM1" npad L n (npad = n rounded up to 128, A (+) I), then the L byte planes of M = (P D A)^T (row form y M = c, y = x^T), plane-major int8 n x n,
#               then c = P D b as int32 (b_i = c_i / 2^k_i: a random right-hand side in [-127, 127] scaled by the rows)
#   out.int     the integer system for SPEX: "n nnz", nnz lines "i j a" of P D A (1-based, a an exact integer), n lines c
#   out.fp64    FP64 references: x from LAPACK dgesv (dense, partial pivoting) and from SuperLU (scipy spsolve), 2n doubles
# Use: python3 -I mm2oz.py matrix.mtx[.gz] out
import sys, numpy as np, scipy.io, scipy.sparse as sp, scipy.sparse.linalg as spl, scipy.linalg as la, time

A = sp.coo_matrix(scipy.io.mmread(sys.argv[1])).tocsr()   # symmetric files come back expanded
n = A.shape[0]
assert A.shape[0] == A.shape[1], "square only"
A.sum_duplicates(); A.eliminate_zeros()
rng = np.random.RandomState(1)
c0 = rng.randint(-127, 128, n).astype(np.int64)
# row scaling: the denominators of doubles are powers of two; k_i = the largest in row i
ints, kexp = [], np.zeros(n, dtype=np.int64)
for i in range(n):
    lo, hi = A.indptr[i], A.indptr[i + 1]
    fr = [float(v).as_integer_ratio() for v in A.data[lo:hi]]
    k = max((d.bit_length() - 1 for _, d in fr), default=0)
    kexp[i] = k
    ints.append([(int(A.indices[j]), nu * (1 << (k - (d.bit_length() - 1)))) for j, (nu, d) in zip(range(lo, hi), fr)])
b = c0 / np.ldexp(1.0, kexp)   # b_i = c_i / 2^k_i, exact in double unless it underflows
assert np.all(b * np.ldexp(1.0, kexp) == c0), "b does not round-trip"
# FP64 references, timed
Ad = A.toarray()
t = time.time(); lu, piv = la.lu_factor(Ad); x_lapack = la.lu_solve((lu, piv), b); t_lapack = time.time() - t
t = time.time(); x_superlu = spl.spsolve(A.tocsc(), b); t_superlu = time.time() - t
perm = np.arange(n)
for i, p in enumerate(piv): perm[i], perm[p] = perm[p], perm[i]   # LAPACK's row interchanges -> P (row i of PA = row perm[i] of A)
# integer system P D A, c = P D b
maxbits = max((abs(v).bit_length() for r in ints for _, v in r), default=1)
L = (maxbits + 1 + 7) // 8   # balanced signed bytes: |a| < 2^(8L - 1)
npad = -(-n // 128) * 128   # the GPU LU takes multiples of 128: A (+) I, c padded with zeros, so x_pad = 0
M = np.zeros((L, npad, npad), dtype=np.int8)   # M = (P D A)^T: M[j][col][row] = byte j of (PDA)[row][col]
for i in range(n, npad): M[0, i, i] = 1
c = np.concatenate([c0[perm], np.zeros(npad - n, dtype=np.int64)])
for r in range(n):
    for col, v in ints[perm[r]]:
        for j in range(L):   # balanced base-256 digits
            d = ((v + 128) & 255) - 128
            M[j, col, r] = d
            v = (v - d) >> 8
        assert v == 0
with open(sys.argv[2] + ".oz", "wb") as f:
    f.write(b"OZM1"); f.write(np.array([npad, L, n], dtype=np.int32).tobytes())
    f.write(M.tobytes()); f.write(c.astype(np.int32).tobytes())
with open(sys.argv[2] + ".int", "w") as f:
    f.write(f"{n} {A.nnz}\n")
    for r in range(n):
        for col, v in ints[perm[r]]: f.write(f"{r + 1} {col + 1} {v}\n")
    f.write("\n".join(str(int(v)) for v in c[:n]) + "\n")
np.concatenate([x_lapack, x_superlu]).astype(np.float64).tofile(sys.argv[2] + ".fp64")
print(f"n {n} (padded {npad}) nnz {A.nnz} entry bits {maxbits} planes L {L} row exponents {kexp.min()}..{kexp.max()}  "
      f"LAPACK {t_lapack:.3f} s  SuperLU {t_superlu:.3f} s  |x| {np.abs(x_lapack).max():.3e}")
