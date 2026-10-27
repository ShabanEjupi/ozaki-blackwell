#!/usr/bin/env python3
# fp64_err.py DIR n npad -- how wrong FP64 is on this system: LAPACK dgesv and SuperLU against the exact x (det.cu OZ_FILE
# writes the exact solution rounded to doubles). Relative error in the max norm, the worst entry, and the correct digits.
import sys, numpy as np
d, n, npad = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
x = np.fromfile(d + "/s.oz.x", dtype=np.float64)[:n]; f = np.fromfile(d + "/s.fp64", dtype=np.float64)
for name, y in (("LAPACK", f[:n]), ("SuperLU", f[n:2 * n])):
    e = np.abs(y - x).max() / np.abs(x).max(); w = np.abs(y - x) / np.maximum(np.abs(x), 1e-300)
    print(f"  FP64 {name}: normwise rel. error {e:.2e} ({max(0.0, -np.log10(max(e, 1e-17))):.1f} digits), worst entry rel. error {np.nanmax(w):.2e}, entries with no correct digit {int((w >= 1).sum())}/{n}")
