#!/usr/bin/env python3
# real_table.py -- the real-system table of README / paper from /opt/ozaki/real/*: GPU time to the certified solution
# (A^-1 + lifting + reconstruction + all numerators; det/d apart), SPEX Left LU time, digest agreement, FP64 digits.
import re, os, sys, subprocess
K = {"west0989": "5.7e12", "qh1484": "5.6e17", "fidap032": "5.2e20", "fidap006": "3.5e21", "cry2500": "4.4e17", "fidapm07": "6.8e19",
     "fidap026": "4.7e18", "mhd3200a": "6.5e39", "fidap014": "2.6e16", "lns_3937": "1.0e17", "fidap012": "2.2e20", "mhd4800a": "3.2e47",
     "sherman3": "6.9e16"}   # NIST Matrix Market "condition number (est.)"
R = "/opt/ozaki/real"
rows = []
for N in K:
    d = f"{R}/{N}"
    if not os.path.exists(f"{d}/gpu.log"): continue
    c, g = open(f"{d}/conv.log").read(), open(f"{d}/gpu.log").read()
    s = open(f"{d}/spex.log").read() if os.path.exists(f"{d}/spex.log") else ""
    m = re.search(r"n (\d+) \(padded \d+\) nnz (\d+) entry bits (\d+)", c)
    t = lambda k: float(re.search(k + r" ([0-9.]+)", g).group(1))
    tg = t(r"\[A\^-1") + t(r"steps") + t(r"ratrec") + float(re.search(r"digest [0-9a-f]+; ([0-9.]+) s", g).group(1))
    dg = re.search(r"digest ([0-9a-f]+);", g).group(1); ok = g.count(": ok") == 2
    sm = re.search(r"  ([0-9.]+) s  D \d+ bits.*digest ([0-9a-f]+)", s)
    D = re.search(r"D (\d+) bits, numerators up to", g).group(1)
    e = subprocess.run(["python3", "-I", os.path.dirname(os.path.abspath(__file__)) + "/fp64_err.py", d, m.group(1), "0"], capture_output=True, text=True).stdout
    fd = re.findall(r"\(([0-9.]+) digits\).*?no correct digit (\d+)/", e)
    spex = (f"{float(sm.group(1)):.1f}" if sm else ("> 7200" if "SPEX" not in s and os.path.exists(f"{d}/spex.log") else "--"))
    same = ("yes" if sm and sm.group(2) == dg else ("--" if not sm else "NO"))
    ratio = f"{float(sm.group(1)) / tg:.0f}x" if sm else ""
    rows.append(f"| {N} | {m.group(1)} | {m.group(2)} | {K[N]} | {m.group(3)} | {D} | {tg:.2f} | {spex} | {ratio} | {same} | {'ok' if ok else 'FAIL'} | {fd[0][0]} ({fd[0][1]}) | {fd[1][0]} ({fd[1][1]}) |")
print("| matrix | n | nnz | kappa (NIST est.) | entry bits | denominator bits | GPU s | SPEX s | | same x | y M = c mod 2 primes | FP64 LAPACK digits (entries with none) | FP64 SuperLU digits (entries with none) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---:|")
print("\n".join(rows))
