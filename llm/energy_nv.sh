#!/bin/bash
# energy_nv.sh [N] [seconds] [layer] -- joules per correctly rounded NVFP4 product (llm/nvexact.cu NV_ENERGY) against the
# native block-scaled FP4 MMA, Nemotron 3 Ultra expert 0 (up_proj 5120 x 2048, down_proj 2048 x 5120), N tokens:
# nvidia-smi power every 100 ms, averaged over each phase.
N=${1:-4096}; T=${2:-10}; L=${3:-49}; cd $(dirname $0)/..
nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm --format=csv,noheader,nounits -lms 100 > /tmp/pw.csv & P=$!
sleep 2
NV_ENERGY=$T ./out_nv /opt/ozaki/nv/L$L backbone.layers.$L.mixer.experts.0 $N 1 | grep ENERGY > /tmp/ph.txt
kill $P
python3 - <<'PY'
import datetime
pw = []
for l in open("/tmp/pw.csv"):
    ts, w, clk = [x.strip() for x in l.split(",")]
    pw.append((datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(w), float(clk)))
for l in open("/tmp/ph.txt"):
    _, name, a, b, it, M, N, K = l.split(); a, b, it, M, N, K = float(a), float(b), int(it), int(M), int(N), int(K)
    s = [(w, c) for t, w, c in pw if a + 0.5 < t < b - 0.2]
    avg = sum(w for w, c in s) / len(s); clk = sum(c for w, c in s) / len(s); per = (b - a) / it
    print("%-17s %4d products %d x %d x %d: %.3f ms each, %.1f W, SM %.0f MHz, %.2f mJ per product, %.0f GOP/J"
          % (name, it, M, K, N, 1e3 * per, avg, clk, 1e3 * avg * per, 2 * M * K * N / (avg * per) / 1e9))
PY
