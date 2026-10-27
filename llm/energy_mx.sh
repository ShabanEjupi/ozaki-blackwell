#!/bin/bash
# energy_mx.sh [N] [seconds] -- joules per correctly rounded MXFP4 product (llm/mxexact.cu MX_ENERGY) against the BF16
# baseline, real K3 expert 0 w1 (3072 x 3584), N tokens: nvidia-smi power every 100 ms, averaged over each phase.
N=${1:-4096}; T=${2:-10}; cd $(dirname $0)/..
nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms 100 > /tmp/pw.csv & P=$!
sleep 2
MX_ENERGY=$T ./out_mx /opt/ozaki/k3/L5 language_model.model.layers.5.block_sparse_moe.experts.0.w1 $N 4 1 | grep ENERGY > /tmp/ph.txt
kill $P
python3 - "$N" <<'PY'
import sys, datetime
N = int(sys.argv[1]); pw = []
for l in open("/tmp/pw.csv"):
    ts, w = [x.strip() for x in l.split(",")]
    pw.append((datetime.datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(w)))
for l in open("/tmp/ph.txt"):
    _, name, a, b, it = l.split(); a, b, it = float(a), float(b), int(it)
    w = [p for t, p in pw if a + 0.5 < t < b - 0.2]
    avg = sum(w) / len(w); per = (b - a) / it
    print("%-6s %d products of 3072 x 3584 x %d in %.1f s: %.3f ms each, %.1f W, %.3f J per product, %.1f GFLOP/J"
          % (name, it, N, b - a, 1e3 * per, avg, avg * per, 2 * 3072 * 3584 * N / (avg * per) / 1e9))
PY
