#!/bin/bash
# energy_cpudd.sh [N] [NB] [REPS] [SECS] -- energy per Ax=b solve (kappa 1e20, seed 1) of the blocked CPU double-double LU beside
# the GPU solver, from inside WSL with the machine already held (bench/cpudd_blocked.sh calls it): until 03-11-2026 the paper's
# energy line put the GPU solver against __float128 only. CPU package power: the Windows RAPL counter, 1 s samples streamed
# through powershell.exe (as llm/hold.sh reaches Windows); GPU power: nvidia-smi, 100 ms, each sample stamped here. 10 s idle,
# then REPS blocked CPU solves back to back, then 20 GPU solves after a warm-up (the program's epoch marks bound each window).
# Energy per solve = (mean package + mean GPU power over the window) x the mean solve time; for the CPU the copies of A
# between its solves fall outside the solve time, which can only flatter the CPU.
set -uo pipefail
n=${1:-4096}; nb=${2:-128}; reps=${3:-8}; secs=${4:-600}   # secs: an upper bound; the sampler is stopped after the GPU window
cd "$(dirname "$0")/.." || exit 1
. llm/win.sh || { echo "energy_cpudd: Windows (RAPL) unreachable from here -- nothing measured"; exit 1; }
D=$(mktemp -d)
Q='Get-Counter "\Energy Meter(RAPL_Package0_PKG)\Power" -SampleInterval 1 -MaxSamples '$secs' | ForEach-Object { foreach ($s in $_.CounterSamples) { [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0} {1}", ([DateTimeOffset]$s.Timestamp).ToUnixTimeMilliseconds(), $s.CookedValue) } }'
"$PS" -NoProfile -EncodedCommand "$(printf '%s' "$Q" | iconv -t UTF-16LE | base64 -w0)" \
    < /dev/null 2> /dev/null > $D/rapl & RP=$!   # straight to the file (no buffering pipe), the \r stripped below
SMI=$(command -v nvidia-smi || echo /usr/lib/wsl/lib/nvidia-smi)
$SMI --query-gpu=power.draw --format=csv,noheader,nounits -lms 100 | while read -r p; do echo "$(date +%s.%N) $p"; done > $D/gpu & GP=$!
{ sleep 3; echo "MARK idle-start $(date +%s.%N)"; sleep 10; echo "MARK idle-end $(date +%s.%N)"
  OZ_CPUDD_ONLY=$nb OZ_CPUDD_NOGPU=1 OZ_CPUDD_REPS=$reps ./out_solve $n 1024 20 1 | grep -E 'MARK|blocked'
  sleep 5
  OZ_GPU=1 OZ_REPS=20 ./out_solve $n 1024 20 1 | grep -E 'MARK|all on'; } > $D/marks 2> /dev/null
sleep 2; kill $RP $GP 2> /dev/null; sleep 0.5
cat $D/marks
# mean power in each window: RAPL in mW (the counter's unit; "> 1000 means mW" took an idle package under 1 W for 800 W), GPU in W
awk 'FILENAME ~ /rapl$/ { gsub(/\r/, ""); if (NF == 2) r[++nr] = ($1 / 1000) " " $2; next }
     FILENAME ~ /gpu$/ { g[++ng] = $1 " " $2; next }
     /^MARK/ { m[$2] = $3; next }
     /blocked nb=/ { tc = $(NF-1) }
     /^all on GPU/ { tg += $4; ngs++ }
     END { split("idle cpudd gpu", W, " ");
           for (w = 1; w <= 3; w++) { s = m[W[w] "-start"]; e = m[W[w] "-end"]; ps = pc = gs = gc = 0
               for (i = 1; i <= nr; i++) { split(r[i], a, " "); if (a[1] >= s && a[1] <= e) { ps += a[2]; pc++ } }
               for (i = 1; i <= ng; i++) { split(g[i], a, " "); if (a[1] >= s && a[1] <= e) { gs += a[2]; gc++ } }
               P[w] = pc ? ps / pc / 1000 : -1; G[w] = gc ? gs / gc : -1
               printf "%s: window %.1f s, package %.1f W (%d samples), GPU %.1f W (%d samples)\n", W[w], e - s, P[w], pc, G[w], gc }
           # a window without samples is not a measurement (04-11: 0 package samples printed as -1 W and summed)
           if (P[2] < 0 || P[3] < 0 || G[2] < 0 || G[3] < 0) { print "per solve: NOT MEASURED -- a power window had no samples"; exit 1 }
           ec = tc * (P[2] + G[2]); eg = (m["gpu-end"] - m["gpu-start"]) / 20 * (P[3] + G[3])
           printf "per solve: CPU blocked dd %.2f s x %.1f W = %.0f J | GPU dd %.3f s (window / 20) x %.1f W = %.1f J | CPU/GPU %.1fx\n",
                  tc, P[2] + G[2], ec, (m["gpu-end"] - m["gpu-start"]) / 20, P[3] + G[3], eg, ec / eg }' \
    $D/rapl $D/gpu $D/marks
rm -rf $D
