#!/bin/bash
# fd_check.sh [NL] [N] -- finite differences through the first NL layers (default 4: three dense + one MoE) on N tokens
# (default 128) of the law ActID 2512: one step prints the predicted L(+e) - L(-e) = <g_0, D+ - D-> for every seed in
# SEEDS and every e in EPS (the moves as rounded to BF16; seed 0 = sign(g_0), written to $O/gsign.i8 by that step); two
# forward-only runs per pair in PAIRS ("seed:e ...", default every seed with every e) measure it. The summary is the
# least-squares slope of measured against predicted per e (1 = the backward agrees) and the largest residual. EPS values
# are written as printf %g prints them (5e-05, 0.0001). Correctness only, so it may share the GPU with a training run
# (T=2 threads). GATT=1: the training's GPU attention. Out: $O/{step,fd}.log, $O/summary.txt
NL=${1:-4}; N=${2:-128}; J=/opt/ozaki/jais; I=/opt/ozaki/jlora/step/gzk2512.ids
B=${B:-/opt/ozaki/src/out_jlora}; O=${O:-/opt/ozaki/jlora/fd$NL}; SEEDS=${SEEDS:-1,2,3,4,5}; EPS=${EPS:-0.001,0.002,0.004}
[ -n "$PAIRS" ] || for s in ${SEEDS//,/ }; do for e in ${EPS//,/ }; do PAIRS="$PAIRS $s:$e"; done; done
mkdir -p $O; : > $O/fd.log
r() { env OZ_NL=$NL OMP_NUM_THREADS=${T:-2} OZ_AHEAD=$([ $NL -gt 8 ] && echo 3 || echo 1) OZ_GATT=${GATT:-0} OZ_GSIGN=$O/gsign.i8 "$@" $B $J $I $N 16 32; }
r OZ_FDEPS=$EPS OZ_DSEEDS=$SEEDS > $O/step.log 2> $O/step.err; echo "step rc=$?" >> $O/fd.log
for pe in $PAIRS; do s=${pe%%:*}; e=${pe#*:}
    p=$(r OZ_DSEED=$s OZ_EPS=$e 2>/dev/null | grep -o 'step 1 loss [0-9.]*' | grep -o '[0-9.]*$')
    m=$(r OZ_DSEED=$s OZ_EPS=-$e 2>/dev/null | grep -o 'step 1 loss [0-9.]*' | grep -o '[0-9.]*$')
    echo "seed $s eps $e L+ $p L- $m" >> $O/fd.log
done
awk 'FNR == NR { if ($1 == "predicted") { split($4, a, /[(+)]/); P[$3 " " a[3]] = $8 }; next }
     $1 == "seed" { k = $2 " " $4; d = $6 - $8; e = $4; sxy[e] += d * P[k]; sxx[e] += P[k] * P[k]; n[e]++; D[k] = d }
     END { for (e in n) { s = sxy[e] / sxx[e]; w = 0
             for (k in D) { split(k, b, " "); if (b[2] == e) { r = D[k] - s * P[k]; if (r < 0) r = -r; if (r > w) w = r } }
             printf "eps %s: %d directions, slope %.4f, largest residual %.3e\n", e, n[e], s, w }
           for (k in D) printf "  seed/eps %s: measured %.6e predicted %.6e ratio %.4f\n", k, D[k], P[k], D[k] / P[k] }' $O/step.log $O/fd.log | sort > $O/summary.txt
cat $O/summary.txt
