#!/bin/bash
# real_batch.sh -- the real-matrix study overnight (the laptop's scheduled task ozaki-flint8192 runs /opt/ozaki/flint/run.sh,
# which calls this from a frozen copy in /opt/ozaki/real/bin, so a new ./wsl.sh upload cannot change a running script).
# Pass 1: every matrix of real_list.txt on the GPU (+ FP64 errors), smallest first; pass 2: SPEX with SPEX_T s each.
cd /opt/ozaki/real/bin || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log=/opt/ozaki/real/batch.log; echo "### $(date) build $(cat .version 2>/dev/null | head -1)" >> $log
while read N P; do SPEX_T=0 bash bench/real_run.sh $N https://math.nist.gov/pub/MatrixMarket2/$P.mtx.gz >> $log 2>&1; done < bench/real_list.txt
echo "### pass 2 SPEX $(date)" >> $log
while read N P; do d=/opt/ozaki/real/$N; [ -f $d/s.int ] || continue; grep -q "SPEX Left LU" $d/spex.log 2>/dev/null && continue
    timeout ${SPEX_T:-7200} ./out_spex $d/s.int > $d/spex.log 2>&1; echo "$N SPEX rc=$? $(tail -1 $d/spex.log)" >> $log; done < bench/real_list.txt
echo "### done $(date)" >> $log
