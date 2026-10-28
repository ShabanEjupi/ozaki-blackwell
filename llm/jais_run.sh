#!/bin/bash
# jais_run.sh -- Jais-3-375B-A23B-FP8 exact vs engines s=1/s=4 on the three prompts, unattended (Windows task ozaki-jais-run,
# 28-10-2026), as soon as its download is complete (ozaki-jais-fetch, which finishes after Ultra is deleted and the vhdx
# compacted). First the CPU check of one FP8 expert map of layer 3 (llm/jais_check.py, gate and down) -- a failure stops
# the run. /opt/ozaki/chain.busy while it computes, so the evening SPEX job waits (and it waits for a running SPEX).
# Log /opt/ozaki/chain/chain.log.
C=/opt/ozaki/chain; J=/opt/ozaki/jais
log() { echo "$(date '+%F %T') $*" >> $C/chain.log; }
until grep -q " done:" $J/fetch.log 2>/dev/null; do sleep 300; done
while [ -e /opt/ozaki/chain.busy ] || pgrep -x out_spex > /dev/null; do sleep 60; done   # never under SPEX timings
touch /opt/ozaki/chain.busy
python3 -I $C/ultra_index.py $J > $J/index.txt
for t in gate_proj:1792:6144 down_proj:6144:1792; do IFS=: read n m k <<< "$t"; d=$C/jcheck_$n; mkdir -p $d
    $C/out_jais --check $J model.layers.3.mlp.experts.0.$n $m $k $d 2>> $C/jais.err
    r=$(python3 -I $C/jais_check.py $J $d); rc=$?; log "jais check: $r"
    [ $rc = 0 ] || { log "jais check failed, no run"; rm -f /opt/ozaki/chain.busy; exit 1; }
done
python3 -I $C/ultra_tok.py enc $J $C/ultra_prompts.txt > $C/jais.ids
log "jais start ($(wc -l < $J/index.txt) tensors)"
OZ_AHEAD=3 $C/out_jais $J $C/jais.ids 24 0,1,4 > $C/jais.log 2>> $C/jais.err; log "jais rc=$?"
python3 -I $C/ultra_tok.py dec $J < $C/jais.log > $C/jais.txt
rm -f /opt/ozaki/chain.busy
