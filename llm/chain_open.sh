#!/bin/bash
# chain_open.sh -- the fully open models after Nemotron 3 Ultra, unattended (Windows task ozaki-chain, 27-10-2026):
# 1. when out_ultra has finished and OLMo-2-0325-32B-Instruct is downloaded: exact vs engine s=1/s=4 on the three prompts,
#    then every generated text traced into the model's whole training data (infini-gram, network only);
# 2. when SPEX pass 3 is done and K2-Think-V2 is downloaded: the same on K2-Think-V2.
# /opt/ozaki/chain.busy exists while phase 1 computes: the SPEX job (/opt/ozaki/job/run.sh) waits for it, so its timings
# are taken on an idle CPU. Frozen copy in /opt/ozaki/chain (binary + scripts); log /opt/ozaki/chain/chain.log.
C=/opt/ozaki/chain; B=/opt/ozaki/chain.busy; touch $B
log() { echo "$(date '+%F %T') $*" >> $C/chain.log; }
run() {   # <model dir> <name>
    python3 -I $C/ultra_index.py $1 > $1/index.txt
    python3 -I $C/ultra_tok.py enc $1 $C/ultra_prompts.txt > $C/$2.ids
    log "$2 start ($(wc -l < $1/index.txt) tensors)"
    $C/out_dense $1 $C/$2.ids 24 0,1,4 > $C/$2.log 2> $C/$2.err; log "$2 rc=$?"
    python3 -I $C/ultra_tok.py dec $1 < $C/$2.log > $C/$2.txt
}
while pgrep -x out_ultra > /dev/null; do sleep 60; done
until grep -q " done:" /opt/ozaki/olmo2/fetch.log 2>/dev/null; do sleep 60; done
run /opt/ozaki/olmo2 olmo
rm -f $B
python3 -I $C/olmo_trace.py --seqs $C/olmo.txt 3 > $C/olmo_trace.txt 2>&1; log "trace olmo rc=$?"
until grep -q "pass 3 done" /opt/ozaki/real/batch.log && grep -q " done:" /opt/ozaki/k2/fetch.log 2>/dev/null; do sleep 300; done
run /opt/ozaki/k2 k2
log "chain done"
