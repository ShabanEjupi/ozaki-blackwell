#!/bin/bash
# jlora_steps.sh -- what jlora_run.sh starts once the machine is free (read only then, so it can change while the wait runs).
# 30-10: the 29-10 night run (4096 rows per product, 24 threads) did steps 1-2 of the 61-layer model before the laptop went
# down in Modern Standby (step-1 digests in step_4096.log, dB files in dB_4096). Now: the selftest; step 1 again with 40 rows
# and 7 threads (and the backward's read-ahead) -- its digests and dB bytes must equal the 4096 run's; the finite-difference
# check; the public corpus tokenized; then training on it: every step a new batch of 8 x 1024 tokens, 2 held-out
# sequences in the same forward, a checkpoint after every step, in runs of 2 steps -- between runs it waits for AC and yields
# to a SPEX measurement that asks for the machine (/opt/ozaki/spex.want); /opt/ozaki/jlora/train/STOP ends it.
J=/opt/ozaki/jais; D=/opt/ozaki/jlora/step; C=/opt/ozaki/jlora/corpus; T=/opt/ozaki/jlora/train; cd /opt/ozaki/src; mkdir -p $T
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
ac() { [ "$($PS -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' 2>/dev/null | tr -dc 0-9)" != 1 ]; }   # never on battery
log() { echo "$(date '+%F %T') $*" >> $D/run.log; }
if [ ! -s $T/ckpt.bin ]; then   # 30-10: with a checkpoint the training has begun -- straight to it (a restart with a new binary)
./out_jlora --selftest $J >> $D/run.log 2>&1; log "selftest rc=$?"
until ac; do sleep 60; done
mkdir -p $D/dB_40; log "step 1 OZ_NB=40 threads=7"
OZ_AHEAD=3 OZ_NB=40 OMP_NUM_THREADS=7 OZ_STEPS=1 OZ_FDEPS=0.004 OZ_DSEEDS=11,12 ./out_jlora $J $D/gzk2512.ids 1024 16 32 $D/dB_40 > $D/step_40.log 2> $D/step_40.err
log "rc=$? $(grep -h ' loss \|digest\|time' $D/step_40.log | tr '\n' ' ')"
n=0; for f in $D/dB_4096/*.f32; do cmp -s $f $D/dB_40/$(basename $f) || { n=$((n + 1)); log "DIFFER $(basename $f)"; }; done
log "step-1 dB files: $(ls $D/dB_4096 | wc -l), $n differ; digests: $(diff <(grep 'step 1 dB digest' $D/step_4096.log | cut -d';' -f1) <(grep 'step 1 dB digest' $D/step_40.log | cut -d';' -f1) > /dev/null && echo identical || echo DIFFER)"
for s in 11 12; do for e in 0.004 -0.004; do
    until ac; do sleep 60; done
    log "fd seed $s eps $e: $(OZ_AHEAD=3 OZ_DSEED=$s OZ_EPS=$e ./out_jlora $J $D/gzk2512.ids 1024 16 32 2>/dev/null | grep -o 'loss [0-9.]*')"
done; done
fi
[ -s $C/public-2910.u32 ] || python3 -I $(pwd)/llm/korpusi_prep.py tok $J $C/public-2910.jsonl $C/public-2910 1024 >> $D/run.log 2>&1
log "public corpus: $(head -c 300 $C/public-2910.json 2>/dev/null | tr '\n' ' ')"
data=$C/public-2910.u32; [ -s $data ] || data=$C/sample.u32; b=8; fails=0
while [ ! -e $T/STOP ]; do
    while [ -e /opt/ozaki/spex.want ]; do rm -f /opt/ozaki/chain.busy; sleep 60; done; touch /opt/ozaki/chain.busy
    until ac; do sleep 60; done
    OZ_AHEAD=3 OZ_NEXT=1 OZ_BATCH=$b OZ_EVAL=2 OZ_STEPS=2 OZ_LR=1e-4 OZ_CKPT=$T/ckpt.bin ./out_jlora $J $data 1024 16 32 >> $T/train.log 2>> $T/train.err
    rc=$?; log "train rc=$rc batch $b: $(grep -h 'loss\|time\|peak' $T/train.log | tail -6 | tr '\n' ' ')"
    if [ $rc != 0 ]; then fails=$((fails + 1)); [ $fails -ge 2 ] && { log "train: two failures, stop"; break; }; b=4; else fails=0; fi
done
