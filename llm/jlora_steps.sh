#!/bin/bash
# jlora_steps.sh -- what jlora_run.sh starts once the machine is free (read only then, so it can change while the wait runs):
# the FP64 selftest of the backward; 3 AdamW steps (lr 1e-4) of the 61-layer model on 1024 tokens, twice -- 4096 rows per
# product and 24 threads, then 40 rows and 7 threads: every step's dB, dA and g_0 digests must agree and the step-1 dB
# files must be the same bytes; then the finite-difference check, two directions at e = 0.004.
J=/opt/ozaki/jais; D=/opt/ozaki/jlora/step; cd /opt/ozaki/src
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
ac() { [ "$($PS -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' 2>/dev/null | tr -dc 0-9)" != 1 ]; }   # never on battery
log() { echo "$(date '+%F %T') $*" >> $D/run.log; }
./out_jlora --selftest $J >> $D/run.log 2>&1; log "selftest rc=$?"
for nb in 4096 40; do
    th=$([ $nb = 4096 ] && echo 24 || echo 7); mkdir -p $D/dB_$nb
    until ac; do sleep 60; done
    log "3 steps OZ_NB=$nb threads=$th"
    OZ_AHEAD=3 OZ_NB=$nb OMP_NUM_THREADS=$th OZ_STEPS=3 OZ_LR=1e-4 OZ_FDEPS=0.004 OZ_DSEEDS=11,12 ./out_jlora $J $D/gzk2512.ids 1024 16 32 $D/dB_$nb > $D/step_$nb.log 2> $D/step_$nb.err
    log "rc=$? $(grep -h ' loss \|loss by\|digest' $D/step_$nb.log | tr '\n' ' ')"
done
n=0; for f in $D/dB_4096/*.f32; do cmp -s $f $D/dB_40/$(basename $f) || { n=$((n + 1)); log "DIFFER $(basename $f)"; }; done
log "step-1 dB files: $(ls $D/dB_4096 | wc -l), $n differ; digests: $(diff <(grep digest $D/step_4096.log) <(grep digest $D/step_40.log) > /dev/null && echo identical || echo DIFFER)"
for s in 11 12; do for e in 0.004 -0.004; do
    until ac; do sleep 60; done
    log "fd seed $s eps $e: $(OZ_AHEAD=3 OZ_DSEED=$s OZ_EPS=$e ./out_jlora $J $D/gzk2512.ids 1024 16 32 2>/dev/null | grep -o 'loss [0-9.]*')"
done; done
# the public corpus (Kosovo institutions, crawled 29-10): 3 steps on a batch of 4 sequences x 1024 tokens of the shuffled
# sample (4096 tokens per step: 4x the tokens for the same 750 GB of reads), then the whole public part tokenized
C=/opt/ozaki/jlora/corpus
until ac; do sleep 60; done
log "corpus: 3 steps, batch 4 x 1024 (sample.u32 from 100)"
OZ_AHEAD=3 OZ_BATCH=4 OZ_FIRST=100 OZ_STEPS=3 OZ_LR=1e-4 ./out_jlora $J $C/sample.u32 1024 16 32 > $D/corpus.log 2> $D/corpus.err
log "rc=$? $(grep -h ' loss \|loss by\|digest\|time' $D/corpus.log | tr '\n' ' ')"
[ -s $C/public-2910.u32 ] || python3 -I $(pwd)/llm/korpusi_prep.py tok $J $C/public-2910.jsonl $C/public-2910 1024 >> $D/run.log 2>&1
log "public corpus: $(head -c 300 $C/public-2910.json 2>/dev/null | tr '\n' ' ')"
