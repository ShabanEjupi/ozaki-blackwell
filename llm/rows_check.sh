#!/bin/bash
# rows_check.sh -- once, from its own tree /opt/ozaki/src-rows (the training's /opt/ozaki/src stays untouched), after the
# clean SPEX measurement: the experts' products on rows gathered on the GPU (rows_up + bmm with rdev: a layer's input
# quantized once, no host gather and no upload per expert, in the forward and in the MoE backward) --
#  * llm/jlora_test.sh with the training's binary and with the new one: the same digests and losses;
#  * the new one with OZ_NB=40 (an expert's rows cut into several products: the gathered rows at n0 > 0): the same digests;
#  * one 4-layer step as the training runs it (8 x 1024 tokens + 2 held out, all threads) with OZ_PROF=1, old binary and
#    new, the machine held for them (spex.want: the training yields between its 2-step runs): the phases into the log.
# Every check passes: the new binary becomes /opt/ozaki/src/out_jlora (cp, then one rename: the next run takes it; the old
# one stays as out_jlora.pre-rows). Log: /opt/ozaki/job/rows.log
S=/opt/ozaki/src-rows; R=/opt/ozaki/src; J=/opt/ozaki/jais; C=/opt/ozaki/jlora/corpus; L=/opt/ozaki/job/rows.log; cd $S || exit 1
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' '), waiting for the SPEX run"
while [ -e /opt/ozaki/spex.want ] || pgrep -x out_spex > /dev/null; do sleep 5; done
log "SPEX done: $(tail -2 /opt/ozaki/real/batch.log | tr '\n' ' ')"
touch /opt/ozaki/spex.want; trap 'rm -f /opt/ozaki/spex.want' EXIT
[ -e /opt/ozaki/chain.busy ] && log "a training run had begun: until it ends the timed steps share the GPU"
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jlora_new llm/jais_lora.cu -lgomp >> $L 2>&1 || { log "build FAILED"; exit 1; }
log "built"
bad=0
# jlora_test.sh runs ./out_jlora_t: first the training's binary, then the new one; digests and losses (not times) must agree
keep() { sed 's/ sync=.*//' "$1"; }
cp $R/out_jlora out_jlora_t && bash llm/jlora_test.sh > rows.old 2>&1
cp out_jlora_new out_jlora_t && bash llm/jlora_test.sh > rows.new 2>&1
log "4 layers, training's binary: $(keep rows.old | tr '\n' ' ')"
log "4 layers, gathered rows: $(keep rows.new | tr '\n' ' ')"
if ! diff <(keep rows.old) <(keep rows.new) > /dev/null || ! grep -q 'resume: same bytes' rows.new || grep -q 'rc=[1-9]' rows.new; then bad=$((bad + 1)); log "4 layers DIFFER"; fi
T=/opt/ozaki/jlora/t
dig() { grep -ho 'dB digest [0-9a-f]*\|held-out loss [0-9.]*' "$1" | tr '\n' ' '; }
OZ_NL=4 OMP_NUM_THREADS=2 OZ_BATCH=2 OZ_AHEAD=1 OZ_EVAL=1 OZ_NB=40 ./out_jlora_new $J $C/sample.u32 128 16 32 > $T/nb40.log 2> $T/nb40.err
log "OZ_NB=40: rc=$? $(dig $T/nb40.log) (eval run: $(dig $T/eval.log))"
[ -n "$(dig $T/nb40.log)" ] && [ "$(dig $T/nb40.log)" = "$(dig $T/eval.log)" ] || { bad=$((bad + 1)); log "OZ_NB=40 DIFFERS"; }
# the timed step: the training's sizes and threads, from the training's data, once per binary
data=$C/public-2910.u32; [ -s $data ] || data=$C/sample.u32
for b in old new; do
    x=$R/out_jlora; [ $b = new ] && x=./out_jlora_new
    OZ_NL=4 OZ_PROF=1 OZ_AHEAD=3 OZ_GATT=1 OZ_BATCH=8 OZ_EVAL=2 OZ_STEPS=1 $x $J $data 1024 16 32 > $T/prof_$b.log 2> $T/prof_$b.err
    log "step, $b binary: rc=$? $(grep -h 'time\|dB digest' $T/prof_$b.log | tr '\n' ' ')"
    grep '^prof ' $T/prof_$b.err | while read -r l; do log "  $b $l"; done
done
[ "$(grep -h 'dB digest' $T/prof_old.log | cut -d';' -f1)" = "$(grep -h 'dB digest' $T/prof_new.log | cut -d';' -f1)" ] || { bad=$((bad + 1)); log "timed step: digests DIFFER"; }
if [ $bad != 0 ]; then log "NOT deployed: $bad checks failed"; exit 1; fi
cp $R/out_jlora $R/out_jlora.pre-rows
cp out_jlora_new $R/out_jlora.rows && mv -f $R/out_jlora.rows $R/out_jlora && log "deployed: $R/out_jlora = $(cat .version | tr '\n' ' ') (old: out_jlora.pre-rows)"
