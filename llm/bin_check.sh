#!/bin/bash
# bin_check.sh <label> -- a new training binary checked against the running one before it takes the training's place. Its
# tree sits in /opt/ozaki/src-<label>:
#     OZ_DEST=/opt/ozaki/src-<label> ./wsl.sh "setsid -f bash llm/bin_check.sh <label> > /dev/null 2>&1 < /dev/null"
# (the training's /opt/ozaki/src stays untouched). Builds first, then takes the machine the way prof_moe.sh does (spex.want:
# the training yields between its 2-step runs, night_checks.sh's SPEX waits in turn) and waits for the running training run
# to end, so that the timed steps have the GPU to themselves. Then:
#  * llm/jlora_test.sh with the training's binary and with the new one: the same digests and losses;
#  * the new one with OZ_NB=40 (an expert's rows cut into several products): the same digests as the eval run;
#  * one 4-layer step as the training runs it (8 x 1024 tokens + 2 held out, all threads) with OZ_PROF=1, old binary and
#    new: the same dB/dA/g_0 digests (and some at all); times and phases into the log.
# Every check passes: the new binary becomes /opt/ozaki/src/out_jlora (deploy_bin.sh: cp, then one rename -- the next run
# takes it; the old one stays as out_jlora.pre-<label>). Log: /opt/ozaki/job/<label>.log
# Used: 31-10 rows (01bf4fa: the experts' rows gathered on the GPU), 03-11 notrim (aab266d: glibc's heap never trimmed).
LB=${1:?label}; S=/opt/ozaki/src-$LB; R=/opt/ozaki/src; J=/opt/ozaki/jais; C=/opt/ozaki/jlora/corpus; L=/opt/ozaki/job/$LB.log
cd $S || exit 1
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
ac() { [ "$($PS -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d "\r ")" != 1 ]; }
log() { echo "$(date '+%F %T') $*" >> $L; }
export PATH=/usr/local/cuda-12.8/bin:$PATH
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jlora_new llm/jais_lora.cu -lgomp >> $L 2>&1 || { log "build FAILED"; exit 1; }
log "built; waiting for the machine (spex.want, then the end of the running training run, AC)"
while [ -e /opt/ozaki/spex.want ]; do sleep 60; done; echo "$LB <$$>" > /opt/ozaki/spex.want; trap 'rm -f /opt/ozaki/spex.want' EXIT
while ! ac || pgrep -x out_jlora > /dev/null || pgrep -x out_spex > /dev/null || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done
log "machine held"
bad=0
# jlora_test.sh runs ./out_jlora_t: first the training's binary, then the new one; digests and losses (not times) must agree
keep() { sed 's/ sync=.*//' "$1"; }
cp $R/out_jlora out_jlora_t && bash llm/jlora_test.sh > test.old 2>&1
cp out_jlora_new out_jlora_t && bash llm/jlora_test.sh > test.new 2>&1
log "4 layers, training's binary: $(keep test.old | tr '\n' ' ')"
log "4 layers, new binary: $(keep test.new | tr '\n' ' ')"
if ! diff <(keep test.old) <(keep test.new) > /dev/null || ! grep -aq 'resume: same bytes' test.new || grep -aq 'rc=[1-9]' test.new; then
    bad=$((bad + 1)); log "4 layers DIFFER"; fi
T=/opt/ozaki/jlora/t
dig() { grep -aho 'dB digest [0-9a-f]*\|held-out loss [0-9.]*' "$1" | tr '\n' ' '; }
OZ_NL=4 OMP_NUM_THREADS=2 OZ_BATCH=2 OZ_AHEAD=1 OZ_EVAL=1 OZ_NB=40 ./out_jlora_new $J $C/sample.u32 128 16 32 > $T/nb40.log 2> $T/nb40.err
log "OZ_NB=40: rc=$? $(dig $T/nb40.log) (eval run: $(dig $T/eval.log))"
[ -n "$(dig $T/nb40.log)" ] && [ "$(dig $T/nb40.log)" = "$(dig $T/eval.log)" ] || { bad=$((bad + 1)); log "OZ_NB=40 DIFFERS"; }
# the timed step: the training's sizes and threads, from the training's data, once per binary
data=$C/public-2910.u32; [ -s $data ] || data=$C/sample.u32
for b in old new; do
    x=$R/out_jlora; [ $b = new ] && x=./out_jlora_new
    OZ_NL=4 OZ_PROF=1 OZ_AHEAD=3 OZ_GATT=1 OZ_BATCH=8 OZ_EVAL=2 OZ_STEPS=1 $x $J $data 1024 16 32 > $T/prof_$b.log 2> $T/prof_$b.err
    rc=$?; [ $rc = 0 ] || { bad=$((bad + 1)); log "timed step, $b binary: rc=$rc"; }
    log "step, $b binary: rc=$rc $(grep -ah 'time\|dB digest' $T/prof_$b.log | tr '\n' ' ')"
    grep -a '^prof ' $T/prof_$b.err | while read -r l; do log "  $b $l"; done
done
d() { grep -ah 'dB digest' $T/prof_$1.log | cut -d';' -f1; }
[ -n "$(d new)" ] && [ "$(d old)" = "$(d new)" ] || { bad=$((bad + 1)); log "timed step: digests DIFFER or missing"; }
if [ $bad != 0 ]; then log "NOT deployed: $bad checks failed"; exit 1; fi
. llm/deploy_bin.sh; deploy_bin out_jlora_new $LB
