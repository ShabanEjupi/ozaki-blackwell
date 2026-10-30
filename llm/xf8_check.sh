#!/bin/bash
# xf8_check.sh -- run once by the Windows task ozaki-xf8 (30-10-2026), from its own tree /opt/ozaki/src-xf8 (the training's
# /opt/ozaki/src stays untouched): after the clean SPEX measurement (/opt/ozaki/spex.want gone) it builds tgemm.cu's XF8 --
# the experts' forward, fp8b_gemm's product tiled -- and checks it while the training runs (2 threads):
#  * out_bgemm: fp8b_gemm against the XF8 tiles + f8x_fixup, gate (1792 x 6144) and down (6144 x 1792) of L3 e0 and L40 e100,
#    40 / 432 / 4096 tokens (432 ~ a training step's tokens an expert): 0 outputs differ at p 8 and at p 24;
#  * --dgtest (the backward's FP8 tiles, unchanged): ok;
#  * llm/jlora_test.sh (4 layers: L3 is MoE) with the training's binary and with the new one: the same digests and losses.
# All of it: the new binary becomes /opt/ozaki/src/out_jlora (cp, then one rename: the next 2-step run takes it; the old one
# stays as out_jlora.<its version>). Log: /opt/ozaki/job/xf8.log
S=/opt/ozaki/src-xf8; R=/opt/ozaki/src; J=/opt/ozaki/jais; L=/opt/ozaki/job/xf8.log; cd $S || exit 1
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' '), waiting for the SPEX run"
while [ -e /opt/ozaki/spex.want ]; do sleep 60; done
log "SPEX done: $(tail -2 /opt/ozaki/real/batch.log | tr '\n' ' ')"
export PATH=/usr/local/cuda-12.8/bin:$PATH OMP_NUM_THREADS=2
B="nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp"
{ $B -o out_bgemm llm/bgemm.cu -lgomp && $B -o out_jlora_new llm/jais_lora.cu -lgomp; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
log "built"
bad=0
for e in 3.mlp.experts.0 40.mlp.experts.100; do for n in 40 432 4096; do
    for m in "gate_proj 1792 6144" "down_proj 6144 1792"; do set -- $m
        ./out_bgemm $J model.layers.$e.$1 $2 $3 $n > xf8.out 2>&1
        r=$(grep '^FP8 p' xf8.out); log "L$e $1 N=$n: $(echo "$r" | tr '\n' ' ')"
        [ "$(echo "$r" | grep -c ': 0 of ')" = 2 ] || { bad=$((bad + 1)); log "  DIFFER or missing: $(tail -3 xf8.out | tr '\n' ' ')"; }
    done
done; done
./out_jlora_new --dgtest $J 3 0 341 > xf8.out 2>&1; log "$(grep '^dgtest' xf8.out | tr '\n' ' ')"; grep -q '^dgtest: ok' xf8.out || bad=$((bad + 1))
# jlora_test.sh runs ./out_jlora_t: first the training's binary, then the new one; digests and losses (not times) must agree
keep() { sed 's/ sync=.*//' "$1"; }
cp $R/out_jlora out_jlora_t && bash llm/jlora_test.sh > xf8.old 2>&1
cp out_jlora_new out_jlora_t && bash llm/jlora_test.sh > xf8.new 2>&1
log "4 layers, training's binary: $(keep xf8.old | tr '\n' ' ')"
log "4 layers, XF8: $(keep xf8.new | tr '\n' ' ')"
if ! diff <(keep xf8.old) <(keep xf8.new) > /dev/null || ! grep -q 'resume: same bytes' xf8.new || grep -q 'rc=[1-9]' xf8.new; then bad=$((bad + 1)); log "4 layers DIFFER"; fi
if [ $bad != 0 ]; then log "NOT deployed: $bad checks failed"; exit 1; fi
cp $R/out_jlora $R/out_jlora.pre-xf8   # (/opt/ozaki/src/.version is the last wsl.sh tree, not always the binary's)
cp out_jlora_new $R/out_jlora.xf8 && mv -f $R/out_jlora.xf8 $R/out_jlora && log "deployed: $R/out_jlora = $(cat .version | tr '\n' ' ') (old: out_jlora.pre-xf8)"
