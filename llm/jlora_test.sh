#!/bin/bash
# jlora_test.sh -- the 4-layer checks of the training features (2 threads: a run may be going): the held-out rows and the
# backward's read-ahead change no digest; a run resumed from its checkpoint ends on the same bytes as one run.
J=/opt/ozaki/jais; C=/opt/ozaki/jlora/corpus/sample.u32; T=/opt/ozaki/jlora/t; mkdir -p $T; rm -f $T/*.bin*
x() { local n=$1; shift; env OZ_NL=4 OMP_NUM_THREADS=2 OZ_BATCH=2 "$@" ./out_jlora_t $J $C 128 16 32 > $T/$n.log 2> $T/$n.err; echo "$n rc=$? $(grep -ho 'dB digest [0-9a-f]*\|held-out loss [0-9.]*\|step [0-9] loss [0-9.]*' $T/$n.log | tr '\n' ' ') sync=$(grep -c 'read synchronously' $T/$n.err) $(grep -h 'peak\|time' $T/$n.log | tail -2 | tr '\n' ' ')"; }
x plain OZ_AHEAD=1
x eval OZ_AHEAD=1 OZ_EVAL=1
x noring OZ_EVAL=1
x two OZ_AHEAD=1 OZ_EVAL=1 OZ_NEXT=1 OZ_STEPS=2 OZ_CKPT=$T/two.bin
x one1 OZ_AHEAD=1 OZ_EVAL=1 OZ_NEXT=1 OZ_STEPS=1 OZ_CKPT=$T/one.bin
x one2 OZ_AHEAD=1 OZ_EVAL=1 OZ_NEXT=1 OZ_STEPS=1 OZ_CKPT=$T/one.bin
cmp $T/two.bin $T/one.bin && echo "resume: same bytes"
