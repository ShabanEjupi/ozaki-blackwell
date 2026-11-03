#!/bin/bash
# speed_all.sh -- the paper's precision table (FP64, double-double, triple-double, beyond) measured again with the machine
# held, every line logged: the README/paper speeds date from 15-10 and 22-10 and the code has changed since (03-11: dd32 at
# n = 4096 ran at 3.38 TFLOPS beside the training, the paper says 1.76). Accuracy comes from the same runs (exact references).
#   OZ_DEST=/opt/ozaki/src-speed ./wsl.sh "setsid -f bash bench/speed_all.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/speed.log
L=/opt/ozaki/job/speed.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
{ nvcc -O3 -arch=sm_120 -std=c++17 -o out_test bench/ozaki_test.cu &&
  g++ -O3 -c bench/ref_quad.cpp -o out_rq.o && nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp -o out_dd bench/dd_test.cu out_rq.o -lquadmath -lmpfr -lgmp -lgomp &&
  nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp -o out_td bench/td_test.cu -lmpfr -lgmp -lgomp &&
  nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp,-march=native -o out_mp bench/mp_test.cu -lqd -lmpfr -lgmp -lgomp; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold speed; log "machine held"
for b in out_test out_dd out_td out_mp; do log "== $b"; ./$b >> $L 2>&1; log "$b rc=$?"; done
log "done, machine given back"
