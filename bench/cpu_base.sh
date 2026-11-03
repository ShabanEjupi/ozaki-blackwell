#!/bin/bash
# cpu_base.sh -- the paper's CPU baselines measured at n = 2048 itself, five runs each, the median (cpu_med.h), with the machine
# held: until 03-11-2026 they were runs at n = 256 / 512 (in cache) scaled by n^3, and those small runs varied up to 2.8x.
# CPU triple-double (td_test.cu), MPFR 160 bits (td_test.cu), QD qd_real and MPFR 212/318/424 bits (mp_test.cu), each in a
# loop i, k, j over 24 threads; the GPU lines of the same runs beside them.
# out_td with -march=native (03-11-2026 23:4x): without it td_fma's std::fma is a call into libm and the omp simd loop over j
# never vectorized (objdump of the 17:11 binary: 3 calls to fma@plt, no ymm register; with it: no call, AVX2 throughout, the FMAs
# as vfmsub) -- most likely why the 22-10 "td AVX" 10.7 s was 29.5 s on 03-11; the rerun measures it. -ffp-contract=off keeps
# every other host rounding as it was.
#   OZ_DEST=/opt/ozaki/src-cpubase ./wsl.sh "setsid -f bash bench/cpu_base.sh > /dev/null 2>&1 < /dev/null"
# Log: /opt/ozaki/job/cpu_base.log
L=/opt/ozaki/job/cpu_base.log; cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
log() { echo "$(date '+%F %T') $*" >> $L; }
log "start $(cat .version 2>/dev/null | tr '\n' ' ')"
{ nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp,-march=native,-ffp-contract=off -o out_td bench/td_test.cu -lmpfr -lgmp -lgomp &&
  nvcc -O3 -arch=sm_120 -std=c++17 -Xcompiler -fopenmp,-march=native -o out_mp bench/mp_test.cu -lqd -lmpfr -lgmp -lgomp; } >> $L 2>&1 || { log "build FAILED"; exit 1; }
. llm/hold.sh; hold cpubase; log "machine held"
for b in out_td out_mp; do log "== $b"; OZ_CPU_N=${OZ_CPU_N:-2048} OZ_CPU_REP=${OZ_CPU_REP:-5} ./$b >> $L 2>&1; log "$b rc=$?"; done
log "done, machine given back"
