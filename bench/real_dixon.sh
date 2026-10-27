#!/bin/bash
# real_dixon.sh NAME... -- real systems above the HOL buffers (n > ~6000 in 12 GB): the exact solution by Dixon lifting with
# S primes per step (det.cu OZ_FILE without OZ_HOL: every digit of every entry kept on the host, the matrix transposed so
# that the column lifting gives the row-form y of the file), then the FP64 errors. Run from a frozen copy (the scheduled task
# ozaki-job): bench/real_run.sh must have converted NAME first (/opt/ozaki/real/NAME/s.oz).
cd $(dirname $0)/..
for N in "$@"; do
    D=/opt/ozaki/real/$N
    read np L n0 < <(python3 -c "import struct;print(*struct.unpack('3i',open('$D/s.oz','rb').read()[4:16]))")
    echo "$N: $(cat $D/conv.log)"
    t0=$(date +%s.%N)
    rm -f $D/t_sol   # the moment the certified solution is printed (the det/d step that follows is not part of the solve)
    OZ_NODET=1 OZ_HADAMARD=1 OZ_FILE=$D/s.oz OZ_PLANES=$L timeout 3600 stdbuf -oL ./out_det $np 32 0 1 2>&1 |
        while IFS= read -r l; do echo "$l"; case "$l" in *"solution x = num"*) date +%s.%N > $D/t_sol;; esac; done > $D/gpu_dixon.log
    echo "  GPU solve $( [ -f $D/t_sol ] && echo "$(cat $D/t_sol) - $t0" | bc ) s (wall to the certified solution) $(grep -E 'file:|head|FILE|failed|RATREC|LIFT' $D/gpu_dixon.log | tr '\n' ' ' | cut -c1-900)"
    python3 -I bench/fp64_err.py $D $n0 $np
done
