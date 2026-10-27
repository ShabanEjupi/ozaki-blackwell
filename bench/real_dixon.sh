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
    OZ_HADAMARD=1 OZ_FILE=$D/s.oz OZ_PLANES=$L ./out_det $np 32 0 1 > $D/gpu_dixon.log 2>&1
    echo "  GPU rc=$? $(grep -E 'file:|head|FILE|failed|RATREC|LIFT' $D/gpu_dixon.log | tr '\n' ' ' | cut -c1-900)"
    python3 -I bench/fp64_err.py $D $n0 $np
done
