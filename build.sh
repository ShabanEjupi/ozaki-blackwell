#!/bin/bash
# build.sh -- builds out_dev inside WSL and fails loudly: a "| grep error || true" once hid a failed build and the old
# binary ran the benchmarks (22-10-2026). Use: ./wsl.sh './build.sh && ./out_dev ...'
set -o pipefail
export PATH=/usr/local/cuda-12.8/bin:$PATH
[ -f out_ref.o ] || g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o || exit 1
rm -f out_dev
nvcc -O3 -arch=sm_120 -std=c++17 --fmad=false ${NVFLAGS:-} -Xcompiler -fopenmp,-ffp-contract=off,-march=native -o out_dev bench/solve.cu out_ref.o \
  -lquadmath -lgomp 2>&1 | grep -v "^$" | grep -iE "error|warning" ; [ -x out_dev ] || { echo "BUILD FAILED"; exit 1; }
echo "built $(cat .version 2>/dev/null | head -1)"
