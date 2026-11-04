#!/bin/bash
# build_solvers.sh [solve] [soltd] -- builds the Ax=b solvers (out_solve: double-double, with the __float128 and CPU dd LUs;
# out_soltd: triple-double, with the CPU td and MPFR LUs) with the one set of flags every job uses; default: both. nice'd, so
# it can run beside the training. One copy on purpose (04-11-2026): six job scripts carried their own copy of these lines,
# and on 03-11 a copy without -march=native left the CPU triple-double baseline unvectorized (std::fma a libm call), so its
# 293x measured the build, not the GPU.
cd "$(dirname "$0")/.." || exit 1
export PATH=/usr/local/cuda-12.8/bin:$PATH
F="-O3 -arch=sm_120 -std=c++17 --fmad=false -Xcompiler -fopenmp,-ffp-contract=off,-march=native"
for t in ${@:-solve soltd}; do
    case $t in
        solve) nice -n 19 g++ -O3 -fopenmp -c bench/solve_ref.cpp -o out_ref.o &&
               nice -n 19 nvcc $F -o out_solve bench/solve.cu out_ref.o -lquadmath -lgomp || exit 1 ;;
        soltd) nice -n 19 nvcc $F -o out_soltd bench/solve_td.cu -lmpfr -lgmp -lgomp || exit 1 ;;
        *) echo "build_solvers.sh: unknown target '$t' (solve, soltd)" >&2; exit 2 ;;
    esac
done
