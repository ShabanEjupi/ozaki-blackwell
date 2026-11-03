#!/bin/bash
# gemmul8_build.sh -- builds RIKEN's GEMMul8 (Ozaki-II, the authors' public code, MIT; cloned at /opt/ozaki/GEMMul8) for this
# laptop: sm_120, CUDA backend, INT8 moduli, gemm only, with nvcc 13.4 and cuBLAS 13 from the pip wheels in /opt/ozaki/cu13
# (its cuBLASLt batch modes and its cuBLAS-emulation rows need cuBLAS 13). Three traps on 03-11-2026, each a failed build:
#  - the wheels ship only versioned libraries (libcublas.so.13, ...): "-lcublas -lcudart" finds nothing in them;
#  - the wheels' lib/stubs is a link to CUDA 12.8's stubs, and nvcc puts lib/stubs FIRST: "-lcublas" took 12.8's stub,
#    which has no cublasSetEmulationStrategy / cublasSetFixedPointEmulation* ("undefined reference");
#  - the wheels' nvml.h was a link to a header never installed (cuda-nvml-dev-12-8, from NVIDIA's apt repo).
# So the cuBLAS 13 libraries are named by file (-l:libcublas.so.13), with an rpath into the wheels for the run; libcuda and
# libnvidia-ml come from the stubs at link time and from the driver (/usr/lib/wsl/lib, by soname) at run time. Only the
# static library is needed (the test links it; the shared one is GEMMul8's cuBLAS hook, not used here). nice -n 19, 2 jobs:
# it may run beside a SPEX run. Before this script, emu_compare.sh's GEMMul8 rows ended in "./test: No such file".
#   OZ_DEST=/opt/ozaki/src-emu ./wsl.sh "bash bench/gemmul8_build.sh"
set -euo pipefail
P=/opt/ozaki/cu13/lib/python3.12/site-packages/nvidia/cu13; G=/opt/ozaki/GEMMul8
[ -e /usr/local/cuda-12.8/include/nvml.h ] || apt-get install -y -q cuda-nvml-dev-12-8 > /dev/null
ln -sf /usr/local/cuda-12.8/include/nvml.h $P/include/nvml.h
M=(BACKEND=cuda OPS=gemm OZ2_BACKENDS=INT8 GPU_ARCH=120 CUDA_PATH=$P HOST_CXX=/usr/bin/g++)
cd $G && nice -n 19 make -j2 "${M[@]}" lib/libgemmul8.a
cd $G/test && nice -n 19 make -j2 "${M[@]}" LIBS="-L$P/lib -l:libcublas.so.13 -l:libcublasLt.so.13 -l:libcudart.so.13 \
    -lcuda -lnvidia-ml -ldl ../lib/libgemmul8.a -Xlinker -rpath=$P/lib"
if ldd ./test | grep 'not found'; then echo "gemmul8_build: unresolved libraries"; exit 1; fi
ldd ./test | grep -E 'cublas|cudart|libcuda|nvidia-ml' | sed 's/ (0x[0-9a-f]*)//'
echo "gemmul8_build: $G/test built ($(git -C $G rev-parse --short HEAD 2>/dev/null || echo '?'))"
