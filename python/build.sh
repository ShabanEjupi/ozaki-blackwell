#!/bin/bash
# build.sh -- libozaki_blackwell.so (SASS for sm_80, 86, 89, 90, 100, 120 + PTX of sm_120 for newer GPUs; cudart, libstdc++
# and libgcc static, so it needs only the NVIDIA driver) and the wheel, on the GPU machine (WSL2 Ubuntu 24.04, nvcc 12.8):
#   OZ_DEST=/opt/ozaki/src-pkg ./wsl.sh "setsid -f bash python/build.sh > /opt/ozaki/job/pkg_build.log 2>&1 < /dev/null"
# TEST=1 then runs `python -m ozaki_blackwell` from the unpacked wheel with the machine held (llm/hold.sh): at n = 4096 the
# triple-double plan alone takes ~6 GB of VRAM, which the training must not meet. Log: /opt/ozaki/job/pkg_build.log
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/local/cuda-12.8/bin:$PATH
echo "$(date '+%F %T') start $(cat ../.version 2>/dev/null | tr '\n' ' ')"
V=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' ozaki_blackwell/__init__.py)
G=""
for a in 80 86 89 90 100 120; do G="$G -gencode arch=compute_$a,code=sm_$a"; done
G="$G -gencode arch=compute_120,code=compute_120"
# libstdc++ static (its GLIBCXX versions would pin the wheel to new systems); libgcc_s stays shared, as manylinux allows:
# a static libgcc brings _dl_find_object@GLIBC_2.35 from its unwinder. glibc_compat.c answers the 2.36/2.38 references.
time nice -n 19 nvcc -O3 -std=c++17 -shared --threads 4 -Xcompiler -fPIC,-fvisibility=hidden -cudart static \
    -Xcompiler -static-libstdc++ -Xlinker --exclude-libs,ALL -DOZB_VERSION="\"$V\"" $G -o ozaki_blackwell/libozaki_blackwell.so capi.cu glibc_compat.c
L=ozaki_blackwell/libozaki_blackwell.so
ls -l $L
readelf -d $L | grep NEEDED
objdump -T $L | grep -E 'GLIBC_2\.3[5-9]|GLIBCXX|CXXABI' && { echo "symbols newer than glibc 2.34 or from libstdc++ remain"; exit 1; }
echo "exported: $(nm -D --defined-only $L | awk '{print $3}' | tr "\n" " ")"
GLIBC=$(objdump -T $L | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1)
echo "highest glibc symbol: $GLIBC"
cp ../LICENSE ozaki_blackwell/LICENSE
python3 make_wheel.py "$V" "manylinux_$(echo ${GLIBC#GLIBC_} | tr . _)_x86_64"
if [ "${TEST:-1}" = 1 ]; then   # the test runs the unpacked wheel, not this tree
    W=$(ls -t "$PWD"/dist/*.whl | head -1)
    rm -rf /tmp/ozb_unpacked && mkdir -p /tmp/ozb_unpacked
    python3 -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$W" /tmp/ozb_unpacked
    . ../llm/hold.sh; hold pkgtest
    echo "$(date '+%F %T') machine held"
    (cd /tmp/ozb_unpacked && timeout 600 python3 -m ozaki_blackwell 4096) || echo "self-test rc=$?"
    echo "$(date '+%F %T') done, machine given back"
fi
