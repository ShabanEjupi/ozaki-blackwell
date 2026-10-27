#!/bin/bash
# hol8_check.sh -- the 8-bit residue chain against the 16-bit one (OZ_HOL16=1): equal heads, and the times. ./wsl.sh 'bash bench/hol8_check.sh'
set -o pipefail
for c in "8 256" "16 256" "8 512" "16 512" "8 1024" "16 1024" "8 2048" "16 2048" "8 4096"; do
    set -- $c
    a=$(OZ_PLANES=$1 OZ_HOL=1 OZ_HOL16=1 ./out_det $2 32 0 1 | grep -E "HOL:|head") || { echo "L=$1 n=$2 16-bit FAILED"; continue; }
    b=$(OZ_PLANES=$1 OZ_HOL=1 ./out_det $2 32 0 1 | grep -E "HOL:|head|HOL: a residue") || { echo "L=$1 n=$2 8-bit FAILED"; continue; }
    ha=$(echo "$a" | grep -o "head [0-9a-f-]*"); hb=$(echo "$b" | grep -o "head [0-9a-f-]*")
    echo "L=$1 n=$2  16: $ha $(echo "$a" | grep -o "residue chain [0-9.]*") $(echo "$a" | grep -o "total [0-9.]* s")  8: $hb $(echo "$b" | grep -o "residue chain [0-9.]*") $(echo "$b" | grep -o "total [0-9.]* s") $(echo "$b" | grep -o "W [0-9]*, WQ [0-9]*")  $([ "$ha" = "$hb" ] && echo SAME || echo DIFFERENT)"
done
