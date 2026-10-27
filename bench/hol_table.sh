#!/bin/bash
# hol_table.sh -- the large-entry table of README / paper: high-order lifting (OZ_HOL=1, best of 2) against S-prime Dixon,
# GPU only (FLINT's own times come from separate runs: CPU and GPU share the laptop's power budget). Use: ./wsl.sh 'bash bench/hol_table.sh'
for c in "8 256" "16 256" "32 256" "8 512" "16 512" "32 512" "8 1024" "16 1024" "4 2048" "8 2048" "16 2048" "4 4096" "8 4096"; do
    set -- $c
    for r in 1 2; do OZ_PLANES=$1 OZ_HOL=1 ./out_det $2 32 0 1 | sed "s/^/HOL L=$1 /"; done
    OZ_PLANES=$1 ./out_det $2 32 0 1 | sed "s/^/DIX L=$1 /"
done
