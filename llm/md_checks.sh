#!/bin/bash
# md_checks.sh -- the GPU-resident MoE backward (jais_lora.cu: moe_back_dev) against the host path on one MoE layer of the real
# model, every output of dx2 bit for bit: the training's 10 240 rows; experts cut into chunks (OZ_NB=40); groups of one
# (OZ_EG=1); outputs sent to the host fallbacks of both paths (OZ_GCAP=16: the fixup list cut short); then the grouped
# products against one call per expert (--egtest: grp_setup is shared by both group functions); last, the phases of both
# paths (OZ_PROF=1). Meant as bin_check.sh's pre-check, with the machine held:
#     OZ_DEST=/opt/ozaki/src-<label> ./wsl.sh 'BC_PRE="bash llm/md_checks.sh" setsid -f bash llm/bin_check.sh <label> ...'
set -eo pipefail
B=${B:-./out_jlora_new}; J=/opt/ozaki/jais; L=${MD_LAYER:-3}
$B --mdtest $J $L 10240
OZ_NB=40 $B --mdtest $J $L 2048
OZ_EG=1 $B --mdtest $J $L 2048
OZ_GCAP=16 $B --mdtest $J $L 512
$B --egtest $J $L 8
OZ_PROF=1 $B --mdtest $J $L 10240 2>&1 | grep -a -E '^(prof|mdtest)'
