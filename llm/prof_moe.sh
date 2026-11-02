#!/bin/bash
# prof_moe.sh -- a clean profile of the training step's MoE work (02-11-2026): the 31-10 profile shared the GPU with the
# training and every phase that waits on the GPU (copies, products) came out several times too long. Waits until
# night_checks.sh has finished its GPU checks, takes spex.want (the training yields between runs; SPEX at 22:00 waits for
# it in turn), then one step of the first 8 layers (3 dense + 5 MoE) at the training's size (8 x 1024 tokens + 2 held out,
# 24 threads, OZ_GATT=1, OZ_PROF=1) with the training's binary, twice (the second with a warm page cache is the one to
# read; the prof lines go to stderr). Out: /opt/ozaki/jlora/prof8/{1,2}.{log,err}, the phases summed in prof8/summary.txt
J=/opt/ozaki/jais; C=/opt/ozaki/jlora/corpus/public-2910.u32; O=/opt/ozaki/jlora/prof8; L=/opt/ozaki/job/night.log
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
ac() { [ "$($PS -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d "\r ")" != 1 ]; }
log() { echo "$(date '+%F %T') prof: $*" >> $L; }
until grep -q "GPU checks done" $L 2>/dev/null; do sleep 60; done
while [ -e /opt/ozaki/spex.want ]; do sleep 60; done; echo "prof <$$>" > /opt/ozaki/spex.want
while ! ac || pgrep -x out_jlora > /dev/null || pgrep -x out_spex > /dev/null || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done
mkdir -p $O; cd /opt/ozaki/src; log "start"
for i in 1 2; do
    OZ_PROF=1 OZ_NL=8 OZ_AHEAD=3 OZ_GATT=1 OZ_NEXT=1 OZ_BATCH=8 OZ_EVAL=2 OMP_NUM_THREADS=24 ./out_jlora $J $C 1024 16 32 > $O/$i.log 2> $O/$i.err
    log "run $i rc=$? $(grep -a -o 'step 1 time.*' $O/$i.log)"
done
# the summary: every phase summed over the 8 layers, seconds, largest first
grep -a 'prof model.layers' $O/2.err | sed 's/^ *prof model\.layers\.\([0-9]*\)\.: //' > $O/phases.txt
awk -F'\n' 'BEGIN { } { n = split($0, w, " "); key = ""; for (i = 1; i <= n; i++) { if (w[i] ~ /^[0-9]+\.[0-9]+$/) { s[key] += w[i]; key = "" } else key = (key == "" ? w[i] : key " " w[i]) } }
    END { for (k in s) printf "%8.2f  %s\n", s[k], k }' $O/phases.txt | sort -rn > $O/summary.txt
grep -a -E '^pass|step 1 time' $O/2.log >> $O/summary.txt
rm -f /opt/ozaki/spex.want; log "done; spex.want removed; $(head -5 $O/summary.txt | tr '\n' ';')"
