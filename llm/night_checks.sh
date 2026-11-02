#!/bin/bash
# night_checks.sh -- what the papers still need from an idle laptop (02-11-2026). The LoRA training yields between its runs
# while /opt/ozaki/spex.want exists; this waits for the running one to end, then:
#  1. step 1 of the 61-layer model on 1024 tokens of ActID 2512 with the training's binary and settings (OZ_GATT=1), OZ_NB
#     4096 on 24 threads against OZ_NB 40 on 7: the 122 dB files byte for byte and the digests. The 29/30-10 run of this
#     test had 3 uncertified outputs (before the host superaccumulator); now every output is exact;
#  2. finite differences through all 61 layers (fd_check.sh 61 1024 with the binary of $NEW, which adds the direction
#     sign(g_0)): seeded +-1 directions at e = 0.004 are below the forward's rounding noise there (30-10), sign(g_0) is not;
#  3. SPEX Left LU on the three systems it did not finish in 2 h (fidapm07, fidap026, fidap014): one a night from 22:00,
#     8 h at most, the CPU alone (spex.want only for those hours: the days stay with the training and the voice job).
# The home crawler stops for the measurements and starts again after. Log /opt/ozaki/job/night.log
NEW=${NEW:-/opt/ozaki/src-fd}; J=/opt/ozaki/jais; I=/opt/ozaki/jlora/step/gzk2512.ids; L=/opt/ozaki/job/night.log
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
ac() { [ "$($PS -NoProfile -Command "(Get-CimInstance Win32_Battery).BatteryStatus" 2>/dev/null | tr -d "\r ")" != 1 ]; }
log() { echo "$(date '+%F %T') $*" >> $L; }
crawler() {   # stop | start the home crawler (its process by pid: a pattern given to pkill -f also matches this shell)
    if [ $1 = stop ]; then $PS -NoProfile -Command "Stop-ScheduledTask -TaskName korpusi-shtepi" > /dev/null 2>&1
        for p in $(pgrep -f '[h]arvest.py --sources sources-shtepi'); do kill $p; done; sleep 30
    else $PS -NoProfile -Command "Start-ScheduledTask -TaskName korpusi-shtepi" > /dev/null 2>&1; fi; }
idle() { while ! ac || pgrep -x out_jlora > /dev/null || pgrep -x out_spex > /dev/null || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done; }
(while true; do echo "$(date +%T) up $(uptime -s) vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> /opt/ozaki/job/heartbeat; sleep 60; done) &
while [ -e /opt/ozaki/spex.want ]; do sleep 60; done   # someone else measures
touch /opt/ozaki/spex.want; log "spex.want set; waiting for the training run, AC"; idle; crawler stop
log "idle: $(uptime | sed 's/.*load/load/'), GPU $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)"

# 1. reproducibility, the training's binary
D=/opt/ozaki/jlora/repro; rm -rf $D; mkdir -p $D/dB_4096 $D/dB_40; cd /opt/ozaki/src
for c in "4096 24" "40 7"; do set -- $c
    OZ_AHEAD=3 OZ_GATT=1 OZ_NB=$1 OMP_NUM_THREADS=$2 OZ_STEPS=1 ./out_jlora $J $I 1024 16 32 $D/dB_$1 > $D/step_$1.log 2> $D/step_$1.err
    log "step 1, OZ_NB=$1, $2 threads: rc=$? $(grep -a -ho 'step 1 dB digest.*' $D/step_$1.log) $(grep -a -ho 'forward [0-9.]* s' $D/step_$1.log)"
done
n=0; d=0; for f in $D/dB_4096/*; do n=$((n+1)); cmp -s $f $D/dB_40/$(basename $f) || d=$((d+1)); done
log "reproducibility: $n dB files, $d differ; distinct digest lines: $(grep -a -ho 'dB digest [0-9a-f]*, dA digest [0-9a-f]*, g_0 digest [0-9a-f]*' $D/step_4096.log $D/step_40.log | sort -u | wc -l) (1 = identical)"

# 2. finite differences through 61 layers (the new binary; its step's digests must be the training binary's)
O=/opt/ozaki/jlora/fd61 B=$NEW/out_jlora T=24 SEEDS=0,13,14 EPS=2.5e-05,5e-05,0.0001,0.0002,0.004 GATT=1 \
    PAIRS="0:2.5e-05 0:5e-05 0:0.0001 0:0.0002 13:0.004 14:0.004" bash $NEW/llm/fd_check.sh 61 1024 > /dev/null 2>&1
log "fd 61 layers: $(grep -a -ho 'step 1 dB digest.*' /opt/ozaki/jlora/fd61/step.log)"; log "$(grep -a -h 'sign direction' /opt/ozaki/jlora/fd61/step.log)"
while read -r l; do log "fd61 $l"; done < /opt/ozaki/jlora/fd61/summary.txt
rm -f /opt/ozaki/spex.want; crawler start; log "GPU checks done; spex.want removed"

# 3. SPEX, CPU alone, one system a night from 22:00 (8 h at most)
for N in fidapm07 fidap026 fidap014; do
    s=$(( $(date -d 22:00 +%s) - $(date +%s) )); [ $s -lt 0 ] && s=$(( s + 86400 )); log "SPEX $N waits $s s for 22:00"; sleep $s
    while [ -e /opt/ozaki/spex.want ]; do sleep 60; done; touch /opt/ozaki/spex.want; idle; crawler stop
    d=/opt/ozaki/real/$N; cd /opt/ozaki/real/bin; echo "### pass 4 SPEX $N (8 h, idle machine), $(date)" >> /opt/ozaki/real/batch.log
    log "SPEX $N starts"; timeout 28800 ./out_spex $d/s.int > $d/spex_idle.log 2>&1; r=$?
    echo "$N SPEX idle rc=$r $(tail -1 $d/spex_idle.log)" >> /opt/ozaki/real/batch.log; log "SPEX $N rc=$r $(tail -1 $d/spex_idle.log | cut -c1-160)"
    rm -f /opt/ozaki/spex.want; crawler start
done
echo "### pass 4 done $(date)" >> /opt/ozaki/real/batch.log; log "done"
kill %1
