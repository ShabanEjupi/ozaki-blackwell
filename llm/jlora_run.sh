#!/bin/bash
# jlora_run.sh -- the full LoRA step of Jais-3-375B-A23B-FP8 (61 layers, 1024 tokens of a Kosovo law from Gazeta Zyrtare,
# r 16, alpha 32), unattended (Windows task ozaki-lora, S4U, 29-10-2026). It waits until SPEX and the ozaki-job task are
# done (SPEX timings need an idle machine) and for AC power, pauses model-lokal (as ozaki-job does), then:
#   the FP64 selftest of the backward; the step twice -- 4096 rows per product and 24 threads, then 40 rows and 7 threads:
#   every dB file must be the same bytes; then the finite-difference check, two directions at e = 0.004.
# Log /opt/ozaki/jlora/step/run.log; the steps' output in step_<nb>.log.
J=/opt/ozaki/jais; D=/opt/ozaki/jlora/step; B=/opt/ozaki/src; cd $B
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
pw() { $PS -NoProfile -Command "$1" 2>/dev/null | tr -d '\r'; }
ac() { [ "$(pw '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d ' ')" != 1 ]; }
log() { echo "$(date '+%F %T') $*" >> $D/run.log; }
log "waiting: SPEX, ozaki-job, AC"
while pgrep -x out_spex > /dev/null || [ "$(pw '(Get-ScheduledTask -TaskName ozaki-job).State')" = Running ] || ! ac || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done
touch /opt/ozaki/chain.busy; pw 'Disable-ScheduledTask -TaskName model-lokal | Out-Null; Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force'
log "start ($(git -C $B rev-parse --short HEAD 2>/dev/null || cat $B/.version 2>/dev/null))"
./out_jlora --selftest $J >> $D/run.log 2>&1; log "selftest rc=$?"
for nb in 4096 40; do
    th=$([ $nb = 4096 ] && echo 24 || echo 7); mkdir -p $D/dB_$nb
    until ac; do sleep 60; done
    log "step OZ_NB=$nb threads=$th"
    OZ_AHEAD=3 OZ_NB=$nb OMP_NUM_THREADS=$th OZ_FDEPS=0.004 OZ_DSEEDS=11,12 ./out_jlora $J $D/gzk2512.ids 1024 16 32 $D/dB_$nb > $D/step_$nb.log 2> $D/step_$nb.err
    log "rc=$? $(grep -h 'loss\|digest\|^step' $D/step_$nb.log | tr '\n' ' ')"
done
n=0; for f in $D/dB_4096/*.f32; do cmp -s $f $D/dB_40/$(basename $f) || { n=$((n + 1)); log "DIFFER $(basename $f)"; }; done
log "dB files compared: $(ls $D/dB_4096 | wc -l) files, $n differ"
for s in 11 12; do for e in 0.004 -0.004; do
    until ac; do sleep 60; done
    log "fd seed $s eps $e: $(OZ_AHEAD=3 OZ_DSEED=$s OZ_EPS=$e ./out_jlora $J $D/gzk2512.ids 1024 16 32 2>/dev/null | grep -o 'loss [0-9.]*')"
done; done
rm -f /opt/ozaki/chain.busy
pw 'Enable-ScheduledTask -TaskName model-lokal | Out-Null; Start-ScheduledTask -TaskName model-lokal'
log "done"
