#!/bin/bash
# jlora_run.sh -- the full LoRA training of Jais-3-375B-A23B-FP8 (61 layers, 1024 tokens of a Kosovo law from Gazeta Zyrtare,
# r 16, alpha 32), unattended (Windows task ozaki-lora, S4U, 29-10-2026). It waits until SPEX and the ozaki-job task are done
# (SPEX timings need an idle machine) and for AC power, pauses model-lokal as ozaki-job does, then runs llm/jlora_steps.sh,
# read only then. Log /opt/ozaki/jlora/step/run.log.
J=/opt/ozaki/jais; D=/opt/ozaki/jlora/step; B=/opt/ozaki/src; cd $B
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
pw() { $PS -NoProfile -Command "$1" 2>/dev/null | tr -d '\r'; }
ac() { [ "$(pw '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d ' ')" != 1 ]; }
log() { echo "$(date '+%F %T') $*" >> $D/run.log; }
log "waiting: SPEX, ozaki-job, AC"
while pgrep -x out_spex > /dev/null || [ "$(pw '(Get-ScheduledTask -TaskName ozaki-job).State')" = Running ] || ! ac || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done
touch /opt/ozaki/chain.busy; pw 'Disable-ScheduledTask -TaskName model-lokal | Out-Null; Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force'
log "start ($(git -C $B rev-parse --short HEAD 2>/dev/null || cat $B/.version 2>/dev/null))"
bash $B/llm/jlora_steps.sh
rm -f /opt/ozaki/chain.busy
pw 'Enable-ScheduledTask -TaskName model-lokal | Out-Null; Start-ScheduledTask -TaskName model-lokal'
log "done"
