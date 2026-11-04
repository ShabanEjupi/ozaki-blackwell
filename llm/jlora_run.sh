#!/bin/bash
# jlora_run.sh -- the full LoRA training of Jais-3-375B-A23B-FP8 (61 layers, 1024 tokens of a Kosovo law from Gazeta Zyrtare,
# r 16, alpha 32), unattended (Windows task ozaki-lora, S4U, 29-10-2026). It waits until SPEX and the ozaki-job task are done
# (SPEX timings need an idle machine) and for AC power, pauses model-lokal as ozaki-job does, then runs llm/jlora_steps.sh,
# read only then. Log /opt/ozaki/jlora/step/run.log.
J=/opt/ozaki/jais; D=/opt/ozaki/jlora/step; B=/opt/ozaki/src; cd $B
W=$(dirname "${BASH_SOURCE[0]}")/win.sh; [ -f "$W" ] || W=/opt/ozaki/src/llm/win.sh; . "$W"   # PS, pw, ac and a live interop socket (llm/win.sh)
log() { echo "$(date '+%F %T') $*" >> $D/run.log; }
log "waiting: SPEX, ozaki-job, AC"
# chain.busy from before this WSL boot is a dead run's (31-10: the laptop was switched off at 01:24 mid-step, the marker stayed
# and this loop would have waited for it forever): every process that set it is gone
if [ -e /opt/ozaki/chain.busy ] && [ "$(stat -c %Y /opt/ozaki/chain.busy)" -lt $(( $(date +%s) - $(cut -d. -f1 /proc/uptime) )) ]; then
    log "stale chain.busy ($(date -r /opt/ozaki/chain.busy '+%F %T'), before this boot) removed"; rm -f /opt/ozaki/chain.busy; fi
# the same for spex.want (02-11): a measurement or the voice job that held it died with the boot, and jlora_steps.sh would
# yield to it forever
if [ -e /opt/ozaki/spex.want ] && [ "$(stat -c %Y /opt/ozaki/spex.want)" -lt $(( $(date +%s) - $(cut -d. -f1 /proc/uptime) )) ]; then
    log "stale spex.want ($(date -r /opt/ozaki/spex.want '+%F %T'): $(head -c 40 /opt/ozaki/spex.want), before this boot) removed"; rm -f /opt/ozaki/spex.want; fi
while pgrep -x out_spex > /dev/null || [ "$(pw '(Get-ScheduledTask -TaskName ozaki-job).State')" = Running ] || ! ac || [ -e /opt/ozaki/chain.busy ]; do sleep 60; done
touch /opt/ozaki/chain.busy; pw 'Disable-ScheduledTask -TaskName model-lokal | Out-Null; Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force'
log "start ($(git -C $B rev-parse --short HEAD 2>/dev/null || cat $B/.version 2>/dev/null))"
bash $B/llm/jlora_steps.sh
rm -f /opt/ozaki/chain.busy
# model-lokal stays off at the end: Shabani, 29-10-2026 -- paused without limit while the training runs (no Enable here)
log "done"
