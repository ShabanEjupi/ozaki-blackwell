#!/bin/bash
# batch_run.sh -- the Windows task ozaki-batch (at logon and on demand): runs /opt/ozaki/b/job.sh (a copy the session writes;
# a script is never rewritten while bash reads it) with the GPU marked busy for the night jobs (/opt/ozaki/chain.busy).
# Never on battery: 28-10 the charger came out at 12:41, the job (24 threads + GPU + NVMe) kept running on battery, the
# embedded controller timed out three times and the laptop powered off at 13:40. So the job starts only on AC, is stopped
# (its whole process group) within a minute of the charger leaving, and restarts from the beginning when it is back --
# also after a reboot (logon trigger) until job.done exists.
B=/opt/ozaki/b; W=$(dirname "${BASH_SOURCE[0]}")/win.sh; [ -f "$W" ] || W=/opt/ozaki/src/llm/win.sh; . "$W"   # PS, pw, ac and a live interop socket (llm/win.sh)
log() { echo "$(date '+%F %T') $*" >> $B/job.log; }
[ -e $B/job.done ] && exit 0
[ -e $B/run.lock ] && kill -0 "$(cat $B/run.lock)" 2>/dev/null && exit 0; echo $$ > $B/run.lock
trap 'rm -f /opt/ozaki/chain.busy $B/run.lock; [ -n "$pid" ] && kill -- -$pid 2>/dev/null' EXIT
while :; do
    if ! ac; then log "on battery: waiting for the charger"; until ac; do sleep 60; done; log "charger back"; fi
    touch /opt/ozaki/chain.busy; log "job start"
    setsid bash $B/job.sh >> $B/job.log 2>&1 & pid=$!
    while kill -0 $pid 2>/dev/null; do sleep 60; ac || { log "charger out: job stopped"; kill -- -$pid; wait $pid; pid=; continue 2; }; done
    wait $pid; rc=$?; pid=; log "job rc=$rc"; [ $rc = 0 ] && touch $B/job.done; rm -f /opt/ozaki/chain.busy; exit $rc
done
