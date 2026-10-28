#!/bin/bash
# batch_run.sh -- the Windows task ozaki-batch: runs /opt/ozaki/b/job.sh (a copy the session writes; a script is never
# rewritten while bash reads it) with the GPU marked busy for the night jobs (/opt/ozaki/chain.busy).
B=/opt/ozaki/b; touch /opt/ozaki/chain.busy; trap 'rm -f /opt/ozaki/chain.busy' EXIT
echo "$(date '+%F %T') job start" >> $B/job.log
bash $B/job.sh >> $B/job.log 2>&1; echo "$(date '+%F %T') job rc=$?" >> $B/job.log
