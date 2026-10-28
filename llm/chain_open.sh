#!/bin/bash
# chain_open.sh -- the day's GPU work on the laptop, unattended (Windows task ozaki-chain). 28-10-2026 order, after the WSL VM
# vanished at 18:52 on 27-10 and took SPEX pass 3, the K2 download and the night with it:
# 1. when K2-Think-V2 is downloaded: exact vs engine s=1/s=4 on the three prompts (OLMo 2 ran on 27-10);
# 2. Nemotron 3 Ultra run2 with read-ahead (OZ_AHEAD=3) on the same prompts as run1; when it exits 0, uncertified 0 and its
#    exact transcripts equal run1's token for token, the 329 GiB of Ultra shards are deleted (Shaban, 27-10: "Ultra is
#    deleted after run2") so the Jais-3 FP8 download (ozaki-jais-fetch, waits for disk space) can finish. The space
#    comes back to Windows C: only offline: fstrim here, then compact.request asks the Windows task ozaki-compact for
#    wsl --shutdown + diskpart compact vdisk, after which it restarts ozaki-jais-fetch (28-10: the vhdx had filled C:).
# /opt/ozaki/chain.busy exists while the chain computes: the SPEX job (/opt/ozaki/job/run.sh, evenings) waits for it, so its
# timings are taken on an idle machine. Frozen copy in /opt/ozaki/chain (binaries + scripts); log /opt/ozaki/chain/chain.log;
# a heartbeat every minute in /opt/ozaki/heartbeat (the second the VM dies, with memory and VRAM in use).
C=/opt/ozaki/chain; B=/opt/ozaki/chain.busy; U=/opt/ozaki/ultra; NV=/opt/ozaki/nv/full; touch $B
log() { echo "$(date '+%F %T') $*" >> $C/chain.log; }
(while true; do echo "$(date '+%F %T') up $(uptime -s) mem $(free -m | awk '/Mem/{print $3}') MB vram $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> /opt/ozaki/heartbeat; sleep 60; done) &
run() {   # <model dir> <name>
    python3 -I $C/ultra_index.py $1 > $1/index.txt
    python3 -I $C/ultra_tok.py enc $1 $C/ultra_prompts.txt > $C/$2.ids
    log "$2 start ($(wc -l < $1/index.txt) tensors)"
    $C/out_dense $1 $C/$2.ids 24 0,1,4 > $C/$2.log 2> $C/$2.err; log "$2 rc=$?"
    python3 -I $C/ultra_tok.py dec $1 < $C/$2.log > $C/$2.txt
}
until grep -q " done:" /opt/ozaki/k2/fetch.log 2>/dev/null; do sleep 60; done
run /opt/ozaki/k2 k2
log "ultra run2 start"
cd $U && OZ_AHEAD=3 $C/out_ultra $NV prompts.ids 24 0,1,4 > run2.log 2> run2.err; rc=$?
python3 -I $C/ultra_tok.py dec $NV < run2.log > run2.txt
same=$(cmp -s <(grep "^seq prompt . mode 0:" run1.log) <(grep "^seq prompt . mode 0:" run2.log) && echo yes || echo no)
log "ultra run2 rc=$rc, $(grep '^uncertified' run2.log), exact transcripts equal to run1: $same"
if [ $rc = 0 ] && grep -q "^uncertified 0$" run2.log && [ $same = yes ]; then
    rm -f $NV/*.safetensors; log "ultra shards deleted, $(df -h --output=avail /opt/ozaki | tail -1) free"
    log "$(fstrim -v / 2>&1)"; touch /mnt/c/Users/NB/model-lokal/wsl/compact.request
fi
rm -f $B
log "chain done"
kill %1
