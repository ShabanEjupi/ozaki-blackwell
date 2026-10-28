#!/bin/bash
# ultra_fetch.sh -- the whole Nemotron 3 Ultra NVFP4 checkpoint (352.3 GB, 113 shards) into /opt/ozaki/nv/full inside the
# WSL ext4 disk (read at NVMe speed later, not through /mnt/c). Resumable (curl -C -), every file checked against the sha256
# that Hugging Face lists (lfs.oid); a file is renamed into place only when its hash matches. Run by the scheduled task
# ozaki-ultra-fetch (Shaban, 27-10-2026: "at home, however long it takes"); progress in /opt/ozaki/nv/full/fetch.log.
# Any other repository: R=<org/name> D=<folder> (27-10: allenai/OLMo-2-0325-32B-Instruct, whose whole training data is indexed by infini-gram).
# MINFREE=<GB>: before each file wait until the disk keeps MINFREE GB free after it (27-10: Jais-3 FP8 starts in the space K2 leaves and finishes after Ultra is deleted).
R=${R:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4}; D=${D:-/opt/ozaki/nv/full}; mkdir -p $D; cd $D
log() { echo "$(date '+%F %T') $*" >> $D/fetch.log; }
curl -sf "https://huggingface.co/api/models/$R/tree/main?recursive=true" -o tree.json || { log "tree failed"; exit 1; }
python3 -I -c '
import json
for f in json.load(open("tree.json")):
    if f.get("type") == "file": print(f["path"], f["size"], (f.get("lfs") or {}).get("oid", "-"))' > files.txt
log "start: $(wc -l < files.txt) files, $(awk '{s+=$2} END {printf "%.1f GB", s/1e9}' files.txt)"
while read -r p size sha; do
    [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$size" ] && continue
    mkdir -p "$(dirname "$p")"
    w=0; while [ $(( $(df -B1 --output=avail $D | tail -1) - size )) -lt $(( ${MINFREE:-0} * 1000000000 )) ]; do
        [ $w = 0 ] && log "$p: waiting for disk space ($(df -h --output=avail $D | tail -1) free, MINFREE ${MINFREE}G)"; w=1; sleep 300; done
    for try in 1 2 3 4 5; do
        curl -sfL -C - --retry 20 --retry-delay 10 -o "$p.part" "https://huggingface.co/$R/resolve/main/$p"
        [ "$(stat -c %s "$p.part" 2>/dev/null)" = "$size" ] && break
        log "$p: try $try ended at $(stat -c %s "$p.part" 2>/dev/null) of $size"; sleep 30
    done
    if [ "$sha" != "-" ] && [ "$(sha256sum "$p.part" | cut -d' ' -f1)" != "$sha" ]; then log "$p: sha256 mismatch, removed"; rm -f "$p.part"; continue; fi
    mv "$p.part" "$p"; log "$p ok ($size)"
done < files.txt
log "done: $(ls $D/*.safetensors 2>/dev/null | wc -l) shards, $(du -sh $D | cut -f1)"
