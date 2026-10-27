#!/usr/bin/env python3
# k3_fetch.py <shard> <regex> <outdir> -- fetch selected tensors of one Kimi K3 safetensors shard with HTTP Range requests
# (only the bytes of the chosen tensors travel; a whole shard is ~16 GB). Writes <name>.bin per tensor and meta.json.
# Source: https://huggingface.co/moonshotai/Kimi-K3 (2 779 931 837 184 parameters; routed experts MXFP4, group 32, E8M0 scales).
import json, os, re, struct, sys, urllib.request

REPO = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"

def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b - 1), "User-Agent": "ozaki-blackwell"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != b - a:
        raise SystemExit("short read %d of %d at %d" % (len(data), b - a, a))
    return data

shard, pat, out = sys.argv[1], re.compile(sys.argv[2]), sys.argv[3]
url = REPO + shard
os.makedirs(out, exist_ok=True)
n = struct.unpack("<Q", get(url, 0, 8))[0]
hdr = json.loads(get(url, 8, 8 + n))
meta, total = {}, 0
for name, t in sorted(hdr.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1):
    if name == "__metadata__" or not pat.search(name):
        continue
    a, b = t["data_offsets"]
    path = os.path.join(out, name + ".bin")
    if not (os.path.exists(path) and os.path.getsize(path) == b - a):
        with open(path + ".part", "wb") as f:
            f.write(get(url, 8 + n + a, 8 + n + b))
        os.replace(path + ".part", path)
    meta[name] = {"dtype": t["dtype"], "shape": t["shape"]}
    total += b - a
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
print("%d tensors, %.1f MB -> %s" % (len(meta), total / 1e6, out))
