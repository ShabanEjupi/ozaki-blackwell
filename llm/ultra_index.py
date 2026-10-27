# ultra_index.py -- one text index of every tensor in a safetensors checkpoint (Nemotron 3 Ultra, OLMo 2), from the headers
# read from each shard (8-byte length + JSON): name dtype shape file offset nbytes.
# python3 -I llm/ultra_index.py /opt/ozaki/nv/full > /opt/ozaki/nv/full/index.txt ; --show prints one layer of each kind
import json, os, re, sys
d = sys.argv[1]
rows = []
for f in sorted(os.listdir(d)):
    if not f.endswith(".safetensors"): continue
    with open(os.path.join(d, f), "rb") as fh:
        n = int.from_bytes(fh.read(8), "little"); h = json.loads(fh.read(n)); base = 8 + n
    for k, v in h.items():
        if k == "__metadata__": continue
        a, b = v["data_offsets"]
        rows.append((k, v["dtype"], "x".join(map(str, v["shape"])) or "-", f, base + a, b - a))
if "--show" in sys.argv:
    for k, t, s, f, a, n in rows:
        m = re.match(r"backbone\.layers\.(\d+)\.", k)
        if (not m or m.group(1) in ("0", "1", "7")) and not re.search(r"experts\.([1-9]\d*)\.", k):
            print(k, t, s, f)
    sys.exit()
for r in rows: print(*r)
