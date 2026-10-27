# olmo_trace.py -- where does a generated text come from? Every maximal word span of it that occurs verbatim in the
# training data of OLMo-2-0325-32B-Instruct (infini-gram index v4_olmo-2-0325-32b-instruct_llama: all 4.6T training
# tokens, pre- and post-training; https://api.infini-gram.io, docs infini-gram.readthedocs.io/en/latest/api.html), with
# its count and one document that contains the longest span.
# python3 -I llm/olmo_trace.py "text" [min words, default 3]   (run where the internet is: ssh ampere)
import json, sys, time, urllib.request
API, IDX = "https://api.infini-gram.io/", "v4_olmo-2-0325-32b-instruct_llama"
def q(**kw):
    kw["index"] = IDX
    for t in range(6):
        try:
            r = urllib.request.Request(API, json.dumps(kw).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=60) as f: out = json.load(f)
            if "error" not in out: return out
            err = out["error"]
        except Exception as e: err = str(e)
        time.sleep(2 * (t + 1))
    raise SystemExit(f"infini-gram: {err}")
text, minw = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3
w = text.split()
spans, best = [], {}
for i in range(len(w)):   # longest j with count(w[i:j]) > 0; counts only shrink as a span grows
    j, c = i, 0
    while j < len(w):
        n = q(query_type="count", query=" ".join(w[i:j + 1]))["count"]
        if n == 0: break
        j, c = j + 1, n
    if j - i >= minw and not any(a <= i and j <= b for a, b, _ in spans): spans.append((i, j, c))
print(f"text: {text!r}\nindex: {IDX}\nmaximal spans of >= {minw} words found verbatim in the training data:")
for a, b, c in spans: print(f"  [{a}:{b}] {b - a} words, {c} occurrences: {' '.join(w[a:b])!r}")
if spans:
    a, b, c = max(spans, key=lambda s: (s[1] - s[0], -s[2]))
    f = q(query_type="find", query=" ".join(w[a:b]))
    for s, (lo, hi) in enumerate(f["segment_by_shard"]):
        if hi > lo:
            d = q(query_type="get_doc_by_rank", s=s, rank=lo, max_disp_len=80, query=" ".join(w[a:b]))   # query is required
            snip = "".join(p[0] if isinstance(p, (list, tuple)) else str(p) for p in d.get("spans", []))
            md = d.get("metadata", "{}"); md = json.loads(md) if isinstance(md, str) else md
            src = (md.get("metadata") or {}).get("id") or md.get("path")
            print(f"longest span, one source document: doc {d['doc_ix']} ({d['doc_len']} tokens), {md.get('path')}, source {src}\n  ...{snip[:400]!r}...")
            break
else: print("  none: no span of that length occurs verbatim")
