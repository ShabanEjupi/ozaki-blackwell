# ultra_summary.py -- the exact path against the FP32 engines (split-K 1 and 4) from an out_ultra / out_dense log:
# per prompt, the generated text of each path, the first generated position where an engine picks another token, and how
# far its top logit is from the exact one while the tokens still agree. Plus the shadow statistics and digests.
# python3 -I llm/ultra_summary.py <model dir (tokenizer.json)> <log>
import re, sys, tokenizers
tok = tokenizers.Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
steps, seqs, extra = {}, {}, []
for line in open(sys.argv[2]):
    m = re.match(r"step (\d+) prompt (\d+) mode (\d+) token (\d+) next (\d+) top1 (\S+) top2 (\d+) (\S+)", line)
    if m:
        t, p, md, _, nx, l1, _, l2 = m.groups(); steps[(int(p), int(md), int(t))] = (int(nx), float(l1), float(l2)); continue
    m = re.match(r"seq prompt (\d+) mode (\d+): (.*)", line)
    if m: seqs[(int(m.group(1)), int(m.group(2)))] = [int(v) for v in m.group(3).split()]; continue
    if line.startswith(("shadow", "uncertified", "digest")): extra.append(line.rstrip())
for p in sorted({k[0] for k in seqs}):
    modes = sorted(md for (q, md) in seqs if q == p)
    print(f"prompt {p}:")
    for md in modes: print(f"  mode {md} ({'exact' if md == 0 else f'engine split-K {md}'}): {tok.decode(seqs[(p, md)])!r}")
    ex = seqs[(p, 0)]
    for md in modes:
        if md == 0: continue
        en = seqs[(p, md)]
        div = next((i for i in range(min(len(ex), len(en))) if ex[i] != en[i]), None)
        same = [t for (q, m_, t) in steps if q == p and m_ == md and (div is None or t < div - 1)]
        d = [abs(steps[(p, md, t)][1] - steps[(p, 0, t)][1]) for t in same if (p, 0, t) in steps]
        print(f"  engine s={md}: " + (f"first different token at position {div} ({tok.decode([ex[div]])!r} exact vs {tok.decode([en[div]])!r})" if div is not None else "same tokens throughout")
              + (f"; |top1 logit - exact| before that: max {max(d):.3f}, mean {sum(d) / len(d):.3f} over {len(d)} steps" if d else ""))
    margins = [steps[(p, 0, t)][1] - steps[(p, 0, t)][2] for (q, m_, t) in steps if q == p and m_ == 0]
    print(f"  exact top1-top2 margin: min {min(margins):.3f} over {len(margins)} steps")
for e in extra: print(e)
