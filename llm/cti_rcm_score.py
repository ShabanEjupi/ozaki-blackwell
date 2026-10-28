# cti_rcm_score.py -- CTI-RCM (CVE description -> CWE) from what Jais-3 generated after the answer prefix "CWE-" (cti_prep.py
# with NOTHINK=1 PREFIX="CWE-": the official prompt, the template's empty thinking, then the prefix, so the next tokens are the
# number): the answer is "CWE-" + the digits the model wrote first. The official protocol lets the model justify first and
# reads the last line; the prefix asks for the id at once (a direct answer, documented as such). Scores mid-run too.
# python3 -I llm/cti_rcm_score.py <stdout of out_jais_batch, OZ_PERPROMPT=1> <prompts .ids> <answers .gt> <model dir>
import re, sys, collections, tokenizers
out, ids, gtf, model = sys.argv[1:5]
plen = [len(l.split()) for l in open(ids)]; gt = open(gtf).read().split()
tok = tokenizers.Tokenizer.from_file(model + "/tokenizer.json")
nxt = {}   # position t >= plen - 1 predicts generated token t + 1 - plen
for line in open(out):
    w = line.split()
    if w[:1] == ["step"]:
        t, p = int(w[1]), int(w[3])
        if t + 1 >= plen[p]: nxt.setdefault(p, {})[t] = int(w[9])
gen = {p: tok.decode([d[t] for t in sorted(d)]) for p, d in nxt.items()}
pick = {p: ("CWE-" + m.group(1) if (m := re.match(r"\s*(\d+)", g)) else "-") for p, g in gen.items()}
n = len(pick); ok = sum(pick[p] == gt[p] for p in pick); none = sum(v == "-" for v in pick.values())
top, c = collections.Counter(gt).most_common(1)[0]
print(f"CTI-RCM: {ok}/{n} = {100 * ok / max(n, 1):.2f}% (no number in {none}; majority class {top} {100 * c / len(gt):.1f}%; "
      f"tokens so far {min((len(d) for d in nxt.values()), default=0)})")
for p in sorted(gen)[:3]: print(p, gt[p], repr(gen[p]))
