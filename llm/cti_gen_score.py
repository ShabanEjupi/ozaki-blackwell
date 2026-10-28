# cti_gen_score.py -- CTI-MCQ from what Jais-3 generated after the prompt (the step lines of out_jais_batch with
# OZ_PERPROMPT=1, printed after every pass, so it scores mid-run too): the answer is the first standalone A-D in the
# generated text (after the empty thinking the model writes "The answer is B) ..." in 147 of the first 157, so the letter
# logit at the first position measures the wrong token); also the A-D logit answer from the score lines, for comparison.
# python3 -I llm/cti_gen_score.py <stdout> <prompts .ids> <answers .gt> <model dir>
import re, sys, tokenizers
out, ids, gtf, model = sys.argv[1:5]
plen = [len(l.split()) for l in open(ids)]; gt = open(gtf).read().split()
tok = tokenizers.Tokenizer.from_file(model + "/tokenizer.json")
nxt, lgt = {}, {}   # position t >= plen - 1 predicts generated token t + 1 - plen
for line in open(out):
    w = line.split()
    if w[:1] == ["step"]:
        t, p = int(w[1]), int(w[3])
        if t + 1 >= plen[p]: nxt.setdefault(p, {})[t] = int(w[9])
    elif w[:2] == ["score", "prompt"]:
        lg = [float(w[4 + 2 * i]) for i in range(4)]; lgt[int(w[2])] = "ABCD"[max(range(4), key=lambda i: (lg[i], -i))]
gen = {p: tok.decode([d[t] for t in sorted(d)]) for p, d in nxt.items()}
pick = {p: (m.group(1) if (m := re.search(r"(?<![A-Za-z])([ABCD])(?![a-z])", g)) else "-") for p, g in gen.items()}
n = len(pick); ok = sum(pick[p] == gt[p] for p in pick); none = sum(v == "-" for v in pick.values())
print(f"generated answer: {ok}/{n} = {100 * ok / max(n, 1):.2f}% (no letter in {none}, tokens so far {min((len(d) for d in nxt.values()), default=0)}); "
      f"A-D logit at the first position: {sum(lgt[p] == gt[p] for p in lgt)}/{len(lgt)}; majority letter {max(gt.count(c) for c in 'ABCD') / len(gt) * 100:.1f}%")
for p in sorted(gen)[:3]: print(p, gt[p], repr(gen[p]))
