# cti_score.py -- CTI-MCQ accuracy from out_jais_batch's "score prompt p lA A lB B lC C lD D" lines: the answer is the
# letter with the largest logit (A first on a tie); also how often the greedy next token is one of the letters at all.
# python3 -I llm/cti_score.py <out_jais_batch stdout> <answers .gt> [letter ids, default 34,35,36,37]
import sys
gt = open(sys.argv[2]).read().split()
ids = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "34,35,36,37").split(",")]
ans, top = {}, {}
for line in open(sys.argv[1]):
    w = line.split()
    if w[:2] == ["score", "prompt"]:
        lg = [float(w[4 + 2 * i]) for i in range(4)]
        ans[int(w[2])] = "ABCD"[max(range(4), key=lambda i: (lg[i], -i))]
    elif w[:1] == ["step"]:
        top[int(w[3])] = int(w[9])   # the last step line of a prompt: its greedy next token
n = len(ans); ok = sum(ans[p] == gt[p] for p in ans)
print(f"CTI-MCQ: {ok}/{n} = {100 * ok / max(n, 1):.2f}% (majority letter {max(gt.count(c) for c in 'ABCD') / len(gt) * 100:.1f}%), "
      f"greedy token a letter in {sum(top.get(p) in ids for p in ans)}/{n}")
for c in "ABCD": print(c, "chosen", sum(a == c for a in ans.values()), "true", sum(gt[p] == c for p in ans))
