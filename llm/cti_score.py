# cti_score.py -- CTI-MCQ accuracy from out_jais_batch's "score prompt p idA lA idB lB idC lC idD lD" lines: the answer is
# the letter with the largest logit (A first on a tie); also how often the greedy next token is one of the letters at all.
# The letter ids are read from the score lines themselves (28-10-2026: a default of 34..37, right only for the empty-thinking
# protocol, reported "a letter in 0/1063" on the answer-prefix run, whose letters are 358,408,353,421).
# python3 -I llm/cti_score.py <out_jais_batch stdout> <answers .gt>
import sys
gt = open(sys.argv[2]).read().split()
ans, top, ids = {}, {}, set()
for line in open(sys.argv[1]):
    w = line.split()
    if w[:2] == ["score", "prompt"]:
        lg = [float(w[4 + 2 * i]) for i in range(4)]
        ids.update(int(w[3 + 2 * i]) for i in range(4))
        ans[int(w[2])] = "ABCD"[max(range(4), key=lambda i: (lg[i], -i))]
    elif w[:1] == ["step"]:
        top[int(w[3])] = int(w[9])   # the last step line of a prompt: its greedy next token
n = len(ans); ok = sum(ans[p] == gt[p] for p in ans)
print(f"CTI-MCQ: {ok}/{n} = {100 * ok / max(n, 1):.2f}% (majority letter {max(gt.count(c) for c in 'ABCD') / len(gt) * 100:.1f}%), "
      f"greedy token a letter in {sum(top.get(p) in ids for p in ans)}/{n} (letter ids {sorted(ids)})")
for c in "ABCD": print(c, "chosen", sum(a == c for a in ans.values()), "true", sum(gt[p] == c for p in ans))
