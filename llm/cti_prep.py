# cti_prep.py -- CTIBench (AI4Sec/cti-bench, CC-BY-NC-SA-4.0) multiple choice as token ids for out_jais_batch: each
# question's official `Prompt` column as one user message through the checkpoint's own chat_template.jinja (rendered as
# transformers does: its {% generation %} tag is a no-op outside training), tokenized with tokenizer.json (the template
# writes the BOS itself). Writes <out>.ids (one prompt per line), <out>.gt (the answer letters) and prints the ids of the
# answer letters for OZ_SCORE.
# NOTHINK=1: the assistant turn opened with an empty thinking block, as the template itself writes one
# ("<ifm|think>\n</ifm|think>" + content), so the next token is the answer: its A-D logits are read in one pass.
# python3 -I llm/cti_prep.py <model dir> <cti-mcq.tsv> <chat_template.jinja> <out prefix> [first n]
import csv, os, sys, jinja2, tokenizers
from jinja2.ext import Extension
from jinja2 import nodes

class Generation(Extension):   # transformers' AssistantTracker: {% generation %}...{% endgeneration %} marks, renders as is
    tags = {"generation"}
    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_id", []), [], [], body).set_lineno(lineno)
    def _id(self, caller): return caller()

def fail(m): raise Exception(m)
model, tsv, tpl, out = sys.argv[1:5]
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t"))
if len(sys.argv) > 5: rows = rows[:int(sys.argv[5])]
env = jinja2.Environment(extensions=[Generation, "jinja2.ext.loopcontrols"], trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = fail
t = env.from_string(open(tpl, encoding="utf-8").read())
tok = tokenizers.Tokenizer.from_file(model + "/tokenizer.json")
with open(out + ".ids", "w") as fi, open(out + ".gt", "w") as fg:
    for i, r in enumerate(rows):
        s = t.render(messages=[{"role": "user", "content": r["Prompt"]}], add_generation_prompt=True, bos_token="<|ifm|begin_of_text|>")
        if os.environ.get("NOTHINK") == "1": s += "</ifm|think>"
        ids = tok.encode(s, add_special_tokens=False).ids
        if i == 0: sys.stderr.write(repr(s[-400:]) + "\n" + str(ids[:3]) + " ... " + str(ids[-12:]) + "\n")
        print(*ids, file=fi); print(r["GT"].strip().upper(), file=fg)
lens = [len(l.split()) for l in open(out + ".ids")]
sys.stderr.write(f"{len(lens)} prompts, {sum(lens)} tokens, longest {max(lens)}\n")
print(",".join(str(tok.encode(c, add_special_tokens=False).ids[0]) for c in "ABCD"),
      ",".join(str(tok.encode(" " + c, add_special_tokens=False).ids[0]) for c in "ABCD"))
