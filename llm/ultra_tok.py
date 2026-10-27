# ultra_tok.py -- prompts to token ids for out_ultra, and its output back to text, with the checkpoint's tokenizer.json
# python3 -I llm/ultra_tok.py enc <model dir> llm/ultra_prompts.txt > prompts.ids   (no BOS: add_bos_token is false)
# python3 -I llm/ultra_tok.py dec <model dir> < out_ultra.log
import sys, tokenizers
tok = tokenizers.Tokenizer.from_file(sys.argv[2] + "/tokenizer.json")
if sys.argv[1] == "enc":
    for p in open(sys.argv[3], encoding="utf-8").read().splitlines():
        if p: print(*tok.encode(p, add_special_tokens=False).ids)
else:
    for line in sys.stdin:
        if line.startswith("seq "):
            head, ids = line.split(":", 1)
            print(head, repr(tok.decode([int(v) for v in ids.split()])))
