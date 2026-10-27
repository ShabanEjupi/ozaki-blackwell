# ultra_tok.py -- prompts to token ids for out_ultra, and its output back to text, with the checkpoint's tokenizer.json
# python3 -I llm/ultra_tok.py enc <model dir> llm/ultra_prompts.txt > prompts.ids
# Special tokens as the model's own tokenizer adds them: none when tokenizer_config says add_bos_token false (Nemotron 3
# Ultra), else tokenizer.json's post-processor decides (K2-Think-V2: <|begin_of_text|>; OLMo 2: none).
# python3 -I llm/ultra_tok.py dec <model dir> < out_ultra.log
import json, os, sys, tokenizers
tok = tokenizers.Tokenizer.from_file(sys.argv[2] + "/tokenizer.json")
tc = os.path.join(sys.argv[2], "tokenizer_config.json")
special = (json.load(open(tc)) if os.path.exists(tc) else {}).get("add_bos_token") is not False
if sys.argv[1] == "enc":
    for p in open(sys.argv[3], encoding="utf-8").read().splitlines():
        if p: print(*tok.encode(p, add_special_tokens=special).ids)
else:
    for line in sys.stdin:
        if line.startswith("seq "):
            head, ids = line.split(":", 1)
            print(head, repr(tok.decode([int(v) for v in ids.split()])))
