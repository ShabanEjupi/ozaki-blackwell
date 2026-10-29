# korpusi_prep.py -- the crawled public corpus (dogana-rag/harvest korpusi.sh, ews_dokumentet.py) into training sequences.
# clean <out.jsonl> <harvest jsonl>... : per source (the file), a line that recurs in >= 20 % of its documents (and >= 5)
#   is the site's frame (menus, footers: Gazeta Zyrtare's "Shqip English Srpski ... FILLIMI") and goes; the harvester's
#   passage marks go; a url's last line wins (--append keeps changed pages); documents under 500 characters go; an exact
#   duplicate of an earlier text goes. Out: {"src", "url", "text"} per line.
# tok <model dir> <clean.jsonl> <out prefix> [tokens per sequence, 1024]: the model's own tokenizer, documents joined by
#   its EOS into one stream, cut into sequences: <out>.u32 (uint32 ids, row after row) and <out>.json (counts per source).
# python3 -I llm/korpusi_prep.py clean ~/korpusi/clean.jsonl ~/korpusi/*/harvest/*.jsonl ~/korpusi/email/harvest/*.jsonl
import collections, hashlib, json, os, re, struct, sys

# a law's own structure recurs in every law: "Neni 1", "Kapitulli I" are never the frame
HEAD = re.compile(r"(neni|article|[cč]lan|kapitulli|chapter|pjesa|part|seksioni|section)\s+\w+\s*$", re.I)


def clean(out, files):
    seen, stats = set(), {}
    with open(out, "w", encoding="utf-8") as o:
        for fn in files:
            src = os.path.basename(fn).rsplit(".", 1)[0]
            docs = {}
            for line in open(fn, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue   # a line cut by a killed run
                docs[d["url"]] = d
            df, nk = collections.Counter(), collections.Counter()   # per kind: the PDFs of a site have no menus
            for d in docs.values():
                nk[d.get("kind")] += 1
                df.update((d.get("kind"), x) for x in set(x.strip() for x in d["text"].split("\n") if x.strip()))
            frame = {kx for kx, c in df.items() if c >= 5 and c >= 0.2 * nk[kx[0]] and not HEAD.match(kx[1])}
            kept = 0
            for d in docs.values():
                lines = [x.strip() for x in d["text"].split("\n")]
                lines = [x for x in lines if x and (d.get("kind"), x) not in frame and not re.fullmatch(r"[⁂\s]*", x)]
                text = re.sub(r"[ \t]{2,}", " ", "\n".join(lines)).strip()
                h = hashlib.sha1(text.encode()).digest()
                if len(text) < 500 or h in seen:
                    continue
                seen.add(h)
                o.write(json.dumps({"src": src, "url": d["url"], "text": text}, ensure_ascii=False) + "\n")
                kept += 1
            stats[src] = {"docs": len(docs), "kept": kept, "frame_lines": len(frame)}
            print(src, json.dumps(stats[src]), flush=True)
    return stats


def tok(model, inp, prefix, seq=1024):
    import tokenizers
    t = tokenizers.Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
    cfg = json.load(open(os.path.join(model, "config.json")))
    eos = cfg.get("eos_token_id"); eos = eos[0] if isinstance(eos, list) else eos
    buf, n, per = [], 0, collections.Counter()
    with open(prefix + ".u32", "wb") as o:
        for line in open(inp, encoding="utf-8"):
            d = json.loads(line)
            ids = t.encode(d["text"], add_special_tokens=False).ids + [eos]
            per[d["src"]] += len(ids)
            buf += ids
            while len(buf) >= seq:
                o.write(struct.pack("<%dI" % seq, *buf[:seq])); buf = buf[seq:]; n += 1
    json.dump({"sequences": n, "tokens_per_sequence": seq, "eos": eos, "tokens_by_source": dict(per.most_common())},
              open(prefix + ".json", "w"), indent=1)
    print(n, "sequences of", seq, "tokens;", sum(per.values()), "tokens")


if __name__ == "__main__":
    if sys.argv[1] == "clean":
        clean(sys.argv[2], sys.argv[3:])
    else:
        tok(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 1024)
