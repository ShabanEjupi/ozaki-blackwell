# jais_lora_prep.py <model dir> <out dir> <N tokens> [layer] [seed] -- real inputs for lora.cu (OZ_W / OZ_X):
# W = layer L's q_proj (BF16, [out][in] as stored), X = the BF16 input of that projection for N token ids:
# RMSNorm(embedding) = BF16(w * x / rms), eps from config.json, in float32 as the model card's reference does.
# The ids are drawn uniformly (seeded) from the vocabulary: real embedding rows, not a sentence.
import sys, json
import numpy as np
mdir, odir, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
L = int(sys.argv[4]) if len(sys.argv) > 4 else 0
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 2026
cfg = json.load(open(mdir + "/config.json"))
H, eps = cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-6)
idx = {}
for line in open(mdir + "/index.txt"):
    nm, dt, sh, fl, off, n = line.split(); idx[nm] = (fl, int(off), int(n))
def raw(name, rows=None, row0=0, width=0):
    fl, off, n = idx[name]
    if rows is None: return np.fromfile("%s/%s" % (mdir, fl), dtype=np.uint16, count=n // 2, offset=off)
    return np.fromfile("%s/%s" % (mdir, fl), dtype=np.uint16, count=rows * width, offset=off + 2 * row0 * width)
def f32(b): return (b.astype(np.uint32) << 16).view(np.float32)
def bf16(f):   # round to nearest even
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
P = "model.layers.%d." % L
q = raw(P + "self_attn.q_proj.weight")
Dout = q.size // H
q.tofile(odir + "/W.bf16")
nw = f32(raw(P + "input_layernorm.weight"))
V = cfg["vocab_size"]
ids = np.random.default_rng(seed).integers(2, V, size=N)
X = np.empty((N, H), dtype=np.uint16)
for j, t in enumerate(ids):
    e = f32(raw("model.embed_tokens.weight", 1, int(t), H))
    r = np.float32(1) / np.sqrt(np.mean(e * e, dtype=np.float32) + np.float32(eps), dtype=np.float32)
    X[j] = bf16(nw * (e * r))
X.tofile(odir + "/X.bf16")
print("q_proj L%d: %d x %d, X: %d x %d, ids %s..." % (L, Dout, H, N, H, ids[:4]))
