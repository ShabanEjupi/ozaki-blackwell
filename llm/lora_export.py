#!/usr/bin/env python3
# lora_export.py <ckpt.bin> <out dir> [alpha (32)] [base model name] -- jais_lora.cu's checkpoint (OZ_CKPT) as a PEFT adapter:
# adapter_model.safetensors (BF16, the values the forward used: every FP32 master rounded to nearest even) and
# adapter_config.json, loadable with peft.PeftModel.from_pretrained(base, <out dir>) on the K2-Horizon / Jais-3 base.
# The checkpoint: int32 {layers, rank, hidden, steps}, then per layer A_q, A_v, B_q, B_v (FP32) and the AdamW moments
# m_i, v_i (FP64), each as uint64 count + data. A is (rank x hidden), B is (out x rank): PEFT's lora_A / lora_B shapes.
import json, os, struct, sys
from array import array


def main(ckpt, out, alpha="32", base="Jais-3-375B-A23B-FP8"):
    with open(ckpt, "rb") as f:
        nl, rk, dh, steps = struct.unpack("<4i", f.read(16))

        def vec(code):
            n, = struct.unpack("<Q", f.read(8)); a = array(code); a.frombytes(f.read(n * a.itemsize))
            if len(a) != n: sys.exit(f"{ckpt}: truncated")
            return a

        tensors = {}
        for L in range(nl):
            aq, av, bq, bv = (vec("f") for _ in range(4))
            for _ in range(8): vec("d")   # the moments: training state, not the adapter
            if not bq: sys.exit(f"{ckpt}: B of layer {L} is empty (no step applied)")
            p = f"base_model.model.model.layers.{L}.self_attn."
            tensors[p + "q_proj.lora_A.weight"] = (aq, [rk, dh]); tensors[p + "v_proj.lora_A.weight"] = (av, [rk, dh])
            tensors[p + "q_proj.lora_B.weight"] = (bq, [len(bq) // rk, rk]); tensors[p + "v_proj.lora_B.weight"] = (bv, [len(bv) // rk, rk])
    head, blobs, off = {"__metadata__": {"format": "pt", "steps": str(steps)}}, [], 0
    for name, (a, shape) in tensors.items():
        u = array("I"); u.frombytes(a.tobytes())
        b = array("H", [((x + 0x7FFF + ((x >> 16) & 1)) >> 16) & 0xFFFF for x in u])   # round to nearest even (no NaNs here)
        blob = b.tobytes(); head[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(blob)]}
        blobs.append(blob); off += len(blob)
    h = json.dumps(head, separators=(",", ":")).encode(); h += b" " * (-len(h) % 8)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "adapter_model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h)
        for blob in blobs: f.write(blob)
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base, "r": rk, "lora_alpha": float(alpha),
           "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"], "bias": "none", "fan_in_fan_out": False, "inference_mode": True}
    with open(os.path.join(out, "adapter_config.json"), "w") as f: json.dump(cfg, f, indent=2)
    print(f"{out}: {nl} layers, rank {rk}, {steps} steps, {off / 1e6:.1f} MB")


if __name__ == "__main__":
    main(*sys.argv[1:5]) if len(sys.argv) >= 3 else sys.exit("usage: lora_export.py <ckpt.bin> <out dir> [alpha] [base]")
