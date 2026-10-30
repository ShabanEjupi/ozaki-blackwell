#!/bin/bash
# train.sh <model dir> <text .jsonl> <out dir> [runs] -- one command from documents to a PEFT adapter: rank-16 LoRA on q_proj and
# v_proj of every layer of Jais-3-375B-A23B-FP8, streamed from disk on one 12 GB GPU, every product of the forward and the
# backward correctly rounded (jais_lora.cu), so a run is a function of its inputs alone and resumes bit for bit.
#   <text .jsonl>: one {"text": ...} per line (korpusi_prep.py clean writes them); tokenized once with the model's own
#   tokenizer into 1024-token sequences (<out>/data.u32), the last OZ_EVAL (2) held out.
#   Each run is OZ_STEPS (2) steps of OZ_BATCH (8) sequences, resumed from <out>/ckpt.bin; after each run the checkpoint is
#   written as <out>/adapter (adapter_model.safetensors + adapter_config.json, peft.PeftModel.from_pretrained). [runs]: how
#   many (default: until <out>/STOP exists). The log is <out>/train.log: per step the loss, the held-out loss, the time.
# Licensed under LICENSE (restricted, not open source): private study and open academic research only.
set -euo pipefail
cd "$(dirname "$0")/.."
J=${1:?model dir}; TXT=${2:?text .jsonl}; O=${3:?out dir}; RUNS=${4:-0}; mkdir -p "$O"
[ -x out_jlora ] || nvcc -O3 -arch=sm_120a -std=c++17 -Xcompiler -fopenmp -o out_jlora llm/jais_lora.cu -lgomp
[ -s "$O/data.u32" ] || python3 -I llm/korpusi_prep.py tok "$J" "$TXT" "$O/data" 1024
run=0
while [ ! -e "$O/STOP" ] && { [ "$RUNS" = 0 ] || [ $run -lt "$RUNS" ]; }; do
    OZ_AHEAD=${OZ_AHEAD:-3} OZ_GATT=${OZ_GATT:-1} OZ_NEXT=1 OZ_BATCH=${OZ_BATCH:-8} OZ_EVAL=${OZ_EVAL:-2} OZ_STEPS=${OZ_STEPS:-2} \
        OZ_LR=${OZ_LR:-1e-4} OZ_CKPT="$O/ckpt.bin" ./out_jlora "$J" "$O/data.u32" 1024 16 32 >> "$O/train.log" 2>> "$O/train.err"
    python3 -I llm/lora_export.py "$O/ckpt.bin" "$O/adapter" 32
    run=$((run + 1)); grep -h 'held-out loss\|time' "$O/train.log" | tail -2
done
