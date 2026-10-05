#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="math/outputs/vllm_eval_external_v1"
MODEL="math/outputs/math_v5_all_solutions_sft_merged_r96_e08"

mkdir -p "$OUT_DIR"


echo "Using model: $MODEL"
test -d "$MODEL" || { echo "ERROR: model folder missing: $MODEL"; exit 1; }
test -f "$MODEL/model.safetensors" || echo "WARNING: model.safetensors not found"

python math/scripts/eval/evaluate_math_vllm_pass8.py \
  --model_dir "$MODEL" \
  --out_jsonl "$OUT_DIR/v5_all_solutions_sft_external_v1_2k.jsonl" \
  --num_completions 8 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --max_tokens 2048 \
  --max_model_len 8192 \
  --gpu_memory_utilization 0.90 \
  2>&1 | tee "$OUT_DIR/v5_all_solutions_sft_external_v1_2k.log"
