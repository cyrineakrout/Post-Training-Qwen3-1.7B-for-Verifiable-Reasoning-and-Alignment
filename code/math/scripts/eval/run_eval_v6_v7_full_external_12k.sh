#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="math/outputs/vllm_eval_external_v1_12k"

mkdir -p "$OUT_DIR"


declare -A MODELS
MODELS["v6"]="math/outputs/math_v6_50k_long_merged_ckpt1500"
MODELS["v7_ckpt1600"]="math/outputs/math_v7_100k_merged_ckpt1600"
MODELS["v7_ckpt2000"]="math/outputs/math_v7_100k_merged_ckpt2000"
MODELS["v7_ckpt2400"]="math/outputs/math_v7_100k_merged_ckpt2400"

for NAME in v6 v7_ckpt1600 v7_ckpt2000 v7_ckpt2400; do
  MODEL="${MODELS[$NAME]}"

  if [ -d "$MODEL" ]; then
    echo
    echo "===== EVALUATING $NAME full external 12k ====="
    echo "MODEL=$MODEL"

    python math/scripts/eval/evaluate_math_vllm_pass8.py \
      --model_dir "$MODEL" \
      --out_jsonl "$OUT_DIR/${NAME}_external_v1_12k.jsonl" \
      --num_completions 8 \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 12000 \
      --max_model_len 40960 \
      --gpu_memory_utilization 0.90 \
      2>&1 | tee "$OUT_DIR/${NAME}_external_v1_12k.log"
  else
    echo
    echo "MISSING MODEL FOR $NAME: $MODEL"
  fi
done
