#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="math/outputs/vllm_eval_external_v1"

mkdir -p "$OUT_DIR"


declare -A MODELS
MODELS["v1"]="math/outputs/math_v1_early_sft_merged"
MODELS["v2"]="math/outputs/math_v2_revised_sft_merged"
MODELS["v3"]="math/outputs/math_v3_curated_sft_merged"
MODELS["v4"]="math/outputs/math_v4_strong_sft_merged"
MODELS["v5_all_solutions_sft"]="math/outputs/math_v5_all_solutions_sft_merged_r96_e08"

for NAME in v1 v2 v3 v4 v5_all_solutions_sft; do
  MODEL="${MODELS[$NAME]}"

  if [ -d "$MODEL" ]; then
    echo
    echo "===== EVALUATING $NAME with 2k tokens ====="
    echo "MODEL=$MODEL"

    python math/scripts/eval/evaluate_math_vllm_pass8.py \
      --model_dir "$MODEL" \
      --out_jsonl "$OUT_DIR/${NAME}_external_v1_2k.jsonl" \
      --num_completions 8 \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 2048 \
      --max_model_len 8192 \
      --gpu_memory_utilization 0.90 \
      2>&1 | tee "$OUT_DIR/${NAME}_external_v1_2k.log"
  else
    echo
    echo "MISSING MODEL FOR $NAME: $MODEL"
  fi
done
