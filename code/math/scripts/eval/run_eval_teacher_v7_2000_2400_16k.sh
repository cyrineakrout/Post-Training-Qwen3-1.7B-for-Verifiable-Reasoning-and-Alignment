#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

TEACHER_EVAL="math/data/validation_math_converted.jsonl"
OUT_DIR="math/outputs/vllm_eval_teacher_v7_late"

mkdir -p "$OUT_DIR"

echo "Using teacher eval: $TEACHER_EVAL"
test -f "$TEACHER_EVAL" || { echo "ERROR: missing $TEACHER_EVAL"; exit 1; }
wc -l "$TEACHER_EVAL"

for STEP in 2000 2400; do
  MODEL="math/outputs/math_v7_100k_merged_ckpt${STEP}"

  if [ -d "$MODEL" ]; then
    echo
    echo "===== EVALUATING ckpt$STEP teacher samples 16k ====="
    echo "MODEL=$MODEL"

    python math/scripts/eval/evaluate_math_vllm_pass8.py \
      --model_dir "$MODEL" \
      --eval_file "$TEACHER_EVAL" \
      --out_jsonl "$OUT_DIR/v7_ckpt${STEP}_teacher_16k.jsonl" \
      --num_completions 8 \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 16384 \
      --max_model_len 40960 \
      --gpu_memory_utilization 0.90 \
      2>&1 | tee "$OUT_DIR/v7_ckpt${STEP}_teacher_16k.log"
  else
    echo
    echo "Missing model: $MODEL"
  fi
done
