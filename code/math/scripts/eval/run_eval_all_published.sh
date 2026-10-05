#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

EXPERIMENTS=(v1 v2 v3 v4_strong_sft v4_grpo v5 v6 v7)

for experiment in "${EXPERIMENTS[@]}"; do
  echo
  echo "===== Evaluating published math model: $experiment ====="
  python math/scripts/eval/evaluate_math_vllm_pass8.py --experiment "$experiment" "$@"
done
