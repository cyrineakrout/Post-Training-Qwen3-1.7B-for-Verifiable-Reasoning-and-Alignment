import json
import csv
from pathlib import Path

# Collect all summary files that may exist
summary_files = {
    # External benchmark v1, 2k versions
    "baseline_2k": "math/outputs/vllm_eval_external_v1/baseline_external_v1_2k.summary.json",
    "v1_2k": "math/outputs/vllm_eval_external_v1/v1_external_v1_2k.summary.json",
    "v2_2k": "math/outputs/vllm_eval_external_v1/v2_external_v1_2k.summary.json",
    "v3_2k": "math/outputs/vllm_eval_external_v1/v3_external_v1_2k.summary.json",
    "v4_2k": "math/outputs/vllm_eval_external_v1/v4_external_v1_2k.summary.json",
    "v4_grpo_ref_2k": "math/outputs/vllm_eval_external_v1/v4_grpo_ref_external_v1_2k.summary.json",
    "v5_all_solutions_sft_2k": "math/outputs/vllm_eval_external_v1/v5_all_solutions_sft_external_v1_2k.summary.json",

    # External benchmark v1, longer versions
    "v6_8k": "math/outputs/vllm_eval_external_v1/v6_external_v1_8k.summary.json",
    "v6_12k": "math/outputs/vllm_eval_external_v1_12k/v6_external_v1_12k.summary.json",
    "v7_ckpt1600_8k": "math/outputs/vllm_eval_external_v1/v7_ckpt1600_external_v1_8k.summary.json",
    "v7_ckpt1600_12k": "math/outputs/vllm_eval_external_v1_12k/v7_ckpt1600_external_v1_12k.summary.json",
    "v7_ckpt2000_8k": "math/outputs/vllm_eval_external_v1/v7_ckpt2000_external_v1_8k.summary.json",
    "v7_ckpt2400_8k": "math/outputs/vllm_eval_external_v1/v7_ckpt2400_external_v1_8k.summary.json",

    # Random50 16k checkpoint selection
    "v6_random50_16k": "math/outputs/vllm_eval_random50_16k/v6_random50_16k.summary.json",
    "v7_ckpt1600_random50_16k": "math/outputs/vllm_eval_random50_16k/v7_ckpt1600_random50_16k.summary.json",
    "v7_ckpt2000_random50_16k": "math/outputs/vllm_eval_late_ckpts_random50/v7_ckpt2000_random50_16k.summary.json",
    "v7_ckpt2400_random50_16k": "math/outputs/vllm_eval_late_ckpts_random50/v7_ckpt2400_random50_16k.summary.json",

    # Teacher validation converted
    "v7_ckpt1600_teacher16k": "math/outputs/vllm_eval_teacher_v7_late/v7_ckpt1600_teacher_16k.summary.json",
    "v7_ckpt2000_teacher16k": "math/outputs/vllm_eval_teacher_v7_late/v7_ckpt2000_teacher_16k.summary.json",
    "v7_ckpt2400_teacher16k": "math/outputs/vllm_eval_teacher_v7_late/v7_ckpt2400_teacher_16k.summary.json",
}

rows = []

for name, path_str in summary_files.items():
    path = Path(path_str)

    if not path.exists():
        rows.append({
            "model_eval": name,
            "score": "",
            "count": "",
            "all_none": "",
            "eval_file": "",
            "pass_at_n": "",
            "status": "missing",
            "path": path_str,
        })
        continue

    try:
        s = json.loads(path.read_text())
        correct = s.get("correct", "")
        total = s.get("total", "")
        score = s.get("score", "")
        score_str = f"{float(score):.4f}" if score != "" else ""

        rows.append({
            "model_eval": name,
            "score": score_str,
            "count": f"{correct}/{total}",
            "all_none": s.get("all_none_examples", ""),
            "eval_file": s.get("eval_file", ""),
            "pass_at_n": s.get("pass_at_n", ""),
            "status": "ok",
            "path": path_str,
        })
    except Exception as e:
        rows.append({
            "model_eval": name,
            "score": "",
            "count": "",
            "all_none": "",
            "eval_file": "",
            "pass_at_n": "",
            "status": f"error: {e}",
            "path": path_str,
        })

# Preserve the declared experiment trajectory instead of sorting by score.
rows_sorted = rows

out_dir = Path("math/outputs/eval_summaries")
out_dir.mkdir(parents=True, exist_ok=True)

csv_path = out_dir / "summary_all_math_evals.csv"
json_path = out_dir / "summary_all_math_evals.json"

with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "model_eval",
            "score",
            "count",
            "all_none",
            "eval_file",
            "pass_at_n",
            "status",
            "path",
        ],
    )
    w.writeheader()
    w.writerows(rows_sorted)

json_path.write_text(json.dumps(rows_sorted, indent=2, ensure_ascii=False) + "\n")

print("Wrote:", csv_path)
print("Wrote:", json_path)
print()
print("===== AVAILABLE RESULTS =====")
for r in rows_sorted:
    if r["status"] == "ok":
        print(f'{r["model_eval"]:35s}  {r["score"]:>7s}  {r["count"]:>8s}  all_none={r["all_none"]}')

print()
print("===== MISSING RESULTS =====")
for r in rows_sorted:
    if r["status"] != "ok":
        print(f'{r["model_eval"]:35s}  {r["status"]}')
