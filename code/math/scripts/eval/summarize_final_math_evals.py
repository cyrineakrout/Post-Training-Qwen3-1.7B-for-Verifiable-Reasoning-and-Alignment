import json
import csv
from pathlib import Path

OUT_DIR = Path("math/outputs/eval_summaries")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main benchmark = final_evaluation/math_benchmark_external_v1.jsonl in the shared HF dataset repo.
# Early models evaluated with 2k.
# Long-context v6/v7 evaluated with 12k.
items = [
    {
        "model": "Baseline Qwen3-1.7B",
        "version": "",
        "method": "Prompt only",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/baseline_external_v1_2k.summary.json",
        "official_ci": "",
        "notes": "Base model",
    },
    {
        "model": "Math v1 - Early SFT",
        "version": "v1",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/v1_external_v1_2k.summary.json",
        "official_ci": "",
        "notes": "Early SFT",
    },
    {
        "model": "Math v2 - Revised SFT",
        "version": "v2",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/v2_external_v1_2k.summary.json",
        "official_ci": "",
        "notes": "Revised SFT",
    },
    {
        "model": "Math v3 - Curated SFT",
        "version": "v3",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/v3_external_v1_2k.summary.json",
        "official_ci": "",
        "notes": "Curated SFT",
    },
    {
        "model": "Math v4 - Strong SFT",
        "version": "v4",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/v4_external_v1_2k.summary.json",
        "official_ci": "39%",
        "notes": "Strong SFT",
    },
    {
        "model": "Math v4 - GRPO reference",
        "version": "v4",
        "method": "SFT + GRPO",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/v4_grpo_ref_external_v1_2k.summary.json",
        "official_ci": "not selected",
        "notes": "GRPO refinement from v4",
    },
    {
        "model": "Math v5 - All-solutions SFT",
        "version": "v5",
        "method": "LoRA SFT r96 e0.8",
        "eval_setting": "external_v1_2k",
        "summary": "math/outputs/vllm_eval_external_v1/v5_all_solutions_sft_external_v1_2k.summary.json",
        "official_ci": "not pushed",
        "notes": "All-solutions training",
    },
    {
        "model": "Math v6 - 50k long",
        "version": "v6",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_12k",
        "summary": "math/outputs/vllm_eval_external_v1_12k/v6_external_v1_12k.summary.json",
        "official_ci": "~48%",
        "notes": "50k AceReason-style mix",
    },
    {
        "model": "Math v7 - 100k checkpoint 1600",
        "version": "v7",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_12k",
        "summary": "math/outputs/vllm_eval_external_v1_12k/v7_ckpt1600_external_v1_12k.summary.json",
        "official_ci": "52%",
        "notes": "100k mix, 1600 steps ≈ 0.51 epoch; current safe CI model",
    },
    {
        "model": "Math v7 - 100k checkpoint 2000",
        "version": "v7",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_12k",
        "summary": "math/outputs/vllm_eval_external_v1_12k/v7_ckpt2000_external_v1_12k.summary.json",
        "official_ci": "pending",
        "notes": "100k mix, 2000 steps ≈ 0.64 epoch",
    },
    {
        "model": "Math v7 - 100k checkpoint 2400",
        "version": "v7",
        "method": "LoRA SFT",
        "eval_setting": "external_v1_12k",
        "summary": "math/outputs/vllm_eval_external_v1_12k/v7_ckpt2400_external_v1_12k.summary.json",
        "official_ci": "pending",
        "notes": "100k mix, 2400 steps ≈ 0.77 epoch",
    },
]

rows = []

for item in items:
    p = Path(item["summary"])
    row = {
        "model": item["model"],
        "version": item["version"],
        "method": item["method"],
        "eval_setting": item["eval_setting"],
        "local_pass8": "",
        "count": "",
        "all_none": "",
        "eval_file": "",
        "pass_at_n": "",
        "official_ci": item["official_ci"],
        "status": "missing",
        "notes": item["notes"],
        "summary_path": item["summary"],
    }

    if p.exists():
        try:
            s = json.loads(p.read_text())
            row["local_pass8"] = f'{float(s.get("score", 0.0)):.4f}'
            row["count"] = f'{s.get("correct", "")}/{s.get("total", "")}'
            row["all_none"] = s.get("all_none_examples", "")
            row["eval_file"] = s.get("eval_file", "")
            row["pass_at_n"] = s.get("pass_at_n", "")
            row["status"] = "ok"
        except Exception as e:
            row["status"] = f"error: {e}"

    rows.append(row)

# Sort for display: keep intended version order, not score order.
csv_path = OUT_DIR / "final_math_results_summary.csv"
json_path = OUT_DIR / "final_math_results_summary.json"

with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "version",
            "method",
            "eval_setting",
            "local_pass8",
            "count",
            "all_none",
            "official_ci",
            "status",
            "notes",
            "eval_file",
            "pass_at_n",
            "summary_path",
        ],
    )
    w.writeheader()
    w.writerows(rows)

json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

print("Wrote:", csv_path)
print("Wrote:", json_path)

print("\n===== FINAL MATH RESULTS SUMMARY =====")
for r in rows:
    print(
        f'{r["version"]:15s} | {r["eval_setting"]:16s} | '
        f'{r["local_pass8"]:>7s} | {r["count"]:>8s} | '
        f'all_none={str(r["all_none"]):>3s} | CI={r["official_ci"]} | {r["status"]}'
    )

print("\n===== BEST AVAILABLE BY LOCAL PASS@8 =====")
ok_rows = [r for r in rows if r["status"] == "ok" and r["local_pass8"]]
for r in sorted(ok_rows, key=lambda x: float(x["local_pass8"]), reverse=True):
    print(
        f'{r["version"]:15s} | {r["local_pass8"]} | {r["count"]} | '
        f'all_none={r["all_none"]} | {r["eval_setting"]}'
    )
