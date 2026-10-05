"""
Evaluate a merged multilingual model using m3's evaluate/ package.

Loads the model via vLLM, generates completions, saves a generations JSONL,
then scores with the same boxed-answer extraction the CI uses.

Usage:
  python code/multilinguality/scripts/evaluate.py --path baseline
  python code/multilinguality/scripts/evaluate.py --path sft_grpo --full-eval
  python code/multilinguality/scripts/evaluate.py --path my_experiment --full-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force vLLM V0 engine — V1 uses subprocess IPC that hangs on Kubernetes pods
# ("Engine core initialization failed. Failed core proc(s): {}")
os.environ.setdefault("VLLM_USE_V1", "0")

ML_ROOT = Path(__file__).resolve().parents[1]   # code/multilinguality/
CODE_ROOT = Path(__file__).resolve().parents[2]  # code/
REPO_ROOT = Path(__file__).resolve().parents[3]  # standard-project-m3-camykaz/
sys.path.insert(0, str(CODE_ROOT))

CONFIG_PATH = ML_ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

from vllm import LLM, SamplingParams
from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer
from evaluate.pass_at_k import compute_pass_at_k_for_dataset
from multilinguality.scripts.utils import GRPO_PATHS, SFT_PATHS, with_lang, with_thinking

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

SYSTEM_PROMPT     = CFG["prompts"]["inference_system"]
PATHS_CFG         = CFG["paths"]
ALL_PATHS         = ("baseline",) + SFT_PATHS + GRPO_PATHS
EVAL_METHOD       = "boxed"

DEFAULT_EVAL_FILE = REPO_ROOT / "validation_samples" / "multilingual.jsonl"
FULL_EVAL_FILE    = ml_path("multilinguality/data/eval.jsonl")


# ── prompt building ───────────────────────────────────────────────────────────

def build_prompt(raw_prompt: str, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": raw_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ── generation ────────────────────────────────────────────────────────────────

def generate(samples: list[dict], model_path: str) -> list[str]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [build_prompt(s["prompt"], tokenizer) for s in samples]

    llm = LLM(model=model_path, dtype="bfloat16", trust_remote_code=True,
              max_model_len=32768, gpu_memory_utilization=0.8, enforce_eager=True)
    sp  = SamplingParams(max_tokens=4096, temperature=0.6, top_p=0.95, top_k=20)
    outputs = llm.generate(prompts, sp)
    return [o.outputs[0].text for o in outputs]


# ── scoring ───────────────────────────────────────────────────────────────────

def score_completions(samples: list[dict], completions: list[str]) -> dict:
    per_problem_correct: list[int] = []
    detailed: list[dict] = []
    for sample, completion in zip(samples, completions):
        reference = sample["answer"].strip().upper()
        extracted = extract_benchmark_answer(completion, EVAL_METHOD, reference)
        correct   = int(is_correct_benchmark_answer(extracted, reference, EVAL_METHOD))
        per_problem_correct.append(correct)
        detailed.append({
            "prompt":    sample["prompt"],
            "reference": reference,
            "extracted": extracted,
            "correct":   correct,
            "language":  sample.get("language", ""),
            "source":    sample.get("source", ""),
        })
    metrics = compute_pass_at_k_for_dataset(per_problem_correct, n=1, k_values=[1])
    return {"metrics": metrics, "detailed": detailed}


def print_summary(result: dict, full_eval: bool) -> None:
    metrics  = result["metrics"]
    detailed = result["detailed"]
    total    = len(detailed)
    correct  = sum(d["correct"] for d in detailed)
    print(f"\nOverall : {correct}/{total} = {correct/total*100:.1f}%  "
          f"pass@1={metrics.get('pass@1', 0):.4f}")

    by_lang: dict[str, list] = {}
    for d in detailed:
        by_lang.setdefault(d.get("language", "?"), []).append(d)
    if len(by_lang) > 1:
        print("\nPer-language accuracy:")
        for lang, items in sorted(by_lang.items()):
            n = sum(i["correct"] for i in items)
            print(f"  {lang:<10} {n}/{len(items)} = {n/len(items)*100:.1f}%")

    if full_eval:
        by_source: dict[str, list] = {}
        for d in detailed:
            by_source.setdefault(d.get("source", "?"), []).append(d)
        print("\nPer-source accuracy:")
        for src, items in sorted(by_source.items()):
            n = sum(i["correct"] for i in items)
            print(f"  {src:<22} {n}/{len(items)} = {n/len(items)*100:.1f}%")

    no_box = sum(1 for d in detailed if d["extracted"] is None)
    print(f"\n\\boxed{{}} failure: {no_box}/{total} = {no_box/total*100:.1f}%")


def save_outputs(samples: list[dict], completions: list[str], result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "prompt":      s["prompt"],
            "answer":      s["answer"],
            "language":    s.get("language", ""),
            "source":      s.get("source", ""),
            "completions": [c],
        }
        for s, c in zip(samples, completions)
    ]
    gen_path = out_dir / "generations.jsonl"
    with open(gen_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generations → {gen_path}")

    scores_path = out_dir / "scores.json"
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Scores     → {scores_path}")


# ── model resolution ─────────────────────────────────────────────────────────

def resolve_model_dir(args) -> str:
    path_cfg = PATHS_CFG.get(args.path)
    if path_cfg is not None and "merged_dir" in path_cfg:
        lang_for_path = (
            args.language
            if args.path in GRPO_PATHS or args.path == "distillation"
            else "english"
        )
        return str(ml_path(with_thinking(
            with_lang(path_cfg["merged_dir"], lang_for_path),
            not args.thinking_off,
        )))
    elif path_cfg is not None and "base_model" in path_cfg:
        return path_cfg["base_model"]
    else:
        return str(ml_path(f"multilinguality/outputs/{args.path}_merged"))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="Model path. Known: " + ", ".join(ALL_PATHS) +
                             ". Or any custom folder name under outputs/.")
    parser.add_argument("--full-eval", action="store_true",
                        help="Use the full ~500-question eval set (default: 10-question CI file).")
    parser.add_argument("--language", choices=["english", "multi"], default="english",
                        help="Must match the --language used during training.")
    parser.add_argument("--thinking-off", action="store_true",
                        help="Evaluate the _toff merged variant. Default: _ton.")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save generations + scores JSON.")
    args = parser.parse_args()

    eval_file = FULL_EVAL_FILE if args.full_eval else DEFAULT_EVAL_FILE
    if not eval_file.exists():
        print(f"Eval file not found: {eval_file}")
        if args.full_eval:
            print("Run: python code/multilinguality/scripts/prepare_multilingual_eval.py")
        sys.exit(1)

    model_dir = resolve_model_dir(args)
    print(f"Model : {model_dir}")
    print(f"Eval  : {eval_file}")

    samples = [json.loads(l) for l in eval_file.open(encoding="utf-8")]
    print(f"Loaded {len(samples)} examples.\n")

    completions = generate(samples, model_dir)
    result = score_completions(samples, completions)
    print_summary(result, args.full_eval)

    suffix  = "full" if args.full_eval else "ci"
    out_dir = Path(args.output_dir) if args.output_dir else ml_path(
        f"multilinguality/outputs/eval_{args.path}_{suffix}"
    )
    save_outputs(samples, completions, result, out_dir)


if __name__ == "__main__":
    main()
