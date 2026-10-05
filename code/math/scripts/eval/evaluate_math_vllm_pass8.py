import argparse
import json
import re
import csv
import sys
from pathlib import Path
from collections import Counter

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import download_final_evaluation

EXPERIMENT_CONFIG = ROOT_DIR / "math" / "configs" / "experiments.json"


def extract_boxed(text):
    if text is None:
        return None

    # Prefer last \boxed{...}
    matches = []
    i = 0
    while True:
        j = text.find("\\boxed", i)
        if j == -1:
            break
        k = text.find("{", j)
        if k == -1:
            break
        depth = 0
        end = None
        for idx in range(k, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end is not None:
            matches.append(text[k+1:end])
            i = end + 1
        else:
            break

    if matches:
        return matches[-1].strip()

    # fallback final answer patterns
    m = re.search(r"final answer\s*[:=]\s*([^\n]+)", text, flags=re.I)
    if m:
        return m.group(1).strip()

    return None


def normalize_answer(x):
    if x is None:
        return None
    x = str(x).strip()

    # remove common wrappers
    x = x.replace("\\left", "").replace("\\right", "")
    x = x.replace("$", "")
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
    x = x.strip()

    # remove spaces
    x = re.sub(r"\s+", "", x)

    # normalize simple punctuation
    x = x.rstrip(".,")

    # normalize leading decimal .356 -> 0.356
    if x.startswith("."):
        x = "0" + x

    # normalize \frac{a}{b} vs a/b only lightly
    x = x.replace("\\dfrac", "\\frac")
    x = x.replace("\\tfrac", "\\frac")

    return x


def get_expected(row):
    meta = row.get("metadata", {})
    for k in ["answer", "expected", "final_answer", "target", "reference"]:
        if k in meta and meta[k] not in [None, ""]:
            return meta[k]

    for k in ["answer", "expected", "final_answer", "target", "reference"]:
        if k in row and row[k] not in [None, ""]:
            return row[k]

    # If assistant solution exists in eval row, try boxed from it
    for m in row.get("messages", []):
        if m.get("role") == "assistant":
            b = extract_boxed(m.get("content", ""))
            if b:
                return b

    return None


def get_prompt_messages(row):
    messages = row.get("messages", [])
    return [m for m in messages if m.get("role") != "assistant"]


def resolve_repo_path(path_value):
    path = Path(path_value)
    if path.exists() or path.is_absolute():
        return path
    return ROOT_DIR / path


def resolve_experiment(args):
    if not args.experiment:
        return

    config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    try:
        experiment = config["experiments"][args.experiment]
    except KeyError as exc:
        choices = ", ".join(sorted(config["experiments"]))
        raise ValueError(f"Unknown experiment {args.experiment!r}. Choose one of: {choices}") from exc

    args.hf_repo = config["model_repo"]
    args.hf_subfolder = experiment["hf_subfolder"]
    if args.max_tokens is None:
        args.max_tokens = experiment["eval_max_tokens"]
    if args.max_model_len is None:
        args.max_model_len = experiment["eval_max_model_len"]
    if not args.out_jsonl:
        args.out_jsonl = f"math/outputs/published_eval/{args.experiment}.jsonl"
    print(f"Experiment: {args.experiment} - {experiment['description']}")
    print(f"Evaluation budget: max_tokens={args.max_tokens}, max_model_len={args.max_model_len}")


def resolve_model_dir(args):
    if args.model_dir:
        return args.model_dir

    patterns = [f"{args.hf_subfolder}/*", f"{args.hf_subfolder}/**/*"] if args.hf_subfolder else None
    snapshot_dir = Path(snapshot_download(
        repo_id=args.hf_repo,
        revision=args.hf_revision,
        allow_patterns=patterns,
    ))
    model_dir = snapshot_dir / args.hf_subfolder if args.hf_subfolder else snapshot_dir
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"No config.json found for Hugging Face model at {model_dir}. "
            "Check --hf_repo and --hf_subfolder."
        )

    print(f"Downloaded Hugging Face model: {args.hf_repo}")
    if args.hf_subfolder:
        print(f"Using Hugging Face model subfolder: {args.hf_subfolder}")
    print(f"Resolved model directory: {model_dir}")
    return str(model_dir)


def main():
    ap = argparse.ArgumentParser()
    model_source = ap.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--model_dir", help="Local model directory or root-level Hugging Face model ID.")
    model_source.add_argument("--hf_repo", help="Hugging Face model repository containing the submitted model.")
    model_source.add_argument("--experiment", help="Published experiment name from math/configs/experiments.json, for example v7.")
    ap.add_argument("--hf_subfolder", help="Model subfolder inside --hf_repo, for example v7.")
    ap.add_argument("--hf_revision", default="main", help="Hugging Face revision to download. Default: main.")
    ap.add_argument(
        "--chat_template_file",
        help="Optional Jinja chat template to apply instead of the model tokenizer template.",
    )
    ap.add_argument("--eval_file")
    ap.add_argument("--out_jsonl", help="Output JSONL path. Defaults by experiment when --experiment is used.")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--num_completions", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=None, help="Generation-token budget. Uses the experiment profile when omitted.")
    ap.add_argument("--max_model_len", type=int, default=None, help="vLLM context length. Uses the experiment profile when omitted.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    args = ap.parse_args()

    resolve_experiment(args)

    if args.max_tokens is None:
        args.max_tokens = 4096
    if args.max_model_len is None:
        args.max_model_len = 16384

    if not args.out_jsonl:
        ap.error("--out_jsonl is required unless --experiment supplies a default")

    if args.hf_subfolder and not args.hf_repo:
        ap.error("--hf_subfolder requires --hf_repo")

    model_dir = resolve_model_dir(args)

    if args.eval_file:
        eval_path = resolve_repo_path(args.eval_file)
        print(f"Using local evaluation file: {eval_path}")
    else:
        eval_path = Path(download_final_evaluation())
        print(f"Downloaded Hugging Face evaluation file: {eval_path}")
    rows = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if args.chat_template_file:
        template_path = resolve_repo_path(args.chat_template_file)
        if not template_path.exists():
            raise FileNotFoundError(f"Chat template file does not exist: {template_path}")
        tok.chat_template = template_path.read_text(encoding="utf-8")
        print(f"Using chat template: {template_path}")


    prompts = []
    kept = []
    for row in rows:
        expected = get_expected(row)
        if expected is None:
            continue

        messages = get_prompt_messages(row)
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)
        kept.append((row, expected))

    print(f"Loaded {len(kept)} evaluable examples from {eval_path}")

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )

    sampling = SamplingParams(
        n=args.num_completions,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    outs = llm.generate(prompts, sampling)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total = 0
    none_count = 0
    source_counter = Counter()
    source_correct = Counter()

    with out_path.open("w", encoding="utf-8") as f:
        for (row, expected), out in zip(kept, outs):
            meta = row.get("metadata", {})
            source = meta.get("benchmark_eval_source") or meta.get("source") or meta.get("dataset") or "unknown"
            source_counter[source] += 1

            exp_norm = normalize_answer(expected)

            completions = [o.text for o in out.outputs]
            extracted = [extract_boxed(c) for c in completions]
            extracted_norm = [normalize_answer(x) for x in extracted]

            if all(x is None for x in extracted):
                none_count += 1

            ok = any(x == exp_norm for x in extracted_norm if x is not None)

            total += 1
            if ok:
                correct += 1
                source_correct[source] += 1

            rec = {
                "source": source,
                "expected": expected,
                "expected_norm": exp_norm,
                "extracted": extracted,
                "extracted_norm": extracted_norm,
                "pass8": ok,
                "metadata": meta,
                "first_completion": completions[0] if completions else "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print("=" * 80)
            print("Sample", total, "Source:", source)
            print("Expected:", expected)
            print("Extracted:", extracted)
            print("Pass@8:", ok)

    score = correct / total if total else 0.0
    print("=" * 100)
    print(f"Pass@{args.num_completions}: {correct}/{total} = {score:.4f}")
    print("All-none examples:", none_count)
    print("By source:")
    for s, n in source_counter.most_common():
        c = source_correct[s]
        print(f"  {s}: {c}/{n} = {c/n:.4f}")

    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "experiment": args.experiment,
        "model_dir": model_dir,
        "hf_repo": args.hf_repo,
        "hf_subfolder": args.hf_subfolder,
        "hf_revision": args.hf_revision if args.hf_repo else None,
        "chat_template_file": args.chat_template_file,
        "eval_file": str(eval_path),
        "pass_at_n": args.num_completions,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "correct": correct,
        "total": total,
        "score": score,
        "all_none_examples": none_count,
        "by_source": {
            s: {
                "correct": source_correct[s],
                "total": n,
                "score": source_correct[s] / n if n else 0.0,
            }
            for s, n in source_counter.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print("Wrote:", out_path)
    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
