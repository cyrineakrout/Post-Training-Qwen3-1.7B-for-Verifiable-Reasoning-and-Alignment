import argparse
import re
import sys
from pathlib import Path

GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
CODE_ROOT = GENERAL_KNOWLEDGE_ROOT.parent  # code/ — contains the evaluate/ package
for _p in [str(GENERAL_KNOWLEDGE_ROOT / "configs"), str(GENERAL_KNOWLEDGE_ROOT), str(SCRIPTS_DIR), str(CODE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config_from_args
from eval_utils import (
    build_run_name,
    flatten_score_metrics,
    generate_completion_rows,
    generate_completion_rows_vllm,
    load_jsonl,
    log_wandb_eval_artifacts,
    save_json,
    save_jsonl,
    score_generation_rows,
)
from evaluate.extract_answer import extract_boxed_answer
from train_sft import force_default_system_prompt, force_qwen_thinking_mode


def normalize_choice_answer(answer):
    if answer is None:
        return None

    answer = str(answer).strip()
    boxed_answer = extract_boxed_answer(answer, strip_double_curly_brace=True)

    if boxed_answer is not None:
        answer = boxed_answer.strip()

    answer = answer.upper()
    match = re.search(r"[A-Z]", answer)

    if match:
        return match.group(0)

    return answer


def extract_fallback_choice(response):
    patterns = [
        r"(?im)^\s*(?:final\s+answer|answer)\s*[:\-]\s*\**\(?([A-Z])\)?\**\b",
        r"(?i)\b(?:final|correct)\s+answer\s+is\s*(?:option\s*)?\**\(?([A-Z])\)?\**\b",
        r"(?i)\bcorrect\s+option\s+is\s*\**\(?([A-Z])\)?\**\b",
        r"(?is)\A\s*\**\(?([A-Z])\)?\**\s*[\).:-]",
    ]

    for pattern in patterns:
        match = re.search(pattern, response)

        if match:
            return match.group(1)

    return None


def extract_debug_prediction(response):
    boxed_raw = extract_boxed_answer(response, strip_double_curly_brace=True)

    if boxed_raw is not None:
        return boxed_raw, "boxed"

    fallback_raw = extract_fallback_choice(response)

    if fallback_raw is not None:
        return fallback_raw, "fallback"

    return None, "none"


def parse_args():
    config_path, config = load_config_from_args()
    model_config = config["model"]
    data_config = config["data"]
    lora_config = config["lora"]
    generation_config = config["generation"]
    wandb_config = config["wandb"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the general knowledge config JSON file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "lora"],
        default="lora",
        help="Evaluate the base model or a LoRA adapter.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=model_config.get("hf_merged_model", model_config["base_model"]),
        help="HF repo ID or local path to the merged model.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=model_config.get("hf_merged_subfolder"),
        help="Subfolder inside the HF repo where the merged model lives (e.g. models/qwen3_gk_v12b).",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=lora_config["adapter_path"],
        help="Path to the LoRA adapter. Used only with --model_type lora.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=data_config["validation_path"],
        help="Path to benchmark-style validation JSONL.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Where to save generated completions. Defaults depend on model type.",
    )
    parser.add_argument(
        "--scores_path",
        type=str,
        default=None,
        help="Where to save detailed scores. Defaults depend on model type.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Generation token budget. Defaults depend on model type.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=generation_config["temperature"],
        help="Use 0.0 for deterministic evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation during evaluation.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for fast batched inference. Only supported for merged/base models.",
    )

    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=8192,
        help="Maximum context length for vLLM. Use 8192 for max_new_tokens=4096.",
    )

    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.90,
        help="GPU memory utilization for vLLM.",
    )

    parser.add_argument(
        "--vllm_dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
        help="dtype for vLLM inference.",
    )
    parser.add_argument("--top_p", type=float, default=generation_config["top_p"])
    parser.add_argument("--top_k", type=int, default=generation_config.get("top_k", 20))
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=generation_config.get("repetition_penalty", 1.0),
        help="Repetition penalty for vLLM generation.",
    )
    parser.add_argument(
        "--score_generations",
        action="store_true",
        default=generation_config["score_generations"],
        help="Score saved generations with the knowledge benchmark scorer.",
    )
    parser.add_argument(
        "--no_score_generations",
        action="store_false",
        dest="score_generations",
        help="Skip scoring saved generations.",
    )
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        default=generation_config["log_wandb"],
        help="Log generation metrics and artifacts to W&B.",
    )
    parser.add_argument(
        "--no_log_wandb",
        action="store_false",
        dest="log_wandb",
        help="Disable W&B logging for this evaluation run.",
    )
    parser.add_argument("--wandb_project", type=str, default=wandb_config["project"])
    parser.add_argument("--wandb_entity", type=str, default=wandb_config["entity"])
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name. Defaults to a descriptive generated name.",
    )
    parser.add_argument(
        "--inject_system_prompt",
        action="store_true",
        default=model_config.get("inject_system_prompt", False),
        help=(
            "Inject the generation_system_prompt into the tokenizer template before eval. "
            "Use for baseline (unfinetuned) models that have no baked-in system prompt."
        ),
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=10,
        help="Number of scored examples to print in the terminal.",
    )

    return parser.parse_args()


def defaults_for_model_type(args, generation_config):
    if args.model_type == "base":
        output_path = args.output_path or generation_config["base_output_path"]
        scores_path = args.scores_path or generation_config["base_scores_path"]
        max_new_tokens = args.max_new_tokens or generation_config["base_max_new_tokens"]
    else:
        output_path = args.output_path or generation_config["lora_output_path"]
        scores_path = args.scores_path or generation_config["lora_scores_path"]
        max_new_tokens = args.max_new_tokens or generation_config["max_new_tokens"]

    return output_path, scores_path, max_new_tokens


def resolve_model_path(base_model: str, subfolder) -> str:
    """
    If subfolder is set, download that subfolder from the HF repo to a local
    cache dir and return the local path.  vLLM and from_pretrained (without
    subfolder kwarg) can then both point at it without extra config.
    """
    if not subfolder:
        return base_model
    import os
    local_dir = os.path.join(
        "/tmp", f"gk_{base_model.replace('/', '_')}_{subfolder.replace('/', '_')}"
    )
    print(f"Downloading {base_model}/{subfolder} from HF Hub ...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=base_model,
        allow_patterns=[f"{subfolder}/*", f"{subfolder}/**/*"],
        local_dir=local_dir,
    )
    return os.path.join(local_dir, subfolder)


def load_model_and_tokenizer(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os
    # Subfolder only applies when evaluating an HF-hosted merged model (--model_type base).
    # For LoRA eval the base is Qwen/Qwen3-1.7B (no subfolder), and for locally trained
    # merged models the path is already a local directory (no HF download needed).
    subfolder = None
    if args.model_type == "base" and not os.path.isdir(args.base_model):
        subfolder = getattr(args, "subfolder", None)
    model_path = resolve_model_path(args.base_model, subfolder)

    if args.model_type == "base":
        print(f"Loading tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        print(f"Loading tokenizer from adapter: {args.adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "auto"
        device = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = None
        device = "mps"
    else:
        dtype = torch.float32
        device_map = "auto"
        device = "cpu"

    print(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=args.model_type == "lora",
    )

    if device_map is None:
        model.to(device)

    if args.model_type == "lora":
        from peft import PeftModel

        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def print_scored_examples(rows, score_result, max_examples):
    if max_examples <= 0:
        return

    detailed_results = score_result["detailed_results"]
    n_examples = min(max_examples, len(rows), len(detailed_results))

    if n_examples == 0:
        return

    print("=" * 100)
    print(f"Showing first {n_examples} scored examples")

    for i in range(n_examples):
        row = rows[i]
        detail = detailed_results[i]
        completion = row["completions"][0].strip()
        gold = normalize_choice_answer(detail["reference"])
        raw_pred, pred_source = extract_debug_prediction(completion)
        pred = normalize_choice_answer(raw_pred)
        correct = pred == gold

        print("=" * 100)
        print(f"Example {i + 1}/{len(rows)}")
        print(f"Gold: {detail['reference']}")
        print(f"Pred: {pred}")
        print(f"Correct: {correct}")
        print(f"Source: {pred_source}")
        print("\nResponse:")
        print(completion)


def print_eval_summary(rows, score_result, model_type, base_model, output_path):
    detailed_results = score_result["detailed_results"]
    total = min(len(rows), len(detailed_results))
    boxed_correct = 0
    fallback_correct = 0
    boxed_parse_failures = 0
    fallback_parse_failures = 0

    for row, detail in zip(rows, detailed_results):
        completion = row["completions"][0]
        gold = normalize_choice_answer(detail["reference"])
        boxed_raw = extract_boxed_answer(completion, strip_double_curly_brace=True)
        boxed_pred = normalize_choice_answer(boxed_raw)
        raw_pred, _ = extract_debug_prediction(completion)
        pred = normalize_choice_answer(raw_pred)

        boxed_correct += int(boxed_pred == gold)
        fallback_correct += int(pred == gold)

        if boxed_raw is None:
            boxed_parse_failures += 1

        if raw_pred is None:
            fallback_parse_failures += 1

    boxed_accuracy = boxed_correct / total if total else 0.0
    fallback_accuracy = fallback_correct / total if total else 0.0
    pass_at_1 = score_result["metrics"].get("pass@1", 0.0)

    print("\n" + "=" * 100)
    print("GENERAL KNOWLEDGE EVALUATION RESULTS")
    print(f"Model type: {model_type}")
    print(f"Base model: {base_model}")
    print(f"Official pass@1: {pass_at_1:.2%}")
    print(f"Fallback correct: {fallback_correct}/{total}")
    print(f"Fallback accuracy: {fallback_accuracy:.2%}")
    print(f"Boxed correct: {boxed_correct}/{total}")
    print(f"Boxed accuracy: {boxed_accuracy:.2%}")
    print(f"Boxed parse failures: {boxed_parse_failures}/{total}")
    print(f"Fallback parse failures: {fallback_parse_failures}/{total}")
    print(f"Saved generations to: {output_path}")
def print_source_breakdown(rows, score_result):
    """Print per-source accuracy. Requires `source` in each row."""
    from collections import defaultdict

    detailed = score_result["detailed_results"]
    correct_by_source = defaultdict(lambda: [0, 0])
    parse_fail_by_source = defaultdict(int)

    for row, detail in zip(rows, detailed):
        src = row.get("source", "unknown")
        correct_by_source[src][1] += 1
        comps = detail.get("completions", [])
        any_correct = any(c.get("correct", False) for c in comps)
        if any_correct:
            correct_by_source[src][0] += 1
        # parse fail = no extracted answer
        if comps and all(c.get("extracted") is None for c in comps):
            parse_fail_by_source[src] += 1

    print("\n" + "=" * 100)
    print("PER-SOURCE BREAKDOWN")
    print("=" * 100)

    # Top-level grouping (mmlu, mmlu_pro, gpqa, etc.)
    top_level = defaultdict(lambda: [0, 0, 0])  # [correct, total, parse_fails]
    for src, (c, t) in correct_by_source.items():
        if src.startswith("mmlu_pro"):
            prefix = "mmlu_pro"
        elif src.startswith("mmlu_"):
            prefix = "mmlu"
        elif src.startswith("gpqa"):
            prefix = "gpqa"
        else:
            prefix = src.split("_")[0] if "_" in src else src
        top_level[prefix][0] += c
        top_level[prefix][1] += t
        top_level[prefix][2] += parse_fail_by_source.get(src, 0)

    print("\nBy top-level source:")
    for prefix, (c, t, pf) in sorted(top_level.items()):
        if t > 0:
            acc = c / t
            print(f"  {prefix:20s}  acc={acc:.4f}  ({c}/{t})  parse_fails={pf}")

    print("\nBy detailed source (sorted by total count):")
    sorted_sources = sorted(correct_by_source.items(), key=lambda kv: kv[1][1], reverse=True)
    for src, (c, t) in sorted_sources:
        if t > 0:
            acc = c / t
            pf = parse_fail_by_source.get(src, 0)
            print(f"  {src:60s}  acc={acc:.4f}  ({c}/{t})  parse_fails={pf}")

    print("=" * 100)

def main():
    args = parse_args()
    _, config = load_config_from_args()
    generation_config = config["generation"]
    prompt_config = config["prompts"]
    wandb_config = config["wandb"]

    output_path, scores_path, max_new_tokens = defaults_for_model_type(
        args,
        generation_config,
    )
    wandb_run = None
    if args.log_wandb:
        import wandb

        model_name_for_run = (
            args.base_model if args.model_type == "base" else args.adapter_path
        )
        run_name = args.run_name or build_run_name(
            task=f"{args.model_type}-eval",
            model_name=model_name_for_run,
            dataset_path=args.input_path,
            method="gk",
        )
        wandb_tags = list(wandb_config.get("tags", [])) + [
            "eval",
            args.model_type,
        ]

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=wandb_tags,
            job_type="eval",
            config={
                "run_type": "eval",
                "model_type": args.model_type,
                "base_model": args.base_model,
                "adapter_path": args.adapter_path if args.model_type == "lora" else None,
                "input_path": args.input_path,
                "output_path": output_path,
                "scores_path": scores_path,
                "max_new_tokens": max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "method": f"{args.model_type}_generation",
                "benchmark_method": "knowledge",
            },
        )


    # Auto-download the eval file from HF if the local path is missing.
    input_path = args.input_path
    if not Path(input_path).exists():
        data_config = config.get("data", {})
        hf_repo = data_config.get("hf_dataset_repo")
        hf_eval_file = data_config.get("hf_eval_file", "datasets/eval_mmlu_mmlupro_supergpqa_nemotron_provided.jsonl")
        if hf_repo:
            print(f"Eval file not found locally — downloading {hf_eval_file} from {hf_repo} ...")
            from huggingface_hub import hf_hub_download
            dest_dir = Path(input_path).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=hf_repo,
                filename=hf_eval_file,
                repo_type="model",
                local_dir=str(dest_dir.parent),
            )
            src = dest_dir.parent / hf_eval_file
            if src != Path(input_path):
                src.rename(input_path)
            print(f"Saved to {input_path}")
        else:
            raise FileNotFoundError(
                f"Eval file not found: {input_path}\n"
                "Run: python scripts/download_data.py"
            )

    examples = load_jsonl(input_path)

    if args.use_vllm:
        if args.model_type != "base":
            raise ValueError(
                "vLLM path only supports merged/base models. "
                "Merge the LoRA first, then run with --model_type base --use_vllm."
            )

        import os as _os
        _subfolder = getattr(args, "subfolder", None) if not _os.path.isdir(args.base_model) else None
        rows = generate_completion_rows_vllm(
            model_path=resolve_model_path(args.base_model, _subfolder),
            examples=examples,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            desc=f"Generating {args.model_type} completions with vLLM",
            max_model_len=args.vllm_max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            dtype=args.vllm_dtype,
            repetition_penalty=args.repetition_penalty,
            system_prompt=prompt_config["generation_system_prompt"] if args.inject_system_prompt else None,
        )

    else:
        model, tokenizer = load_model_and_tokenizer(args)

        rows = generate_completion_rows(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            desc=f"Generating {args.model_type} completions",
            batch_size=args.batch_size,
            repetition_penalty=args.repetition_penalty,
            system_prompt=prompt_config["generation_system_prompt"] if args.inject_system_prompt else None,
        )
    save_jsonl(rows, output_path)
    print(f"Saved generations to: {output_path}")

    score_result = None
    metrics_prefix = f"{args.model_type}_eval"
    if args.score_generations:
        score_result = score_generation_rows(rows, method="boxed")
        save_json(score_result, scores_path)
        metrics = flatten_score_metrics(score_result, prefix=metrics_prefix)
        print(f"{args.model_type.title()} eval metrics: {metrics}")
        print(f"Saved scores to: {scores_path}")
        print_scored_examples(rows, score_result, args.print_examples)
        print_eval_summary(
            rows=rows,
            score_result=score_result,
            model_type=args.model_type,
            base_model=args.base_model,
            output_path=output_path,
        )
        print_source_breakdown(rows, score_result)

    if args.log_wandb:
        import wandb

        if score_result is not None:
            wandb.log(flatten_score_metrics(score_result, prefix=metrics_prefix))

        log_wandb_eval_artifacts(
            wandb=wandb,
            artifact_name=f"{wandb.run.name}-generations",
            generations_path=output_path,
            scores_path=scores_path if score_result is not None else None,
        )

        wandb.finish()


if __name__ == "__main__":
    main()
