import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_vllm_dtype() -> str:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def use_vllm() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_model(model_cfg: dict) -> str:
    """Return local model path, downloading from HF if not already present."""
    local = model_cfg.get("local_path", "")
    if local and Path(local).exists():
        print(f"Using local model: {local}")
        return local

    hf_repo = model_cfg.get("hf_repo")
    hf_sub  = model_cfg.get("hf_subfolder", "")
    if not hf_repo:
        raise FileNotFoundError(
            f"Model not found at '{local}' and no hf_repo specified in config."
        )

    print(f"Downloading model {hf_repo}/{hf_sub} ...")
    from huggingface_hub import snapshot_download

    patterns = [f"{hf_sub}/*", f"{hf_sub}/**/*"] if hf_sub else None
    tmp = Path(local).parent / ".hf_model_cache" if local else Path("models/.hf_model_cache")
    snapshot_download(repo_id=hf_repo, allow_patterns=patterns, local_dir=str(tmp))

    src = tmp / hf_sub if hf_sub else tmp
    dest = Path(local) if local else src

    if local and src != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(src), str(dest))
        shutil.rmtree(str(tmp), ignore_errors=True)

    print(f"Model ready at: {dest}")
    return str(dest)


def resolve_eval_file(data_cfg: dict) -> str:
    """Return local eval file path, downloading from HF if not already present."""
    local   = data_cfg.get("eval_path", "")
    if local and Path(local).exists():
        return local

    hf_repo = data_cfg.get("hf_repo")
    hf_file = data_cfg.get("hf_eval_file")
    if not hf_repo or not hf_file:
        raise FileNotFoundError(
            f"Eval file not found at '{local}' and no hf_repo/hf_eval_file in config."
        )

    print(f"Downloading eval file {hf_file} from {hf_repo} ...")
    from huggingface_hub import hf_hub_download

    dest = Path(local) if local else Path("data") / Path(hf_file).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=hf_repo, filename=hf_file, local_dir=str(dest.parent / ".hf_data_cache"))

    if Path(downloaded) != dest:
        shutil.copy2(downloaded, str(dest))

    print(f"Eval file ready at: {dest}")
    return str(dest)

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


DEFAULT_EVAL_FILE = Path(__file__).resolve().parents[1] / "data" / "group_eval.jsonl"
RESULTS_FILE = Path(__file__).resolve().parents[1] / "results.txt"

# Mirrors the system block hardcoded in chat_template.jinja (group_model_merged_runB_r64_e060).
# When apply_chat_template uses that jinja the system message in `messages` is skipped and
# the template injects this prompt itself; this constant is the fallback for other tokenizers.
SYSTEM_PROMPT = (
    "You are a careful reasoning assistant optimized for four benchmark types: "
    "mathematics, general knowledge, multilingual questions, and safety questions.\n\n"
    "Core behavior:\n"
    "- Read the full question and all answer choices before deciding.\n"
    "- First identify the task type: math, knowledge, multilingual, safety, or mixed reasoning.\n"
    "- For multiple-choice questions, choose the single best answer from the provided options.\n"
    "- For math questions, solve step by step, verify arithmetic, and give the exact final result when possible.\n"
    "- For general knowledge questions, rely on established facts and avoid guessing from irrelevant details.\n"
    "- For multilingual questions, understand the question in its original language and reason accordingly.\n"
    "- For safety questions, choose the safest, least harmful, most responsible, and most policy-compliant option.\n"
    "- If choices are provided, never invent a new answer outside the choices.\n\n"
    "Output format:\n"
    "- Always reason inside <think> ... </think>.\n"
    "- Always finish with exactly one final answer inside one LaTeX \\boxed{...}.\n"
    "- If answer choices are provided, put only the option letter inside the box, for example \\boxed{C}.\n"
    "- If the task is free-response math, put only the exact mathematical answer inside the box.\n"
    "- Do not put explanations, units, or extra words inside the box unless explicitly required.\n"
    "- Do not write anything after the boxed answer."
)

MINIMAL_SYSTEM_PROMPT = (
    "Answer the question. "
    "Always put your final answer inside \\boxed{}. "
    "For multiple-choice questions, put only the option letter inside the box, for example \\boxed{A}."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_num}: {exc}"
                ) from exc

    return rows


def normalize_text(text: Any) -> str:
    text = str(text)
    text = text.replace("<|im_end|>", "")
    text = text.replace("<|im_start|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_messages(prompt: str, no_system_prompt: bool = False, minimal_system_prompt: bool = False) -> list[dict[str, str]]:
    if no_system_prompt:
        return [{"role": "user", "content": str(prompt).strip()}]
    system = MINIMAL_SYSTEM_PROMPT if minimal_system_prompt else SYSTEM_PROMPT
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": str(prompt).strip()},
    ]


def get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def extract_last_boxed(text: str) ->Optional[ str]:
    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    chars = []

    while i < len(text) and depth > 0:
        char = text[i]

        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                chars.append(char)
        else:
            chars.append(char)

        i += 1

    if depth == 0:
        return "".join(chars).strip()

    return None


def extract_choice(text: str) ->Optional[ str]:
    text = normalize_text(text)

    patterns = [
        r"\\boxed\{\s*([A-J])\s*\}",
        r"\bfinal\s*answer\s*(?:is|:)?\s*([A-J])\b",
        r"\banswer\s*(?:is|:)?\s*([A-J])\b",
        r"^\s*([A-J])[\.\)]",
        r"^\s*([A-J])\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            return match.group(1).upper()

    return None


def normalize_answer(text: Any) -> str:
    text = normalize_text(text)

    boxed = extract_last_boxed(text)
    if boxed is not None:
        text = boxed

    choice = extract_choice(text)
    if choice is not None:
        return choice

    text = text.lower()

    replacements = {
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        " ": "",
        "$": "",
        ".": "",
        ",": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.strip()
    text = text.strip("{}")
    return text


def is_correct(prediction: str, reference: str) -> bool:
    pred_choice = extract_choice(prediction)
    ref_choice = extract_choice(reference)

    if ref_choice is not None:
        return pred_choice == ref_choice

    pred_boxed = extract_last_boxed(prediction)

    if pred_boxed is not None:
        pred_norm = normalize_answer(pred_boxed)
    else:
        pred_norm = normalize_answer(prediction)

    ref_norm = normalize_answer(reference)

    return pred_norm == ref_norm


def get_eval_source(row: dict[str, Any]) -> str:
    domain = row.get("_domain", "")

    src = row.get("source") or ""
    if not src:
        metadata = row.get("metadata")
        if isinstance(metadata, dict):
            src = str(metadata.get("source") or "")

    src = src.strip()
    if domain and src and src != "unknown":
        return f"{domain}/{src}"
    return domain or src or "unknown"


def prepare_eval_examples(examples: list[dict[str, Any]], tokenizer, no_system_prompt: bool = False, minimal_system_prompt: bool = False):
    prepared = []

    for ex in examples:
        if "prompt" in ex:
            prompt = ex["prompt"]
            answer = get_first(ex, ["answer", "reference", "target", "label"])

            if answer is None:
                raise ValueError(
                    "Each benchmark-style evaluation row must contain answer/reference/target/label."
                )

            messages = build_messages(str(prompt), no_system_prompt=no_system_prompt, minimal_system_prompt=minimal_system_prompt)

        elif "question" in ex:
            prompt = ex["question"]
            answer = get_first(ex, ["answer", "reference", "target", "label"])

            if answer is None:
                raise ValueError(
                    "Each question-style evaluation row must contain answer/reference/target/label."
                )

            messages = build_messages(str(prompt), no_system_prompt=no_system_prompt, minimal_system_prompt=minimal_system_prompt)

        elif "messages" in ex and "metadata" in ex:
            messages = ex["messages"][:-1]
            prompt = messages[-1]["content"]
            answer = ex["metadata"].get("answer")

            if answer is None:
                answer = ex["messages"][-1]["content"]

        elif "messages" in ex:
            messages = ex["messages"][:-1]
            prompt = messages[-1]["content"]
            answer = ex["messages"][-1]["content"]

        else:
            raise ValueError(f"Unsupported evaluation row format: {ex.keys()}")

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        domain = "unknown"
        if "metadata" in ex and isinstance(ex["metadata"], dict):
            domain = ex["metadata"].get("domain", "unknown")

        prepared.append(
            {
                "prompt": str(prompt),
                "answer": str(answer),
                "input_text": input_text,
                "source": get_eval_source(ex),
                "domain": domain,
            }
        )

    return prepared



@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
):
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    ).to(model.device)

    if do_sample:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    else:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_answers_vllm(
    llm,
    prepared: list[dict],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    math_n: int = 8,
) -> list[list[str]]:
    """Return a list of prediction lists. Math examples get `math_n` candidates (pass@k),
    all others get 1 candidate (pass@1)."""
    math_indices    = [i for i, ex in enumerate(prepared) if ex.get("domain") == "math"]
    nonmath_indices = [i for i, ex in enumerate(prepared) if ex.get("domain") != "math"]

    results: list[list[str]] = [[] for _ in prepared]

    def make_params(n):
        return SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else -1,
            repetition_penalty=repetition_penalty,
            n=n,
        )

    if nonmath_indices:
        texts = [prepared[i]["input_text"] for i in nonmath_indices]
        outputs = llm.generate(texts, make_params(1))
        for i, out in zip(nonmath_indices, outputs):
            results[i] = [out.outputs[0].text.strip()]

    if math_indices:
        texts = [prepared[i]["input_text"] for i in math_indices]
        outputs = llm.generate(texts, make_params(math_n))
        for i, out in zip(math_indices, outputs):
            results[i] = [o.text.strip() for o in out.outputs]

    return results


def save_results(args, correct: int, total: int, by_source: dict, model_dir: str, eval_file: str) -> None:
    from datetime import datetime

    overall_acc = correct / total if total else 0.0
    lines = []
    lines.append("=" * 70)
    lines.append(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Model      : {model_dir}")
    lines.append(f"Eval file  : {eval_file}")
    if args.limit:
        lines.append(f"Limit      : {args.limit}")
    if args.notes:
        lines.append(f"Notes      : {args.notes}")
    lines.append(f"Overall    : {correct}/{total} = {overall_acc*100:.1f}%")
    lines.append("\nPer-source accuracy:")
    for source, stats in sorted(by_source.items()):
        n, t = stats["correct"], stats["total"]
        lines.append(f"  {source:<40} {n}/{t} = {n/max(t,1)*100:.1f}%")
    lines.append("")
    with open(RESULTS_FILE, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nResults appended to {RESULTS_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None,
                        help="Path to experiment config JSON (e.g. configs/lora_direct_v1.json). "
                             "Downloads model and eval file from HF automatically if not local.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local path to merged model. Overrides config model path.")
    parser.add_argument("--notes", type=str, default=None,
                        help="Free-text notes appended to the results entry.")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="Path to eval JSONL. Overrides config eval path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--sample", action="store_true", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for faster batched inference (requires vllm package).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for vLLM tensor parallelism.",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Omit the system prompt entirely.",
    )
    parser.add_argument(
        "--minimal-system-prompt",
        action="store_true",
        help="Use a minimal system prompt (format instruction only, no task guidance).",
    )

    args = parser.parse_args()

    # Load config and apply defaults (CLI args override config values).
    cfg = load_config(args.config) if args.config else {}
    gen_cfg = cfg.get("generation", {})

    model_dir = args.model_dir or (resolve_model(cfg["model"]) if cfg.get("model") else None)
    if not model_dir:
        parser.error("Provide --model_dir or --config with a model.hf_repo entry.")

    eval_file = args.eval_file or (resolve_eval_file(cfg["data"]) if cfg.get("data") else str(DEFAULT_EVAL_FILE))
    max_new_tokens     = args.max_new_tokens     or gen_cfg.get("max_new_tokens",     16384)
    do_sample          = args.sample             if args.sample is not None else gen_cfg.get("do_sample", True)
    temperature        = args.temperature        if args.temperature is not None else gen_cfg.get("temperature", 0.6)
    top_p              = args.top_p              if args.top_p is not None else gen_cfg.get("top_p", 0.95)
    top_k              = args.top_k              if args.top_k is not None else gen_cfg.get("top_k", 20)
    repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else gen_cfg.get("repetition_penalty", 1.03)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = read_jsonl(Path(eval_file))
    print(f"Loaded {len(examples)} examples from {eval_file}")
    if args.limit is not None:
        examples = examples[: args.limit]
    prepared = prepare_eval_examples(examples, tokenizer, no_system_prompt=args.no_system_prompt, minimal_system_prompt=args.minimal_system_prompt)

    correct = 0
    total = 0
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})

    if args.use_vllm or use_vllm():
        print("Backend: vLLM")
        llm = LLM(
            model=model_dir,
            dtype=get_vllm_dtype(),
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        all_predictions = generate_answers_vllm(
            llm=llm,
            prepared=prepared,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        for idx, (ex, preds) in enumerate(zip(prepared, all_predictions), start=1):
            is_math = ex.get("domain") == "math"
            ok = any(is_correct(p, ex["answer"]) for p in preds)
            total += 1
            correct += int(ok)
            source = ex["source"]
            by_source[source]["total"] += 1
            by_source[source]["correct"] += int(ok)
            print("=" * 80)
            print(f"Sample {idx}  [{'pass@8' if is_math else 'pass@1'}]")
            print("Source:", source)
            print("Expected:", ex["answer"])
            print("Predicted:", normalize_text(preds[0])[:900])
            print("Correct:", ok)

    else:
        print("Backend: HuggingFace transformers")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=get_torch_dtype(),
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        for idx, ex in enumerate(prepared, start=1):
            pred = generate_answer(
                model=model,
                tokenizer=tokenizer,
                input_text=ex["input_text"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

            ok = is_correct(pred, ex["answer"])
            total += 1
            correct += int(ok)
            source = ex["source"]
            by_source[source]["total"] += 1
            by_source[source]["correct"] += int(ok)
            print("=" * 80)
            print(f"Sample {idx}")
            print("Source:", source)
            print("Expected:", ex["answer"])
            print("Predicted:", normalize_text(pred)[:900])
            print("Correct:", ok)

    print("=" * 80)
    print("SUMMARY")
    print(f"Accuracy: {correct}/{total} = {correct / max(total, 1):.4f}")

    print("\nBy source:")
    for source, stats in sorted(by_source.items()):
        source_correct = stats["correct"]
        source_total = stats["total"]
        print(
            f"- {source}: "
            f"{source_correct}/{source_total} = "
            f"{source_correct / max(source_total, 1):.4f}"
        )

    save_results(args, correct, total, by_source, model_dir, eval_file)


if __name__ == "__main__":
    main()
