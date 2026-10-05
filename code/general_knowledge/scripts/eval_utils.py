import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT.parent  # code/ — contains the evaluate/ package
for _p in [str(PROJECT_ROOT), str(CODE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def slugify(value: str) -> str:
    slug = []

    for char in str(value):
        if char.isalnum():
            slug.append(char.lower())
        elif char in {".", "-", "_"}:
            slug.append(char.lower())
        else:
            slug.append("-")

    cleaned = "".join(slug).strip("-")

    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")

    return cleaned or "run"


def model_slug(model_name: str) -> str:
    return slugify(Path(model_name).name)


def build_run_name(
    task: str,
    model_name: str,
    dataset_path: str,
    method: Optional[str] = None,
) -> str:
    parts = [slugify(task), model_slug(model_name), slugify(Path(dataset_path).stem)]

    if method:
        parts.append(slugify(method))

    return "-".join(parts)


def load_jsonl(path: str, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
    examples = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}:\n{line[:200]}"
                ) from e

            if max_examples is not None and len(examples) >= max_examples:
                break

    return examples


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(data: Dict[str, Any], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_messages(prompt: str) -> List[Dict[str, str]]:
    """
    Build the messages list for evaluation.

    We intentionally pass USER-ONLY (no system message) so this matches
    what CI does: CI calls `apply_chat_template(messages, add_generation_prompt=True)`
    with `messages` containing only the user prompt.

    With the chat-template injection from `force_default_system_prompt` in
    `train_sft.py`, the template auto-prepends the SFT system prompt at
    render time when no system message is provided. So the rendered prompt
    is identical between training (which provides system explicitly) and
    eval (which doesn't, and lets the template inject it).

    Running the local eval this way also acts as an integration test for
    the chat-template injection: if accuracy collapses, the injection
    isn't working on the saved tokenizer.

    The `system_prompt` parameter is kept for backward compatibility with
    callers but is intentionally unused.
    """
    return [
        {"role": "user", "content": prompt},
    ]

def generate_completion_rows(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    desc: str,
    batch_size: int = 8,
    top_k: int = 20,
    repetition_penalty: float = 1.0,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    import torch
    from tqdm import tqdm

    rows = []
    do_sample = temperature > 0.0

    model.eval()

    # Important for batched decoder-only models
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared = []

    for ex in examples:
        if "prompt" in ex:
            prompt = ex["prompt"]
            answer = ex.get("answer", ex.get("reference"))

            if answer is None:
                raise ValueError(
                    "Each benchmark-style evaluation row must contain 'answer' or 'reference'."
                )

            messages = build_messages(prompt)

        elif "messages" in ex and "metadata" in ex:
            messages = ex["messages"][:-1]
            prompt = messages[-1]["content"]
            answer = ex["metadata"].get("answer")

            if answer is None:
                raise ValueError(
                    "Each SFT-style evaluation row must contain metadata['answer']."
                )

        else:
            raise ValueError(f"Unsupported evaluation row format: {ex.keys()}")

        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if "source" in ex:
            src = ex["source"]
        elif "metadata" in ex and isinstance(ex["metadata"], dict):
            src = ex["metadata"].get("source", "unknown")
        else:
            src = "unknown"

        prepared.append(
            {
                "prompt": prompt,
                "answer": str(answer),
                "input_text": input_text,
                "source": src,
            }
        )

    for start in tqdm(range(0, len(prepared), batch_size), desc=desc):
        batch = prepared[start : start + batch_size]
        input_texts = [item["input_text"] for item in batch]

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        input_width = inputs["input_ids"].shape[1]

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": repetition_penalty,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_tokens = outputs[:, input_width:]

        completions = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        for item, completion in zip(batch, completions):
            rows.append(
                {
                    "prompt": item["prompt"],
                    "answer": item["answer"],
                    "completions": [completion],
                    "source": item["source"],
                }
            )

    return rows
def generate_completion_rows_vllm(
    model_path: str,
    examples: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    desc: str,
    top_k: int = -1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "bfloat16",
    repetition_penalty: float = 1.0,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    prompts = []
    metadata_rows = []

    for ex in tqdm(examples, desc=f"Preparing {desc} prompts"):
        if "prompt" in ex:
            prompt = ex["prompt"]
            answer = ex.get("answer", ex.get("reference"))

            if answer is None:
                raise ValueError(
                    "Each benchmark-style evaluation row must contain 'answer' or 'reference'."
                )

            messages = build_messages(prompt)

        elif "messages" in ex and "metadata" in ex:
            messages = ex["messages"][:-1]
            prompt = messages[-1]["content"]
            answer = ex["metadata"].get("answer")

            if answer is None:
                raise ValueError(
                    "Each SFT-style evaluation row must contain metadata['answer']."
                )

        else:
            raise ValueError(f"Unsupported evaluation row format: {ex.keys()}")

        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if "source" in ex:
            src = ex["source"]
        elif "metadata" in ex and isinstance(ex["metadata"], dict):
            src = ex["metadata"].get("source", "unknown")
        else:
            src = "unknown"

        prompts.append(input_text)
        metadata_rows.append(
            {
                "prompt": prompt,
                "answer": str(answer),
                "source": src,
            }
        )

    do_sample = temperature > 0.0

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else -1,
        repetition_penalty=repetition_penalty,
    )

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    outputs = llm.generate(
        prompts,
        sampling_params,
        use_tqdm=True,
    )

    rows = []

    for meta, output in zip(metadata_rows, outputs):
        completion = output.outputs[0].text

        rows.append(
            {
                "prompt": meta["prompt"],
                "answer": meta["answer"],
                "completions": [completion],
                "source": meta["source"],
            }
        )

    return rows

def score_generation_rows(
    rows: List[Dict[str, Any]],
    method: str = "knowledge",
) -> Dict[str, Any]:
    from evaluate.score import score_generations

    return score_generations(rows, method=method)


def flatten_score_metrics(
    score_result: Dict[str, Any],
    prefix: str,
) -> Dict[str, float]:
    metrics = {
        f"{prefix}/{name.replace('@', '_at_')}": value
        for name, value in score_result["metrics"].items()
    }
    metrics[f"{prefix}/n_problems"] = score_result["n_problems"]
    metrics[f"{prefix}/n_completions"] = score_result["n_completions"]
    return metrics


def log_wandb_eval_artifacts(
    wandb,
    artifact_name: str,
    generations_path: str,
    scores_path: Optional[str] = None,
) -> None:
    artifact = wandb.Artifact(artifact_name, type="evaluation")
    artifact.add_file(generations_path)

    if scores_path:
        artifact.add_file(scores_path)

    wandb.log_artifact(artifact)
