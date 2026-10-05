import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import download_final_evaluation



def extract_boxed(text: str):
    if text is None:
        return None

    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    answer = []

    while i < len(text) and depth > 0:
        char = text[i]

        if char == "{":
            depth += 1
            answer.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                answer.append(char)
        else:
            answer.append(char)

        i += 1

    if depth == 0:
        return "".join(answer).strip()

    return None


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    return text.strip()


def normalize_answer(answer):
    if answer is None:
        return None

    answer = str(answer).strip()

    boxed = extract_boxed(answer)
    if boxed is not None:
        answer = boxed

    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()

    answer = answer.strip().rstrip(".")

    answer = answer.replace("\\dfrac", "\\frac")
    answer = answer.replace("\\tfrac", "\\frac")
    answer = answer.replace("\\left", "")
    answer = answer.replace("\\right", "")
    answer = answer.replace("\\,", "")
    answer = answer.replace("\\!", "")
    answer = answer.replace("°", "^\\circ")
    answer = answer.replace("^{\\circ}", "^\\circ")

    answer = re.sub(r"\s+", "", answer)

    if answer.startswith("{") and answer.endswith("}"):
        answer = answer[1:-1]

    return answer


def answers_match(predicted, expected) -> bool:
    predicted_norm = normalize_answer(predicted)
    expected_norm = normalize_answer(expected)

    if predicted_norm is None or expected_norm is None:
        return False

    if predicted_norm == expected_norm:
        return True

    try:
        pred_float = float(predicted_norm.replace(",", ""))
        exp_float = float(expected_norm.replace(",", ""))
        return abs(pred_float - exp_float) <= 1e-6
    except Exception:
        return False


def read_jsonl(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_num}: {exc}") from exc

    return rows


def row_to_prompt_answer_source(row: dict[str, Any]):
    source = clean_text(row.get("source")) or "unknown"

    # Format: {"prompt": "...", "answer": "..."}
    if "prompt" in row:
        prompt = clean_text(row.get("prompt"))
        answer = clean_text(
            row.get("answer")
            or row.get("reference")
            or row.get("target")
            or row.get("final_answer")
        )

        if not prompt or not answer:
            raise ValueError("Prompt-format row must contain prompt and answer/reference.")

        return prompt, answer, source

    # Format: {"messages": [...], "metadata": {...}}
    messages = row.get("messages")

    if isinstance(messages, list):
        user_parts = []
        assistant_text = ""

        for msg in messages:
            role = msg.get("role")
            content = clean_text(msg.get("content"))

            if not content:
                continue

            if role == "user":
                user_parts.append(content)
            elif role == "assistant":
                assistant_text = content

        prompt = clean_text("\n\n".join(user_parts))

        metadata = row.get("metadata")
        answer = ""

        if isinstance(metadata, dict):
            answer = clean_text(
                metadata.get("answer")
                or metadata.get("reference")
                or metadata.get("target")
                or metadata.get("final_answer")
            )
            source = clean_text(metadata.get("source")) or source

        if not answer:
            boxed = extract_boxed(assistant_text)
            answer = boxed if boxed is not None else clean_text(assistant_text)

        if prompt and answer:
            return prompt, answer, source

    raise ValueError(f"Unsupported eval row format: {row.keys()}")


def load_validation_samples(eval_file: Path, limit: int):
    rows = read_jsonl(eval_file)

    samples = []
    skipped = 0

    for row in rows:
        try:
            prompt, answer, source = row_to_prompt_answer_source(row)
            samples.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "source": source,
                }
            )

            if limit > 0 and len(samples) >= limit:
                break

        except Exception:
            skipped += 1

    print(f"Loaded eval samples: {len(samples)} from {eval_file} | skipped={skipped}")

    return samples


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--eval_file")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--num_completions", type=int, default=8)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=8)

    args = parser.parse_args()

    eval_file = Path(args.eval_file or download_final_evaluation())

    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    samples = load_validation_samples(eval_file, args.limit)

    boxed_any_count = 0
    pass_count = 0
    total_completions = 0
    boxed_completions = 0
    by_source = {}

    for i, sample in enumerate(samples, start=1):
        prompt = sample["prompt"]
        expected = str(sample["answer"]).strip()
        expected_norm = normalize_answer(expected)
        source = sample.get("source", "unknown")

        messages = [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem clearly and concisely. "
                    "Always end with exactly one final answer in \\boxed{}.\n\n"
                    f"{prompt}"
                ),
            }
        ]

        chat_prompt = apply_chat_template(tokenizer, messages)

        inputs = tokenizer(
            chat_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model.device)

        answers = []
        correct = False

        for _ in range(args.num_completions):
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            boxed = extract_boxed(completion)
            boxed_norm = normalize_answer(boxed)

            total_completions += 1

            if boxed is not None:
                boxed_completions += 1

            answers.append(boxed)

            if boxed_norm == expected_norm or answers_match(boxed, expected):
                correct = True

        if any(answer is not None for answer in answers):
            boxed_any_count += 1

        if correct:
            pass_count += 1

        if source not in by_source:
            by_source[source] = {"correct": 0, "total": 0}

        by_source[source]["total"] += 1

        if correct:
            by_source[source]["correct"] += 1

        print("=" * 80)
        print(f"Sample {i}")
        print(f"Source: {source}")
        print(f"Expected: {expected}")
        print(f"Expected normalized: {expected_norm}")
        print(f"Answers: {answers}")
        print(f"Pass@{args.num_completions}: {correct}")

    print("=" * 80)
    print("SUMMARY")
    print(f"Problems with at least one boxed answer: {boxed_any_count}/{len(samples)}")
    print(f"Boxed completions: {boxed_completions}/{total_completions}")
    print(
        f"Pass@{args.num_completions}: "
        f"{pass_count}/{len(samples)} = {pass_count / max(len(samples), 1):.4f}"
    )

    print("\nBy source:")
    for source, stats in sorted(by_source.items()):
        correct = stats["correct"]
        total = stats["total"]
        print(f"- {source}: {correct}/{total} = {correct / max(total, 1):.4f}")


if __name__ == "__main__":
    main()
