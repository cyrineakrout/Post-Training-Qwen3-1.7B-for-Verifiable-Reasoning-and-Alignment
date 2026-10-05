#!/usr/bin/env python3

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm
import json, random



INSTRUCTION_RE = re.compile(
    r"^Answer the following multiple choice question\.\s*"
    r"The last line of your response should be in the following format:.*?\n\n",
    flags=re.IGNORECASE | re.DOTALL,
)


def clean_nemotron_question(content: str) -> str:
    """
    Remove Nemotron instruction prefix and embedded A:/B:/... choices.
    Keep only question text.
    """
    content = str(content).strip()

    content = INSTRUCTION_RE.sub("", content).strip()

    # Remove embedded choices from the original prompt.
    # We will rebuild the choices cleanly from the options field.
    content = re.sub(
        r"\n\s*A\s*:\s*.+$",
        "",
        content,
        flags=re.DOTALL,
    ).strip()

    content = re.sub(r"\s+", " ", content).strip()
    return content


def get_nemotron_prompt_content(ex: Dict[str, Any]) -> str:
    """
    Nemotron format:
      responses_create_params["input"][0]["content"]
    """
    params = ex.get("responses_create_params", {}) or {}
    input_messages = params.get("input", []) or []

    if input_messages and isinstance(input_messages[0], dict):
        return str(input_messages[0].get("content", "")).strip()

    raw_input = ex.get("input", "")

    if isinstance(raw_input, list) and raw_input and isinstance(raw_input[0], dict):
        return str(raw_input[0].get("content", "")).strip()

    if isinstance(raw_input, str):
        return raw_input.strip()

    return ""


def extract_nemotron_options(ex: Dict[str, Any]):
    """
    Nemotron options look like:
      [
        {"A": "text", "B": None, ...},
        {"A": None, "B": "text", ...},
        ...
      ]

    Keep only non-null choices.
    Allow 2 to 20 choices: A through T.
    """
    choices_by_letter = {}

    raw_options = ex.get("options", []) or []

    for opt in raw_options:
        if not isinstance(opt, dict):
            continue

        for letter, text in opt.items():
            if text is None:
                continue

            letter = str(letter).strip().upper()
            text = str(text).strip()

            if len(letter) == 1 and "A" <= letter <= "T" and text:
                choices_by_letter[letter] = text

    if not choices_by_letter:
        return None, None

    letters = sorted(choices_by_letter.keys())
    expected_letters = [chr(ord("A") + i) for i in range(len(letters))]

    # Require contiguous A, B, C, ...
    if letters != expected_letters:
        return None, None

    choices = [choices_by_letter[l] for l in expected_letters]

    if len(choices) < 2 or len(choices) > 20:
        return None, None

    return choices_by_letter, choices


def has_pass_rate_one(ex: Dict[str, Any]) -> bool:
    """
    Optional quality filter: keep examples where at least one reward profile has pass_rate == 1.0.
    """
    for rp in ex.get("reward_profiles", []) or []:
        try:
            if float(rp.get("pass_rate", -1)) == 1.0:
                return True
        except Exception:
            pass
    return False


def has_qwen_pass_rate_one(ex: Dict[str, Any]) -> bool:
    """
    Optional quality filter: keep examples where Qwen/Qwen3-30B-A3B has pass_rate == 1.0.
    """
    for rp in ex.get("reward_profiles", []) or []:
        try:
            if (
                rp.get("model_hf_path") == "Qwen/Qwen3-30B-A3B"
                and float(rp.get("pass_rate", -1)) == 1.0
            ):
                return True
        except Exception:
            pass
    return False


def make_sft_row(question: str, choices, answer_index: int, source: str, system_prompt: str):
    letters = [chr(ord("A") + i) for i in range(len(choices))]
    answer_letter = letters[answer_index]

    choices_text = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    )

    user_prompt = f"{question}\n\nChoices:\n{choices_text}"

    # Your short controlled-thinking target.
    # This assumes your training/eval template opens <think> before the assistant content.
    assistant_content = (
        "<think>\n"
        "I will identify the best option.\n"
        "</think>\n\n"
        f"\\boxed{{{answer_letter}}}"
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": source,
            "num_choices": len(choices),
            "answer": answer_letter,
            "has_explanation": False,
            "short_think_format": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        default="nvidia/Nemotron-RL-knowledge-mcqa",
    )
    parser.add_argument(
        "--split",
        default="train",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap after filtering.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dataset before filtering/writing.",
    )
    parser.add_argument(
        "--require_any_pass1",
        action="store_true",
        help="Keep only rows where at least one reward profile has pass_rate == 1.0.",
    )
    parser.add_argument(
        "--require_qwen_pass1",
        action="store_true",
        help="Keep only rows where Qwen/Qwen3-30B-A3B has pass_rate == 1.0.",
    )
    parser.add_argument(
        "--system_prompt",
        default=(
            "You are a careful multiple-choice general-knowledge assistant. "
            "Choose exactly one correct option from the choices. "
            "At the end, write the final answer as exactly one capital letter inside \\boxed{}. "
            "The boxed content must contain only the option letter, for example \\boxed{C}."
        ),
    )

    args = parser.parse_args()
    random.seed(args.seed)

    ds = load_dataset(args.dataset_name, split=args.split)

    indices = list(range(len(ds)))
    if args.shuffle:
        random.shuffle(indices)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_empty = 0
    skipped_options = 0
    skipped_answer = 0
    skipped_pass_filter = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx in tqdm(indices, desc="Filtering Nemotron"):
            ex = ds[idx]

            if args.require_any_pass1 and not has_pass_rate_one(ex):
                skipped_pass_filter += 1
                continue

            if args.require_qwen_pass1 and not has_qwen_pass_rate_one(ex):
                skipped_pass_filter += 1
                continue

            content = get_nemotron_prompt_content(ex)
            if not content:
                skipped_empty += 1
                continue

            question = clean_nemotron_question(content)
            if not question:
                skipped_empty += 1
                continue

            choices_by_letter, choices = extract_nemotron_options(ex)
            if choices_by_letter is None or choices is None:
                skipped_options += 1
                continue

            answer_letter = str(ex.get("expected_answer", "")).strip().upper()

            if (
                not answer_letter
                or len(answer_letter) != 1
                or not ("A" <= answer_letter <= "T")
                or answer_letter not in choices_by_letter
            ):
                skipped_answer += 1
                continue

            answer_index = ord(answer_letter) - ord("A")

            if answer_index < 0 or answer_index >= len(choices):
                skipped_answer += 1
                continue

            row = make_sft_row(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source="nemotron_rl_knowledge_mcqa",
                system_prompt=args.system_prompt,
            )

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

            if args.max_examples is not None and kept >= args.max_examples:
                break

    print("=" * 80)
    print(f"Dataset:               {args.dataset_name}")
    print(f"Split:                 {args.split}")
    print(f"Output:                {out_path}")
    print(f"Kept:                  {kept}")
    print(f"Skipped empty:         {skipped_empty}")
    print(f"Skipped bad options:   {skipped_options}")
    print(f"Skipped bad answer:    {skipped_answer}")
    print(f"Skipped pass filter:   {skipped_pass_filter}")
    print("=" * 80)
    inp = Path("general_knowledge/data/nemotron_clean_valid_qwenpass1_2to20.jsonl")
    eval_out = Path("general_knowledge/data/nemotron_eval_heldout_300.jsonl")
    train_out = Path("general_knowledge/data/nemotron_train_pool_after_heldout.jsonl")

    random.seed(42)

    rows = []
    with inp.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    random.shuffle(rows)

    eval_rows = rows[:300]
    train_rows = rows[300:]

    with eval_out.open("w", encoding="utf-8") as f:
        for r in eval_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with train_out.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total: {len(rows)}")
    print(f"Heldout eval: {len(eval_rows)} -> {eval_out}")
    print(f"Train pool: {len(train_rows)} -> {train_out}")




if __name__ == "__main__":
    main()
    