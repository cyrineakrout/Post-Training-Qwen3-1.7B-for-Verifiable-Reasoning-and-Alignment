"""
Build v13 thinking-mode training set from v12 data.

Converts examples that have reasoning text into Qwen3 thinking format:

  BEFORE (v12 format):
    assistant: "The request involves violence. Option A is safest...
                Final answer: \\boxed{A}"

  AFTER (v13 thinking format):
    assistant: "<think>\\nThe request involves violence. Option A is safest...\\n</think>\\n\\boxed{A}"

Examples with bare \\boxed{X} answers (BBQ, SALAD) are SKIPPED — empty
<think> blocks caused the v11 failure and must be avoided.

Sources included (~38k):
  - PKU-Alignment/BeaverTails  : ~12k  (has reasoning)
  - Anthropic/hh-rlhf          :  ~8k  (has reasoning)
  - allenai/social_i_qa        :  ~8k  (has reasoning)
  - hendrycks/ethics           :  ~6k  (has reasoning)
  - google/civil_comments      :  ~7k  (has reasoning)

Sources skipped:
  - walledai/BBQ               : ~20k  (bare \\boxed{X}, no reasoning)
  - OpenSafetyLab/SALAD        :  ~947 (bare \\boxed{X}, no reasoning)

Usage:
  python safety/scripts/build_v13_thinking_sft.py

Then train:
  python safety/scripts/train_safety_lora.py \\
      --config safety/configs/safety_lora_combined_v13_thinking.yaml \\
      --thinking
"""

import json
import re
import random
from pathlib import Path

SEED = 42
DEV_SIZE = 500

INPUT_TRAIN = "safety/data/combined_v12_train.jsonl"
INPUT_DEV   = "safety/data/combined_v12_dev.jsonl"

OUTPUT_TRAIN = "safety/data/combined_v13_thinking_train.jsonl"
OUTPUT_DEV   = "safety/data/combined_v13_thinking_dev.jsonl"

# Pattern to extract reasoning + answer from "reasoning... Final answer: \boxed{X}"
_FINAL_ANSWER_RE = re.compile(
    r"^(.*?)\s*Final answer:\s*\\boxed\{([A-T])\}\s*$",
    re.DOTALL | re.IGNORECASE,
)

# Also handle bare "\boxed{X}" (no reasoning prefix)
_BARE_BOXED_RE = re.compile(r"^\\boxed\{([A-T])\}$")


def convert_to_thinking(content: str) -> str | None:
    """
    Convert assistant content to Qwen3 thinking format.

    Returns None if the content is bare \\boxed{X} (no reasoning to wrap).
    Returns the converted string otherwise.
    """
    content = content.strip()

    # Case 1: bare \boxed{X} — skip (no reasoning)
    if _BARE_BOXED_RE.match(content):
        return None

    # Case 2: "reasoning... Final answer: \boxed{X}"
    m = _FINAL_ANSWER_RE.match(content)
    if m:
        reasoning = m.group(1).strip()
        letter = m.group(2)
        if not reasoning:
            return None  # no actual reasoning, skip
        return f"<think>\n{reasoning}\n</think>\n\\boxed{{{letter}}}"

    # Case 3: anything else — skip (can't parse safely)
    return None


def convert_example(ex: dict) -> dict | None:
    """Convert a single training example to thinking format."""
    messages = ex.get("messages", [])
    if not messages:
        return None

    # Find assistant message
    new_messages = []
    converted = False
    for msg in messages:
        if msg["role"] == "assistant":
            new_content = convert_to_thinking(msg["content"])
            if new_content is None:
                return None  # skip this example
            new_messages.append({"role": msg["role"], "content": new_content})
            converted = True
        else:
            new_messages.append(msg)

    if not converted:
        return None

    return {**ex, "messages": new_messages}


def process_file(input_path: str, output_path: str):
    converted = 0
    skipped = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            new_ex = convert_example(ex)
            if new_ex is None:
                skipped += 1
            else:
                fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
                converted += 1

    print(f"  {input_path}")
    print(f"    Converted : {converted:,}")
    print(f"    Skipped   : {skipped:,} (bare \\boxed{{X}} — BBQ/SALAD)")
    return converted


def verify_sample(path: str, n: int = 3):
    """Print a few examples to verify the format looks right."""
    print(f"\nSample converted examples from {path}:")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            assistant_content = next(
                m["content"] for m in ex["messages"] if m["role"] == "assistant"
            )
            print(f"\n  Example {i+1}:")
            print(f"    {repr(assistant_content[:200])}")


def main():
    random.seed(SEED)

    print("Building v13 thinking training set...")
    print("=" * 60)

    print("\nProcessing TRAIN split:")
    n_train = process_file(INPUT_TRAIN, OUTPUT_TRAIN)

    print("\nProcessing DEV split:")
    n_dev = process_file(INPUT_DEV, OUTPUT_DEV)

    print("\n" + "=" * 60)
    print(f"Train: {n_train:,} examples → {OUTPUT_TRAIN}")
    print(f"Dev  : {n_dev:,} examples → {OUTPUT_DEV}")

    verify_sample(OUTPUT_TRAIN)


if __name__ == "__main__":
    main()
