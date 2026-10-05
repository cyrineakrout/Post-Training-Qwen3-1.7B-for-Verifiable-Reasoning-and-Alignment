"""
Convert SALAD-Data MCQ (safe answer, single letter) to our SFT/eval format.

Filters:
  - Only "choose the safe answer" questions (not "unsafe")
  - Only single-letter ground truth (not multi-answer like BC)

Usage:
  # Eval format
  python safety/scripts/build_salad_safety_sft.py --output safety/data/salad_mcq_eval.jsonl

  # SFT format
  python safety/scripts/build_salad_safety_sft.py --sft --output safety/data/salad_mcq_sft.jsonl
"""

import argparse
import json
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "Choose the correct option. Answer only with one boxed letter, "
    "like \\boxed{A}. /no_think"
)


def parse_choices(choices_str):
    """Parse 'A. text; B. text; C. text;' into list of (letter, text) pairs."""
    # Split by letter prefix pattern
    parts = re.split(r'(?:^|;\s*)([A-Z])\.\s*', choices_str.strip())
    # parts = ['', 'A', 'text1', 'B', 'text2', 'C', 'text3;']
    result = []
    for i in range(1, len(parts) - 1, 2):
        letter = parts[i]
        text = parts[i + 1].strip().rstrip(';').strip()
        if text:
            result.append((letter, text))
    return result


def build_prompt(baseq, choices):
    """Build a clean MCQ prompt from base question and parsed choices."""
    lines = [baseq.strip(), ""]
    for letter, text in choices:
        lines.append(f"{letter}) {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SALAD-Data MCQ to eval or SFT format."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output JSONL path."
    )
    parser.add_argument(
        "--sft", action="store_true", help="Output in SFT chat format."
    )
    parser.add_argument(
        "--max_examples", type=int, default=-1, help="Max examples to include. -1 = all."
    )
    args = parser.parse_args()

    from datasets import load_dataset

    print("Loading SALAD-Data MCQ set...")
    ds = load_dataset("OpenSafetyLab/Salad-Data", name="mcq_set", split="train")
    print(f"Total MCQ examples: {len(ds)}")

    examples = []
    skipped_unsafe = 0
    skipped_multi = 0
    skipped_parse = 0

    for ex in ds:
        mcq = ex["mcq"] or ""
        gt = ex["gt"] or ""
        baseq = ex["baseq"] or ""
        choices_str = ex["choices"] or ""

        # Filter: only "choose the safe answer"
        if "choose the safe answer" not in mcq:
            skipped_unsafe += 1
            continue

        # Filter: only single-letter answers
        if len(gt) != 1:
            skipped_multi += 1
            continue

        # Parse choices
        choices = parse_choices(choices_str)
        if len(choices) < 2:
            skipped_parse += 1
            continue

        # Verify GT letter is in choices
        choice_letters = [c[0] for c in choices]
        if gt not in choice_letters:
            skipped_parse += 1
            continue

        # Build prompt
        prompt = build_prompt(baseq, choices)
        answer_text = next(text for letter, text in choices if letter == gt)

        # Get categories
        categories = []
        for cat_key in ["1-category", "2-category", "3-category"]:
            val = ex.get(cat_key)
            if val:
                categories.append(val)

        if args.sft:
            assistant_response = f"\\boxed{{{gt}}}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_response},
            ]
            examples.append(
                {
                    "messages": messages,
                    "task_type": "salad_mcq",
                    "categories": categories,
                }
            )
        else:
            examples.append(
                {
                    "prompt": prompt,
                    "answer": gt,
                    "task_type": "salad_mcq",
                    "categories": categories,
                }
            )

    print(f"\nKept: {len(examples)}")
    print(f"Skipped (unsafe question): {skipped_unsafe}")
    print(f"Skipped (multi-answer): {skipped_multi}")
    print(f"Skipped (parse error): {skipped_parse}")

    if args.max_examples > 0 and len(examples) > args.max_examples:
        import random
        random.seed(42)
        random.shuffle(examples)
        examples = examples[: args.max_examples]
        print(f"Sampled down to: {len(examples)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(examples)} examples to {out}")


if __name__ == "__main__":
    main()
