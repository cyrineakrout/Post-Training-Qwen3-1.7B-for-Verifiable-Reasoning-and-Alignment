"""
Build SFT data from BBQ (Bias Benchmark for QA).

BBQ covers 11 social-bias categories directly relevant to SafetyBench:
  age, disability_status, gender_identity, nationality, physical_appearance,
  race_ethnicity, race_x_ses, race_x_gender, religion, ses, sexual_orientation

Each example: 3-option MCQ (A/B/C) — directly usable without format conversion.

Two question types are included:
  - disambig:  context is sufficient → clear right answer (bias/fairness judgment)
  - ambig:     context is not sufficient → correct answer is the "unknown" option
                (trains model to avoid assumptions)

Usage:
  python safety/scripts/build_bbq_safety_sft.py \
      --output safety/data/bbq_safety_sft.jsonl \
      --max_examples 8000 \
      --seed 42

HuggingFace dataset: walledai/BBQ  (also available as heegyu/bbq)
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)

LETTERS = "ABCDEFGHIJKLMNOPQRST"

# Category-specific preambles to add variety
CATEGORY_PREAMBLES = {
    "Age": "Consider the following situation about age-related bias:",
    "Disability_status": "Consider the following situation about disability-related bias:",
    "Gender_identity": "Consider the following situation about gender identity:",
    "Nationality": "Consider the following situation about nationality-related bias:",
    "Physical_appearance": "Consider the following situation about physical appearance bias:",
    "Race_ethnicity": "Consider the following situation about race and ethnicity:",
    "Race_x_SES": "Consider the following situation about race and socioeconomic status:",
    "Race_x_gender": "Consider the following situation about race and gender:",
    "Religion": "Consider the following situation about religious bias:",
    "SES": "Consider the following situation about socioeconomic status:",
    "Sexual_orientation": "Consider the following situation about sexual orientation bias:",
}

DEFAULT_PREAMBLE = "Consider the following situation:"


def build_bbq_example(row: dict) -> dict | None:
    """Convert a single BBQ row to SFT format."""
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    choices = row.get("choices") or []
    label = row.get("answer")
    category = (row.get("category") or "").strip()

    if not all([context, question]) or len(choices) < 2:
        return None
    if not isinstance(label, int) or label >= len(choices):
        return None

    options = [c.strip() for c in choices]
    answer_letter = LETTERS[label]

    preamble = CATEGORY_PREAMBLES.get(category, DEFAULT_PREAMBLE)

    user_prompt = (
        f"{preamble}\n\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        + "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": f"\\boxed{{{answer_letter}}}"},
        ],
        "metadata": {
            "source": "walledai/BBQ",
            "task_type": "bias_fairness_mcq",
            "category": category,
            "context_condition": row.get("context_condition", ""),
            "question_polarity": row.get("question_polarity", ""),
            "num_options": 3,
            "answer": answer_letter,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build BBQ bias/fairness MCQ SFT data."
    )
    parser.add_argument(
        "--output", default="safety/data/bbq_safety_sft.jsonl"
    )
    parser.add_argument(
        "--max_examples", type=int, default=8000,
        help="Max examples to keep (default 8000)."
    )
    parser.add_argument(
        "--disambig_only", action="store_true",
        help="Only include disambiguated examples (clear right answer). "
             "Default: include both ambig and disambig."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # walledai/BBQ uses one split per category (not a single "train" split)
    BBQ_SPLITS = [
        "age", "disabilityStatus", "genderIdentity", "nationality",
        "physicalAppearance", "raceEthnicity", "raceXSes", "raceXGender",
        "religion", "ses", "sexualOrientation",
    ]

    print("Loading walledai/BBQ (all category splits) …")
    all_rows = []
    for split_name in BBQ_SPLITS:
        try:
            split_ds = load_dataset("walledai/BBQ", split=split_name)
            all_rows.extend(list(split_ds))
            print(f"  {split_name:<25} {len(split_ds):,} rows")
        except Exception as e:
            print(f"  [SKIP] {split_name}: {e}")

    from datasets import Dataset
    ds = Dataset.from_list(all_rows)
    print(f"  Total rows: {len(ds):,}")

    rows = list(ds)
    random.shuffle(rows)

    if args.disambig_only:
        rows = [r for r in rows if r.get("context_condition") == "disambig"]
        print(f"  After disambig filter: {len(rows)}")

    counters = Counter()
    kept = []

    for row in rows:
        if len(kept) >= args.max_examples:
            break
        ex = build_bbq_example(row)
        if ex is None:
            counters["skipped"] += 1
            continue
        kept.append(ex)
        counters["written"] += 1
        counters[f"cat:{ex['metadata']['category']}"] += 1
        counters[f"cond:{ex['metadata']['context_condition']}"] += 1

    with output_path.open("w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nWritten : {counters['written']}")
    print(f"Skipped : {counters['skipped']}")
    print(f"Output  : {output_path}")

    print("\nCategory distribution:")
    for k, v in sorted(
        ((k, v) for k, v in counters.items() if k.startswith("cat:")),
        key=lambda x: -x[1]
    ):
        print(f"  {k[4:]:<35} {v}")

    print("\nContext condition distribution:")
    for k, v in sorted(
        ((k, v) for k, v in counters.items() if k.startswith("cond:")),
        key=lambda x: -x[1]
    ):
        print(f"  {k[5:]:<15} {v}")


if __name__ == "__main__":
    main()
