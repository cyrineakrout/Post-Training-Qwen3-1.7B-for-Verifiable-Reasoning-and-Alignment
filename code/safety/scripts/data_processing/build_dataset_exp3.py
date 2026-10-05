"""
Build combined_v10_nosb training set (~36k examples, NO SafetyBench).

Replaces thu-coai/SafetyBench (10,928 examples) with:
  - walledai/BBQ       : 8,000  (bias/fairness MCQ — closest to SafetyBench)
  - OpenSafetyLab/SALAD:  5,000  (harmful-content MCQ — broader harm categories)

Keeps all other v5_all sources unchanged:
  - PKU-Alignment/BeaverTails : 5,500
  - Anthropic/hh-rlhf         : 5,000
  - allenai/social_i_qa       : 5,000
  - google/civil_comments     : 4,600
  - hendrycks/ethics          : 3,176

Total target: ~36,276 examples

Usage (from project root on RCP):

  # Step 1 – build BBQ data (if not already built)
  python safety/scripts/build_bbq_safety_sft.py \
      --output safety/data/bbq_safety_sft.jsonl \
      --max_examples 8000

  # Step 2 – build SALAD data (if not already built)
  python safety/scripts/build_salad_safety_sft.py \
      --sft \
      --output safety/data/salad_mcq_sft.jsonl

  # Step 3 – build combined v10
  python safety/scripts/build_combined_v10_nosb.py

  # Step 4 – train
  python safety/scripts/train_safety_lora.py \
      --config safety/configs/safety_lora_combined_v10_nosb.yaml
"""

import json
import random
from pathlib import Path


SEED = 42
DEV_SIZE = 500

# Input files (all on RCP under safety/data/)
SOURCES = [
    # (path, optional max, label)
    ("safety/data/combined_v5_safetybench_train.jsonl",  None,  "v5_all_nosb"),  # filtered below
    ("safety/data/bbq_safety_sft.jsonl",                 8_000, "bbq"),
    ("safety/data/salad_mcq_sft.jsonl",                  8_000, "salad"),
]

OUTPUT_TRAIN = "safety/data/combined_v10_nosb_train.jsonl"
OUTPUT_DEV   = "safety/data/combined_v10_nosb_dev.jsonl"

# Sources to EXCLUDE when pulling from v5_all
EXCLUDE_SOURCES = {"thu-coai/SafetyBench"}


def load_jsonl(path: str, max_rows=None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def save_jsonl(rows: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(rows):,} → {path}")


def main():
    random.seed(SEED)

    all_rows: list[dict] = []

    for src_path, max_rows, label in SOURCES:
        p = Path(src_path)
        if not p.exists():
            print(f"  [SKIP] {src_path} not found — run the appropriate build script first.")
            continue

        rows = load_jsonl(src_path)

        # For v5_all: filter out SafetyBench rows
        if label == "v5_all_nosb":
            before = len(rows)
            rows = [
                r for r in rows
                if r.get("metadata", {}).get("source", "") not in EXCLUDE_SOURCES
            ]
            print(f"  v5_all: {before:,} → {len(rows):,} after removing SafetyBench")
        else:
            if max_rows and len(rows) > max_rows:
                random.shuffle(rows)
                rows = rows[:max_rows]

        print(f"  {label:<20} {len(rows):,} examples")
        all_rows.extend(rows)

    print(f"\nTotal before split: {len(all_rows):,}")

    random.shuffle(all_rows)

    dev_rows   = all_rows[:DEV_SIZE]
    train_rows = all_rows[DEV_SIZE:]

    save_jsonl(train_rows, OUTPUT_TRAIN)
    save_jsonl(dev_rows,   OUTPUT_DEV)

    # Source distribution in train
    from collections import Counter
    source_counts = Counter(
        r.get("metadata", {}).get("source", "unknown") for r in train_rows
    )
    print("\nSource distribution in train set:")
    for src, cnt in source_counts.most_common():
        print(f"  {cnt:6,}  {src}")


if __name__ == "__main__":
    main()
