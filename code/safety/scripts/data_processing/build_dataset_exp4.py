"""
Build combined_v12 training set (~62k examples, all sources scaled up).

All sources built fresh from scratch — NO SafetyBench, no dependency on v5_all.

Source composition:
  - PKU-Alignment/BeaverTails (30k_train) : 12,000
  - Anthropic/hh-rlhf                     :  8,000
  - SIQA + ETHICS + Civil (mixed)         : 21,000  (8k SIQA + 6k ETHICS + 7k Civil)
  - walledai/BBQ                          : 20,000
  - OpenSafetyLab/SALAD                   :    947
  Total                                    ≈ 61,947

Usage (from project root on RCP — run builds FIRST, can run in parallel):

  # Terminal 1 — BeaverTails 12k
  python safety/scripts/build_beavertails_refusal_sft.py \
      --split 30k_train --max_examples 12000 \
      --output_path safety/data/beavertails_v12_sft.jsonl

  # Terminal 2 — HH-RLHF 8k
  python safety/scripts/build_hh_rlhf_sft.py \
      --max_examples 8000 \
      --output_path safety/data/hh_rlhf_v12_sft.jsonl

  # Terminal 3 — Mixed (SIQA 8k + ETHICS 6k + Civil 7k = 21k)
  python safety/scripts/build_mixed_safety_sft.py \
      --output_path safety/data/mixed_safety_v12_sft.jsonl \
      --max_siqa 8000 --max_ethics 6000 \
      --max_civil_tox 5000 --max_civil_bias 2000

  # Terminal 4 — BBQ 20k
  python safety/scripts/build_bbq_safety_sft.py \
      --max_examples 20000 \
      --output safety/data/bbq_safety_v12_sft.jsonl

  # SALAD is already at its max (~947), reuse existing file:
  #   safety/data/salad_mcq_sft.jsonl

  # Combine all into train/dev split
  python safety/scripts/build_combined_v12.py

  # Train
  python safety/scripts/train_safety_lora.py \
      --config safety/configs/safety_lora_combined_v12.yaml
"""

import json
import random
from collections import Counter
from pathlib import Path


SEED = 42
DEV_SIZE = 500

# (path, max_rows_to_use, label)
# max_rows=None means use all rows in the file
SOURCES = [
    ("safety/data/beavertails_v12_sft.jsonl",  12_000, "beavertails"),
    ("safety/data/hh_rlhf_v12_sft.jsonl",       8_000, "hh_rlhf"),
    ("safety/data/mixed_safety_v12_sft.jsonl",  21_000, "mixed"),
    ("safety/data/bbq_safety_v12_sft.jsonl",    20_000, "bbq"),
    ("safety/data/salad_mcq_sft.jsonl",           None, "salad"),
]

OUTPUT_TRAIN = "safety/data/combined_v12_train.jsonl"
OUTPUT_DEV   = "safety/data/combined_v12_dev.jsonl"


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
    missing = []

    for src_path, max_rows, label in SOURCES:
        p = Path(src_path)
        if not p.exists():
            missing.append(src_path)
            print(f"  [MISSING] {src_path}")
            continue

        rows = load_jsonl(src_path)

        if max_rows and len(rows) > max_rows:
            random.shuffle(rows)
            rows = rows[:max_rows]

        print(f"  {label:<20} {len(rows):,} examples")
        all_rows.extend(rows)

    if missing:
        print(f"\n[!] {len(missing)} source file(s) not found. Run the build scripts first:")
        for m in missing:
            print(f"    {m}")
        return

    print(f"\nTotal before split: {len(all_rows):,}")

    random.shuffle(all_rows)

    dev_rows   = all_rows[:DEV_SIZE]
    train_rows = all_rows[DEV_SIZE:]

    save_jsonl(train_rows, OUTPUT_TRAIN)
    save_jsonl(dev_rows,   OUTPUT_DEV)

    # Source distribution in train set
    source_counts = Counter(
        r.get("metadata", {}).get("source", "unknown") for r in train_rows
    )
    print("\nSource distribution in train set:")
    for src, cnt in source_counts.most_common():
        print(f"  {cnt:6,}  {src}")


if __name__ == "__main__":
    main()
