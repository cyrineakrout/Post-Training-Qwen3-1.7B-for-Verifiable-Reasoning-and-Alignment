"""
Build a DPO dataset for safety model fine-tuning.

Two sources of pairs:
  1. Hard negatives  — actual wrong predictions from model inference on a
                       TRAINING-LIKE pool (NOT the clean eval set).
                       chosen=\\boxed{gold}, rejected=\\boxed{model_pred}
  2. Augmentation   — random wrong options from SFT training files
                       chosen=\\boxed{gold}, rejected=\\boxed{random_wrong}


Run inference on training pool to get predictions:
  python safety/scripts/eval_safety_accuracy.py \
      --model_name safety/models/qwen3_safety_merged_combined_v5_all \
      --data_path safety/data/safetybench_10k.jsonl \
      --output_path safety/data/v5all_sb_predictions.jsonl \
      --no_wandb

Then build DPO data:
  python safety/scripts/build_safety_dpo.py \
      --predictions_path safety/data/v5all_sb_predictions.jsonl \
      --augment_paths safety/data/mixed_safety_v3_moresiqa.jsonl \
                      safety/data/safetybench_10k.jsonl \
      --output safety/data/safety_dpo_v1.jsonl \
      --max_augment 4000 \
      --seed 42
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRST"

SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_boxed_letter(text: str):
    """Extract letter from \\boxed{X}."""
    m = re.search(r'\\boxed\{([A-T])\}', text)
    return m.group(1).upper() if m else None


def parse_option_letters(user_content: str):
    """Return sorted list of option letters found in user message."""
    hits = re.findall(r'(?m)^([A-T])[).]\s', user_content)
    return sorted(set(hits))


def make_dpo_pair(prompt: str, chosen_letter: str, rejected_letter: str) -> dict:
    """TRL messages-format DPO example."""
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    return {
        "chosen":   base + [{"role": "assistant", "content": f"\\boxed{{{chosen_letter}}}"}],
        "rejected": base + [{"role": "assistant", "content": f"\\boxed{{{rejected_letter}}}"}],
        "metadata": {
            "chosen_letter":   chosen_letter,
            "rejected_letter": rejected_letter,
        },
    }


# ── Source 1: hard negatives from eval predictions ─────────────────────────────

def load_hard_negatives(predictions_path: str) -> list[dict]:
    """
    Read eval output JSONL (from eval_safety_accuracy.py --output_path).
    Returns DPO pairs only for examples where the model was wrong.
    """
    pairs = []
    path = Path(predictions_path)
    if not path.exists():
        print(f"  WARNING: predictions file not found: {predictions_path}")
        return pairs

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if row.get("correct", True):
                continue  # skip correct predictions

            prompt   = row.get("prompt", "").strip()
            gold     = str(row.get("answer", "")).strip().upper()
            pred     = str(row.get("pred", "")).strip().upper()

            if not prompt or not gold or not pred or gold == pred:
                continue
            if len(gold) != 1 or len(pred) != 1:
                continue

            pair = make_dpo_pair(prompt, gold, pred)
            pair["metadata"]["source_type"] = "hard_negative"
            pairs.append(pair)

    print(f"  Hard negatives loaded: {len(pairs)}")
    return pairs


# ── Source 2: augmentation from SFT training files ────────────────────────────

def load_augmentation_pairs(
    sft_paths: list[str],
    max_pairs: int,
    rng: random.Random,
) -> list[dict]:
    """
    Load SFT examples, parse options, create (gold, random_wrong) DPO pairs.
    Samples from SIQA + SafetyBench training files.
    """
    candidates = []

    for path_str in sft_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"  WARNING: augment file not found: {path_str}")
            continue

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                messages = row.get("messages", [])
                metadata = row.get("metadata", {})

                user_msg = next(
                    (m["content"] for m in messages if m.get("role") == "user"),
                    None,
                )
                asst_msg = next(
                    (m["content"] for m in messages if m.get("role") == "assistant"),
                    None,
                )

                if not user_msg or not asst_msg:
                    continue

                gold = extract_boxed_letter(asst_msg)
                if gold is None:
                    # try metadata
                    gold = str(metadata.get("answer", "")).strip().upper() or None
                if gold is None or len(gold) != 1:
                    continue

                options = parse_option_letters(user_msg)
                wrong_options = [o for o in options if o != gold]
                if not wrong_options:
                    continue

                candidates.append((user_msg.strip(), gold, wrong_options))

    if not candidates:
        print("  No augmentation candidates found.")
        return []

    rng.shuffle(candidates)
    candidates = candidates[:max_pairs]

    pairs = []
    for prompt, gold, wrong_options in candidates:
        rejected = rng.choice(wrong_options)
        pair = make_dpo_pair(prompt, gold, rejected)
        pair["metadata"]["source_type"] = "augmentation"
        pairs.append(pair)

    print(f"  Augmentation pairs built: {len(pairs)}")
    return pairs


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build DPO dataset from eval wrong predictions + augmentation."
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Output JSONL from eval_safety_accuracy.py --output_path.",
    )
    parser.add_argument(
        "--augment_paths",
        type=str,
        nargs="*",
        default=[],
        help="SFT JSONL files to sample augmentation pairs from.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="safety/data/safety_dpo_v1.jsonl",
    )
    parser.add_argument(
        "--max_augment",
        type=int,
        default=4000,
        help="Max augmentation pairs to add on top of hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("=" * 70)
    print("BUILDING SAFETY DPO DATASET")
    print("=" * 70)

    # Source 1: hard negatives (model's actual wrong predictions)
    print("\n[1/2] Loading hard negatives from eval predictions...")
    hard_pairs = load_hard_negatives(args.predictions_path)

    # Source 2: augmentation (random wrong options from training data)
    print("\n[2/2] Building augmentation pairs from SFT files...")
    aug_pairs = load_augmentation_pairs(
        args.augment_paths,
        args.max_augment,
        rng,
    )

    all_pairs = hard_pairs + aug_pairs
    rng.shuffle(all_pairs)

    # Stats
    total = len(all_pairs)
    chosen_dist  = Counter(p["metadata"]["chosen_letter"]  for p in all_pairs)
    rejected_dist = Counter(p["metadata"]["rejected_letter"] for p in all_pairs)

    print(f"\nTotal DPO pairs  : {total}")
    print(f"  Hard negatives : {len(hard_pairs)}")
    print(f"  Augmentation   : {len(aug_pairs)}")
    print(f"\nChosen  distribution : { {k: chosen_dist[k]  for k in sorted(chosen_dist)} }")
    print(f"Rejected distribution: { {k: rejected_dist[k] for k in sorted(rejected_dist)} }")

    # Write
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\nSaved {total} DPO pairs to: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
