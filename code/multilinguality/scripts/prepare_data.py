"""
Prepare all multilingual training data in one pass.

Usage:
  python prepare_data.py [--path PATH]

  PATH controls what gets written:
    sft                  — no exclusion, no GRPO reservation, everything → SFT
    distillation_sft     — exclude distillation IDs, no GRPO reservation, rest → SFT
    simple_sft           — no exclusion, no GRPO reservation, everything → SFT
    cold_grpo            — no exclusion, reserve GRPO, rest → SFT
    sft_grpo             — no exclusion, reserve GRPO, distillation data joins SFT
    distillation_sft_grpo— exclude distillation IDs, reserve GRPO, rest → SFT (default)

Algorithm:
  1. Load all available examples per language (source_limits apply).
  2. Optionally exclude IDs present in the distillation training file (path-dependent).
  3. Shuffle each language pool with a fixed seed.
  4. Optionally reserve first GRPO_PER_LANG (600) examples per language → GRPO set.
  5. Pool the SFT remainder and apply 3× soft balance.
  6. Write output files.
"""

from typing import Optional
import argparse
import json
import random
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

CONFIG_PATH = ML_ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

from multilinguality.scripts.data_sources import load_all

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

DATA_CFG   = CFG["data"]
SFT_CFG    = CFG["sft"]
DIST_CFG   = CFG["distillation"]
PROMPT_CFG = CFG["prompts"]
LETTERS    = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

GRPO_PER_LANG   = DATA_CFG["max_per_lang"]      # 600
DIST_PER_LANG   = DIST_CFG.get("max_per_lang", GRPO_PER_LANG)
SOFT_BALANCE_MX = 3                              # SFT cap = 3× smallest language pool


def to_grpo_record(ex: dict) -> Optional[dict]:
    correct_index = ord(ex["answer"]) - ord("A")
    if correct_index >= len(ex["options"]) or not ex["options"][correct_index]:
        return None
    options_block = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(ex["options"]))
    return {
        "messages": [
            {"role": "user", "content": f"{ex['question']}\n\nOptions:\n{options_block}"},
        ],
        "option_texts":   ex["options"],
        "correct_index":  correct_index,
        "correct_letter": ex["answer"],
        "language":       ex.get("lang", ""),
    }


def to_sft_record(ex: dict) -> Optional[dict]:
    correct_index = ord(ex["answer"]) - ord("A")
    if correct_index >= len(ex["options"]) or not ex["options"][correct_index]:
        return None
    options_block = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(ex["options"]))
    return {
        "_id": ex["id"],
        "messages": [
            {"role": "system",    "content": PROMPT_CFG["sft_system"]},
            {"role": "user",      "content": f"{ex['question']}\n\nOptions:\n{options_block}"},
            {"role": "assistant", "content": (
                f"Option {ex['answer']} is the correct answer. {ex['options'][correct_index]}\n\n"
                f"\\boxed{{{ex['answer']}}}"
            )},
        ],
    }


def write_jsonl(records: list, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            if rec is None:
                skipped += 1
                continue
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    print(f"  {label}: {written} written, {skipped} skipped → {path}")


def main():
    VALID_PATHS = ["sft", "distillation_sft", "simple_sft", "cold_grpo", "sft_grpo", "distillation_sft_grpo"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="distillation_sft_grpo",
                        choices=VALID_PATHS,
                        help="Training path (controls data routing)")
    args = parser.parse_args()
    path = args.path

    seed      = DATA_CFG["seed"]
    limits    = DATA_CFG.get("source_limits")
    languages = DATA_CFG["languages"]

    exclude_distillation = "distillation" in path
    reserve_grpo         = "grpo" in path

    print(f"Path: {path}")
    print(f"  carve distillation pool: {exclude_distillation} ({DIST_PER_LANG}/lang)")
    print(f"  reserve GRPO pool:       {reserve_grpo} ({GRPO_PER_LANG}/lang)\n")

    raw_pools:  dict[str, list] = {}
    dist_pools: dict[str, list] = {}
    grpo_pools: dict[str, list] = {}
    sft_pools:  dict[str, list] = {}

    for lang in languages:
        all_exs = load_all([lang], max_per_lang=5000, seed=seed, limits=limits)
        random.Random(seed).shuffle(all_exs)
        for ex in all_exs:
            ex["lang"] = lang
        raw_pools[lang] = all_exs

        cursor = 0

        # 1. Distillation slice (carved first so it never overlaps GRPO/SFT)
        if exclude_distillation:
            dist_pools[lang] = all_exs[cursor : cursor + DIST_PER_LANG]
            cursor += DIST_PER_LANG
        else:
            dist_pools[lang] = []

        remainder = all_exs[cursor:]

        # 2. GRPO slice from the remainder
        if reserve_grpo:
            grpo_pools[lang] = remainder[:GRPO_PER_LANG]
            sft_pools[lang]  = remainder[GRPO_PER_LANG:]
        else:
            grpo_pools[lang] = []
            sft_pools[lang]  = remainder

        print(f"  {lang}: {len(all_exs)} total → "
              f"{len(dist_pools[lang])} distillation + "
              f"{len(grpo_pools[lang])} GRPO + "
              f"{len(sft_pools[lang])} SFT remainder")

    min_sft = min(len(p) for p in sft_pools.values())
    sft_cap = min_sft * SOFT_BALANCE_MX
    print(f"\nSFT soft balance: min pool = {min_sft}, cap = {sft_cap} ({SOFT_BALANCE_MX}× min)")

    sft_examples: list = []
    for lang in languages:
        pool = sft_pools[lang][:sft_cap]
        print(f"  {lang}: {len(pool)} SFT examples (pool had {len(sft_pools[lang])})")
        sft_examples.extend(pool)
    random.Random(seed).shuffle(sft_examples)

    grpo_examples: list = []
    for lang in languages:
        grpo_examples.extend(grpo_pools[lang])
    random.Random(seed).shuffle(grpo_examples)

    if exclude_distillation:
        print("\nWriting distillation pool...")
        dist_records = [ex for pool in dist_pools.values() for ex in pool]
        random.Random(seed).shuffle(dist_records)
        write_jsonl(dist_records, ml_path(DIST_CFG["pool_file"]), "distillation_pool")

    print("\nWriting raw pool file...")
    pool_records = [
        ex
        for lang in languages
        for ex in raw_pools[lang]
    ]
    write_jsonl(pool_records, ml_path(DATA_CFG["raw_pool_file"]), "ml_raw_pool")

    print("\nWriting GRPO file...")
    write_jsonl(
        [to_grpo_record(ex) for ex in grpo_examples],
        ml_path(DATA_CFG["train_file"]),
        "grpo_train",
    )

    print("\nWriting SFT file...")
    write_jsonl(
        [to_sft_record(ex) for ex in sft_examples],
        ml_path(SFT_CFG["train_file"]),
        "sft_train",
    )

    n_dist = sum(len(p) for p in dist_pools.values())
    print(f"\nDone. distillation_pool={n_dist}, GRPO={len(grpo_examples)}, SFT={len(sft_examples)}")


if __name__ == "__main__":
    main()
