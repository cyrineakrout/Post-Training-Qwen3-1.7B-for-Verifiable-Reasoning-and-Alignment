"""
Download multilinguality datasets from HuggingFace.

Run this once before training or evaluating any experiment.
Files are saved to data/.

Usage (from code/multilinguality/):

  python scripts/download_data.py                # training + eval
  python scripts/download_data.py --eval-only    # only eval.jsonl
  python scripts/download_data.py --train-only   # only training JSONL files
"""

import argparse
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

with open(ML_ROOT / "config.json") as f:
    CFG = json.load(f)

TRAIN_REPO = CFG["hf_data"]["train_repo"]
EVAL_REPO  = CFG["hf_data"]["eval_repo"]

TRAIN_FILES = [
    "grpo_train.jsonl",
    "sft_train.jsonl",
    "distillation_pool.jsonl",
    "distillation_train_english.jsonl",
    "distillation_train_multi.jsonl",
]

EVAL_FILE = "eval.jsonl"
DATA_DIR  = ML_ROOT / "data"


def download(repo_id: str, filename: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download
    if dest.exists():
        print(f"  [skip] {filename}")
        return
    print(f"  {filename} ...", end=" ", flush=True)
    try:
        hf_hub_download(repo_id=repo_id, filename=filename,
                        repo_type="dataset", local_dir=str(DATA_DIR))
        print("done")
    except Exception as e:
        print(f"SKIP ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        print(f"\n=== Training data from {TRAIN_REPO} ===")
        for f in TRAIN_FILES:
            download(TRAIN_REPO, f, DATA_DIR / f)

    if not args.train_only:
        print(f"\n=== Eval data from {EVAL_REPO} ===")
        download(EVAL_REPO, EVAL_FILE, DATA_DIR / EVAL_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()
