"""
Upload multilinguality datasets to HuggingFace.

Two repos are used:
  TRAIN_REPO  — training data (grpo_train, sft_train, distillation_train_*)
  EVAL_REPO   — held-out eval set (eval.jsonl)

Repos are created automatically if they don't exist.

Usage (from code/multilinguality/):

  # Upload everything that exists locally
  python scripts/upload_data.py

  # Upload only the eval set
  python scripts/upload_data.py --eval-only

  # Upload only training data
  python scripts/upload_data.py --train-only

  # Dry-run (prints what would be uploaded, no actual upload)
  python scripts/upload_data.py --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

with open(ML_ROOT / "config.json") as f:
    CFG = json.load(f)

TRAIN_REPO = CFG["hf_data"]["train_repo"]
EVAL_REPO  = CFG["hf_data"]["eval_repo"]

# All training files, in order of how they're generated
TRAIN_FILES = [
    ML_ROOT / "data" / "grpo_train.jsonl",
    ML_ROOT / "data" / "sft_train.jsonl",
    ML_ROOT / "data" / "distillation_pool.jsonl",
    ML_ROOT / "data" / "distillation_train_english.jsonl",
    ML_ROOT / "data" / "distillation_train_multi.jsonl",
]

EVAL_FILE = ML_ROOT / "data" / "eval.jsonl"


def ensure_repo(api, repo_id: str, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] would create dataset repo: {repo_id}")
        return
    from huggingface_hub.utils import RepositoryNotFoundError
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"  repo exists: {repo_id}")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False)
        print(f"  created repo: {repo_id}")


def upload_file(api, local_path: Path, repo_id: str, dry_run: bool, retries: int = 3) -> None:
    if not local_path.exists():
        print(f"  [skip] not found locally: {local_path.name}")
        return
    size_mb = local_path.stat().st_size / 1_048_576
    if dry_run:
        print(f"  [dry-run] would upload {local_path.name}  ({size_mb:.1f} MB)  → {repo_id}")
        return
    import time
    for attempt in range(1, retries + 1):
        try:
            print(f"  uploading {local_path.name}  ({size_mb:.1f} MB)  ...", end=" ", flush=True)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=local_path.name,
                repo_id=repo_id,
                repo_type="dataset",
            )
            print("done")
            return
        except Exception as e:
            print(f"failed (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    print(f"  [ERROR] gave up uploading {local_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only",  action="store_true", help="Upload only eval.jsonl")
    parser.add_argument("--train-only", action="store_true", help="Upload only training JSONL files")
    parser.add_argument("--dry-run",    action="store_true", help="Print what would be uploaded, no actual upload")
    args = parser.parse_args()

    do_train = not args.eval_only
    do_eval  = not args.train_only

    from huggingface_hub import HfApi
    api = HfApi()

    if do_train:
        print(f"\n=== Training data → {TRAIN_REPO} ===")
        ensure_repo(api, TRAIN_REPO, args.dry_run)
        for f in TRAIN_FILES:
            upload_file(api, f, TRAIN_REPO, args.dry_run)

    if do_eval:
        print(f"\n=== Eval data → {EVAL_REPO} ===")
        ensure_repo(api, EVAL_REPO, args.dry_run)
        upload_file(api, EVAL_FILE, EVAL_REPO, args.dry_run)

    if not args.dry_run:
        print("\nAll done.")
        if do_train:
            print(f"  Training data: https://huggingface.co/datasets/{TRAIN_REPO}")
        if do_eval:
            print(f"  Eval data:     https://huggingface.co/datasets/{EVAL_REPO}")
    else:
        print("\n(dry-run complete — nothing was uploaded)")


if __name__ == "__main__":
    main()
