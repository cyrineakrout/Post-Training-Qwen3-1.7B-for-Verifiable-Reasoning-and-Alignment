"""Download prepared math training and evaluation data from Hugging Face.

Run from code/:
    python math/scripts/download_data.py
    python math/scripts/download_data.py --eval-only
    python math/scripts/download_data.py --version v7_100k
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

DATASET_REPO = "cs-552-2026-camykaz/math_model"
DATA_DIR = Path("math/data/hf")

VERSION_FILES = {
    "v1_early_sft": ["train.jsonl", "validation.jsonl"],
    "v2_revised_sft": ["train.jsonl", "validation.jsonl"],
    "v3_curated_sft": ["train.jsonl", "validation.jsonl", "train_with_meta.jsonl", "validation_with_meta.jsonl"],
    "v4_strong_sft": ["train.jsonl", "validation.jsonl", "train_with_meta.jsonl", "validation_with_meta.jsonl", "grpo_ref.jsonl"],
    "v5_all_solutions_sft": ["train.jsonl", "validation.jsonl"],
    "v6_50k_long": ["train.jsonl", "validation.jsonl", "manifest.json"],
    "v7_100k": ["train.jsonl", "validation.jsonl", "manifest.json"],
    "final_evaluation": ["math_benchmark_external_v1.jsonl", "manifest.json"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=sorted(VERSION_FILES), help="Download only one prepared dataset version.")
    parser.add_argument("--eval-only", action="store_true", help="Download only the final evaluation benchmark.")
    args = parser.parse_args()

    if args.eval_only:
        versions = ["final_evaluation"]
    elif args.version:
        versions = [args.version]
    else:
        versions = list(VERSION_FILES)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading prepared math data from {DATASET_REPO} to {DATA_DIR}/ ...")
    for version in versions:
        for filename in VERSION_FILES[version]:
            hub_file = f"{version}/{filename}"
            print(f"  {hub_file} ...", end=" ", flush=True)
            path = hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=hub_file,
                local_dir=str(DATA_DIR),
            )
            print(path)
    print("Done.")


if __name__ == "__main__":
    main()
