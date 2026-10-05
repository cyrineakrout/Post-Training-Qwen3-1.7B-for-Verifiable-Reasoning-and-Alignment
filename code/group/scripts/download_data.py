"""
Download group model datasets from HuggingFace.

Usage (from code/group/):

  python scripts/download_data.py
"""

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

DATASET_REPO = "cs-552-2026-camykaz/group_data"
MODEL_REPO   = "cs-552-2026-camykaz/group_model_lora_direct_v1"
GK_MODEL_REPO = "cyrineakrout/gk-models"
MATH_DATA_REPO = "cs-552-2026-camykaz/math_model"
SAFETY_DATA_REPO = "zguir23/safety_data"
DATA_DIR = Path("data")

# Training data from group_data dataset repo
TRAIN_FILES = [
    "group_sft_final_train.jsonl",
    "group_sft_final_val.jsonl",
]

# Eval file from the model repo (under lora_direct_v1/eval/)
EVAL_FILES = [
    ("lora_direct_v1/eval/group_eval.jsonl", "group_eval.jsonl"),
]

# Source GK SFT file used when rebuilding the combined group dataset.
GK_TRAIN_FILE = ("datasets/sft_train_v12.jsonl", "gk_sft_train.jsonl")

# Source math SFT file used when rebuilding the combined group dataset.
MATH_TRAIN_FILE = ("v4_strong_sft/train.jsonl", "math_v4_strong_sft_train.jsonl")

# Source safety SFT file used when rebuilding the combined group dataset.
SAFETY_TRAIN_FILE = ("exp3_clean_mix_train.jsonl", "safety_combined_train.jsonl")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading training data from {DATASET_REPO} → {DATA_DIR}/")
    for f in TRAIN_FILES:
        dest = DATA_DIR / f
        if dest.exists():
            print(f"  [skip] {f}")
            continue
        print(f"  {f} ...", end=" ", flush=True)
        try:
            hf_hub_download(repo_id=DATASET_REPO, filename=f,
                            repo_type="dataset", local_dir=str(DATA_DIR))
            print("done")
        except Exception:
            print("SKIP (not found)")

    print(f"\nDownloading eval data from {MODEL_REPO} → {DATA_DIR}/")
    for hf_file, local_name in EVAL_FILES:
        dest = DATA_DIR / local_name
        if dest.exists():
            print(f"  [skip] {local_name}")
            continue
        print(f"  {local_name} ...", end=" ", flush=True)
        try:
            downloaded = hf_hub_download(repo_id=MODEL_REPO, filename=hf_file,
                                         repo_type="model", local_dir=str(DATA_DIR / ".cache"))
            shutil.copy2(downloaded, str(dest))
            print("done")
        except Exception:
            print("SKIP (not found)")

    hf_file, local_name = GK_TRAIN_FILE
    dest = DATA_DIR / local_name
    print(f"\nDownloading GK training data from {GK_MODEL_REPO} → {dest}")
    if dest.exists():
        print(f"  [skip] {local_name}")
    else:
        print(f"  {local_name} ...", end=" ", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=GK_MODEL_REPO,
                filename=hf_file,
                repo_type="model",
                local_dir=str(DATA_DIR / ".cache"),
            )
            shutil.copy2(downloaded, str(dest))
            print("done")
        except Exception:
            print("SKIP (not found)")

    hf_file, local_name = MATH_TRAIN_FILE
    dest = DATA_DIR / local_name
    print(f"\nDownloading math training data from {MATH_DATA_REPO} → {dest}")
    if dest.exists():
        print(f"  [skip] {local_name}")
    else:
        print(f"  {local_name} ...", end=" ", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=MATH_DATA_REPO,
                filename=hf_file,
                repo_type="dataset",
                local_dir=str(DATA_DIR / ".cache"),
            )
            shutil.copy2(downloaded, str(dest))
            print("done")
        except Exception:
            print("SKIP (not found)")

    hf_file, local_name = SAFETY_TRAIN_FILE
    dest = DATA_DIR / local_name
    print(f"\nDownloading safety training data from {SAFETY_DATA_REPO} → {dest}")
    if dest.exists():
        print(f"  [skip] {local_name}")
    else:
        print(f"  {local_name} ...", end=" ", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=SAFETY_DATA_REPO,
                filename=hf_file,
                repo_type="dataset",
                local_dir=str(DATA_DIR / ".cache"),
            )
            shutil.copy2(downloaded, str(dest))
            print("done")
        except Exception:
            print("SKIP (not found)")

    print("\nDone.")


if __name__ == "__main__":
    main()
