#!/usr/bin/env python3
"""
Download all training datasets and the evaluation file from HuggingFace.

Usage:
    python scripts/download_data.py

Files are saved to  general_knowledge/data/datasets/
"""
from pathlib import Path

HF_MODEL_REPO = "cyrineakrout/gk-models"
HF_DATA_REPO  = "cyrineakrout/gk_data_backup"

# (repo_id, filename, repo_type)
DATASET_FILES = [
    # SFT training datasets (model repo, nested under datasets/)
    (HF_MODEL_REPO, "datasets/eval_mmlu_mmlupro_supergpqa_nemotron_provided.jsonl", "model"),
    (HF_MODEL_REPO, "datasets/sft_train_v6.jsonl",  "model"),
    (HF_MODEL_REPO, "datasets/sft_train_v10.jsonl", "model"),
    (HF_MODEL_REPO, "datasets/sft_train_v11.jsonl", "model"),
    (HF_MODEL_REPO, "datasets/sft_train_v12.jsonl", "model"),
    (HF_MODEL_REPO, "datasets/sft_train_v14.jsonl", "model"),
    # GRPO training dataset (dataset repo, flat file)
    (HF_DATA_REPO,  "grpo_hard_gpqa_supergpqa_hard.jsonl", "dataset"),
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "datasets"


def main():
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading datasets to: {DATA_DIR}\n")

    for repo_id, hf_path, repo_type in DATASET_FILES:
        filename = Path(hf_path).name
        local_path = DATA_DIR / filename
        if local_path.exists():
            print(f"[skip] Already exists: {local_path}")
            continue
        print(f"[download] {repo_id}/{hf_path} ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=hf_path,
            repo_type=repo_type,
            local_dir=str(DATA_DIR.parent),
        )
        # hf_hub_download preserves path structure under local_dir.
        # For model repo files nested under datasets/, the file lands at DATA_DIR/filename.
        # For dataset repo flat files, it lands directly at DATA_DIR.parent/filename — move it.
        flat_src = DATA_DIR.parent / filename
        if flat_src.exists() and flat_src != local_path:
            flat_src.rename(local_path)
        print(f"  -> saved to {local_path}")

    print("\nAll datasets ready.")


if __name__ == "__main__":
    main()
