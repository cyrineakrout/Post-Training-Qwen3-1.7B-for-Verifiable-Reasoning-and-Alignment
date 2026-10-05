"""
Download all Safety datasets and models from HuggingFace.

Run this once before training or evaluating any experiment.
Datasets are saved to data/, models to models/.

Usage (from code/safety/):

  # Download everything (datasets + all models)
  python scripts/download_data.py

  # Download only the eval set + best model (fastest, enough to run eval)
  python scripts/download_data.py --eval_only

  # Download a single specific model
  python scripts/download_data.py --model exp3_clean_mix_32k_best
"""

import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

DATASET_REPO = "zguir23/safety_data"
MODEL_REPO   = "zguir23/safety-models"
DATA_DIR     = Path("data")
MODELS_DIR   = Path("models")

DATASET_FILES = [
    "safety_eval_v2_clean.jsonl",
    "exp2_bt_siqa_ethics_civil_train.jsonl",
    "exp3_clean_mix_train.jsonl",
    "exp3_clean_mix_dev.jsonl",
    "exp4_clean_mix_wildguardmix_train.jsonl",
    "exp4_clean_mix_wildguardmix_dev.jsonl",
    "exp5_large_mix_bbq_train.jsonl",
    "exp5_large_mix_bbq_dev.jsonl",
    "thinking_t2_synthetic_cot_train.jsonl",
    "thinking_t2_synthetic_cot_dev.jsonl",
    "thinking_t3_real_cot_train.jsonl",
    "dpo_dataset.jsonl",
]

EVAL_FILES = ["safety_eval_v2_clean.jsonl"]

ALL_MODELS = [
    "exp1_beavertails_5k",
    "exp2_bt_siqa_ethics_civil_10k",
    "exp3_clean_mix_32k_best",
    "exp4_clean_mix_wildguardmix_48k",
    "exp5_large_mix_bbq_58k",
    "dpo_d1_single_pass",
    "dpo_d2_iterative_collapsed",
    "dpo_d3_iterative_softer_collapsed",
    "thinking_t1_no_reasoning_traces",
    "thinking_t2_synthetic_cot",
    "thinking_t3_real_cot_distillation",
]


def download_dataset_file(filename: str):
    dest = DATA_DIR / filename
    if dest.exists():
        print(f"  [skip] {filename}")
        return
    print(f"  {filename} ...", end=" ", flush=True)
    try:
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        print("done")
    except Exception as e:
        print(f"SKIP (not found on HF)")


def download_model(model_name: str):
    dest = MODELS_DIR / model_name
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {model_name}")
        return
    print(f"  {model_name} ...", end=" ", flush=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=f"{model_name}/**",
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False,
    )
    print("done")


def main():
    parser = argparse.ArgumentParser(description="Download safety datasets and models from HuggingFace.")
    parser.add_argument("--eval_only", action="store_true",
                        help="Download only eval set + best model (exp3). Fastest option.")
    parser.add_argument("--model", type=str, default=None,
                        help="Download a single model by name.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        # Single model download
        if args.model not in ALL_MODELS:
            print(f"[ERROR] Unknown model '{args.model}'. Choose from:")
            for m in ALL_MODELS:
                print(f"  {m}")
            return
        print(f"Downloading model: {args.model}")
        download_model(args.model)
        print(f"\nReady. Run eval:")
        print(f"  python scripts/eval_safety_accuracy.py \\")
        print(f"      --model_name models/{args.model} \\")
        print(f"      --data_path data/safety_eval_v2_clean.jsonl --no_wandb")

    elif args.eval_only:
        # Eval set + best model only
        print(f"=== Downloading eval set ===")
        download_dataset_file("safety_eval_v2_clean.jsonl")
        print(f"\n=== Downloading best model (exp3_clean_mix_32k_best) ===")
        download_model("exp3_clean_mix_32k_best")
        print(f"\nReady. Run eval:")
        print(f"  python scripts/eval_safety_accuracy.py \\")
        print(f"      --model_name models/exp3_clean_mix_32k_best \\")
        print(f"      --data_path data/safety_eval_v2_clean.jsonl --no_wandb")

    else:
        # Download everything
        print(f"=== Downloading all datasets from {DATASET_REPO} ===")
        for f in DATASET_FILES:
            download_dataset_file(f)

        print(f"\n=== Downloading all models from {MODEL_REPO} ===")
        for m in ALL_MODELS:
            download_model(m)

        print("\nAll done!")
        print("Run eval on best model:")
        print("  python scripts/eval_safety_accuracy.py \\")
        print("      --model_name models/exp3_clean_mix_32k_best \\")
        print("      --data_path data/safety_eval_v2_clean.jsonl --no_wandb")


if __name__ == "__main__":
    main()
