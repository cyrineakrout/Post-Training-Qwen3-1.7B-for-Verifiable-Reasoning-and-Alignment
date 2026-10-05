"""
Download trained models from HuggingFace into outputs/.

Usage (from code/multilinguality/):

  python scripts/download_models.py                        # all models
  python scripts/download_models.py --model sft_grpo_ton_merged  # one model
"""

import argparse
import json
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]

with open(ML_ROOT / "config.json") as f:
    CFG = json.load(f)

MODEL_REPO  = CFG["hf_model"]["repo"]
OUTPUTS_DIR = ML_ROOT / "outputs"


def download(model_name: str) -> None:
    from huggingface_hub import snapshot_download
    # Ensure local folder ends with _merged so evaluate.py finds it
    local_name = model_name if model_name.endswith("_merged") else model_name + "_merged"
    dest = OUTPUTS_DIR / local_name
    if dest.exists() and any(dest.iterdir()):
        print(f"  [skip] {local_name}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  {model_name} → {local_name} ...", end=" ", flush=True)
    try:
        snapshot_download(repo_id=MODEL_REPO, revision=model_name,
                          local_dir=str(dest))
        print("done")
    except Exception as e:
        print(f"FAILED ({e})")


def list_models() -> list[str]:
    from huggingface_hub import HfApi
    refs = HfApi().list_repo_refs(repo_id=MODEL_REPO, repo_type="model")
    return sorted(b.name for b in refs.branches if b.name != "main")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Download a single model folder")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model:
        print(f"\n=== Downloading {args.model} from {MODEL_REPO} ===")
        download(args.model)
    else:
        models = list_models()
        print(f"\n=== Downloading all {len(models)} models from {MODEL_REPO} ===")
        for m in models:
            download(m)

    print("\nDone.")


if __name__ == "__main__":
    main()
