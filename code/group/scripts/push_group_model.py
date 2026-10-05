"""
Push a merged group model to HuggingFace.

Pushes to a named branch (default: group_model) on the target repo.
Use --also-main to also update main in the same run.

Usage (from code/group/):

  # Push to the official team repo
  python scripts/push_group_model.py --model-dir group/outputs/distillation_sft_grpo_merged

  # Push to a specific branch
  python scripts/push_group_model.py --model-dir ... --branch my_run_v2

  # Push to mmsellam's personal repo
  python scripts/push_group_model.py --model-dir ... --repo mmsellam

  # Also update main (CI-active model)
  python scripts/push_group_model.py --model-dir ... --also-main

Repo IDs come from env vars (override as needed):
  HF_GROUP_REPO         official repo (default: cs-552-2026-camykaz/group_model)
  HF_GROUP_MMSELLAM_REPO mmsellam repo (default: mmsellam/group_model)
"""

import argparse
import os
import sys
from pathlib import Path

OFFICIAL_REPO_ID = os.environ.get("HF_GROUP_REPO",          "cs-552-2026-camykaz/group_model")
MMSELLAM_REPO_ID = os.environ.get("HF_GROUP_MMSELLAM_REPO", "mmsellam/group_model")


def check_required_files(model_dir: Path) -> None:
    required = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]

    has_single  = (model_dir / "model.safetensors").exists()
    has_sharded = (model_dir / "model.safetensors.index.json").exists()
    if not has_single and not has_sharded:
        missing.append("model.safetensors or model.safetensors.index.json")

    if missing:
        raise FileNotFoundError(
            "Missing required files in model dir:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + f"\n\nChecked: {model_dir}"
        )
    print("All required checkpoint files present.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Path to merged model directory.")
    parser.add_argument("--branch", default="group_model",
                        help="HF branch to push to (default: group_model).")
    parser.add_argument("--also-main", action="store_true",
                        help="Also push to main in the same run (updates CI-active model).")
    parser.add_argument("--repo", choices=("official", "mmsellam"), default="official",
                        help="Target repo (default: official).")
    args = parser.parse_args()

    repo_id   = OFFICIAL_REPO_ID if args.repo == "official" else MMSELLAM_REPO_ID
    private   = args.repo == "mmsellam"
    model_dir = Path(args.model_dir)
    branches  = [args.branch]
    if args.also_main and args.branch != "main":
        branches.append("main")

    print("=" * 100)
    print("Pushing group model to HuggingFace")
    print("=" * 100)
    print(f"Model dir : {model_dir}")
    print(f"Repo      : {repo_id} ({'private' if private else 'public'})")
    print(f"Branches  : {branches}")
    print("=" * 100)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    check_required_files(model_dir)

    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()

    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

        for branch in branches:
            if branch == "main":
                print(f"\nWARNING: pushing to main — this will update the CI-active model.")
            if branch != "main":
                try:
                    api.create_branch(repo_id=repo_id, repo_type="model", branch=branch)
                except Exception:
                    pass  # branch already exists

            print(f"\n→ https://huggingface.co/{repo_id}/tree/{branch}")
            upload_folder(
                folder_path=str(model_dir),
                repo_id=repo_id,
                repo_type="model",
                revision=branch,
                commit_message=f"Upload group model to {branch}",
                ignore_patterns=[
                    "*.log", "checkpoint-*", "runs/*",
                    "__pycache__/*", ".ipynb_checkpoints/*",
                ],
            )
            print(f"  Upload to '{branch}' complete.")

    except HfHubHTTPError as e:
        print("\nUpload failed with a HuggingFace HTTP error.")
        print("Most likely cause: no valid token or token has no write access.")
        print(f"Set HF_TOKEN env var before running this script.")
        print(f"Repo: {repo_id}")
        raise e

    print("=" * 100)
    print("Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
