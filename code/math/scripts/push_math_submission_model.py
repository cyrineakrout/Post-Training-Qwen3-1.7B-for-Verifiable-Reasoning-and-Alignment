"""Validate and publish one selected merged math model to Hugging Face."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_REPO_ID = "cs-552-2026-camykaz/math_model"
REQUIRED_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    files = {path.name for path in model_dir.iterdir() if path.is_file()}
    missing = sorted(REQUIRED_FILES - files)
    if missing:
        raise FileNotFoundError(f"Missing required model files in {model_dir}: {missing}")
    if not any(name.endswith(".safetensors") for name in files):
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")

    template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    if "\\boxed" not in template or "add_generation_prompt" not in template:
        raise ValueError(f"{model_dir}/chat_template.jinja does not satisfy the math submission contract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Merged model directory to publish.")
    parser.add_argument(
        "--repo_id",
        default=os.environ.get("MATH_MODEL_REPO", DEFAULT_REPO_ID),
        help="Destination Hugging Face model repository.",
    )
    parser.add_argument("--private", action="store_true", help="Create the destination repository as private.")
    parser.add_argument("--path_in_repo", default="", help="Optional destination subfolder inside the model repository.")
    parser.add_argument(
        "--commit_message",
        default="Publish selected math submission model",
        help="Hugging Face commit message.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    validate_model_dir(model_dir)

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=args.path_in_repo or None,
        commit_message=args.commit_message,
    )
    destination = f"https://huggingface.co/{args.repo_id}"
    if args.path_in_repo:
        destination += f"/tree/main/{args.path_in_repo}"
    print(f"Published {model_dir} to {destination}")


if __name__ == "__main__":
    main()
