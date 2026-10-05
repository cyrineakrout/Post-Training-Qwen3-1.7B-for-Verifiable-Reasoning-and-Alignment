import os
import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import create_repo, hf_hub_download, login, upload_folder


DEFAULT_DATASET_REPO = "cs-552-2026-camykaz/math_model"
FINAL_EVAL_VERSION = "final_evaluation"
FINAL_EVAL_FILE = "math_benchmark_external_v1.jsonl"


def dataset_repo_id() -> str:
    return os.environ.get("MATH_DATASET_REPO", DEFAULT_DATASET_REPO)


def publish_dataset_version(
    version: str,
    files: dict[str, str | Path],
    *,
    commit_message: str | None = None,
) -> None:
    """Upload one version's generated dataset files into its Hub subfolder."""
    if os.environ.get("MATH_SKIP_HF_UPLOAD", "0") == "1":
        print("Skipping Hugging Face upload because MATH_SKIP_HF_UPLOAD=1")
        return

    resolved = {name: Path(path) for name, path in files.items()}
    missing = [str(path) for path in resolved.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot publish missing dataset files: {missing}")

    repo_id = dataset_repo_id()
    login(skip_if_logged_in=True)
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"math-{version}-") as tmp:
        upload_dir = Path(tmp)
        for hub_name, source in resolved.items():
            destination = upload_dir / hub_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        upload_folder(
            folder_path=str(upload_dir),
            path_in_repo=version,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message or f"Publish math dataset {version}",
        )

    print(f"Published {version} to https://huggingface.co/datasets/{repo_id}/tree/main/{version}")


def load_dataset_version(
    version: str,
    *,
    files: dict[str, str] | None = None,
):
    """Load a prepared math dataset version directly from the shared Hub repo."""
    split_files = files or {
        "train": "train.jsonl",
        "validation": "validation.jsonl",
    }
    data_files = {split: f"{version}/{name}" for split, name in split_files.items()}
    return load_dataset(dataset_repo_id(), data_files=data_files)


def download_dataset_version_file(version: str, filename: str) -> str:
    """Download one generated file for a downstream preparation step."""
    return hf_hub_download(
        repo_id=dataset_repo_id(),
        filename=f"{version}/{filename}",
        repo_type="dataset",
    )


def publish_final_evaluation(eval_file: str | Path, manifest_file: str | Path) -> None:
    publish_dataset_version(
        FINAL_EVAL_VERSION,
        {
            FINAL_EVAL_FILE: eval_file,
            "manifest.json": manifest_file,
        },
        commit_message="Publish final math evaluation benchmark",
    )


def download_final_evaluation() -> str:
    return download_dataset_version_file(FINAL_EVAL_VERSION, FINAL_EVAL_FILE)
