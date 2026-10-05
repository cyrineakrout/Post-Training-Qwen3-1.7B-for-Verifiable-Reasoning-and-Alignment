"""Validate math pipelines without starting training, merging, or vLLM inference."""

import argparse
import ast
import importlib
import inspect
import json
import os
import re
import shlex
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT_DIR / "math" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from hf_dataset_utils import dataset_repo_id
from runtime_utils import resolve_attention_implementation

HUB_FILES = {
    "v1_early_sft": ["train.jsonl", "validation.jsonl"],
    "v2_revised_sft": ["train.jsonl", "validation.jsonl"],
    "v3_curated_sft": ["train.jsonl", "validation.jsonl", "train_with_meta.jsonl", "validation_with_meta.jsonl"],
    "v4_strong_sft": ["train.jsonl", "validation.jsonl", "train_with_meta.jsonl", "validation_with_meta.jsonl", "grpo_ref.jsonl"],
    "v5_all_solutions_sft": ["train.jsonl", "validation.jsonl"],
    "v6_50k_long": ["train.jsonl", "validation.jsonl", "manifest.json"],
    "v7_100k": ["train.jsonl", "validation.jsonl", "manifest.json"],
    "final_evaluation": ["math_benchmark_external_v1.jsonl", "manifest.json"],
}
DEFAULT_SUBMISSION_MODEL_REPO = "cs-552-2026-camykaz/math_model"
EXPERIMENT_CONFIG = ROOT_DIR / "math" / "configs" / "experiments.json"
SUBMISSION_MODEL_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}
MERGE_INPUTS = {
    "v1": "math/outputs/math_v1_early_sft_lora",
    "v2": "math/outputs/math_v2_revised_sft_lora",
    "v3": "math/outputs/math_v3_curated_sft_lora",
    "v4": "math/outputs/math_v4_strong_sft_lora",
    "v4_grpo": "math/outputs/math_v4_grpo_ref",
    "v5": "math/outputs/math_v5_all_solutions_sft_lora_r96_e08",
    "v6": "math/outputs/math_v6_50k_long_lora/checkpoint-1500",
}
def check_dependencies() -> None:
    print("\n[dependencies]")
    missing = []
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl", "huggingface_hub", "vllm"):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError:
            missing.append(name)
            print(f"MISSING {name}")
            continue
        print(f"OK {name} {getattr(module, '__version__', '')}".rstrip())
    if missing:
        raise ModuleNotFoundError(
            f"Missing validation dependencies: {', '.join(missing)}. "
            "Install them with: pip install -r math/requirements.txt"
        )
    print(f"OK attention backend: {resolve_attention_implementation()}")


def check_scripts() -> None:
    from transformers import Trainer, TrainingArguments
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    trainer_classes = {
        cls.__name__: cls
        for cls in (Trainer, TrainingArguments, SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig)
    }

    print("\n[scripts]")
    for path in sorted(SCRIPTS_DIR.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
    print("OK all Python scripts compile")
    for path in sorted(SCRIPTS_DIR.glob("v*/*.py")):
        if not path.name.startswith("train_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            cls = trainer_classes.get(node.func.id)
            if cls is None:
                continue
            signature = inspect.signature(cls)
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            unsupported = [k.arg for k in node.keywords if k.arg and k.arg not in signature.parameters and not accepts_kwargs]
            if unsupported:
                raise TypeError(f"{path}: {node.func.id} unsupported arguments: {unsupported}")
        print(f"OK trainer API: {path.relative_to(ROOT_DIR)}")


def check_readme_commands() -> None:
    print("\n[readme commands]")
    readme = ROOT_DIR / "math" / "README.md"
    text = readme.read_text(encoding="utf-8")
    commands = []

    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        logical_line = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                logical_line += line[:-1].strip() + " "
                continue
            commands.append((logical_line + line).strip())
            logical_line = ""

    checked = 0
    for command in commands:
        command_without_redirect = command.split("2>&1", 1)[0].strip()
        parts = shlex.split(command_without_redirect)
        while parts and "=" in parts[0] and not parts[0].startswith("--"):
            parts.pop(0)
        if len(parts) < 2 or parts[0] not in {"python", "bash"}:
            continue
        script = ROOT_DIR / parts[1]
        if not script.is_file():
            raise FileNotFoundError(f"README command references missing script: {parts[1]}")
        checked += 1

    print(f"OK {checked} README Python/bash command references")


def check_experiment_config() -> None:
    print("\n[experiment config]")
    config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    required = {"description", "hf_subfolder", "train_script", "merge_script", "eval_max_tokens", "eval_max_model_len"}
    for name, experiment in config["experiments"].items():
        missing = required - set(experiment)
        if missing:
            raise ValueError(f"{name}: missing experiment config keys {sorted(missing)}")
        if experiment["eval_max_tokens"] <= 0 or experiment["eval_max_model_len"] <= 0:
            raise ValueError(f"{name}: evaluation budgets must be positive")
        if experiment["eval_max_tokens"] >= experiment["eval_max_model_len"]:
            raise ValueError(f"{name}: eval_max_model_len must exceed eval_max_tokens")
        for key in ("train_script", "merge_script"):
            if not (ROOT_DIR / experiment[key]).is_file():
                raise FileNotFoundError(f"{name}: missing {key}: {experiment[key]}")
    print(f"OK {len(config['experiments'])} published experiment mappings and evaluation profiles")


def check_published_models() -> None:
    print("\n[published experiment models]")
    config = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    files = set(HfApi().list_repo_files(config["model_repo"], repo_type="model"))
    for name, experiment in config["experiments"].items():
        prefix = experiment["hf_subfolder"] + "/"
        required = {prefix + filename for filename in SUBMISSION_MODEL_FILES}
        missing = sorted(required - files)
        weights = [path for path in files if path.startswith(prefix) and path.endswith(".safetensors")]
        if missing or not weights:
            raise FileNotFoundError(f"{name}: missing files={missing}, weights={weights}")
        print(f"OK {name}: {experiment['hf_subfolder']}")


def first_jsonl_row(repo_id: str, version: str, filename: str) -> dict:
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=f"{version}/{filename}")
    with open(path, encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def check_hub() -> None:
    print("\n[hugging-face data]")
    repo_id = dataset_repo_id()
    files = set(HfApi().list_repo_files(repo_id, repo_type="dataset"))
    missing = [f"{v}/{f}" for v, names in HUB_FILES.items() for f in names if f"{v}/{f}" not in files]
    if missing:
        raise FileNotFoundError(f"Missing Hub files in {repo_id}: {missing}")
    print(f"OK all {sum(map(len, HUB_FILES.values()))} required files exist in {repo_id}")
    for version, filenames in HUB_FILES.items():
        filename = next((name for name in filenames if name.endswith(".jsonl")), None)
        if filename is None:
            continue
        row = first_jsonl_row(repo_id, version, filename)
        expected = {"messages", "metadata"} if version == "final_evaluation" else {"messages"}
        if not expected.issubset(row):
            raise ValueError(f"{version}/{filename}: expected keys {sorted(expected)}, got {sorted(row)}")
        print(f"OK schema: {version}/{filename}")


def check_submission_model() -> None:
    print("\n[public submission model]")
    repo_id = os.environ.get("MATH_MODEL_REPO", DEFAULT_SUBMISSION_MODEL_REPO)
    files = set(HfApi().list_repo_files(repo_id, repo_type="model"))
    missing = sorted(SUBMISSION_MODEL_FILES - files)
    weights = sorted(name for name in files if name.endswith(".safetensors"))
    if missing:
        raise FileNotFoundError(f"Missing required model files in {repo_id}: {missing}")
    if not weights:
        raise FileNotFoundError(f"No safetensors weights found in {repo_id}")

    template_path = hf_hub_download(repo_id=repo_id, repo_type="model", filename="chat_template.jinja")
    template = Path(template_path).read_text(encoding="utf-8")
    if "\\boxed" not in template or "add_generation_prompt" not in template:
        raise ValueError(f"{repo_id}/chat_template.jinja does not satisfy the math submission contract")

    print(f"OK required root files and {len(weights)} safetensors weight file(s) in {repo_id}")


def check_merge_inputs() -> None:
    print("\n[local merge inputs]")
    for version, relative in MERGE_INPUTS.items():
        adapter_config = ROOT_DIR / relative / "adapter_config.json"
        state = "READY" if adapter_config.exists() else "not present yet"
        print(f"{state}: {version} -> {adapter_config.relative_to(ROOT_DIR)}")
    print("INFO v7 merge input is selected explicitly with --checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hub",
        "--skip_hub",
        dest="skip_hub",
        action="store_true",
        help="Skip live Hugging Face checks.",
    )
    args = parser.parse_args()
    check_readme_commands()
    check_experiment_config()
    check_dependencies()
    check_scripts()
    if not args.skip_hub:
        check_hub()
        check_submission_model()
        check_published_models()
    check_merge_inputs()
    print("\nVALIDATION PASSED: no training, merging, or vLLM inference was started.")


if __name__ == "__main__":
    main()
