import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import publish_dataset_version, publish_final_evaluation


DATA_DIR = ROOT_DIR / "math" / "data"

VERSIONS = {
    "v1_early_sft": {
        "train.jsonl": DATA_DIR / "math_v1_early_sft_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v1_early_sft_val.jsonl",
    },
    "v2_revised_sft": {
        "train.jsonl": DATA_DIR / "math_v2_revised_sft_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v2_revised_sft_val.jsonl",
    },
    "v3_curated_sft": {
        "train.jsonl": DATA_DIR / "math_v3_curated_sft_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v3_curated_sft_val.jsonl",
        "train_with_meta.jsonl": DATA_DIR / "math_v3_curated_sft_train_with_meta.jsonl",
        "validation_with_meta.jsonl": DATA_DIR / "math_v3_curated_sft_val_with_meta.jsonl",
    },
    "v4_strong_sft": {
        "train.jsonl": DATA_DIR / "math_v4_strong_sft_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v4_strong_sft_val.jsonl",
        "train_with_meta.jsonl": DATA_DIR / "math_v4_strong_sft_train_with_meta.jsonl",
        "validation_with_meta.jsonl": DATA_DIR / "math_v4_strong_sft_val_with_meta.jsonl",
        "grpo_ref.jsonl": DATA_DIR / "math_v4_grpo_ref_train_5000.jsonl",
    },
    "v5_all_solutions_sft": {
        "train.jsonl": DATA_DIR / "math_v5_all_solutions_sft_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v5_all_solutions_sft_val.jsonl",
    },
    "v6_50k_long": {
        "train.jsonl": DATA_DIR / "math_v6_50k_long_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v6_50k_long_val.jsonl",
        "manifest.json": DATA_DIR / "math_v6_50k_long_manifest.json",
    },
    "v7_100k": {
        "train.jsonl": DATA_DIR / "math_v7_100k_train.jsonl",
        "validation.jsonl": DATA_DIR / "math_v7_100k_val.jsonl",
        "manifest.json": DATA_DIR / "math_v7_100k_manifest.json",
    },
}


def main() -> None:
    for version, files in VERSIONS.items():
        publish_dataset_version(version, files)

    publish_final_evaluation(
        DATA_DIR / "math_benchmark_external_v1.jsonl",
        DATA_DIR / "math_benchmark_external_v1_manifest.json",
    )


if __name__ == "__main__":
    main()
