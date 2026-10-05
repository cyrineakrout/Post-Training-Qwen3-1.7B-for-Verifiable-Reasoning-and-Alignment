import subprocess
import sys
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

STEPS = [
    ["math/scripts/v1_early_sft/prepare_math_v1_early_sft.py"],
    ["math/scripts/v2_revised_sft/prepare_math_v2_revised_sft.py"],
    ["math/scripts/v3_curated_sft/prepare_math_v3_curated_sft.py"],
    ["math/scripts/v4_strong_sft/prepare_math_v4_strong_sft.py"],
    ["math/scripts/v4_strong_sft/prepare_math_v4_grpo_ref.py"],
    ["math/scripts/v5_all_solutions_sft/prepare_math_v5_all_solutions_sft.py"],
    ["math/scripts/v6_50k_long/prepare_math_v6_50k_long.py"],
    ["math/scripts/v7_100k/prepare_math_v7_100k.py"],
]


def main() -> None:
    steps = [list(command) for command in STEPS]
    if os.environ.get("MATH_SKIP_HF_UPLOAD", "0") == "1":
        steps[4].extend(["--input", "math/data/math_v4_strong_sft_train_with_meta.jsonl"])
        steps[5].extend(
            [
                "--train_in",
                "math/data/math_v4_strong_sft_train_with_meta.jsonl",
                "--val_in",
                "math/data/math_v4_strong_sft_val_with_meta.jsonl",
            ]
        )

    for index, command in enumerate(steps, start=1):
        script = command[0]
        print("=" * 100, flush=True)
        print(f"[{index}/{len(steps)}] Preparing and publishing {script}", flush=True)
        print("=" * 100, flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
