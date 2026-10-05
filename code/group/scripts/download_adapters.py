"""
Download the lora_direct_v1 domain adapters from HuggingFace.

The adapters were trained on Qwen/Qwen3-1.7B for four domains
(safety, general knowledge, multilingual, math) and pushed to:
  cs-552-2026-camykaz/group_model_lora_direct_v1

Usage (run from code/group/):
  python scripts/download_adapters.py

Downloads to:
  adapters/safety/
  adapters/general_knowledge/
  adapters/multilingual/
  adapters/math/
"""

from pathlib import Path
from huggingface_hub import snapshot_download

HF_REPO = "cs-552-2026-camykaz/group_model_lora_direct_v1"

ADAPTERS = {
    "safety":            "lora_direct_v1/adapters/safety",
    "general_knowledge": "lora_direct_v1/adapters/general_knowledge",
    "multilingual":      "lora_direct_v1/adapters/multilingual",
    "math":              "lora_direct_v1/adapters/math",
}

LOCAL_DIR = Path(__file__).resolve().parents[1] / "adapters"


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading adapters from {HF_REPO}")

    for name, hf_prefix in ADAPTERS.items():
        dest = LOCAL_DIR / name
        if dest.exists() and any(dest.iterdir()):
            print(f"  [skip] {name}  (already at {dest})")
            continue

        print(f"  {name} ...", end=" ", flush=True)
        downloaded = snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=[
                f"{hf_prefix}/adapter_config.json",
                f"{hf_prefix}/adapter_model.safetensors",
                f"{hf_prefix}/tokenizer.json",
                f"{hf_prefix}/tokenizer_config.json",
                f"{hf_prefix}/chat_template.jinja",
            ],
            local_dir=str(dest.parent / "_tmp_download"),
            local_dir_use_symlinks=False,
        )

        src = Path(downloaded) / hf_prefix
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            target = dest / f.name
            if not target.exists():
                import shutil
                shutil.copy2(str(f), str(target))

        import shutil
        shutil.rmtree(str(dest.parent / "_tmp_download"), ignore_errors=True)
        print("done")

    print(f"\nAll adapters downloaded to {LOCAL_DIR}/")
    print("\nNext: merge adapters")
    print("  python scripts/merge_adapters.py \\")
    print(f"      --safety_adapter      adapters/safety \\")
    print(f"      --gk_adapter          adapters/general_knowledge \\")
    print(f"      --ml_adapter          adapters/multilingual \\")
    print(f"      --math_adapter        adapters/math \\")
    print(f"      --output_dir          models/lora_direct_v1_reproduced")


if __name__ == "__main__":
    main()
