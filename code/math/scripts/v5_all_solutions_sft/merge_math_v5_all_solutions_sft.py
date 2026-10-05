import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[3]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_ADAPTER_DIR = str(ROOT_DIR / "math" / "outputs" / "math_v5_all_solutions_sft_lora_r96_e08")
DEFAULT_OUTPUT_DIR = str(ROOT_DIR / "math" / "outputs" / "math_v5_all_solutions_sft_merged_r96_e08")

BASE_MODEL = os.environ.get("MATH_BASE_MODEL", DEFAULT_BASE_MODEL)
ADAPTER_DIR = os.environ.get("MATH_ADAPTER_DIR", DEFAULT_ADAPTER_DIR)
OUTPUT_DIR = os.environ.get("MATH_MERGED_DIR", DEFAULT_OUTPUT_DIR)


def main() -> None:
    print("=" * 100)
    print("[math-v5] Merge all-solutions SFT adapter into base model")
    print("=" * 100)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Adapter dir: {ADAPTER_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")

    adapter_dir = Path(ADAPTER_DIR)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[math-v5] Merged model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
