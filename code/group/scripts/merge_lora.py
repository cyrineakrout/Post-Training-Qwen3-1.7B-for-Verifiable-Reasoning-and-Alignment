import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

GROUP_ROOT = Path(__file__).resolve().parents[1]
SCRATCH    = GROUP_ROOT.parent.parent


BASE_MODEL = os.environ.get("GROUP_BASE_MODEL", "Qwen/Qwen3-1.7B")
ADAPTER_DIR = os.environ.get("GROUP_ADAPTER_DIR", "group/outputs/group_lora_final_r96_e06")
MERGED_DIR = os.environ.get("GROUP_MERGED_DIR", "group/outputs/group_model_merged_final")


def main():
    print("=" * 100)
    print("[group] Merge LoRA")
    print("=" * 100)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Adapter dir: {ADAPTER_DIR}")
    print(f"Merged dir:  {MERGED_DIR}")

    merged_dir = Path(MERGED_DIR)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_DIR,
        trust_remote_code=True,
        use_fast=True,
    )

    model.save_pretrained(
        MERGED_DIR,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(MERGED_DIR)

    print("=" * 100)
    print(f"[group] Merged model saved to {MERGED_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
