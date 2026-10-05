import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

ROOT_DIR = Path(__file__).resolve().parents[3]
MATH_DIR = ROOT_DIR / "math"
OUTPUT_DIR = MATH_DIR / "outputs"

BASE_MODEL = "Qwen/Qwen3-1.7B"
LORA_DIR = OUTPUT_DIR / "math_v1_early_sft_lora"
MERGED_DIR = OUTPUT_DIR / "math_v1_early_sft_merged"


def force_thinking_off(tokenizer):
    """
    For Milestone 2, we prioritize reliable boxed answers.
    Non-thinking mode avoids long <think> loops and improves answer extraction.
    """
    if tokenizer.chat_template is None:
        raise ValueError("Tokenizer has no chat_template.")

    marker = "{%- set enable_thinking = false %}"

    if "{%- set enable_thinking = true %}" in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{%- set enable_thinking = true %}",
            marker,
        )
    elif marker not in tokenizer.chat_template:
        tokenizer.chat_template = marker + "\n" + tokenizer.chat_template

    return tokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        str(LORA_DIR),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer = force_thinking_off(tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    model = model.merge_and_unload()

    model.save_pretrained(
        str(MERGED_DIR),
        safe_serialization=True,
        max_shard_size="4GB",
    )

    tokenizer.save_pretrained(str(MERGED_DIR))

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.1,
        max_new_tokens=2048,
        bos_token_id=151643,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
    )

    generation_config.save_pretrained(str(MERGED_DIR))

    print(f"Merged model saved to {MERGED_DIR}")


if __name__ == "__main__":
    main()