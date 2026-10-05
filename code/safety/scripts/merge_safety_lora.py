import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Import force_safety_system_prompt from the training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_safety_lora import force_safety_system_prompt

_FORCE_NO_THINK_BEGIN = "{# === FORCE_QWEN_NO_THINK_BEGIN === #}"
_FORCE_NO_THINK_END = "{# === FORCE_QWEN_NO_THINK_END === #}"


def force_qwen_no_think(tokenizer):
    """
    Force Qwen3 chat template to disable thinking by default.
    Injects {%- set enable_thinking = false %} at the top of the Jinja
    template so thinking is disabled regardless of what the caller passes,
    even if CI calls apply_chat_template with enable_thinking=True.
    """
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return tokenizer

    template = tokenizer.chat_template

    # Remove previous patch if already present (idempotent).
    if _FORCE_NO_THINK_BEGIN in template and _FORCE_NO_THINK_END in template:
        before = template.split(_FORCE_NO_THINK_BEGIN, 1)[0]
        after = template.split(_FORCE_NO_THINK_END, 1)[1].lstrip("\n")
        template = before + after

    injection = (
        f"{_FORCE_NO_THINK_BEGIN}\n"
        "{%- set enable_thinking = false %}\n"
        f"{_FORCE_NO_THINK_END}\n"
    )

    tokenizer.chat_template = injection + template
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA safety adapter into Qwen3-1.7B for HF/vLLM submission."
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Base model used for LoRA training.",
    )

    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the trained LoRA adapter folder.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the merged full model.",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Keep thinking mode enabled in the merged model (for v13+). "
             "By default thinking is forced OFF via the chat template.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("LOADING TOKENIZER")
    print(f"Base model   : {args.base_model}")
    print(f"Adapter path : {args.adapter_path}")
    print(f"Output dir   : {output_dir}")

    # Prefer tokenizer from adapter because your training script saved it there.
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 100)
    print("LOADING BASE MODEL")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    print("=" * 100)
    print("LOADING LORA ADAPTER")

    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
    )

    print("=" * 100)
    print("MERGING LORA INTO BASE MODEL")

    model = model.merge_and_unload()
    model.eval()

    print("=" * 100)
    print("SAVING MERGED MODEL")

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Bake safety system prompt into chat template.
    # Thinking ON models use a different prompt (no /no_think directive).
    if args.thinking:
        thinking_prompt = (
            "Choose the correct option. "
            "Think carefully step by step, then answer with one boxed letter, like \\boxed{A}."
        )
        print(f"Thinking mode: ON — using thinking-friendly system prompt")
        print(f"  Prompt: {thinking_prompt}")
        force_safety_system_prompt(tokenizer, system_prompt=thinking_prompt)
    else:
        force_safety_system_prompt(tokenizer)
        force_qwen_no_think(tokenizer)
    tokenizer.save_pretrained(output_dir)
    print("=" * 100)
    print("SAVING GENERATION CONFIG")

    gen_config = GenerationConfig.from_pretrained(args.base_model)

    # Deterministic decoding for MCQ/boxed-answer safety task.
    gen_config.do_sample = False
    gen_config.temperature = None
    gen_config.top_p = None
    gen_config.top_k = None

    # Course document says CI uses a 16384 max-new-token budget.
    gen_config.max_new_tokens = 16384

    # Qwen3 token ids from the project document.
    gen_config.bos_token_id = tokenizer.bos_token_id
    gen_config.eos_token_id = [151645, 151643]
    gen_config.pad_token_id = tokenizer.pad_token_id

    print("Generation config token IDs:")
    print(f"  bos_token_id: {gen_config.bos_token_id}")
    print(f"  eos_token_id: {gen_config.eos_token_id}")
    print(f"  pad_token_id: {gen_config.pad_token_id}")
    print(f"  max_new_tokens: {gen_config.max_new_tokens}")
    print(f"  do_sample: {gen_config.do_sample}")

    gen_config.save_pretrained(output_dir)
    print("=" * 100)
    print("MERGE COMPLETE")
    print(f"Merged model saved to: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
