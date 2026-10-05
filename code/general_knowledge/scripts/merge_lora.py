import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    # Generation config flags — use defaults for no-think models.
    # For CoT models (v10, v11) pass --do_sample with sampling params.
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Enable sampling in generation_config (required for CoT models).")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.03)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging LoRA into base model...")
    model = model.merge_and_unload()
    

    print("Loading tokenizer from adapter path...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path,
        trust_remote_code=True,
    )

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output_dir)

    print("Saving generation_config.json...")
    gen_config = GenerationConfig.from_pretrained(args.base_model)
    gen_config.max_new_tokens = 16384
    gen_config.pad_token_id = tokenizer.pad_token_id
    gen_config.eos_token_id = [151645, 151643]
    gen_config.bos_token_id = tokenizer.bos_token_id

    if args.do_sample:
        gen_config.do_sample = True
        gen_config.temperature = args.temperature
        gen_config.top_p = args.top_p
        gen_config.top_k = args.top_k
        gen_config.repetition_penalty = args.repetition_penalty
    else:
        gen_config.do_sample = False
        gen_config.temperature = None
        gen_config.top_p = None
        gen_config.top_k = None
        gen_config.repetition_penalty = None

    gen_config.save_pretrained(output_dir)

    print("Done.")
    print(f"Merged checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()