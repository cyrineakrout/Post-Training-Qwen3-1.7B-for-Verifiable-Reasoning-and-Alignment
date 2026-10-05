"""
DPO fine-tuning for the safety model on top of a merged SFT model.

Applies a LoRA DPO adapter on top of the merged SFT model.
The frozen base (reference model) is the same model with LoRA disabled,
so only one model is loaded in memory.

Usage:
  python safety/scripts/train_safety_dpo.py \
      --config safety/configs/safety_dpo_v1.yaml
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def apply_template(messages: list[dict], tokenizer) -> str:
    """Apply chat template to a message list, return a string."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )


def build_dpo_row(example: dict, tokenizer) -> dict | None:
    """
    Convert a DPO JSONL row to TRL DPO format:
      prompt   = system + user formatted string (with generation marker)
      chosen   = assistant answer only  e.g. \\boxed{B}
      rejected = assistant answer only  e.g. \\boxed{C}

    TRL concatenates: prompt + chosen (or rejected) internally.
    Returning full conversations for chosen/rejected would duplicate the prompt.
    """
    chosen_msgs   = example.get("chosen", [])
    rejected_msgs = example.get("rejected", [])

    if not chosen_msgs or not rejected_msgs:
        return None

    # Prompt = all messages before the assistant turn
    prompt_msgs = [m for m in chosen_msgs if m.get("role") != "assistant"]

    chosen_asst = next(
        (m.get("content", "") for m in chosen_msgs   if m.get("role") == "assistant"), ""
    ).strip()
    rejected_asst = next(
        (m.get("content", "") for m in rejected_msgs if m.get("role") == "assistant"), ""
    ).strip()

    if not chosen_asst or not rejected_asst:
        return None

    try:
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
        )

    return {
        "prompt":   prompt_str,
        "chosen":   chosen_asst,
        "rejected": rejected_asst,
    }


def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for safety model.")
    parser.add_argument(
        "--config",
        type=str,
        default="safety/configs/safety_dpo_v1.yaml",
    )
    parser.add_argument(
        "--train_limit",
        type=int,
        default=-1,
        help="Limit training examples. -1 = all.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    model_path  = cfg["model"]["base_model"]
    output_dir  = cfg["model"]["output_dir"]
    train_path  = cfg["data"]["train_path"]
    dev_path    = cfg["data"].get("dev_path")
    train_limit = None if args.train_limit == -1 else args.train_limit
    lora_cfg    = cfg["lora"]
    dpo_cfg     = cfg["dpo"]
    train_cfg   = cfg["training"]
    run_name    = cfg.get("wandb", {}).get("run_name", "safety-dpo")

    print("=" * 100)
    print("LOADING TOKENIZER")
    print(f"Model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("=" * 100)
    print("LOADING TRAINING DATA")

    raw_train = load_jsonl(train_path, limit=train_limit)
    print(f"  Loaded {len(raw_train)} raw DPO pairs")

    train_rows = [build_dpo_row(r, tokenizer) for r in raw_train]
    train_rows = [r for r in train_rows if r is not None]
    print(f"  Valid DPO pairs: {len(train_rows)}")

    train_dataset = Dataset.from_list(train_rows)

    eval_dataset = None
    if dev_path:
        raw_dev  = load_jsonl(dev_path)
        dev_rows = [build_dpo_row(r, tokenizer) for r in raw_dev]
        dev_rows = [r for r in dev_rows if r is not None]
        eval_dataset = Dataset.from_list(dev_rows) if dev_rows else None
        print(f"  Dev pairs: {len(dev_rows)}")

    print("=" * 100)
    print("LOADING MODEL")

    use_cuda = torch.cuda.is_available()
    dtype    = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Apply LoRA for DPO (reference model = frozen base, policy = LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # W&B
    report_to = [] if args.no_wandb else ["wandb"]

    if not args.no_wandb:
        import wandb
        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg.get("wandb", {}).get("entity"),
            name=run_name,
            group="safety-dpo",
            job_type="train",
            tags=["safety", "dpo", "lora"],
            config={
                "base_model":  model_path,
                "dpo_beta":    dpo_cfg["beta"],
                "lora_r":      lora_cfg["r"],
                "lora_alpha":  lora_cfg["alpha"],
                "learning_rate": train_cfg["learning_rate"],
                "num_train_epochs": train_cfg["num_train_epochs"],
                "train_pairs": len(train_rows),
            },
        )

    print("=" * 100)
    print("STARTING DPO TRAINING")

    dpo_training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        weight_decay=float(train_cfg["weight_decay"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        eval_steps=int(train_cfg["eval_steps"]) if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        save_total_limit=2,
        fp16=use_cuda,
        bf16=False,
        gradient_checkpointing=True,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
        seed=int(train_cfg["seed"]),
        # DPO-specific
        beta=float(dpo_cfg["beta"]),
        max_length=int(train_cfg.get("max_seq_length", 512)),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # None = use LoRA's frozen base as reference
        args=dpo_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    print("=" * 100)
    print("SAVING DPO ADAPTER")
    final_dir = Path(output_dir) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    if not args.no_wandb:
        import wandb
        wandb.finish()

    print(f"DPO adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()
