import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


SAFETY_SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)

_SAFETY_SYS_BEGIN = "{# === SAFETY_DEFAULT_SYSTEM_BEGIN === #}"
_SAFETY_SYS_END = "{# === SAFETY_DEFAULT_SYSTEM_END === #}"


def force_safety_system_prompt(tokenizer, system_prompt: str = SAFETY_SYSTEM_PROMPT):
    """
    Inject a default system message into the chat template so that CI
    (which calls apply_chat_template without a system message) still
    sees our safety system prompt.

    Also bakes an empty <think></think> block into the generation prompt
    so the model never generates chain-of-thought regardless of what
    system prompt the CI passes.
    """
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return tokenizer

    template = tokenizer.chat_template

    # Strip previous injection if present (idempotent).
    if _SAFETY_SYS_BEGIN in template and _SAFETY_SYS_END in template:
        before = template.split(_SAFETY_SYS_BEGIN, 1)[0]
        after = template.split(_SAFETY_SYS_END, 1)[1].lstrip("\n")
        template = before + after

    escaped = (
        system_prompt
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )

    injection = (
        f"{_SAFETY_SYS_BEGIN}\n"
        "{%- if messages and messages[0].role != 'system' %}\n"
        f"    {{%- set messages = [{{'role': 'system', 'content': '{escaped}'}}] + messages %}}\n"
        "{%- endif %}\n"
        f"{_SAFETY_SYS_END}\n"
    )

    tokenizer.chat_template = injection + template
    return tokenizer


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str, limit: int | None = None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_texts(example, tokenizer, thinking: bool = False):
    """
    Creates:
    - prompt_text: system + user + assistant generation marker
    - full_text: system + user + assistant answer
    We mask prompt tokens so the model only learns the assistant answer.

    thinking=True  → enable_thinking=True  (v13+ with <think> blocks)
    thinking=False → enable_thinking=False (v10/v12 default, /no_think)
    """
    messages = example["messages"]

    prompt_messages = messages[:-1]
    full_messages = messages

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=thinking,
    )

    return prompt_text, full_text


def tokenize_example(example, tokenizer, max_seq_length: int, thinking: bool = False):
    prompt_text, full_text = build_texts(example, tokenizer, thinking=thinking)

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]

    labels = input_ids.copy()

    prompt_len = len(prompt["input_ids"])
    prompt_len = min(prompt_len, len(labels))

    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            input_ids.append(
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            attention_mask.append(
                f["attention_mask"] + [0] * pad_len
            )
            labels.append(
                f["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f}%)"
    )


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace("_", "-")


def default_run_name(cfg) -> str:
    base_model = short_model_name(cfg["model"]["base_model"])
    train_stem = Path(cfg["data"]["train_path"]).stem.replace("_", "-")
    return f"{base_model}-safety-lora-sft-{train_stem}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Qwen3 safety model with LoRA SFT."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="safety/configs/safety_lora_sft.yaml",
    )

    parser.add_argument(
        "--train_limit",
        type=int,
        default=-1,
        help="Limit training examples for debugging. Use -1 for all.",
    )

    parser.add_argument(
        "--eval_limit",
        type=int,
        default=-1,
        help="Limit eval examples for debugging. Use -1 for all.",
    )

    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode (enable_thinking=True in chat template). "
             "Use with v13+ datasets that have <think> blocks in assistant responses.",
    )

    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Override the W&B run name from the config.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg["model"]["base_model"]
    output_dir = cfg["model"]["output_dir"]
    run_name = (
        args.run_name
        or cfg.get("wandb", {}).get("run_name")
        or default_run_name(cfg)
    )

    train_path = cfg["data"]["train_path"]
    dev_path = cfg["data"]["dev_path"]

    train_limit = None if args.train_limit == -1 else args.train_limit
    eval_limit = None if args.eval_limit == -1 else args.eval_limit

    print("=" * 100)
    print("LOADING TOKENIZER")
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 100)
    print("LOADING DATA")
    print(f"Train path: {train_path}")
    print(f"Dev path: {dev_path}")

    train_rows = load_jsonl(train_path, limit=train_limit)
    dev_rows = load_jsonl(dev_path, limit=eval_limit)

    print(f"Train examples: {len(train_rows)}")
    print(f"Dev examples: {len(dev_rows)}")

    max_seq_length = int(cfg["training"]["max_seq_length"])

    thinking_mode = args.thinking
    print(f"Thinking mode: {'ON' if thinking_mode else 'OFF'}")

    train_dataset = Dataset.from_list(train_rows).map(
        lambda ex: tokenize_example(ex, tokenizer, max_seq_length, thinking=thinking_mode),
        remove_columns=["messages"],
    )

    dev_dataset = Dataset.from_list(dev_rows).map(
        lambda ex: tokenize_example(ex, tokenizer, max_seq_length, thinking=thinking_mode),
        remove_columns=["messages"],
    )

    print("=" * 100)
    print("LOADING MODEL")

    use_cuda = torch.cuda.is_available()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_cfg = cfg["lora"]

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

    report_to = [] if args.no_wandb else ["wandb"]

    training_cfg = cfg["training"]

    if not args.no_wandb:
        import wandb

        wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg.get("wandb", {}).get("entity"),
            name=run_name,
            group="safety-lora-sft",
            job_type="train",
            tags=[
                "safety",
                "lora",
                "sft",
                short_model_name(model_name),
            ],
            config={
                "run_name": run_name,
                "task": "safety",
                "base_model": model_name,
                "method": "lora_sft",
                "adapter_type": "lora",
                "thinking_mode": "off",
                "output_dir": output_dir,
                "train_path": train_path,
                "dev_path": dev_path,
                "train_limit": args.train_limit,
                "eval_limit": args.eval_limit,
                "num_train_epochs": float(training_cfg["num_train_epochs"]),
                "learning_rate": float(training_cfg["learning_rate"]),
                "per_device_train_batch_size": int(
                    training_cfg["per_device_train_batch_size"]
                ),
                "gradient_accumulation_steps": int(
                    training_cfg["gradient_accumulation_steps"]
                ),
                "max_seq_length": int(training_cfg["max_seq_length"]),
                "warmup_ratio": float(training_cfg["warmup_ratio"]),
                "weight_decay": float(training_cfg["weight_decay"]),
                "lora_r": int(lora_cfg["r"]),
                "lora_alpha": int(lora_cfg["alpha"]),
                "lora_dropout": float(lora_cfg["dropout"]),
                "lora_target_modules": lora_cfg["target_modules"],
                "seed": int(training_cfg["seed"]),
            },
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        learning_rate=float(training_cfg["learning_rate"]),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        warmup_ratio=float(training_cfg["warmup_ratio"]),
        weight_decay=float(training_cfg["weight_decay"]),
        logging_steps=int(training_cfg["logging_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        eval_steps=int(training_cfg["eval_steps"]),
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        fp16=use_cuda,
        bf16=False,
        gradient_checkpointing=True,
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
        seed=int(training_cfg["seed"]),
    )

    collator = DataCollatorForCausalLM(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
    )

    print("=" * 100)
    print("STARTING TRAINING")
    trainer.train()

    print("=" * 100)
    print("SAVING FINAL MODEL")
    final_dir = Path(output_dir) / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(final_dir)

    # Bake the safety system prompt into the chat template so CI
    # (which passes no system message) still sees it.
    tokenizer = force_safety_system_prompt(tokenizer)
    tokenizer.save_pretrained(final_dir)

    if not args.no_wandb:
        import wandb

        wandb.finish()

    print(f"Saved LoRA adapter to: {final_dir}")


if __name__ == "__main__":
    main()
