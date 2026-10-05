import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


# ============================================================
# Paths / environment variables
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import download_dataset_version_file

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = str(ROOT_DIR / "math" / "outputs" / "math_v3_curated_sft_lora")

BASE_MODEL = os.environ.get("MATH_BASE_MODEL", DEFAULT_BASE_MODEL)
OUTPUT_DIR = os.environ.get("MATH_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)

MAX_SEQ_LEN = int(os.environ.get("MATH_MAX_SEQ_LEN", "2048"))

LR = float(os.environ.get("MATH_LR", "7e-6"))
NUM_EPOCHS = float(os.environ.get("MATH_NUM_EPOCHS", "1.0"))

BATCH_SIZE = int(os.environ.get("MATH_BATCH_SIZE", "1"))
EVAL_BATCH_SIZE = int(os.environ.get("MATH_EVAL_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("MATH_GRAD_ACCUM", "16"))

WARMUP_RATIO = float(os.environ.get("MATH_WARMUP_RATIO", "0.03"))
WEIGHT_DECAY = float(os.environ.get("MATH_WEIGHT_DECAY", "0.0"))

LORA_R = int(os.environ.get("MATH_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("MATH_LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("MATH_LORA_DROPOUT", "0.05"))

SAVE_STEPS = int(os.environ.get("MATH_SAVE_STEPS", "250"))
EVAL_STEPS = int(os.environ.get("MATH_EVAL_STEPS", "125"))
LOGGING_STEPS = int(os.environ.get("MATH_LOGGING_STEPS", "10"))

SEED = int(os.environ.get("MATH_SEED", "52"))


# ============================================================
# Utilities
# ============================================================

def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(json.loads(line))

    return rows


def get_assistant_start_index(
    tokenizer: AutoTokenizer,
    prompt_messages: list[dict[str, str]],
    full_messages: list[dict[str, str]],
    max_seq_len: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Returns input_ids, attention_mask, labels.
    Labels are masked for system/user/prompt tokens and active only on assistant tokens.
    """

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_tokenized = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )

    full_tokenized = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids = full_tokenized["input_ids"]
    attention_mask = full_tokenized["attention_mask"]

    labels = input_ids.copy()

    prompt_len = min(len(prompt_tokenized["input_ids"]), len(labels))

    for i in range(prompt_len):
        labels[i] = -100

    # If truncation removed all assistant tokens, keep last token trainable to avoid Trainer crash.
    if all(label == -100 for label in labels):
        labels[-1] = input_ids[-1]

    return input_ids, attention_mask, labels


# ============================================================
# Dataset
# ============================================================

class ChatSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_seq_len: int):
        self.path = path
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if len(self.rows) == 0:
            raise ValueError(f"No rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]

        if "messages" not in row:
            raise ValueError(f"Missing messages field at index {idx}")

        messages = row["messages"]

        if len(messages) < 3:
            raise ValueError(f"Bad row at index {idx}: expected at least system/user/assistant messages")

        if messages[-1]["role"] != "assistant":
            raise ValueError(f"Bad row at index {idx}: last message must be assistant")

        prompt_messages = messages[:-1]

        input_ids, attention_mask, labels = get_assistant_start_index(
            tokenizer=self.tokenizer,
            prompt_messages=prompt_messages,
            full_messages=messages,
            max_seq_len=self.max_seq_len,
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ============================================================
# Data collator
# ============================================================

class DataCollatorForChatSFT:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id

        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            length = len(feature["input_ids"])
            pad_len = max_len - length

            input_ids.append(
                torch.cat(
                    [
                        feature["input_ids"],
                        torch.full((pad_len,), pad_id, dtype=torch.long),
                    ],
                    dim=0,
                )
            )

            attention_mask.append(
                torch.cat(
                    [
                        feature["attention_mask"],
                        torch.zeros(pad_len, dtype=torch.long),
                    ],
                    dim=0,
                )
            )

            labels.append(
                torch.cat(
                    [
                        feature["labels"],
                        torch.full((pad_len,), -100, dtype=torch.long),
                    ],
                    dim=0,
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


# ============================================================
# Main
# ============================================================

def main() -> None:
    train_file = os.environ.get("MATH_TRAIN_FILE") or download_dataset_version_file("v3_curated_sft", "train.jsonl")
    val_file = os.environ.get("MATH_VAL_FILE") or download_dataset_version_file("v3_curated_sft", "validation.jsonl")

    print("=" * 100)
    print("[math-v3] Qwen clean strong LoRA SFT")
    print("=" * 100)
    print(f"Base model:     {BASE_MODEL}")
    print(f"Train Hub file: {train_file}")
    print(f"Val Hub file:   {val_file}")
    print(f"Output dir:     {OUTPUT_DIR}")
    print(f"Max seq len:    {MAX_SEQ_LEN}")
    print(f"LR:             {LR}")
    print(f"Epochs:         {NUM_EPOCHS}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Grad accum:     {GRAD_ACCUM}")
    print(f"LoRA r:         {LORA_R}")
    print(f"LoRA alpha:     {LORA_ALPHA}")
    print(f"LoRA dropout:   {LORA_DROPOUT}")
    print("=" * 100)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    if not Path(train_file).exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    if not Path(val_file).exists():
        raise FileNotFoundError(f"Val file not found: {val_file}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSFTDataset(train_file, tokenizer, MAX_SEQ_LEN)
    eval_dataset = ChatSFTDataset(val_file, tokenizer, MAX_SEQ_LEN)

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples:   {len(eval_dataset)}")

    data_collator = DataCollatorForChatSFT(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        learning_rate=LR,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,

        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,

        bf16=True,
        fp16=False,

        optim="adamw_torch",
        lr_scheduler_type="cosine",

        gradient_checkpointing=True,
        remove_unused_columns=False,

        report_to="none",
        seed=SEED,
        data_seed=SEED,

        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("=" * 100)
    print("[math-v3] Starting training...")
    print("=" * 100)

    trainer.train()

    print("=" * 100)
    print("[math-v3] Saving LoRA adapter...")
    print("=" * 100)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[math-v3] Saved LoRA adapter to {OUTPUT_DIR}")

    print("=" * 100)
    print("[math-v3] Done.")
    print("=" * 100)


if __name__ == "__main__":
    main()
