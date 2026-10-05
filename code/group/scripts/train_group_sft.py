import argparse
import os
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

GROUP_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = os.environ.get("GROUP_BASE_MODEL", "Qwen/Qwen3-1.7B")
TRAIN_FILE = os.environ.get("GROUP_TRAIN_FILE", str(GROUP_ROOT / "data/group_sft_final_train.jsonl"))
VAL_FILE   = os.environ.get("GROUP_VAL_FILE",   str(GROUP_ROOT / "data/group_sft_final_val.jsonl"))
OUTPUT_DIR = os.environ.get("GROUP_OUTPUT_DIR", str(GROUP_ROOT / "outputs/group_lora_final_r96_e06"))

# Must match distillation.py — only used when --exclude-distillation is passed
_DISTILL_N    = 2150
_DISTILL_SEED = 42

RESUME_FROM_CHECKPOINT = os.environ.get("GROUP_RESUME_FROM_CHECKPOINT", None)

LR = float(os.environ.get("GROUP_LR", "8e-6"))
NUM_EPOCHS = float(os.environ.get("GROUP_NUM_EPOCHS", "0.6"))
MAX_SEQ_LEN = int(os.environ.get("GROUP_MAX_SEQ_LEN", "2048"))
BATCH_SIZE = int(os.environ.get("GROUP_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GROUP_GRAD_ACCUM", "16"))

LORA_R = int(os.environ.get("GROUP_LORA_R", "96"))
LORA_ALPHA = int(os.environ.get("GROUP_LORA_ALPHA", "192"))
LORA_DROPOUT = float(os.environ.get("GROUP_LORA_DROPOUT", "0.05"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-distillation", action="store_true",
                        help="Exclude the distillation slice (first %d rows of seed-%d shuffle) "
                             "from SFT training to keep SFT and distillation sets disjoint." % (_DISTILL_N, _DISTILL_SEED))
    args = parser.parse_args()

    print("=" * 100)
    print("[group] Train final group SFT")
    print("=" * 100)
    print(f"Base model: {BASE_MODEL}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Val file:   {VAL_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"LR: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Max seq len: {MAX_SEQ_LEN}")
    print(f"LoRA r/alpha/dropout: {LORA_R}/{LORA_ALPHA}/{LORA_DROPOUT}")
    if args.exclude_distillation:
        print(f"Excluding first {_DISTILL_N} rows (distillation slice, seed={_DISTILL_SEED})")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_dataset("json", data_files=TRAIN_FILE, split="train")
    if args.exclude_distillation:
        rows = list(train_ds)
        random.Random(_DISTILL_SEED).shuffle(rows)
        rows = rows[_DISTILL_N:]
        train_ds = Dataset.from_list(rows)
        print(f"Train examples after exclusion: {len(train_ds)}")
    val_ds = load_dataset("json", data_files=VAL_FILE, split="train")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(format_example, remove_columns=val_ds.column_names)

    print(train_ds)
    print(val_ds)
    print("Example text:")
    print(train_ds[0]["text"][:1500])

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=OUTPUT_DIR,

        max_length=MAX_SEQ_LEN,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=4,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 100)
    print(f"[group] Saved LoRA adapter to {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
