import os
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import load_dataset_version

BASE_MODEL = os.environ.get("MATH_BASE_MODEL", "Qwen/Qwen3-1.7B")
OUTPUT_DIR = os.environ.get("MATH_OUTPUT_DIR", str(ROOT_DIR / "math/outputs/math_v4_strong_sft_lora"))
RESUME_FROM_CHECKPOINT = os.environ.get("MATH_RESUME_FROM_CHECKPOINT", None)

LR = float(os.environ.get("MATH_LR", "1e-5"))
NUM_EPOCHS = float(os.environ.get("MATH_NUM_EPOCHS", "0.5"))
MAX_SEQ_LEN = int(os.environ.get("MATH_MAX_SEQ_LEN", "2048"))
GRAD_ACCUM = int(os.environ.get("MATH_GRAD_ACCUM", "16"))
BATCH_SIZE = int(os.environ.get("MATH_BATCH_SIZE", "1"))

LORA_R = int(os.environ.get("MATH_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("MATH_LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("MATH_LORA_DROPOUT", "0.05"))


def main() -> None:
    print("=" * 100)
    print("[math-v4] Train LoRA strong SFT")
    print("=" * 100)
    print(f"Base model: {BASE_MODEL}")
    print("Dataset: v4_strong_sft from Hugging Face Hub")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"LR: {LR}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Max seq len: {MAX_SEQ_LEN}")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset_version("v4_strong_sft")
    train_ds = dataset["train"]
    val_ds = dataset["validation"]

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
    print(train_ds[0]["text"][:1200])

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False
    # No k-bit preparation: we are training LoRA in bf16, not 4-bit/8-bit.

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
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=0.05,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=300,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=3,
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

    if RESUME_FROM_CHECKPOINT:
        print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
        trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    else:
        trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 100)
    print(f"[math-v4] Saved LoRA adapter to {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
