"""
GRPO training for the group model, warm-started from a merged SFT model.

Usage (from code/group/):

  python scripts/train_group_grpo.py

Key env vars (all have defaults):
  GROUP_GRPO_BASE       path to merged base model (default: group/outputs/distillation_sft_merged)
  GROUP_GRPO_DATA       path to GRPO training JSONL  (default: group/data/group_grpo_train.jsonl)
  GROUP_GRPO_OUT        output LoRA dir             (default: group/outputs/distillation_sft_grpo_lora)
  GROUP_GRPO_LR         learning rate               (default: 1e-5)
  GROUP_GRPO_EPOCHS     number of epochs            (default: 1)
  GROUP_GRPO_BATCH      per-device batch size       (default: 2)
  GROUP_GRPO_GRAD_ACCUM gradient accumulation steps (default: 4)
  GROUP_GRPO_NGEN       num_generations for GRPO    (default: 8)
  GROUP_GRPO_MAX_TOKENS max completion length       (default: 4096)
  GROUP_GRPO_TEMP       sampling temperature        (default: 1.1)
  GROUP_GRPO_LORA_R     LoRA rank                   (default: 64)
  GROUP_GRPO_LORA_ALPHA LoRA alpha                  (default: 128)
  WANDB_PROJECT         W&B project                 (default: cs552-mnlp-group)
  WANDB_ENTITY          W&B entity                  (default: MNLP26)
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

GROUP_ROOT = Path(__file__).resolve().parents[1]
SCRATCH    = GROUP_ROOT.parent.parent  # project root

BASE_MODEL  = os.environ.get("GROUP_GRPO_BASE",       str(SCRATCH / "group/outputs/distillation_sft_merged"))
DATA_FILE   = os.environ.get("GROUP_GRPO_DATA",       str(SCRATCH / "group/data/group_grpo_train.jsonl"))
OUTPUT_DIR  = os.environ.get("GROUP_GRPO_OUT",        str(SCRATCH / "group/outputs/distillation_sft_grpo_lora"))

LR           = float(os.environ.get("GROUP_GRPO_LR",         "1e-5"))
NUM_EPOCHS   = float(os.environ.get("GROUP_GRPO_EPOCHS",     "1"))
BATCH_SIZE   = int(os.environ.get("GROUP_GRPO_BATCH",        "2"))
GRAD_ACCUM   = int(os.environ.get("GROUP_GRPO_GRAD_ACCUM",   "4"))
NUM_GENS     = int(os.environ.get("GROUP_GRPO_NGEN",         "8"))
MAX_TOKENS   = int(os.environ.get("GROUP_GRPO_MAX_TOKENS",   "4096"))
TEMPERATURE  = float(os.environ.get("GROUP_GRPO_TEMP",       "1.1"))
LORA_R       = int(os.environ.get("GROUP_GRPO_LORA_R",       "64"))
LORA_ALPHA   = int(os.environ.get("GROUP_GRPO_LORA_ALPHA",   "128"))
LORA_DROPOUT = float(os.environ.get("GROUP_GRPO_LORA_DROPOUT", "0.05"))

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "cs552-mnlp-group")
WANDB_ENTITY  = os.environ.get("WANDB_ENTITY",  "MNLP26")

SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "For every question, put the final answer inside exactly one LaTeX \\boxed{} expression. "
    "If the question is multiple-choice, put only the option letter inside the box, for example \\boxed{C}. "
    "If the question is free-form or mathematical, put only the final expression, number, set, or "
    "requested object inside the box. "
    "Use brief reasoning only when useful. "
    "Do not write anything after the boxed answer."
)

BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")

_acc = {"zero": 0, "total": 0, "batches": 0}


def reward_fn(completions: list, metadata: list, **_) -> list[float]:
    rewards = []
    for completion, meta in zip(completions, metadata):
        if isinstance(completion, list):
            completion = completion[-1].get("content", "") if completion else ""
        if isinstance(completion, dict):
            completion = completion.get("content", "")
        expected = str(meta.get("answer", "")).strip().upper()
        m = BOXED_RE.search(str(completion))
        if m:
            got = m.group(1).strip().upper()
            reward = 1.0 if got == expected else 0.2
        else:
            reward = 0.0
        print(f"[reward] expected={expected}  got={m.group(1).strip() if m else None}  r={reward}", flush=True)
        rewards.append(reward)

    _acc["zero"]   += sum(1 for r in rewards if r == 0.0)
    _acc["total"]  += len(rewards)
    _acc["batches"] += 1

    if _acc["batches"] % 20 == 0:
        wandb.log({
            "reward/zero_frac":  _acc["zero"] / max(1, _acc["total"]),
            "reward/zero_count": _acc["zero"],
        })
        _acc["zero"] = _acc["total"] = _acc["batches"] = 0

    return rewards


def load_jsonl(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart", action="store_true",
                        help="Delete output dir before training to force a fresh run.")
    args = parser.parse_args()

    out_dir = Path(OUTPUT_DIR)

    if args.restart and out_dir.exists():
        print(f"--restart: removing {out_dir}")
        shutil.rmtree(out_dir)

    print("=" * 100)
    print("[group] GRPO training")
    print("=" * 100)
    print(f"Base model : {BASE_MODEL}")
    print(f"Data file  : {DATA_FILE}")
    print(f"Output dir : {out_dir}")
    print(f"LR={LR}  epochs={NUM_EPOCHS}  batch={BATCH_SIZE}  grad_accum={GRAD_ACCUM}")
    print(f"num_generations={NUM_GENS}  max_tokens={MAX_TOKENS}  temp={TEMPERATURE}")
    print(f"LoRA r={LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    dataset = load_jsonl(DATA_FILE)

    def add_system(example):
        example["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + example["messages"]
        return example

    dataset = dataset.map(add_system)
    dataset = dataset.rename_column("messages", "prompt")
    print(f"Training examples: {len(dataset)}")

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="grpo_group",
        config={
            "lr": LR, "epochs": NUM_EPOCHS, "batch": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "num_generations": NUM_GENS,
            "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE,
            "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
        },
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_generations=NUM_GENS,
        num_iterations=1,
        max_completion_length=MAX_TOKENS,
        temperature=TEMPERATURE,
        logging_steps=20,
        save_steps=200,
        report_to="wandb",
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=32768,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    resume = True if any(out_dir.glob("checkpoint-*")) else None
    print(f"Starting GRPO training (resume={resume})...")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
