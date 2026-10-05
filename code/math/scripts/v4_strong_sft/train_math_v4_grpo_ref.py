import inspect
import os
import re
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import load_dataset_version

BASE_MODEL = os.environ.get(
    "MATH_BASE_MODEL",
    str(ROOT_DIR / "math/outputs/math_v4_strong_sft_merged"),
)
OUTPUT_DIR = os.environ.get(
    "MATH_OUTPUT_DIR",
    str(ROOT_DIR / "math/outputs/math_v4_grpo_ref"),
)

MAX_PROMPT_LENGTH = int(os.environ.get("MATH_MAX_PROMPT_LENGTH", "1024"))
MAX_COMPLETION_LENGTH = int(os.environ.get("MATH_MAX_COMPLETION_LENGTH", "512"))

RESUME_FROM_CHECKPOINT = os.environ.get("MATH_RESUME_FROM_CHECKPOINT", None)

LR = float(os.environ.get("MATH_LR", "1e-6"))
MAX_STEPS = int(os.environ.get("MATH_MAX_STEPS", "50"))

NUM_GENERATIONS = int(os.environ.get("MATH_NUM_GENERATIONS", "4"))
BATCH_SIZE = int(os.environ.get("MATH_BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.environ.get("MATH_GRAD_ACCUM", "4"))

LORA_R = int(os.environ.get("MATH_LORA_R", "32"))
LORA_ALPHA = int(os.environ.get("MATH_LORA_ALPHA", "64"))
LORA_DROPOUT = float(os.environ.get("MATH_LORA_DROPOUT", "0.05"))


def strip_math_wrappers(s: str) -> str:
    s = str(s or "").strip()

    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()

    if s.startswith("\\[") and s.endswith("\\]"):
        s = s[2:-2].strip()

    return s.strip()


def normalize_answer(s: str) -> str:
    s = strip_math_wrappers(s)
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace(" ", "")
    s = s.replace("\n", "")
    s = s.strip(".")
    return s


def extract_last_boxed(text: str) -> str | None:
    text = str(text or "")
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not starts:
        return None

    start = starts[-1] + len("\\boxed{")
    depth = 1
    i = start
    chars = []

    while i < len(text) and depth > 0:
        ch = text[i]

        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                chars.append(ch)
        else:
            chars.append(ch)

        i += 1

    if depth == 0:
        return "".join(chars).strip()

    return None


def completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(completion)


def correctness_reward(completions, answer, **kwargs):
    rewards = []

    for comp, gold in zip(completions, answer):
        text = completion_to_text(comp)
        boxed = extract_last_boxed(text)

        if boxed is None:
            rewards.append(-0.2)
            continue

        pred = normalize_answer(boxed)
        ref = normalize_answer(gold)

        reward = 0.05

        if pred == ref:
            reward = 1.0

        # Mild anti-gaming penalties.
        if text.count("\\boxed{") > 1:
            reward -= 0.1

        if len(text.split()) > 650:
            reward -= 0.1

        rewards.append(float(reward))

    return rewards


def format_reward(completions, **kwargs):
    rewards = []

    for comp in completions:
        text = completion_to_text(comp)
        boxed_count = text.count("\\boxed{")

        score = 0.0

        if boxed_count >= 1:
            score += 0.05

        if boxed_count == 1:
            score += 0.05

        if "\\boxed{" in text[-120:]:
            score += 0.05

        rewards.append(score)

    return rewards


def main():
    print("=" * 100)
    print("[math-v4] GRPO / RLVR")
    print("=" * 100)
    print(f"Base model: {BASE_MODEL}")
    print("Dataset: v4_strong_sft/grpo_ref.jsonl from Hugging Face Hub")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"LR: {LR}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Num generations: {NUM_GENERATIONS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Grad accum: {GRAD_ACCUM}")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset_version(
        "v4_strong_sft",
        files={"train": "grpo_ref.jsonl"},
    )["train"]

    print(dataset)
    print("Example:", dataset[0])

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

    # TRL/GRPOConfig arguments differ slightly across versions.
    # Build kwargs, then keep only arguments supported by the installed GRPOConfig.
    grpo_kwargs = {
        "output_dir": OUTPUT_DIR,
        "learning_rate": LR,
        "max_steps": MAX_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "num_generations": NUM_GENERATIONS,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
        "bf16": True,
        "fp16": False,
        "logging_steps": 5,
        "save_steps": 500,
        "save_total_limit": 20,
        "report_to": "none",
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
    }

    supported = set(inspect.signature(GRPOConfig.__init__).parameters)
    filtered_kwargs = {k: v for k, v in grpo_kwargs.items() if k in supported}
    dropped_kwargs = sorted(set(grpo_kwargs) - set(filtered_kwargs))

    print("GRPOConfig supported kwargs:", sorted(filtered_kwargs))
    print("GRPOConfig dropped kwargs:", dropped_kwargs)

    args = GRPOConfig(**filtered_kwargs)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[correctness_reward, format_reward],
        args=args,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("=" * 100)
    print(f"[math-v4] Saved GRPO adapter to {OUTPUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
