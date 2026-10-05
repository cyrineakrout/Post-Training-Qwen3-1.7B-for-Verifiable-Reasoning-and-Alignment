"""
GRPO training for multilingual reasoning on Qwen3-1.7B.

Reward: binary — 1.0 if \\boxed{correct_letter} is present, 0.0 otherwise.

Paths:
  cold_grpo              — GRPO from base model (no warm-start)
  sft_grpo               — GRPO warm-started from sft_merged
  distillation_sft_grpo  — GRPO warm-started from distillation_sft_merged
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

ML_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

CONFIG_PATH = ML_ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

from multilinguality.scripts.utils import BOXED_RE, LANG_LABELS, detect_language, GRPO_PATHS, with_lang, with_thinking

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

MODEL_CFG  = CFG["model"]
DATA_CFG   = CFG["data"]
TRAIN_CFG  = CFG["training"]
LORA_CFG   = CFG["lora"]
WANDB_CFG  = CFG["wandb"]
PATHS_CFG  = CFG["paths"]
PROMPT_CFG = CFG["prompts"]

CHAT_TEMPLATE_PATH = ML_ROOT / "chat_template.jinja"


def apply_chat_template(tokenizer, thinking_on: bool):
    if thinking_on:
        return tokenizer
    tokenizer.chat_template = CHAT_TEMPLATE_PATH.read_text()
    return tokenizer


def make_reward_fn(language_mode: str):
    """Factory returning a reward function that optionally enforces language consistency."""
    def binary_reward(
        completions: list,
        correct_index: list[int],
        language: Optional[list] = None,
        **_,
    ) -> list[float]:
        rewards = []
        for i, (completion, cidx) in enumerate(zip(completions, correct_index)):
            if isinstance(completion, list):
                completion = completion[-1] if completion else ""
            if isinstance(completion, dict):
                completion = completion.get("content", "")
            if not isinstance(completion, str):
                completion = str(completion)

            if language_mode == "multi" and language is not None:
                lang_code = language[i] if isinstance(language, list) else language
                target = LANG_LABELS.get(lang_code, "")
                if target and not detect_language(completion, target):
                    print(f"[DBG] lang_fail lang={lang_code}  reward=0.0", flush=True)
                    rewards.append(0.0)
                    continue

            m = BOXED_RE.search(completion)
            if m and ord(m.group(1)) - ord("A") == cidx:
                reward = 1.0
            elif m:
                reward = 0.2
            else:
                reward = 0.0
            print(f"[DBG] boxed={m.group(0) if m else None}  reward={reward}", flush=True)
            rewards.append(reward)
        return rewards

    return binary_reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, choices=GRPO_PATHS,
                        help="Training path: cold_grpo | sft_grpo | distillation_sft_grpo")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint in output_dir.")
    parser.add_argument("--restart", action="store_true",
                        help="Delete output_dir before training to force a fresh run.")
    parser.add_argument("--thinking-on", action="store_true",
                        help="Keep thinking enabled during rollouts. Default: thinking off.")
    parser.add_argument("--thinking-off", action="store_true",
                        help="Use the _toff merged base. Default: _ton (matches baseline).")
    parser.add_argument("--language", choices=["english", "multi"], default="english",
                        help="'multi': reward=0 when completion is not in the question's language.")
    parser.add_argument("--suffix", default="",
                        help="Append suffix to the lora output dir, e.g. '4k' → sft_grpo4k_lora.")
    return parser.parse_args()


def main():
    args = parse_args()

    path_cfg = PATHS_CFG[args.path]
    out_dir  = ml_path(with_lang(path_cfg["lora_dir"], args.language))
    if args.suffix:
        out_dir = out_dir.parent / (out_dir.name.replace("_lora", args.suffix + "_lora"))

    if args.restart and out_dir.exists():
        import shutil
        print(f"--force: removing {out_dir}")
        shutil.rmtree(out_dir)

    system_prompt = PROMPT_CFG["inference_system"]
    if args.path == "cold_grpo":
        base = MODEL_CFG["base_model"]
    elif args.path == "sft_grpo":
        base = str(ml_path(with_thinking(PATHS_CFG["sft"]["merged_dir"], not args.thinking_off)))
    elif args.path == "distillation_sft_grpo":
        base = str(ml_path(with_thinking(with_lang(PATHS_CFG["distillation_sft"]["merged_dir"], args.language), not args.thinking_off)))
    else:
        raise ValueError(f"Unknown GRPO path: {args.path!r}. Expected one of {GRPO_PATHS}")

    data_file = str(ml_path(DATA_CFG["train_file"]))
    print(f"[path={args.path}] base={base}  data={data_file}  out={out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer = apply_chat_template(tokenizer, thinking_on=args.thinking_on)

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    dataset = load_dataset("json", data_files=data_file, split="train")

    def add_system(example):
        example["messages"] = [{"role": "system", "content": system_prompt}] + example["messages"]
        return example
    dataset = dataset.map(add_system)
    dataset = dataset.rename_column("messages", "prompt")
    print(f"  {len(dataset)} training examples")

    wandb.init(
        project=WANDB_CFG["project"],
        entity=WANDB_CFG["entity"],
        tags=WANDB_CFG["tags"] + [args.path],
        name=args.path,
        config={**MODEL_CFG, **DATA_CFG, **TRAIN_CFG, **LORA_CFG},
    )

    lora_config = LoraConfig(
        r=LORA_CFG["r"],
        lora_alpha=LORA_CFG["alpha"],
        lora_dropout=LORA_CFG["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    grpo_config = GRPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=TRAIN_CFG["num_train_epochs"],
        per_device_train_batch_size=TRAIN_CFG["batch_size"],
        gradient_accumulation_steps=TRAIN_CFG["gradient_accumulation_steps"],
        learning_rate=TRAIN_CFG["learning_rate"],
        num_generations=TRAIN_CFG["num_generations"],
        num_iterations=TRAIN_CFG["num_iterations"],
        max_completion_length=TRAIN_CFG["max_new_tokens"],
        temperature=TRAIN_CFG["temperature"],
        beta=TRAIN_CFG["beta"],
        logging_steps=TRAIN_CFG["logging_steps"],
        save_steps=TRAIN_CFG["save_steps"],
        report_to="wandb",
        bf16=True,
        fp16=False,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=32768,
    )

    reward_fn = make_reward_fn(args.language)
    if args.language == "multi":
        print("Language reward: enabled — completions in wrong language get reward=0.0")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"Starting GRPO training ({args.path})...")
    trainer.train(resume_from_checkpoint=args.resume or None)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"LoRA adapter saved to {out_dir}")

    wandb.finish()


if __name__ == "__main__":
    main()
