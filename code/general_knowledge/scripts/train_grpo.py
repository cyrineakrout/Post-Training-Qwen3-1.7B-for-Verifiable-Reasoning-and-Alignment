import argparse
import re
import sys
from pathlib import Path

GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(GENERAL_KNOWLEDGE_ROOT / "configs"), str(GENERAL_KNOWLEDGE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config_from_args
from eval_utils import build_run_name, load_jsonl


def parse_args():
    config_path, config = load_config_from_args()
    model_config = config["model"]
    training_config = config["training"]
    wandb_config = config["wandb"]
    lora_defaults = config["lora"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument(
        "--model_path",
        type=str,
        default=model_config["base_model"],
        help="Merged SFT model (or base model) to warm-start GRPO from.",
    )
    parser.add_argument("--train_file", type=str, default=training_config["train_file"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=wandb_config["project"])
    parser.add_argument("--wandb_entity", type=str, default=wandb_config["entity"])

    # GRPO generation
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        help="Completions sampled per prompt (G in GRPO). Higher = more stable advantage estimates.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Max new tokens per rollout. Short-stub format needs only ~30 tokens; 512 is generous.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)

    # Optimisation
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device batch size. Keep at 1 for GRPO to avoid OOM.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.1,
        help="Lower than SFT to prevent policy collapse.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on training examples (useful for quick smoke tests).",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=lora_defaults["r"])
    parser.add_argument("--lora_alpha", type=int, default=lora_defaults["alpha"])
    parser.add_argument("--lora_dropout", type=float, default=lora_defaults["dropout"])

    # Reward shaping
    parser.add_argument(
        "--format_penalty",
        type=float,
        default=-0.5,
        help="Reward assigned when no \\boxed{X} is found. Encourages format compliance.",
    )

    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from, or 'latest' to auto-detect in output_dir.",
    )

    args = parser.parse_args()
    args.wandb_config = wandb_config
    return args


def extract_boxed_answer(text: str):
    """Return the last \\boxed{X} letter found, or None."""
    matches = re.findall(r"\\boxed\{([A-Ta-t])\}", text)
    return matches[-1].upper() if matches else None


def make_reward_fn(format_penalty: float):
    """
    Returns a reward function compatible with TRL GRPOTrainer.

    GRPOTrainer passes extra dataset columns as kwargs, so `answer`
    (our dataset column) is automatically forwarded here.
    """
    def reward_fn(completions: list, answer: list, **kwargs) -> list:
        rewards = []
        for completion, ans in zip(completions, answer):
            predicted = extract_boxed_answer(completion)
            if predicted is None:
                rewards.append(format_penalty)
            elif predicted == ans.strip().upper():
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards
    return reward_fn


def main():
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOTrainer, GRPOConfig

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    use_wandb = not args.disable_wandb
    run_name = args.run_name or build_run_name(
        task="grpo",
        model_name=args.model_path,
        dataset_path=args.train_file,
        method="gk",
    )

    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=list(args.wandb_config.get("tags", [])) + ["train", "lora", "grpo", "rlvr"],
            job_type="train",
            config={
                "model_path": args.model_path,
                "train_file": args.train_file,
                "num_generations": args.num_generations,
                "max_completion_length": args.max_completion_length,
                "temperature": args.temperature,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "format_penalty": args.format_penalty,
            },
        )

    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading training data: {args.train_file}")
    raw_examples = load_jsonl(args.train_file, max_examples=args.max_examples)

    prompts, answers = [], []
    for ex in raw_examples:
        messages = ex["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(prompt_text)
        answers.append(ex["metadata"]["answer"])

    dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})
    print(f"GRPO dataset: {len(dataset)} examples")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=run_name if use_wandb else None,
        report_to="wandb" if use_wandb else "none",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[make_reward_fn(args.format_penalty)],
        args=grpo_config,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    resume = args.resume_from_checkpoint
    if resume == "latest":
        import os
        ckpts = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        resume = os.path.join(args.output_dir, ckpts[-1]) if ckpts else None
        print(f"Resuming from: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    print("Saving GRPO LoRA adapter...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"Saved GRPO adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
