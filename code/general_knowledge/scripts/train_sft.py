import argparse
import sys
from pathlib import Path

GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
for _p in [str(GENERAL_KNOWLEDGE_ROOT / "configs"), str(GENERAL_KNOWLEDGE_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config_from_args
from eval_utils import (
    build_run_name,
    flatten_score_metrics,
    generate_completion_rows,
    load_jsonl,
    log_wandb_eval_artifacts,
    save_json,
    save_jsonl,
    score_generation_rows,
)


def parse_args():
    config_path, config = load_config_from_args()
    model_config = config["model"]
    training_config = config["training"]
    wandb_config = config["wandb"]
    lora_config = config["lora"]
    generation_config = config["generation"]
    prompt_config = config["prompts"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the general knowledge config JSON file.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        default=training_config["train_file"],
        help="Path to the SFT training JSONL file.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=training_config["eval_file"],
        help="Path to the SFT validation JSONL file for eval/loss.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=training_config["output_dir"],
        help="Directory where the LoRA adapter will be saved.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name. Defaults to a descriptive generated name.",
    )

    parser.add_argument("--wandb_project", type=str, default=wandb_config["project"])
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=wandb_config["entity"],
        help="W&B team/entity name.",
    )

    parser.add_argument("--base_model", type=str, default=model_config["base_model"])

    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=training_config["num_train_epochs"],
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=training_config["learning_rate"],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=training_config["warmup_ratio"],
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=training_config["max_grad_norm"],
    )
    parser.add_argument("--batch_size", type=int, default=training_config["batch_size"])
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=training_config["eval_batch_size"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=training_config["gradient_accumulation_steps"],
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
        )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=training_config["logging_steps"],
    )
    parser.add_argument("--save_steps", type=int, default=training_config["save_steps"])
    parser.add_argument(
        "--eval_strategy",
        type=str,
        choices=["no", "steps", "epoch"],
        default=training_config["eval_strategy"],
    )
    parser.add_argument("--eval_steps", type=int, default=training_config.get("eval_steps", 500))

    parser.add_argument("--lora_r", type=int, default=lora_config["r"])
    parser.add_argument("--lora_alpha", type=int, default=lora_config["alpha"])
    parser.add_argument("--lora_dropout", type=float, default=lora_config["dropout"])

    parser.add_argument(
        "--max_length",
        type=int,
        default=training_config.get("max_length", 2048),
        help="Maximum sequence length for SFT tokenization.",
    )

    parser.add_argument(
        "--no_assistant_only_loss",
        action="store_true",
        help="Disable assistant-only loss and train on the full formatted text.",
    )

    parser.add_argument(
        "--final_eval_file",
        type=str,
        default=training_config["final_eval_file"],
        help="Benchmark-style JSONL used for final generative accuracy.",
    )
    parser.add_argument(
        "--final_eval_max_examples",
        type=int,
        default=training_config["final_eval_max_examples"],
        help="Optional cap for final generative accuracy examples.",
    )
    parser.add_argument(
        "--final_eval_generations_path",
        type=str,
        default=training_config["final_eval_generations_path"],
    )
    parser.add_argument(
        "--final_eval_scores_path",
        type=str,
        default=training_config["final_eval_scores_path"],
    )
    parser.add_argument(
        "--final_eval_max_new_tokens",
        type=int,
        default=generation_config["max_new_tokens"],
    )
    parser.add_argument(
        "--final_eval_temperature",
        type=float,
        default=generation_config["temperature"],
    )
    parser.add_argument("--final_eval_top_p", type=float, default=generation_config["top_p"])
    parser.add_argument(
        "--disable_final_eval",
        action="store_true",
        default=not training_config["final_eval_enabled"],
        help="Skip final generation-based accuracy evaluation.",
    )

    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default= "true",
        help="Disable W&B logging.",
    )

    args = parser.parse_args()

    args.training_config = training_config
    args.wandb_config = wandb_config
    args.prompt_config = prompt_config

    return args


def force_qwen_thinking_mode(tokenizer, enable_thinking: bool = True):
    """
    Force Qwen chat template thinking mode.

    This matters because when we use assistant_only_loss=True, we let SFTTrainer
    apply the chat template directly from the tokenizer. So we need the template
    itself to encode thinking=False.
    """

    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return tokenizer

    template = tokenizer.chat_template

    # Avoid adding repeated settings if the script is run several times.
    template = template.replace("{%- set enable_thinking = true %}\n", "")
    template = template.replace("{%- set enable_thinking = false %}\n", "")

    setting = (
        "{%- set enable_thinking = true %}\n"
        if enable_thinking
        else "{%- set enable_thinking = false %}\n"
    )

    tokenizer.chat_template = setting + template
    return tokenizer


# Sentinels used to make `force_default_system_prompt` idempotent.
_GK_SYS_SENTINEL_BEGIN = "{# === GK_DEFAULT_SYSTEM_BEGIN === #}"
_GK_SYS_SENTINEL_END = "{# === GK_DEFAULT_SYSTEM_END === #}"


def force_default_system_prompt(tokenizer, system_prompt: str):
    """
    Inject a default system message into the chat template.
    """

    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return tokenizer

    template = tokenizer.chat_template

    # Strip a previous injection (if any) so we can update the prompt safely.
    if _GK_SYS_SENTINEL_BEGIN in template and _GK_SYS_SENTINEL_END in template:
        before = template.split(_GK_SYS_SENTINEL_BEGIN, 1)[0]
        after = template.split(_GK_SYS_SENTINEL_END, 1)[1].lstrip("\n")
        template = before + after

    # Escape the prompt for use inside a Jinja single-quoted string literal.
    escaped = (
        system_prompt
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )

    injection = (
        f"{_GK_SYS_SENTINEL_BEGIN}\n"
        "{%- if messages and messages[0].role != 'system' %}\n"
        f"    {{%- set messages = [{{'role': 'system', 'content': '{escaped}'}}] + messages %}}\n"
        "{%- endif %}\n"
        f"{_GK_SYS_SENTINEL_END}\n"
    )

    tokenizer.chat_template = injection + template
    return tokenizer


def main():
    args = parse_args()
    training_config = args.training_config
    wandb_config = args.wandb_config
    prompt_config = args.prompt_config

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    use_wandb = not args.disable_wandb
    run_name = args.run_name or training_config.get("run_name") or build_run_name(
        task="sft-lora-train",
        model_name=args.base_model,
        dataset_path=args.train_file,
        method="gk",
    )

    completion_only_loss = not args.no_assistant_only_loss

    wandb_tags = list(wandb_config.get("tags", [])) + [
        "train",
        "lora",
        "sft",
    ]

    if completion_only_loss:
        wandb_tags.append("completion_only_loss")
    else:
        wandb_tags.append("full_text_loss")

    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=wandb_tags,
            job_type="train",
            config={
                "run_type": "train",
                "model_type": "lora",
                "base_model": args.base_model,
                "train_file": args.train_file,
                "output_dir": args.output_dir,
                "method": "lora_sft",
                "adapter_output_dir": args.output_dir,
                "thinking_mode": "off",
                "completion_only_loss": completion_only_loss,
                "eval_file": args.eval_file,
                "eval_strategy": args.eval_strategy,
                "eval_steps": args.eval_steps,
                "learning_rate": args.learning_rate,
                "num_train_epochs": args.num_train_epochs,
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "max_length": args.max_length,
                "final_eval_file": args.final_eval_file,
                "final_eval_max_examples": args.final_eval_max_examples,
                "warmup_ratio": training_config.get("warmup_ratio", 0.0),
                "max_grad_norm": training_config.get("max_grad_norm", 1.0),
            },
        )

    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # enable_thinking=True: prompt ends with <|im_start|>assistant\n (no extra tokens).
    # enable_thinking=False: template appends <think>\n\n</think>\n\n, which corrupts
    # CoT training where completions already start with <think>.
    # Read from config: set True for CoT runs (v10, v11), False for no-think runs.
    enable_thinking = training_config.get("enable_thinking", False)
    tokenizer = force_qwen_thinking_mode(tokenizer, enable_thinking=enable_thinking)

    # Bake the SFT system prompt into the chat template as the default-when-missing,
    # so CI inference (which does not pass a system message) sees the same context
    # the model was trained on.
    tokenizer = force_default_system_prompt(
        tokenizer,
        system_prompt=prompt_config["sft_system_prompt"],
    )

    print(f"Loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    data_files = {"train": args.train_file}
    use_eval_dataset = bool(args.eval_file) and args.eval_strategy != "no"

    if use_eval_dataset:
        data_files["eval"] = args.eval_file

    print(f"Loading SFT data: {data_files}")
    dataset = load_dataset(
        "json",
        data_files=data_files,
    )
    def to_prompt_completion(example):
        messages = example["messages"]

        prompt_messages = messages[:-1]
        assistant_message = messages[-1]["content"]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        completion = assistant_message

        if tokenizer.eos_token is not None:
            completion = completion + tokenizer.eos_token

        return {
            "prompt": prompt,
            "completion": completion,
        }


    print("Converting messages dataset to prompt/completion format...")
    dataset = dataset.map(
        to_prompt_completion,
        remove_columns=dataset["train"].column_names,
    )
    def formatting_func(example):
        """
        Used only when assistant_only_loss is disabled.

        For assistant_only_loss=True, we should let SFTTrainer handle the
        conversational 'messages' format directly so it can mask assistant tokens.
        """
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
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

    training_args_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": "wandb" if use_wandb else "none",
        "run_name": run_name if use_wandb else None,
        "warmup_ratio": training_config.get("warmup_ratio", 0.05),
        "max_grad_norm": training_config.get("max_grad_norm", 1.0),
        "bf16": torch.cuda.is_available(),
        "fp16": False,
        "remove_unused_columns": False,
        "assistant_only_loss": False,
        "completion_only_loss": True,
        "packing": False,
        "max_length": args.max_length,
    }

    if use_eval_dataset:
        training_args_kwargs["eval_strategy"] = args.eval_strategy
        training_args_kwargs["eval_steps"] = args.eval_steps

    training_args = SFTConfig(**training_args_kwargs)

    print("=" * 80)
    print(f"completion_only_loss: {completion_only_loss}")
    print(f"max_length: {args.max_length}")
    print("=" * 80)

    trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["eval"] if use_eval_dataset else None,
    formatting_func=None,
    peft_config=lora_config,
    args=training_args,
)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("saving model...")
    #tokenizer = force_qwen_thinking_mode(tokenizer, enable_thinking=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if not args.disable_final_eval:
        print(f"Running final generative eval on: {args.final_eval_file}")
        eval_examples = load_jsonl(
            args.final_eval_file,
            max_examples=args.final_eval_max_examples,
        )
        generation_rows = generate_completion_rows(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=eval_examples,
            max_new_tokens=args.final_eval_max_new_tokens,
            temperature=args.final_eval_temperature,
            top_p=args.final_eval_top_p,
            desc="Final generative eval",
        )
        save_jsonl(generation_rows, args.final_eval_generations_path)

        score_result = score_generation_rows(generation_rows, method="boxed")
        save_json(score_result, args.final_eval_scores_path)
        final_metrics = flatten_score_metrics(score_result, prefix="final_eval")
        print(f"Final eval metrics: {final_metrics}")

        if use_wandb:
            import wandb

            wandb.log(final_metrics)
            log_wandb_eval_artifacts(
                wandb=wandb,
                artifact_name=f"{run_name}-final-eval",
                generations_path=args.final_eval_generations_path,
                scores_path=args.final_eval_scores_path,
            )

    if use_wandb:
        import wandb

        wandb.finish()

    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()