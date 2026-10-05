"""
SFT for multilingual reasoning on Qwen3-1.7B.

Paths:
  distillation       — teacher-trace distillation from base model, thinking ON
  sft                — cold-start format teaching from base model, thinking OFF
  distillation_sft   — format teaching starting from distillation_merged, thinking OFF
"""

import argparse
import json
import random
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

CONFIG_PATH = ML_ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

from multilinguality.scripts.utils import SFT_PATHS, with_lang, with_thinking

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

MODEL_CFG  = CFG["model"]
SFT_CFG    = CFG["sft"]
DIST_CFG   = CFG["distillation"]
LORA_CFG   = CFG["lora"]
WANDB_CFG  = CFG["wandb"]
PATHS_CFG  = CFG["paths"]

CHAT_TEMPLATE_PATH = ML_ROOT / "chat_template.jinja"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, choices=SFT_PATHS,
                        help="Training path: distillation | sft | distillation_sft")
    parser.add_argument("--language", choices=["english", "multi"], default="english",
                        help="Distillation language mode — selects distillation_train_{english|multi}.jsonl. "
                             "Only used when --path distillation.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    parser.add_argument("--thinking-off", action="store_true",
                        help="Use the _toff merged base. Default: _ton.")
    parser.add_argument("--suffix", default="",
                        help="Append suffix to the lora output dir, e.g. '4k' → sft4k_lora.")
    args = parser.parse_args()

    import torch
    import wandb
    import shutil
    from datasets import Dataset, load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    path_cfg = PATHS_CFG[args.path]
    lang_for_path = args.language if args.path in ("distillation", "distillation_sft") else "english"
    out_dir = ml_path(with_lang(path_cfg["lora_dir"], lang_for_path))
    if args.suffix:
        out_dir = out_dir.parent / (out_dir.name.replace("_lora", args.suffix + "_lora"))

    if args.restart and out_dir.exists():
        print(f"--restart: removing {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    thinking_on = (args.path == "distillation")

    if args.path == "distillation":
        hypers     = DIST_CFG
        base       = DIST_CFG["train_file"].replace(".jsonl", "")
        train_file = ml_path(f"{base}_{args.language}.jsonl")
        if not train_file.exists():
            raise FileNotFoundError(
                f"Distillation file not found: {train_file}\n"
                f"Run: python code/multilinguality/scripts/generate_distillation_data.py --language {args.language}"
            )
        base_model = MODEL_CFG["base_model"]
        tags       = ["sft", "distillation", "thinking-on", "multilingual", "qwen3-1.7b"]
    elif args.path == "sft":
        hypers     = SFT_CFG
        train_file = ml_path(SFT_CFG["train_file"])
        base_model = MODEL_CFG["base_model"]
        tags       = ["sft", "cold-start", "multilingual", "qwen3-1.7b"]
    elif args.path == "distillation_sft":
        hypers     = SFT_CFG
        train_file = ml_path(SFT_CFG["train_file"])
        base_model = str(ml_path(with_thinking(with_lang(PATHS_CFG["distillation"]["merged_dir"], args.language), not args.thinking_off)))
        tags       = ["sft", "distillation-sft", "multilingual", "qwen3-1.7b"]
    elif args.path == "simple_sft":
        hypers     = SFT_CFG
        train_file = ml_path(SFT_CFG["train_file"])
        base_model = MODEL_CFG["base_model"]
        tags       = ["sft", "simple", "greedy", "multilingual", "qwen3-1.7b"]
    else:
        raise ValueError(f"Unknown SFT path: {args.path!r}. Expected one of {SFT_PATHS}")

    print(f"[path={args.path}] base={base_model}  out={out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if not thinking_on:
        tokenizer.chat_template = CHAT_TEMPLATE_PATH.read_text()

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.path == "distillation":
        records  = [r for l in train_file.open(encoding="utf-8")
                    if "messages" in (r := json.loads(l))]
        random.Random(42).shuffle(records)
        n_val    = max(1, int(len(records) * 0.05))
        train_ds = Dataset.from_list(records[n_val:])
        val_ds   = Dataset.from_list(records[:n_val])
        print(f"Distillation data: {len(train_ds)} train / {len(val_ds)} val")
    else:
        train_ds = load_dataset("json", data_files=str(train_file), split="train")

    def to_prompt_completion(example):
        messages   = example["messages"]
        prompt     = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        completion = messages[-1]["content"] + tokenizer.eos_token
        return {"prompt": prompt, "completion": completion}

    train_ds = train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)

    lora_config = LoraConfig(
        r=LORA_CFG["r"],
        lora_alpha=LORA_CFG["alpha"],
        lora_dropout=LORA_CFG["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    wandb.init(project=WANDB_CFG["project"], entity=WANDB_CFG["entity"],
               tags=tags, name=args.path)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=hypers["num_train_epochs"],
        per_device_train_batch_size=hypers["batch_size"],
        gradient_accumulation_steps=hypers["gradient_accumulation_steps"],
        learning_rate=hypers["learning_rate"],
        logging_steps=hypers["logging_steps"],
        save_steps=hypers["save_steps"],
        eval_strategy="no",
        max_length=hypers["max_length"],
        completion_only_loss=True,
        report_to="wandb",
        bf16=True,
        fp16=False,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        peft_config=lora_config,
        args=sft_config,
    )

    print(f"Starting SFT training ({args.path})...")
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Adapter saved to {out_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
