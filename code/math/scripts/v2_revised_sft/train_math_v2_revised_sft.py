import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import load_dataset_version

MATH_DIR = ROOT_DIR / "math"
OUTPUT_DIR = MATH_DIR / "outputs" / "math_v2_revised_sft_lora"

BASE_MODEL = "Qwen/Qwen3-1.7B"


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Important for training stability with gradient checkpointing.
    model.config.use_cache = False

    dataset = load_dataset_version("v2_revised_sft")

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # Batch setup for A100 40GB.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        # Lower LR than v1 because MATH is harder and more diverse.
        learning_rate=1e-5,
        num_train_epochs=1,

        # Logging/checkpointing.
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,

        # A100 supports bf16.
        bf16=True,

        # Memory helper.
        gradient_checkpointing=True,

        # Avoid wandb unless explicitly enabled.
        report_to="none",

        # Needed because the dataset contains "messages".
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        formatting_func=formatting_func,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"LoRA v2 adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
