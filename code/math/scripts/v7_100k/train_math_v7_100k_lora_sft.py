import os
import json
import math
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import dataset_repo_id, load_dataset_version
from runtime_utils import resolve_attention_implementation


# ============================================================
# Config from environment
# ============================================================

BASE_MODEL = os.environ.get("MATH_BASE_MODEL", "Qwen/Qwen3-1.7B")
DATASET_VERSION = "v7_100k"
OUTPUT_DIR = os.environ.get("MATH_OUTPUT_DIR", "math/outputs/math_v7_100k_lora")

MAX_SEQ_LEN = int(os.environ.get("MATH_MAX_SEQ_LEN", "16000"))
LR = float(os.environ.get("MATH_LR", "7.5e-5"))
EPOCHS = float(os.environ.get("MATH_NUM_EPOCHS", "1.0"))
BATCH_SIZE = int(os.environ.get("MATH_BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("MATH_GRAD_ACCUM", "32"))
MAX_STEPS = int(os.environ.get("MATH_MAX_STEPS", "-1"))
RESUME_FROM_CHECKPOINT = os.environ.get("MATH_RESUME_FROM_CHECKPOINT", None)

SAVE_STEPS = int(os.environ.get("MATH_SAVE_STEPS", "100"))
LOGGING_STEPS = int(os.environ.get("MATH_LOGGING_STEPS", "20"))

# For speed: do not evaluate during training. We will evaluate selected checkpoints after.
DO_EVAL = os.environ.get("MATH_DO_EVAL", "0") == "1"
EVAL_STEPS = int(os.environ.get("MATH_EVAL_STEPS", "500"))

SEED = int(os.environ.get("MATH_SEED", "42"))

LORA_R = int(os.environ.get("MATH_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("MATH_LORA_ALPHA", "128"))
LORA_DROPOUT = float(os.environ.get("MATH_LORA_DROPOUT", "0.05"))

MAP_NUM_PROC = int(os.environ.get("MATH_MAP_NUM_PROC", "4"))
ATTN_IMPLEMENTATION = resolve_attention_implementation()

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


print("=" * 100)
print("V7 LORA SFT CONFIG")
print("BASE_MODEL:", BASE_MODEL)
print("DATASET_REPO:", dataset_repo_id())
print("DATASET_VERSION:", DATASET_VERSION)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("MAX_SEQ_LEN:", MAX_SEQ_LEN)
print("LR:", LR)
print("EPOCHS:", EPOCHS)
print("BATCH_SIZE:", BATCH_SIZE)
print("GRAD_ACCUM:", GRAD_ACCUM)
print("EFFECTIVE_BATCH:", BATCH_SIZE * GRAD_ACCUM)
print("MAX_STEPS:", MAX_STEPS)
print("RESUME_FROM_CHECKPOINT:", RESUME_FROM_CHECKPOINT)
print("SAVE_STEPS:", SAVE_STEPS)
print("LOGGING_STEPS:", LOGGING_STEPS)
print("DO_EVAL:", DO_EVAL)
print("EVAL_STEPS:", EVAL_STEPS)
print("LORA_R:", LORA_R)
print("LORA_ALPHA:", LORA_ALPHA)
print("LORA_DROPOUT:", LORA_DROPOUT)
print("MAP_NUM_PROC:", MAP_NUM_PROC)
print("ATTN_IMPLEMENTATION:", ATTN_IMPLEMENTATION)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
print("=" * 100)


# ============================================================
# Chat template: strong 46–56% style
# ============================================================

MATH_CHAT_TEMPLATE = r"""{%- set enable_thinking = true %}
{%- set boxed_answer_instruction = 'For math problems, reason carefully and put the final answer exactly once inside \\boxed{...}.' %}
{%- if messages and messages[0]['role'] == 'system' %}
{%- set messages = [{'role': 'system', 'content': messages[0]['content'] + '\n' + boxed_answer_instruction}] + messages[1:] %}
{%- else %}
{%- set messages = [{'role': 'system', 'content': boxed_answer_instruction}] + messages %}
{%- endif %}
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
{%- if add_generation_prompt %}<think>
{% endif %}"""


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    use_fast=True,
)

tokenizer.chat_template = MATH_CHAT_TEMPLATE
tokenizer.model_max_length = 131072
tokenizer.padding_side = "right"

# Strong config style: pad = <|endoftext|> = 151643
tokenizer.pad_token = "<|endoftext|>"

print("tokenizer.pad_token:", tokenizer.pad_token)
print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
print("tokenizer.eos_token:", tokenizer.eos_token)
print("tokenizer.eos_token_id:", tokenizer.eos_token_id)


# ============================================================
# Dataset tokenization
# ============================================================

def get_messages(example):
    messages = example.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Each row must have a messages list.")
    return messages


def split_prompt_and_full(messages):
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        raise ValueError("No assistant message found.")

    prompt_messages = messages[:last_assistant_idx]
    full_messages = messages[:last_assistant_idx + 1]

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

    return prompt_text, full_text


def tokenize_one(example):
    messages = get_messages(example)

    try:
        prompt_text, full_text = split_prompt_and_full(messages)
    except Exception:
        return {
            "input_ids": [tokenizer.eos_token_id],
            "attention_mask": [1],
            "labels": [-100],
        }

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]

    prompt_len = min(len(prompt_ids), len(full_ids))

    # Left-truncate to preserve final answer.
    if len(full_ids) > MAX_SEQ_LEN:
        start = len(full_ids) - MAX_SEQ_LEN
        input_ids = full_ids[start:]

        labels = []
        for original_idx in range(start, len(full_ids)):
            if original_idx < prompt_len:
                labels.append(-100)
            else:
                labels.append(full_ids[original_idx])
    else:
        input_ids = full_ids
        labels = [-100] * prompt_len + full_ids[prompt_len:]

    attention_mask = [1] * len(input_ids)

    if len(labels) != len(input_ids):
        labels = labels[:len(input_ids)]
        if len(labels) < len(input_ids):
            labels += [-100] * (len(input_ids) - len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class CausalCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        pad_id = self.tokenizer.pad_token_id

        input_ids_batch = []
        attention_batch = []
        labels_batch = []

        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f["attention_mask"]
            labels = f["labels"]

            pad_len = max_len - len(input_ids)

            input_ids_batch.append(input_ids + [pad_id] * pad_len)
            attention_batch.append(attention_mask + [0] * pad_len)
            labels_batch.append(labels + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


print("Loading datasets...")
raw = load_dataset_version(DATASET_VERSION)
print(raw)

print("Tokenizing train dataset...")
train_ds = raw["train"].map(
    tokenize_one,
    remove_columns=raw["train"].column_names,
    num_proc=MAP_NUM_PROC,
    desc="Tokenizing train",
)

eval_ds = None
if DO_EVAL and "validation" in raw:
    print("Tokenizing validation dataset...")
    eval_ds = raw["validation"].map(
        tokenize_one,
        remove_columns=raw["validation"].column_names,
        num_proc=MAP_NUM_PROC,
        desc="Tokenizing validation",
    )

print("Train tokenized:", train_ds)
if eval_ds is not None:
    print("Eval tokenized:", eval_ds)


# ============================================================
# Model + LoRA
# ============================================================

print("Loading base model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    attn_implementation=ATTN_IMPLEMENTATION,
    trust_remote_code=True,
)

model.config.use_cache = False

model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("total_params:", total_params)
print("trainable_params:", trainable_params)
print("trainable_fraction:", trainable_params / total_params)


# ============================================================
# Training
# ============================================================

eval_strategy = "steps" if DO_EVAL and eval_ds is not None else "no"

args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,
    max_grad_norm=1.0,

    bf16=True,
    fp16=False,
    tf32=True,

    logging_steps=LOGGING_STEPS,

    save_steps=SAVE_STEPS,
    save_strategy="steps",
    save_total_limit=20,

    eval_strategy=eval_strategy,
    eval_steps=EVAL_STEPS,

    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    report_to=[],
    seed=SEED,

    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=CausalCollator(tokenizer),
    processing_class=tokenizer,
)

print("=" * 100)
print("Starting LoRA SFT...")
print("=" * 100)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)

print("=" * 100)
print("Saving final adapter/tokenizer...")
print("=" * 100)

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save generation/config/template files in the same robust style.
generation_config = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "transformers_version": "5.7.0",
}

with open(Path(OUTPUT_DIR) / "generation_config.json", "w", encoding="utf-8") as f:
    json.dump(generation_config, f, indent=2)
    f.write("\n")

with open(Path(OUTPUT_DIR) / "chat_template.jinja", "w", encoding="utf-8") as f:
    f.write(MATH_CHAT_TEMPLATE)

tok_cfg_path = Path(OUTPUT_DIR) / "tokenizer_config.json"
if tok_cfg_path.exists():
    tok_cfg = json.loads(tok_cfg_path.read_text())
    tok_cfg.pop("chat_template", None)
    tok_cfg["pad_token"] = "<|endoftext|>"
    tok_cfg["eos_token"] = "<|im_end|>"
    tok_cfg["bos_token"] = None
    tok_cfg["model_max_length"] = 131072
    tok_cfg["tokenizer_class"] = "Qwen2Tokenizer"
    tok_cfg["is_local"] = True
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2) + "\n")

metadata = {
    "base_model": BASE_MODEL,
    "dataset_repo": dataset_repo_id(),
    "dataset_version": DATASET_VERSION,
    "output_dir": OUTPUT_DIR,
    "max_seq_length": MAX_SEQ_LEN,
    "lr": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "effective_batch": BATCH_SIZE * GRAD_ACCUM,
    "max_steps": MAX_STEPS,
    "lora": True,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "trainable_params": trainable_params,
    "total_params": total_params,
    "save_steps": SAVE_STEPS,
    "do_eval": DO_EVAL,
}

with open(Path(OUTPUT_DIR) / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)
    f.write("\n")

print("Done.")
