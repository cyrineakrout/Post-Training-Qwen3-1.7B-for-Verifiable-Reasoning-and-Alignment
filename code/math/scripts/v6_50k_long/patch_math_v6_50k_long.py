import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT_DIR / "math" / "outputs" / "math_v6_50k_long_merged_ckpt1500"
model_dir = Path(os.environ.get("MATH_MODEL_DIR", DEFAULT_MODEL_DIR))

if not model_dir.exists():
    raise FileNotFoundError(f"Model dir not found: {model_dir}")

CHAT_TEMPLATE = r"""{%- set boxed_instruction = "You are a mathematical reasoning model. Solve the problem carefully. Use concise scratch work with only the essential intermediate steps. Return the final answer in exact form whenever possible. Put the final answer inside a LaTeX \\boxed{} expression. Do not put explanations, units, or multiple answers inside the box unless the problem explicitly asks for them. " %}
{%- if messages|length > 0 and messages[0]['role'] == 'system' %}
    {%- set system_message = boxed_instruction + "\n\n" + messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = boxed_instruction %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '<|im_start|>system\n' + system_message + '<|im_end|>\n' }}
{%- for message in loop_messages %}
    {{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<think>\n' }}
{%- endif %}"""

# 1. chat_template.jinja
(model_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")

# 2. tokenizer_config.json
tok_path = model_dir / "tokenizer_config.json"
tok = json.loads(tok_path.read_text(encoding="utf-8"))

tok["chat_template"] = CHAT_TEMPLATE
tok["add_prefix_space"] = False
tok["clean_up_tokenization_spaces"] = False
tok["bos_token"] = None
tok["eos_token"] = "<|im_end|>"
tok["pad_token"] = "<|endoftext|>"
tok["split_special_tokens"] = False
tok["truncation_side"] = "left"
tok["model_max_length"] = 131072
tok["tokenizer_class"] = "Qwen2Tokenizer"

tok_path.write_text(json.dumps(tok, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# 3. generation_config.json
gen = {
    "bos_token_id": 151643,
    "do_sample": True,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "max_new_tokens": 4096
}
(model_dir / "generation_config.json").write_text(
    json.dumps(gen, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)

# 4. config.json
cfg_path = model_dir / "config.json"
cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

cfg["bos_token_id"] = None
cfg["eos_token_id"] = 151645
cfg["pad_token_id"] = 151643
cfg["use_cache"] = False

# Keep dtype style already present, but ensure one exists.
if "dtype" not in cfg and "torch_dtype" in cfg:
    cfg["dtype"] = cfg["torch_dtype"]
if "dtype" not in cfg:
    cfg["dtype"] = "bfloat16"

cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

# 5. sanity required files
required = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors",
]
missing = [f for f in required if not (model_dir / f).exists()]
if missing:
    raise FileNotFoundError("Missing required files: " + ", ".join(missing))

print("Patched submission files in:", model_dir)
print("Required files are present.")
