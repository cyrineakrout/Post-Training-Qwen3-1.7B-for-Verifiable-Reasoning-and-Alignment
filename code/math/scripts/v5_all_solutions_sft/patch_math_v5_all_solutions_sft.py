import json
import os
from pathlib import Path


MODEL_DIR = Path(os.environ.get("MATH_MODEL_DIR", "math/outputs/math_v5_all_solutions_sft_merged_r96_e08"))

CHAT_TEMPLATE = r"""{%- set enable_thinking = false %}
{%- set boxed_instruction = "You are a mathematical reasoning model. Solve the problem carefully. Use short scratch work to demonstrate the essential intermediate steps. Answer before the token budget is exhausted. Return the final answer in exact form whenever possible. Put only the final answer inside a LaTeX \\boxed{} expression. Do not put explanations or units inside the box. If the problem asks for all values, all functions, all solutions, or all possible cases, include the complete list inside the box." %}
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
    {{- '<|im_start|>assistant\n' }}
    {%- if not enable_thinking %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"""


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def main():
    if not MODEL_DIR.exists():
        raise SystemExit(f"Missing model dir: {MODEL_DIR}")

    (MODEL_DIR / "chat_template.jinja").write_text(CHAT_TEMPLATE)

    tok_cfg_path = MODEL_DIR / "tokenizer_config.json"
    tok_cfg = load_json(tok_cfg_path)
    tok_cfg["chat_template"] = CHAT_TEMPLATE
    tok_cfg["bos_token"] = None
    tok_cfg["eos_token"] = "<|im_end|>"
    tok_cfg["pad_token"] = "<|endoftext|>"
    save_json(tok_cfg, tok_cfg_path)

    gen_cfg = {
        "bos_token_id": 151643,
        "do_sample": True,
        "eos_token_id": [151645, 151643],
        "max_new_tokens": 4096,
        "pad_token_id": 151643,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.9
    }
    save_json(gen_cfg, MODEL_DIR / "generation_config.json")

    cfg_path = MODEL_DIR / "config.json"
    cfg = load_json(cfg_path)
    if cfg:
        cfg["dtype"] = "bfloat16"
        cfg["torch_dtype"] = "bfloat16"
        save_json(cfg, cfg_path)

    print(f"Patched math v5 submission files in {MODEL_DIR}")


if __name__ == "__main__":
    main()
