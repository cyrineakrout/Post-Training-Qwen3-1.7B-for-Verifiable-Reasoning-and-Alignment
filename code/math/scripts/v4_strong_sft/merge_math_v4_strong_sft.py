import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[3]

BASE_MODEL = os.environ.get("MATH_BASE_MODEL", "Qwen/Qwen3-1.7B")
ADAPTER_DIR = os.environ.get("MATH_ADAPTER_DIR", str(ROOT_DIR / "math/outputs/math_v4_strong_sft_lora"))
MERGED_DIR = os.environ.get("MATH_MERGED_DIR", str(ROOT_DIR / "math/outputs/math_v4_strong_sft_merged"))


CHAT_TEMPLATE = r"""{%- if tools %}
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
    {%- else %}
        {{- '<|im_start|>system\nYou are a mathematical reasoning model. Solve the problem carefully and concisely. Return the final answer in exact form when possible. Put only the final answer inside one LaTeX \\boxed{} expression.<|im_end|>\n' }}
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
    {{- '<think>\n\n</think>\n\n' }}
{%- endif %}"""


GENERATION_CONFIG = {
    "do_sample": True,
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "eos_token_id": [151645, 151643],
    "pad_token_id": 151643,
}


def main() -> None:
    print("=" * 100)
    print("[math-v4] Merge LoRA")
    print("=" * 100)
    print(f"Base model:  {BASE_MODEL}")
    print(f"Adapter dir: {ADAPTER_DIR}")
    print(f"Output dir:  {MERGED_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()

    merged_dir = Path(MERGED_DIR)
    merged_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(
        merged_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(merged_dir)

    # Write standalone chat template.
    (merged_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")

    # Patch tokenizer_config.json explicitly.
    tok_cfg_path = merged_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
    tok_cfg["chat_template"] = CHAT_TEMPLATE
    tok_cfg["eos_token"] = "<|im_end|>"
    tok_cfg["pad_token"] = "<|endoftext|>"
    tok_cfg_path.write_text(json.dumps(tok_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Patch config.json explicitly.
    cfg_path = merged_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["eos_token_id"] = 151645
    cfg["pad_token_id"] = 151643
    cfg["use_cache"] = True
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Patch generation config.
    (merged_dir / "generation_config.json").write_text(
        json.dumps(GENERATION_CONFIG, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(f"[math-v4] Merged model saved to {MERGED_DIR}")
    print("Patched:")
    print("- chat_template.jinja")
    print("- tokenizer_config.json")
    print("- generation_config.json")
    print("- config.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
