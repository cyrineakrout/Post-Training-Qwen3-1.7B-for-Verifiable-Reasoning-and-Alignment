import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from runtime_utils import resolve_attention_implementation


CHAT_TEMPLATE = r"""{%- set enable_thinking = true %}
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


def patch_files(out_dir: Path):
    # generation config
    gen = {
        "bos_token_id": 151643,
        "do_sample": True,
        "eos_token_id": [151645, 151643],
        "pad_token_id": 151643,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
        "transformers_version": "5.7.0",
    }
    (out_dir / "generation_config.json").write_text(
        json.dumps(gen, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # chat template
    (out_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")

    # config
    cfg_path = out_dir / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["bos_token_id"] = None
    cfg["eos_token_id"] = 151645
    cfg["pad_token_id"] = 151643
    cfg["use_cache"] = False
    cfg["dtype"] = "bfloat16"
    cfg["transformers_version"] = "5.7.0"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # tokenizer config
    tok_path = out_dir / "tokenizer_config.json"
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    tok.pop("chat_template", None)
    tok["add_prefix_space"] = False
    tok["backend"] = "tokenizers"
    tok["bos_token"] = None
    tok["clean_up_tokenization_spaces"] = False
    tok["eos_token"] = "<|im_end|>"
    tok["is_local"] = True
    tok["local_files_only"] = False
    tok["model_max_length"] = 131072
    tok["pad_token"] = "<|endoftext|>"
    tok["split_special_tokens"] = False
    tok["tokenizer_class"] = "Qwen2Tokenizer"
    tok["unk_token"] = None
    tok_path.write_text(json.dumps(tok, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    if not (ckpt / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing LoRA checkpoint adapter_config.json: {ckpt}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Base model:", args.base_model)
    print("LoRA checkpoint:", ckpt)
    print("Output dir:", out_dir)
    attn_implementation = resolve_attention_implementation()
    print("Attention implementation:", attn_implementation)

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    tok.chat_template = CHAT_TEMPLATE
    tok.model_max_length = 131072
    tok.pad_token = "<|endoftext|>"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )

    model = PeftModel.from_pretrained(base, str(ckpt))
    model = model.merge_and_unload()

    print("Saving merged model...")
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    patch_files(out_dir)

    print("Done:", out_dir)


if __name__ == "__main__":
    main()
