"""
Merge a LoRA adapter into its training base and save a full model.

Usage:
  python merge_lora.py --path PATH [--language english|multi] [--lora-dir OVERRIDE]

Paths and their merge bases:
  distillation         — base_model,              thinking ON  (intermediate)
  sft                  — base_model,              thinking OFF (intermediate)
  distillation_sft     — distillation_merged,     thinking OFF (intermediate)
  simple_sft           — base_model,              thinking OFF (final)
  cold_grpo            — base_model,              thinking OFF (final)
  sft_grpo             — sft_merged,              thinking OFF (final)
  distillation_sft_grpo— distillation_sft_merged, thinking OFF (final)

Final paths (simple_sft + grpo paths) get the system prompt baked into the chat
template and a GenerationConfig saved alongside the model.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ML_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

CONFIG_PATH = ML_ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

from multilinguality.scripts.utils import ALL_PATHS, GRPO_PATHS, with_lang, with_thinking

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

BASE_MODEL              = CFG["model"]["base_model"]
CHAT_TEMPLATE_TOFF_PATH = ML_ROOT / "chat_template.jinja"
CHAT_TEMPLATE_TON_PATH  = ML_ROOT / "chat_template_ton.jinja"
GEN_CFG                 = CFG["merge"]
SYSTEM_PROMPT           = CFG["prompts"]["inference_system"]
PATHS_CFG               = CFG["paths"]

_SYS_BEGIN = "{# === MULTILINGUAL_SYSTEM_BEGIN === #}"
_SYS_END   = "{# === MULTILINGUAL_SYSTEM_END === #}"


def inject_system_prompt(tokenizer, system_prompt: str):
    """Bake system_prompt into the chat template so inference needs no system message."""
    template = tokenizer.chat_template

    if _SYS_BEGIN in template and _SYS_END in template:
        before = template.split(_SYS_BEGIN, 1)[0]
        after  = template.split(_SYS_END, 1)[1].lstrip("\n")
        template = before + after

    escaped = (
        system_prompt
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )

    injection = (
        f"{_SYS_BEGIN}\n"
        "{%- if messages and messages[0].role != 'system' %}\n"
        f"    {{%- set messages = [{{'role': 'system', 'content': '{escaped}'}}] + messages %}}\n"
        "{%- endif %}\n"
        f"{_SYS_END}\n"
    )

    tokenizer.chat_template = injection + template
    return tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="Which adapter to merge. Known paths: "
                             + ", ".join(ALL_PATHS) + ". For custom paths use --lora-dir.")
    parser.add_argument("--language", choices=["english", "multi"], default="english",
                        help="Must match the --language used during training.")
    parser.add_argument("--lora-dir", default=None,
                        help="Override LoRA adapter path (e.g. a mid-training checkpoint subdir).")
    parser.add_argument("--thinking-off", action="store_true",
                        help="Bake the non-thinking chat template into the merged model. "
                             "Default: thinking ON.")
    parser.add_argument("--final", action="store_true",
                        help="Inject system prompt and save GenerationConfig. "
                             "Default: auto-detected from path (simple_sft + grpo paths).")
    parser.add_argument("--suffix", default="",
                        help="Append suffix to lora/merged dir names.")
    args = parser.parse_args()

    lang_for_path = args.language if args.path in GRPO_PATHS or args.path == "distillation" else "english"
    thinking_on   = not args.thinking_off
    path_cfg      = PATHS_CFG.get(args.path)

    def apply_suffix(path_str: str, suffix: str) -> str:
        for end in ("_lora", "_merged"):
            if path_str.endswith(end):
                return path_str[:-len(end)] + suffix + end
        return path_str + suffix

    if path_cfg:
        base_lora   = with_lang(path_cfg["lora_dir"], lang_for_path)
        base_merged = with_lang(path_cfg["merged_dir"], lang_for_path)
        lora_dir    = Path(args.lora_dir) if args.lora_dir else ml_path(base_lora)
        merged_dir  = ml_path(with_thinking(apply_suffix(base_merged, args.suffix), thinking_on))
    else:
        lora_dir   = Path(args.lora_dir) if args.lora_dir else ml_path(f"multilinguality/outputs/{args.path}_lora")
        merged_dir = ml_path(f"multilinguality/outputs/{args.path}_merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    # ── resolve merge base ────────────────────────────────────────────────────
    if not path_cfg or args.path in ("distillation", "sft", "simple_sft", "cold_grpo"):
        merge_base = BASE_MODEL
    elif args.path == "sft_grpo":
        merge_base = str(ml_path(with_thinking(PATHS_CFG["sft"]["merged_dir"], thinking_on)))
    elif args.path == "distillation_sft":
        merge_base = str(ml_path(with_thinking(with_lang(PATHS_CFG["distillation"]["merged_dir"], lang_for_path), thinking_on)))
    elif args.path == "distillation_sft_grpo":
        merge_base = str(ml_path(with_thinking(with_lang(PATHS_CFG["distillation_sft"]["merged_dir"], lang_for_path), thinking_on)))
    else:
        raise ValueError(f"Unknown path: {args.path!r}")

    print(f"[path={args.path}] merge_base={merge_base}  lora={lora_dir}  out={merged_dir}")

    tokenizer = AutoTokenizer.from_pretrained(merge_base, trust_remote_code=True)
    template_path = CHAT_TEMPLATE_TON_PATH if thinking_on else CHAT_TEMPLATE_TOFF_PATH
    tokenizer.chat_template = template_path.read_text()

    is_final = args.final or (args.path == "simple_sft" or args.path in GRPO_PATHS)
    if is_final:
        tokenizer = inject_system_prompt(tokenizer, SYSTEM_PROMPT)

    print(f"Loading base model: {merge_base}")
    base = AutoModelForCausalLM.from_pretrained(
        merge_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Merging LoRA adapter...")
    model = PeftModel.from_pretrained(base, str(lora_dir))
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(merged_dir))

    if args.path == "simple_sft":
        GenerationConfig(
            do_sample=False,
            max_new_tokens=32,
            bos_token_id=151643,
            eos_token_id=151645,
            pad_token_id=151643,
        ).save_pretrained(str(merged_dir))
        print("Saved greedy generation_config (do_sample=False, max_new_tokens=32).")
    elif args.path in GRPO_PATHS:
        GenerationConfig(
            do_sample=True,
            temperature=GEN_CFG["generation_temperature"],
            top_p=GEN_CFG["generation_top_p"],
            top_k=GEN_CFG["generation_top_k"],
            repetition_penalty=1.1,
            bos_token_id=151643,
            eos_token_id=[151645, 151643],
            pad_token_id=151643,
        ).save_pretrained(str(merged_dir))
        print("Saved sampling generation_config.")

    print("Done.")


if __name__ == "__main__":
    main()
