"""
Merge 4 domain LoRA adapters (safety, GK, multilingual, math) into a single
group model using weighted linear combination of deltas.

Why linear over TIES:
  TIES trims + sign-elects on the delta tensors. When adapters have very
  different ranks and training scales, this step prunes too aggressively
  and zeros out most domain signal. Linear combination keeps all deltas
  and scales them by weight — simpler but more robust when domains conflict.

Usage (config-driven, downloads adapters from HF automatically):
  python scripts/merge_adapters.py --config configs/lora_direct_v1.json

Usage (explicit local paths):
  python scripts/merge_adapters.py \
      --safety_adapter  adapters/safety \
      --gk_adapter      adapters/general_knowledge \
      --ml_adapter      adapters/multilingual \
      --math_adapter    adapters/math \
      --output_dir      models/lora_direct_v1_reproduced
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


BASE_MODEL = "Qwen/Qwen3-1.7B"

SYSTEM_PROMPT = (
    "You are a careful reasoning assistant optimized for four benchmark types: "
    "mathematics, general knowledge, multilingual questions, and safety questions.\n\n"
    "Core behavior:\n"
    "- Read the full question and all answer choices before deciding.\n"
    "- First identify the task type: math, knowledge, multilingual, safety, or mixed reasoning.\n"
    "- For multiple-choice questions, choose the single best answer from the provided options.\n"
    "- For math questions, solve step by step, verify arithmetic, and give the exact final result when possible.\n"
    "- For general knowledge questions, rely on established facts and avoid guessing from irrelevant details.\n"
    "- For multilingual questions, understand the question in its original language and reason accordingly.\n"
    "- For safety questions, choose the safest, least harmful, most responsible, and most policy-compliant option.\n"
    "- If choices are provided, never invent a new answer outside the choices.\n\n"
    "Output format:\n"
    "- Always reason inside <think> ... </think>.\n"
    "- Always finish with exactly one final answer inside one LaTeX \\boxed{...}.\n"
    "- If answer choices are provided, put only the option letter inside the box, for example \\boxed{C}.\n"
    "- If the task is free-response math, put only the exact mathematical answer inside the box.\n"
    "- Do not put explanations, units, or extra words inside the box unless explicitly required.\n"
    "- Do not write anything after the boxed answer."
)

_SYS_BEGIN = "{# GROUP_DEFAULT_SYSTEM_BEGIN #}"
_SYS_END   = "{# GROUP_DEFAULT_SYSTEM_END #}"


def inject_system_prompt(tokenizer):
    template = tokenizer.chat_template or ""
    if _SYS_BEGIN in template and _SYS_END in template:
        before = template.split(_SYS_BEGIN, 1)[0]
        after  = template.split(_SYS_END,   1)[1].lstrip("\n")
        template = before + after

    escaped = (
        SYSTEM_PROMPT
        .replace("\\", "\\\\")
        .replace("'",  "\\'")
        .replace("\n", "\\n")
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


def patch_tokenizer_config(out_dir: Path):
    cfg_path = out_dir / "tokenizer_config.json"
    if not cfg_path.exists():
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir), trust_remote_code=True)
    cfg["chat_template"] = tokenizer.chat_template
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("  tokenizer_config.json patched")


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def resolve_adapter(name: str, adapter_cfg: dict) -> str:
    """Return local adapter path, downloading from HF if not already present."""
    local = adapter_cfg.get("local_path", "")
    if local and Path(local).exists():
        print(f"  [{name}] Using local adapter: {local}")
        return local

    hf_repo = adapter_cfg.get("hf_repo")
    hf_path = adapter_cfg.get("hf_path", "")
    if not hf_repo:
        raise FileNotFoundError(
            f"Adapter '{name}' not found at '{local}' and no hf_repo in config."
        )

    print(f"  [{name}] Downloading from {hf_repo}/{hf_path} ...")
    from huggingface_hub import snapshot_download

    patterns = [f"{hf_path}/*", f"{hf_path}/**/*"] if hf_path else None
    tmp = Path(local).parent / f".hf_adapter_cache_{name}" if local else Path(f"adapters/.hf_adapter_cache_{name}")
    snapshot_download(repo_id=hf_repo, allow_patterns=patterns, local_dir=str(tmp))

    src = tmp / hf_path if hf_path else tmp
    dest = Path(local) if local else src

    if local and src != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(src), str(dest))
        shutil.rmtree(str(tmp), ignore_errors=True)

    print(f"  [{name}] Adapter ready at: {dest}")
    return str(dest)


def parse_args():
    ap = argparse.ArgumentParser(description="Merge domain LoRA adapters into a group model")
    ap.add_argument("--config", type=str, default=None,
                    help="Path to experiment config JSON (e.g. configs/lora_direct_v1.json). "
                         "Downloads adapters from HF automatically if not local.")
    ap.add_argument("--safety_adapter", default=None, help="Path to safety LoRA adapter dir (overrides config)")
    ap.add_argument("--gk_adapter",     default=None, help="Path to GK LoRA adapter dir (overrides config)")
    ap.add_argument("--ml_adapter",     default=None, help="Path to multilingual LoRA adapter dir (overrides config)")
    ap.add_argument("--math_adapter",   default=None, help="Path to math LoRA adapter dir (overrides config)")
    ap.add_argument("--output_dir",     default=None, help="Output directory for merged model (overrides config)")
    ap.add_argument("--safety_w", type=float, default=None, help="Safety adapter weight (overrides config)")
    ap.add_argument("--gk_w",     type=float, default=None, help="GK adapter weight (overrides config)")
    ap.add_argument("--ml_w",     type=float, default=None, help="Multilingual adapter weight (overrides config)")
    ap.add_argument("--math_w",   type=float, default=None, help="Math adapter weight (overrides config)")
    return ap.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config) if args.config else {}
    merge_cfg = cfg.get("merge", {})
    adapter_cfgs = merge_cfg.get("adapters", {})

    # Resolve adapter paths (CLI overrides config, config downloads from HF if needed)
    def get_adapter_path(cli_val, name):
        if cli_val:
            return cli_val
        if name in adapter_cfgs:
            return resolve_adapter(name, adapter_cfgs[name])
        raise ValueError(f"No path for adapter '{name}': provide --{name}_adapter or --config with merge.adapters.{name}")

    safety_path = get_adapter_path(args.safety_adapter, "safety")
    gk_path     = get_adapter_path(args.gk_adapter,     "general_knowledge")
    ml_path     = get_adapter_path(args.ml_adapter,     "multilingual")
    math_path   = get_adapter_path(args.math_adapter,   "math")

    # Resolve weights (CLI overrides config, default 0.25)
    safety_w = args.safety_w if args.safety_w is not None else adapter_cfgs.get("safety", {}).get("weight", 0.25)
    gk_w     = args.gk_w     if args.gk_w     is not None else adapter_cfgs.get("general_knowledge", {}).get("weight", 0.25)
    ml_w     = args.ml_w     if args.ml_w     is not None else adapter_cfgs.get("multilingual", {}).get("weight", 0.25)
    math_w   = args.math_w   if args.math_w   is not None else adapter_cfgs.get("math", {}).get("weight", 0.25)

    output_dir = args.output_dir or merge_cfg.get("output_dir", "models/lora_direct_v1_reproduced")
    base_model = merge_cfg.get("base_model", BASE_MODEL)

    adapters = {
        "safety":            (safety_path, safety_w),
        "general_knowledge": (gk_path,     gk_w),
        "multilingual":      (ml_path,     ml_w),
        "math":              (math_path,   math_w),
    }

    print("=" * 80)
    print("Direct LoRA merge  (manual weighted delta accumulation)")
    print(f"  base             : {base_model}")
    for name, (path, w) in adapters.items():
        print(f"  {name:20s}: weight={w:.2f}  path={path}")
    print(f"  output_dir       : {output_dir}")
    print("=" * 80)

    for name, (path, _) in adapters.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} adapter not found: {path}")

    print("\nLoading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    adapter_names = list(adapters.keys())
    first_name = adapter_names[0]
    first_path, _ = adapters[first_name]

    print(f"\nLoading adapter: {first_name}  ({first_path})")
    model = PeftModel.from_pretrained(base, first_path, adapter_name=first_name)

    for name in adapter_names[1:]:
        path, _ = adapters[name]
        print(f"Loading adapter: {name}  ({path})")
        model.load_adapter(path, adapter_name=name)

    adapter_weights = {n: adapters[n][1] for n in adapter_names}

    # Manually accumulate weighted deltas across adapters.
    # PEFT's built-in linear combination requires identical ranks; our adapters
    # have r=16/32/64/96, so we materialize each delta (alpha/r * B @ A) and
    # sum them ourselves — rank differences don't matter at this level.
    print("\nAccumulating weighted deltas...")
    for module_name, module in model.named_modules():
        if not hasattr(module, "get_delta_weight"):
            continue
        delta_sum = None
        for adapter_name in adapter_names:
            if adapter_name not in getattr(module, "lora_A", {}):
                continue
            delta = module.get_delta_weight(adapter_name).to(torch.float32)
            w = adapter_weights[adapter_name]
            delta_sum = w * delta if delta_sum is None else delta_sum + w * delta
        if delta_sum is not None:
            base_weight = module.base_layer.weight.data.to(torch.float32)
            module.base_layer.weight.data = (base_weight + delta_sum).to(torch.bfloat16)

    print("Unloading LoRA layers...")
    # Zero out lora_A so merge_and_unload adds zero deltas — our weighted
    # deltas are already baked into base_layer.weight above.
    for module in model.modules():
        if hasattr(module, "lora_A"):
            for adapter_name in adapter_names:
                if adapter_name in module.lora_A:
                    module.lora_A[adapter_name].weight.data.zero_()
    merged = model.merge_and_unload()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        first_path,
        trust_remote_code=True,
        use_fast=True,
    )

    template_path = Path(__file__).resolve().parents[1] / "configs" / "chat_template.jinja"
    if template_path.exists():
        tokenizer.chat_template = template_path.read_text(encoding="utf-8")
        print(f"  Applied chat template from {template_path}")
    else:
        tokenizer = inject_system_prompt(tokenizer)
        print("  Injected system prompt into base Qwen3 chat_template")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving merged model to {out_dir} ...")
    merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(str(out_dir))
    patch_tokenizer_config(out_dir)

    GenerationConfig(
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.03,
        max_new_tokens=16384,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
    ).save_pretrained(str(out_dir))

    print("\n" + "=" * 80)
    print(f"Done. Merged model → {out_dir}")
    print(f"\nEvaluate:")
    if args.config:
        print(f"  python scripts/eval_group.py --config {args.config} --model_dir {out_dir}")
    else:
        print(f"  python scripts/eval_group.py --model_dir {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
