"""
TIES merge of 4 specialist models into a single group model.

Usage (from project root on RCP):

    # Basic — uses paths in group/configs/merge_ties.yaml
    python group/scripts/merge_ties_group.py

    # Override individual model paths if your layout differs
    python group/scripts/merge_ties_group.py \
        --safety_model   safety/models/qwen3_safety_merged_combined_v5_all \
        --gk_model       general_knowledge/merged_models/gk_merged \
        --ml_model       multilinguality/merged_model \
        --math_model     math/merged_model \
        --output_dir     group/outputs/group_model_merged

What this script does:
    1. Validates that all 4 model directories exist.
    2. Patches the merge_ties.yaml with the actual paths.
    3. Runs mergekit-yaml to perform TIES merging.
    4. Injects the group system prompt into the tokenizer chat_template
       (same approach as merge_lora_group.py) so CI gets the right prompt.
    5. Saves a generation_config.json with repetition_penalty to prevent loops.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from transformers import AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cluster_paths import shared
from compat import apply_tokenizer_patch, get_torch_dtype
apply_tokenizer_patch()


# ---------------------------------------------------------------------------
# Group system prompt — must match GROUP_SYSTEM_PROMPT in prepare_group_sft_final.py
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "For every question, put the final answer inside exactly one LaTeX \\boxed{} expression. "
    "If the question is multiple-choice, put only the option letter inside the box, for example \\boxed{C}. "
    "If the question is free-form or mathematical, put only the final expression, number, set, or requested object inside the box. "
    "Use brief reasoning only when useful. "
    "Do not write anything after the boxed answer."
)

# Path to the proven custom chat_template from runB — if present, it replaces
# the base Qwen3 template entirely (bakes in system prompt + few-shot examples
# with <think> for all domains). Preferred over SYSTEM_PROMPT injection above.
CUSTOM_CHAT_TEMPLATE = Path("group/outputs/group_model_merged_runB_r64_e060/chat_template.jinja")

_SYS_BEGIN = "{# GROUP_DEFAULT_SYSTEM_BEGIN #}"
_SYS_END   = "{# GROUP_DEFAULT_SYSTEM_END #}"

BASE_CONFIG = Path("group/configs/merge_ties.yaml")
DEFAULT_OUT = str(shared("group/outputs/group_model_merged"))

# HF fallback: if a local model path from the yaml doesn't exist, download from here.
HF_SPECIALIST_REPO = "cs-552-2026-camykaz/merge_ties"
HF_SPECIALIST_SUBFOLDERS = {
    "safety":            "safety",
    "general_knowledge": "general_knowledge",
    "multilingual":      "multilingual",
    "math":              "math",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_specialist_model(path: str, domain: str) -> str:
    """Return path, downloading from HF if not present locally."""
    p = Path(path)
    if p.exists() and (p / "config.json").exists():
        return path

    hf_sub = HF_SPECIALIST_SUBFOLDERS.get(domain)
    if not hf_sub:
        return path  # let check_model_dir report the error

    print(f"  '{path}' not found locally — downloading {domain} from {HF_SPECIALIST_REPO}/{hf_sub} ...")
    from huggingface_hub import snapshot_download
    p.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_SPECIALIST_REPO,
        allow_patterns=[f"{hf_sub}/*"],
        local_dir=str(p),
    )
    # snapshot_download puts files under local_dir/hf_sub/; move them up one level.
    downloaded = p / hf_sub
    if downloaded.exists():
        import shutil
        for item in downloaded.iterdir():
            target = p / item.name
            if not target.exists():
                shutil.move(str(item), str(target))
        downloaded.rmdir()
    print(f"  ✓ {domain} downloaded to {p}")
    return str(p)


def check_model_dir(path: str, name: str) -> Path:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {name} model not found: {p}")
        print("        Update the path with --{}_model or edit merge_ties.yaml".format(
            name.lower().replace(" ", "_")
        ))
        sys.exit(1)
    if not (p / "config.json").exists():
        print(f"[ERROR] {p} exists but has no config.json — is it a full merged model?")
        sys.exit(1)
    print(f"  ✓ {name:15s} {p}")
    return p


def patch_yaml(base_yaml: Path, paths: dict, output_path: str) -> str:
    """
    Load the base YAML, replace model paths with the actual ones,
    write to a temp file, return the temp path.
    """
    with open(base_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    labels = ["safety", "general_knowledge", "multilingual", "math"]
    models_cfg = cfg.get("models", [])

    for i, label in enumerate(labels):
        if i < len(models_cfg):
            models_cfg[i]["model"] = str(paths[label])

    cfg["models"] = models_cfg

    # Write to temp file so we don't mutate the committed config.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="ties_patched_"
    )
    yaml.dump(cfg, tmp, allow_unicode=True, default_flow_style=False)
    tmp.close()
    return tmp.name


def force_group_system_prompt(tokenizer):
    """Inject group system prompt into chat_template (idempotent)."""
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return tokenizer

    template = tokenizer.chat_template

    # Strip previous injection if present.
    if _SYS_BEGIN in template and _SYS_END in template:
        before = template.split(_SYS_BEGIN, 1)[0]
        after  = template.split(_SYS_END,   1)[1].lstrip("\n")
        template = before + after

    escaped = (
        SYSTEM_PROMPT
        .replace("\\", "\\\\")
        .replace("'",  "\\'")
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


def patch_tokenizer_config(merged_dir: Path):
    """Make sure tokenizer_config.json reflects the patched chat_template."""
    config_path = merged_dir / "tokenizer_config.json"
    if not config_path.exists():
        return
    with open(config_path, "r") as f:
        cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    cfg["chat_template"] = tokenizer.chat_template
    with open(config_path, "w") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("  ✓ tokenizer_config.json patched with group system prompt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TIES merge → group model")
    p.add_argument("--safety_model", default=None)
    p.add_argument("--gk_model",     default=None)
    p.add_argument("--ml_model",     default=None)
    p.add_argument("--math_model",   default=None)
    p.add_argument("--output_dir",   default=DEFAULT_OUT)
    p.add_argument("--config",       default=str(BASE_CONFIG))
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load base config to get default paths
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    base_paths = [m["model"] for m in base_cfg.get("models", [])]

    safety_path = args.safety_model or (base_paths[0] if len(base_paths) > 0 else None)
    gk_path     = args.gk_model     or (base_paths[1] if len(base_paths) > 1 else None)
    ml_path     = args.ml_model     or (base_paths[2] if len(base_paths) > 2 else None)
    math_path   = args.math_model   or (base_paths[3] if len(base_paths) > 3 else None)

    print("\n" + "=" * 70)
    print("STEP 1 — Resolving model paths (download from HF if needed)")
    print("=" * 70)

    safety_path = resolve_specialist_model(safety_path, "safety")
    gk_path     = resolve_specialist_model(gk_path,     "general_knowledge")
    ml_path     = resolve_specialist_model(ml_path,     "multilingual")
    math_path   = resolve_specialist_model(math_path,   "math")

    paths = {
        "safety":            check_model_dir(safety_path, "Safety"),
        "general_knowledge": check_model_dir(gk_path,     "General KN"),
        "multilingual":      check_model_dir(ml_path,     "Multilingual"),
        "math":              check_model_dir(math_path,   "Math"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Patch YAML with real paths
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 — Preparing merge config")
    print("=" * 70)

    patched_yaml = patch_yaml(Path(args.config), paths, args.output_dir)
    print(f"  Patched config → {patched_yaml}")

    # ------------------------------------------------------------------
    # 3. Run mergekit
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3 — Running TIES merge (mergekit)")
    print("=" * 70)

    # mergekit 0.1.4 + pydantic 2.x: ConfiguredModuleArchitecture and
    # ConfiguredModelArchitecture have unresolved torch.dtype forward refs that
    # raise at plan time.  Importing torch first and calling model_rebuild()
    # on both classes before mergekit runs is the fix.
    prelude = (
        "import sys, torch; "
        "from mergekit.architecture import ConfiguredModuleArchitecture; "
        "from mergekit.plan import ConfiguredModelArchitecture; "
        "ConfiguredModuleArchitecture.model_rebuild(); "
        "ConfiguredModelArchitecture.model_rebuild(); "
        "from mergekit.scripts.run_yaml import main; "
        "sys.exit(main())"
    )
    cmd = [
        sys.executable, "-c", prelude,
        patched_yaml,
        str(output_dir),
        "--cuda",
        "--copy-tokenizer",
        "--trust-remote-code",
        "--lazy-unpickle",
    ]

    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print("\n[ERROR] mergekit failed. Check logs above.")
        sys.exit(result.returncode)

    print("\n  ✓ TIES merge complete")

    # ------------------------------------------------------------------
    # 4. Inject group system prompt / custom chat template into tokenizer
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4 — Injecting group chat template")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(str(output_dir), trust_remote_code=True)

    if CUSTOM_CHAT_TEMPLATE.exists():
        # Prefer the proven full template (bakes in system prompt + few-shot
        # <think> examples for all domains) over the partial injection below.
        tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE.read_text(encoding="utf-8")
        print(f"  ✓ Applied custom chat_template from {CUSTOM_CHAT_TEMPLATE}")
    else:
        tokenizer = force_group_system_prompt(tokenizer)
        print("  ✓ Injected system prompt into base Qwen3 chat_template")

    tokenizer.save_pretrained(str(output_dir))
    patch_tokenizer_config(output_dir)

    # ------------------------------------------------------------------
    # 5. Save generation config with repetition penalty (prevents loops)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5 — Saving generation config")
    print("=" * 70)

    gen_config = GenerationConfig(
        do_sample=False,             # greedy for MCQ (deterministic, better for safety/GK/multilingual)
        temperature=1.0,
        repetition_penalty=1.05,     # prevents infinite loops in math thinking
        no_repeat_ngram_size=8,
        max_new_tokens=1024,
        bos_token_id=151643,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
    )
    gen_config.save_pretrained(str(output_dir))
    print("  ✓ generation_config.json saved")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Merged group model → {output_dir}")
    print("\nNext steps:")
    print("  1. Evaluate: python group/scripts/evaluate_group.py --model_dir", output_dir)
    print("  2. If score looks good: python group/scripts/push_group_model.py")


if __name__ == "__main__":
    main()
