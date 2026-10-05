"""Shared constants and utilities for multilinguality scripts."""

import re

# ── regex ──────────────────────────────────────────────────────────────────────
BOXED_RE = re.compile(r"\\boxed\{([A-Z])\}")
THINK_RE  = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# ── language maps ──────────────────────────────────────────────────────────────
LANG_NAMES = {
    "IT_IT": "Italian",
    "ES_LA": "Spanish",
    "ZH_CN": "Chinese",
    "HI_IN": "Hindi",
    "RU_RU": "Russian",
}

# langdetect language codes
LANG_LABELS = {
    "IT_IT": "it",
    "ES_LA": "es",
    "ZH_CN": "zh-cn",
    "HI_IN": "hi",
    "RU_RU": "ru",
}

# ── training path constants ────────────────────────────────────────────────────
GRPO_PATHS = ("cold_grpo", "sft_grpo", "distillation_sft_grpo")
SFT_PATHS  = ("distillation", "sft", "distillation_sft", "simple_sft")
ALL_PATHS  = SFT_PATHS + GRPO_PATHS


def with_thinking(path_str: str, thinking_on: bool) -> str:
    """Append _ton or _toff to a merged dir path to distinguish thinking variants."""
    suffix = "_ton" if thinking_on else "_toff"
    if path_str.endswith("_merged"):
        return path_str[:-len("_merged")] + suffix + "_merged"
    return path_str + suffix


def with_lang(path_str: str, language: str) -> str:
    """Append _multi to lora/merged dir names when language='multi'.

    English (default) leaves paths unchanged for backward compatibility.
    """
    if language == "english":
        return path_str
    for suffix in ("_lora", "_merged"):
        if path_str.endswith(suffix):
            return path_str[: -len(suffix)] + "_multi" + suffix
    return path_str + "_multi"


# ── language detection ─────────────────────────────────────────────────────────
def detect_language(text: str, target_label: str, min_conf: float = 0.7) -> bool:
    """Return True if *text* is in *target_label* with at least *min_conf* confidence.

    Extracts the <think> block when present, falls back to the full text.
    Returns True on empty input or langdetect errors (permissive by default).
    """
    from langdetect import detect_langs
    m = THINK_RE.search(text)
    chunk = (m.group(1) if m else text)[:1000].replace("\n", " ").strip()
    if not chunk:
        return True
    try:
        langs = detect_langs(chunk)
        for lang in langs:
            if lang.lang == target_label:
                return lang.prob >= min_conf
        return False
    except Exception:
        return True
