import importlib.util
import os


def resolve_attention_implementation() -> str:
    """Use FlashAttention when available, otherwise fall back to PyTorch SDPA."""
    override = os.environ.get("MATH_ATTN_IMPLEMENTATION")
    if override:
        return override
    if importlib.util.find_spec("flash_attn") is not None:
        return "flash_attention_2"
    return "sdpa"
