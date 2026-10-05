#!/usr/bin/env python3
"""
Generate CoT reasoning for hard MCQ examples (SuperGPQA + GPQA) using a
Qwen3 teacher model via vLLM. Only keeps examples where the generated answer
matches the reference. Output format matches mmlupro_cot_train_labeled_correct.jsonl.

Usage:
  python generate_cot_qwen.py \
      --teacher_model Qwen/Qwen3-7B \
      --input_file general_knowledge/data/sft_train_v6_mmluaux50k_nempool50k_gpqaextall_supergpqaall.jsonl \
      --output_file general_knowledge/data/supergpqa_gpqa_cot_qwen3_7b.jsonl

  # Resume an interrupted run:
  python generate_cot_qwen.py ... --resume
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = GENERAL_KNOWLEDGE_ROOT / "configs"

for _p in [str(CONFIGS_DIR), str(GENERAL_KNOWLEDGE_ROOT), str(SCRIPT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import load_config_from_args

DEFAULT_SOURCES = ("gpqa", "supergpqa", "mmlu")

COT_SYSTEM_PROMPT = (
    "You are a careful multiple-choice general-knowledge assistant. "
    "Always show your reasoning step by step inside the <think> block before giving the final answer. "
    "Keep your reasoning concise and focused. "
    "Then finish with exactly one capital letter inside \\boxed{}. "
    "Do not answer without explicit reasoning."
)

_BOXED_RE = re.compile(r"\\boxed\{([A-T])\}", re.IGNORECASE)
_FINAL_ANSWER_RE = re.compile(
    r"Final answer[:\s]+(?:\\boxed\{)?([A-T])\}?", re.IGNORECASE
)


def extract_teacher_answer(text: str) -> Optional[str]:
    """Extract final answer from after </think> to avoid stray letters in reasoning."""
    if not text:
        return None

    # Strip <think>...</think>, look for answer in remainder
    after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    m = _BOXED_RE.search(after_think)
    if m:
        return m.group(1).upper()

    m = _FINAL_ANSWER_RE.search(after_think)
    if m:
        return m.group(1).upper()

    m = re.search(
        r"(?:the\s+(?:correct\s+)?answer\s+is\s*[:\-]?\s*|answer\s*[:\-]\s*)([A-T])\b",
        after_think, re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()

    # Fallback: last \boxed{} anywhere in full output
    all_matches = _BOXED_RE.findall(text)
    if all_matches:
        return all_matches[-1].upper()

    return None


def load_examples(input_file: str, sources: tuple) -> List[Dict[str, Any]]:
    examples = []
    from collections import Counter
    counts = Counter()
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = row.get("metadata", {}).get("source", "")
            prefix = next((s for s in sources if src.startswith(s)), None)
            if prefix:
                examples.append(row)
                counts[prefix] += 1
    print(f"Loaded {len(examples)} examples from sources: {dict(counts)}")
    return examples


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    config_path, config = load_config_from_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument(
        "--input_file",
        type=str,
        default="general_knowledge/data/sft_train_v6_mmluaux50k_nempool50k_gpqaextall_supergpqaall.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="general_knowledge/data/supergpqa_gpqa_cot_qwen3_7b.jsonl",
    )
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--sources",
        type=str,
        default=",".join(DEFAULT_SOURCES),
        help="Comma-separated source prefixes to include. Default: gpqa,supergpqa,mmlu",
    )
    parser.add_argument("--system_prompt", type=str, default=COT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Cap on input examples to process. Default: all.",
    )
    parser.add_argument(
        "--max_kept", type=int, default=None,
        help="Stop once this many correct CoT examples are collected. Default: no limit.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["auto", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-written examples in output file.",
    )
    args = parser.parse_args()

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    examples = load_examples(args.input_file, sources)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
        print(f"Capped to {len(examples)} examples")

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(".ckpt")

    already_processed = 0
    already_kept = 0
    if args.resume and ckpt_path.exists():
        with open(ckpt_path) as _f:
            ckpt = json.load(_f)
        already_processed = ckpt.get("processed", 0)
        already_kept = ckpt.get("kept", 0)
        examples = examples[already_processed:]
        print(f"Resuming: skipping first {already_processed} processed ({already_kept} kept)")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"Loading teacher model: {args.teacher_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    llm = LLM(
        model=args.teacher_model,
        tokenizer=args.teacher_model,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    kept = already_kept
    skipped_wrong = 0
    skipped_no_answer = 0
    skipped_too_long = 0
    total_processed = already_processed
    total_in = already_processed + len(examples)
    max_input_len = args.max_model_len - args.max_new_tokens

    write_mode = "a" if already_processed > 0 else "w"

    with open(out_path, write_mode, encoding="utf-8") as f_out:
        for batch_start in range(0, len(examples), args.batch_size):
            if args.max_kept is not None and kept >= args.max_kept:
                print(f"  Reached max_kept={args.max_kept}, stopping early.")
                break
            batch = examples[batch_start : batch_start + args.batch_size]

            prompts = []
            valid_rows = []
            for row in batch:
                user_content = next(
                    m["content"] for m in row["messages"] if m["role"] == "user"
                )
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_content},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                token_len = len(tokenizer.encode(prompt))
                if token_len > max_input_len:
                    skipped_too_long += 1
                    total_processed += 1
                    continue
                prompts.append(prompt)
                valid_rows.append(row)

            if not prompts:
                continue

            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for row, output in zip(valid_rows, outputs):
                total_processed += 1
                generated = output.outputs[0].text.strip()
                reference = row["metadata"].get("answer", "").strip().upper()

                extracted = extract_teacher_answer(generated)

                if extracted is None:
                    skipped_no_answer += 1
                    continue

                if "</think>" not in generated or r"\boxed{" not in generated:
                    skipped_no_answer += 1
                    continue

                if extracted != reference:
                    skipped_wrong += 1
                    continue

                user_content = next(
                    m["content"] for m in row["messages"] if m["role"] == "user"
                )

                out_row = {
                    "messages": [
                        {"role": "system", "content": args.system_prompt},
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": generated},
                    ],
                    "metadata": {
                        **row.get("metadata", {}),
                        "has_cot": True,
                        "cot_source": f"qwen3_{args.teacher_model.split('/')[-1]}",
                    },
                }
                f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                f_out.flush()
                kept += 1

            pct = 100 * total_processed / total_in if total_in else 0
            print(
                f"  [{total_processed:,}/{total_in:,} = {pct:.1f}%]  "
                f"kept={kept:,}  wrong={skipped_wrong:,}  "
                f"no_answer={skipped_no_answer:,}  too_long={skipped_too_long:,}",
                flush=True,
            )
            with open(ckpt_path, "w") as _f:
                json.dump({"processed": total_processed, "kept": kept}, _f)

    print(f"\nDone.")
    print(f"  Kept (correct CoT):     {kept:,} ({100*kept/max(total_in,1):.1f}%)")
    print(f"  Skipped (wrong answer): {skipped_wrong:,}")
    print(f"  Skipped (no answer):    {skipped_no_answer:,}")
    print(f"  Skipped (too long):     {skipped_too_long:,}")
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
