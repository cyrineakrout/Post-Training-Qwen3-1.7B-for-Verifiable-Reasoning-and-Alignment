"""
Build v14 CoT distillation data using Claude API as teacher.

MUCH faster than local generation (1-2 hours for 31k vs 10+ hours on GPU).
Better reasoning quality than Qwen3-8B.

Cost estimate (claude-3-5-haiku-20241022):
  Input:  31k × ~230 tokens × $0.80/1M  ≈  $5.70
  Output: 31k × ~300 tokens × $4.00/1M  ≈ $37.20
  Total:  ~$43 for full dataset

With $20 credits: covers ~13-14k new examples.
Use --resume to continue after adding more credits.

Usage:
  export ANTHROPIC_API_KEY="sk-ant-..."

  # First time (or resume after GPU run):
  python safety/scripts/build_v14_cot_claude.py \\
      --output_train safety/data/combined_v14_thinking_train.jsonl \\
      --output_dev  safety/data/combined_v14_thinking_dev.jsonl \\
      --resume

  # Test with 50 examples first:
  python safety/scripts/build_v14_cot_claude.py \\
      --output_train safety/data/combined_v14_thinking_train.jsonl \\
      --output_dev  safety/data/combined_v14_thinking_dev.jsonl \\
      --limit 50 --resume
"""

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import anthropic

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Think carefully step by step, then answer with one boxed letter, like \\boxed{A}."
)

RESPONSE_FORMAT_INSTRUCTION = (
    "\n\nPlease format your response exactly as:\n"
    "<think>\n[your step-by-step reasoning]\n</think>\n\\boxed{X}"
)

# ── Answer extraction ─────────────────────────────────────────────────────────
_BOXED_RE       = re.compile(r"\\boxed\{([A-T])\}", re.IGNORECASE)
_FINAL_ANS_RE   = re.compile(r"(?:final answer|answer)[:\s]*(?:\\boxed\{)?([A-T])\}?", re.IGNORECASE)
_BOLD_RE        = re.compile(r"\*\*(?:Option\s+)?([A-T])\*\*", re.IGNORECASE)
_OPTION_RE      = re.compile(r"\bOption\s+([A-T])\b", re.IGNORECASE)
_THE_ANSWER_RE  = re.compile(r"the\s+(?:correct\s+)?answer\s+is\s*[:\-]?\s*([A-T])\b", re.IGNORECASE)


def extract_gold(content: str) -> str | None:
    if not content:
        return None
    m = _BOXED_RE.search(content)
    if m:
        return m.group(1).upper()
    m = _FINAL_ANS_RE.search(content)
    if m:
        return m.group(1).upper()
    return None


def extract_pred(output: str) -> str | None:
    if not output:
        return None
    after = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()

    for pattern in [_BOXED_RE, _FINAL_ANS_RE, _THE_ANSWER_RE, _BOLD_RE, _OPTION_RE]:
        m = pattern.search(after)
        if m:
            return m.group(1).upper()

    # Last \boxed anywhere
    all_boxed = _BOXED_RE.findall(output)
    if all_boxed:
        return all_boxed[-1].upper()

    # Isolated letter at end
    m = re.search(r"\b([A-T])\s*$", after.strip())
    if m:
        return m.group(1).upper()

    return None


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def count_lines(path: str) -> int:
    if not Path(path).exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


async def call_claude(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    user_content: str,
    model: str,
    max_tokens: int,
    retries: int = 3,
) -> str | None:
    """Call Claude API with retry on rate limit errors."""
    async with semaphore:
        for attempt in range(retries):
            try:
                msg = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": user_content + RESPONSE_FORMAT_INSTRUCTION,
                        }
                    ],
                )
                return msg.content[0].text if msg.content else None
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5  # 5s, 10s, 20s
                print(f"  [rate limit] waiting {wait}s...", flush=True)
                await asyncio.sleep(wait)
            except Exception as e:
                print(f"  [error] {type(e).__name__}: {e}", flush=True)
                if attempt < retries - 1:
                    await asyncio.sleep(2)
        return None


async def process_split(
    client: anthropic.AsyncAnthropic,
    input_path: str,
    output_path: str,
    model: str,
    max_tokens: int,
    concurrency: int,
    limit: int | None,
    resume: bool,
):
    examples = load_jsonl(input_path, limit=limit)
    total_in = len(examples)
    print(f"  Loaded {total_in:,} examples from {input_path}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    already_done = 0
    if resume and Path(output_path).exists():
        already_done = count_lines(output_path)
        examples = examples[already_done:]
        print(f"  Resuming: skipping first {already_done:,} already done")

    kept            = already_done
    skipped_gold    = 0
    skipped_wrong   = 0
    skipped_parse   = 0
    total_processed = already_done
    start_time      = time.time()

    semaphore = asyncio.Semaphore(concurrency)
    write_mode = "a" if already_done > 0 else "w"

    with open(output_path, write_mode, encoding="utf-8") as fout:
        # Process in chunks to show progress and write incrementally
        chunk_size = concurrency * 4  # process 4 waves at a time

        for chunk_start in range(0, len(examples), chunk_size):
            chunk = examples[chunk_start : chunk_start + chunk_size]

            # Extract gold answers and build inputs
            tasks    = []
            golds    = []
            orig_exs = []
            valid    = []

            for ex in chunk:
                asst = next(
                    (m["content"] for m in ex["messages"] if m["role"] == "assistant"),
                    None,
                )
                gold = extract_gold(asst) if asst else None
                user_msg = next(
                    (m["content"] for m in ex["messages"] if m["role"] == "user"),
                    None,
                )

                if gold is None or user_msg is None:
                    skipped_gold += 1
                    valid.append(False)
                    golds.append(None)
                    orig_exs.append(ex)
                    tasks.append(None)
                    total_processed += 1
                    continue

                valid.append(True)
                golds.append(gold)
                orig_exs.append(ex)
                tasks.append(
                    call_claude(client, semaphore, user_msg, model, max_tokens)
                )
                total_processed += 1

            # Run all valid tasks concurrently
            valid_tasks = [t for t, v in zip(tasks, valid) if v]
            if valid_tasks:
                results = await asyncio.gather(*valid_tasks)
            else:
                results = []

            result_idx = 0
            for i, (ex, is_valid) in enumerate(zip(orig_exs, valid)):
                if not is_valid:
                    continue

                output = results[result_idx]
                result_idx += 1

                if output is None:
                    skipped_parse += 1
                    continue

                pred = extract_pred(output)
                if pred is None:
                    skipped_parse += 1
                    continue

                if pred != golds[i]:
                    skipped_wrong += 1
                    continue

                user_msg = next(m["content"] for m in ex["messages"] if m["role"] == "user")
                new_ex = {
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": user_msg},
                        {"role": "assistant", "content": output},
                    ]
                }
                fout.write(json.dumps(new_ex, ensure_ascii=False) + "\n")
                fout.flush()
                kept += 1

            # Progress
            elapsed  = time.time() - start_time
            rate     = max(total_processed - already_done, 1) / elapsed * 3600
            pct      = 100 * total_processed / total_in if total_in else 0
            new_kept = kept - already_done
            print(
                f"  [{total_processed:,}/{total_in:,} = {pct:.1f}%]  "
                f"kept={kept:,}  wrong={skipped_wrong:,}  "
                f"no_parse={skipped_parse:,}  "
                f"speed={rate:.0f} ex/hr",
                flush=True,
            )

    print(f"\n  ── Split summary ──────────────────────────────────")
    print(f"  Input examples  : {total_in:,}")
    print(f"  Kept (output)   : {kept:,}  ({100*kept/max(total_in,1):.1f}%)")
    print(f"  Skipped no gold : {skipped_gold:,}")
    print(f"  Skipped wrong   : {skipped_wrong:,}")
    print(f"  Skipped no parse: {skipped_parse:,}")
    print(f"  Output path     : {output_path}")
    return kept


async def main_async(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY") or args.api_key
    if not api_key:
        raise ValueError(
            "Set ANTHROPIC_API_KEY environment variable or pass --api_key"
        )

    client = anthropic.AsyncAnthropic(api_key=api_key)

    print("=" * 80)
    print(f"Claude teacher: {args.model}")
    print(f"Concurrency   : {args.concurrency}")
    print(f"Max tokens    : {args.max_tokens}")
    print(f"Resume        : {args.resume}")
    print(f"Limit         : {args.limit or 'all'}")

    print("=" * 80)
    print("Processing TRAIN split...")
    n_train = await process_split(
        client,
        args.input_train, args.output_train,
        args.model, args.max_tokens,
        args.concurrency, args.limit, args.resume,
    )

    if not args.skip_dev:
        print("=" * 80)
        print("Processing DEV split...")
        n_dev = await process_split(
            client,
            args.input_dev, args.output_dev,
            args.model, args.max_tokens,
            args.concurrency, args.limit, args.resume,
        )
    else:
        n_dev = 0

    print("=" * 80)
    print(f"DONE.  Train: {n_train:,}   Dev: {n_dev:,}")
    print()
    print("Next step — train the student:")
    print("  python safety/scripts/train_safety_lora.py \\")
    print("      --config safety/configs/safety_lora_combined_v14_thinking.yaml \\")
    print("      --thinking")


def main():
    parser = argparse.ArgumentParser(
        description="Generate v14 CoT data using Claude API (faster + better than local 8B)."
    )
    parser.add_argument("--input_train",  default="safety/data/combined_v10_nosb_train.jsonl")
    parser.add_argument("--input_dev",    default="safety/data/combined_v10_nosb_dev.jsonl")
    parser.add_argument("--output_train", default="safety/data/combined_v14_thinking_train.jsonl")
    parser.add_argument("--output_dev",   default="safety/data/combined_v14_thinking_dev.jsonl")
    parser.add_argument(
        "--model",
        default="claude-3-5-haiku-20241022",
        help="Claude model. Options: claude-3-5-haiku-20241022 (cheap), claude-3-5-sonnet-20241022 (better)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="Concurrent API calls. Haiku allows up to 50/min, keep at 20 to be safe.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Max output tokens per example.",
    )
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--resume",   action="store_true")
    parser.add_argument("--skip_dev", action="store_true")
    parser.add_argument("--api_key",  type=str, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
