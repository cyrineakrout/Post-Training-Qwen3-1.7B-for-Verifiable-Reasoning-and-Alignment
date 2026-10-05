#!/usr/bin/env python3
"""
Convert nvidia/Llama-Nemotron-Post-Training-Dataset SFT/science split
into the project's SFT training format.

The science split has 708k MCQ examples with real DeepSeek-R1 reasoning
in <think> blocks — the reasoning data we couldn't generate ourselves.

Key filtering steps:
  1. Extract the final answer letter from the output text
  2. Verify the extracted letter is actually a valid choice in the question
     (guards against picking up a stray letter from mid-reasoning)
  3. Rebuild the assistant turn as <think>...</think>\n\n\boxed{X}

Usage:
  python prepare_nemotron_science.py \
      --max_examples 30000 \
      --output general_knowledge/data/nemotron_science_30k.jsonl \
      --audit_n 200        # spot-check this many extractions before saving
"""

import argparse
import json
import re
import random
import sys
from pathlib import Path
from typing import Optional

GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(GENERAL_KNOWLEDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERAL_KNOWLEDGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_sft_data as sft_data
from config import load_config_from_args


# ---------------------------------------------------------------------------
# Answer extraction
#
# Outputs end in various ways:
#   (a) "The answer is (X)."                   — most common
#   (b) "**The answer is (X)**."               — bold variant
#   (c) answer inside <think> block before </think>
#   (d) no clear marker                        — skip these
#
# Key fix: don't anchor with $ — the answer often appears just before
# </think>, not at the very end of the string. We find ALL occurrences
# and take the last one.
#
# We then VERIFY the extracted letter is an actual choice in the question.
# ---------------------------------------------------------------------------

# Matches "the answer is X" or "the answer is (X)" with optional bold.
# No $ anchor — answer may be inside the think block before </think>.
_P_EXPLICIT = re.compile(
    r"\*{0,2}[Tt]he\s+answer\s+is\s+\(?([A-J])\)?\*{0,2}[.\s]*"
)

# Choice line detector: "A: ..." or "A. ..." at line start
_P_CHOICE_LINE = re.compile(r"^([A-J])[:\.]", re.MULTILINE)


def extract_answer(output: str) -> Optional[str]:
    text = output.strip()

    # Find ALL explicit "the answer is X" occurrences, take the last one.
    # Last occurrence is the final conclusion, not a mid-reasoning mention.
    hits = _P_EXPLICIT.findall(text)
    if hits:
        return hits[-1].upper()

    return None


def valid_choices(user_content: str) -> set:
    """Return the set of choice letters present in the question (e.g. {'A','B',...,'J'})."""
    return set(_P_CHOICE_LINE.findall(user_content))


def build_assistant_content(output: str, answer: str) -> str:
    """
    Keep the <think> block from the original output, replace the trailing
    prose with \boxed{X}.

    Cases:
      - output has </think>: keep everything up to and including </think>,
        then append \boxed{X}
      - output starts with <think> but no closing tag: close it, then \boxed{X}
      - output has no think tags at all: wrap in <think>...</think>, then \boxed{X}
    """
    text = output.strip()
    end = text.find("</think>")

    if text.startswith("<think>") and end != -1:
        think_block = text[: end + len("</think>")]
        return f"{think_block}\n\n\\boxed{{{answer}}}"
    elif text.startswith("<think>"):
        return f"{text}\n</think>\n\n\\boxed{{{answer}}}"
    else:
        return f"<think>\n{text}\n</think>\n\n\\boxed{{{answer}}}"


def convert(max_examples: Optional[int], seed: int, audit_n: int):
    from datasets import load_dataset
    from tqdm import tqdm

    rng = random.Random(seed)

    print("Streaming nvidia/Llama-Nemotron-Post-Training-Dataset SFT/science ...")
    ds = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset",
        "SFT",
        split="science",
        streaming=True,
    )

    # Reservoir sample for a random subset without loading 708k into RAM.
    # Oversample by 2x to account for ~35% skip rate during conversion.
    reservoir = []
    cap = (max_examples * 2) if max_examples and max_examples > 0 else None

    for i, ex in enumerate(tqdm(ds, desc="reservoir sampling")):
        if cap is None:
            reservoir.append(ex)
        elif len(reservoir) < cap:
            reservoir.append(ex)
        else:
            j = rng.randint(0, i)
            if j < cap:
                reservoir[j] = ex

    rng.shuffle(reservoir)
    print(f"Reservoir size: {len(reservoir)}")

    rows = []
    stats = {
        "no_input": 0,
        "no_answer_extracted": 0,
        "answer_not_in_choices": 0,  # key filter: extracted letter isn't a real choice
        "kept": 0,
        "by_pattern": {"explicit": 0},
    }

    for ex in tqdm(reservoir, desc="converting"):
        input_msgs = ex.get("input", [])
        user_content = next(
            (
                m.get("content", "").strip()
                for m in reversed(input_msgs)
                if m.get("role") == "user"
            ),
            None,
        )
        if not user_content:
            stats["no_input"] += 1
            continue

        output = ex.get("output", "").strip()

        # Track which pattern matched
        answer = extract_answer(output)
        if answer:
            stats["by_pattern"]["explicit"] += 1

        if not answer:
            stats["no_answer_extracted"] += 1
            continue

        # Validate: the extracted letter must be one of the actual choices
        choices = valid_choices(user_content)
        if answer not in choices:
            stats["answer_not_in_choices"] += 1
            continue

        num_choices = len(choices)
        assistant_content = build_assistant_content(output, answer)

        rows.append({
            "messages": [
                {"role": "system", "content": sft_data.SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "source": "nemotron_science",
                "num_choices": num_choices,
                "answer": answer,
                "has_explanation": True,
            },
        })
        stats["kept"] += 1
        if max_examples and len(rows) >= max_examples:
            break

    # ------------------------------------------------------------------
    # Audit: print a random sample so you can eyeball quality
    # ------------------------------------------------------------------
    if audit_n > 0 and rows:
        print(f"\n{'='*70}")
        print(f"AUDIT: {min(audit_n, len(rows))} random examples")
        print(f"{'='*70}")
        sample = rng.sample(rows, min(audit_n, len(rows)))
        for i, r in enumerate(sample):
            asst = r["messages"][2]["content"]
            think_end = asst.find("</think>")
            think_tail = asst[max(0, think_end - 120): think_end] if think_end != -1 else asst[-120:]
            after_think = asst[think_end + len("</think>"):].strip() if think_end != -1 else ""
            print(f"\n[{i}] answer={r['metadata']['answer']}  choices={r['metadata']['num_choices']}")
            print(f"  user: {r['messages'][1]['content'][:100]!r}")
            print(f"  think tail: {think_tail!r}")
            print(f"  after think: {after_think!r}")

    print(f"\n{'='*70}")
    print(f"Stats:")
    print(f"  Kept:                    {stats['kept']}")
    print(f"  Skipped no input:        {stats['no_input']}")
    print(f"  Skipped no answer:       {stats['no_answer_extracted']}")
    print(f"  Skipped bad extraction:  {stats['answer_not_in_choices']}  <- letter not in choices")
    print(f"  Matched via 'the answer is': {stats['by_pattern']['explicit']}")
    print(f"{'='*70}")

    return rows


def main():
    config_path, config = load_config_from_args()
    sft_data.SYSTEM_PROMPT = config["prompts"]["sft_system_prompt"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=config_path)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=30000,
        help="Number of examples to sample from the 708k pool.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="general_knowledge/data/nemotron_science_30k.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--audit_n",
        type=int,
        default=10,
        help="Number of random examples to print for manual inspection (0 to skip).",
    )
    args = parser.parse_args()

    rows = convert(
        max_examples=args.max_examples,
        seed=args.seed,
        audit_n=args.audit_n,
    )

    if not rows:
        print("No rows produced — check the stats above.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(rows)} examples to {out}")


if __name__ == "__main__":
    main()
