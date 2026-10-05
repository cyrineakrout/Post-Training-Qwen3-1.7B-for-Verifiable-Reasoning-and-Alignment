"""
Build SFT data from WildGuardMix (allenai/wildguardmix) for the safety model.

WildGuardMix contains (prompt, response, labels) tuples.
Labels: prompt_harm_label, response_harm_label, response_refusal_label.

Conversion strategy:
  1. PAIRED MCQ (main): group by prompt → for prompts that have both a
     harmful-compliance response AND a safe-refusal response, build a
     2-option MCQ: "Which response is safest?" (refusal = correct answer).
     Also generate a flipped variant: "Which response is most harmful?"
     (compliance = correct answer). This directly trains harm detection.

  2. UNPAIRED MCQ (fallback): for prompts with only one response, pair them
     with a random response from the distractor pool (opposite label).
     For benign + compliant examples: refusal-calibration MCQs (compliance is
     correct — the model should NOT always refuse).

Usage (from project root):
  python safety/scripts/build_wildguardmix_sft.py \\
      --output safety/data/wildguardmix_sft.jsonl \\
      --max_examples 15000

  python safety/scripts/build_wildguardmix_sft.py \\
      --output safety/data/wildguardmix_sft.jsonl \\
      --max_examples 15000 --flip_ratio 0.3
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)

LETTERS = "ABCDEFGHIJKLMNOPQRST"

# ── Question templates ─────────────────────────────────────────────────────────

SAFE_PICK_TEMPLATES = [
    "Which response is the safest and most appropriate?",
    "Which of the following responses best handles this request safely?",
    "Which response would be most responsible?",
    "Select the response that avoids harm and provides safe guidance:",
    "Which reply best protects the user from harm?",
    "Which response should an ethical AI give to this request?",
]

HARMFUL_PICK_TEMPLATES = [
    "Which of the following responses would cause the most harm?",
    "Which response is the most dangerous or irresponsible?",
    "Which reply would be most harmful to the user or others?",
    "Identify the response that poses the greatest safety risk:",
    "Which response fails to protect the user from harm?",
]

SAFE_REFUSAL_PICK_TEMPLATES = [
    "For this request, which response correctly handles potential safety concerns?",
    "Which response most appropriately addresses this potentially sensitive request?",
]

# Generic safe refusals used as distractors for benign calibration examples
GENERIC_REFUSALS = [
    (
        "I can't help with harmful, illegal, unsafe, or deceptive actions. "
        "Instead, I can suggest safe, legal, and responsible alternatives."
    ),
    (
        "I can't assist with that request because it could cause harm or violate laws. "
        "A safer option is to seek legal, ethical, and responsible guidance."
    ),
    (
        "I can't provide instructions for harmful or unsafe behavior. "
        "I can help with prevention, safety planning, or lawful alternatives instead."
    ),
    (
        "I'm sorry, but I can't help with that. It may involve harmful, unethical, "
        "or illegal actions. Please consider safer and more constructive alternatives."
    ),
    (
        "That request asks for something potentially dangerous or unethical. "
        "I'll redirect to safe, responsible, and constructive options instead."
    ),
]


def build_mcq_prompt(question: str, options: list[str]) -> str:
    lines = [question, ""]
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}) {opt}")
    return "\n".join(lines)


def build_sft_row(
    prompt_text: str,
    options: list[str],
    correct_idx: int,
    source: str,
    category: str = "",
    subcategory: str = "",
    task_type: str = "wildguardmix_mcq",
) -> dict:
    question = build_mcq_prompt(prompt_text, options)
    answer_letter = LETTERS[correct_idx]
    categories = [c for c in [category, subcategory] if c and c.lower() not in ("", "none", "nan")]

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": f"\\boxed{{{answer_letter}}}"},
        ],
        "metadata": {
            "source": "allenai/wildguardmix",
            "task_type": task_type,
            "answer": answer_letter,
            "categories": categories,
        },
    }


def make_paired_examples(
    prompt: str,
    harmful_response: str,
    safe_response: str,
    category: str,
    subcategory: str,
    flip_ratio: float,
    rng: random.Random,
) -> list[dict]:
    """
    Given a prompt and two responses (one harmful, one safe), generate MCQ(s).
    - Main: "which is safer?" → safe_response is correct
    - Flipped (flip_ratio): "which is most harmful?" → harmful_response is correct
    """
    rows = []

    # Randomise option order to avoid position bias
    if rng.random() < 0.5:
        options = [safe_response, harmful_response]
        correct_safe_idx = 0
        correct_harm_idx = 1
    else:
        options = [harmful_response, safe_response]
        correct_safe_idx = 1
        correct_harm_idx = 0

    # --- safe-pick variant ---
    question = rng.choice(SAFE_PICK_TEMPLATES)
    full_prompt = f"{prompt}\n\n{question}"
    rows.append(
        build_sft_row(
            full_prompt, options, correct_safe_idx,
            source="allenai/wildguardmix",
            category=category, subcategory=subcategory,
            task_type="wildguardmix_safe_pick",
        )
    )

    # --- harmful-pick variant (flip) ---
    if rng.random() < flip_ratio:
        question = rng.choice(HARMFUL_PICK_TEMPLATES)
        full_prompt = f"{prompt}\n\n{question}"
        rows.append(
            build_sft_row(
                full_prompt, options, correct_harm_idx,
                source="allenai/wildguardmix",
                category=category, subcategory=subcategory,
                task_type="wildguardmix_harm_pick",
            )
        )

    return rows


def make_unpaired_harmful_example(
    prompt: str,
    harmful_response: str,
    distractor_pool: list[str],
    category: str,
    subcategory: str,
    rng: random.Random,
) -> dict | None:
    """
    For harmful prompts with only a compliance response: pair with a random
    safe refusal from the distractor pool.
    """
    if not distractor_pool:
        return None

    safe_resp = rng.choice(distractor_pool)

    # Shuffle options
    if rng.random() < 0.5:
        options = [safe_resp, harmful_response]
        correct_idx = 0
    else:
        options = [harmful_response, safe_resp]
        correct_idx = 1

    question = rng.choice(SAFE_PICK_TEMPLATES)
    full_prompt = f"{prompt}\n\n{question}"
    return build_sft_row(
        full_prompt, options, correct_idx,
        source="allenai/wildguardmix",
        category=category, subcategory=subcategory,
        task_type="wildguardmix_safe_pick",
    )


def make_benign_calibration_example(
    prompt: str,
    compliant_response: str,
    category: str,
    rng: random.Random,
) -> dict:
    """
    For benign prompts where compliance is correct (refusal is wrong).
    Teaches the model not to refuse safe requests.
    """
    refusal = rng.choice(GENERIC_REFUSALS)

    if rng.random() < 0.5:
        options = [compliant_response, refusal]
        correct_idx = 0
    else:
        options = [refusal, compliant_response]
        correct_idx = 1

    question = rng.choice(SAFE_REFUSAL_PICK_TEMPLATES)
    full_prompt = f"{prompt}\n\n{question}"
    return build_sft_row(
        full_prompt, options, correct_idx,
        source="allenai/wildguardmix",
        category=category, subcategory="benign_calibration",
        task_type="wildguardmix_benign_calibration",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build SFT data from WildGuardMix."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="safety/data/wildguardmix_sft.jsonl",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=15_000,
        help="Max SFT examples to generate. -1 = no limit.",
    )
    parser.add_argument(
        "--flip_ratio",
        type=float,
        default=0.3,
        help="Fraction of paired examples to also generate a harm-pick variant.",
    )
    parser.add_argument(
        "--benign_ratio",
        type=float,
        default=0.15,
        help="Fraction of benign calibration examples to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    print("Loading WildGuardMix from HuggingFace...")
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
    except Exception:
        # Fallback: try without config name
        ds = load_dataset("allenai/wildguardmix", split="train")

    print(f"Loaded {len(ds):,} examples")
    print(f"Columns: {ds.column_names}")
    print(f"Sample: {dict(list(ds[0].items())[:6])}")

    # ── Group by prompt ────────────────────────────────────────────────────────
    # For each unique prompt, collect all (response, labels) pairs
    by_prompt: dict[str, list[dict]] = defaultdict(list)

    for ex in tqdm(ds, desc="Grouping by prompt"):
        prompt = (ex.get("prompt") or ex.get("instruction") or "").strip()
        if not prompt:
            continue

        response = (ex.get("response") or ex.get("output") or "").strip()
        if not response:
            continue

        # Normalise label fields (different versions may use different names)
        prompt_harm   = str(ex.get("prompt_harm_label",      ex.get("prompt_label", ""))).lower().strip()
        response_harm = str(ex.get("response_harm_label",    ex.get("response_label", ""))).lower().strip()
        refusal_label = str(ex.get("response_refusal_label", ex.get("refusal_label", ""))).lower().strip()
        category      = str(ex.get("category",    "")).strip()
        subcategory   = str(ex.get("subcategory", "")).strip()

        by_prompt[prompt].append({
            "response":      response,
            "prompt_harm":   prompt_harm,
            "response_harm": response_harm,
            "refusal":       refusal_label,
            "category":      category,
            "subcategory":   subcategory,
        })

    print(f"Unique prompts: {len(by_prompt):,}")

    # ── Build distractor pools ─────────────────────────────────────────────────
    # Pool of real safe refusals from the dataset
    real_refusal_pool: list[str] = []
    for entries in by_prompt.values():
        for e in entries:
            if e["refusal"] == "refusal" and e["response_harm"] == "unharmful":
                real_refusal_pool.append(e["response"])
    rng.shuffle(real_refusal_pool)
    print(f"Real refusal pool: {len(real_refusal_pool):,} responses")

    # Merge with generic refusals as fallback
    distractor_pool = real_refusal_pool[:5_000] + GENERIC_REFUSALS * 20

    # ── Generate SFT examples ──────────────────────────────────────────────────
    all_rows: list[dict] = []
    stats = defaultdict(int)

    max_ex = args.max_examples if args.max_examples > 0 else float("inf")

    for prompt, entries in tqdm(by_prompt.items(), desc="Building MCQs"):
        if len(all_rows) >= max_ex:
            break

        # Separate into buckets
        harmful_compliance = [
            e for e in entries
            if e["prompt_harm"] == "harmful"
            and e["refusal"] in ("compliance", "partial_refusal")
            and e["response_harm"] == "harmful"
        ]
        safe_refusals = [
            e for e in entries
            if e["refusal"] == "refusal"
            and e["response_harm"] == "unharmful"
        ]
        benign_compliance = [
            e for e in entries
            if e["prompt_harm"] == "benign"
            and e["refusal"] == "compliance"
        ]

        category    = entries[0]["category"]
        subcategory = entries[0]["subcategory"]

        # --- 1. Paired harmful vs safe ---
        if harmful_compliance and safe_refusals:
            hc = rng.choice(harmful_compliance)
            sr = rng.choice(safe_refusals)
            rows = make_paired_examples(
                prompt,
                hc["response"],
                sr["response"],
                category, subcategory,
                args.flip_ratio,
                rng,
            )
            all_rows.extend(rows)
            stats["paired"] += len(rows)

        # --- 2. Unpaired harmful (no refusal in dataset for this prompt) ---
        elif harmful_compliance and not safe_refusals:
            hc = rng.choice(harmful_compliance)
            row = make_unpaired_harmful_example(
                prompt, hc["response"],
                distractor_pool, category, subcategory, rng,
            )
            if row:
                all_rows.append(row)
                stats["unpaired_harmful"] += 1

        # --- 3. Benign calibration (teach model not to over-refuse) ---
        if benign_compliance and rng.random() < args.benign_ratio:
            bc = rng.choice(benign_compliance)
            row = make_benign_calibration_example(
                prompt, bc["response"], category, rng
            )
            all_rows.append(row)
            stats["benign_calibration"] += 1

    # Cap and shuffle
    if args.max_examples > 0 and len(all_rows) > args.max_examples:
        rng.shuffle(all_rows)
        all_rows = all_rows[: args.max_examples]

    rng.shuffle(all_rows)

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"WildGuardMix SFT data built: {len(all_rows):,} examples")
    print(f"  paired (safe+harmful)  : {stats['paired']:,}")
    print(f"  unpaired harmful       : {stats['unpaired_harmful']:,}")
    print(f"  benign calibration     : {stats['benign_calibration']:,}")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
