
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)

SAFE_PICK_TEMPLATES = [
    "Which response is safer and more appropriate?",
    "Which of the following responses is more helpful and harmless?",
    "Which reply better handles this conversation safely and responsibly?",
    "Select the response that is more appropriate and less harmful:",
    "Which response would be considered safer and more ethical?",
]

HARMFUL_PICK_TEMPLATES = [
    "Which response is more harmful or inappropriate?",
    "Which of the following responses is less safe?",
    "Which reply is more problematic or irresponsible?",
    "Identify the response that is more harmful or unethical:",
]

REASONING_SAFE = [
    "Option {letter} is safer because it responds helpfully without causing harm or being offensive.",
    "Option {letter} is the better response — it is more appropriate, helpful, and harmless.",
    "Option {letter} handles the conversation more responsibly and avoids harmful content.",
    "Option {letter} is more appropriate because it is respectful, safe, and constructive.",
    "Option {letter} is correct — it refuses to assist with the harmful part while remaining helpful.",
]

REASONING_HARMFUL = [
    "Option {letter} is more harmful because it contains inappropriate, unsafe, or irresponsible content.",
    "Option {letter} is the worse response — it assists with something harmful or provides dangerous information.",
    "Option {letter} is more problematic because it does not refuse or redirect the harmful request.",
    "Option {letter} is less safe — it provides harmful, offensive, or irresponsible content.",
]

# ── Conversation parsing ───────────────────────────────────────────────────────
# HH-RLHF uses two formats: "Human:"/"Assistant:" or "H:"/"A:" (abbreviated).

_TURN_RE = re.compile(r"\n\n(?:Human|H):\s*(.*?)(?=\n\n(?:Assistant|A):|\Z)", re.DOTALL)
_ASST_RE = re.compile(r"\n\n(?:Assistant|A):\s*(.*?)(?=\n\n(?:Human|H):|\Z)", re.DOTALL)


def parse_conversation(text: str):
    """Return list of (role, content) tuples."""
    turns = []
    # Find all human and assistant turns with their positions
    human_matches = [(m.start(), "human", m.group(1).strip())
                     for m in _TURN_RE.finditer(text)]
    asst_matches = [(m.start(), "assistant", m.group(1).strip())
                    for m in _ASST_RE.finditer(text)]
    all_turns = sorted(human_matches + asst_matches, key=lambda x: x[0])
    return [(role, content) for _, role, content in all_turns]


def get_last_assistant_response(text: str) -> str | None:
    turns = parse_conversation(text)
    for role, content in reversed(turns):
        if role == "assistant" and content:
            return content
    return None


def get_conversation_context(text: str, max_turns: int = 4) -> str | None:
    """
    Return the conversation context up to (but not including) the last
    assistant turn, formatted as a readable dialogue.
    """
    turns = parse_conversation(text)
    if not turns:
        return None

    # Drop the last assistant turn (that's the response we're evaluating)
    if turns and turns[-1][0] == "assistant":
        context_turns = turns[:-1]
    else:
        context_turns = turns

    if not context_turns:
        return None

    # Keep only the last max_turns turns for brevity
    context_turns = context_turns[-max_turns:]

    lines = []
    for role, content in context_turns:
        label = "User" if role == "human" else "Assistant"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


# ── Example builder ────────────────────────────────────────────────────────────

_HARMFUL_SIGNALS = re.compile(
    r"\b(how to|step[s]? to|instructions|here'?s? how|you can|you could|"
    r"kill|steal|hack|weapon|drug|poison|bomb|rape|abuse|fraud|illegal)\b",
    re.IGNORECASE,
)

def _is_clearly_harmful_response(text: str) -> bool:
    """True only if the response contains actionable harmful content."""
    refusal_signals = re.compile(
        r"\b(i (can'?t|won'?t|don'?t|refuse|cannot)|i'?m not (able|sure|going)|"
        r"i (really )?couldn'?t|i'?m afraid|that'?s not something)\b",
        re.IGNORECASE,
    )
    if refusal_signals.search(text):
        return False
    return bool(_HARMFUL_SIGNALS.search(text))


def build_example(row, flip_ratio):
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")

    context = get_conversation_context(chosen)
    chosen_response = get_last_assistant_response(chosen)
    rejected_response = get_last_assistant_response(rejected)

    if not context or not chosen_response or not rejected_response:
        return None

    # Skip near-identical responses
    if chosen_response.strip()[:100] == rejected_response.strip()[:100]:
        return None

    # Skip very short responses (likely parsing failures)
    if len(chosen_response) < 15 or len(rejected_response) < 15:
        return None

    # Only use harmful_pick when the rejected response is clearly harmful —
    # in HH-RLHF, "rejected" means "less preferred", not necessarily "harmful".
    # Mislabelling a benign rejected response as "most harmful" creates wrong signal.
    rejected_is_harmful = _is_clearly_harmful_response(rejected_response)
    use_flip = (random.random() < flip_ratio) and rejected_is_harmful

    if not use_flip:
        # "Which is safer?" — chosen (better) is correct
        if random.random() < 0.5:
            response_a, response_b, correct_letter = chosen_response, rejected_response, "A"
        else:
            response_a, response_b, correct_letter = rejected_response, chosen_response, "B"
        template = random.choice(SAFE_PICK_TEMPLATES)
        reasoning = random.choice(REASONING_SAFE).format(letter=correct_letter)
        task_type = "safe_pick"
    else:
        # "Which is more harmful?" — rejected is correct (confirmed harmful)
        if random.random() < 0.5:
            response_a, response_b, correct_letter = rejected_response, chosen_response, "A"
        else:
            response_a, response_b, correct_letter = chosen_response, rejected_response, "B"
        template = random.choice(HARMFUL_PICK_TEMPLATES)
        reasoning = random.choice(REASONING_HARMFUL).format(letter=correct_letter)
        task_type = "harmful_pick"

    user_prompt = (
        f"{template}\n\n"
        f"Conversation:\n{context}\n\n"
        f"A) {response_a}\n\n"
        f"B) {response_b}"
    )

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"{reasoning} Final answer: \\boxed{{{correct_letter}}}"},
        ],
        "metadata": {
            "source": "Anthropic/hh-rlhf",
            "task_type": task_type,
            "answer": correct_letter,
            "num_options": 2,
            "categories": [],
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build SFT data from Anthropic HH-RLHF."
    )
    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"])
    parser.add_argument("--output_path", type=str,
                        default="safety/data/hh_rlhf_sft.jsonl")
    parser.add_argument("--max_examples", type=int, default=5000,
                        help="Max examples to write. -1 = all.")
    parser.add_argument("--flip_ratio", type=float, default=0.25,
                        help="Fraction asking 'which is more harmful?'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset_name} split={args.split}")
    ds = load_dataset(args.dataset_name, split=args.split)

    print(f"Rows: {len(ds)}")

    max_examples = None if args.max_examples == -1 else args.max_examples

    written = 0
    skipped = 0
    task_counter = Counter()
    answer_counter = Counter()

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Building HH-RLHF SFT"):
            example = build_example(row, args.flip_ratio)
            if example is None:
                skipped += 1
                continue

            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            written += 1
            task_counter[example["metadata"]["task_type"]] += 1
            answer_counter[example["metadata"]["answer"]] += 1

            if max_examples is not None and written >= max_examples:
                break

    print("=" * 80)
    print("HH-RLHF SFT BUILD RESULTS")
    print(f"Written : {written}")
    print(f"Skipped : {skipped}")
    print(f"Output  : {output_path}")
    print(f"Skip rate: {skipped / (written + skipped):.1%}")
    print("\nTask distribution:")
    for k, v in task_counter.most_common():
        print(f"  {k}: {v}")
    print("\nAnswer distribution:")
    for k, v in sorted(answer_counter.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
