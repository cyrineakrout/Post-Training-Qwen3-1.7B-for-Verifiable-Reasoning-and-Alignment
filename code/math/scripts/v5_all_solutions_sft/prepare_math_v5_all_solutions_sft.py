import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import download_dataset_version_file, publish_dataset_version


SYSTEM_PROMPT = (
    "You are a mathematical reasoning model. Solve the problem carefully. "
    "Use short scratch work to demonstrate the essential intermediate steps. "
    "Return the final answer in exact form whenever possible. "
    "Put only the final answer inside one LaTeX \\boxed{} expression. "
    "Do not put explanations or units inside the box. "
    "If the problem asks for all values, all functions, all solutions, or all possible cases, "
    "make sure the boxed answer contains the complete list of valid answers."
)


ALL_SOLUTION_PATTERNS = [
    r"\bfind all\b",
    r"\ball possible\b",
    r"\ball values\b",
    r"\ball solutions\b",
    r"\ball functions\b",
    r"\bdetermine all\b",
    r"\bclassify\b",
    r"\bwhich functions\b",
    r"\bpossible values\b",
    r"\bpossible pairs\b",
    r"\bpossible triples\b",
    r"\bfor all\b.*\bfind\b",
    r"\bfunction\b",
    r"\bfunctions\b",
]


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_think_tags(text):
    if not isinstance(text, str):
        return text
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def has_boxed(text):
    return isinstance(text, str) and "\\boxed{" in text


def is_all_solution_problem(user_text):
    t = user_text.lower()
    return any(re.search(p, t) for p in ALL_SOLUTION_PATTERNS)


def normalize_row(row, source_name="math_v5"):
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None

    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
    assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), None)

    if not user or not assistant:
        return None

    assistant = strip_think_tags(assistant)

    if not has_boxed(assistant):
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ],
        "metadata": {
            "source": row.get("metadata", {}).get("source", source_name),
            "all_solution_focus": is_all_solution_problem(user),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in")
    ap.add_argument("--val_in")
    ap.add_argument("--train_out", default="math/data/math_v5_all_solutions_sft_train.jsonl")
    ap.add_argument("--val_out", default="math/data/math_v5_all_solutions_sft_val.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--all_solution_oversample", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    train_in = args.train_in or download_dataset_version_file("v4_strong_sft", "train_with_meta.jsonl")
    val_in = args.val_in or download_dataset_version_file("v4_strong_sft", "validation_with_meta.jsonl")

    raw_train = read_jsonl(train_in)
    raw_val = read_jsonl(val_in)

    train = []
    special = []

    for row in raw_train:
        item = normalize_row(row)
        if item is None:
            continue
        train.append(item)
        if item["metadata"]["all_solution_focus"]:
            special.append(item)

    # Oversample all-solution / classification / function problems
    augmented = list(train)
    for _ in range(args.all_solution_oversample):
        augmented.extend(special)

    rng.shuffle(augmented)

    val = []
    for row in raw_val:
        item = normalize_row(row)
        if item is not None:
            val.append(item)

    write_jsonl(augmented, args.train_out)
    write_jsonl(val, args.val_out)

    print("=" * 80)
    print(f"Raw train: {len(raw_train)}")
    print(f"Base usable train: {len(train)}")
    print(f"All-solution focus examples: {len(special)}")
    print(f"Oversample factor: {args.all_solution_oversample}")
    print(f"Final train: {len(augmented)} -> {args.train_out}")
    print(f"Val: {len(val)} -> {args.val_out}")

    print("Train metadata:", Counter(x["metadata"]["all_solution_focus"] for x in augmented))
    print("\nExample:")
    print(json.dumps(augmented[0], ensure_ascii=False, indent=2)[:2000])

    publish_dataset_version(
        "v5_all_solutions_sft",
        {"train.jsonl": args.train_out, "validation.jsonl": args.val_out},
    )


if __name__ == "__main__":
    main()
