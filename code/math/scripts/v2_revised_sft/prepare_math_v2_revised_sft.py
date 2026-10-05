import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

SEED = 42

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import publish_dataset_version

MATH_DIR = ROOT_DIR / "math"
OUTPUT_DIR = MATH_DIR / "data"
TRAIN_OUT = OUTPUT_DIR / "math_v2_revised_sft_train.jsonl"
VAL_OUT = OUTPUT_DIR / "math_v2_revised_sft_val.jsonl"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem clearly and concisely. "
    "Always end with exactly one final answer in \\boxed{}."
)


def extract_last_boxed(text: str):
    """
    Extract the last \\boxed{...} expression, including nested braces
    such as \\boxed{\\frac{3}{4}}.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    answer_chars = []

    while i < len(text) and depth > 0:
        char = text[i]

        if char == "{":
            depth += 1
            answer_chars.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                answer_chars.append(char)
        else:
            answer_chars.append(char)

        i += 1

    if depth == 0:
        return "".join(answer_chars).strip()

    return None


def has_valid_boxed(text: str) -> bool:
    return extract_last_boxed(text) is not None


def normalize_gsm8k_answer(answer: str):
    """
    GSM8K answers usually contain:
    reasoning...
    #### final_answer
    """
    if "####" in answer:
        reasoning, final = answer.split("####", 1)
        final = final.strip()
        reasoning = reasoning.strip()
    else:
        return None

    if not final:
        return None

    assistant = (
        f"{reasoning}\n\n"
        f"Therefore, the final answer is \\boxed{{{final}}}."
    )

    return assistant


def normalize_math_solution(solution: str):
    """
    MATH solutions often already contain \\boxed{}.
    We keep the solution, but ensure it ends with exactly one standardized final boxed answer.
    Handles nested boxed answers like \\boxed{\\frac{3}{4}}.
    """
    solution = solution.strip()
    boxed = extract_last_boxed(solution)

    if boxed is None:
        return None

    assistant = (
        f"{solution}\n\n"
        f"Therefore, the final answer is \\boxed{{{boxed}}}."
    )

    if not has_valid_boxed(assistant):
        return None

    return assistant


def make_example(problem: str, assistant: str):
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": problem.strip(),
            },
            {
                "role": "assistant",
                "content": assistant.strip(),
            },
        ]
    }


def load_gsm8k_examples():
    print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")

    train_examples = []
    val_examples = []

    for row in ds["train"]:
        assistant = normalize_gsm8k_answer(row["answer"])
        if assistant is not None:
            train_examples.append(make_example(row["question"], assistant))

    for row in ds["test"]:
        assistant = normalize_gsm8k_answer(row["answer"])
        if assistant is not None:
            val_examples.append(make_example(row["question"], assistant))

    print(f"GSM8K train examples: {len(train_examples)}")
    print(f"GSM8K val examples: {len(val_examples)}")

    return train_examples, val_examples


def load_math_examples():
    print("Loading MATH dataset...")

    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    train_examples = []
    val_examples = []

    for subject in subjects:
        print(f"Loading MATH subject: {subject}")
        ds = load_dataset("EleutherAI/hendrycks_math", subject)

        for row in ds["train"]:
            assistant = normalize_math_solution(row["solution"])
            if assistant is not None:
                train_examples.append(make_example(row["problem"], assistant))

        for row in ds["test"]:
            assistant = normalize_math_solution(row["solution"])
            if assistant is not None:
                val_examples.append(make_example(row["problem"], assistant))

    print(f"MATH train examples with boxed answers: {len(train_examples)}")
    print(f"MATH val examples with boxed answers: {len(val_examples)}")

    return train_examples, val_examples


def write_jsonl(path: Path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {path}")


def check_dataset_examples(examples, name: str, limit: int = 5):
    """
    Quick sanity check: every assistant message should have a valid boxed answer.
    Prints a few examples for inspection.
    """
    bad = []

    for idx, ex in enumerate(examples):
        assistant = ex["messages"][-1]["content"]
        boxed = extract_last_boxed(assistant)

        if boxed is None:
            bad.append(idx)

    print(f"{name}: {len(examples) - len(bad)}/{len(examples)} examples have valid boxed answers.")

    if bad:
        print(f"First bad indices in {name}: {bad[:10]}")
        raise ValueError(f"{name} contains malformed boxed answers.")

    print(f"\nPreview from {name}:")
    for ex in examples[:limit]:
        assistant = ex["messages"][-1]["content"]
        boxed = extract_last_boxed(assistant)
        print(f"- boxed answer: {boxed}")


def main():
    random.seed(SEED)

    gsm_train, gsm_val = load_gsm8k_examples()
    math_train, math_val = load_math_examples()

    # V2 mix: mostly competition math, but keep GSM8K for clean reasoning.
    random.shuffle(gsm_train)
    random.shuffle(math_train)
    random.shuffle(gsm_val)
    random.shuffle(math_val)

    # Use all MATH + a controlled amount of GSM8K.
    # You can increase GSM8K later, but V2 should focus on competition math.
    gsm_train_keep = min(len(gsm_train), 3000)
    gsm_val_keep = min(len(gsm_val), 300)

    train = math_train + gsm_train[:gsm_train_keep]
    val = math_val[:500] + gsm_val[:gsm_val_keep]

    random.shuffle(train)
    random.shuffle(val)

    check_dataset_examples(train, "train")
    check_dataset_examples(val, "validation")

    write_jsonl(TRAIN_OUT, train)
    write_jsonl(VAL_OUT, val)

    print("\nDone.")
    print(f"Train file: {TRAIN_OUT}")
    print(f"Validation file: {VAL_OUT}")
    print(f"Total train examples: {len(train)}")
    print(f"Total validation examples: {len(val)}")

    publish_dataset_version(
        "v2_revised_sft",
        {"train.jsonl": TRAIN_OUT, "validation.jsonl": VAL_OUT},
    )


if __name__ == "__main__":
    main()
