import json
import re
import sys
from pathlib import Path

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import publish_dataset_version

MATH_DIR = ROOT_DIR / "math"
OUTPUT_DIR = MATH_DIR / "data"
TRAIN_FILE = OUTPUT_DIR / "math_v1_early_sft_train.jsonl"
VAL_FILE = OUTPUT_DIR / "math_v1_early_sft_val.jsonl"

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve the problem step by step. "
    "Always put the final answer inside \\boxed{}."
)


def extract_gsm8k_final_answer(answer: str) -> str:
    """
    GSM8K answers usually end with:
        #### 72

    This function extracts only the final answer, e.g.:
        72
    """
    if "####" in answer:
        return answer.split("####")[-1].strip()

    return answer.strip()


def remove_gsm8k_final_marker(answer: str) -> str:
    """
    Remove GSM8K's final marker:
        #### 72

    We keep only the reasoning part, then add our own boxed final answer.
    """
    if "####" in answer:
        return answer.split("####")[0].strip()

    return answer.strip()


def clean_calculator_annotations(text: str) -> str:
    """
    GSM8K often contains calculator annotations like:
        48/2 = <<48/2=24>>24

    This converts them into:
        48/2 = 24

    This makes the reasoning cleaner for SFT.
    """
    return re.sub(r"<<[^>]+>>", "", text)


def normalize_final_answer(answer: str) -> str:
    """
    Light normalization of the final answer.
    Removes commas in numbers, e.g.:
        1,000 -> 1000
    """
    answer = answer.strip()

    # Remove commas inside numbers
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)

    return answer


def build_assistant_answer(raw_solution: str, final_answer: str) -> str:
    """
    Build the final assistant response:
        reasoning
        Therefore, the final answer is \\boxed{...}.
    """
    reasoning = remove_gsm8k_final_marker(raw_solution)
    reasoning = clean_calculator_annotations(reasoning)
    final_answer = normalize_final_answer(final_answer)

    return (
        reasoning.strip()
        + f"\n\nTherefore, the final answer is \\boxed{{{final_answer}}}."
    )


def format_example(question: str, raw_solution: str):
    """
    Convert one GSM8K example into chat-format messages.
    """
    final_answer = extract_gsm8k_final_answer(raw_solution)
    assistant_answer = build_assistant_answer(raw_solution, final_answer)

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question.strip(),
            },
            {
                "role": "assistant",
                "content": assistant_answer,
            },
        ]
    }


def write_jsonl(path: Path, examples):
    """
    Write examples to JSONL.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main")

    train_examples = []
    val_examples = []

    print("Formatting training set...")
    for ex in dataset["train"]:
        train_examples.append(
            format_example(
                question=ex["question"],
                raw_solution=ex["answer"],
            )
        )

    print("Formatting validation set...")
    for ex in dataset["test"]:
        val_examples.append(
            format_example(
                question=ex["question"],
                raw_solution=ex["answer"],
            )
        )

    write_jsonl(TRAIN_FILE, train_examples)
    write_jsonl(VAL_FILE, val_examples)

    print(f"Saved {len(train_examples)} training examples to {TRAIN_FILE}")
    print(f"Saved {len(val_examples)} validation examples to {VAL_FILE}")

    publish_dataset_version(
        "v1_early_sft",
        {"train.jsonl": TRAIN_FILE, "validation.jsonl": VAL_FILE},
    )

    print("\nExample:")
    print(json.dumps(train_examples[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
