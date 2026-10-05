import json
import re
from pathlib import Path

DATA_FILES = [
    Path("math/data/math_v1_early_sft_train.jsonl"),
    Path("math/data/math_v1_early_sft_val.jsonl"),
]

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed(text: str):
    """
    Extract the first boxed answer from a text.
    Example:
        Therefore, the answer is \\boxed{72}.
    Returns:
        72
    """
    match = BOXED_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def check_file(path: Path):
    if not path.exists():
        print(f"\nFile not found: {path}")
        return

    total = 0
    boxed_count = 0
    missing_boxed = []
    multiple_boxed = []
    empty_boxed = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total += 1
            example = json.loads(line)

            messages = example.get("messages", [])
            assistant_text = messages[-1]["content"] if messages else ""

            matches = BOXED_PATTERN.findall(assistant_text)

            if len(matches) == 0:
                missing_boxed.append(line_number)
            elif len(matches) > 1:
                multiple_boxed.append(line_number)
                boxed_count += 1
            else:
                boxed_answer = matches[0].strip()
                if boxed_answer == "":
                    empty_boxed.append(line_number)
                else:
                    boxed_count += 1

    print(f"\nChecking: {path}")
    print(f"Total examples: {total}")
    print(f"Examples with boxed answer: {boxed_count}")
    print(f"Missing boxed answer: {len(missing_boxed)}")
    print(f"Multiple boxed answers: {len(multiple_boxed)}")
    print(f"Empty boxed answers: {len(empty_boxed)}")

    if missing_boxed:
        print(f"First missing boxed lines: {missing_boxed[:10]}")

    if multiple_boxed:
        print(f"First multiple boxed lines: {multiple_boxed[:10]}")

    if empty_boxed:
        print(f"First empty boxed lines: {empty_boxed[:10]}")


def main():
    for path in DATA_FILES:
        check_file(path)


if __name__ == "__main__":
    main()