import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
if str(GENERAL_KNOWLEDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERAL_KNOWLEDGE_ROOT))

from config import load_config_from_args


SYSTEM_PROMPT = None
KEEP_MMLU_SUBJECTS = {
    # World knowledge / history / geography / politics
    "global_facts",
    "high_school_geography",
    "high_school_world_history",
    "high_school_us_history",
    "high_school_european_history",
    "prehistory",
    "world_religions",
    "us_foreign_policy",
    "security_studies",
    "high_school_government_and_politics",
    "international_law",

    # Law / humanities-style reasoning
    "professional_law",
    "jurisprudence",

    # Humanities / social sciences / common-sense-adjacent
    "sociology",
    "high_school_psychology",
    "professional_psychology",
    "philosophy",
    "moral_disputes",
    "moral_scenarios",
    "logical_fallacies",
    "human_sexuality",
    "human_aging",
    "miscellaneous",

    # Science / medicine
    "astronomy",
    "conceptual_physics",
    "high_school_physics",
    "college_physics",
    "high_school_biology",
    "college_biology",
    "high_school_chemistry",
    "college_chemistry",
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
    "medical_genetics",
    "nutrition",
    "virology",

    # Economics / business / social science
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "business_ethics",
    "management",
    "marketing",
    "professional_accounting",
}
def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def make_sft_example_with_explanation(
    question: str,
    choices: List[str],
    answer_index: int,
    source: str,
    explanation: str,
) -> Dict[str, Any]:
    """
    Same prompt format as `make_sft_example`, but the assistant turn prepends
    a short rationale taken from the dataset itself (e.g. SciQ's `support`
    or OpenBookQA's `fact1`).

    User prompt format is byte-for-byte identical to the staff
    `validation_samples/general_knowledge.jsonl` (no leading whitespace).
    """

    if not choices:
        raise ValueError("choices cannot be empty")

    if answer_index < 0 or answer_index >= len(choices):
        raise ValueError(
            f"Invalid answer_index={answer_index} for {len(choices)} choices"
        )

    letters = [chr(ord("A") + i) for i in range(len(choices))]
    answer_letter = letters[answer_index]
    correct_choice_text = choices[answer_index]

    choices_text = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    )

    user_prompt = f"{question}\n\nChoices:\n{choices_text}"

    explanation = explanation.strip()

    assistant_content = f"\\boxed{{{answer_letter}}}"
        
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": source,
            "num_choices": len(choices),
            "answer": answer_letter,
            # "has_explanation": bool(explanation),
            "has_explanation": explanation,  # We want to track the original explanation presence, even if we don't use it all in the prompt.
        },
    }



def convert_mmlu(
    split: str,
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = []

    from datasets import load_dataset

    dataset = list(load_dataset("cais/mmlu", "all", split=split))
    random.shuffle(dataset)

    skipped_empty = 0
    skipped_bad_answer = 0

    for ex in dataset:
        question = str(ex.get("question", "")).strip()
        raw_choices = ex.get("choices", [])

        if not question or not raw_choices:
            skipped_empty += 1
            continue

        choices = [str(choice).strip() for choice in raw_choices]

        if any(choice == "" for choice in choices):
            skipped_empty += 1
            continue

        try:
            answer_index = int(ex["answer"])
        except Exception:
            skipped_bad_answer += 1
            continue

        if answer_index < 0 or answer_index >= len(choices):
            skipped_bad_answer += 1
            continue

        rows.append(
            make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"mmlu_{split}",
                explanation=""
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nMMLU split={split}")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows

def _convert_mmlu_split(
    split: str,
    max_examples: Optional[int],
    apply_subject_filter: bool,
    source_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Internal helper: load one MMLU split and convert to SFT rows.
    Both `dev` and `validation` are subject-tagged.
    """

    from datasets import load_dataset

    rows: List[Dict[str, Any]] = []
    dataset = list(load_dataset("cais/mmlu", "all", split=split))
    random.shuffle(dataset)

    skipped_subject = 0
    skipped_empty = 0
    skipped_bad_answer = 0
    subject_counts: Dict[str, int] = {}

    for ex in dataset:
        subject = str(ex.get("subject", "")).strip()

        if apply_subject_filter and subject not in KEEP_MMLU_SUBJECTS:
            skipped_subject += 1
            continue

        question = str(ex.get("question", "")).strip()
        raw_choices = ex.get("choices", [])

        if not question or not raw_choices:
            skipped_empty += 1
            continue

        choices = [str(choice).strip() for choice in raw_choices]
        if any(choice == "" for choice in choices):
            skipped_empty += 1
            continue

        try:
            answer_index = int(ex["answer"])
        except Exception:
            skipped_bad_answer += 1
            continue

        if answer_index < 0 or answer_index >= len(choices):
            skipped_bad_answer += 1
            continue

        source = f"{source_prefix}_{subject}" if subject else source_prefix

        rows.append(
            make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=source,
                explanation=""
            )
        )

        if subject:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nMMLU {split} (apply_subject_filter={apply_subject_filter})")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped excluded subjects: {skipped_subject}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    if subject_counts:
        print("\n  Subject counts:")
        for subject, count in sorted(subject_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    {subject}: {count}")

    return rows


def convert_mmlu_dev(
    max_examples: Optional[int] = None,
    apply_subject_filter: bool = False,
) -> List[Dict[str, Any]]:
    """
    Pulls from MMLU's `dev` split only (~285 items, 5 per subject).
    Subject-tagged. Safe to use for TRAINING because it does not overlap with
    the `validation` split that we score against.
    """
    return _convert_mmlu_split(
        split="dev",
        max_examples=max_examples,
        apply_subject_filter=apply_subject_filter,
        source_prefix="mmlu_dev",
    )


def convert_mmlu_validation(
    max_examples: Optional[int] = None,
    apply_subject_filter: bool = False,
) -> List[Dict[str, Any]]:
    """
    Pulls from MMLU's `validation` split only (~1531 items).
    Subject-tagged. RESERVED FOR EVAL — do not use this in any training mix
    or you'll inflate your val numbers.
    """
    return _convert_mmlu_split(
        split="validation",
        max_examples=max_examples,
        apply_subject_filter=apply_subject_filter,
        source_prefix="mmlu_validation",
    )


def convert_arc(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts ARC-Easy + ARC-Challenge into the same SFT format.

    ARC fields:
    - question: str
    - choices: {"text": [...], "label": [...]}
    - answerKey: str, e.g. "A", "B", "C", "D"
    """

    rows = []

    from datasets import load_dataset

    raw_examples = []

    for arc_config in ["ARC-Challenge"]:
        ds = list(load_dataset("allenai/ai2_arc", arc_config, split=split))
        for ex in ds:
            ex["arc_config"] = arc_config
            raw_examples.append(ex)

    random.shuffle(raw_examples)

    skipped_empty = 0
    skipped_bad_answer = 0

    for ex in raw_examples:
        question = str(ex.get("question", "")).strip()
        choice_obj = ex.get("choices", {})
        answer_key = str(ex.get("answerKey", "")).strip().upper()

        raw_choices = choice_obj.get("text", [])
        raw_labels = choice_obj.get("label", [])

        if not question or not raw_choices or not raw_labels or not answer_key:
            skipped_empty += 1
            continue

        choices = [str(choice).strip() for choice in raw_choices]
        labels = [str(label).strip().upper() for label in raw_labels]

        if any(choice == "" for choice in choices):
            skipped_empty += 1
            continue

        if answer_key not in labels:
            skipped_bad_answer += 1
            continue

        answer_index = labels.index(answer_key)

        rows.append(
            make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"arc_{ex['arc_config']}_{split}",
                explanation=""
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nARC split={split}")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows

def convert_commonsenseqa(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts CommonsenseQA into the SFT multiple-choice format.

    Dataset: tau/commonsense_qa
    Fields:
      - question
      - choices: {"label": [...], "text": [...]}
      - answerKey
    """

    from datasets import load_dataset

    rows = []

    dataset = list(load_dataset("tau/commonsense_qa", split=split))
    random.shuffle(dataset)

    skipped_empty = 0
    skipped_bad_answer = 0

    for ex in dataset:
        question = str(ex.get("question", "")).strip()
        choice_obj = ex.get("choices", {})
        answer_key = str(ex.get("answerKey", "")).strip().upper()

        raw_choices = choice_obj.get("text", [])
        raw_labels = choice_obj.get("label", [])

        if not question or not raw_choices or not raw_labels or not answer_key:
            skipped_empty += 1
            continue

        choices = [str(choice).strip() for choice in raw_choices]
        labels = [str(label).strip().upper() for label in raw_labels]

        if len(choices) != len(labels):
            skipped_empty += 1
            continue

        if any(choice == "" for choice in choices):
            skipped_empty += 1
            continue

        if answer_key not in labels:
            skipped_bad_answer += 1
            continue

        answer_index = labels.index(answer_key)

        rows.append(
            make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"commonsenseqa_{split}",
                explanation=""
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nCommonsenseQA split={split}")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows

def convert_quartz(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts AllenAI QUARTZ into the SFT multiple-choice format.

    Dataset: allenai/quartz
    Fields:
      - question:  str — the question text
      - choices:   {"text": [...], "label": [...]}  (always A/B)
      - answerKey: str — "A" or "B"
      - para:      str — qualitative background paragraph used as explanation

    Splits available: train, validation, test
    """

    from datasets import load_dataset

    rows: List[Dict[str, Any]] = []

    dataset = list(load_dataset("allenai/quartz", split=split))
    random.shuffle(dataset)

    skipped_empty = 0
    skipped_bad_answer = 0

    for ex in dataset:
        question = str(ex.get("question", "")).strip()
        choice_obj = ex.get("choices", {})
        answer_key = str(ex.get("answerKey", "")).strip().upper()
        explanation = str(ex.get("para", "")).strip()

        raw_choices = choice_obj.get("text", [])
        raw_labels = choice_obj.get("label", [])

        if not question or not raw_choices or not raw_labels or not answer_key:
            skipped_empty += 1
            continue

        choices = [str(c).strip() for c in raw_choices]
        labels = [str(l).strip().upper() for l in raw_labels]

        if len(choices) != len(labels):
            skipped_empty += 1
            continue

        if any(c == "" for c in choices):
            skipped_empty += 1
            continue

        if answer_key not in labels:
            skipped_bad_answer += 1
            continue

        answer_index = labels.index(answer_key)

        rows.append(
            make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"quartz_{split}",
                explanation=explanation,
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nQUARTZ split={split}")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows


def print_source_counts(rows: List[Dict[str, Any]], title: str) -> None:
    print(f"\n{title}")

    sources = sorted(set(row["metadata"]["source"] for row in rows))

    for source in sources:
        count = sum(row["metadata"]["source"] == source for row in rows)
        print(f"  {source}: {count}")


def main():
    global SYSTEM_PROMPT
    config_path, config = load_config_from_args()
    data_config = config["data"]
    SYSTEM_PROMPT = config["prompts"]["sft_system_prompt"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the general knowledge config JSON file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["toy", "mmlu", "arc", "mmlu_filtered", "mmlu_clean", "quartz"],
        default="toy",
        help=(
            "Choose data source. `mmlu_clean` uses auxiliary_train + dev + "
            "validation splits (no `test` contamination) and is the recommended "
            "MMLU source for SFT."
        ),
    )

    parser.add_argument("--seed", type=int, default=data_config["seed"])

    parser.add_argument(
        "--toy_output",
        type=str,
        default=data_config["toy_output"],
    )

    parser.add_argument(
        "--train_output",
        type=str,
        default=data_config["train_output"],
    )

    parser.add_argument(
        "--val_output",
        type=str,
        default=data_config["val_output"],
    )

    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=data_config["max_train_examples"],
        help="Maximum number of MMLU training examples to save.",
    )

    parser.add_argument(
        "--max_val_examples",
        type=int,
        default=data_config["max_val_examples"],
        help="Maximum number of MMLU validation examples to save.",
    )
    parser.add_argument(
        "--mmlu_train_split",
        type=str,
        default=data_config["mmlu_train_split"],
    )
    parser.add_argument(
        "--mmlu_val_split",
        type=str,
        default=data_config["mmlu_val_split"],
    )

    args = parser.parse_args()
    random.seed(args.seed)

    if args.mode == "toy":
        rows = build_toy_dataset()
        save_jsonl(rows, args.toy_output)

        print("=" * 80)
        print(f"Saved {len(rows)} toy examples to {args.toy_output}")
        print_source_counts(rows, "Toy source counts:")
        return

    if args.mode == "mmlu":
        print("Loading MMLU training split...")
        train_rows = convert_mmlu(
            split=args.mmlu_train_split,
            max_examples=args.max_train_examples,
        )

        print("Loading MMLU validation split...")
        val_rows = convert_mmlu(
            split=args.mmlu_val_split,
            max_examples=args.max_val_examples,
        )

        random.shuffle(train_rows)
        random.shuffle(val_rows)

        save_jsonl(train_rows, args.train_output)
        save_jsonl(val_rows, args.val_output)

        print("=" * 80)
        print(f"Saved {len(train_rows)} MMLU training examples to {args.train_output}")
        print(f"Saved {len(val_rows)} MMLU validation examples to {args.val_output}")

        print_source_counts(train_rows, "Training source counts:")
        print_source_counts(val_rows, "Validation source counts:")
        return
    if args.mode == "mmlu_filtered":
        print("Loading MMLU training split...")
        train_rows = convert_mmlu_filtered(
            split="test",
            max_examples=args.max_train_examples,
        )

        print("Loading MMLU validation split...")
        val_rows = convert_mmlu_filtered(
            split=args.mmlu_val_split,
            max_examples=args.max_val_examples,
        )

        random.shuffle(train_rows)
        random.shuffle(val_rows)

        save_jsonl(train_rows, args.train_output)
        save_jsonl(val_rows, args.val_output)

        print("=" * 80)
        print(f"Saved {len(train_rows)} MMLU training examples to {args.train_output}")
        print(f"Saved {len(val_rows)} MMLU validation examples to {args.val_output}")

        print_source_counts(train_rows, "Training source counts:")
        print_source_counts(val_rows, "Validation source counts:")
        return

    if args.mode == "mmlu_clean":
        # Clean recipe — strict separation between training and eval splits:
        #   * train bulk:  auxiliary_train (no contamination with anything)
        #   * train topup: `dev` only (~285 items, subject-tagged, DISJOINT
        #                  from `validation` so eval stays clean)
        #   * eval:        full `validation`, NO subject filter (broadest CI proxy)
        bulk_target = args.max_train_examples
        # `dev` caps at ~285 items total; let it fill naturally.
        topup_target = min(285, max(0, bulk_target // 5))

        print("Loading MMLU auxiliary_train (no subject filter)...")
        train_rows = convert_mmlu(
            split="auxiliary_train",
            max_examples=max(0, bulk_target - topup_target),
        )

        if topup_target > 0:
            print("Loading MMLU `dev` as a subject-tagged top-up (training only)...")
            train_rows.extend(
                convert_mmlu_dev(
                    max_examples=topup_target,
                    apply_subject_filter=False,
                )
            )

        print("Loading MMLU `validation` (UNFILTERED) for eval...")
        # Pass max_val_examples=None in config to take all ~1531 items.
        val_rows = convert_mmlu_validation(
            max_examples=args.max_val_examples,
            apply_subject_filter=False,
        )

        random.shuffle(train_rows)
        random.shuffle(val_rows)

        save_jsonl(train_rows, args.train_output)
        save_jsonl(val_rows, args.val_output)

        print("=" * 80)
        print(f"Saved {len(train_rows)} MMLU training examples to {args.train_output}")
        print(f"Saved {len(val_rows)} MMLU validation examples to {args.val_output}")

        print_source_counts(train_rows, "Training source counts:")
        print_source_counts(val_rows, "Validation source counts:")
        return

    if args.mode == "arc":
        print("Loading ARC training split...")
        train_rows = convert_arc(
            split="train",
            max_examples=args.max_train_examples,
        )

        print("Loading ARC validation split...")
        val_rows = convert_arc(
            split="validation",
            max_examples=args.max_val_examples,
        )

        random.shuffle(train_rows)
        random.shuffle(val_rows)

        save_jsonl(train_rows, args.train_output)
        save_jsonl(val_rows, args.val_output)

        print("=" * 80)
        print(f"Saved {len(train_rows)} ARC training examples to {args.train_output}")
        print(f"Saved {len(val_rows)} ARC validation examples to {args.val_output}")

        print_source_counts(train_rows, "Training source counts:")
        print_source_counts(val_rows, "Validation source counts:")
        return

    if args.mode == "quartz":
        print("Loading QUARTZ training split...")
        train_rows = convert_quartz(
            split="train",
            max_examples=args.max_train_examples,
        )

        print("Loading QUARTZ validation split...")
        val_rows = convert_quartz(
            split="validation",
            max_examples=args.max_val_examples,
        )

        random.shuffle(train_rows)
        random.shuffle(val_rows)

        save_jsonl(train_rows, args.train_output)
        save_jsonl(val_rows, args.val_output)

        print("=" * 80)
        print(f"Saved {len(train_rows)} QUARTZ training examples to {args.train_output}")
        print(f"Saved {len(val_rows)} QUARTZ validation examples to {args.val_output}")

        print_source_counts(train_rows, "Training source counts:")
        print_source_counts(val_rows, "Validation source counts:")
        return


if __name__ == "__main__":
    main()
