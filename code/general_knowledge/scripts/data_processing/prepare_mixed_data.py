import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_CJK_PATTERN = re.compile(r"[一-鿿぀-ヿ가-힯]")
_CJK_THRESHOLD = 5


def filter_non_english(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Drop rows that contain more than _CJK_THRESHOLD CJK characters in any
    user or assistant message. These are non-English examples (Chinese,
    Japanese, Korean) that would confuse the model's output format.
    """
    kept, removed = [], 0
    for row in rows:
        noisy = False
        for msg in row.get("messages", []):
            if msg.get("role") in ("user", "assistant"):
                if len(_CJK_PATTERN.findall(msg.get("content", ""))) > _CJK_THRESHOLD:
                    noisy = True
                    break
        if noisy:
            removed += 1
        else:
            kept.append(row)
    if removed:
        print(f"  [filter_non_english] Removed {removed} non-English rows, kept {len(kept)}")
    return kept


GENERAL_KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

if str(GENERAL_KNOWLEDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERAL_KNOWLEDGE_ROOT))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import load_config_from_args

# Import the existing data utilities from prepare_sft_data.py
import prepare_sft_data as sft_data

import json


def load_sample_jsonl(
    path: str,
    max_examples: Optional[int] = None,
    seed: int = 42,
    source_name: str = "jsonl",
) -> List[Dict[str, Any]]:
    """
    Load a JSONL file and randomly sample max_examples rows.
    If max_examples is None or <= 0, keep all rows.
    """
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"\nLoaded {len(rows)} examples from {path}")

    random.seed(seed)
    random.shuffle(rows)

    if max_examples is not None and max_examples > 0:
        if len(rows) < max_examples:
            raise ValueError(
                f"Requested {max_examples} examples from {source_name}, "
                f"but file only contains {len(rows)} examples: {path}"
            )
        rows = rows[:max_examples]

    print(f"  Kept {len(rows)} examples from {source_name}")
    return rows

def load_nemotron_pool_with_coverage(
    path: str,
    max_examples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample from the original Nemotron RL pool with full option-count coverage.

    Always keeps ALL examples with >10 options (only ~87 exist in the pool),
    then fills the remaining quota randomly from the <=10 option examples.
    This ensures the model sees 11-20 option questions without losing the
    more common 4-10 option distribution.
    """
    rng = random.Random(seed)

    over10 = []
    under10 = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            n = ex.get("metadata", {}).get("num_choices", 0)
            if n > 10:
                over10.append(ex)
            else:
                under10.append(ex)

    rng.shuffle(under10)
    fill = max(0, max_examples - len(over10))
    rows = over10 + under10[:fill]
    rng.shuffle(rows)

    # Strip think blocks entirely — the chat template adds the empty think block
    # itself at generation time (enable_thinking=False), so the completion
    # should be just \boxed{X} to avoid doubled think blocks during training.
    for ex in rows:
        msgs = ex.get("messages", [])
        for msg in msgs:
            if msg.get("role") == "assistant":
                content = msg["content"]
                if "<think>" in content and "</think>" in content:
                    after = content.split("</think>")[-1].lstrip("\n")
                    msg["content"] = after

    from collections import Counter
    counts = Counter(r.get("metadata", {}).get("num_choices", 0) for r in rows)
    print(f"\nNemotron pool (coverage-sampled)")
    print(f"  Kept: {len(rows)} (>{10} options: {len(over10)}, rest: {len(rows)-len(over10)})")
    print(f"  Option count distribution: {dict(sorted(counts.items()))}")
    return rows


def convert_sciq(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts SciQ into SFT multiple-choice format.

    SciQ fields:
    - question
    - correct_answer
    - distractor1
    - distractor2
    - distractor3
    - support
    """

    from datasets import load_dataset

    rows = []

    dataset = list(load_dataset("sciq", split=split))
    random.shuffle(dataset)

    skipped_empty = 0

    for ex in dataset:
        question = str(ex.get("question", "")).strip()
        correct = str(ex.get("correct_answer", "")).strip()
        support = str(ex.get("support", "")).strip()

        distractors = [
            str(ex.get("distractor1", "")).strip(),
            str(ex.get("distractor2", "")).strip(),
            str(ex.get("distractor3", "")).strip(),
        ]

        if not question or not correct or any(d == "" for d in distractors):
            skipped_empty += 1
            continue

        choices = distractors + [correct]
        random.shuffle(choices)

        answer_index = choices.index(correct)

        rows.append(
            sft_data.make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"sciq_{split}",
                explanation="",
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nSciQ split={split}")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped empty/bad text: {skipped_empty}")

    return rows


def convert_openbookqa(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts OpenBookQA into the SFT format with a real rationale.

    Dataset: allenai/openbookqa
    Config: "additional"  (provides `fact1`, the seed science fact for the
            question — same role as SciQ's `support`).

    Rows where `fact1` is missing or empty fall back to the templated
    rationale from `make_sft_example` (no real rationale, just choice echo).
    """

    from datasets import load_dataset

    rows = []

    dataset = list(load_dataset("allenai/openbookqa", "additional", split=split))
    random.shuffle(dataset)

    skipped_empty = 0
    skipped_bad_answer = 0
    with_fact = 0

    for ex in dataset:
        question = str(ex.get("question_stem", "")).strip()
        choice_obj = ex.get("choices", {})
        answer_key = str(ex.get("answerKey", "")).strip().upper()
        fact1 = str(ex.get("fact1", "")).strip()

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

        if fact1:
            with_fact += 1
            rows.append(
                sft_data.make_sft_example_with_explanation(
                    question=question,
                    choices=choices,
                    answer_index=answer_index,
                    source=f"openbookqa_{split}",
                    explanation="",
                )
            )
        else:
            rows.append(
                sft_data.make_sft_example_with_explanation(
                    question=question,
                    choices=choices,
                    answer_index=answer_index,
                    source=f"openbookqa_{split}",
                    explanation=""
                )
            )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nOpenBookQA split={split}")
    print(f"  Kept examples: {len(rows)} (with fact1 rationale: {with_fact})")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows


def shorten_explanation(text: str, max_chars: int = 350) -> str:
    text = " ".join(str(text).strip().split())

    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]
    last_sentence_end = max(
        cut.rfind("."),
        cut.rfind("?"),
        cut.rfind("!"),
    )

    if last_sentence_end > 100:
        return cut[: last_sentence_end + 1]

    return cut.rstrip() + "..."


def convert_gpqa(
    config_name: str = "gpqa_extended",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts GPQA into the SFT multiple-choice format.
    """

    from datasets import load_dataset

    rows = []
    skipped_empty = 0

    dataset = list(load_dataset("Idavidrein/gpqa", config_name, split="train"))
    random.shuffle(dataset)

    for ex in dataset:
        question = str(ex.get("Question", "")).strip()
        correct = str(ex.get("Correct Answer", "")).strip()
        distractors = [
            str(ex.get("Incorrect Answer 1", "")).strip(),
            str(ex.get("Incorrect Answer 2", "")).strip(),
            str(ex.get("Incorrect Answer 3", "")).strip(),
        ]

        explanation = shorten_explanation(
            str(ex.get("Explanation", "")).strip(),
            max_chars=500,
        )

        if not question or not correct or any(not d for d in distractors):
            skipped_empty += 1
            continue

        choices = distractors + [correct]
        random.shuffle(choices)
        answer_index = choices.index(correct)

        rows.append(
            sft_data.make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"gpqa_{config_name}",
                explanation="",
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nGPQA config={config_name}")
    print(
        f"  Kept examples: {len(rows)} "
        f"(with explanation rationale: "
        f"{sum(1 for r in rows if r['metadata'].get('has_explanation'))})"
    )
    print(f"  Skipped empty/bad text: {skipped_empty}")

    return rows


def load_supergpqa_eval_questions(eval_file: str) -> set:
    """Return the set of question texts from SuperGPQA examples in the eval file."""
    questions = set()
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "supergpqa" not in row.get("source", ""):
                continue
            prompt = row.get("prompt", "")
            # question is everything before '\n\nChoices:'
            q = prompt.split("\n\nChoices:")[0].strip()
            if q:
                questions.add(q)
    return questions


def convert_supergpqa(
    max_examples: Optional[int] = None,
    eval_file: Optional[str] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Converts SuperGPQA (m-a-p/SuperGPQA) into SFT multiple-choice format,
    filtering out any questions that appear in eval_file to avoid contamination.

    SuperGPQA has 26k+ graduate-level questions across many disciplines with
    variable numbers of options (answer_letter can be A-J or beyond).
    """
    from datasets import load_dataset

    eval_questions: set = set()
    if eval_file:
        eval_questions = load_supergpqa_eval_questions(eval_file)
        print(f"\nSuperGPQA: loaded {len(eval_questions)} eval questions to exclude")

    dataset = list(load_dataset("m-a-p/SuperGPQA", split="train"))
    rng = random.Random(seed)
    rng.shuffle(dataset)

    rows = []
    skipped_eval = 0
    skipped_empty = 0
    skipped_bad_answer = 0

    for ex in dataset:
        question = str(ex.get("question", "")).strip()
        if not question:
            skipped_empty += 1
            continue

        if question in eval_questions:
            skipped_eval += 1
            continue

        options_raw = ex.get("options", [])
        if isinstance(options_raw, str):
            try:
                import ast
                options_raw = ast.literal_eval(options_raw)
            except Exception:
                skipped_empty += 1
                continue

        choices = [str(o).strip() for o in options_raw]
        if not choices or any(c == "" for c in choices):
            skipped_empty += 1
            continue

        answer_letter = str(ex.get("answer_letter", "")).strip().upper()
        if not answer_letter or len(answer_letter) != 1:
            skipped_bad_answer += 1
            continue

        answer_index = ord(answer_letter) - ord("A")
        if answer_index < 0 or answer_index >= len(choices):
            skipped_bad_answer += 1
            continue

        discipline = str(ex.get("discipline", "")).strip().lower().replace(" ", "_")
        field = str(ex.get("field", "")).strip().lower().replace(" ", "_")
        source = f"supergpqa_{discipline}_{field}"

        rows.append(
            sft_data.make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=source,
                explanation="",
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nSuperGPQA")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped (in eval): {skipped_eval}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows


def convert_mmlu_pro(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts MMLU-Pro into the SFT multiple-choice format.

    Dataset: TIGER-Lab/MMLU-Pro

    Typical fields:
      - question
      - options / choices
      - answer
      - answer_index
      - category

    MMLU-Pro is useful because it has harder academic questions and often
    more than 4 answer choices, sometimes up to 10.
    """

    from datasets import load_dataset

    rows = []
    skipped_empty = 0
    skipped_bad_answer = 0

    dataset = list(load_dataset("TIGER-Lab/MMLU-Pro", split=split))
    random.shuffle(dataset)

    for ex in dataset:
        question = str(ex.get("question", "")).strip()

        # MMLU-Pro usually uses "options"
        raw_choices = ex.get("options", None)
        if raw_choices is None:
            raw_choices = ex.get("choices", None)

        if not question or not raw_choices:
            skipped_empty += 1
            continue

        choices = [str(choice).strip() for choice in raw_choices]

        if any(choice == "" for choice in choices):
            skipped_empty += 1
            continue

        answer_index = None

        # Case 1: dataset provides answer_index
        if "answer_index" in ex and ex["answer_index"] is not None:
            try:
                answer_index = int(ex["answer_index"])
            except Exception:
                answer_index = None

        # Case 2: dataset provides answer as a letter, e.g. "B"
        if answer_index is None:
            answer = str(ex.get("answer", "")).strip()

            if len(answer) == 1 and answer.isalpha():
                answer_index = ord(answer.upper()) - ord("A")

        # Case 3: dataset provides answer as exact text
        if answer_index is None:
            answer = str(ex.get("answer", "")).strip()
            if answer in choices:
                answer_index = choices.index(answer)

        if answer_index is None or answer_index < 0 or answer_index >= len(choices):
            skipped_bad_answer += 1
            continue

        category = str(ex.get("category", "unknown")).strip().lower().replace(" ", "_")

        rows.append(
            sft_data.make_sft_example_with_explanation(
                question=question,
                choices=choices,
                answer_index=answer_index,
                source=f"mmlu_pro_{split}_{category}",
                explanation=""
            )
        )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nMMLU-Pro split={split}")
    print(f"  Kept examples: {len(rows)}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows

def convert_medmcqa(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Converts MedMCQA into the SFT multiple-choice format.

    Keeps only single-choice questions.

    Dataset: openlifescienceai/medmcqa
    Typical fields:
      - question
      - opa, opb, opc, opd
      - cop: correct option index, usually 0-3
      - choice_type: "single" or "multi"
      - exp: explanation, sometimes available
    """

    from datasets import load_dataset

    rows = []

    dataset = list(load_dataset("openlifescienceai/medmcqa", split=split))
    random.shuffle(dataset)

    skipped_empty = 0
    skipped_bad_answer = 0
    skipped_not_single = 0
    with_exp = 0

    for ex in dataset:
        choice_type = str(ex.get("choice_type", "")).strip().lower()

        # Keep only single-choice questions
        if choice_type and choice_type != "single":
            skipped_not_single += 1
            continue

        question = str(ex.get("question", "")).strip()

        choices = [
            str(ex.get("opa", "")).strip(),
            str(ex.get("opb", "")).strip(),
            str(ex.get("opc", "")).strip(),
            str(ex.get("opd", "")).strip(),
        ]

        if not question or any(choice == "" for choice in choices):
            skipped_empty += 1
            continue

        try:
            answer_index = int(ex.get("cop"))
        except Exception:
            skipped_bad_answer += 1
            continue

        if answer_index < 0 or answer_index >= len(choices):
            skipped_bad_answer += 1
            continue

        explanation = str(ex.get("exp", "")).strip()

        if explanation:
            with_exp += 1
            rows.append(
                sft_data.make_sft_example_with_explanation(
                    question=question,
                    choices=choices,
                    answer_index=answer_index,
                    source=f"medmcqa_{split}_single",
                    explanation="",
                )
            )
        else:
            rows.append(
                sft_data.make_sft_example_with_explanation(
                    question=question,
                    choices=choices,
                    answer_index=answer_index,
                    source=f"medmcqa_{split}_single",
                    explanation="",
                )
            )

        if max_examples is not None and len(rows) >= max_examples:
            break

    print(f"\nMedMCQA split={split}")
    print(f"  Kept single-choice examples: {len(rows)} (with explanation: {with_exp})")
    print(f"  Skipped non-single-choice: {skipped_not_single}")
    print(f"  Skipped empty/bad text: {skipped_empty}")
    print(f"  Skipped bad answer index: {skipped_bad_answer}")

    return rows
def build_mixed_dataset(
    mmlu_aux_examples: int,
    mmlu_dev_examples: int,
    arc_examples: int,
    openbookqa_examples: int,
    sciq_examples: int,
    commonsenseqa_examples: int = 0,
    medmcqa_examples: int = 0,
    gpqa_examples: int = 0,
    gpqa_config: str = "gpqa_extended",
    mmlu_pro_examples: int = 0,
    mmlu_pro_split: str = "validation",
    nemotron_examples: int = 0,
    nemotron_file: Optional[str] = None,
    nemotron_science_examples: int = 0,
    nemotron_science_file: Optional[str] = None,
    nemotron_pool_examples: int = 0,
    nemotron_pool_file: Optional[str] = None,
    supergpqa_examples: int = 0,
    supergpqa_eval_file: Optional[str] = None,
    seed: int = 42,
    ) -> List[Dict[str, Any]]:
    """
    Builds the mixed training set.

    MMLU is split between two TRAINING sources, both of which are DISJOINT
    from the MMLU `validation` split that we evaluate on:
      * `auxiliary_train`: bulk training data, no subject metadata.
      * `dev`: small subject-tagged top-up (~285 items max, 5 per subject).

    NOTHING from `validation` is included here, so the local val numbers
    stay honest.
    """

    rows = []

    if mmlu_aux_examples > 0:
        rows.extend(
            sft_data.convert_mmlu(
                split="auxiliary_train",
                max_examples=mmlu_aux_examples,
            )
        )

    if mmlu_dev_examples > 0:
        rows.extend(
            sft_data.convert_mmlu_dev(
                max_examples=mmlu_dev_examples,
                apply_subject_filter=False,
            )
        )

    if arc_examples > 0:
        rows.extend(
            sft_data.convert_arc(
                split="train",
                max_examples=arc_examples,
            )
        )

    if openbookqa_examples > 0:
        rows.extend(
            convert_openbookqa(
                split="train",
                max_examples=openbookqa_examples,
            )
        )

    if sciq_examples > 0:
        rows.extend(
            convert_sciq(
                split="train",
                max_examples=sciq_examples,
            )
        )
    if gpqa_examples > 0:
        rows.extend(
            convert_gpqa(
                config_name=gpqa_config,
                max_examples=gpqa_examples,
            )
        )
    if mmlu_pro_examples > 0:
        rows.extend(
            convert_mmlu_pro(
                split=mmlu_pro_split,
                max_examples=mmlu_pro_examples,
            )
        )
    if commonsenseqa_examples > 0:
        rows.extend(
            sft_data.convert_commonsenseqa(
                split="train",
                max_examples=commonsenseqa_examples,
            )
        )

    if medmcqa_examples > 0:
        rows.extend(
            convert_medmcqa(
                split="train",
                max_examples=medmcqa_examples,
            )
        )
    if nemotron_examples > 0:
        if not nemotron_file:
            raise ValueError(
                "--nemotron_examples was set, but --nemotron_file was not provided. "
                "Pass the filtered Nemotron JSONL file."
            )

        rows.extend(
            load_sample_jsonl(
                path=nemotron_file,
                max_examples=nemotron_examples,
                seed=seed,
                source_name="nemotron_filtered",
            )
        )

    if nemotron_science_examples > 0:
        if not nemotron_science_file:
            raise ValueError(
                "--nemotron_science_examples was set, but --nemotron_science_file was not provided. "
                "Run prepare_nemotron_science.py first to generate the JSONL file."
            )
        rows.extend(
            load_sample_jsonl(
                path=nemotron_science_file,
                max_examples=nemotron_science_examples,
                seed=seed,
                source_name="nemotron_science",
            )
        )

    if nemotron_pool_examples > 0:
        if not nemotron_pool_file:
            raise ValueError(
                "--nemotron_pool_examples was set, but --nemotron_pool_file was not provided."
            )
        rows.extend(
            load_nemotron_pool_with_coverage(
                path=nemotron_pool_file,
                max_examples=nemotron_pool_examples,
                seed=seed,
            )
        )

    if supergpqa_examples > 0:
        rows.extend(
            convert_supergpqa(
                max_examples=supergpqa_examples,
                eval_file=supergpqa_eval_file,
                seed=seed,
            )
        )

    rows = filter_non_english(rows)
    random.shuffle(rows)

    print("\nMixed training dataset")
    print(f"  Total examples: {len(rows)}")
    print(f"  MMLU auxiliary_train requested: {mmlu_aux_examples}")
    print(f"  MMLU dev requested: {mmlu_dev_examples}")
    print(f"  ARC examples requested: {arc_examples}")
    print(f"  OpenBookQA examples requested: {openbookqa_examples}")
    print(f"  SciQ examples requested: {sciq_examples}")
    print(f"  GPQA examples requested: {gpqa_examples}")
    print(f"  GPQA config: {gpqa_config}")
    print(f"  MMLU-Pro examples requested: {mmlu_pro_examples}")
    print(f"  MMLU-Pro split: {mmlu_pro_split}")
    print(f"  CommonsenseQA examples requested: {commonsenseqa_examples}")
    print(f"  MedMCQA examples requested: {medmcqa_examples}")
    print(f"  Nemotron examples requested: {nemotron_examples}")
    print(f"  Nemotron file: {nemotron_file}")
    print(f"  Nemotron science examples requested: {nemotron_science_examples}")
    print(f"  Nemotron science file: {nemotron_science_file}")
    print(f"  Nemotron pool examples requested: {nemotron_pool_examples}")
    print(f"  Nemotron pool file: {nemotron_pool_file}")
    print(f"  SuperGPQA examples requested: {supergpqa_examples}")
    print(f"  SuperGPQA eval file (excluded): {supergpqa_eval_file}")

    return rows


def main():
    config_path, config = load_config_from_args()
    data_config = config["data"]

    # Important:
    # make_sft_example inside prepare_sft_data.py uses the global SYSTEM_PROMPT
    # from that module, so we need to set it there.
    sft_data.SYSTEM_PROMPT = config["prompts"]["sft_system_prompt"]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=config_path,
        help="Path to the general knowledge config JSON file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=data_config["seed"],
    )

    parser.add_argument(
        "--mmlu_aux_examples",
        type=int,
        default=6000,
        help=(
            "Number of MMLU `auxiliary_train` examples (no subject metadata, "
            "no contamination with `test`). This is the bulk MMLU source."
        ),
    )

    parser.add_argument(
        "--mmlu_dev_examples",
        type=int,
        default=0,
        help=(
            "Number of MMLU `dev` examples (subject-tagged, ~285 items max "
            "= 5 per subject * 57 subjects). DISJOINT from `validation`, so "
            "safe to use as a training coverage top-up. `validation` is "
            "reserved for evaluation."
        ),
    )

    parser.add_argument(
        "--mmlu_examples",
        type=int,
        default=None,
        help=(
            "DEPRECATED legacy flag. If set, overrides --mmlu_aux_examples and "
            "disables --mmlu_dev_val_examples. Kept for backward compatibility."
        ),
    )
    parser.add_argument(
    "--nemotron_examples",
    type=int,
    default=0,
    help="Number of NVIDIA Nemotron RL Knowledge MCQA examples.",
    )
    parser.add_argument(
    "--nemotron_file",
    type=str,
    default=None,
    help="Path to filtered Nemotron JSONL file to sample from.",
    )

    parser.add_argument(
        "--arc_examples",
        type=int,
        default=0,
        help="Number of ARC examples.",
    )

    parser.add_argument(
        "--openbookqa_examples",
        type=int,
        default=0,
        help="Number of OpenBookQA examples (now uses the `additional` config with fact1).",
    )

    parser.add_argument(
        "--sciq_examples",
        type=int,
        default=0,
        help="Number of SciQ examples.",
    )

    parser.add_argument(
        "--train_output",
        type=str,
        default="general_knowledge/data/sft_train_mixed_clean_mmluaux6000_dev285_arc750_obqa500_sciq1000.jsonl",
        help="Where to save the mixed SFT training JSONL.",
    )
    parser.add_argument(
        "--gpqa_examples",
        type=int,
        default=0,
        help="Number of GPQA examples to include. Start small, e.g. 200 or 300.",
    )

    parser.add_argument(
        "--gpqa_config",
        type=str,
        default="gpqa_extended",
        choices=["gpqa_diamond", "gpqa_main", "gpqa_extended", "gpqa_experts"],
        help="GPQA config to use. Prefer gpqa_extended for training; reserve gpqa_diamond for eval.",
    )
    parser.add_argument(
        "--mmlu_pro_examples",
        type=int,
        default=0,
        help="Number of MMLU-Pro examples to include. Start with 500.",
    )

    parser.add_argument(
        "--mmlu_pro_split",
        type=str,
        default="test",
        choices=["test", "validation", "train"],
        help="MMLU-Pro split to use. Usually use test for training only if allowed by your project rules.",
    )
    parser.add_argument(
        "--commonsenseqa_examples",
        type=int,
        default=0,
        help="Number of CommonsenseQA examples.",
    )

    parser.add_argument(
        "--medmcqa_examples",
        type=int,
        default=0,
        help="Number of MedMCQA examples.",
    )

    parser.add_argument(
        "--nemotron_science_examples",
        type=int,
        default=0,
        help="Number of Nemotron science examples to include.",
    )
    parser.add_argument(
        "--nemotron_science_file",
        type=str,
        default=None,
        help="Path to JSONL produced by prepare_nemotron_science.py.",
    )
    parser.add_argument(
        "--nemotron_pool_examples",
        type=int,
        default=0,
        help="Number of examples from the original Nemotron RL pool (covers 2-20 options). All >10-option examples are always included first.",
    )
    parser.add_argument(
        "--nemotron_pool_file",
        type=str,
        default=None,
        help="Path to nemotron_train_pool_after_heldout.jsonl.",
    )
    parser.add_argument(
        "--supergpqa_examples",
        type=int,
        default=0,
        help=(
            "Number of SuperGPQA examples to include (m-a-p/SuperGPQA, 26k+ total). "
            "Use --supergpqa_eval_file to exclude eval questions automatically."
        ),
    )
    parser.add_argument(
        "--supergpqa_eval_file",
        type=str,
        default=None,
        help=(
            "Path to the eval JSONL containing SuperGPQA examples to exclude from training. "
            "Questions are matched by text so no IDs are needed."
        ),
    )

    args = parser.parse_args()
    random.seed(args.seed)

    if args.mmlu_examples is not None:
        print(
            "[deprecated] --mmlu_examples set; routing all MMLU into "
            "auxiliary_train and skipping the dev top-up."
        )
        mmlu_aux_examples = args.mmlu_examples
        mmlu_dev_examples = 0
    else:
        mmlu_aux_examples = args.mmlu_aux_examples
        mmlu_dev_examples = args.mmlu_dev_examples

    train_rows = build_mixed_dataset(
        mmlu_aux_examples=mmlu_aux_examples,
        mmlu_dev_examples=mmlu_dev_examples,
        arc_examples=args.arc_examples,
        openbookqa_examples=args.openbookqa_examples,
        sciq_examples=args.sciq_examples,
        commonsenseqa_examples=args.commonsenseqa_examples,
        medmcqa_examples=args.medmcqa_examples,
        gpqa_examples=args.gpqa_examples,
        gpqa_config=args.gpqa_config,
        mmlu_pro_examples=args.mmlu_pro_examples,
        mmlu_pro_split=args.mmlu_pro_split,
        nemotron_examples=args.nemotron_examples,
        nemotron_file=args.nemotron_file,
        nemotron_science_examples=args.nemotron_science_examples,
        nemotron_science_file=args.nemotron_science_file,
        nemotron_pool_examples=args.nemotron_pool_examples,
        nemotron_pool_file=args.nemotron_pool_file,
        supergpqa_examples=args.supergpqa_examples,
        supergpqa_eval_file=args.supergpqa_eval_file,
        seed=args.seed,
    )

    # Normalize all system prompts to the config's sft_system_prompt so that
    # training and CI inference (which injects this same prompt via the template
    # default) see identical system context.
    for row in train_rows:
        for msg in row.get("messages", []):
            if msg.get("role") == "system":
                msg["content"] = sft_data.SYSTEM_PROMPT

    sft_data.save_jsonl(train_rows, args.train_output)

    print("=" * 80)
    print(f"Saved {len(train_rows)} mixed training examples to {args.train_output}")

    sft_data.print_source_counts(train_rows, "Training source counts:")


if __name__ == "__main__":
    main()