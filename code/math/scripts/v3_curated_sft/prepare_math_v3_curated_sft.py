import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any


try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import publish_dataset_version

MATH_DIR = ROOT_DIR / "math"
DATA_DIR = MATH_DIR / "data"

TRAIN_OUT = DATA_DIR / "math_v3_curated_sft_train.jsonl"
VAL_OUT = DATA_DIR / "math_v3_curated_sft_val.jsonl"

TRAIN_META_OUT = DATA_DIR / "math_v3_curated_sft_train_with_meta.jsonl"
VAL_META_OUT = DATA_DIR / "math_v3_curated_sft_val_with_meta.jsonl"


# ============================================================
# Global config
# ============================================================

SEED = 52

SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. "
    "Solve carefully and concisely. "
    "Use the key steps needed to reach the answer. "
    "Do not use multiple-choice letters. "
    "Do not repeat yourself. "
    "Always end with exactly one final answer in \\boxed{}. "
    "After the boxed answer, stop immediately."
)

# Goal: recover more Hendrycks while keeping OlympiadContest clean.
TARGET_COUNTS = {
    "hendrycks_math": 7000,
    "olympiad_contest_20k": 7000,
}

MAX_TOTAL_EXAMPLES = 14000

VAL_FRACTION = 0.02
MAX_VAL_PER_SOURCE = 250

MIN_PROMPT_CHARS = 20
MAX_PROMPT_CHARS = 8000

MIN_SOLUTION_WORDS = 35
MAX_SOLUTION_WORDS = 800

MIN_SOLUTION_CHARS = 70
MAX_SOLUTION_CHARS = 4500

MAX_ANSWER_CHARS = 180

DIFFICULTY_THRESHOLD = 3

SOURCE_DIFFICULTY_THRESHOLDS = {
    "hendrycks_math": 2,
    "olympiad_contest_20k": 4,
}

BOXED = "\\boxed{"

HENDRYCKS_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


# ============================================================
# Filters
# ============================================================

BAD_DATASET_MARKERS = [
    "aime 2024",
    "aime2024",
    "aime 2025",
    "aime2025",
    "imo 2025",
    "imo2025",
    "olympiadbench",
    "competition_math:test",
    "hendrycks_math:test",
    "math test",
]

PROOF_PATTERNS = [
    r"\bprove\b",
    r"\bshow that\b",
    r"\bdemonstrate\b",
    r"\bestablish\b",
    r"démontrer",
    r"demontrer",
    r"montrer que",
    r"prouver que",
    r"\\blacksquare",
    r"\bqed\b",
]

MCQ_PATTERNS = [
    r"answer choices",
    r"answer choice",
    r"multiple choice",
    r"\\bmcq\\b",
    r"options:",
    r"given options",
    r"correct option",
    r"enter the letter",
    r"corresponding option",
    r"option a",
    r"option b",
    r"option c",
    r"option d",
    r"option e",
    r"the answer is option",
    r"\\textbf\{\([A-Ea-e]\)\}",
    r"\\mathrm\{\([A-Ea-e]\)\}",
    r"\*\*\([A-Ea-e]\)\*\*",
    r"\([A-Ea-e]\)\s*[^()\n]{1,120}",
    r"(?m)^\s*[A-Ea-e]\s*[:.)]\s+\S+",
    r"(?m)^\s*\([A-Ea-e]\)\s+\S+",
]

BAD_PROMPT_MARKERS = [
    "read the following material",
    "please read carefully",
    "teacher",
    "blackboard",
    "fill in",
    "fill in the blank",
    "task 1",
    "task 2",
    "full score",
    "mathematics test",
    "enter the letter",
    "correct option",
    "given options",
    "corresponding option",
    "<br/>",
]

CODE_MARKERS = [
    "```",
    "```python",
    "python program",
    "write a python",
    "print(",
    "input(",
    "def ",
    "import ",
    "return ",
    "answer =",
]

BAD_ANSWER_FRAGMENTS = [
    "proven",
    "hence",
    "therefore",
    "the example with",
    "continuing",
    "inductively",
    "20 points",
    "points",
    "qed",
    "blacksquare",
    ", and",
    "is divisible by",
    "is a multiple of",
    "this proves",
    "conclusion",
    "solution",
    "proof",
    "expressed as",
    "common fraction",
    "answer is",
    "the final answer",
    "winning strategy",
    "opponent",
]

FORMULA_PROMPT_MARKERS = [
    "find the general term",
    "determine a_n",
    "as a function of",
    "find a_n",
    "find the formula",
    "general formula",
    "find all",
    "determine all",
    "all solutions",
    "solve for",
]

HARD_KEYWORDS = [
    "integer",
    "positive integer",
    "prime",
    "divisible",
    "divisibility",
    "modulo",
    "remainder",
    "congruent",
    "\\pmod",
    "polynomial",
    "coefficient",
    "roots",
    "complex",
    "sequence",
    "recurrence",
    "functional equation",
    "triangle",
    "circle",
    "angle",
    "geometry",
    "area",
    "perimeter",
    "probability",
    "combinatorics",
    "permutation",
    "combination",
    "ways",
    "\\binom",
    "minimum",
    "maximum",
    "least",
    "greatest",
    "find all",
    "determine all",
    "\\sum",
    "\\prod",
]

FINAL_ANSWER_PATTERNS = [
    re.compile(r"(?:final answer|answer)\s*(?:is|:|=)\s*(.+)$", re.IGNORECASE),
    re.compile(r"therefore,?\s*(?:the )?(?:answer|result)\s*(?:is|=)\s*(.+)$", re.IGNORECASE),
    re.compile(r"hence,?\s*(?:the )?(?:answer|result)\s*(?:is|=)\s*(.+)$", re.IGNORECASE),
]


# ============================================================
# Text helpers
# ============================================================

def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} rows to {path}")


def canonical_prompt(text: str) -> str:
    text = clean_text(text).lower()

    # Remove Asymptote snippets to deduplicate visually identical geometry prompts better.
    text = re.sub(r"\[asy\].*?\[/asy\]", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"asy\].*?\[/asy", "", text, flags=re.DOTALL | re.IGNORECASE)

    text = text.replace("\\!", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(canonical_prompt(prompt).encode("utf-8")).hexdigest()


def has_marker(text: str, markers: list[str]) -> bool:
    low = clean_text(text).lower()
    return any(marker.lower() in low for marker in markers)


def has_regex(text: str, patterns: list[str]) -> bool:
    text = clean_text(text)
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


# ============================================================
# Answer extraction
# ============================================================

def extract_boxed(text: str) -> str | None:
    text = clean_text(text)

    start = text.rfind(BOXED)
    if start == -1:
        return None

    i = start + len(BOXED)
    depth = 1
    answer = []

    while i < len(text) and depth > 0:
        ch = text[i]

        if ch == "{":
            depth += 1
            answer.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                answer.append(ch)
        else:
            answer.append(ch)

        i += 1

    if depth == 0:
        return clean_text("".join(answer))

    return None


def strip_answer(answer: str) -> str:
    answer = clean_text(answer)
    answer = answer.strip(" .\n\t")

    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()

    if answer.startswith("\\(") and answer.endswith("\\)"):
        answer = answer[2:-2].strip()

    if answer.startswith("\\[") and answer.endswith("\\]"):
        answer = answer[2:-2].strip()

    return answer.strip(" .\n\t")


def normalize_answer(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return ""

    boxed = extract_boxed(text)
    if boxed is not None:
        return strip_answer(boxed)

    return strip_answer(text)


def is_proof_prompt(prompt: str) -> bool:
    return has_regex(prompt, PROOF_PATTERNS)


def is_mcq(prompt: str, solution: str = "", answer: str = "") -> bool:
    text = clean_text(prompt + "\n" + solution + "\n" + answer)

    if has_regex(text, MCQ_PATTERNS):
        return True

    answer_clean = clean_text(answer).strip(" .")

    if re.fullmatch(r"[A-Ea-e]", answer_clean):
        return True

    if re.match(r"^[A-Ea-e]\s*[:.)]", answer_clean):
        return True

    return False


def answer_looks_bad(answer: str, prompt: str = "") -> bool:
    answer = strip_answer(answer)
    low = answer.lower()
    prompt_low = clean_text(prompt).lower()

    if not answer:
        return True

    if len(answer) > MAX_ANSWER_CHARS:
        return True

    if "\n" in answer:
        return True

    if BOXED in answer:
        return True

    # Single-letter answers are often bad extractions like \boxed{p}.
    # Exception: allow i only in complex expressions if not single.
    if re.fullmatch(r"[a-zA-Z]", answer):
        return True

    if re.match(r"^[A-Ea-e]\s*[:.)]", answer):
        return True

    if "\\textbf" in low or "\\mathrm" in low or "option" in low:
        return True

    if any(fragment in low for fragment in BAD_ANSWER_FRAGMENTS):
        return True

    if answer in {",", ".", ";", ":", ", and"}:
        return True

    if len(answer.split()) > 10:
        return True

    if "\\text{" in answer and len(answer.split()) > 4:
        return True

    if has_marker(answer, CODE_MARKERS):
        return True

    # Reject chained intermediate equations unless the prompt asks for formulas/solutions.
    if answer.count("=") >= 2 and not any(marker in prompt_low for marker in FORMULA_PROMPT_MARKERS):
        return True

    # Reject sentence-like non-math answers.
    has_math_signal = bool(
        re.search(r"[0-9]", answer)
        or "\\" in answer
        or "^" in answer
        or "_" in answer
        or "=" in answer
        or "<" in answer
        or ">" in answer
        or "{" in answer
        or "}" in answer
        or "!" in answer
        or "," in answer
        or "(" in answer
        or ")" in answer
    )

    if not has_math_signal and len(answer.split()) >= 3:
        return True

    return False


def trim_after_last_boxed(solution: str) -> str:
    solution = clean_text(solution)

    start = solution.rfind(BOXED)
    if start == -1:
        return solution

    i = start + len(BOXED)
    depth = 1

    while i < len(solution) and depth > 0:
        ch = solution[i]

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

        i += 1

    if depth == 0:
        return solution[:i].strip()

    return solution


def extract_explicit_final_answer(solution: str, prompt: str = "") -> str:
    """
    Safe extraction only.
    No fallback to arbitrary last equation.
    """
    solution = clean_text(solution)

    boxed = extract_boxed(solution)
    if boxed is not None:
        answer = normalize_answer(boxed)
        if not answer_looks_bad(answer, prompt):
            return answer

    lines = [clean_text(line) for line in solution.splitlines() if clean_text(line)]

    for line in reversed(lines[-10:]):
        low = line.lower()

        if (
            "final answer" not in low
            and "answer is" not in low
            and "therefore" not in low
            and "hence" not in low
        ):
            continue

        for pattern in FINAL_ANSWER_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue

            candidate = match.group(1)
            candidate = candidate.split("\n")[0]
            candidate = candidate.strip(" .")
            candidate = normalize_answer(candidate)

            if not answer_looks_bad(candidate, prompt):
                return candidate

    return ""


# ============================================================
# Row parsing
# ============================================================

def parse_llama_chat_text(text: str) -> tuple[str, str] | None:
    text = clean_text(text)

    user_start = "<|start_header_id|>user<|end_header_id|>"
    assistant_start = "<|start_header_id|>assistant<|end_header_id|>"
    eot = "<|eot_id|>"

    if user_start not in text or assistant_start not in text:
        return None

    try:
        prompt = text.split(user_start, 1)[1].split(eot, 1)[0]
        solution = text.split(assistant_start, 1)[1].split(eot, 1)[0]
    except Exception:
        return None

    prompt = clean_text(prompt)
    solution = clean_text(solution)

    if not prompt or not solution:
        return None

    return prompt, solution


def get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]

    return None


def normalize_row(row: dict[str, Any], source: str, origin: str) -> tuple[str, str, str] | None:
    # kevin009 dataset format
    if "text" in row and row["text"] is not None:
        parsed = parse_llama_chat_text(row["text"])
        if parsed is None:
            return None

        prompt, solution = parsed
        answer = extract_explicit_final_answer(solution, prompt)
        return prompt, solution, answer

    # generic instruction/solution format
    prompt = clean_text(
        get_first(row, ["problem", "question", "prompt", "input", "query", "instruction"])
    )

    solution = clean_text(
        get_first(row, ["solution", "output", "response", "completion", "assistant", "rationale", "reasoning"])
    )

    answer = normalize_answer(
        get_first(row, ["final_answer", "answer", "reference", "target", "label", "correct_answer"]) or ""
    )

    if not answer:
        answer = extract_explicit_final_answer(solution, prompt)

    return prompt, solution, answer


# ============================================================
# Difficulty and example construction
# ============================================================

def difficulty_score(prompt: str, solution: str, source: str) -> int:
    text = (prompt + "\n" + solution).lower()
    score = 0

    for keyword in HARD_KEYWORDS:
        if keyword.lower() in text:
            score += 1

    if "\\sum" in text or "\\prod" in text or "\\binom" in text:
        score += 2

    if "$" in prompt or "\\" in prompt:
        score += 1

    # Hendrycks is high-trust competition MATH data.
    if source == "hendrycks_math":
        score += 2

    words = len(solution.split())

    if 90 <= words <= 700:
        score += 2
    elif words < 45:
        score -= 1
    elif words > 900:
        score -= 2

    return score


def make_concise_solution(solution: str, answer: str, prompt: str) -> str:
    solution = clean_text(solution)
    answer = normalize_answer(answer)

    if answer_looks_bad(answer, prompt):
        return ""

    boxed = extract_boxed(solution)

    if boxed is not None:
        boxed_answer = normalize_answer(boxed)

        if answer_looks_bad(boxed_answer, prompt):
            return ""

        solution = trim_after_last_boxed(solution)
    else:
        solution = solution.rstrip() + f"\n\nTherefore, the final answer is \\boxed{{{answer}}}."

    solution = trim_after_last_boxed(solution)
    solution = clean_text(solution)

    words = len(solution.split())

    if words < MIN_SOLUTION_WORDS or words > MAX_SOLUTION_WORDS:
        return ""

    if len(solution) < MIN_SOLUTION_CHARS or len(solution) > MAX_SOLUTION_CHARS:
        return ""

    return solution


def make_sft_example(
    prompt: str,
    solution: str,
    answer: str,
    source: str,
    origin: str,
) -> dict[str, Any] | None:
    prompt = clean_text(prompt)
    solution = clean_text(solution)
    answer = normalize_answer(answer)

    if not prompt or not solution or not answer:
        return None

    if len(prompt) < MIN_PROMPT_CHARS or len(prompt) > MAX_PROMPT_CHARS:
        return None

    combined = prompt + "\n" + solution + "\n" + answer + "\n" + origin

    if has_marker(combined, BAD_DATASET_MARKERS):
        return None

    if is_proof_prompt(prompt):
        return None

    if has_marker(prompt, BAD_PROMPT_MARKERS):
        return None

    if has_marker(prompt, CODE_MARKERS) or has_marker(solution, CODE_MARKERS):
        return None

    if is_mcq(prompt, solution, answer):
        return None

    if answer_looks_bad(answer, prompt):
        return None

    concise_solution = make_concise_solution(solution, answer, prompt)

    if not concise_solution:
        return None

    boxed = extract_boxed(concise_solution)
    if boxed is None:
        return None

    boxed = normalize_answer(boxed)

    if answer_looks_bad(boxed, prompt):
        return None

    score = difficulty_score(prompt, concise_solution, source)
    min_score = SOURCE_DIFFICULTY_THRESHOLDS.get(source, DIFFICULTY_THRESHOLD)

    if score < min_score:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": concise_solution},
        ],
        "metadata": {
            "source": source,
            "origin": origin,
            "answer": boxed,
            "prompt_hash": prompt_hash(prompt),
            "difficulty_score": score,
            "solution_chars": len(concise_solution),
            "solution_words": len(concise_solution.split()),
        },
    }


# ============================================================
# Loaders
# ============================================================

def load_hendrycks_math() -> list[dict[str, Any]]:
    if load_dataset is None:
        print("datasets unavailable; skipping hendrycks math")
        return []

    examples = []

    for config in HENDRYCKS_CONFIGS:
        try:
            ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")
            origin = f"EleutherAI/hendrycks_math:{config}:train"
            print(f"Loaded {origin}")
        except Exception as exc:
            print(f"Could not load EleutherAI/hendrycks_math {config}: {exc}")
            continue

        before = len(examples)

        for row in ds:
            normalized = normalize_row(row, "hendrycks_math", origin)

            if normalized is None:
                continue

            prompt, solution, answer = normalized
            ex = make_sft_example(prompt, solution, answer, "hendrycks_math", origin)

            if ex is not None:
                examples.append(ex)

        print(f"Prepared {len(examples) - before} from {config}")

    print(f"Prepared total hendrycks_math: {len(examples)}")
    return examples


def load_olympiad_contest_20k() -> list[dict[str, Any]]:
    if load_dataset is None:
        print("datasets unavailable; skipping olympiad contest 20k")
        return []

    try:
        ds = load_dataset("kevin009/olympiad-math-contest-llama3-20k", split="train")
        origin = "kevin009/olympiad-math-contest-llama3-20k:train"
        print(f"Loaded {origin}")
    except Exception as exc:
        print(f"Could not load olympiad contest 20k: {exc}")
        return []

    examples = []

    for row in ds:
        normalized = normalize_row(row, "olympiad_contest_20k", origin)

        if normalized is None:
            continue

        prompt, solution, answer = normalized
        ex = make_sft_example(prompt, solution, answer, "olympiad_contest_20k", origin)

        if ex is not None:
            examples.append(ex)

    print(f"Prepared total olympiad_contest_20k: {len(examples)}")
    return examples


# ============================================================
# Selection / split
# ============================================================

def deduplicate(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []

    examples = sorted(
        examples,
        key=lambda ex: (
            -ex["metadata"].get("difficulty_score", 0),
            ex["metadata"].get("solution_words", 9999),
        ),
    )

    for ex in examples:
        h = ex["metadata"]["prompt_hash"]

        if h in seen:
            continue

        seen.add(h)
        output.append(ex)

    return output


def sample_source(examples: list[dict[str, Any]], source: str, target: int) -> list[dict[str, Any]]:
    examples = deduplicate(examples)

    examples = sorted(
        examples,
        key=lambda ex: (
            -ex["metadata"].get("difficulty_score", 0),
            ex["metadata"].get("solution_words", 9999),
        ),
    )

    selected = examples[: min(target, len(examples))]
    random.shuffle(selected)

    print(f"Selected {len(selected)}/{len(examples)} from {source}, target={target}")
    return selected


def split_train_val(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    val = []

    sources = sorted(set(ex["metadata"]["source"] for ex in examples))

    for source in sources:
        rows = [ex for ex in examples if ex["metadata"]["source"] == source]
        random.shuffle(rows)

        val_size = min(MAX_VAL_PER_SOURCE, max(1, int(len(rows) * VAL_FRACTION)))

        val.extend(rows[:val_size])
        train.extend(rows[val_size:])

    random.shuffle(train)
    random.shuffle(val)

    return train, val


def strip_meta(ex: dict[str, Any]) -> dict[str, Any]:
    return {"messages": ex["messages"]}


# ============================================================
# Reporting
# ============================================================

def print_stats(examples: list[dict[str, Any]], name: str) -> None:
    counts = {}
    scores = []
    words = []

    for ex in examples:
        source = ex["metadata"]["source"]
        counts[source] = counts.get(source, 0) + 1
        scores.append(ex["metadata"]["difficulty_score"])
        words.append(ex["metadata"]["solution_words"])

    print(f"\n{name}: {len(examples)} examples")

    for source, count in sorted(counts.items()):
        print(f"- {source}: {count}")

    if scores:
        print(f"Difficulty: min={min(scores)} max={max(scores)} avg={sum(scores) / len(scores):.2f}")

    if words:
        print(f"Solution words: min={min(words)} max={max(words)} avg={sum(words) / len(words):.2f}")


def preview(examples: list[dict[str, Any]], k: int = 8) -> None:
    print("\nPreview:")

    for ex in examples[:k]:
        print("=" * 100)
        print("SOURCE:", ex["metadata"]["source"])
        print("ORIGIN:", ex["metadata"]["origin"])
        print("SCORE:", ex["metadata"]["difficulty_score"])
        print("WORDS:", ex["metadata"]["solution_words"])
        print("ANSWER:", ex["metadata"]["answer"])
        print("PROMPT:", ex["messages"][1]["content"][:600])
        print("ASSISTANT:", ex["messages"][-1]["content"][:800])


# ============================================================
# Main
# ============================================================

def main() -> None:
    random.seed(SEED)

    print("=" * 100)
    print("Preparing v3 Qwen clean math dataset")
    print("=" * 100)
    print("Sources used:")
    print("- EleutherAI/hendrycks_math train")
    print("- kevin009/olympiad-math-contest-llama3-20k train")
    print("Sources NOT used:")
    print("- AI-MO/NuminaMath-CoT")
    print("- AI-MO/olympiads")
    print("- olympiad_stepwise_20k")
    print("- competition_math duplicate clones")
    print("- all benchmark/test splits")
    print("=" * 100)

    grouped = {
        "hendrycks_math": load_hendrycks_math(),
        "olympiad_contest_20k": load_olympiad_contest_20k(),
    }

    selected = []

    for source, target in TARGET_COUNTS.items():
        selected.extend(sample_source(grouped.get(source, []), source, target))

    selected = deduplicate(selected)
    random.shuffle(selected)

    if len(selected) > MAX_TOTAL_EXAMPLES:
        selected = selected[:MAX_TOTAL_EXAMPLES]

    train, val = split_train_val(selected)

    print_stats(train, "TRAIN")
    print_stats(val, "VALIDATION")
    preview(train)

    write_jsonl(TRAIN_OUT, [strip_meta(ex) for ex in train])
    write_jsonl(VAL_OUT, [strip_meta(ex) for ex in val])

    write_jsonl(TRAIN_META_OUT, train)
    write_jsonl(VAL_META_OUT, val)

    print("\nDone.")
    print(f"Train file: {TRAIN_OUT}")
    print(f"Val file:   {VAL_OUT}")
    print(f"Train meta: {TRAIN_META_OUT}")
    print(f"Val meta:   {VAL_META_OUT}")
    print(f"Train examples: {len(train)}")
    print(f"Val examples:   {len(val)}")
    print("\nCheck distribution:")
    print("grep -o '\"source\": \"[^\"]*\"' math/data/math_v3_curated_sft_train_with_meta.jsonl | sort | uniq -c")

    publish_dataset_version(
        "v3_curated_sft",
        {
            "train.jsonl": TRAIN_OUT,
            "validation.jsonl": VAL_OUT,
            "train_with_meta.jsonl": TRAIN_META_OUT,
            "validation_with_meta.jsonl": VAL_META_OUT,
        },
    )


if __name__ == "__main__":
    main()
