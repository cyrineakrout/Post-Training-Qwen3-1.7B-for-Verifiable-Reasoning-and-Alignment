import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset


ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import publish_dataset_version

DATA_DIR = ROOT_DIR / "math" / "data"

TRAIN_OUT = DATA_DIR / "math_v4_strong_sft_train.jsonl"
VAL_OUT = DATA_DIR / "math_v4_strong_sft_val.jsonl"

TRAIN_META_OUT = DATA_DIR / "math_v4_strong_sft_train_with_meta.jsonl"
VAL_META_OUT = DATA_DIR / "math_v4_strong_sft_val_with_meta.jsonl"

SEED = 52

SYSTEM_PROMPT = (
    "You are a mathematical reasoning model. "
    "Solve the problem carefully and concisely. "
    "Return the final answer in exact form when possible. "
    "Put only the final answer inside one LaTeX \\boxed{} expression."
)

# Broad SFT like the groups around 0.39, but lightly filtered.
TARGET_COUNTS = {
    "numina_math_cot": 30000,
    "hendrycks_math": 7000,
    "olympiad_contest_20k": 10000,
}

MAX_TOTAL_EXAMPLES = 47000
VAL_FRACTION = 0.01
MAX_VAL_PER_SOURCE = 250

MIN_PROMPT_CHARS = 15
MAX_PROMPT_CHARS = 7000

MIN_SOLUTION_WORDS = 25
MAX_SOLUTION_WORDS = 900

MAX_SOLUTION_CHARS = 6000
MAX_ANSWER_CHARS = 120

RANDOM_SCAN_LIMIT_NUMINA = 250000

HENDRYCKS_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

BAD_PROMPT_PATTERNS = [
    r"\bmultiple choice\b",
    r"\banswer choices\b",
    r"\bchoose\b",
    r"\boptions?\b",
    r"\boption [a-e]\b",
    r"(?m)^\s*\([A-Ea-e]\)",
    r"(?m)^\s*[A-Ea-e]\s*[\).:]",
    r"\\textbf\{\([A-Ea-e]\)\}",
    r"\\mathrm\{\([A-Ea-e]\)\}",
    r"<br\s*/?>",
    r"\bpython\b",
    r"```",
]

# Light proof filtering: only prompts that explicitly ask for a proof.
PROOF_PROMPT_PATTERNS = [
    r"^\s*prove\b",
    r"^\s*prove:",
    r"^\s*show that\b",
    r"^\s*demonstrate\b",
    r"^\s*establish\b",
    r"\bdémontrer\b",
    r"\bmontrer que\b",
    r"\bprouver que\b",
]

# Extra light-cleaning rules requested.
BAD_GAME_OR_STRATEGY_PROMPT_PATTERNS = [
    r"\\boxed\{p\}",
    r"\bwinning strategy\b",
    r"\bwho will win\b",
    r"\bwill win\b",
    r"\bgame\b",
    r"\bplayers?\b",
    r"\bPetya\b",
    r"\bVasya\b",
    r"\bAlice\b",
    r"\bBob\b",
    r"\bKolya\b",
    r"\bVitya\b",
    r"\bAnna\b",
    r"\bEdda\b",
    r"\bMagni\b",
    r"\bWhite\b",
    r"\bBlack\b",
]

BAD_ANSWER_FRAGMENTS = [
    "therefore",
    "hence",
    "proven",
    "proof",
    "qed",
    "the final answer",
    "answer is",
    "winning strategy",
    "opponent",
    "points",
    "20 points",
    "this proves",
    "conclusion",
]

BOXED = "\\boxed{"


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    text = str(x)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {path}")


def canonical_prompt(prompt: str) -> str:
    prompt = clean_text(prompt).lower()
    prompt = re.sub(r"\[asy\].*?\[/asy\]", "", prompt, flags=re.DOTALL)
    prompt = re.sub(r"\s+", " ", prompt)
    return prompt.strip()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(canonical_prompt(prompt).encode("utf-8")).hexdigest()


def has_regex(text: str, patterns: list[str]) -> bool:
    text = clean_text(text)
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def extract_last_boxed(text: str) -> str | None:
    text = clean_text(text)
    starts = [m.start() for m in re.finditer(re.escape(BOXED), text)]
    if not starts:
        return None

    start = starts[-1] + len(BOXED)
    i = start
    depth = 1
    chars = []

    while i < len(text) and depth > 0:
        ch = text[i]

        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                chars.append(ch)
        else:
            chars.append(ch)

        i += 1

    if depth == 0:
        return clean_text("".join(chars))

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


def normalize_answer(answer: Any) -> str:
    text = clean_text(answer)
    if not text:
        return ""

    boxed = extract_last_boxed(text)
    if boxed:
        return strip_answer(boxed)

    return strip_answer(text)


def answer_looks_bad(answer: str) -> bool:
    answer = strip_answer(answer)
    low = answer.lower()

    if not answer:
        return True

    if len(answer) > MAX_ANSWER_CHARS:
        return True

    if "\n" in answer:
        return True

    if BOXED in answer:
        return True

    # Important new rule: remove text-answer examples.
    if "\\text{" in answer:
        return True

    # Remove Chinese / non-Latin text-answer style outputs.
    if re.search(r"[\u4e00-\u9fff]", answer):
        return True

    if re.fullmatch(r"[A-Ea-e]", answer):
        return True

    if re.fullmatch(r"[a-zA-Z]", answer):
        return True

    if any(x in low for x in BAD_ANSWER_FRAGMENTS):
        return True

    if len(answer.split()) > 12:
        return True

    if answer in {",", ".", ";", ":", ", and"}:
        return True

    return False


def extract_explicit_answer(solution: str, fallback_answer: Any = "") -> str:
    solution = clean_text(solution)

    boxed = extract_last_boxed(solution)
    if boxed:
        ans = normalize_answer(boxed)
        if not answer_looks_bad(ans):
            return ans

    ans = normalize_answer(fallback_answer)
    if ans and not answer_looks_bad(ans):
        return ans

    # Conservative final-answer patterns.
    lines = [clean_text(x) for x in solution.splitlines() if clean_text(x)]

    for line in reversed(lines[-10:]):
        low = line.lower()
        if "final answer" not in low and "answer is" not in low and "therefore" not in low:
            continue

        math_candidates = re.findall(r"\$([^$]{1,220})\$", line)
        if math_candidates:
            cand = normalize_answer(math_candidates[-1])
            if not answer_looks_bad(cand):
                return cand

        m = re.search(r"(?:final answer|answer)\s*(?:is|:|=)\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            cand = normalize_answer(m.group(1))
            if not answer_looks_bad(cand):
                return cand

    return ""


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


def normalize_row(row: dict[str, Any]) -> tuple[str, str, str] | None:
    # kevin009 llama-chat style.
    if "text" in row and row["text"] is not None:
        parsed = parse_llama_chat_text(row["text"])
        if parsed is None:
            return None
        prompt, solution = parsed
        answer = extract_explicit_answer(solution)
        return prompt, solution, answer

    prompt = clean_text(
        get_first(row, ["problem", "question", "prompt", "input", "query", "instruction"])
    )

    solution = clean_text(
        get_first(row, ["solution", "output", "response", "completion", "assistant", "rationale", "reasoning"])
    )

    raw_answer = get_first(row, ["answer", "final_answer", "reference", "target", "label", "correct_answer"])
    answer = extract_explicit_answer(solution, raw_answer)

    if not prompt or not solution:
        return None

    return prompt, solution, answer


def is_bad_prompt(prompt: str, solution: str, answer: str) -> bool:
    if len(prompt) < MIN_PROMPT_CHARS or len(prompt) > MAX_PROMPT_CHARS:
        return True

    if has_regex(prompt, BAD_PROMPT_PATTERNS):
        return True

    if has_regex(prompt, PROOF_PROMPT_PATTERNS):
        return True

    # New light-cleaning rule: remove game/winner/strategy prompts.
    if has_regex(prompt, BAD_GAME_OR_STRATEGY_PROMPT_PATTERNS):
        return True

    combined = prompt + "\n" + solution + "\n" + answer

    if has_regex(combined, BAD_PROMPT_PATTERNS):
        return True

    return False


def make_thinking_solution(solution: str, answer: str) -> str:
    solution = clean_text(solution)
    answer = normalize_answer(answer)

    if answer_looks_bad(answer):
        return ""

    # Remove final boxed answer from inside thinking if it exists.
    # The successful groups usually have final answer outside </think>.
    solution_no_tail = solution

    boxed = extract_last_boxed(solution)
    if boxed:
        idx = solution.rfind(BOXED)
        if idx != -1:
            solution_no_tail = solution[:idx].strip()

    solution_no_tail = clean_text(solution_no_tail)

    if not solution_no_tail:
        return ""

    words = len(solution_no_tail.split())

    if words < MIN_SOLUTION_WORDS or words > MAX_SOLUTION_WORDS:
        return ""

    if len(solution_no_tail) > MAX_SOLUTION_CHARS:
        return ""

    return f"<think>\n{solution_no_tail}\n</think>\n\n\\boxed{{{answer}}}"


def make_example(
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

    if is_bad_prompt(prompt, solution, answer):
        return None

    if answer_looks_bad(answer):
        return None

    assistant = make_thinking_solution(solution, answer)
    if not assistant:
        return None

    boxed = extract_last_boxed(assistant)

    if not boxed or answer_looks_bad(boxed):
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "source": source,
            "origin": origin,
            "answer": normalize_answer(boxed),
            "prompt_hash": prompt_hash(prompt),
            "solution_words": len(solution.split()),
        },
    }


def load_hendrycks() -> list[dict[str, Any]]:
    examples = []

    for config in HENDRYCKS_CONFIGS:
        origin = f"EleutherAI/hendrycks_math:{config}:train"
        print(f"Loading {origin}")
        ds = load_dataset("EleutherAI/hendrycks_math", config, split="train")

        before = len(examples)

        for row in ds:
            normalized = normalize_row(row)
            if normalized is None:
                continue

            prompt, solution, answer = normalized
            ex = make_example(prompt, solution, answer, "hendrycks_math", origin)
            if ex:
                examples.append(ex)

        print(f"Prepared {len(examples) - before} from {config}")

    print(f"Total hendrycks_math: {len(examples)}")
    return examples


def load_numina() -> list[dict[str, Any]]:
    origin = "AI-MO/NuminaMath-CoT:train"
    print(f"Loading {origin}")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

    examples = []
    scanned = 0

    ds = ds.shuffle(seed=SEED)

    for row in ds:
        scanned += 1

        if scanned > RANDOM_SCAN_LIMIT_NUMINA:
            break

        normalized = normalize_row(row)
        if normalized is None:
            continue

        prompt, solution, answer = normalized
        ex = make_example(prompt, solution, answer, "numina_math_cot", origin)
        if ex:
            examples.append(ex)

        if len(examples) >= TARGET_COUNTS["numina_math_cot"] * 2:
            break

    print(f"Total numina_math_cot: {len(examples)} from scanned={scanned}")
    return examples


def load_olympiad_contest() -> list[dict[str, Any]]:
    origin = "kevin009/olympiad-math-contest-llama3-20k:train"
    print(f"Loading {origin}")
    ds = load_dataset("kevin009/olympiad-math-contest-llama3-20k", split="train")

    examples = []

    for row in ds:
        normalized = normalize_row(row)
        if normalized is None:
            continue

        prompt, solution, answer = normalized
        ex = make_example(prompt, solution, answer, "olympiad_contest_20k", origin)
        if ex:
            examples.append(ex)

    print(f"Total olympiad_contest_20k: {len(examples)}")
    return examples


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []

    random.shuffle(rows)

    for row in rows:
        h = row["metadata"]["prompt_hash"]
        if h in seen:
            continue

        seen.add(h)
        output.append(row)

    return output


def sample_source(rows: list[dict[str, Any]], source: str, target: int) -> list[dict[str, Any]]:
    rows = deduplicate(rows)
    random.shuffle(rows)

    selected = rows[: min(target, len(rows))]
    print(f"Selected {len(selected)}/{len(rows)} from {source}, target={target}")

    return selected


def split_train_val(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    val = []

    sources = sorted(set(row["metadata"]["source"] for row in rows))

    for source in sources:
        source_rows = [row for row in rows if row["metadata"]["source"] == source]
        random.shuffle(source_rows)

        val_n = min(MAX_VAL_PER_SOURCE, max(1, int(len(source_rows) * VAL_FRACTION)))

        val.extend(source_rows[:val_n])
        train.extend(source_rows[val_n:])

    random.shuffle(train)
    random.shuffle(val)

    return train, val


def strip_meta(row: dict[str, Any]) -> dict[str, Any]:
    return {"messages": row["messages"]}


def print_stats(rows: list[dict[str, Any]], name: str) -> None:
    counts = {}

    for row in rows:
        src = row["metadata"]["source"]
        counts[src] = counts.get(src, 0) + 1

    print(f"\n{name}: {len(rows)}")

    for src, count in sorted(counts.items()):
        print(f"- {src}: {count}")


def preview(rows: list[dict[str, Any]], k: int = 5) -> None:
    print("\nPreview")

    for row in rows[:k]:
        print("=" * 100)
        print("SOURCE:", row["metadata"]["source"])
        print("ANSWER:", row["metadata"]["answer"])
        print("PROMPT:", row["messages"][1]["content"][:600])
        print("ASSISTANT:", row["messages"][2]["content"][:900])


def main() -> None:
    random.seed(SEED)

    print("=" * 100)
    print("Preparing math v4 strong SFT dataset, lightly cleaned")
    print("=" * 100)
    print("Sources:")
    print("- AI-MO/NuminaMath-CoT train")
    print("- EleutherAI/hendrycks_math train")
    print("- kevin009/olympiad-math-contest-llama3-20k train")
    print("Format:")
    print("<think> solution </think> + boxed final answer")
    print("Light filters:")
    print("- remove \\boxed{p} prompts")
    print("- remove games / winning strategy prompts")
    print("- remove \\text{...} answers")
    print("- remove answers longer than 120 chars")
    print("=" * 100)

    grouped = {
        "numina_math_cot": load_numina(),
        "hendrycks_math": load_hendrycks(),
        "olympiad_contest_20k": load_olympiad_contest(),
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
    print_stats(val, "VAL")
    preview(train)

    write_jsonl(TRAIN_OUT, [strip_meta(row) for row in train])
    write_jsonl(VAL_OUT, [strip_meta(row) for row in val])
    write_jsonl(TRAIN_META_OUT, train)
    write_jsonl(VAL_META_OUT, val)

    print("\nDone.")
    print(f"Train: {TRAIN_OUT}")
    print(f"Val:   {VAL_OUT}")
    print("\nCheck distribution:")
    print("grep -o '\"source\": \"[^\"]*\"' math/data/math_v4_strong_sft_train_with_meta.jsonl | sort | uniq -c")
    print("wc -l math/data/math_v4_strong_sft_train.jsonl")
    print("\nCheck bad patterns:")
    print("grep -i \"winning strategy\" math/data/math_v4_strong_sft_train_with_meta.jsonl | head")
    print("grep -i \"who will win\" math/data/math_v4_strong_sft_train_with_meta.jsonl | head")
    print("grep -i \"\\\\boxed{p}\" math/data/math_v4_strong_sft_train_with_meta.jsonl | head")
    print("grep -i \"\\\\text{\" math/data/math_v4_strong_sft_train_with_meta.jsonl | head")

    publish_dataset_version(
        "v4_strong_sft",
        {
            "train.jsonl": TRAIN_OUT,
            "validation.jsonl": VAL_OUT,
            "train_with_meta.jsonl": TRAIN_META_OUT,
            "validation_with_meta.jsonl": VAL_META_OUT,
        },
    )


if __name__ == "__main__":
    main()
