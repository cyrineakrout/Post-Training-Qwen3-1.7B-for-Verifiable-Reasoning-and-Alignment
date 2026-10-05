import json
import random
import hashlib
import sys
from pathlib import Path
from collections import Counter

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import publish_final_evaluation

SEED = 2026
rng = random.Random(SEED)

OUT = Path("math/data/math_benchmark_external_v1.jsonl")
MANIFEST = Path("math/data/math_benchmark_external_v1_manifest.json")

SYSTEM_PROMPT = (
    "You are a mathematical reasoning model. Solve the problem carefully. "
    "Return the final answer in exact form whenever possible. "
    "Put the final answer inside a LaTeX \\boxed{} expression."
)

SOURCES = [
    # benchmark_name, hf_name/local_path, config, split, target_count, is_local
    ("course_math", "validation_samples/math.jsonl", None, None, 10, True),
    ("math500", "HuggingFaceH4/MATH-500", None, "test", 200, False),
    ("aime_validation", "AI-MO/aimo-validation-aime", None, "train", 100, False),
    ("aime_2024", "Maxwell-Jia/AIME_2024", None, "train", 30, False),
    ("aime_1983_2024", "di-zhang-fdu/AIME_1983_2024", None, "train", 200, False),
    ("aime_2025", "yentinglin/aime_2025", None, "train", 30, False),
]

def norm(x):
    return " ".join(str(x).lower().split())

def hash_problem(problem):
    return hashlib.sha256(norm(problem).encode("utf-8")).hexdigest()

def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def extract_problem_answer(row):
    """
    Supports:
    - MATH-500: problem, solution, answer
    - AI-MO: problem, solution, answer
    - Maxwell AIME: Problem, Solution, Answer
    - di-zhang AIME: Question, Answer
    - yentinglin AIME: problem, answer, solution
    - chat/message rows
    """
    problem = None
    solution = None
    answer = None

    # problem field
    for k in ["problem", "Problem", "question", "Question", "prompt", "input"]:
        if row.get(k) is not None:
            problem = str(row[k])
            break

    # solution field
    for k in ["solution", "Solution", "response", "output", "completion"]:
        if row.get(k) is not None:
            solution = str(row[k])
            break

    # answer field
    for k in ["answer", "Answer", "final_answer", "target"]:
        if row.get(k) is not None:
            answer = str(row[k])
            break

    # messages fallback
    if problem is None and isinstance(row.get("messages"), list):
        for m in row["messages"]:
            if m.get("role") == "user":
                problem = str(m.get("content", ""))
            elif m.get("role") == "assistant":
                solution = str(m.get("content", ""))

    # metadata answer fallback
    if answer is None and isinstance(row.get("metadata"), dict):
        for k in ["answer", "reference", "final_answer"]:
            if row["metadata"].get(k) is not None:
                answer = str(row["metadata"][k])
                break

    if answer is None and solution:
        # This is weaker, but lets evaluator extract from solution if needed.
        answer = solution

    return problem, answer, solution

def make_eval_row(problem, answer, solution, source_name, raw_source):
    if solution and "\\boxed" in solution:
        assistant = solution
    elif solution:
        assistant = solution.rstrip() + f"\n\nFinal answer: \\boxed{{{answer}}}"
    else:
        assistant = f"Final answer: \\boxed{{{answer}}}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "benchmark_eval_source": source_name,
            "raw_source": raw_source,
            "answer": answer,
            "benchmark_eval_version": "math_benchmark_external_v1",
        },
    }

def load_source(source_name, path_or_name, config, split, target, is_local):
    rows = []

    if is_local:
        raw_rows = read_jsonl(path_or_name)
    else:
        print("=" * 100)
        print("Loading:", source_name, path_or_name, split)
        if config:
            ds = load_dataset(path_or_name, config, split=split, streaming=True)
        else:
            ds = load_dataset(path_or_name, split=split, streaming=True)

        raw_rows = []
        max_scan = max(2000, target * 20)
        for i, row in enumerate(ds):
            if i >= max_scan:
                break
            raw_rows.append(row)

    rng.shuffle(raw_rows)

    for row in raw_rows:
        problem, answer, solution = extract_problem_answer(row)
        if not problem or not answer:
            continue

        rows.append(make_eval_row(
            problem=problem,
            answer=answer,
            solution=solution,
            source_name=source_name,
            raw_source=path_or_name,
        ))

        if len(rows) >= target:
            break

    return rows

def main():
    selected = []
    seen = set()
    used = Counter()
    available_after_filter = Counter()

    for source_name, path_or_name, config, split, target, is_local in SOURCES:
        try:
            rows = load_source(source_name, path_or_name, config, split, target, is_local)
        except Exception as e:
            print(f"FAILED source {source_name}: {type(e).__name__}: {e}")
            continue

        available_after_filter[source_name] = len(rows)

        for row in rows:
            problem = ""
            for m in row["messages"]:
                if m["role"] == "user":
                    problem = m["content"]
                    break

            h = hash_problem(problem)
            if h in seen:
                continue
            seen.add(h)

            selected.append(row)
            used[source_name] += 1

            if used[source_name] >= target:
                break

    rng.shuffle(selected)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(OUT),
        "seed": SEED,
        "total": len(selected),
        "used": dict(used),
        "available_after_filter": dict(available_after_filter),
        "sources": [
            {
                "benchmark_source": s[0],
                "dataset": s[1],
                "split": s[3],
                "target": s[4],
                "is_local": s[5],
            }
            for s in SOURCES
        ],
        "note": "External benchmark-style math evaluation set for checkpoint selection. Do not train on this file.",
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print("=" * 100)
    print("Wrote:", OUT)
    print("Total:", len(selected))
    print("Used:", dict(used))
    print("Manifest:", MANIFEST)

    publish_final_evaluation(OUT, MANIFEST)

if __name__ == "__main__":
    main()
