import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

GROUP_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]


GROUP_SYSTEM_PROMPT = (
    "You are a careful assistant. "
    "For every question, put the final answer inside exactly one LaTeX \\boxed{} expression. "
    "If the question is multiple-choice, put only the option letter inside the box, for example \\boxed{C}. "
    "If the question is free-form or mathematical, put only the final expression, number, set, or requested object inside the box. "
    "Use brief reasoning only when useful. "
    "Do not write anything after the boxed answer."
)


def read_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_boxed_letter(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"\\boxed\{([A-T])\}", text)
    if m:
        return m.group(1)
    m = re.search(r"(?:answer|option|final)[^A-T]{0,20}\b([A-T])\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"\s*[A-T]\s*", text):
        return text.strip().upper()
    return None


def has_boxed(text):
    return isinstance(text, str) and "\\boxed{" in text


def get_user_assistant(row):
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None, None
    user = next((m.get("content", "") for m in messages if m.get("role") == "user"), None)
    assistant = next((m.get("content", "") for m in messages if m.get("role") == "assistant"), None)
    return user, assistant


def format_raw_mc_question(row):
    question = row.get("question") or row.get("prompt") or row.get("input")
    options = row.get("options") or row.get("choices")
    if not question or not isinstance(options, list) or not options:
        return None

    lines = [str(question).strip(), ""]
    for i, option in enumerate(options):
        label = chr(ord("A") + i)
        lines.append(f"{label}. {str(option).strip()}")
    return "\n".join(lines).strip()


def normalize_math(row):
    user, assistant = get_user_assistant(row)
    if not user or not assistant:
        return None

    assistant = assistant.strip()
    if not has_boxed(assistant):
        return None

    return {
        "messages": [
            {"role": "system", "content": GROUP_SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "domain": "math",
            "source": row.get("metadata", {}).get("source", "math_v4"),
        },
    }


def normalize_mc(row, domain):
    user, assistant = get_user_assistant(row)

    meta = row.get("metadata", {})
    answer = meta.get("answer") or meta.get("label") or meta.get("correct_answer")
    if answer is None:
        answer = row.get("answer") or row.get("label") or row.get("correct_answer")

    letter = None
    if isinstance(answer, str) and re.fullmatch(r"[A-T]", answer.strip()):
        letter = answer.strip().upper()
    if letter is None and assistant:
        letter = extract_boxed_letter(assistant)

    if letter is None:
        return None

    if not user:
        user = format_raw_mc_question(row)
    if not user:
        return None

    return {
        "messages": [
            {"role": "system", "content": GROUP_SYSTEM_PROMPT},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": f"\\boxed{{{letter}}}"},
        ],
        "metadata": {
            "domain": domain,
            "source": meta.get("source", domain),
            "answer": letter,
            "num_choices": meta.get("num_choices") or meta.get("num_options") or len(row.get("options") or []),
            "language": meta.get("language") or row.get("language") or row.get("lang"),
        },
    }


def sample_balanced(rows, n, seed, key_fn=None):
    if n <= 0:
        return []
    if len(rows) <= n:
        return rows

    rng = random.Random(seed)

    if key_fn is None:
        return rng.sample(rows, n)

    buckets = defaultdict(list)
    for r in rows:
        buckets[key_fn(r)].append(r)

    selected = []
    keys = list(buckets.keys())
    rng.shuffle(keys)

    per_bucket = max(1, n // max(1, len(keys)))

    for k in keys:
        bucket = buckets[k]
        rng.shuffle(bucket)
        take = min(per_bucket, len(bucket))
        selected.extend(bucket[:take])

    if len(selected) < n:
        selected_ids = set(id(x) for x in selected)
        remaining = [r for k in keys for r in buckets[k] if id(r) not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    rng.shuffle(selected)
    return selected[:n]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--math_file", default=str(GROUP_ROOT / "data/math_v4_strong_sft_train.jsonl"))
    ap.add_argument("--knowledge_file", default=str(GROUP_ROOT / "data/gk_sft_train.jsonl"))
    ap.add_argument("--multilingual_file", default=str(CODE_ROOT / "multilinguality/data/ml_raw_pool.jsonl"))
    ap.add_argument("--safety_file", default=str(GROUP_ROOT / "data/safety_combined_train.jsonl"))

    ap.add_argument("--out_train", default=str(GROUP_ROOT / "data/group_sft_final_train_rebuilt.jsonl"))
    ap.add_argument("--out_val", default=str(GROUP_ROOT / "data/group_sft_final_val_rebuilt.jsonl"))

    ap.add_argument("--n_math", type=int, default=15000)
    ap.add_argument("--n_knowledge", type=int, default=25000)
    ap.add_argument("--n_multilingual", type=int, default=10000)
    ap.add_argument("--n_safety", type=int, default=25000)
    ap.add_argument("--val_size", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    datasets = []

    raw_math = read_jsonl(args.math_file)
    math_rows = [x for x in (normalize_math(r) for r in raw_math) if x]
    math_rows = sample_balanced(math_rows, args.n_math, args.seed)
    datasets.extend(math_rows)
    print(f"math: raw={len(raw_math)} used={len(math_rows)}")

    raw_know = read_jsonl(args.knowledge_file)
    know_rows = [x for x in (normalize_mc(r, "knowledge") for r in raw_know) if x]
    know_rows = sample_balanced(
        know_rows,
        args.n_knowledge,
        args.seed,
        key_fn=lambda r: r["metadata"].get("num_choices") or "unknown",
    )
    datasets.extend(know_rows)
    print(f"knowledge: raw={len(raw_know)} used={len(know_rows)}")

    raw_multi = read_jsonl(args.multilingual_file)
    multi_rows = [x for x in (normalize_mc(r, "multilingual") for r in raw_multi) if x]
    multi_rows = sample_balanced(
        multi_rows,
        args.n_multilingual,
        args.seed,
        key_fn=lambda r: r["metadata"].get("language") or "unknown",
    )
    datasets.extend(multi_rows)
    print(f"multilingual: raw={len(raw_multi)} used={len(multi_rows)}")

    raw_safety = read_jsonl(args.safety_file)
    safety_rows = [x for x in (normalize_mc(r, "safety") for r in raw_safety) if x]
    safety_rows = sample_balanced(
        safety_rows,
        args.n_safety,
        args.seed,
        key_fn=lambda r: r["metadata"].get("num_choices") or "unknown",
    )
    datasets.extend(safety_rows)
    print(f"safety: raw={len(raw_safety)} used={len(safety_rows)}")

    rng.shuffle(datasets)

    val_size = min(args.val_size, max(1, len(datasets) // 20))
    val = datasets[:val_size]
    train = datasets[val_size:]

    write_jsonl(train, args.out_train)
    write_jsonl(val, args.out_val)

    print("=" * 80)
    print(f"Wrote train={len(train)} to {args.out_train}")
    print(f"Wrote val={len(val)} to {args.out_val}")

    print("Train domain counts:", Counter(r["metadata"]["domain"] for r in train))
    print("Val domain counts:", Counter(r["metadata"]["domain"] for r in val))

    print("\nExample:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
