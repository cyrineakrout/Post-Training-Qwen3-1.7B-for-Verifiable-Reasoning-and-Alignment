import json
import random
import hashlib
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import download_final_evaluation, publish_dataset_version

SEED = 42
rng = random.Random(SEED)

DATASET_NAME = "nvidia/AceReason-1.1-SFT"

TRAIN_OUT = Path("math/data/math_v7_100k_train.jsonl")
VAL_OUT = Path("math/data/math_v7_100k_val.jsonl")
MANIFEST_OUT = Path("math/data/math_v7_100k_manifest.json")

# Neutral filenames, but manifest records the actual construction.
TRAIN_TARGET = 100_000
VAL_TARGET = 5_000

# Your chosen balance.
TRAIN_BUCKET_TARGETS = {
    "short": 65_000,
    "medium": 27_000,
    "long": 8_000,
}

# Keep validation harder and more diagnostic.
VAL_BUCKET_TARGETS = {
    "short": 1_500,
    "medium": 1_750,
    "long": 1_750,
}

# Slightly different from the other group, but still close to the source quality.
# Total = 100k.
TRAIN_SOURCE_TARGETS = {
    "OpenMathReasoning": 75_000,
    "NuminaMath-CoT": 25_000,
}

VAL_SOURCE_TARGETS = {
    "OpenMathReasoning": 3_750,
    "NuminaMath-CoT": 1_250,
}

SYSTEM_PROMPT = (
    "You are a mathematical reasoning model. Solve the problem carefully. "
    "Put the final answer exactly once inside a LaTeX \\boxed{} expression."
)

# The final evaluation benchmark downloaded from the Hub is appended at runtime.
EXCLUDE_FILES = []

TARGET_SOURCES = set(TRAIN_SOURCE_TARGETS) | set(VAL_SOURCE_TARGETS)


def stable_hash(text: str) -> str:
    text = " ".join(str(text).lower().split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def extract_problem_from_any(row):
    if isinstance(row.get("messages"), list):
        for m in row["messages"]:
            if m.get("role") == "user":
                return str(m.get("content", "")).strip()

    for k in ["problem", "question", "prompt", "input", "Problem", "Question"]:
        if row.get(k):
            return str(row[k]).strip()

    return ""


def build_exclude_hashes():
    hashes = set()
    for path in EXCLUDE_FILES:
        for row in read_jsonl(path):
            problem = extract_problem_from_any(row)
            if problem:
                hashes.add(stable_hash(problem))
    return hashes


def get_source(row):
    # AceReason rows often have metadata/source-like fields.
    candidates = []

    for k in ["source", "dataset", "data_source", "origin", "subset"]:
        if row.get(k):
            candidates.append(str(row[k]))

    meta = row.get("metadata")
    if isinstance(meta, dict):
        for k in ["source", "dataset", "data_source", "origin", "subset"]:
            if meta.get(k):
                candidates.append(str(meta[k]))

    text = " ".join(candidates)

    if "OpenMathReasoning" in text:
        return "OpenMathReasoning"
    if "NuminaMath-CoT" in text or "Numina" in text:
        return "NuminaMath-CoT"

    # Fallback: sometimes source appears in id/name.
    full = json.dumps(row, ensure_ascii=False)[:2000]
    if "OpenMathReasoning" in full:
        return "OpenMathReasoning"
    if "NuminaMath" in full or "Numina" in full:
        return "NuminaMath-CoT"

    return "other"


def get_messages(row):
    if isinstance(row.get("messages"), list):
        messages = row["messages"]
        if any(m.get("role") == "user" for m in messages) and any(m.get("role") == "assistant" for m in messages):
            return messages

    problem = None
    solution = None

    for k in ["problem", "question", "prompt", "input", "Problem", "Question"]:
        if row.get(k):
            problem = str(row[k]).strip()
            break

    for k in ["solution", "answer", "response", "output", "completion", "Solution", "Answer"]:
        if row.get(k):
            solution = str(row[k]).strip()
            break

    if not problem or not solution:
        return None

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": solution},
    ]


def assistant_text(messages):
    parts = []
    for m in messages:
        if m.get("role") == "assistant":
            parts.append(str(m.get("content", "")))
    return "\n".join(parts)


def user_text(messages):
    parts = []
    for m in messages:
        if m.get("role") == "user":
            parts.append(str(m.get("content", "")))
    return "\n".join(parts)


def has_final_answer(messages):
    text = assistant_text(messages)
    # We prefer boxed/final-answer style, but do not require strict boxed because some high-quality CoT has final answer text.
    markers = ["\\boxed", "final answer", "answer is", "therefore", "Thus"]
    return any(m.lower() in text.lower() for m in markers)


def is_math_like(messages):
    txt = user_text(messages)
    if len(txt) < 10:
        return False
    math_chars = sum(ch in txt for ch in ["=", "+", "-", "\\", "^", "_", "{", "}", "(", ")", "$"])
    keywords = ["integer", "triangle", "polynomial", "probability", "number", "equation", "geometry", "prove", "find"]
    return math_chars >= 1 or any(k in txt.lower() for k in keywords)


def estimate_tokens(messages):
    # Approximate token count without loading tokenizer. Good enough for bucket construction.
    text = "\n".join(m.get("content", "") for m in messages)
    return max(1, int(len(text) / 3.7))


def bucket_of(n_tokens):
    # Designed for 16k context.
    # short: fits easily
    # medium: substantial reasoning
    # long: important 16k-style reasoning
    if n_tokens < 4096:
        return "short"
    if n_tokens < 10000:
        return "medium"
    return "long"


def normalize_row(row, source, bucket, n_tokens):
    messages = get_messages(row)
    return {
        "messages": messages,
        "metadata": {
            "source": source,
            "bucket": bucket,
            "approx_tokens": n_tokens,
            "dataset": DATASET_NAME,
            "version": "math_v7_100k",
        },
    }


def can_add(split_name, source, bucket, used_source, used_bucket, source_targets, bucket_targets):
    if source not in source_targets:
        return False
    if used_source[source] >= source_targets[source]:
        return False
    if used_bucket[bucket] >= bucket_targets[bucket]:
        return False
    return True


def main():
    final_eval_file = download_final_evaluation()
    if final_eval_file not in EXCLUDE_FILES:
        EXCLUDE_FILES.append(final_eval_file)

    exclude_hashes = build_exclude_hashes()
    print("exclude_hashes:", len(exclude_hashes))

    train_rows = []
    val_rows = []

    train_source = Counter()
    train_bucket = Counter()
    val_source = Counter()
    val_bucket = Counter()

    counts = Counter()
    seen = set(exclude_hashes)

    raw_seen_source = Counter()
    raw_seen_bucket = Counter()

    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    # We scan enough rows to fill quotas. Stop when both train and val are full.
    for row in ds:
        counts["seen"] += 1

        messages = get_messages(row)
        if messages is None:
            counts["bad_schema"] += 1
            continue

        problem = user_text(messages)
        if not problem:
            counts["no_problem"] += 1
            continue

        h = stable_hash(problem)
        if h in seen:
            counts["duplicate_or_excluded"] += 1
            continue

        source = get_source(row)
        raw_seen_source[source] += 1

        if source not in TARGET_SOURCES:
            counts["non_target_source"] += 1
            continue

        if not is_math_like(messages):
            counts["not_math_like"] += 1
            continue

        if not has_final_answer(messages):
            counts["no_final_marker"] += 1
            continue

        n_tokens = estimate_tokens(messages)
        bucket = bucket_of(n_tokens)
        raw_seen_bucket[bucket] += 1

        # Remove extremely long examples that will be truncated too hard at 16k.
        # Keep up to around 18k approximate tokens because estimate is rough.
        if n_tokens > 18000:
            counts["too_long"] += 1
            continue

        # Prefer adding to validation first for a stable hard validation set.
        if len(val_rows) < VAL_TARGET and can_add(
            "val", source, bucket, val_source, val_bucket, VAL_SOURCE_TARGETS, VAL_BUCKET_TARGETS
        ):
            seen.add(h)
            val_rows.append(normalize_row(row, source, bucket, n_tokens))
            val_source[source] += 1
            val_bucket[bucket] += 1
            continue

        if len(train_rows) < TRAIN_TARGET and can_add(
            "train", source, bucket, train_source, train_bucket, TRAIN_SOURCE_TARGETS, TRAIN_BUCKET_TARGETS
        ):
            seen.add(h)
            train_rows.append(normalize_row(row, source, bucket, n_tokens))
            train_source[source] += 1
            train_bucket[bucket] += 1
            continue

        counts["quota_full_or_not_matching"] += 1

        if len(train_rows) >= TRAIN_TARGET and len(val_rows) >= VAL_TARGET:
            break

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)

    with TRAIN_OUT.open("w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with VAL_OUT.open("w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_NAME,
        "version": "math_v7_100k",
        "seed": SEED,
        "train_output": str(TRAIN_OUT),
        "val_output": str(VAL_OUT),
        "train_written": len(train_rows),
        "val_written": len(val_rows),
        "train_source_targets": TRAIN_SOURCE_TARGETS,
        "train_bucket_targets": TRAIN_BUCKET_TARGETS,
        "val_source_targets": VAL_SOURCE_TARGETS,
        "val_bucket_targets": VAL_BUCKET_TARGETS,
        "train_source_counts": dict(train_source),
        "train_bucket_counts": dict(train_bucket),
        "val_source_counts": dict(val_source),
        "val_bucket_counts": dict(val_bucket),
        "raw_seen_source": dict(raw_seen_source),
        "raw_seen_bucket": dict(raw_seen_bucket),
        "exclude_hashes": len(exclude_hashes),
        "counts": dict(counts),
        "note": "Neutral filenames are used, but this manifest records the real sampling choices for reproducibility.",
    }

    MANIFEST_OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 100)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    publish_dataset_version(
        "v7_100k",
        {
            "train.jsonl": TRAIN_OUT,
            "validation.jsonl": VAL_OUT,
            "manifest.json": MANIFEST_OUT,
        },
    )


if __name__ == "__main__":
    main()
