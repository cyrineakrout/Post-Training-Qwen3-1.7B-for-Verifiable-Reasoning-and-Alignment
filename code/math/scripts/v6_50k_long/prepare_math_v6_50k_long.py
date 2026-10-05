import json
import random
import hashlib
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

OUT_TRAIN = Path("math/data/math_v6_50k_long_train.jsonl")
OUT_VAL = Path("math/data/math_v6_50k_long_val.jsonl")
MANIFEST = Path("math/data/math_v6_50k_long_manifest.json")
LOG = Path("math/data/prepare_math_v6_50k_long.log")

TARGET_TRAIN = 50000
TARGET_VAL = 1000

# Source targets inspired by the strong model strategy.
SOURCE_TRAIN_TARGETS = {
    "OpenMathReasoning": 34300,
    "NuminaMath-CoT": 15700,
}

# Mostly short traces, with a small number of medium/long traces.
TRAIN_BUCKET_TARGETS = {
    "short": 47500,
    "medium": 2000,
    "long": 500,
}

# Validation is harder/longer to help model selection.
VAL_BUCKET_TARGETS = {
    "medium": 470,
    "long": 530,
}

SYSTEM_PROMPT = (
    "You are a mathematical reasoning model. Solve the problem carefully. "
    "Use concise scratch work with only the essential intermediate steps. "
    "Return the final answer in exact form whenever possible. "
    "Put the final answer inside a LaTeX \\boxed{} expression."
)

# The final evaluation benchmark downloaded from the Hub is appended at runtime.
EXCLUDE_FILES = []

MAX_TOTAL_CHARS = 65000
MIN_ASSISTANT_CHARS = 80


def norm_text(x: str) -> str:
    return " ".join(str(x).lower().split())


def sha(x: str) -> str:
    return hashlib.sha256(norm_text(x).encode("utf-8")).hexdigest()


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def extract_messages(row):
    """
    Convert common SFT schemas into messages:
    [
      {"role": "system", ...},
      {"role": "user", ...},
      {"role": "assistant", ...}
    ]
    """

    if isinstance(row.get("messages"), list):
        msgs = []
        for m in row["messages"]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in {"system", "user", "assistant"} and content:
                msgs.append({"role": role, "content": str(content)})
        if any(m["role"] == "user" for m in msgs) and any(m["role"] == "assistant" for m in msgs):
            return msgs

    for conv_key in ["conversations", "conversation"]:
        if isinstance(row.get(conv_key), list):
            msgs = []
            for m in row[conv_key]:
                if not isinstance(m, dict):
                    continue
                role = m.get("role") or m.get("from")
                content = m.get("content") or m.get("value")
                if not content:
                    continue

                if role in {"human", "user"}:
                    role = "user"
                elif role in {"gpt", "assistant", "model"}:
                    role = "assistant"
                elif role == "system":
                    role = "system"
                else:
                    continue

                msgs.append({"role": role, "content": str(content)})

            if any(m["role"] == "user" for m in msgs) and any(m["role"] == "assistant" for m in msgs):
                return msgs

    prompt = None
    answer = None

    for key in ["problem", "question", "prompt", "input", "query"]:
        if row.get(key):
            prompt = str(row[key])
            break

    for key in ["solution", "answer", "response", "output", "completion"]:
        if row.get(key):
            answer = str(row[key])
            break

    if prompt and answer:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]

    return None


def user_text(msgs):
    return "\n".join(m["content"] for m in msgs if m["role"] == "user").strip()


def assistant_text(msgs):
    return "\n".join(m["content"] for m in msgs if m["role"] == "assistant").strip()


def source_name(row):
    # Try metadata/source fields
    for key in ["source", "dataset", "origin"]:
        if row.get(key):
            return str(row[key])

    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ["source", "dataset", "origin"]:
            if meta.get(key):
                return str(meta[key])

    # Fallback: inspect row text
    text = json.dumps(row, ensure_ascii=False)
    if "OpenMathReasoning" in text:
        return "OpenMathReasoning"
    if "NuminaMath" in text or "NuminaMath-CoT" in text:
        return "NuminaMath-CoT"

    return "unknown"


def canonical_source(src: str) -> str:
    s = src.lower()
    if "openmath" in s:
        return "OpenMathReasoning"
    if "numina" in s:
        return "NuminaMath-CoT"
    return "other"


def is_math_row(row, msgs):
    src = canonical_source(source_name(row))
    if src in {"OpenMathReasoning", "NuminaMath-CoT"}:
        return True

    text = (user_text(msgs) + "\n" + assistant_text(msgs)).lower()
    markers = [
        "\\boxed", "equation", "integer", "polynomial", "geometry",
        "triangle", "probability", "number theory", "solve", "function",
        "matrix", "sequence", "combinatorics"
    ]
    return any(m in text for m in markers)


def bucket_by_chars(msgs):
    total = len("\n".join(m["content"] for m in msgs))
    if total < 16000:
        return "short"
    if total < 45000:
        return "medium"
    return "long"


def has_final_marker(msgs):
    a = assistant_text(msgs).lower()
    return ("\\boxed" in a) or ("final answer" in a) or ("answer:" in a)


def build_exclude_hashes():
    ex = set()
    for path in EXCLUDE_FILES:
        for row in read_jsonl(path):
            msgs = extract_messages(row)
            if msgs:
                q = user_text(msgs)
            else:
                q = row.get("problem") or row.get("question") or row.get("prompt") or ""
            if q:
                ex.add(sha(q))
    return ex


def normalize_output(row, msgs, source, bucket, split):
    # Ensure system prompt exists.
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    return {
        "messages": msgs,
        "metadata": {
            "dataset": DATASET_NAME,
            "source": source,
            "bucket": bucket,
            "split": split,
            "version": "math_v6_50k_long",
        },
    }


def quota_available_train(source, bucket, source_counts, bucket_counts):
    if source not in SOURCE_TRAIN_TARGETS:
        return False
    if source_counts[source] >= SOURCE_TRAIN_TARGETS[source]:
        return False
    if bucket_counts[bucket] >= TRAIN_BUCKET_TARGETS.get(bucket, 0):
        return False
    return True


def quota_available_val(bucket, val_bucket_counts):
    return val_bucket_counts[bucket] < VAL_BUCKET_TARGETS.get(bucket, 0)


def main():
    final_eval_file = download_final_evaluation()
    if final_eval_file not in EXCLUDE_FILES:
        EXCLUDE_FILES.append(final_eval_file)

    exclude_hashes = build_exclude_hashes()
    seen = set(exclude_hashes)

    train = []
    val = []

    source_counts = Counter()
    bucket_counts = Counter()
    val_bucket_counts = Counter()
    val_source_counts = Counter()
    raw_source_counts = Counter()
    counts = Counter()

    print(f"Loading streaming dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)

    for row in ds:
        counts["seen"] += 1

        msgs = extract_messages(row)
        if not msgs:
            counts["bad_schema"] += 1
            continue

        q = user_text(msgs)
        a = assistant_text(msgs)

        if not q or not a:
            counts["missing_q_or_a"] += 1
            continue

        if len(a) < MIN_ASSISTANT_CHARS:
            counts["assistant_too_short"] += 1
            continue

        total_chars = len("\n".join(m["content"] for m in msgs))
        if total_chars > MAX_TOTAL_CHARS:
            counts["too_long"] += 1
            continue

        if not is_math_row(row, msgs):
            counts["not_math"] += 1
            continue

        if not has_final_marker(msgs):
            counts["no_final_marker"] += 1
            continue

        q_hash = sha(q)
        if q_hash in seen:
            counts["duplicate_or_excluded"] += 1
            continue

        source = canonical_source(source_name(row))
        raw_source_counts[source] += 1

        if source not in SOURCE_TRAIN_TARGETS:
            counts["non_target_source"] += 1
            continue

        bucket = bucket_by_chars(msgs)

        # Fill train first with strict source and bucket quotas.
        if len(train) < TARGET_TRAIN and quota_available_train(source, bucket, source_counts, bucket_counts):
            seen.add(q_hash)
            train.append(normalize_output(row, msgs, source, bucket, "train"))
            source_counts[source] += 1
            bucket_counts[bucket] += 1

        # Then fill validation with medium/long examples from same target sources.
        elif len(val) < TARGET_VAL and quota_available_val(bucket, val_bucket_counts):
            seen.add(q_hash)
            val.append(normalize_output(row, msgs, source, bucket, "val"))
            val_bucket_counts[bucket] += 1
            val_source_counts[source] += 1

        else:
            counts["quota_full"] += 1

        if counts["seen"] % 10000 == 0:
            print(
                f"seen={counts['seen']} train={len(train)} val={len(val)} "
                f"sources={dict(source_counts)} buckets={dict(bucket_counts)} "
                f"val_buckets={dict(val_bucket_counts)}"
            )

        if len(train) >= TARGET_TRAIN and len(val) >= TARGET_VAL:
            break

    rng.shuffle(train)
    rng.shuffle(val)

    OUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    with OUT_TRAIN.open("w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with OUT_VAL.open("w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_NAME,
        "seed": SEED,
        "train_output": str(OUT_TRAIN),
        "val_output": str(OUT_VAL),
        "train_written": len(train),
        "val_written": len(val),
        "source_train_targets": SOURCE_TRAIN_TARGETS,
        "train_bucket_targets": TRAIN_BUCKET_TARGETS,
        "val_bucket_targets": VAL_BUCKET_TARGETS,
        "source_counts": dict(source_counts),
        "bucket_counts": dict(bucket_counts),
        "val_source_counts": dict(val_source_counts),
        "val_bucket_counts": dict(val_bucket_counts),
        "raw_source_counts_seen": dict(raw_source_counts),
        "exclude_hashes": len(exclude_hashes),
        "counts": dict(counts),
    }

    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=" * 100)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    publish_dataset_version(
        "v6_50k_long",
        {
            "train.jsonl": OUT_TRAIN,
            "validation.jsonl": OUT_VAL,
            "manifest.json": MANIFEST,
        },
    )


if __name__ == "__main__":
    main()
