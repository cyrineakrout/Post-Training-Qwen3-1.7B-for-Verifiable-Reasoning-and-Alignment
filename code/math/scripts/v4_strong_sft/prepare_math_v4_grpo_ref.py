import argparse
import json
import random
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_DIR / "math" / "scripts"))

from hf_dataset_utils import download_dataset_version_file, publish_dataset_version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output", default="math/data/math_v4_grpo_ref_train_5000.jsonl")
    parser.add_argument("--max_examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--prefer_hard", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    input_path = args.input or download_dataset_version_file("v4_strong_sft", "train_with_meta.jsonl")

    rows = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta = row.get("metadata", {})
            messages = row.get("messages", [])

            answer = meta.get("answer", "")
            source = meta.get("source", "unknown")

            if not answer:
                continue

            # GRPO needs only system + user. The model generates assistant output.
            prompt_messages = [m for m in messages if m.get("role") in {"system", "user"}]

            if len(prompt_messages) < 2:
                continue

            user_text = prompt_messages[-1].get("content", "")

            # Keep first GRPO run memory-safe.
            if len(user_text) > 3500:
                continue

            rows.append({
                "prompt": prompt_messages,
                "answer": answer,
                "source": source,
            })

    if args.prefer_hard:
        hard_sources = {"hendrycks_math", "olympiad_contest_20k"}
        hard = [r for r in rows if r["source"] in hard_sources]
        easy = [r for r in rows if r["source"] not in hard_sources]
        random.shuffle(hard)
        random.shuffle(easy)
        rows = hard + easy
    else:
        random.shuffle(rows)

    rows = rows[: args.max_examples]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1

    print(f"Wrote {len(rows)} rows to {args.output}")
    for src, n in sorted(counts.items()):
        print(f"- {src}: {n}")

    publish_dataset_version(
        "v4_strong_sft",
        {"grpo_ref.jsonl": args.output},
        commit_message="Publish math dataset v4 GRPO reference",
    )


if __name__ == "__main__":
    main()
