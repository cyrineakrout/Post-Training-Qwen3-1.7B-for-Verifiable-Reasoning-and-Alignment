"""
Generate group distillation data from the prepared SFT training pool.

Reads group_sft_final_train.jsonl (output of build_group_dataset.py),
shuffles with a fixed seed, runs Qwen3-14B as teacher to generate
thinking traces, filters by domain:
  - MCQ (metadata.answer is a letter): keep if teacher produces the correct boxed letter
  - Math (metadata.domain == "math"):  keep if teacher produces any \\boxed{}
  - Other: keep all teacher responses

Resumes automatically if the output file already exists.
After all pending examples are processed, re-attempts rejected ones once.

Always uses vLLM (A100 40GB), bfloat16.

Usage (from code/group/):

  python scripts/distillation.py [--n-distill 2150] [--batch-size 8]

Key env vars:
  GROUP_DIST_IN   path to input JSONL   (default: group/data/group_sft_final_train.jsonl)
  GROUP_DIST_OUT  path to output JSONL  (default: group/data/group_distillation_train.jsonl)
  GROUP_DIST_BATCH  vLLM batch size     (default: 8)
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

GROUP_ROOT = Path(__file__).resolve().parents[1]

IN_PATH  = os.environ.get("GROUP_DIST_IN",  str(GROUP_ROOT / "data/group_sft_final_train.jsonl"))
OUT_PATH = os.environ.get("GROUP_DIST_OUT", str(GROUP_ROOT / "data/group_distillation_train.jsonl"))

TEACHER_MODEL  = "Qwen/Qwen3-14B"
N_DISTILL      = 2150
N_SAMPLES      = 1
MIN_CORRECT    = 1
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.6
SEED           = 42

BOXED_LETTER_RE = re.compile(r"\\boxed\{([A-Za-z])\}")
BOXED_ANY_RE    = re.compile(r"\\boxed\{")

TEACHER_SYSTEM = (
    "You are a knowledgeable multilingual expert. "
    "Think through the question carefully in English. "
    "For multiple-choice questions, end your response with \\boxed{LETTER} on its own line "
    "where LETTER is the correct option (A, B, C, …). "
    "For math problems, end with \\boxed{answer} on its own line. "
    "For other questions, provide a thorough and accurate response."
)


def read_jsonl(path) -> list:
    rows = []
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing: {path}")
        return rows
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def extract_user_content(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            return m["content"]
    return ""


def is_mcq(example: dict) -> bool:
    answer = example.get("metadata", {}).get("answer", "")
    return bool(answer) and bool(re.fullmatch(r"[A-T]", str(answer).strip()))


def is_math(example: dict) -> bool:
    return example.get("metadata", {}).get("domain") == "math"


def make_prompt(user_content: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": TEACHER_SYSTEM},
         {"role": "user",   "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch_vllm(prompts: list, n_samples: int, llm) -> list:
    from vllm import SamplingParams
    sp = SamplingParams(n=n_samples, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=0.95)
    t0 = time.time()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"[vLLM] batch of {len(prompts)} in {time.time()-t0:.1f}s", flush=True)
    return [[o.text for o in output.outputs] for output in outputs]


def to_sft_record(example: dict, response: str, uid: str) -> dict:
    return {
        "_id":     uid,
        "_source": example.get("metadata", {}).get("domain", "unknown"),
        "messages": [
            {"role": "system",    "content": TEACHER_SYSTEM},
            {"role": "user",      "content": extract_user_content(example["messages"])},
            {"role": "assistant", "content": response.strip()},
        ],
    }


def process_examples(examples, out_path, llm, tokenizer, batch_size):
    kept = rejected = 0
    with open(out_path, "a", encoding="utf-8") as fh:
        for start in range(0, len(examples), batch_size):
            batch   = examples[start:start + batch_size]
            prompts = [make_prompt(extract_user_content(ex["messages"]), tokenizer) for ex in batch]

            n_samples = N_SAMPLES if any(is_mcq(ex) for ex in batch) else 1
            responses_per_prompt = generate_batch_vllm(prompts, n_samples, llm)

            for ex, responses in zip(batch, responses_per_prompt):
                uid = ex["_uid"]

                if is_mcq(ex):
                    correct_letter = str(ex["metadata"]["answer"]).strip().upper()
                    correct_resps  = [
                        r for r in responses
                        if (m := BOXED_LETTER_RE.search(r)) and m.group(1).upper() == correct_letter
                    ]
                    if len(correct_resps) < MIN_CORRECT:
                        fh.write(json.dumps({"_id": uid, "_kept": False}, ensure_ascii=False) + "\n")
                        rejected += 1
                        continue
                    chosen = correct_resps[0]

                elif is_math(ex):
                    valid = [r for r in responses if BOXED_ANY_RE.search(r)]
                    if not valid:
                        fh.write(json.dumps({"_id": uid, "_kept": False}, ensure_ascii=False) + "\n")
                        rejected += 1
                        continue
                    chosen = valid[0]

                else:
                    chosen = responses[0]

                fh.write(json.dumps(to_sft_record(ex, chosen, uid), ensure_ascii=False) + "\n")
                kept += 1

            done = start + len(batch)
            print(f"[{done}/{len(examples)}]  kept={kept}  rejected={rejected}", flush=True)

    return kept, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-distill",  type=int, default=N_DISTILL,
                        help="Number of examples to attempt distilling.")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("GROUP_DIST_BATCH", "8")))
    args = parser.parse_args()

    in_path  = Path(IN_PATH)
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path}")
    rows = read_jsonl(str(in_path))
    rng  = random.Random(SEED)
    rng.shuffle(rows)
    sampled = rows[:args.n_distill]
    print(f"Sampled {len(sampled)} from {len(rows)} total (seed={SEED})")

    for i, row in enumerate(sampled):
        row["_uid"] = f"distill_{i}"

    # Resume: skip already-processed IDs
    accepted_ids: set = set()
    rejected_ids: set = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    uid = rec.get("_id")
                    if not uid:
                        continue
                    if rec.get("_kept") is False:
                        rejected_ids.add(uid)
                    else:
                        accepted_ids.add(uid)
                except json.JSONDecodeError:
                    pass
        rejected_ids -= accepted_ids
        print(f"Resuming — {len(accepted_ids)} kept, {len(rejected_ids)} rejected so far")

    done_ids = accepted_ids | rejected_ids
    pending  = [ex for ex in sampled if ex["_uid"] not in done_ids]
    retry    = [ex for ex in sampled if ex["_uid"] in rejected_ids]

    if not pending and not retry:
        print("Nothing to do.")
        return

    print(f"\nTo process: {len(pending)} pending  +  {len(retry)} retries")

    print(f"\nLoading teacher model: {TEACHER_MODEL} (vLLM, bfloat16)")
    from transformers import AutoTokenizer
    from vllm import LLM
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    llm = LLM(model=TEACHER_MODEL, dtype="bfloat16", trust_remote_code=True,
               max_model_len=8192, gpu_memory_utilization=0.8)

    total_kept = total_rejected = 0

    if pending:
        k, r = process_examples(pending, out_path, llm, tokenizer, args.batch_size)
        total_kept += k
        total_rejected += r
        print(f"\nFirst pass done — kept={k}  rejected={r}")

    if retry:
        print(f"\nRetrying {len(retry)} previously rejected examples...")
        k, r = process_examples(retry, out_path, llm, tokenizer, args.batch_size)
        total_kept += k
        total_rejected += r
        print(f"\nRetry pass done — kept={k}  rejected={r}")

    print(f"\nTotal kept={total_kept}  rejected={total_rejected}  →  {out_path}")


if __name__ == "__main__":
    main()
