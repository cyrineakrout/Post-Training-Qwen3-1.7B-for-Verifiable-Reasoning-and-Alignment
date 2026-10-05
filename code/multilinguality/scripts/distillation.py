"""
Generate distillation data using Qwen3-14B as teacher (self-consistency sampling).

For each example, N_SAMPLES traces are generated. The example is kept only if
MIN_CORRECT or more traces agree with the ground-truth answer. The first correct
trace is written as the SFT example.

Output: SFT-compatible JSONL with <think>...</think> trace followed by \\boxed{X}.
"""

from typing import Optional
import argparse
import json
import random
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

CONFIG_PATH = ML_ROOT / "config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

from multilinguality.scripts.utils import BOXED_RE, LANG_NAMES, LANG_LABELS, detect_language

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

DATA_CFG   = CFG["data"]
DIST_CFG   = CFG["distillation"]
PROMPT_CFG = CFG["prompts"]

TEACHER_MODEL  = "Qwen/Qwen3-14B"
BATCH_SIZE     = 8
N_SAMPLES      = 1
MIN_CORRECT    = 1
MAX_NEW_TOKENS = 4096
TEMPERATURE    = 0.6
LETTERS        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

TEACHER_SYSTEM = (
    "You are a multilingual expert. Think through this multiple-choice question "
    "step by step in English. End with \\boxed{LETTER} on its own line, "
    "where LETTER is the correct option."
)

TEACHER_SYSTEM_TMPL = (
    "You are a multilingual expert. Think through this multiple-choice question "
    "step by step in {lang}. Do not use English. End with \\boxed{{LETTER}} on its own line, "
    "where LETTER is the correct option."
)

STUDENT_SYSTEM = PROMPT_CFG["grpo_only_system"]


def _fmt_user(ex: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(ex["options"]))
    return f"{ex['question']}\n\nOptions:\n{opts}"


def load_examples() -> list[dict]:
    pool_file = ml_path(DIST_CFG["pool_file"])
    if not pool_file.exists():
        raise FileNotFoundError(
            f"Distillation pool not found: {pool_file}\n"
            "Run first: python scripts/prepare_data.py --path distillation_sft_grpo"
        )
    print(f"Loading distillation pool from {pool_file} ...")
    raw = []
    with open(pool_file, encoding="utf-8") as fh:
        for line in fh:
            ex = json.loads(line)
            raw.append(ex)
    from collections import Counter
    lang_counts = Counter(ex.get("lang", ex.get("language", "?")) for ex in raw)
    for lang, n in sorted(lang_counts.items()):
        print(f"  {lang}: {n} examples")
    return [
        {
            "id":           ex["id"],
            "user_content": _fmt_user(ex),
            "answer":       ex["answer"],
            "lang":         ex.get("lang", ex.get("language", "")),
        }
        for ex in raw
    ]


def make_prompt(user_content: str, lang: str, tokenizer, language_mode: str = "english") -> str:
    if language_mode == "multi":
        system = TEACHER_SYSTEM_TMPL.format(lang=LANG_NAMES.get(lang, "the question's language"))
    else:
        system = TEACHER_SYSTEM
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_batch_vllm(prompts: list[str], llm, n_samples: int) -> list[list[str]]:
    from vllm import SamplingParams
    import time
    sp = SamplingParams(n=n_samples, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=0.95)
    t0 = time.time()
    outputs = llm.generate(prompts, sp, use_tqdm=False)
    print(f"[vLLM] batch of {len(prompts)} in {time.time()-t0:.1f}s", flush=True)
    return [[o.text for o in output.outputs] for output in outputs]


def extract_boxed(text: str) -> Optional[str]:
    m = BOXED_RE.search(text)
    return m.group(1).upper() if m else None


def to_sft_record(ex: dict, response: str) -> dict:
    return {
        "_id":   ex["id"],
        "_lang": ex["lang"],
        "messages": [
            {"role": "system",    "content": STUDENT_SYSTEM},
            {"role": "user",      "content": ex["user_content"]},
            {"role": "assistant", "content": response.strip()},
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None,
                        help="Output JSONL path; defaults to distillation_train_{english|multi}.jsonl")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Generations per example.")
    parser.add_argument("--min-correct", type=int, default=MIN_CORRECT,
                        help="Min correct out of n-samples to keep an example.")
    parser.add_argument("--language", choices=["english", "multi"], default="english",
                        help="'english': teacher reasons in English (default). "
                             "'multi': teacher reasons in the question's language + langdetect filter.")
    parser.add_argument("--lang-confidence", type=float, default=0.5,
                        help="Min langdetect confidence to accept a trace's language (default 0.5).")
    args = parser.parse_args()

    n_samples   = args.n_samples
    min_correct = args.min_correct

    if args.output is None:
        base = DIST_CFG["train_file"].replace(".jsonl", "")
        args.output = f"{base}_{args.language}.jsonl"

    if args.language == "multi":
        print(f"Language mode: multilingual (langdetect filter, confidence >= {args.lang_confidence})")
    else:
        print("Language mode: English reasoning (default)")

    out_path = ml_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accepted_ids: set[str] = set()
    rejected_ids: set[str] = set()
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

    all_examples = load_examples()
    done_ids = accepted_ids | rejected_ids
    examples = [ex for ex in all_examples if ex["id"] not in done_ids]
    if not examples:
        examples = [ex for ex in all_examples if ex["id"] in rejected_ids]
        random.shuffle(examples)
        print(f"\nFirst pass complete. Retrying {len(examples)} rejected examples.")
    else:
        print(f"\nTo process: {len(examples)} unprocessed (+ {len(rejected_ids)} queued for retry)")

    from collections import Counter
    lang_counts = Counter(ex["lang"] for ex in examples)
    print("\nSamples per language to process:")
    for lang in DATA_CFG["languages"]:
        n = lang_counts.get(lang, 0)
        n_done = sum(1 for ex in all_examples if ex["lang"] == lang and ex["id"] in done_ids)
        print(f"  {lang}: {n} to process  ({n_done} already done)")

    if not examples:
        print("Nothing to do.")
        return

    from vllm import LLM
    from transformers import AutoTokenizer
    print(f"Loading teacher: {TEACHER_MODEL} (vLLM)")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
    tokenizer.padding_side = "left"
    llm = LLM(model=TEACHER_MODEL, dtype="bfloat16", trust_remote_code=True, max_model_len=8192)
    _gen = lambda prompts: generate_batch_vllm(prompts, llm, n_samples)

    kept = rejected = 0
    bs = args.batch_size
    with open(out_path, "a", encoding="utf-8") as fh:
        for start in range(0, len(examples), bs):
            batch = examples[start : start + bs]
            prompts = [make_prompt(ex["user_content"], ex["lang"], tokenizer, args.language) for ex in batch]
            responses_per_prompt = _gen(prompts)

            for ex, responses in zip(batch, responses_per_prompt):
                correct = [r for r in responses if extract_boxed(r) == ex["answer"]]
                if len(correct) < min_correct:
                    fh.write(json.dumps({"_id": ex["id"], "_kept": False}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                if args.language == "multi":
                    target_label = LANG_LABELS[ex["lang"]]
                    correct = [r for r in correct if detect_language(r, target_label, args.lang_confidence)]
                if not correct:
                    fh.write(json.dumps({"_id": ex["id"], "_kept": False}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                fh.write(json.dumps(to_sft_record(ex, correct[0]), ensure_ascii=False) + "\n")
                kept += 1

            done = start + len(batch)
            print(f"[{done}/{len(examples)}]  kept={kept}  rejected={rejected}", flush=True)

    print(f"\nKept {kept}, rejected {rejected}  →  {out_path}")


if __name__ == "__main__":
    main()
