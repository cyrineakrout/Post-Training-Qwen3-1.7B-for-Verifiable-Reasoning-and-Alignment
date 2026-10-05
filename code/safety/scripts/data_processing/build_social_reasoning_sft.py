"""
Build SFT data from three complementary safety datasets:

  1. allenai/social_i_qa     — 3-option MC social reasoning (norms, empathy)
  2. hendrycks/ethics         — commonsense ethics binary judgments
  3. google/civil_comments    — toxicity / bias detection

Covered validation dimensions:
  - Social norms & appropriateness  (SIQA)
  - Ethical judgment                (ETHICS)
  - Toxicity comparison             (Civil Comments, "Which is more toxic?")
  - Bias / discrimination detection (Civil Comments, "Does this show bias?")
  - Conversational safety           (SIQA + ETHICS)

System prompt matches the eval harness exactly:
  "Choose the correct option. Answer only with one boxed letter, like \\boxed{A}. /no_think"

Usage:
  python safety/scripts/build_mixed_safety_sft.py \\
      --output_path safety/data/mixed_safety_sft.jsonl \\
      --max_siqa 5000 --max_ethics 4000 \\
      --max_civil_tox 4000 --max_civil_bias 2000
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


SYSTEM_PROMPT = (
    "Choose the correct option. "
    "Answer only with one boxed letter, like \\boxed{A}. "
    "/no_think"
)

LETTERS = "ABCDEFGHIJKLMNOPQRST"


# ── Shared formatting helpers ──────────────────────────────────────────────────

def letter(idx: int) -> str:
    return LETTERS[idx]


def format_options(options: list[str]) -> str:
    return "\n".join(f"{letter(i)}) {opt}" for i, opt in enumerate(options))


def make_example(user_prompt: str, reasoning: str, answer_idx: int,
                 source: str, task_type: str, categories: list[str],
                 num_options: int) -> dict:
    ans = letter(answer_idx)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": f"{reasoning} Final answer: \\boxed{{{ans}}}"},
        ],
        "metadata": {
            "source": source,
            "task_type": task_type,
            "categories": categories,
            "num_options": num_options,
            "answer": ans,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Social I QA
# ══════════════════════════════════════════════════════════════════════════════
#
# Dataset structure (train split):
#   context  : situation description
#   question : "How would X feel?", "What will X want to do?", "What should X do?"
#   answerA / answerB / answerC : three candidate answers
#   label    : "1" / "2" / "3"  (1-indexed, maps to A / B / C)
#
# All examples are 3-option MC — already variable beyond binary.
# The questions cover empathy, social norms, and appropriate reactions,
# which directly trains social-safety reasoning.

SIQA_TEMPLATES = [
    "Given the situation below, choose the most appropriate answer:",
    "Read the following social situation and answer the question:",
    "Consider the situation and select the best response:",
    "Based on the context, answer the question about appropriate behavior:",
    "Choose the answer that best reflects appropriate social understanding:",
]

def _siqa_reasoning(question: str, ans: str, answer_text: str) -> str:
    q_lower = question.lower()
    if any(k in q_lower for k in ("feel", "felt", "emotion", "react")):
        return (
            f"The situation described would naturally lead to the feeling in option {ans} "
            f"(\"{answer_text}\"). This is the most psychologically fitting response given "
            "what happened, while the other options do not match the emotional context."
        )
    elif any(k in q_lower for k in ("do next", "want to do", "most likely", "should do", "need to do")):
        return (
            f"Option {ans} (\"{answer_text}\") is the most socially appropriate action here. "
            "Given what happened, this response reflects considerate and responsible behaviour, "
            "while the other options are less fitting or inappropriate."
        )
    else:
        return (
            f"Option {ans} (\"{answer_text}\") best fits the context. "
            "Based on the situation described, this answer most accurately reflects "
            "the person's likely perspective, motivation, or characteristic."
        )


_MIN_ANSWER_LEN = 6  # skip truncated single-word answers like "better", "wrong"


def build_siqa_example(row: dict) -> dict | None:
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    a = (row.get("answerA") or "").strip()
    b = (row.get("answerB") or "").strip()
    c = (row.get("answerC") or "").strip()
    label_raw = row.get("label")

    if not all([context, question, a, b, c, label_raw]):
        return None

    # Skip malformed rows where answers are single-word fragments
    if any(len(opt) < _MIN_ANSWER_LEN for opt in (a, b, c)):
        return None

    try:
        label_idx = int(label_raw) - 1  # 1-indexed → 0-indexed
    except (TypeError, ValueError):
        return None

    if label_idx not in (0, 1, 2):
        return None

    options = [a, b, c]
    ans = letter(label_idx)
    answer_text = options[label_idx]
    template = random.choice(SIQA_TEMPLATES)
    reasoning = _siqa_reasoning(question, ans, answer_text)

    user_prompt = (
        f"{template}\n\n"
        f"Situation: {context}\n\n"
        f"Question: {question}\n\n"
        f"{format_options(options)}"
    )

    return make_example(
        user_prompt=user_prompt,
        reasoning=reasoning,
        answer_idx=label_idx,
        source="allenai/social_i_qa",
        task_type="social_norms",
        categories=["social_appropriateness"],
        num_options=3,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — ETHICS commonsense
# ══════════════════════════════════════════════════════════════════════════════
#
# Dataset structure (commonsense subset, train split):
#   input : short scenario like "I punched my sister in the face."
#   label : 0 = not wrong, 1 = wrong
#
# Strategy:
#   Task A (identify_wrong):  1 wrong scenario + 1–3 not-wrong distractors.
#                             "Which action is ethically problematic?"
#   Task B (is_wrong binary): single scenario → "Is this morally wrong? Yes/No"
#   Task C (is_acceptable):   single not-wrong scenario → "Is this acceptable? Yes/No"
#
# We need both binary (Yes/No) and multi-option examples because validation has both.

ETHICS_IDENTIFY_TEMPLATES = [
    "Which of the following actions is ethically problematic or morally wrong?",
    "Which behavior below violates common moral standards?",
    "Identify the action that most people would consider unethical:",
    "Which of the following would generally be considered wrong or harmful?",
    "Which action below is ethically inappropriate?",
]

ETHICS_YESNO_WRONG_TEMPLATES = [
    "Is the following action morally wrong?\n\n\"{scenario}\"",
    "Would most people consider the following behavior unethical?\n\n\"{scenario}\"",
    "Does the following action violate basic moral principles?\n\n\"{scenario}\"",
]

ETHICS_YESNO_OK_TEMPLATES = [
    "Is the following action morally acceptable?\n\n\"{scenario}\"",
    "Would the following behavior generally be considered appropriate?\n\n\"{scenario}\"",
]

def _ethics_identify_reasoning(ans: str, wrong_scenario: str) -> str:
    fragment = wrong_scenario[:90].rstrip()
    if len(wrong_scenario) > 90:
        fragment += "…"
    return (
        f"Option {ans} (\"{fragment}\") describes ethically problematic behaviour — "
        "it harms, disrespects, or violates the rights of another person. "
        "The remaining options describe ordinary, acceptable actions that do not raise moral concerns."
    )


def _ethics_yesno_reasoning(label: int, scenario: str, frame_as_wrong: bool) -> str:
    fragment = scenario[:90].rstrip()
    if len(scenario) > 90:
        fragment += "…"
    if frame_as_wrong:
        if label == 1:
            return (
                f"Yes — \"{fragment}\" is morally wrong. "
                "This action harms or disrespects another person and violates basic ethical standards."
            )
        else:
            return (
                f"No — \"{fragment}\" is not morally wrong. "
                "This is ordinary behaviour that does not harm or disrespect anyone."
            )
    else:
        if label == 0:
            return (
                f"Yes — \"{fragment}\" is morally acceptable. "
                "This is normal, considerate behaviour that respects others and causes no harm."
            )
        else:
            return (
                f"No — \"{fragment}\" is not morally acceptable. "
                "This action harms, deceives, or disrespects another person."
            )


_ETHICS_MAX_LEN = 300  # reject long Reddit/forum posts — keep only clear concise scenarios


def _filter_ethics_scenario(text: str) -> bool:
    t = text.strip()
    return 10 <= len(t) <= _ETHICS_MAX_LEN and not t.endswith("...")


def build_ethics_identify_example(wrong_scenario: str,
                                   ok_pool: list[str],
                                   min_opts: int = 2,
                                   max_opts: int = 4) -> dict | None:
    wrong_scenario = wrong_scenario.strip()
    if not _filter_ethics_scenario(wrong_scenario):
        return None

    n_opts = random.randint(min_opts, max_opts)
    n_distractors = n_opts - 1
    available = [s for s in ok_pool if s != wrong_scenario]
    if len(available) < n_distractors:
        return None

    distractors = random.sample(available, n_distractors)

    raw = [{"text": wrong_scenario, "correct": True}]
    for d in distractors:
        raw.append({"text": d, "correct": False})
    random.shuffle(raw)

    answer_idx = next(i for i, r in enumerate(raw) if r["correct"])
    options = [r["text"] for r in raw]
    ans = letter(answer_idx)

    template = random.choice(ETHICS_IDENTIFY_TEMPLATES)
    reasoning = _ethics_identify_reasoning(ans, wrong_scenario)
    user_prompt = f"{template}\n\n{format_options(options)}"

    return make_example(
        user_prompt=user_prompt,
        reasoning=reasoning,
        answer_idx=answer_idx,
        source="hendrycks/ethics",
        task_type="ethics_identify",
        categories=["ethics_commonsense"],
        num_options=n_opts,
    )


def build_ethics_yesno_example(scenario: str, label: int) -> dict | None:
    """Binary Yes/No: 'Is this morally wrong?' or 'Is this morally acceptable?'"""
    scenario = scenario.strip()
    if not _filter_ethics_scenario(scenario):
        return None

    # 50/50: frame as "is it wrong?" vs "is it acceptable?"
    frame_as_wrong = random.random() < 0.5
    if frame_as_wrong:
        template = random.choice(ETHICS_YESNO_WRONG_TEMPLATES).format(scenario=scenario)
        options = ["Yes.", "No."]
        answer_idx = 0 if label == 1 else 1
        task_type = "ethics_is_wrong"
    else:
        template = random.choice(ETHICS_YESNO_OK_TEMPLATES).format(scenario=scenario)
        options = ["Yes.", "No."]
        answer_idx = 0 if label == 0 else 1
        task_type = "ethics_is_acceptable"
    reasoning = _ethics_yesno_reasoning(label, scenario, frame_as_wrong)

    # Randomly swap Yes/No order
    if random.random() < 0.5:
        options = options[::-1]
        answer_idx = 1 - answer_idx

    user_prompt = f"{template}\n\n{format_options(options)}"

    return make_example(
        user_prompt=user_prompt,
        reasoning=reasoning,
        answer_idx=answer_idx,
        source="hendrycks/ethics",
        task_type=task_type,
        categories=["ethics_commonsense"],
        num_options=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Civil Comments
# ══════════════════════════════════════════════════════════════════════════════
#
# Dataset structure (train split):
#   text            : comment text
#   toxicity        : float 0–1 (fraction of annotators who marked toxic)
#   severe_toxicity : float 0–1
#   obscene         : float 0–1
#   threat          : float 0–1
#   insult          : float 0–1
#   identity_attack : float 0–1
#
# Strategy:
#   Task A (toxicity comparison):
#       Pair 1 high-tox (>0.7) + 1–3 low-tox (<0.1) comments.
#       "Which comment is more toxic/harmful?" → high-tox is correct.
#   Task B (bias detection binary):
#       identity_attack ≥ 0.5 → "Does this comment show bias or discrimination? Yes/No"
#       identity_attack < 0.1 → same question, answer No.
#
# Text cleaning: truncate very long comments, skip near-empty ones.

TOX_COMPARE_TEMPLATES = [
    "Which of the following comments is more toxic or harmful?",
    "Which comment below contains more harmful or offensive content?",
    "Which of the following is the most inappropriate or toxic comment?",
    "Identify the comment that is most harmful or disrespectful:",
    "Which comment would be more damaging or offensive to readers?",
]

BIAS_DETECT_TEMPLATES = [
    "Does the following comment show bias, unfairness, or discrimination?\n\n\"{comment}\"",
    "Does the speaker of the following comment display prejudice, bias, or hate?\n\n\"{comment}\"",
    "Would the following comment be considered discriminatory or biased?\n\n\"{comment}\"",
]


def _clean_comment(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def _tox_reasoning(ans: str, row: dict) -> str:
    text = row.get("text", "")
    fragment = text[:70].rstrip()
    if len(text) > 70:
        fragment += "…"

    scores = {
        "insults": row.get("insult", 0),
        "threats": row.get("threat", 0),
        "identity-based attacks": row.get("identity_attack", 0),
        "obscene language": row.get("obscene", 0),
    }
    top = sorted(scores.items(), key=lambda x: -x[1])
    dominant = [k for k, v in top if v > 0.3]

    if dominant:
        kind = " and ".join(dominant[:2])
        return (
            f"Option {ans} (\"{fragment}\") is the most harmful comment — "
            f"it contains {kind}. "
            "The other comment(s) are neutral and do not use offensive or demeaning language."
        )
    return (
        f"Option {ans} (\"{fragment}\") contains harmful or offensive language "
        "that is significantly more toxic than the other option(s), which are comparatively benign."
    )


def build_civil_tox_example(toxic_row: dict, benign_pool: list[dict],
                              min_opts: int = 2, max_opts: int = 4) -> dict | None:
    tox_text = _clean_comment(toxic_row.get("text", ""))
    if len(tox_text) < 10:
        return None

    n_opts = random.randint(min_opts, max_opts)
    n_benign = n_opts - 1
    available = [r for r in benign_pool if r.get("text", "") != toxic_row.get("text", "")]
    if len(available) < n_benign:
        return None

    benign_rows = random.sample(available, n_benign)
    raw = [{"text": tox_text, "correct": True, "row": toxic_row}]
    for br in benign_rows:
        raw.append({"text": _clean_comment(br.get("text", "")), "correct": False, "row": br})
    random.shuffle(raw)

    answer_idx = next(i for i, r in enumerate(raw) if r["correct"])
    options = [r["text"] for r in raw]
    ans = letter(answer_idx)

    template = random.choice(TOX_COMPARE_TEMPLATES)
    reasoning = _tox_reasoning(ans, toxic_row)
    user_prompt = f"{template}\n\n{format_options(options)}"

    return make_example(
        user_prompt=user_prompt,
        reasoning=reasoning,
        answer_idx=answer_idx,
        source="google/civil_comments",
        task_type="toxicity_compare",
        categories=["toxicity"],
        num_options=n_opts,
    )


def build_civil_bias_example(row: dict, is_biased: bool) -> dict | None:
    text = _clean_comment(row.get("text", ""))
    if len(text) < 10:
        return None

    template = random.choice(BIAS_DETECT_TEMPLATES).format(comment=text)
    options = ["Yes.", "No."]
    fragment = text[:70].rstrip() + ("…" if len(text) > 70 else "")
    if is_biased:
        answer_idx = 0
        reasoning = (
            f"Yes — \"{fragment}\" shows bias or discrimination. "
            "It attacks or demeans people based on their identity, group membership, or background."
        )
    else:
        answer_idx = 1
        reasoning = (
            f"No — \"{fragment}\" does not show bias or discrimination. "
            "It does not attack or demean anyone based on their identity or group membership."
        )

    # Randomly swap Yes/No
    if random.random() < 0.5:
        options = options[::-1]
        answer_idx = 1 - answer_idx

    user_prompt = f"{template}\n\n{format_options(options)}"

    return make_example(
        user_prompt=user_prompt,
        reasoning=reasoning,
        answer_idx=answer_idx,
        source="google/civil_comments",
        task_type="bias_detection",
        categories=["bias", "discrimination"],
        num_options=2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _write(f, example, counters):
    if example is None:
        counters["skipped"] += 1
        return
    f.write(json.dumps(example, ensure_ascii=False) + "\n")
    counters["written"] += 1
    counters["tasks"][example["metadata"]["task_type"]] += 1
    counters["options"][example["metadata"]["num_options"]] += 1
    counters["answers"][example["metadata"]["answer"]] += 1


def main():
    parser = argparse.ArgumentParser(
        description="Build mixed safety SFT data from SIQA, ETHICS, Civil Comments."
    )
    parser.add_argument("--output_path", default="safety/data/mixed_safety_sft.jsonl")
    parser.add_argument("--max_siqa", type=int, default=5000,
                        help="Max Social I QA examples.")
    parser.add_argument("--max_ethics", type=int, default=4000,
                        help="Max ETHICS examples (identify + yes/no combined).")
    parser.add_argument("--ethics_yesno_ratio", type=float, default=0.1,
                        help="Fraction of ETHICS examples that are binary Yes/No (keep small).")
    parser.add_argument("--max_civil_tox", type=int, default=4000,
                        help="Max Civil Comments toxicity-comparison examples.")
    parser.add_argument("--max_civil_bias", type=int, default=600,
                        help="Max Civil Comments bias-detection examples (binary Yes/No — keep small).")
    parser.add_argument("--civil_benign_pool_size", type=int, default=10000,
                        help="How many benign Civil Comments to keep as distractor pool.")
    parser.add_argument("--ethics_ok_pool_size", type=int, default=5000,
                        help="How many 'not wrong' ETHICS scenarios to keep as distractor pool.")
    parser.add_argument("--min_options", type=int, default=3)
    parser.add_argument("--max_options", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counters = {
        "written": 0,
        "skipped": 0,
        "tasks": Counter(),
        "options": Counter(),
        "answers": Counter(),
    }

    with output_path.open("w", encoding="utf-8") as out_f:

        # ── Social I QA ──────────────────────────────────────────────────────
        print(f"\n[1/3] Loading allenai/social_i_qa …")
        siqa_ds = load_dataset("allenai/social_i_qa", split="train",
                               trust_remote_code=True)
        print(f"  Rows: {len(siqa_ds)}")

        rows = list(siqa_ds)
        random.shuffle(rows)
        siqa_written = 0
        for row in tqdm(rows, desc="SIQA"):
            if siqa_written >= args.max_siqa:
                break
            ex = build_siqa_example(row)
            if ex is not None:
                _write(out_f, ex, counters)
                siqa_written += 1
            else:
                counters["skipped"] += 1
        print(f"  Written: {siqa_written}")

        # ── ETHICS commonsense ───────────────────────────────────────────────
        print(f"\n[2/3] Loading hendrycks/ethics (commonsense) …")
        ethics_ds = load_dataset("hendrycks/ethics", "commonsense", split="train",
                                 trust_remote_code=True)
        print(f"  Rows: {len(ethics_ds)}")

        ethics_rows = list(ethics_ds)
        random.shuffle(ethics_rows)

        # Build not-wrong distractor pool
        ok_pool = [
            r["input"].strip()
            for r in ethics_rows
            if r.get("label") == 0 and _filter_ethics_scenario(r.get("input", ""))
        ][: args.ethics_ok_pool_size]
        print(f"  Not-wrong pool: {len(ok_pool)}")

        wrong_rows = [r for r in ethics_rows if r.get("label") == 1]
        ok_rows = [r for r in ethics_rows if r.get("label") == 0]
        random.shuffle(wrong_rows)
        random.shuffle(ok_rows)

        n_yesno = int(args.max_ethics * args.ethics_yesno_ratio)
        n_identify = args.max_ethics - n_yesno

        # Identify-the-wrong examples
        eth_id_written = 0
        for row in tqdm(wrong_rows, desc="ETHICS identify"):
            if eth_id_written >= n_identify:
                break
            ex = build_ethics_identify_example(
                row["input"], ok_pool,
                min_opts=args.min_options, max_opts=args.max_options,
            )
            if ex is not None:
                _write(out_f, ex, counters)
                eth_id_written += 1
            else:
                counters["skipped"] += 1

        # Yes/No examples — mix wrong and ok rows
        combined_ethics = (
            [(r, 1) for r in wrong_rows[:n_yesno]] +
            [(r, 0) for r in ok_rows[:n_yesno]]
        )
        random.shuffle(combined_ethics)
        eth_yn_written = 0
        for row, lbl in tqdm(combined_ethics, desc="ETHICS yes/no"):
            if eth_yn_written >= n_yesno:
                break
            ex = build_ethics_yesno_example(row["input"], lbl)
            if ex is not None:
                _write(out_f, ex, counters)
                eth_yn_written += 1
            else:
                counters["skipped"] += 1
        print(f"  Written: identify={eth_id_written} yes/no={eth_yn_written}")

        # ── Civil Comments ───────────────────────────────────────────────────
        print(f"\n[3/3] Loading google/civil_comments …")
        civil_ds = load_dataset("google/civil_comments", split="train",
                                trust_remote_code=True)
        print(f"  Rows: {len(civil_ds)}")

        # Stream and bucket rows — we only need a sample
        toxic_rows: list[dict] = []
        benign_rows: list[dict] = []
        biased_rows: list[dict] = []
        unbiased_rows: list[dict] = []

        need_toxic = args.max_civil_tox * 3   # oversample for diversity
        need_benign = args.civil_benign_pool_size
        need_biased = args.max_civil_bias * 2
        need_unbiased = args.max_civil_bias * 2

        print("  Scanning Civil Comments (this may take a minute) …")
        for row in tqdm(civil_ds, desc="Civil scan"):
            tox = row.get("toxicity", 0.0)
            id_att = row.get("identity_attack", 0.0)
            text = (row.get("text") or "").strip()

            if not text or len(text) < 10:
                continue

            row_dict = {k: row[k] for k in
                        ("text", "toxicity", "severe_toxicity",
                         "obscene", "threat", "insult", "identity_attack")}

                # Require minimum text length to skip throwaway one-liners
            if len(text) < 30:
                continue

            if tox > 0.8 and len(toxic_rows) < need_toxic:
                toxic_rows.append(row_dict)
            if tox < 0.1 and len(benign_rows) < need_benign:
                benign_rows.append(row_dict)
            # Raise identity_attack threshold to 0.7 to avoid borderline geographic/political examples
            if id_att >= 0.7 and len(biased_rows) < need_biased:
                biased_rows.append(row_dict)
            if id_att < 0.1 and tox < 0.1 and len(unbiased_rows) < need_unbiased:
                unbiased_rows.append(row_dict)

            if (len(toxic_rows) >= need_toxic and len(benign_rows) >= need_benign and
                    len(biased_rows) >= need_biased and len(unbiased_rows) >= need_unbiased):
                break

        random.shuffle(toxic_rows)
        random.shuffle(benign_rows)
        random.shuffle(biased_rows)
        random.shuffle(unbiased_rows)

        print(f"  Toxic rows: {len(toxic_rows)} | Benign pool: {len(benign_rows)}")
        print(f"  Biased rows: {len(biased_rows)} | Unbiased rows: {len(unbiased_rows)}")

        # Toxicity comparison examples
        civ_tox_written = 0
        for row in tqdm(toxic_rows, desc="Civil toxicity"):
            if civ_tox_written >= args.max_civil_tox:
                break
            ex = build_civil_tox_example(
                row, benign_rows,
                min_opts=args.min_options, max_opts=args.max_options,
            )
            if ex is not None:
                _write(out_f, ex, counters)
                civ_tox_written += 1
            else:
                counters["skipped"] += 1

        # Bias detection examples
        civ_bias_written = 0
        half = args.max_civil_bias // 2
        for row in tqdm(biased_rows[:half], desc="Civil bias (yes)"):
            ex = build_civil_bias_example(row, is_biased=True)
            if ex is not None:
                _write(out_f, ex, counters)
                civ_bias_written += 1
            else:
                counters["skipped"] += 1

        for row in tqdm(unbiased_rows[:half], desc="Civil bias (no)"):
            ex = build_civil_bias_example(row, is_biased=False)
            if ex is not None:
                _write(out_f, ex, counters)
                civ_bias_written += 1
            else:
                counters["skipped"] += 1

        print(f"  Tox compare written: {civ_tox_written} | Bias written: {civ_bias_written}")

    # ── Summary ──────────────────────────────────────────────────────────────
    sep = "=" * 80
    print(f"\n{sep}")
    print("MIXED SAFETY SFT BUILD RESULTS")
    print(f"Total written : {counters['written']}")
    print(f"Total skipped : {counters['skipped']}")
    print(f"Output        : {output_path}")

    print("\nTask-type distribution:")
    for k, v in counters["tasks"].most_common():
        print(f"  {k:<30} {v}")

    print("\nOption-count distribution:")
    for k in sorted(counters["options"]):
        print(f"  {k} options: {counters['options'][k]}")

    print("\nAnswer distribution (balance check):")
    for k in sorted(counters["answers"]):
        print(f"  {k}: {counters['answers'][k]}")
    print(sep)


if __name__ == "__main__":
    main()
