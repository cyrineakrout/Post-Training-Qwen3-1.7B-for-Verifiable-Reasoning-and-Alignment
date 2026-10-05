"""
Build a held-out multilingual evaluation set.

Test splits are cleared for use. Focus: regional knowledge, civics, culture,
and professional licensing — no pure science/math/abstract content.

Sources (loaded exhaustively, then weighted-sampled per language):
  IT: INCLUDE test, EXAMS test, Global-MMLU CS test
  ES: INCLUDE test, HEAD-QA test, EXAMS test, Global-MMLU CS test
  ZH: INCLUDE test, CMMLU test (civics/culture/licensing only), C-Eval val*, Global-MMLU CS test
  HI: INCLUDE test, MILU test, Indic-MMLU test, Global-MMLU CS test
  RU: INCLUDE test, Global-MMLU CS test

  * C-Eval test labels are withheld (live competition) — val used instead.

Target: 300 examples per language = 1500 total.
Output: multilinguality/data/eval.jsonl
"""

import csv
import io
import json
import random
import re
import sys
import zipfile
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

ML_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

def ml_path(p: str) -> Path:
    return ML_ROOT / p.removeprefix("multilinguality/")

SEED         = 99
MAX_PER_LANG = 300
LETTERS      = "ABCDEFGHIJKLMNOPQRST"
_LETTER_SET  = set(LETTERS)
_OPT_PREFIX  = re.compile(r"^\s*[\(\[]?[A-Za-z][\)\]\.]?\s*")
LANGUAGES    = ["IT_IT", "ES_LA", "ZH_CN", "HI_IN", "RU_RU"]
OUT_PATH     = "multilinguality/data/eval.jsonl"

SOURCE_SCORES = {
    "IT_IT": {"include": 9, "exams": 7, "global_mmlu_cs": 3},
    "ES_LA": {"include": 9, "head_qa": 9, "exams": 7, "global_mmlu_cs": 3},
    "ZH_CN": {"include": 9, "cmmlu": 8, "ceval": 6, "global_mmlu_cs": 3},
    "HI_IN": {"include": 9, "milu": 10, "indic_mmlu": 4, "global_mmlu_cs": 3},
    "RU_RU": {"include": 9, "global_mmlu_cs": 6},
}

_INCLUDE_CONFIGS = {
    "IT_IT": ["Italian", "it", "it-IT"],
    "ES_LA": ["Spanish", "es", "es-ES", "es-419"],
    "ZH_CN": ["Chinese", "zh", "zh-CN", "Chinese (Simplified)"],
    "HI_IN": ["Hindi", "hi", "hi-IN"],
    "RU_RU": ["Russian", "ru", "ru-RU"],
}

_GLOBAL_MMLU_LANG_EVAL = {"IT_IT": "it", "ES_LA": "es", "ZH_CN": "zh", "HI_IN": "hi", "RU_RU": "ru"}

_CMMLU_EVAL_SUBJECTS = [
    "arts", "business_ethics", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
    "college_law", "economics", "education", "ethnology", "global_facts",
    "high_school_geography", "high_school_politics", "human_sexuality",
    "international_law", "journalism", "jurisprudence", "legal_and_moral_basis",
    "management", "marketing", "marxist_theory", "modern_chinese", "nutrition",
    "philosophy", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "sociology", "sports_science", "traditional_chinese_medicine",
    "world_history", "world_religions",
]


def _strip_prefix(t): return _OPT_PREFIX.sub("", t).strip()

def fmt_prompt(question, options):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{question}\n\nOptions:\n{opts}"

def rec(prompt, answer, language, source):
    return {"prompt": prompt, "answer": answer, "language": language, "source": source}


def weighted_sample(by_source, scores, max_total, rng):
    available = {s: list(exs) for s, exs in by_source.items()
                 if exs and scores.get(s, 0) > 0}
    if not available:
        return []

    total_score = sum(scores[s] for s in available)
    targets = {s: max_total * scores[s] / total_score for s in available}

    int_targets = {s: int(t) for s, t in targets.items()}
    remainder = max_total - sum(int_targets.values())
    for s in sorted(available, key=lambda s: targets[s] - int_targets[s], reverse=True)[:remainder]:
        int_targets[s] += 1

    result, taken, shortfall = [], {}, 0
    for s in sorted(available, key=lambda s: scores.get(s, 0), reverse=True):
        rng.shuffle(available[s])
        take = min(int_targets[s], len(available[s]))
        taken[s] = take
        result.extend(available[s][:take])
        shortfall += int_targets[s] - take

    for s in sorted(available, key=lambda s: scores.get(s, 0), reverse=True):
        if shortfall <= 0:
            break
        extra = min(shortfall, len(available[s]) - taken[s])
        if extra > 0:
            result.extend(available[s][taken[s]:taken[s] + extra])
            taken[s] += extra
            shortfall -= extra

    return result


def load_include_test(languages, limit, seed):
    out = []
    for lang in languages:
        ds = None
        for config in _INCLUDE_CONFIGS.get(lang, []):
            try:
                ds = load_dataset("CohereForAI/include-base-44", config, split="test")
                print(f"  [include/{lang}] loaded config '{config}', {len(ds)} rows")
                break
            except Exception:
                continue
        if ds is None:
            print(f"  [include/{lang}] Skipped — request access at "
                  "huggingface.co/datasets/CohereForAI/include-base-44")
            continue

        opt_keys = ["option_a", "option_b", "option_c", "option_d"]
        rng = random.Random(seed)
        indices = list(range(len(ds)))
        rng.shuffle(indices)

        regional, agnostic = [], []
        for idx in indices:
            if len(regional) + len(agnostic) >= limit:
                break
            row = ds[idx]
            q = str(row.get("question") or "").strip()
            a_raw = row.get("answer")
            if isinstance(a_raw, int):
                if not 0 <= a_raw < len(LETTERS):
                    continue
                a = LETTERS[a_raw]
            else:
                a = str(a_raw or "").strip().upper()
                if not a or a not in _LETTER_SET:
                    continue
            if not q:
                continue
            options = [str(row.get(k) or "").strip() for k in opt_keys]
            if not all(options) or ord(a) - ord("A") >= len(options):
                continue
            rf = str(row.get("regional_feature") or row.get("region_feature") or "").lower()
            entry = rec(fmt_prompt(q, options), a, lang, "include")
            (agnostic if rf == "agnostic" else regional).append(entry)

        out.extend((regional + agnostic)[:limit])
    return out


def load_exams_test(languages, limit, seed):
    _map = {"IT_IT": "Italian", "ES_LA": "Spanish"}
    target = {lang: _map[lang] for lang in languages if lang in _map}
    if not target:
        return []
    try:
        ds = load_dataset("mhardalov/exams", "multilingual", split="test")
    except Exception as e:
        print(f"  [exams/test] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed)
    counts = {v: 0 for v in target.values()}
    rev = {v: k for k, v in target.items()}
    out = []
    for row in ds:
        lang = (row.get("info") or {}).get("language", "").strip()
        if lang not in counts or counts[lang] >= limit:
            continue
        q_obj = row.get("question") or {}
        q = (q_obj.get("stem") or "").strip()
        choices = q_obj.get("choices") or []
        a = (row.get("answerKey") or "").strip().upper()
        if not q or not choices or a not in _LETTER_SET:
            continue
        options = [str(c.get("text", "") if isinstance(c, dict) else c).strip() for c in choices]
        if ord(a) - ord("A") >= len(options):
            continue
        counts[lang] += 1
        out.append(rec(fmt_prompt(q, options), a, rev[lang], "exams"))
        if all(v >= limit for v in counts.values()):
            break
    return out


def load_head_qa_test(languages, limit, seed):
    if "ES_LA" not in languages:
        return []
    try:
        zip_path = hf_hub_download(
            repo_id="dvilares/head_qa",
            filename="data/head-qa-es-en-pdfs.zip",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  [head_qa/test] Skipped: {e}")
        return []
    with zipfile.ZipFile(zip_path) as zf:
        if "HEAD/test_HEAD.json" not in zf.namelist():
            return []
        with zf.open("HEAD/test_HEAD.json") as f:
            data = json.load(f)
    out = []
    for exam in (data.get("exams") or {}).values():
        for question in exam.get("data", []):
            q = (question.get("qtext") or "").strip()
            ra_raw = question.get("ra")
            answers = question.get("answers") or []
            if not q or not isinstance(answers, list) or not answers or ra_raw is None:
                continue
            try:
                ra = int(ra_raw)
            except (ValueError, TypeError):
                continue
            ans_idx = next((i for i, a in enumerate(answers) if a.get("aid") == ra), None)
            if ans_idx is None or ans_idx >= len(LETTERS):
                continue
            options = [str(a.get("atext", "")).strip() for a in answers]
            out.append(rec(fmt_prompt(q, options), LETTERS[ans_idx], "ES_LA", "head_qa"))
    random.Random(seed).shuffle(out)
    return out[:limit]


def load_milu_test(languages, limit, seed):
    if "HI_IN" not in languages:
        return []
    try:
        ds = load_dataset("ai4bharat/MILU", "Hindi", split="test")
    except Exception as e:
        print(f"  [milu/test] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        options = [ex.get(f"option{i}") for i in range(1, 5)]
        if any(o is None or str(o).strip() == "" for o in options):
            continue
        try:
            answer_idx = int(str(ex.get("target")).replace("option", "").strip()) - 1
        except (ValueError, AttributeError, TypeError):
            continue
        if not 0 <= answer_idx < 4:
            continue
        q = (ex.get("question") or "").strip()
        if not q:
            continue
        out.append(rec(fmt_prompt(q, [str(o).strip() for o in options]),
                       LETTERS[answer_idx], "HI_IN", "milu"))
    return out


def load_cmmlu_test(languages, limit, seed):
    if "ZH_CN" not in languages:
        return []
    try:
        zip_path = hf_hub_download("haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset")
    except Exception as e:
        print(f"  [cmmlu/test] Skipped: {e}")
        return []
    per_subject = max(1, limit // len(_CMMLU_EVAL_SUBJECTS))
    rng = random.Random(seed)
    seen: set[str] = set()
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for subj in _CMMLU_EVAL_SUBJECTS:
            entry = f"test/{subj}.csv"
            if entry not in zf.namelist():
                continue
            with zf.open(entry) as f:
                rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))
            rng.shuffle(rows)
            for row in rows[:per_subject]:
                q = (row.get("Question") or "").strip()
                a = (row.get("Answer") or "").strip().upper()
                if not q or a not in _LETTER_SET:
                    continue
                options = [str(row.get(c, "")).strip() for c in "ABCD"]
                if not any(options):
                    continue
                key = q + a
                if key not in seen:
                    seen.add(key)
                    out.append(rec(fmt_prompt(q, options), a, "ZH_CN", "cmmlu"))
    return out[:limit]


def load_ceval_val(languages, limit, seed):
    if "ZH_CN" not in languages:
        return []
    _CEVAL_EVAL_SUBJECTS = {
        "business_administration", "chinese_language_and_literature",
        "civil_servant", "college_economics", "college_humanities",
        "environmental_impact_assessment_engineer", "high_school_chinese",
        "high_school_geography", "high_school_history", "high_school_politics",
        "ideological_and_moral_cultivation", "law", "legal_professional",
        "logic", "modern_chinese_history", "physician", "politics",
        "professional_tour_guide", "tax_accountant", "teacher_qualification",
        "urban_and_rural_planner",
    }
    try:
        subjects = get_dataset_config_names("ceval/ceval-exam")
    except Exception as e:
        print(f"  [ceval/val] Could not list configs: {e}")
        return []
    subjects = [s for s in subjects if s in _CEVAL_EVAL_SUBJECTS]
    per_subject = max(1, limit // max(1, len(subjects)))
    seen: set[str] = set()
    out = []
    for subj in subjects:
        try:
            ds = load_dataset("ceval/ceval-exam", subj, split="val")
        except Exception:
            continue
        ds = ds.shuffle(seed=seed).select(range(min(per_subject, len(ds))))
        for row in ds:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip().upper()
            if not q or a not in _LETTER_SET:
                continue
            options = [str(row.get(c, "")).strip() for c in "ABCD"]
            if not any(options):
                continue
            key = q + a
            if key not in seen:
                seen.add(key)
                out.append(rec(fmt_prompt(q, options), a, "ZH_CN", "ceval"))
    return out[:limit]


def load_indic_mmlu_test(languages, limit, seed):
    if "HI_IN" not in languages:
        return []
    try:
        ds = load_dataset("Cognitive-Lab/Indic-MMLU", "hi", split="test")
    except Exception:
        try:
            ds = load_dataset("Cognitive-Lab/Indic-MMLU", split="test")
        except Exception as e:
            print(f"  [indic_mmlu/test] Skipped: {e}")
            return []
    ds = ds.shuffle(seed=seed).select(range(min(limit, len(ds))))
    out = []
    for row in ds:
        q = (row.get("translated_question") or row.get("question") or "").strip()
        opts_raw = row.get("translated_choices") or row.get("choices") or []
        ans_idx = row.get("answer")
        if not q or not isinstance(opts_raw, list) or not isinstance(ans_idx, int):
            continue
        options = [str(o).strip() for o in opts_raw]
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        out.append(rec(fmt_prompt(q, options), LETTERS[ans_idx], "HI_IN", "indic_mmlu"))
    return out


def load_global_mmlu_test(languages, limit, seed):
    out = []
    for lang in languages:
        gl = _GLOBAL_MMLU_LANG_EVAL.get(lang)
        if not gl:
            continue
        try:
            ds = load_dataset("CohereForAI/Global-MMLU", gl, split="test")
        except Exception as e:
            print(f"  [global_mmlu/test/{lang}] Skipped: {e}")
            continue
        ds = ds.shuffle(seed=seed)
        cs_field = next(
            (f for f in ["cultural_sensitivity_label", "is_culturally_sensitive",
                         "cs_label", "culturally_sensitive"]
             if f in (ds.column_names or [])),
            None
        )
        lang_out = []
        for row in ds:
            if cs_field and str(row.get(cs_field, "")).strip().upper() not in ("CS", "1", "TRUE"):
                continue
            a = (row.get("answer") or "").strip().upper()
            q = (row.get("question") or "").strip()
            if not q or a not in _LETTER_SET:
                continue
            options = [str(row.get(f"option_{c}", "")).strip() for c in "abcd"]
            if not all(options):
                continue
            lang_out.append(rec(fmt_prompt(q, options), a, lang, "global_mmlu_cs"))
            if len(lang_out) >= limit:
                break
        if not lang_out:
            for row in ds:
                a = (row.get("answer") or "").strip().upper()
                q = (row.get("question") or "").strip()
                if not q or a not in _LETTER_SET:
                    continue
                options = [str(row.get(f"option_{c}", "")).strip() for c in "abcd"]
                if not all(options):
                    continue
                lang_out.append(rec(fmt_prompt(q, options), a, lang, "global_mmlu_cs"))
                if len(lang_out) >= limit:
                    break
        out.extend(lang_out)
    return out


def main():
    rng = random.Random(SEED)
    limit = MAX_PER_LANG * 5

    loaders = [
        ("include",        lambda: load_include_test(LANGUAGES, limit, SEED)),
        ("exams",          lambda: load_exams_test(LANGUAGES, limit, SEED)),
        ("head_qa",        lambda: load_head_qa_test(LANGUAGES, limit, SEED)),
        ("cmmlu",          lambda: load_cmmlu_test(LANGUAGES, limit, SEED)),
        ("ceval",          lambda: load_ceval_val(LANGUAGES, limit, SEED)),
        ("milu",           lambda: load_milu_test(LANGUAGES, limit, SEED)),
        ("indic_mmlu",     lambda: load_indic_mmlu_test(LANGUAGES, limit, SEED)),
        ("global_mmlu_cs", lambda: load_global_mmlu_test(LANGUAGES, limit, SEED)),
    ]

    by_lang_source: dict[str, dict[str, list]] = {lang: {} for lang in LANGUAGES}
    seen_prompts: set[str] = set()
    for name, loader in loaders:
        print(f"\nLoading {name}...")
        examples = loader()
        print(f"  got {len(examples)} examples")
        for ex in examples:
            key = ex["prompt"][:120]
            if key in seen_prompts:
                continue
            seen_prompts.add(key)
            by_lang_source[ex["language"]].setdefault(ex["source"], []).append(ex)

    final = []
    print("\nEval set composition:")
    for lang in LANGUAGES:
        scores = SOURCE_SCORES.get(lang, {})
        sampled = weighted_sample(by_lang_source[lang], scores, MAX_PER_LANG, rng)
        by_source = {}
        for ex in sampled:
            by_source.setdefault(ex["source"], 0)
            by_source[ex["source"]] += 1
        print(f"  {lang}: {len(sampled)} — {by_source}")
        final.extend(sampled)

    rng.shuffle(final)
    out_path = ml_path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\nTotal: {len(final)} eval examples → {out_path}")


if __name__ == "__main__":
    main()
