"""
Shared multilingual data sources.

Each loader returns list[dict] with keys:
    id       — stable MD5 identifier for deduplication and resume
    question — question text (includes passage for reading-comprehension questions)
    options  — list of clean option strings, no letter prefix
    answer   — uppercase letter ("A", "B", …)

Add new sources by adding a loader below and registering it in load_all().
"""

from typing import Optional
import csv
import hashlib
import io
import json
import random
import re
import zipfile

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

LETTERS = "ABCDEFGHIJKLMNOPQRST"  # A–T, 20 options max per README
_LETTER_SET = set(LETTERS)

_GLOBAL_MMLU_LANG = {"IT_IT": "it", "ES_LA": "es", "ZH_CN": "zh", "HI_IN": "hi", "RU_RU": "ru"}
_M3EXAM_LANG      = {"IT_IT": "italian", "ZH_CN": "chinese"}
_EXAMS_LANG       = {"IT_IT": "Italian", "ES_LA": "Spanish"}

_OPT_PREFIX = re.compile(r"^\s*[\(\[]?[A-Za-z][\)\]\.]?\s*")


def _eid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _strip_prefix(text: str) -> str:
    return _OPT_PREFIX.sub("", text).strip()


# ── loaders ───────────────────────────────────────────────────────────────────

def load_global_mmlu(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Global MMLU dev split — culturally diverse questions across languages."""
    parts = []
    for lang in languages:
        gl = _GLOBAL_MMLU_LANG.get(lang)
        if not gl:
            continue
        ds = load_dataset("CohereForAI/Global-MMLU", gl, split="dev")
        parts.append(ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds)))))
    if not parts:
        return []
    combined = concatenate_datasets(parts)
    out = []
    for row in combined:
        a = row["answer"].strip().upper()
        q = row["question"].strip()
        if a not in _LETTER_SET or not q:
            continue
        options = [row[f"option_{c}"].strip() for c in "abcd"]
        out.append({"id": _eid(q + a), "question": q, "options": options, "answer": a})
    return out


def load_m3exam(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Italian and Chinese only — M3Exam has no Spanish, Russian, or Hindi."""
    parts = []
    for lang in languages:
        m3 = _M3EXAM_LANG.get(lang)
        if not m3:
            continue
        ds = load_dataset("chiayewken/M3Exam", m3, split="dev")
        parts.append(ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds)))))
    if not parts:
        return []
    combined = concatenate_datasets(parts)
    out = []
    for row in combined:
        a = (row.get("answer_text") or "").strip().upper()
        q = (row.get("question_text") or "").strip()
        bg = (row.get("background") or "").strip()
        opts_raw = row.get("options") or []
        if not isinstance(opts_raw, list) or not q or a not in _LETTER_SET:
            continue
        options = [_strip_prefix(o) for o in opts_raw]
        if ord(a) - ord("A") >= len(options):
            continue
        full_q = f"{bg}\n\n{q}" if bg else q
        out.append({"id": _eid(full_q + a), "question": full_q, "options": options, "answer": a})
    return out


# Exact 67-subject list from the official CMMLU README / zip (dev/).
_CMMLU_SUBJECTS = [
    "agronomy", "anatomy", "ancient_chinese", "arts", "astronomy",
    "business_ethics", "chinese_civil_service_exam", "chinese_driving_rule",
    "chinese_food_culture", "chinese_foreign_policy", "chinese_history",
    "chinese_literature", "chinese_teacher_qualification", "clinical_knowledge",
    "college_actuarial_science", "college_education", "college_engineering_hydrology",
    "college_law", "college_mathematics", "college_medical_statistics", "college_medicine",
    "computer_science", "computer_security", "conceptual_physics",
    "construction_project_management", "economics", "education",
    "electrical_engineering", "elementary_chinese", "elementary_commonsense",
    "elementary_information_and_technology", "elementary_mathematics",
    "ethnology", "food_science", "genetics", "global_facts",
    "high_school_biology", "high_school_chemistry", "high_school_geography",
    "high_school_mathematics", "high_school_physics", "high_school_politics",
    "human_sexuality", "international_law", "journalism", "jurisprudence",
    "legal_and_moral_basis", "logical", "machine_learning", "management",
    "marketing", "marxist_theory", "modern_chinese", "nutrition", "philosophy",
    "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_study", "sociology",
    "sports_science", "traditional_chinese_medicine", "virology",
    "world_history", "world_religions",
]

_cmmlu_zip_path:Optional[ str] = None


def _get_cmmlu_zip() -> str:
    global _cmmlu_zip_path
    if _cmmlu_zip_path is None:
        _cmmlu_zip_path = hf_hub_download(
            repo_id="haonan-li/cmmlu",
            filename="cmmlu_v1_0_1.zip",
            repo_type="dataset",
        )
    return _cmmlu_zip_path


def load_cmmlu(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """CMMLU — Chinese cultural and professional knowledge (ACL 2024). ZH only.
    Loads from the zip (haonan-li/cmmlu uses a legacy script incompatible with
    recent datasets versions). CSV columns: Question, A, B, C, D, Answer.
    """
    if "ZH_CN" not in languages:
        return []
    per_subject = max(1, max_per_lang // len(_CMMLU_SUBJECTS))
    rng = random.Random(seed)
    seen: set[str] = set()
    out = []
    try:
        zip_path = _get_cmmlu_zip()
    except Exception as e:
        print(f"  [CMMLU] Could not download zip: {e}")
        return []
    with zipfile.ZipFile(zip_path) as zf:
        for subj in _CMMLU_SUBJECTS:
            entry = f"dev/{subj}.csv"
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
                eid = _eid(q + a)
                if eid not in seen:
                    seen.add(eid)
                    out.append({"id": eid, "question": q, "options": options, "answer": a})
    return out


def load_italic(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """ITALIC — Italian cultural knowledge benchmark (NAACL 2025). IT only.
    answer field is an integer index into options.
    """
    if "IT_IT" not in languages:
        return []
    try:
        ds = load_dataset("Crisp-Unimib/ITALIC", split="train")
    except Exception as e:
        print(f"  [ITALIC] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
    out = []
    for row in ds:
        q = (row.get("question") or "").strip()
        ans_idx = row.get("answer")
        opts_raw = row.get("options") or []
        if not isinstance(opts_raw, list) or not q or not isinstance(ans_idx, int):
            continue
        options = [str(o).strip() for o in opts_raw]
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        answer = LETTERS[ans_idx]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out


def load_exams(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """EXAMS — Multilingual high-school MCQ from national exams (EMNLP 2020). IT + ES."""
    target_langs = {lang: _EXAMS_LANG[lang] for lang in languages if lang in _EXAMS_LANG}
    if not target_langs:
        return []
    try:
        ds = load_dataset("mhardalov/exams", "multilingual", split="train")
    except Exception as e:
        print(f"  [EXAMS] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed)
    target_lang_set = set(target_langs.values())
    per_lang_count: dict[str, int] = {v: 0 for v in target_lang_set}
    out = []
    for row in ds:
        info = row.get("info") or {}
        lang = (info.get("language") or "").strip()
        if lang not in per_lang_count or per_lang_count[lang] >= max_per_lang:
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
        per_lang_count[lang] += 1
        out.append({"id": _eid(q + a), "question": q, "options": options, "answer": a})
        if all(v >= max_per_lang for v in per_lang_count.values()):
            break
    return out


def load_ceval(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """C-Eval — Chinese exam benchmark (NeurIPS 2023). ZH only. Dev split."""
    if "ZH_CN" not in languages:
        return []
    try:
        subjects = get_dataset_config_names("ceval/ceval-exam")
    except Exception as e:
        print(f"  [C-Eval] Could not list configs: {e}")
        return []
    per_subject = max(1, max_per_lang // max(1, len(subjects)))
    seen: set[str] = set()
    out = []
    for subj in subjects:
        try:
            ds = load_dataset("ceval/ceval-exam", subj, split="dev")
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
            eid = _eid(q + a)
            if eid not in seen:
                seen.add(eid)
                out.append({"id": eid, "question": q, "options": options, "answer": a})
    return out




# Subjects filtered to civics, culture, regional knowledge, and professional licensing.
# Excludes pure science/math (abstract_algebra, college_physics, etc.)
_MMLU_HINDI_SUBJECTS = [
    "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
    "econometrics", "global_facts",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "high_school_psychology",
    "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "management", "marketing", "medical_genetics", "miscellaneous",
    "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
    "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_cultures", "world_religions",
]


def load_mmlu_hindi(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """MMLU translated to Hindi (FreedomIntelligence/MMLU_Hindi). HI only.
    CSV files: mmlu_Hindi/dev/{subject}_dev.csv — no header row,
    columns: question, A, B, C, D, answer_letter.
    Covers civics, culture, professional licensing across selected MMLU subjects.
    """
    if "HI_IN" not in languages:
        return []
    rng = random.Random(seed)
    seen: set[str] = set()
    out = []
    for subj in _MMLU_HINDI_SUBJECTS:
        filename = f"mmlu_Hindi/dev/{subj}_dev.csv"
        try:
            path = hf_hub_download(
                repo_id="FreedomIntelligence/MMLU_Hindi",
                filename=filename,
                repo_type="dataset",
            )
        except Exception:
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        rng.shuffle(rows)
        for row in rows:
            if len(row) < 6:
                continue
            q = row[0].strip()
            a = row[5].strip().upper()
            if not q or a not in _LETTER_SET:
                continue
            options = [row[i].strip() for i in range(1, 5)]
            if not any(options):
                continue
            eid = _eid(q + a)
            if eid not in seen:
                seen.add(eid)
                out.append({"id": eid, "question": q, "options": options, "answer": a})
        if len(out) >= max_per_lang:
            break
    return out[:max_per_lang]


def load_head_qa(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """HEAD-QA — Spanish professional licensing exams (ACL 2019). ES only.
    Data lives in head-qa-es-en-pdfs.zip; bypasses legacy loading script.
    JSON: data["exams"][key]["data"]; ra/qid are strings, aid is int.
    Train split only (2657 Spanish examples).
    """
    if "ES_LA" not in languages:
        return []
    try:
        zip_path = hf_hub_download(
            repo_id="dvilares/head_qa",
            filename="data/head-qa-es-en-pdfs.zip",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  [head_qa] Skipped: {e}")
        return []
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("HEAD/train_HEAD.json") as f:
            data = json.load(f)
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
            ans_idx = next(
                (i for i, a in enumerate(answers) if a.get("aid") == ra), None
            )
            if ans_idx is None or ans_idx >= len(LETTERS):
                continue
            options = [str(a.get("atext", "")).strip() for a in answers]
            answer = LETTERS[ans_idx]
            out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    random.Random(seed).shuffle(out)
    return out[:max_per_lang]


def load_indic_mmlu(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Cognitive-Lab/Indic-MMLU — MMLU translated to Indic languages. HI only.
    Dev split: 285 Hindi examples across 57 MMLU subjects (civics, culture,
    professional topics). answer is 0-indexed into translated_choices list.
    """
    if "HI_IN" not in languages:
        return []
    try:
        ds = load_dataset("Cognitive-Lab/Indic-MMLU", "hi", split="dev")
    except Exception:
        try:
            ds = load_dataset("Cognitive-Lab/Indic-MMLU", split="dev")
        except Exception as e:
            print(f"  [indic_mmlu] Skipped: {e}")
            return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
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
        answer = LETTERS[ans_idx]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out


def load_ruMMLU_pro(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """ruMMLU-Pro — Russian professional MCQ, variable 1-10 options (train split only)."""
    if "RU_RU" not in languages:
        return []
    try:
        ds = load_dataset("t-tech/ruMMLU-pro", split="train")
    except Exception as e:
        print(f"  [ruMMLU-pro] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
    out = []
    for row in ds:
        q = (row.get("question") or "").strip()
        opts_raw = row.get("options") or []
        ans_idx = row.get("answer_index")
        if not q or not isinstance(opts_raw, list) or not isinstance(ans_idx, int):
            continue
        options = [str(o).strip() for o in opts_raw]
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        answer = LETTERS[ans_idx]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out



def load_cmb(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """CMB — Chinese Medical Board exam MCQ (FreedomIntelligence/CMB).
    Downloads CMB-train-merge.json directly (loading script bypassed, same
    approach as CMMLU). Only keeps 单项选择题 (single-choice questions).
    Options stored in row['option'] dict with keys A-F; answer is a single letter.
    """
    if "ZH_CN" not in languages:
        return []
    try:
        path = hf_hub_download(
            repo_id="FreedomIntelligence/CMB",
            filename="CMB-Exam/CMB-train/CMB-train-merge.json",
            repo_type="dataset",
        )
    except Exception as e:
        print(f"  [cmb] Skipped: {e}")
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = records.get("data", [])
    random.Random(seed).shuffle(records)
    out = []
    for row in records:
        if len(out) >= max_per_lang:
            break
        if (row.get("question_type") or "") != "单项选择题":
            continue
        q = (row.get("question") or "").strip()
        a = (row.get("answer") or "").strip().upper()
        opt_dict = row.get("option") or {}
        if not q or not isinstance(opt_dict, dict) or len(a) != 1 or a not in _LETTER_SET:
            continue
        sorted_keys = sorted(k for k in opt_dict if k and opt_dict[k])
        options = [str(opt_dict[k]).strip() for k in sorted_keys if opt_dict[k]]
        if not options or a not in sorted_keys:
            continue
        ans_pos = sorted_keys.index(a)
        answer = LETTERS[ans_pos]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out


def load_aya_es(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Aya global exams Spanish — constitutional/administrative/EU law MCQ. ES only.
    Train split: 466 examples. answer is 1-indexed int into options list.
    """
    if "ES_LA" not in languages:
        return []
    try:
        ds = load_dataset("amayuelas/aya-global-exams-spanish", split="train")
    except Exception as e:
        print(f"  [aya_es] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
    out = []
    for row in ds:
        q = (row.get("question") or "").strip()
        opts_raw = row.get("options") or row.get("choices") or []
        ans = row.get("answer")
        if not q or not isinstance(opts_raw, list):
            continue
        options = [str(o).strip() for o in opts_raw]
        # answer may be int (1-indexed), digit string ("1"), or letter string ("A")
        if isinstance(ans, int):
            ans_idx = ans - 1
        elif isinstance(ans, str):
            s = ans.strip()
            if s.isdigit():
                ans_idx = int(s) - 1
            elif len(s) == 1 and s.upper() in _LETTER_SET:
                ans_idx = ord(s.upper()) - ord("A")
            else:
                continue
        else:
            continue
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        answer = LETTERS[ans_idx]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out


def load_medexpqa_es(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """MedExpQA Spanish — Spanish MIR medical licensing exam. ES only.
    Train split: 434 examples. options is a dict with keys "1"-"5",
    correct_option is 1-indexed int.
    """
    if "ES_LA" not in languages:
        return []
    try:
        ds = load_dataset("NLP-FBK/medexpqa-es", "train", split="train")
    except Exception as e:
        print(f"  [medexpqa_es] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
    out = []
    for row in ds:
        q = (row.get("full_question") or "").strip()
        opt_dict = row.get("options") or {}
        ans = row.get("correct_option")
        if not q or not isinstance(opt_dict, dict) or not isinstance(ans, int):
            continue
        sorted_keys = sorted(k for k in opt_dict if opt_dict.get(k))
        options = [str(opt_dict[k]).strip() for k in sorted_keys]
        ans_idx = ans - 1  # 1-indexed
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        answer = LETTERS[ans_idx]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out


def load_exams_bolivia(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Bolivian university entry exams — law, biochemistry, regional knowledge. ES only.
    Train split: 331 examples. answer is 0-indexed int.
    """
    if "ES_LA" not in languages:
        return []
    try:
        ds = load_dataset("Nairaarnez/exams-spanish-Bolivia", split="train")
    except Exception as e:
        print(f"  [exams_bolivia] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
    out = []
    for row in ds:
        q = (row.get("question") or "").strip()
        opts_raw = row.get("options") or []
        ans = row.get("answer")
        if not q or not isinstance(opts_raw, list) or not isinstance(ans, int):
            continue
        options = [str(o).strip() for o in opts_raw]
        if ans < 0 or ans >= len(options):
            continue
        answer = LETTERS[ans]
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
    return out


# Peruvian ADUNI exam prep: economy (215), philosophy (163), literature (479).
# answer field: 1-indexed for economy/philosophy, 0-indexed for literature.
_PERU_EXAM_REPOS = [
    ("johanobandoc/economy_spanish_exam",            True),
    ("johanobandoc/philosophy_spanish_exam",          True),
    ("johanobandoc/literature_languaje_spanish_exam", False),
]


def load_peru_exams(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Peruvian ADUNI exam prep — economy, philosophy, literature/culture. ES only.
    Combines three repos; answer indexing varies per repo (handled internally).
    """
    if "ES_LA" not in languages:
        return []
    rng = random.Random(seed)
    seen: set[str] = set()
    all_records = []
    for repo_id, one_indexed in _PERU_EXAM_REPOS:
        try:
            ds = load_dataset(repo_id, split="train")
        except Exception as e:
            print(f"  [peru_exams/{repo_id.split('/')[1]}] Skipped: {e}")
            continue
        for row in ds:
            q = (row.get("question") or "").strip()
            opts_raw = row.get("options") or []
            ans = row.get("answer")
            if not q or not isinstance(opts_raw, list):
                continue
            options = [str(o).strip() for o in opts_raw]
            if isinstance(ans, str):
                try:
                    ans = int(ans)
                except ValueError:
                    continue
            if not isinstance(ans, int):
                continue
            ans_idx = ans - 1 if one_indexed else ans
            if ans_idx < 0 or ans_idx >= len(options):
                continue
            answer = LETTERS[ans_idx]
            eid = _eid(q + answer)
            if eid not in seen:
                seen.add(eid)
                all_records.append({"id": eid, "question": q, "options": options, "answer": answer})
    rng.shuffle(all_records)
    return all_records[:max_per_lang]


_KALEIDOSCOPE_LANG = {"ES_LA": "es", "HI_IN": "hi", "RU_RU": "ru"}

_KALEIDOSCOPE_SUBJECTS = {
    "Economics", "Geography", "History", "Language",
    "Social Sciences", "Sociology", "Medicine", "Driving License",
}


def load_kaleidoscope(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Kaleidoscope — native real-world exam MCQ for ES, HI, RU (CohereLabs/kaleidoscope).
    Filtered to civics, culture, regional knowledge, and professional licensing subjects.
    Excluded: Mathematics, Physics, Chemistry, Engineering, Biology, Reasoning.
    answer is 0-indexed int into options list. Only train split available.
    """
    target = {lang: _KALEIDOSCOPE_LANG[lang] for lang in languages if lang in _KALEIDOSCOPE_LANG}
    if not target:
        return []
    try:
        ds = load_dataset("CohereLabs/kaleidoscope", split="train")
    except Exception as e:
        print(f"  [kaleidoscope] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed)
    per_lang_count: dict[str, int] = {iso: 0 for iso in target.values()}
    rev = {iso: lang for lang, iso in target.items()}
    out = []
    for row in ds:
        iso = (row.get("language") or "").strip()
        if iso not in per_lang_count or per_lang_count[iso] >= max_per_lang:
            continue
        if (row.get("category_en") or "") not in _KALEIDOSCOPE_SUBJECTS:
            continue
        q = (row.get("question") or "").strip()
        opts_raw = row.get("options") or []
        ans_idx = row.get("answer")
        if not q or not isinstance(opts_raw, list) or not isinstance(ans_idx, int):
            continue
        options = [str(o).strip() for o in opts_raw]
        if ans_idx < 0 or ans_idx >= len(options):
            continue
        answer = LETTERS[ans_idx]
        per_lang_count[iso] += 1
        out.append({"id": _eid(q + answer), "question": q, "options": options, "answer": answer})
        if all(v >= max_per_lang for v in per_lang_count.values()):
            break
    return out


def load_hindi_mcq(languages: list[str], max_per_lang: int, seed: int) -> list[dict]:
    """Hindi general MCQ — cmeraki/hindi_eval_general_mcq. HI only.
    Train split: 1,653 examples. Subjects: Maths, Physics, Chemistry, Biology, General.
    Columns: QUESTION, A, B, C, D, TARGET (answer letter A-D).
    """
    if "HI_IN" not in languages:
        return []
    try:
        ds = load_dataset("cmeraki/hindi_eval_general_mcq", split="train")
    except Exception as e:
        print(f"  [hindi_mcq] Skipped: {e}")
        return []
    ds = ds.shuffle(seed=seed).select(range(min(max_per_lang, len(ds))))
    out = []
    for row in ds:
        q = (row.get("QUESTION") or "").strip()
        a = (row.get("TARGET") or "").strip().upper()
        if not q or a not in _LETTER_SET:
            continue
        options = [str(row.get(c) or "").strip() for c in "ABCD"]
        if not any(options):
            continue
        out.append({"id": _eid(q + a), "question": q, "options": options, "answer": a})
    return out


# ── aggregate ─────────────────────────────────────────────────────────────────

def load_all(
    languages: list[str],
    max_per_lang: int,
    seed: int,
    limits:Optional[ dict[str, int]] = None,
) -> list[dict]:
    """Load and deduplicate examples from all sources.

    limits: per-source cap overriding max_per_lang (e.g. {"cmmlu": 200}).
    Sources missing from limits fall back to max_per_lang.
    """
    def lim(name: str) -> int:
        return limits[name] if limits and name in limits else max_per_lang

    loaders: list[tuple[str, object]] = [
        ("global_mmlu", lambda: load_global_mmlu(languages, lim("global_mmlu"), seed)),
        ("m3exam",      lambda: load_m3exam(languages, lim("m3exam"), seed)),
        ("cmmlu",       lambda: load_cmmlu(languages, lim("cmmlu"), seed)),
        ("ceval",       lambda: load_ceval(languages, lim("ceval"), seed)),
        ("italic",      lambda: load_italic(languages, lim("italic"), seed)),
        ("exams",       lambda: load_exams(languages, lim("exams"), seed)),
        ("rummlupro",   lambda: load_ruMMLU_pro(languages, lim("rummlupro"), seed)),
        ("mmlu_hindi",    lambda: load_mmlu_hindi(languages, lim("mmlu_hindi"), seed)),
        ("indic_mmlu",    lambda: load_indic_mmlu(languages, lim("indic_mmlu"), seed)),
        ("hindi_mcq",     lambda: load_hindi_mcq(languages, lim("hindi_mcq"), seed)),
        ("kaleidoscope",  lambda: load_kaleidoscope(languages, lim("kaleidoscope"), seed)),
        ("head_qa",       lambda: load_head_qa(languages, lim("head_qa"), seed)),
        ("aya_es",        lambda: load_aya_es(languages, lim("aya_es"), seed)),
        ("medexpqa_es",   lambda: load_medexpqa_es(languages, lim("medexpqa_es"), seed)),
        ("exams_bolivia", lambda: load_exams_bolivia(languages, lim("exams_bolivia"), seed)),
        ("peru_exams",    lambda: load_peru_exams(languages, lim("peru_exams"), seed)),
        ("cmb",           lambda: load_cmb(languages, lim("cmb"), seed)),
    ]
    seen: set[str] = set()
    result: list[dict] = []
    for name, loader in loaders:
        before = len(result)
        for ex in loader():
            if ex["id"] not in seen:
                seen.add(ex["id"])
                result.append(ex)
        added = len(result) - before
        if added:
            print(f"  [{name}] +{added}")
    return result
