# Multilinguality — Reproduction Guide

Fine-tunes Qwen3-1.7B for multilingual multiple-choice reasoning across five
languages: Italian (IT), Spanish (ES), Chinese (ZH), Hindi (HI), Russian (RU).

- **CI submission**: `cs-552-2026-camykaz/multilingual_model_m3`
- **WandB project**: `cs552-mnlp-multilinguality-m3` (entity `MNLP26`)

---

## Environment

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `transformers`, `peft`, `trl`, `vllm`, `datasets`,
`huggingface_hub`, `langdetect`, `wandb`.

> **Note on vLLM:** all inference (evaluation, distillation data generation) uses
> vLLM on A100 GPUs. No CPU/V100 fallback paths exist.

---

## HuggingFace dataset repos

Pre-built datasets are hosted on HuggingFace at:
- **Training data**: `mmsellam/multilinguality-m3-data`
- **Eval data**: `mmsellam/multilinguality-m3-eval`
- **Models**: `mmsellam/multilingual_model`

```bash
# Datasets
python scripts/download_data.py              # training + eval
python scripts/download_data.py --eval-only  # only eval.jsonl

# Models
python scripts/download_models.py                             # all models
python scripts/download_models.py --model sft_grpo_ton_merged # one model
```

> If you prefer to build the data locally instead of downloading, skip this step
> and run `prepare_data.py` / `distillation.py` /
> `prepare_multilingual_eval.py` directly — see [Step 0–1](#step-0--prepare-training-data) below.

---

## Data sourcing

Training data is built by streaming live from HuggingFace via `scripts/data_sources.py`
— ~17 multilingual source datasets (Global-MMLU, M3Exam, CMMLU, EXAMS, RuMMLU-Pro,
MILU, Indic-MMLU, ITALIC, C-Eval, HEAD-QA, etc.).

> **INCLUDE dataset**: `CohereForAI/include-base-44` requires accepting the terms
> on HuggingFace before use. If access is not granted the loader skips it gracefully.

---

## Evaluating the submitted model — one command

```bash
# Run from code/multilinguality/
# Uses validation_samples/multilingual.jsonl (10-question CI file)
python scripts/evaluate.py --path baseline
```

To run the submitted fine-tuned model instead:
```bash
python scripts/evaluate.py --path distillation_sft_grpo
```

> The `--path` argument resolves to a local merged model under `outputs/`. See
> [Model resolution](#model-resolution) for how paths map to directories.

---

## Training paths

Seven fine-tuning paths are supported. They differ in warm-start base and
data routing:

| Path | Warm-start | Data | Type | Thinking |
|---|---|---|---|---|
| `distillation` | Qwen3-1.7B base | Teacher traces (14B) | SFT | ON |
| `sft` | Qwen3-1.7B base | Format-teaching SFT | SFT | OFF |
| `distillation_sft` | `distillation_merged` | Format-teaching SFT | SFT | OFF |
| `simple_sft` | Qwen3-1.7B base | Format-teaching SFT | SFT | OFF |
| `cold_grpo` | Qwen3-1.7B base | GRPO rollouts | GRPO | OFF |
| `sft_grpo` | `sft_merged` | GRPO rollouts | GRPO | OFF |
| `distillation_sft_grpo` | `distillation_sft_merged` | GRPO rollouts | GRPO | OFF |

The two deepest pipelines are:

```
distillation → distillation_sft → distillation_sft_grpo   (teacher-seeded path)
sft          → sft_grpo                                    (cold SFT path)
```

---

## Experiment index

All commands are run from **`code/multilinguality/`**.

### Step 0 — Prepare training data

Must be run once before any training:

```bash
python scripts/prepare_data.py --path distillation_sft_grpo
```

`--path` controls what GRPO and SFT JSONL files to produce (see table above for
which paths use distillation exclusions / GRPO reservation):

```bash
# Default: exclude distillation IDs, reserve GRPO pool, rest → SFT
python scripts/prepare_data.py --path distillation_sft_grpo

# No exclusion, no GRPO reservation (used for sft, simple_sft)
python scripts/prepare_data.py --path sft

# No exclusion, reserve GRPO pool
python scripts/prepare_data.py --path cold_grpo
```

Outputs:
- `data/grpo_train.jsonl` — GRPO training set (600 examples/language)
- `data/sft_train.jsonl` — SFT training set (soft-balanced across languages)
- `data/ml_raw_pool.jsonl` — full unfiltered pool (debugging/inspection)

---

### Step 1 — (Optional) Generate distillation traces

Required only for the `distillation`, `distillation_sft`, and
`distillation_sft_grpo` paths. Uses Qwen3-14B as teacher (vLLM, A100).

```bash
# Teacher reasons in English (default, faster)
python scripts/distillation.py --language english

# Teacher reasons in the question's language + langdetect filter
python scripts/distillation.py --language multi
```

Output: `data/distillation_train_{english|multi}.jsonl`

The script is **resumable**: it skips IDs already in the output file, so it
can be interrupted and restarted safely.

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--language` | `english` | `english`: teacher in English. `multi`: teacher in question's language. |
| `--n-samples` | `1` | Generations per example (self-consistency). |
| `--min-correct` | `1` | Min correct out of `n-samples` to keep example. |
| `--batch-size` | `8` | vLLM batch size. |
| `--lang-confidence` | `0.5` | Min langdetect score for `--language multi`. |

---

### Step 2 — Train

#### SFT paths

```bash
# distillation — teacher-trace warm-start, thinking ON
python scripts/train_sft.py --path distillation --language english

# sft — cold format-teaching, thinking OFF
python scripts/train_sft.py --path sft

# distillation_sft — format-teaching on top of distillation_merged
python scripts/train_sft.py --path distillation_sft --language english

# simple_sft — greedy format-teaching from base
python scripts/train_sft.py --path simple_sft
```

**Key arguments (`train_sft.py`):**

| Argument | Default | Description |
|---|---|---|
| `--path` | required | `distillation`, `sft`, `distillation_sft`, `simple_sft` |
| `--language` | `english` | Selects `distillation_train_{english|multi}.jsonl` (distillation paths only). |
| `--thinking-off` | false | For `distillation_sft`: use the `_toff` distillation merged base. |
| `--restart` | false | Delete existing checkpoint dir and start fresh. |
| `--suffix` | `` | Append suffix to output dir (e.g. `4k` → `sft4k_lora`). |

#### GRPO paths

```bash
# cold_grpo — GRPO directly from base
python scripts/train_grpo.py --path cold_grpo

# sft_grpo — GRPO warm-started from sft_merged
python scripts/train_grpo.py --path sft_grpo

# distillation_sft_grpo — GRPO warm-started from distillation_sft_merged
python scripts/train_grpo.py --path distillation_sft_grpo --language english

# Multi-language reward variant (completion in wrong language → reward=0)
python scripts/train_grpo.py --path distillation_sft_grpo --language multi
```

**Key arguments (`train_grpo.py`):**

| Argument | Default | Description |
|---|---|---|
| `--path` | required | `cold_grpo`, `sft_grpo`, `distillation_sft_grpo` |
| `--language` | `english` | `multi`: reward=0 if completion is not in the question's language. |
| `--thinking-on` | false | Keep thinking enabled during rollouts. Default: thinking OFF. |
| `--thinking-off` | false | Use the `_toff` merged base (for GRPO warm-starts). |
| `--resume` | false | Resume from latest checkpoint in output dir. |
| `--restart` | false | Delete output dir and start fresh. |
| `--suffix` | `` | Append suffix to output dir. |

---

### Step 3 — Merge LoRA adapter

Must be done before evaluation (vLLM requires a merged full model).

```bash
# Intermediate SFT paths (no system prompt injection)
python scripts/merge_lora.py --path distillation --language english
python scripts/merge_lora.py --path sft
python scripts/merge_lora.py --path distillation_sft --language english

# Final paths (system prompt baked in, GenerationConfig saved)
python scripts/merge_lora.py --path simple_sft
python scripts/merge_lora.py --path cold_grpo
python scripts/merge_lora.py --path sft_grpo
python scripts/merge_lora.py --path distillation_sft_grpo --language english

# Thinking variants
python scripts/merge_lora.py --path sft_grpo --thinking-off   # → sft_grpo_toff_merged/
python scripts/merge_lora.py --path sft_grpo                  # → sft_grpo_ton_merged/  (default)
```

**Key arguments (`merge_lora.py`):**

| Argument | Default | Description |
|---|---|---|
| `--path` | required | Any training path. See table above. |
| `--language` | `english` | `multi`: targets `_multi` LoRA adapter. |
| `--thinking-off` | false | Bake the non-thinking chat template. Default: thinking ON. |
| `--lora-dir` | auto | Override LoRA adapter path (e.g. a specific checkpoint subdir). |
| `--final` | auto | Force system-prompt injection + GenerationConfig. Default: auto-detected. |
| `--suffix` | `` | Append suffix to dir names. |

#### Output directory naming

Merged model directories are placed under `outputs/` with this convention:

```
{path}_merged               → outputs/sft_grpo_ton_merged/      (english, thinking ON)
{path}_toff_merged          → outputs/sft_grpo_toff_merged/     (english, thinking OFF)
{path}_multi_ton_merged     → outputs/sft_grpo_multi_ton_merged/ (multi, thinking ON)
```

The `with_lang` decorator appends `_multi` before `_merged`; the `with_thinking`
decorator appends `_ton` or `_toff` also before `_merged`.

---

### Step 4 — Evaluate

#### Quick CI evaluation (10-question validation set)

```bash
# Baseline (no fine-tuning)
python scripts/evaluate.py --path baseline

# Any trained and merged path
python scripts/evaluate.py --path sft_grpo
python scripts/evaluate.py --path distillation_sft_grpo --language english
python scripts/evaluate.py --path distillation_sft_grpo --language multi
```

#### Full evaluation (held-out set, ~1500 questions)

`data/eval.jsonl` is available via `download_data.py`. If you need to rebuild it from scratch:
```bash
python scripts/prepare_multilingual_eval.py
```

```bash
python scripts/evaluate.py --path sft_grpo --full-eval
python scripts/evaluate.py --path distillation_sft_grpo --language english --full-eval
```

**Key arguments (`evaluate.py`):**

| Argument | Default | Description |
|---|---|---|
| `--path` | required | `baseline`, or any trained path name. Unknown paths resolve to `outputs/{path}_merged`. |
| `--full-eval` | false | Use `data/eval.jsonl` (~1500 examples). Default: `validation_samples/multilingual.jsonl` (10 examples). |
| `--language` | `english` | Must match `--language` used at training time. |
| `--thinking-off` | false | Evaluate the `_toff` merged variant. Default: `_ton`. |
| `--output-dir` | auto | Directory to save `generations.jsonl` and `scores.json`. |

Evaluation outputs:
- `outputs/eval_{path}_{ci|full}/generations.jsonl` — one row per example with the generated completion
- `outputs/eval_{path}_{ci|full}/scores.json` — pass@1 and per-example correctness

---

## Model resolution

`evaluate.py` (and `merge_lora.py`) resolve `--path` to a local directory as follows:

1. If `--path baseline` → uses `Qwen/Qwen3-1.7B` directly from HuggingFace.
2. If `--path` matches a known path in `config.json` → resolves to
   `outputs/{path}[_multi]_[ton|toff]_merged/` applying `--language` and `--thinking-off`.
3. Otherwise → resolves to `outputs/{path}_merged/` (custom experiment name).

---

## Full pipeline — deepest path end to end

```bash
# 0. Prepare data (stream from HF, write local JSONL)
python scripts/prepare_data.py --path distillation_sft_grpo

# 1. Generate distillation traces (Qwen3-14B teacher, vLLM)
python scripts/distillation.py --language english

# 2a. SFT — distillation warm-start (thinking ON)
python scripts/train_sft.py --path distillation --language english

# 2b. Merge distillation adapter
python scripts/merge_lora.py --path distillation --language english

# 3a. SFT — format-teaching on top of distillation_merged (thinking OFF)
python scripts/train_sft.py --path distillation_sft --language english

# 3b. Merge distillation_sft adapter
python scripts/merge_lora.py --path distillation_sft --language english

# 4a. GRPO on top of distillation_sft_merged
python scripts/train_grpo.py --path distillation_sft_grpo --language english

# 4b. Merge final adapter (injects system prompt + saves GenerationConfig)
python scripts/merge_lora.py --path distillation_sft_grpo --language english

# 5. Evaluate
python scripts/evaluate.py --path distillation_sft_grpo --language english --full-eval
```

---

## Directory layout

```
multilinguality/
├── requirements.txt
├── README.md
├── config.json                    ← all hyperparameters, path definitions
├── chat_template.jinja            ← thinking OFF chat template
├── chat_template_ton.jinja        ← thinking ON  chat template
├── data/
│   ├── grpo_train.jsonl           ← GRPO training set     (prepare_data.py)
│   ├── sft_train.jsonl            ← SFT training set      (prepare_data.py)
│   ├── ml_raw_pool.jsonl          ← full raw pool         (prepare_data.py)
│   ├── distillation_train_english.jsonl  (distillation.py)
│   ├── distillation_train_multi.jsonl    (distillation.py)
│   └── eval.jsonl                 ← held-out eval set     (prepare_multilingual_eval.py)
├── outputs/
│   ├── distillation_lora/         ← LoRA adapter
│   ├── distillation_ton_merged/   ← merged (thinking ON)
│   ├── sft_lora/
│   ├── sft_ton_merged/
│   ├── sft_toff_merged/
│   ├── distillation_sft_lora/
│   ├── distillation_sft_ton_merged/
│   ├── cold_grpo_lora/
│   ├── cold_grpo_ton_merged/
│   ├── sft_grpo_lora/
│   ├── sft_grpo_ton_merged/       ← default (thinking ON)
│   ├── sft_grpo_toff_merged/      ← --thinking-off
│   ├── distillation_sft_grpo_lora/
│   ├── distillation_sft_grpo_ton_merged/
│   ├── distillation_sft_grpo_multi_ton_merged/   ← --language multi
│   └── eval_*/                    ← evaluation outputs (generations + scores)
└── scripts/
    ├── utils.py                   ← path helpers, regex, language maps
    ├── data_sources.py            ← HF dataset loaders (streams live from HF)
    ├── download_data.py           ← download datasets from HF
    ├── download_models.py         ← download trained models from HF
    ├── prepare_data.py            ← build GRPO + SFT JSONL from live HF data
    ├── distillation.py  ← teacher traces with Qwen3-14B (vLLM)
    ├── prepare_multilingual_eval.py   ← build held-out eval from test splits
    ├── train_sft.py               ← LoRA SFT (TRL SFTTrainer)
    ├── train_grpo.py              ← GRPO with binary reward (TRL GRPOTrainer)
    ├── merge_lora.py              ← PeftModel merge + chat template baking
    └── evaluate.py                ← vLLM generation + m3 evaluate/ package scoring
```

### Data sources in `data_sources.py`

| Source | Languages | Notes |
|---|---|---|
| Global-MMLU (dev) | all 5 | Culturally diverse questions |
| M3Exam (dev) | IT, ZH | Official school exams |
| CMMLU (dev) | ZH | Chinese professional/cultural |
| C-Eval (val) | ZH | Chinese college-level |
| ITALIC | IT | Italian cultural/civic |
| EXAMS | IT, ES | Multilingual school exams |
| RuMMLU-Pro | RU | Russian professional |
| MMLU-Hindi | HI | Hindi language MMLU |
| Indic-MMLU | HI | Indian languages MMLU |
| Hindi-MCQ | HI | Hindi multiple-choice |
| Kaleidoscope | HI | Regional Indian knowledge |
| HEAD-QA | ES | Spanish medical licensing |
| AYA-ES | ES | Cohere AYA Spanish |
| MedExpQA-ES | ES | Spanish medical MCQ |
| EXAMS-Bolivia | ES | Bolivian national exams |
| Peru-Exams | ES | Peruvian national exams |
| CMB | ZH | Chinese medical benchmarks |

Limits per source are set in `config.json` under `data.source_limits`.
Training uses `dev` splits; `prepare_multilingual_eval.py` uses `test` splits.
