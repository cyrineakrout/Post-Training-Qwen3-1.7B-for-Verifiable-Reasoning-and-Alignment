# General Knowledge — Reproduction Guide

All trained models and datasets are hosted on HuggingFace at:
**`cyrineakrout/gk-models`** (https://huggingface.co/cyrineakrout/gk-models)
- **Models**: `cyrineakrout/gk-models/merged_models` (all experiment models)
- **Datasets**: `cyrineakrout/gk-models/datasets`
- **CI submission**: `cs-552-2026-camykaz/general_knowledge_model` (best model — Exp 1, 44% CI)

### Evaluating the CI-submitted model
All commands below should be run from code/general_knowledge/

```bash
python scripts/evaluate.py \
    --config configs/config_t3_cot.json \
    --model_type base \
    --base_model cs-552-2026-camykaz/general_knowledge_model \
    --subfolder "" \
    --use_vllm
```

> `--subfolder ""` overrides the config default so the model is loaded directly from the repo root.

---

## Environment

```bash
pip install -r requirements.txt
```
If you get a `bitsandbytes` error when running the commands below, uninstall it first, then rerun the command

```bash
pip uninstall -y bitsandbytes
```

Key packages: `torch`, `transformers`, `peft`, `trl`, `datasets`, `huggingface_hub`, `vllm`.

---

## Evaluating a trained model — one command

The merged models and evaluation dataset are on HuggingFace.
`evaluate.py` **auto-downloads both** the model and the eval file on first run — no manual setup needed.

```bash
# Run from code/general_knowledge/
python scripts/evaluate.py \
    --config configs/config_exp1_sft.json \
    --model_type base \
    --use_vllm \
    --output_path outputs/exp1_eval_generations.jsonl \
    --scores_path outputs/exp1_eval_scores.json
```

> **vLLM is optional.** If vLLM is not installed, drop `--use_vllm` — the script falls back to standard HuggingFace inference, which is slower but produces similar results.
> ```bash
> python scripts/evaluate.py --config configs/config_exp1_sft.json --model_type base
> ```

Replace `config_exp1_sft.json` with the config for the experiment you want to reproduce.

---

## Experiment index (report → config → one command)

All experiments below are from **General Knowledge appendix** of the report.
Each command is run from `code/general_knowledge/`.

> **Note:** This list covers only the experiments reported in the paper. Many additional models and datasets were produced during development and are available on HuggingFace:
> - All models (including unreported runs): [`cyrineakrout/gk_all_models_backup`](https://huggingface.co/cyrineakrout/gk_all_models_backup)
> - All datasets (including intermediate and exploratory mixes): [`cyrineakrout/gk_data_backup`](https://huggingface.co/datasets/cyrineakrout/gk_data_backup)

---

### Table 6 — Main supervised fine-tuning experiments

#### Baseline — Vanilla Qwen3-1.7B, no-thinking prompt, no fine-tuning (7.51% local)
Config: `config_baseline.json` | Model: `Qwen/Qwen3-1.7B` (downloaded directly from HF)

Evaluates the base model before any fine-tuning. The low accuracy is mostly due to missing or malformed `\boxed{}` outputs — the model has not been trained to follow the required output format.
```bash
python scripts/evaluate.py --config configs/config_baseline.json --model_type base --use_vllm
```

---

#### Baseline Thinking — Vanilla Qwen3-1.7B, thinking enabled, minimal prompt, no fine-tuning (48.4%)
Config: `config_baseline_thinking.json` | Model: `Qwen/Qwen3-1.7B` (downloaded directly from HF)

Evaluates the base model with thinking mode enabled (no `/no_think` flag), using a minimal system prompt and 4096 max new tokens. The model is free to use its full `<think>` reasoning chain before producing a `\boxed{X}` answer.
```bash
python scripts/evaluate.py --config configs/config_baseline_thinking.json --model_type base --use_vllm
```

---

#### Exp. 1 — MMLU-aux 25k + Nemotron 25k + GPQA all + SuperGPQA 10k (47.62% local)
Config: `config_exp1_sft.json` | HF model: `models/qwen3_gk_v12b`
```bash
python scripts/evaluate.py --config configs/config_exp1_sft.json --model_type base --use_vllm
```

---
#### Exp. 2 — v12 base + ARC-Challenge + QUARTZ (48.16% local)
Config: `config_exp2_sft.json` | HF model: `models/qwen3_gk_v14`

```bash
python scripts/evaluate.py --config configs/config_exp2_sft.json --model_type base --use_vllm
```
---

#### Exp. 3 — MMLU-aux 50k + Nemotron 50k + GPQA all + SuperGPQA 26k (49.95% local)
Config: `config_exp3_sft.json` | HF model: `cyrineakrout/gk_all_models_backup` → subfolder `merged_models/qwen3_gk_v6_r32_nothink_ckpt6600`

> **Note:** this model is stored in a different repo (`gk_all_models_backup`) from the others (`gk-models`). The config already points to the correct repo.

This model was also used as the warm-start for the GRPO experiment (G1, Table 4).
```bash
python scripts/evaluate.py --config configs/config_exp3_sft.json --model_type base --use_vllm
```

### Table 7 
---
#### A1 — MMLU-aux 6k + ARC-Challenge 750 + OpenBookQA 250 + SciQ 1k (early small-scale run) (42.19% local)
Config: `config_a1_sft.json` | HF model: `cyrineakrout/gk_all_models_backup` → subfolder `merged_models/qwen3_gk_mmluaux6k_arc750_obqa250_sciq1k_lr2e4_ep1`

Small-dataset early experiment to validate the training pipeline. LoRA r=16, α=32, lr=2e-4, effective batch size 8, max_length=2048, 1 epoch. No warmup.
```bash
python scripts/evaluate.py --config configs/config_a1_sft.json --model_type base --use_vllm
```

---

#### A2 — MMLU-aux 4k + SciQ 1k + GPQA-extended all + MMLU-Pro val all (early small-scale run) (40.85% local)
Config: `config_a2_sft.json` | HF model: `cyrineakrout/gk_all_models_backup` → subfolder `merged_models/qwen3_gk_mmluaux4k_sciq1k_gpqaextall_mmluprovalall_lr5e5_ep1`

Small-dataset experiment introducing GPQA and MMLU-Pro val into the mix, no explanations. LoRA r=16, α=32, lr=5e-5, effective batch size 8, max_length=2048, 1 epoch. No warmup.
```bash
python scripts/evaluate.py --config configs/config_a2_sft.json --model_type base --use_vllm
```

---

#### A3 — MMLU-aux 50k + Nemotron 50k, no GPQA/SuperGPQA (ablation) (44.78% local)
Config: `config_a3_sft.json` | HF model: `cyrineakrout/gk_all_models_backup` → subfolder `merged_models/qwen3_gk_mmluaux50k_nemotron50k`

Ablation removing GPQA and SuperGPQA from the Exp. 3 mix to isolate their contribution. Trained with LoRA r=32, α=64, max_length=2048, lr=5e-5, 1 epoch.
```bash
python scripts/evaluate.py --config configs/config_a3_sft.json --model_type base --use_vllm
```

---

### Table 8 — GRPO experiment

#### G1 — GRPO on hard SuperGPQA/GPQA, initialized from Exp. 3 (50.02% local)
Config: `config_g1_grpo.json` | HF model: `models/qwen3_gk_v6_ckpt2000`

GRPO was applied on top of Exp. 3 SFT using a curated set of hard SuperGPQA and GPQA questions (`grpo_hard_gpqa_supergpqa_hard.jsonl`, ~7k examples). The reward function: `+1.0` correct `\boxed{X}`, `0.0` wrong `\boxed{X}`, `−0.5` no `\boxed{}` found. G=8 rollouts per prompt, max completion 512 tokens, temperature 0.9, top-p 0.95, top-k 20, lr 1e-5, effective batch size 8.

Training dataset: `cyrineakrout/gk_data_backup` → `grpo_hard_gpqa_supergpqa_hard.jsonl` (HuggingFace dataset repo — downloaded automatically by `download_data.py`).

```bash
python scripts/evaluate.py --config configs/config_g1_grpo.json --model_type base --use_vllm
```

---

### Table 9 — Chain-of-thought experiments

#### T1 — Synthetic CoT distillation: MMLU-aux + GPQA + SuperGPQA (50.15% local)
Config: `config_t1_cot.json` | HF model: `models/qwen3_gk_v10`

CoT reasoning generated by a Qwen3 teacher model; only correctly answered and correctly formatted generations kept (~40k examples).
```bash
python scripts/evaluate.py --config configs/config_t1_cot.json --model_type base --use_vllm
```

---

#### T2 — T1 + Nemotron-science CoT rationales (49.10% local)
Config: `config_t2_cot.json` | HF model: `models/qwen3_gk_v11`

T1 data augmented with ~30k Llama-Nemotron science MCQ examples with DeepSeek-R1 CoT rationales (~70k total).
```bash
python scripts/evaluate.py --config configs/config_t2_cot.json --model_type base --use_vllm
```

---

#### T3 — Exp. 1 model (no-think SFT) evaluated with thinking prompt at inference time (53.23% local) BEST MODEL
Config: `config_t3_cot.json` | HF model: `models/qwen3_gk_v12b` 

Tests whether the no-think SFT model (v12b) retains its thinking capability when the thinking system prompt is injected at eval time, without any CoT fine-tuning. Uses the same generation settings as T1/T2 (temperature 0.7, repetition_penalty 1.03, 4096 max new tokens). `inject_system_prompt` overrides the no-think prompt baked in during Exp. 1 training.
```bash
python scripts/evaluate.py --config configs/config_t3_cot.json --model_type base --use_vllm
```

---

## Reproducing training

> Training requires the dataset files locally. Run `python scripts/download_data.py` once before training.

```bash
python scripts/download_data.py   # one-time download of all datasets from HF
```

### Evaluating a locally trained model

After training, evaluate the adapter directly (no merge needed):

```bash
python scripts/evaluate.py \
    --config configs/config_exp1_sft.json \
    --model_type lora \
    --base_model Qwen/Qwen3-1.7B \
    --adapter_path models/qwen3_gk_v12b
```

Or merge first and evaluate the merged model (required for `--use_vllm`):

```bash
# No-think models (Exp. 1, Exp. 2, Exp. 3, G1)
python scripts/merge_lora.py --base_model Qwen/Qwen3-1.7B \
    --adapter_path models/qwen3_gk_v12b --output_dir models/qwen3_gk_v12b_merged

# CoT models (T1, T2) — must add --do_sample to get correct generation_config.json
python scripts/merge_lora.py --base_model Qwen/Qwen3-1.7B \
    --adapter_path models/qwen3_gk_v10 --output_dir models/qwen3_gk_v10_merged \
    --do_sample --temperature 0.6 --repetition_penalty 1.03 --top_k 20 --top_p 0.95

python scripts/evaluate.py \
    --config configs/config_exp1_sft.json \
    --model_type base \
    --base_model models/qwen3_gk_v12b_merged \
    --use_vllm
```

Substitute the config and paths for any other experiment.

---

### Table 2 — SFT experiments

**Exp. 1** (`config_exp1_sft.json`):
```bash
python scripts/train_sft.py --config configs/config_exp1_sft.json
```

**Exp. 2** (`config_exp2_sft.json`):
```bash
python scripts/train_sft.py --config configs/config_exp2_sft.json
```

**Exp. 3** (`config_exp3_sft.json`):
```bash
python scripts/train_sft.py --config configs/config_exp3_sft.json
```

**A1** (`config_a1_sft.json`):
```bash
python scripts/train_sft.py --config configs/config_a1_sft.json
```

**A2** (`config_a2_sft.json`):
```bash
python scripts/train_sft.py --config configs/config_a2_sft.json
```

**A3** (`config_a3_sft.json`):
```bash
python scripts/train_sft.py --config configs/config_a3_sft.json
```

### Table 6 — CoT experiments

**T1** (`config_t1_cot.json`):
```bash
python scripts/train_sft.py --config configs/config_t1_cot.json
```

**T2** (`config_t2_cot.json`):
```bash
python scripts/train_sft.py --config configs/config_t2_cot.json
```

### Table 4 — GRPO experiment (G1)

The GRPO model was initialized from the Exp. 3 SFT (ckpt6600), available on HF at `cyrineakrout/gk-all-models-backup / merged_models/qwen3_gk_v6_r32_nothink_ckpt6600`.

**Training data:** `grpo_hard_gpqa_supergpqa_hard.jsonl` — a curated set of hard SuperGPQA and GPQA questions, available at `cyrineakrout/gk_data_backup` (HF dataset repo). Downloaded automatically by `download_data.py`.

```bash
# Step 1 (option A): re-train the Exp. 3 SFT base from scratch
python scripts/train_sft.py --config configs/config_exp3_sft.json
python scripts/merge_lora.py --base_model Qwen/Qwen3-1.7B \
    --adapter_path models/qwen3_gk_v6_sft --output_dir models/qwen3_gk_v6_sft_merged

# Step 1 (option B): download the pre-trained Exp. 3 SFT model from HF
python -c "
import shutil
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='cyrineakrout/gk-all-models-backup',
    allow_patterns=['merged_models/qwen3_gk_v6_r32_nothink_ckpt6600/**'],
    local_dir='/tmp/gk_sft_download'
)
shutil.move(
    '/tmp/gk_sft_download/merged_models/qwen3_gk_v6_r32_nothink_ckpt6600',
    'models/qwen3_gk_v6_sft_merged'
)
"
# model will be at: models/qwen3_gk_v6_sft_merged

# Step 2: run GRPO (training data is set in the config)
python scripts/train_grpo.py \
    --config configs/config_g1_grpo.json \
    --model_path models/qwen3_gk_v6_sft_merged \
    --output_dir models/qwen3_gk_v6_grpo
```

---

## Data preprocessing

The training datasets uploaded to HuggingFace were built with the scripts in
`scripts/data_processing/`. They do **not** need to be re-run to reproduce results
(the final datasets are already on HF), but are included for full transparency.

### Pipeline overview

```
Raw HF datasets (MMLU, GPQA, Nemotron, ARC, SuperGPQA, ...)
        │
        ▼
prepare_sft_data.py          ← converts individual datasets to SFT format
prepare_mixed_data.py        ← mixes sources for no-think experiments
        │
        ├── v6, v12 datasets (no-think, direct answer \boxed{X})
        │
filter_nemotron_mcqa.py      ← filters Nemotron RL pool → train pool + eval split
prepare_nemotron_science.py  ← converts Nemotron science CoT data
generate_cot_qwen.py         ← generates CoT with a Qwen3 teacher model (vLLM)
postprocess_cot.py           ← cleans/validates generated CoT
        │
        ├── v10, v11 datasets (chain-of-thought)
        
```

### Script descriptions

#### `prepare_sft_data.py`
Core conversion utilities shared by all other data scripts. Converts raw HuggingFace
datasets — MMLU (auxiliary_train, dev splits), ARC-Challenge, OpenBookQA, SciQ,
CommonsenseQA, and QUARTZ — into the project's SFT JSONL format:
`{"messages": [...], "metadata": {"source": ..., "answer": "B", ...}}`.
The global `SYSTEM_PROMPT` is injected at call time by the parent script so every
source in a mixed dataset shares the same system message.

#### `prepare_mixed_data.py`
Main dataset builder for the **no-think** experiments (v6, v12). Takes a config
file and CLI flags that control how many examples to draw from each source, then
assembles and shuffles the combined dataset. Sources supported: MMLU-aux, MMLU-dev,
ARC, OpenBookQA, SciQ, GPQA (extended/main/diamond), MMLU-Pro, CommonsenseQA,
MedMCQA, Nemotron RL pool, Nemotron science, SuperGPQA. Filters non-English
examples (CJK character heuristic) before saving.

| Dataset | Sources used |
|---|---|
| `sft_train_v6.jsonl` | MMLU-aux 50k + Nemotron RL pool 50k + GPQA extended (all) + SuperGPQA (all) |
| `sft_train_v12.jsonl` | MMLU-aux 25k + Nemotron RL pool 25k + GPQA extended (all) + SuperGPQA 10k |

#### `filter_nemotron_mcqa.py`
Filters and converts `nvidia/Nemotron-RL-knowledge-mcqa` into SFT format.
Cleans question text by stripping Nemotron's instruction prefix and embedded
choice lines, then parses variable-count options (2–20 choices, A–T). Optionally
filters by model pass rate (e.g. keep only examples where `Qwen/Qwen3-30B-A3B`
achieves `pass_rate == 1.0`). The filtered pool is split into a 300-example
held-out eval set and a training pool used as the Nemotron component in v6/v12.

#### `generate_cot_qwen.py`
Uses a Qwen3 teacher model (via vLLM) to generate chain-of-thought reasoning for
hard MCQ examples. Takes an existing SFT JSONL, filters to specified sources
(GPQA, SuperGPQA, MMLU by default), generates multi-step `<think>` reasoning for
each question, and **keeps only examples where the generated answer matches the
reference** (self-consistency filter). Used to create the CoT training data
for v10 and v11.

#### `postprocess_cot.py`
Cleans and validates CoT JSONL files produced by `generate_cot_qwen.py`.
Removes malformed examples (missing `</think>` or `\boxed{}`), strips any
extraneous prose between `</think>` and the final answer, and normalises the
assistant turn to exactly `<think>...</think>\n\n\boxed{X}`. Optionally rewrites
all system prompts to the canonical CoT system prompt. Can merge multiple input
files into one output.

#### `prepare_nemotron_science.py`
Converts the `nvidia/Llama-Nemotron-Post-Training-Dataset` SFT/science split
(~708k MCQ examples with DeepSeek-R1 reasoning) into the CoT training format.
Reservoir-samples a random subset to avoid loading the full dataset, extracts the
final answer letter from each output ("the answer is X" pattern), validates that
the extracted letter is an actual listed choice (guards against mid-reasoning
false positives), and wraps the existing reasoning in `<think>...</think>\n\n\boxed{X}`.
30k examples from this source are used in `sft_train_v11.jsonl`.

> **Note on `sft_train_v14.jsonl`:** this dataset was built by augmenting the v12 base mix with ARC-Challenge train and QUARTZ train examples. The final dataset is available directly on HuggingFace.

---

## Directory layout

```
general_knowledge/
├── requirements.txt
├── README.md
├── configs/
│   ├── config.py              # config loader utility (used by all scripts)
│   ├── config_baseline.json           # Baseline: vanilla Qwen3-1.7B, no-think, no fine-tuning (eval only)
│   ├── config_baseline_thinking.json  # Baseline: vanilla Qwen3-1.7B, thinking enabled, minimal prompt (eval only)
│   ├── config_t1_cot.json     # T1 CoT SFT: MMLU-aux 39k + GPQA 1.6k
│   ├── config_t2_cot.json     # T2 CoT SFT: MMLU-aux 39k + GPQA 1.6k + Nemotron-science 30k
│   ├── config_t3_cot.json     # T3 Exp.1 model (v12b) + thinking prompt at inference (eval only)
│   ├── config_exp1_sft.json   # Exp. 1 no-think SFT: MMLU-aux 25k + Nem 25k + GPQA all + SuperGPQA 10k
│   ├── config_exp2_sft.json   # Exp. 2 no-think SFT: Exp. 1 base + ARC + QUARTZ
│   ├── config_exp3_sft.json   # Exp. 3 no-think SFT (intermediate: base model for GRPO)
│   ├── config_a1_sft.json     # A1 no-think SFT early run: MMLU-aux 6k + ARC 750 + OpenBookQA 250 + SciQ 1k
│   ├── config_a2_sft.json     # A2 no-think SFT early run: MMLU-aux 4k + SciQ 1k + GPQA-ext all + MMLU-Pro val all
│   ├── config_a3_sft.json     # A3 no-think SFT ablation: MMLU-aux 50k + Nemotron 50k only
│   └── config_g1_grpo.json    # G1 GRPO RL on top of Exp. 3 SFT
├── scripts/
│   ├── download_data.py       # download training datasets from HF (training only)
│   ├── train_sft.py           # LoRA SFT training
│   ├── train_grpo.py          # GRPO reinforcement learning
│   ├── merge_lora.py          # merge LoRA adapter into base model
│   ├── evaluate.py            # evaluation (auto-downloads model + eval file from HF)
│   ├── eval_utils.py          # shared generation and scoring utilities
│   └── data_processing/       # dataset preparation scripts (see above)
│       ├── generate_cot_qwen.py      # teacher-model CoT generation (vLLM)
│       ├── postprocess_cot.py        # clean/validate generated CoT
│       ├── prepare_sft_data.py       # convert HF datasets to SFT format
│       ├── prepare_mixed_data.py     # assemble no-think training mixes (v6, v12)
│       ├── filter_nemotron_mcqa.py   # filter Nemotron RL pool
│       └── prepare_nemotron_science.py # convert Nemotron science CoT data
├── data/
│   └── datasets/              # populated by download_data.py
├── models/                    # local adapter / merged model outputs
└── outputs/                   # generation JSONL and score files
```
