# Group Model — Reproduction Guide

Multi-domain model trained on four benchmarks: **Math**, **General Knowledge**, **Multilingual**, **Safety**.

- **CI submission**: `cs-552-2026-camykaz/group_model`
- **Training data**: `cs-552-2026-camykaz/group_data`
- **All models**: `zguir23/group-models`
- **Run B retrain model**: `Yassinekouch/mnlp_project_qwen_based_group_model`
- **lora_direct_v1 artifacts**: `cs-552-2026-camykaz/group_model_lora_direct_v1`
- **ties_v1 artifacts**: `cs-552-2026-camykaz/merge_ties`

Experiment configs drive each pipeline — they contain the HF paths so scripts download artifacts automatically on first run:
- [`configs/lora_direct_v1.json`](configs/lora_direct_v1.json) — weighted delta merge (linear)
- [`configs/ties_v1.json`](configs/ties_v1.json) — TIES merge of 4 specialist models
- [`configs/runb_retrain.json`](configs/runb_retrain.json) — safety-heavy joint-SFT Run B retrain

---

## Setup

All commands run from `code/group/`.

```bash
cd code/group
pip install -r requirements.txt
```

**vLLM (optional):** if `vllm` is installed and a CUDA GPU with bfloat16 support is available,
the eval script uses it automatically for fast batched inference (math examples get pass@8).
Without vLLM the script falls back to HuggingFace `transformers` — no code change needed,
just slower. To install vLLM:
```bash
pip install vllm
```

---

## Experiment Index

| Experiment | Strategy | Published model | Evaluate |
| --- | --- | --- | --- |
| Base | Untouched `Qwen/Qwen3-1.7B` baseline | `Qwen/Qwen3-1.7B` | Option A |
| lora_direct_v1 | Equal-weight linear delta merge of four specialist LoRA adapters | `cs-552-2026-camykaz/group_model_lora_direct_v1/lora_direct_v1/group_model` | Option B |
| ties_v1 | TIES merge of four specialist models with domain-weighted coefficients | `cs-552-2026-camykaz/merge_ties/group_model` | Option D |
| runb_retrain | Safety-heavy joint SFT across all four domains; LoRA r=64, 0.6 epoch | `Yassinekouch/mnlp_project_qwen_based_group_model` | Option F |
| distillation SFT + GRPO | multi-task training on joint data from the four domains | `mmsellam/group_model` | Option G |

The Run B retrain differs from the merge experiments: it trains one LoRA jointly
on a 110,400-example four-domain dataset, then merges that adapter into Qwen3.
Its verified training mix is 44,382 safety, 34,490 general knowledge, 19,708
multilingual, and 11,820 math examples.

---
## Option A Evaluate the base model (baseline)

Evaluate unmodified `Qwen/Qwen3-1.7B` on the same eval set with a minimal system prompt.
Uses the eval file from the config; downloads the model directly from HF via `transformers`.

```bash
python scripts/eval_group.py --config configs/lora_direct_v1.json \
    --model_dir Qwen/Qwen3-1.7B \
    --minimal-system-prompt \
    --max_new_tokens 4096
```

## Option B — Evaluate the pre-merged model (linear merge, Best model)

Downloads the merged model and eval file from HF, then evaluates.

```bash
python scripts/eval_group.py --config configs/lora_direct_v1.json
```

---

## Option C — Reproduce the merge from domain adapters, then evaluate

Downloads the four domain adapters from HF, runs weighted delta merge, saves locally.

```bash
# Step 1: merge (downloads adapters, saves to models/lora_direct_v1_reproduced/)
python scripts/merge_adapters.py --config configs/lora_direct_v1.json

# Step 2: evaluate the reproduced merge
python scripts/eval_group.py --config configs/lora_direct_v1.json \
    --model_dir models/lora_direct_v1_reproduced
```

Results are appended to `results.txt`.

---


## Option D — Evaluate the TIES-merged model (download from HF)

Downloads the TIES model and eval file from HF, then evaluates.

```bash
python scripts/eval_group.py --config configs/ties_v1.json
```

## Option E — Reproduce the TIES merge from specialist models, then evaluate

Specialist models are downloaded automatically from `cs-552-2026-camykaz/merge_ties` if not present locally.

```bash
# Step 1: run TIES merge (downloads specialist models from HF if needed)
python scripts/merge_ties.py --config configs/merge_ties.yaml

# Step 2: evaluate
python scripts/eval_group.py --config configs/ties_v1.json \
    --model_dir outputs/group_model_merged
```

Results are appended to `results.txt`.

---
## Option F — Evaluate and reproduce the Run B retrain experiment

The config downloads the complete merged model from the root of
`Yassinekouch/mnlp_project_qwen_based_group_model` and evaluates it on the same
shared group benchmark used by the merge experiments. Its 4,096-token generation
profile matches the published model configuration.

```bash
python scripts/eval_group.py --config configs/runb_retrain.json
```

Quick smoke evaluation:

```bash
python scripts/eval_group.py --config configs/runb_retrain.json \
    --limit 1 \
    --max_new_tokens 32
```

Verified Run B retrain settings:

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-1.7B` |
| Prepared train / validation rows | 110,400 / 1,600 |
| Domain targets before validation split | math 12k, knowledge 35k, multilingual 20k, safety 45k |
| LoRA | r=64, alpha=128, dropout=0.05 |
| Learning rate / epochs / sequence length | `5e-6` / `0.6` / `2048` |
| Final step / best validation checkpoint | 4140 / checkpoint 4000 |

---
## Option G — Evaluate the published distillation SFT + GRPO model

Downloads `mmsellam/group_model` from HuggingFace and evaluates it on the shared
group benchmark.

```bash
python scripts/eval_group.py --config configs/distillation_grpo.json
```

---
## Option H — Reproduce training from scratch (joint SFT)

### Step 1 — Download data
```bash
python scripts/download_data.py
```

> Pre-built dataset already on HF — `download_data.py` fetches it directly.

### Step 2 — Train LoRA (r=96, α=192, lr=8e-6, 0.6 epochs)
```bash
GROUP_BASE_MODEL=Qwen/Qwen3-1.7B \
GROUP_TRAIN_FILE=data/group_sft_final_train.jsonl \
GROUP_VAL_FILE=data/group_sft_final_val.jsonl \
GROUP_OUTPUT_DIR=outputs/group_lora_final \
python scripts/train_group_sft.py
```

### Step 3 — Merge adapter
```bash
GROUP_BASE_MODEL=Qwen/Qwen3-1.7B \
GROUP_ADAPTER_DIR=outputs/group_lora_final \
GROUP_MERGED_DIR=outputs/group_model_merged \
python scripts/merge_lora.py
```

### Step 4 — Evaluate
```bash
python scripts/eval_group.py \
    --model_dir outputs/group_model_merged \
    --config    configs/lora_direct_v1.json
```

## Option I — Distillation SFT + GRPO pipeline

Full pipeline: distillation data generation → merge -> SFT → merge → GRPO → merge.
Requires an A100 40 GB (always uses vLLM + bfloat16).
### Step 0 - Download data

If the data files are not already present, run:

```bash
python scripts/download_data.py
```
### Step 1 — distillation (Qwen3-14B teacher)

This script produces a distilled SFT dataset by running the `Qwen/Qwen3-14B`
teacher via vLLM (bfloat16). It samples from the prepared SFT pool, generates
thinking traces, and applies simple filters:
    - MCQ: keep only responses where the teacher outputs the correct \boxed{LETTER}
    - Math: keep only responses that include a \boxed{...} token
    - Other: keep the teacher response as-is

Requirements and notes:
    - Requires vLLM and a GPU with bfloat16 (A100 40GB recommended).
    - The script resumes automatically if the output JSONL exists and retries
        previously rejected examples once.

Usage (run from `code/group/`):

```bash
# Optional environment overrides:
# GROUP_DIST_IN   - input JSONL (default: data/group_sft_final_train.jsonl)
# GROUP_DIST_OUT  - output JSONL (default: data/group_distillation_train.jsonl)
# GROUP_DIST_BATCH - vLLM batch size (default: 8)

GROUP_DIST_IN=data/group_sft_final_train.jsonl \
GROUP_DIST_OUT=data/group_distillation_train.jsonl \
GROUP_DIST_BATCH=8 \
python scripts/distillation.py --n-distill 2150 --batch-size 8
```

Key args/defaults:
    - `--n-distill` (default 2150) — number of examples to attempt distilling
    - `--batch-size` / `GROUP_DIST_BATCH` (default 8) — vLLM generation batch size

Behavior summary:
    - Runs the teacher to generate SFT-format records and writes JSONL lines to
        the output file. Each kept record contains system/user/assistant messages.
    - Uses a fixed seed (42) for reproducible sampling.

### Step 2 — SFT on distillation data

```bash
GROUP_BASE_MODEL=Qwen/Qwen3-1.7B \
GROUP_TRAIN_FILE=data/group_distillation_train.jsonl \
GROUP_VAL_FILE=data/group_sft_final_val.jsonl \
GROUP_OUTPUT_DIR=outputs/distillation_sft_lora \
python scripts/train_group_sft.py
```

### Step 3 — Merge SFT adapter

```bash
python scripts/merge_lora.py \
    --adapter_path outputs/distillation_sft_lora \
    --output_dir   outputs/distillation_sft_merged
```

### Step 4 — GRPO training on top of merged SFT

Warm-starts from the merged SFT model.

```bash
GROUP_GRPO_BASE=outputs/distillation_sft_merged \
GROUP_GRPO_DATA=data/group_grpo_train.jsonl \
GROUP_GRPO_OUT=outputs/distillation_sft_grpo_lora \
python scripts/train_group_grpo.py
```

Resume a crashed run automatically (re-run the same command).
Force a fresh run with `--restart`.

Key env vars (all have defaults — see script header for full list):

| Variable | Default | Description |
|---|---|---|
| `GROUP_GRPO_BASE` | `outputs/distillation_sft_merged` | Merged SFT model to warm-start from |
| `GROUP_GRPO_DATA` | `data/group_grpo_train.jsonl` | GRPO training JSONL |
| `GROUP_GRPO_OUT` | `outputs/distillation_sft_grpo_lora` | LoRA output dir |
| `GROUP_GRPO_LR` | `1e-5` | Learning rate |
| `GROUP_GRPO_NGEN` | `8` | Generations per prompt |
| `GROUP_GRPO_LORA_R` | `64` | LoRA rank |
| `WANDB_PROJECT` | `cs552-mnlp-group` | W&B project |

### Step 5 — Merge GRPO adapter

```bash
python scripts/merge_lora.py \
    --adapter_path outputs/distillation_sft_grpo_lora \
    --base_model   outputs/distillation_sft_merged \
    --output_dir   outputs/distillation_sft_grpo_merged
```

### Step 6 — Evaluate

```bash
python scripts/eval_group.py \
    --model_dir outputs/distillation_sft_grpo_merged \
    --config    configs/lora_direct_v1.json
```

---

## Merge method (lora_direct_v1)

`merge_adapters.py` accumulates weighted LoRA deltas manually:

```
Δ_final = 0.25 × Δ_safety + 0.25 × Δ_gk + 0.25 × Δ_multilingual + 0.25 × Δ_math
W_merged = W_base + Δ_final
```

The adapters have different LoRA ranks (r=16/32/64/96), so PEFT's `add_weighted_adapter`
cannot be used directly. The script materialises each delta (`alpha/r × B @ A`) and sums
them in float32 before casting back to bfloat16.

---

### Reproduce `group_sft_final_train.jsonl`

1. Download the data. This downloads the general knowledge and safety datasets:
   ```bash
   python scripts/download_data.py
   ```

2. Build the group SFT dataset. This creates the required SFT train and validation files:
   ```bash
   python scripts/build_group_dataset.py
   ```

## HuggingFace repo layout

Repo: `cs-552-2026-camykaz/group_model_lora_direct_v1`

```
lora_direct_v1/
├── group_model/            ← linear-merged model (Option B)
├── adapters/
│   ├── safety/             ← safety LoRA adapter  (r=16)
│   ├── general_knowledge/  ← GK LoRA adapter      (r=16)
│   ├── multilingual/       ← multilingual adapter  (r=16)
│   └── math/               ← math LoRA adapter    (r=96)
└── eval/
    └── group_eval.jsonl    ← evaluation set (shared by all configs)
```

Repo: `cs-552-2026-camykaz/merge_ties`

```
group_model/        ← TIES-merged model (Option E)
general_knowledge/  ← GK specialist model
math/               ← math specialist model
multilingual/       ← multilingual specialist model
safety/             ← safety specialist model
```

Repo: `Yassinekouch/mnlp_project_qwen_based_group_model`

```
config.json              ← Run B retrain merged model at repository root
model.safetensors
generation_config.json
tokenizer.json
tokenizer_config.json
chat_template.jinja
```

---

## Repository layout

```
code/group/
├── README.md
├── requirements.txt
├── configs/
│   ├── lora_direct_v1.json   ← HF paths, adapter weights, generation params (linear merge)
│   ├── ties_v1.json          ← HF paths and generation params (TIES merge)
│   ├── runb_retrain.json     ← HF path, verified training metadata, and generation params
│   ├── chat_template.jinja   ← 4-domain system prompt baked into merged model
│   └── merge_ties.yaml       ← TIES merge config
└── scripts/
    ├── eval_group.py             ← evaluation (supports --config)
    ├── merge_adapters.py         ← weighted delta merge (supports --config)
    ├── merge_lora.py             ← simple single-adapter merge
    ├── merge_ties.py             ← TIES-based merging (alternative approach)
    ├── train_group_sft.py        ← joint SFT training
    ├── train_group_grpo.py       ← GRPO fine-tuning on a merged SFT base
    ├── distillation.py           ← generate distillation data (Qwen3-14B teacher)
    ├── push_group_model.py       ← push merged model to HuggingFace
    ├── build_group_dataset.py    ← assemble multi-domain training dataset
    ├── download_adapters.py      ← standalone adapter downloader
    └── download_data.py          ← download datasets from HF
```

`adapters/`, `data/`, `outputs/`, and `models/` are created at runtime and are not committed.
