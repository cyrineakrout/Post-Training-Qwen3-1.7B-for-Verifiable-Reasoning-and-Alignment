# Math Model

Math specialist models built from `Qwen/Qwen3-1.7B` using supervised fine-tuning,
GRPO, targeted data curation, and long-context reasoning experiments.

- **All merged experiment models**: [`Yassinekouch/mnlp_project_qwen_based_math_model`](https://huggingface.co/Yassinekouch/mnlp_project_qwen_based_math_model)
- **Prepared training and evaluation data**: `cs-552-2026-camykaz/math_model`
- **CI submission**: `cs-552-2026-camykaz/math_model` repository root
- **Selected model**: archived subfolder `v7`, merged from checkpoint 2400
- **CI/archive consistency**: the CI root model and archived `v7` model weights have the same LFS hash

All commands below run from the M3 repository `code` directory:

```bash
cd /scratch/standard-project-m3-camykaz/code
```

## Environment

```bash
pip install -r math/requirements.txt
```

If you get a `bitsandbytes` error after installing the requirements, uninstall
it before running the math scripts:

```bash
pip uninstall -y bitsandbytes
```

The public models and datasets require no authentication for evaluation or
training reproduction. Authentication is needed only when publishing. The vLLM
evaluator requires one CUDA GPU with enough memory for Qwen3-1.7B.

## Evaluate The Submitted Model

This one command downloads the selected v7 model and final evaluation benchmark
from Hugging Face, generates eight completions per problem, and reports pass@8
overall and by source:

```bash
python math/scripts/eval/evaluate_math_vllm_pass8.py --experiment v7
```

It writes detailed generations to `math/outputs/published_eval/v7.jsonl` and a
compact score report to `math/outputs/published_eval/v7.summary.json`.

For a quick end-to-end check:

```bash
python math/scripts/eval/evaluate_math_vllm_pass8.py \
  --experiment v7 \
  --limit 10 \
  --max_tokens 4096 \
  --max_model_len 16384 \
  --out_jsonl math/outputs/published_eval/v7_smoke.jsonl
```

## Experiment Index

Each training script automatically downloads its prepared train and validation
files from `cs-552-2026-camykaz/math_model`. Each `--experiment` evaluation
automatically selects the corresponding published model, output path, generation
budget, and context window from `math/configs/experiments.json`.

| Experiment | Strategy | HF folder | Eval tokens / context | Training command |
| --- | --- | --- | ---: | --- |
| v1 | Early GSM8K SFT baseline | `v1` | 2,048 / 8,192 | `python math/scripts/v1_early_sft/train_math_v1_early_sft.py` |
| v2 | GSM8K + Hendrycks MATH | `v2` | 2,048 / 8,192 | `python math/scripts/v2_revised_sft/train_math_v2_revised_sft.py` |
| v3 | Curated competition math | `v3` | 2,048 / 8,192 | `python math/scripts/v3_curated_sft/train_math_v3_curated_sft.py` |
| v4 strong SFT | Broad thinking-format SFT | `v4_strong_sft` | 4,096 / 16,384 | `python math/scripts/v4_strong_sft/train_math_v4_strong_sft.py` |
| v4 GRPO | GRPO continuation of v4 strong SFT | `v4_grpo` | 4,096 / 16,384 | `python math/scripts/v4_strong_sft/train_math_v4_grpo_ref.py` |
| v5 | All-solutions targeted SFT | `v5` | 4,096 / 16,384 | `python math/scripts/v5_all_solutions_sft/train_math_v5_all_solutions_sft.py` |
| v6 | 50k long-context AceReason mix | `v6` | 12,000 / 40,960 | `python math/scripts/v6_50k_long/train_math_v6_50k_long_lora_sft.py` |
| v7 | 100k long-context AceReason mix; selected checkpoint 2400 | `v7` | 16,384 / 40,960 | `python math/scripts/v7_100k/train_math_v7_100k_lora_sft.py` |

### Strategy Progression

- **v1 - establish the SFT baseline:** trains on GSM8K reasoning traces with
  standardized boxed final answers. This version verifies the basic LoRA
  training, merging, and math-answer formatting pipeline.
- **v2 - move toward competition math:** adds the seven Hendrycks MATH subjects
  while retaining a controlled amount of GSM8K. Compared with v1, it introduces
  harder and more diverse problems without changing the short-context setup.
- **v3 - improve data quality through curation:** replaces the broad v2 mix with
  up to 14k filtered competition examples from Hendrycks MATH and
  OlympiadContest. It applies stronger prompt, solution, and answer-quality
  filters to prioritize reliable reasoning traces.
- **v4 strong SFT - increase scale and reasoning diversity:** expands to up to
  47k examples from NuminaMath-CoT, Hendrycks MATH, and OlympiadContest. It uses
  thinking-format responses and a larger evaluation context than v1-v3.
- **v4 GRPO - test reinforcement learning after strong SFT:** continues from the
  merged v4 strong-SFT model using v4 prompts and boxed-answer rewards. This
  isolates whether GRPO improves the already trained v4 model.
- **v5 - target incomplete multi-answer behavior:** starts from the v4 prepared
  data and oversamples problems asking for all values, functions, solutions, or
  possible cases. Its goal is to reduce answers that find only one valid case.
- **v6 - introduce controlled long-context training:** switches to a 50k-example
  AceReason mix of OpenMathReasoning and NuminaMath-CoT. Most traces remain short,
  with a controlled medium/long tail; checkpoint 1500 is the canonical candidate.
- **v7 - scale the long-context strategy:** expands AceReason training to 100k
  examples with substantially more medium and long traces. Checkpoint 2400 is
  the selected submission model and is evaluated with the largest token budget.

The v6 and v7 preparation scripts exclude questions matching the final
evaluation benchmark to reduce evaluation leakage.

Evaluate any one model by changing the experiment name:

```bash
python math/scripts/eval/evaluate_math_vllm_pass8.py --experiment v4_grpo
```

Evaluate all eight published models sequentially with their configured profiles:

```bash
bash math/scripts/eval/run_eval_all_published.sh
```

Quickly verify all eight evaluation pipelines:

```bash
bash math/scripts/eval/run_eval_all_published.sh \
  --limit 1 \
  --num_completions 1 \
  --max_tokens 32 \
  --max_model_len 4096
```

Explicit `--max_tokens` and `--max_model_len` flags override an experiment's
default profile, allowing controlled comparisons at different token budgets.

## Reproduce Training

Run the training command from the experiment table to reproduce that version.
For example, the selected v7 LoRA run is:

```bash
python math/scripts/v7_100k/train_math_v7_100k_lora_sft.py
```

A smoke run still downloads and loads the model and v7 data, but trains for only
one step:

```bash
MATH_MAX_STEPS=1 MATH_OUTPUT_DIR=/tmp/math_v7_smoke \
python math/scripts/v7_100k/train_math_v7_100k_lora_sft.py
```

Later experiments support environment overrides such as:

```bash
MATH_BASE_MODEL=Qwen/Qwen3-1.7B \
MATH_OUTPUT_DIR=math/outputs/custom_v7 \
MATH_MAX_SEQ_LEN=16000 \
python math/scripts/v7_100k/train_math_v7_100k_lora_sft.py
```

The v4 GRPO run requires the merged v4 strong-SFT model as its base. Train and
merge v4 first, or download the published base:

```bash
hf download Yassinekouch/mnlp_project_qwen_based_math_model \
  --include "v4_strong_sft/**" \
  --local-dir math/outputs/published_models

MATH_BASE_MODEL=math/outputs/published_models/v4_strong_sft \
python math/scripts/v4_strong_sft/train_math_v4_grpo_ref.py
```

The optional full-parameter v7 training command is:

```bash
python math/scripts/v7_100k/train_math_v7_100k_full_sft.py
```

## Prepared Data

Training scripts stream their required prepared files from Hugging Face. A local
copy is optional and can be downloaded under `math/data/hf/`:

```bash
python math/scripts/download_data.py
python math/scripts/download_data.py --version v7_100k
python math/scripts/download_data.py --eval-only
```

The public dataset repository has this layout:

```text
v1_early_sft/{train.jsonl,validation.jsonl}
v2_revised_sft/{train.jsonl,validation.jsonl}
v3_curated_sft/{train.jsonl,validation.jsonl,train_with_meta.jsonl,validation_with_meta.jsonl}
v4_strong_sft/{train.jsonl,validation.jsonl,train_with_meta.jsonl,validation_with_meta.jsonl,grpo_ref.jsonl}
v5_all_solutions_sft/{train.jsonl,validation.jsonl}
v6_50k_long/{train.jsonl,validation.jsonl,manifest.json}
v7_100k/{train.jsonl,validation.jsonl,manifest.json}
final_evaluation/{math_benchmark_external_v1.jsonl,manifest.json}
```

Set `MATH_DATASET_REPO=owner/repository` to use another dataset repository.
Rerunning preparation is not required for reproduction, but the preparation
scripts are retained for transparency. To rebuild and publish all prepared data:

```bash
python math/scripts/prepare_and_publish_all_math_datasets.py
```

Prepare locally without publishing, or publish existing prepared files:

```bash
MATH_SKIP_HF_UPLOAD=1 python math/scripts/prepare_and_publish_all_math_datasets.py
python math/scripts/publish_all_prepared_math_datasets.py
```

## Merge And Push

Merge commands require a completed local adapter or checkpoint and do not use
training data. Canonical merge commands are:

```bash
python math/scripts/v1_early_sft/merge_math_v1_early_sft.py
python math/scripts/v2_revised_sft/merge_math_v2_revised_sft.py
python math/scripts/v3_curated_sft/merge_math_v3_curated_sft.py
python math/scripts/v4_strong_sft/merge_math_v4_strong_sft.py
python math/scripts/v4_strong_sft/merge_math_v4_grpo_ref.py
python math/scripts/v5_all_solutions_sft/merge_math_v5_all_solutions_sft.py
python math/scripts/v6_50k_long/merge_math_v6_50k_long_checkpoint.py

python math/scripts/v7_100k/merge_math_v7_100k_checkpoint.py \
  --checkpoint math/outputs/math_v7_100k_lora/checkpoint-2400 \
  --out_dir math/outputs/math_v7_100k_merged_ckpt2400
```

The v6 and v7 merge scripts install the thinking-enabled math chat template and
generation configuration. Publish a merged model to its experiment subfolder:

```bash
python math/scripts/push_math_submission_model.py \
  --model_dir math/outputs/math_v7_100k_merged_ckpt2400 \
  --repo_id Yassinekouch/mnlp_project_qwen_based_math_model \
  --path_in_repo v7
```

Use the corresponding `path_in_repo` from `math/configs/experiments.json` for
another experiment. The publisher validates configuration, tokenizer, chat
template, and safetensors files before upload.

## Evaluate A Local Model

The same evaluator accepts a local merged model. When `--eval_file` is omitted,
it downloads the final benchmark automatically:

```bash
python math/scripts/eval/evaluate_math_vllm_pass8.py \
  --model_dir path/to/merged_model \
  --out_jsonl math/outputs/local_eval.jsonl \
  --max_tokens 12000 \
  --max_model_len 40960
```

Pass `--eval_file path/to/file.jsonl` to evaluate a different local benchmark.
Completed evaluation files can be summarized with:

```bash
python math/scripts/eval/summarize_all_math_evals.py
python math/scripts/eval/summarize_final_math_evals.py
```

## Validate

Validate README commands, script compatibility, experiment profiles, public Hub
datasets, and all eight published models without starting training or inference:

```bash
python math/scripts/validate_math_pipeline.py
```

Skip live Hugging Face checks when offline:

```bash
python math/scripts/validate_math_pipeline.py --skip-hub
```

## Directory Layout

```text
math/
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- experiments.json              # model, evaluation, training, and merge mapping
|-- data/                             # manifests and optional HF downloads
|-- figures/                          # report plots
|-- outputs/                          # runtime models and evaluation results
`-- scripts/
    |-- download_data.py              # optional prepared-data download
    |-- hf_dataset_utils.py           # shared automatic HF dataset loading
    |-- push_math_submission_model.py # validated model upload
    |-- validate_math_pipeline.py     # local and live-Hub reproducibility checks
    |-- eval/                         # vLLM evaluation and summaries
    `-- v1_early_sft/ ... v7_100k/   # preparation, training, and merge scripts
```
