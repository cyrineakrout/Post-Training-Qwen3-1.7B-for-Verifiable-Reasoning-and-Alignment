# CS-552 MNLP Spring 2026 — Project Starter

Welcome to the EPFL **CS-552 Modern Natural Language Processing** course project. Over the next ~5 weeks your team of 4 will post-train **Qwen3-1.7B** into a small family of reasoning models and submit them to the course leaderboard. The project is **70% of your final grade**.

This repository contains everything you need to get started: required model layout, evaluation contract, validation snapshot, and a description of the CI pipeline that will score your work.

## Project Timeline, Milestones & Deliverables

Please read the [project description](https://docs.google.com/document/d/1TECHv4q_eR0X-HIyW10vHFbcU2bHLSph/edit?usp=sharing&ouid=109194228875252004302&rtpof=true&sd=true) for the details.

## Team setup

### 1. Create your HuggingFace organization

Use your **team name verbatim as the org slug**.

1. Sign in to [huggingface.co](https://huggingface.co/) (each teammate needs an account).
2. Click your avatar (top-right) → **New Organization**.
3. Set **Organization name** *and* **Organization handle** to your team name (e.g. team `Helvetica Roman` → handle `helvetica_roman`, URL `https://huggingface.co/helvetica`).
4. Invite all 4 teammates with **Write** access.
5. Keep all your repos under this organization as public.

### 2. Required model repositories (exactly 5 per team)

Create these five **public** model repos under your team org. Names must match **exactly** — the CI pipeline looks them up by these slugs.

| Repo path | Owner | Evaluated on |
|---|---|---|
| `cs-552-2026-<your-org>/group_model` | whole team | all 4 benchmarks |
| `cs-552-2026-<your-org>/math_model` | one teammate | math |
| `cs-552-2026-<your-org>/general_knowledge_model` | one teammate | general knowledge |
| `cs-552-2026-<your-org>/safety_model` | one teammate | safety |
| `cs-552-2026-<your-org>/multilingual_model` | one teammate | multilinguality |

The four specialty models must each have a different student owner — every team member owns exactly one specialty model and contributes to the group model.

### 3. Per-checkpoint requirements

Every checkpoint you push must satisfy all of the following, or it will fail validation:

- **Starting model:** `Qwen/Qwen3-1.7B`. The starting model cannot be swapped — switching architectures will fail the checkpoint loading and forfeit the milestone.
- **Format:** vLLM-loadable safetensors; `config.json` and weights at the root of the repo. See the starting model repo for details.
- **`generation_config.json`** must be committed alongside the weights. You may tune `temperature`, `top_k`, `top_p`.
- **Tokenizer:** must have a `chat_template` file. The CI sends prompts through `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` — **nothing else is passed**, so any prompt-construction preference must be encoded inside the template itself.
- **Output contract:** every answer must be wrapped in `\boxed{...}`. Anything outside the box is treated as reasoning and not evaluated.

#### Thinking vs. non-thinking mode

Qwen3 supports a "thinking" mode (the model emits a `<think>...</think>` block of reasoning before its real answer) and a non-thinking mode (no `<think>` block). **Pick one and bake it into your `chat_template.jinja`** — the CI will not pass `enable_thinking` as a keyword argument, so a runtime default is the only signal we honor.

The shipped Qwen3 chat template defaults to `enable_thinking=True`. To override that default, edit the Jinja in your tokenizer's `chat_template.jinja` (or the `chat_template` field in `tokenizer_config.json`).

**Force thinking ON** — add this near the top of your `chat_template.jinja`, before any `{% if enable_thinking %}` block:

```jinja
{%- set enable_thinking = true %}
```

**Force thinking OFF** — same idea, with `false`, *and* make sure the empty `<think></think>` block expected by the Qwen3 template is still emitted (otherwise generation drifts):

```jinja
{%- set enable_thinking = false %}
```

Alternatively, for the OFF case you can append `/no_think` to your system message inside the template (Qwen's "soft switch"):

```jinja
{%- set system_message = system_message + ' /no_think' %}
```

Verify your choice locally before pushing:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./your_checkpoint_dir")
print(tok.apply_chat_template(
    [{"role": "user", "content": "What is 2+2?"}],
    tokenize=False,
    add_generation_prompt=True,
))
```

The output should show (or not show) the `<think>` opener according to your choice — if the wrong mode appears here, it will appear in CI too.

You can iterate as often as you want — push a new commit and the next nightly run will pick it up.

---

## The CI evaluation pipeline

### Expacted date of CI online


### When it runs

- **Nightly at 23:59**, automatically.
- Models whose HF `lastModified` timestamp has **not** advanced since the previous run are skipped (no eval, no evaluation report PR). To trigger a re-eval, push an update.

### What it does, in order

1. **Freshness check** — query `huggingface_hub.repo_info(...).last_modified` for each model in the roster.
2. **Validation** — verify the repo exists, has `generation_config.json`, the tokenizer has `chat_template`, and the model loads in vLLM.
3. **Inference** — run vLLM batch inference, **n=8 completions per problem**, using your model's chat template and generation config. Wall-clock cap: 1800 s per model.
4. **Scoring** — extract answers via `\boxed{...}` parsing, compare to ground truth, compute the metrics.
5. **Reporting** — update the public leaderboard **and** open/update an evaluation-report PR on each re-evaluated model repo (see [Automatic evaluation reports](#automatic-evaluation-reports)).

### Hardware budget

- 1× NVIDIA A100 40 GB, FP16 inference.
- Max model length 8192. The generation stops once the `\boxed{...}` answer is generated or the model reaches an EoS token, or the maximmum length is reached.

### Metrics

- **mean@8** = mean correctness across the 8 sampled completions per question (i.e. expected per-attempt accuracy).
- **pass@8** = fraction of questions for which **at least one** of the 8 sampled completions is correct (unbiased pass@k from Chen et al., 2021).
- The group model gets an additional **4-domain average** that determines its leaderboard rank.

---

## Evaluation domains

The exact benchmark datasets are kept secret, but here is what to expect in each domain. Train accordingly.

### Math
Competition-style problems across **algebra, number theory, combinatorics, and geometry**. Free-form numeric or short algebraic answers. Expect questions that require multi-step reasoning before the final boxed answer.

### General knowledge
**Closed-book** factual and reasoning questions across the sciences, humanities, geography, and current events. **Multiple-choice.**

### Safety
Adversarial prompts that probe **refusal calibration**, harm categories, and jailbreak resistance. **Multiple-choice.**

### Multilinguality
Questions in **5 languages** spanning regional knowledge, civics, culture, and professional licensing material. **Multiple-choice.**

### Question formats — train for both

Your group model training mix needs to handle both styles below, while the individual model can stick to one particular format.

**Free-form** — the prompt is a question; your model writes its reasoning and wraps the final answer in `\boxed{...}`:

```
Q: What is the smallest prime greater than 100?
A: ...reasoning... \boxed{101}
```

**Multiple-choice** — the prompt lists labeled options; your model boxes the **letter** of the correct option:

```
Q: Which of the following ...?
A) ...
B) ...
C) ...
D) ...
E) ...
A: ...reasoning... \boxed{C}
```

> **Important — MC option counts vary from 2 to 20.**
> Some items have 2 options (`A, B`); others go up to 20 (`A` … `T`). If your prompt template hardcodes 4 options, or if your training data is all 4-way MC, your model will collapse on the long-tail items.
> Make sure your formatter accepts a variable number of options and that your training set covers the full 2–20 range.

---

## Validation snapshot (`validation/`)

We provide a frozen sample of **10 problems per benchmark** (40 total) so you can sanity-check your inference and prompt formatting **without HF auth and without any risk of leaking eval data**.

```
validation/
├── math.jsonl
├── general_knowledge.jsonl
├── safety.jsonl
├── multilingual.jsonl
└── README.md
```

These are **not** the eval set. Use them to:

- verify your inference produces an extractable `\boxed{...}`;
- exercise both free-form and multiple-choice paths in your prompt template;
- exercise the **variable option-count** behaviour described above;
- smoke-test before pushing a checkpoint to HF.

**Schema** — one JSON object per line, **two fields only**:

```json
{"prompt": "...", "answer": "42"}
```

For multiple-choice items the option list is folded into `prompt` with letter labels (`A) ...`, `B) ...`, …) and `answer` is a single capital letter. For free-form items `answer` is the expected text/number.

---

## Automatic evaluation reports

After each nightly run, **if your model was re-evaluated**, the CI bot opens (or updates an existing) **Pull Request** on your model repo's HuggingFace **Community** tab. The PR adds or replaces a file called `EVAL_REPORT.md` in your repo's root.

The report includes:

- Metric for each benchmark scored against this model;
- 1–2 sample correct completions and 1–2 sample incorrect completions (truncated);
- validation/inference error logs if anything failed.

You will find it at:

```
https://huggingface.co/cs-552-2026-<your-org>/<model_name>/discussions
```

> The PR is **non-blocking** — you do not need to merge it, and merging it does not affect your grade. Reading it is just the fastest debug loop.

If your model was **not** re-evaluated this round (because its `lastModified` did not change), no new PR is opened. Push something to your repo to trigger a re-evaluation on the next nightly run.

---

## Working loop during the project

```
                ┌──────────────────────┐
                │ 1. Train locally     │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ 2. Push to HF Hub    │
                │    (overwrite repo)  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ 3. Wait ≤24 h for    │
                │    nightly CI run    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ 4. Read EVAL_REPORT  │
                │    PR on your repo   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ 5. Iterate           │
                └──────────────────────┘
```

---

## Common failure modes

| Symptom in `EVAL_REPORT.md` | Likely cause |
|---|---|
| `validation: tokenizer has no chat_template` | Save the chat template alongside the tokenizer, e.g. `tokenizer.chat_template = ...; tokenizer.save_pretrained(...)` before pushing. |
| `validation: vLLM failed to load model` | Missing or malformed `config.json` / `generation_config.json`, or you accidentally pushed a non-Qwen3 architecture. |
| `pass@1 ≈ 0` but model "looks fine" locally | Output is missing `\boxed{...}`. Check your system prompt and reinforce boxed-output behaviour in SFT data. |
| `pass@1` collapses on multilingual or knowledge | Your prompt template hardcodes 4 options. Re-check the **2–20 option** requirement. |
| `inference: Timeout (1800s)` | Model is generating too many tokens. Reduce `max_new_tokens` *only* if your answers still complete; otherwise revisit decoding hyperparameters. |

---

## Resources

- Course page: see Moodle (CS-552, Spring 2026).
- HuggingFace Hub docs: <https://huggingface.co/docs/hub>.
- vLLM docs: <https://docs.vllm.ai>.
- Qwen3 model card: <https://huggingface.co/Qwen/Qwen3-1.7B>.

Good luck — and ship early.
