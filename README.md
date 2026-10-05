[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HGMFhRpE)
# CS-552 MNLP Spring 2026 — Milestone 3

Welcome to the EPFL **CS-552 Modern Natural Language Processing** course project — final milestone. Over the next ~2 weeks, your team of 4 will finalize post-training **Qwen3-1.7B** into 5 reasoning models (math, safety, multilinguality, general knowledge, and the group model) and submit them to the course leaderboard along with a final project report. This milestone is worth **50% of your final grade**.

## Project Timeline, Milestones & Deliverables

Please read the [project description](https://docs.google.com/document/d/1TECHv4q_eR0X-HIyW10vHFbcU2bHLSph/edit?usp=sharing&ouid=109194228875252004302&rtpof=true&sd=true) for details.

## Final Report

Use the provided [LaTeX template](https://www.overleaf.com/read/fvsddxcjqssd#4dde29) for the final report, and push it to the `final_report` folder.

## Evaluation CI
The evaluation CI continues running nightly with a maximum length of 4k for one more week, i.e., until May 31. After that, to give you a more accurate estimate of your scores, the CI will run every 48 hours with the full 16k context length.

## Code
Push your code to the `code` folder. The code itself is not graded, but we will run it — so submit it in a runnable form with clear instructions in a README. We expect the performance metrics reported in your final report to closely match (if not exactly match) what we observe in our internal evaluation; fixing seeds helps with reproducibility.

For the models, as in the M2 release, we require your models in the following **public** repos under your team org. Names must match **exactly**, as the CI pipeline looks them up by these slugs.

| Repo path | Owner | Evaluated on |
|---|---|---|
| `cs-552-2026-<your-org>/group_model` | whole team | all 4 benchmarks |
| `cs-552-2026-<your-org>/math_model` | one teammate | math |
| `cs-552-2026-<your-org>/general_knowledge_model` | one teammate | general knowledge |
| `cs-552-2026-<your-org>/safety_model` | one teammate | safety |
| `cs-552-2026-<your-org>/multilingual_model` | one teammate | multilinguality |

`M2-archive` contains the template repository from the M2 release in case you need to refer back to any of its content.
