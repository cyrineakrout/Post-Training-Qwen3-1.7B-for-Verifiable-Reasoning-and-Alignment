# Code

This folder contains the code used to reproduce the benchmark experiments. Each benchmark area is kept in its own subfolder, with scripts, configs, data manifests, and a local README where applicable.

- `math/`: math benchmark data preparation, training, merging, evaluation, and result summarization scripts.
- `safety/`: safety benchmark data processing, supervised fine-tuning, DPO training, LoRA merging, and evaluation scripts.
- `general_knowledge/`: general knowledge benchmark data preparation, SFT/GRPO training, LoRA merging, and evaluation scripts.
- `group/`: group benchmark dataset construction, SFT/GRPO training, adapter merging, and evaluation scripts.
- `multilinguality/`: multilingual benchmark data preparation, distillation, SFT/GRPO training, model merging, and evaluation scripts.
- `evaluate/`: shared benchmark evaluation utilities, including scoring, answer extraction, benchmark definitions, and pass@k computation.

See the README inside each benchmark folder for more detailed reproduction instructions.
