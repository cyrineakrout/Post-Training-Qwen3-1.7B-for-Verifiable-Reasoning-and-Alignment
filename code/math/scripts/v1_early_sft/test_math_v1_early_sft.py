import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[3]
MATH_DIR = ROOT_DIR / "math"
OUTPUT_DIR = MATH_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "math_v1_early_sft_merged"
VALIDATION_FILE = ROOT_DIR / "validation_samples" / "math.jsonl"

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed(text: str):
    match = BOXED_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def load_validation_samples(limit=3):
    samples = []
    with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    samples = load_validation_samples(limit=3)

    for i, sample in enumerate(samples, start=1):
        prompt = sample["prompt"]
        expected = sample["answer"]

        messages = [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem. "
                    "Keep the reasoning concise. "
                    "You must finish with exactly one final answer using LaTeX boxed notation. "
                    "Do not stop before writing the final boxed answer.\n\n"
                    f"{prompt}"
                ),
            }
        ]

        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=3072,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=False)

        boxed = extract_boxed(completion)

        print("=" * 80)
        print(f"Sample {i}")
        print("Prompt:")
        print(prompt)
        print("\nExpected:")
        print(expected)
        print("\nExtracted boxed answer:")
        print(boxed)
        print("\nBoxed format OK:")
        print(boxed is not None)
        print("\nGenerated completion only:")
        print(completion)


if __name__ == "__main__":
    main()