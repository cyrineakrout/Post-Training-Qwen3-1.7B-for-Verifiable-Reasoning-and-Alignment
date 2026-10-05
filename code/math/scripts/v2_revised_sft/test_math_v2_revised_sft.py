import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[3]
MATH_DIR = ROOT_DIR / "math"
OUTPUT_DIR = MATH_DIR / "outputs"

MODEL_DIR = OUTPUT_DIR / "math_v2_revised_sft_merged"
VALIDATION_FILE = ROOT_DIR / "validation_samples" / "math.jsonl"


def extract_boxed(text: str):
    """
    Extract last \boxed{...}, including nested braces like \boxed{\frac{3}{4}}.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    answer = []

    while i < len(text) and depth > 0:
        char = text[i]

        if char == "{":
            depth += 1
            answer.append(char)
        elif char == "}":
            depth -= 1
            if depth > 0:
                answer.append(char)
        else:
            answer.append(char)

        i += 1

    if depth == 0:
        return "".join(answer).strip()

    return None


def load_validation_samples(limit=5):
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

    samples = load_validation_samples(limit=5)

    boxed_ok_count = 0
    exact_match_count = 0

    for i, sample in enumerate(samples, start=1):
        prompt = sample["prompt"]
        expected = str(sample["answer"]).strip()

        messages = [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem clearly and concisely. "
                    "Always end with exactly one final answer in \\boxed{}.\n\n"
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
            temperature=0.3,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=False)

        boxed = extract_boxed(completion)
        boxed_ok = boxed is not None
        exact = boxed == expected if boxed_ok else False

        boxed_ok_count += int(boxed_ok)
        exact_match_count += int(exact)

        print("=" * 80)
        print(f"Sample {i}")
        print("Expected:")
        print(expected)
        print("\nExtracted boxed answer:")
        print(boxed)
        print("\nBoxed format OK:")
        print(boxed_ok)
        print("\nExact match:")
        print(exact)
        print("\nGenerated completion only:")
        print(completion)

    print("=" * 80)
    print("SUMMARY")
    print(f"Boxed rate: {boxed_ok_count}/{len(samples)}")
    print(f"Exact match: {exact_match_count}/{len(samples)}")


if __name__ == "__main__":
    main()