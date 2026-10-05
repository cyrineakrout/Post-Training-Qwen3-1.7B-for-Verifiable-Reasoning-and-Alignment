import json
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT_DIR = Path(__file__).resolve().parents[3]
VALIDATION_FILE = ROOT_DIR / "validation_samples" / "math.jsonl"
BASE_MODEL = "Qwen/Qwen3-1.7B"

BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed(text: str):
    match = BOXED_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def load_first_math_sample():
    with open(VALIDATION_FILE, "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def main():
    sample = load_first_math_sample()
    prompt = sample["prompt"]
    expected = sample["answer"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. "
                "Put the final answer inside \\boxed{}.\n\n"
                f"{prompt}"
            ),
        }
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print("\n===== CHAT TEMPLATE OUTPUT =====")
    print(chat_prompt[:2000])

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    print("\nGenerating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    boxed = extract_boxed(decoded)

    print("\n===== MODEL OUTPUT =====")
    print(decoded)

    print("\n===== EXPECTED ANSWER =====")
    print(expected)

    print("\n===== EXTRACTED BOXED ANSWER =====")
    print(boxed)

    print("\n===== BOXED FORMAT OK? =====")
    print(boxed is not None)


if __name__ == "__main__":
    main()