import json
from pathlib import Path

inp = Path("validation_samples/math.jsonl")
out = Path("math/data/validation_math_converted.jsonl")

rows = []
skipped = 0

with inp.open(encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if not line.strip():
            continue

        r = json.loads(line)

        prompt = r.get("prompt")
        answer = r.get("answer")

        if not prompt or answer in [None, ""]:
            skipped += 1
            continue

        rows.append({
            "messages": [
                {
                    "role": "user",
                    "content": str(prompt).strip()
                }
            ],
            "metadata": {
                "benchmark_eval_source": "teacher_validation_math",
                "source": "validation_samples/math.jsonl",
                "id": f"teacher_math_{idx}",
                "answer": str(answer).strip()
            }
        })

out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Input:", inp)
print("Output:", out)
print("Written:", len(rows))
print("Skipped:", skipped)

if rows:
    print("\nFirst converted row:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:1500])
