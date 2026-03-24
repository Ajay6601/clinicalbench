import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)

def main():
    print("[LOAD] Loading MedQA dataset from HuggingFace...")
    # MedQA: USMLE-style 4-option medical questions with gold answers
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

    # Sample 200 diverse questions
    indices = random.sample(range(len(ds)), min(200, len(ds)))
    samples = [ds[i] for i in indices]

    output_path = Path("data/medqa_200.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sample in enumerate(samples):
        # Extract question, options, and correct answer
        options = sample["options"]  # dict like {"A": "...", "B": "...", ...}
        answer_key = sample["answer_idx"]  # e.g. "A"
        gold_answer = options[answer_key]

        # Format options as readable string for LLM context
        options_str = "\n".join(
            [f"  {k}. {v}" for k, v in sorted(options.items())]
        )

        record = {
            "id": f"medqa_{i:03d}",
            "question": sample["question"],
            "options": options,
            "options_formatted": options_str,
            "answer_key": answer_key,
            "gold_answer": gold_answer,
            "meta_info": sample.get("meta_info", ""),
        }
        records.append(record)

    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Saved {len(records)} questions to {output_path}")

    # Print a sample for sanity check
    print("\n--- Sample question ---")
    s = records[0]
    print(f"Q: {s['question'][:200]}...")
    print(f"Options:\n{s['options_formatted']}")
    print(f"Gold: ({s['answer_key']}) {s['gold_answer']}")


if __name__ == "__main__":
    main()
