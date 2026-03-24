import json
import csv
import os
import sys
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from anthropic import Anthropic

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ant = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DIMENSIONS = ["factuality", "safety", "hallucination", "completeness"]

from judges import (
    FACTUALITY_SYSTEM, SAFETY_SYSTEM, HALLUCINATION_SYSTEM,
    COMPLETENESS_SYSTEM, parse_judge_output,
)

SYSTEMS = {
    "factuality": FACTUALITY_SYSTEM,
    "safety": SAFETY_SYSTEM,
    "hallucination": HALLUCINATION_SYSTEM,
    "completeness": COMPLETENESS_SYSTEM,
}

# ── Reference judge configs ──────────────────────────────────

REFERENCE_JUDGES = [
    {
        "name": "gpt-4o",
        "provider": "openai",
        "model_id": "gpt-4o",
        "output_file": "results/human_annotations.csv",  # calibrate.py reads this
    },
    {
        "name": "claude-sonnet",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
        "output_file": "results/claude_reference_annotations.csv",
    },
]


def call_reference_judge(provider: str, model_id: str, system: str, user: str, dim: str):
    """Call a reference judge model."""
    for attempt in range(3):
        try:
            if provider == "anthropic":
                resp = ant.messages.create(
                    model=model_id,
                    max_tokens=500,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=0.0,
                )
                raw = resp.content[0].text
            else:
                resp = oai.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content

            return parse_judge_output(raw, dim)
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                from judges import JudgeResult
                return JudgeResult(dim, -1, f"ERROR: {e}", [], 0.0, str(e))


def select_diverse_samples(answers_path: Path, n: int, seed: int) -> list:
    """Select diverse samples across models and correctness."""
    all_answers = [json.loads(l) for l in open(answers_path)]
    random.seed(seed)

    by_model = {}
    for a in all_answers:
        by_model.setdefault(a["model_name"], []).append(a)

    samples = []
    per_model = max(1, n // len(by_model))

    for model_name, model_answers in by_model.items():
        correct = [a for a in model_answers if a["is_correct"]]
        incorrect = [a for a in model_answers if not a["is_correct"]]

        n_correct = min(len(correct), per_model // 2)
        n_incorrect = min(len(incorrect), per_model - n_correct)

        random.shuffle(correct)
        random.shuffle(incorrect)
        samples.extend(correct[:n_correct])
        samples.extend(incorrect[:n_incorrect])

    random.shuffle(samples)
    return samples[:n]


def load_existing(path: Path) -> set:
    """Load already-annotated keys."""
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["question_id"], row["model_name"]))
    return done


def run_reference_judge(judge_cfg: dict, samples: list):
    """Run one reference judge across all samples."""
    output_path = Path(judge_cfg["output_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_existing(output_path)
    remaining = [s for s in samples if (s["question_id"], s["model_name"]) not in done]

    name = judge_cfg["name"]
    print(f"\n{'='*50}")
    print(f"Reference judge: {name}")
    print(f"  {len(samples)} total, {len(done)} done, {len(remaining)} remaining")
    print(f"{'='*50}")

    if not remaining:
        print(f"  All samples already annotated by {name}")
        return

    fieldnames = [
        "question_id", "model_name", "dimension",
        "human_score", "human_notes", "is_correct", "timestamp",
    ]
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for sample in tqdm(remaining, desc=f"Judging ({name})"):
            for dim in DIMENSIONS:
                user_prompt = f"""QUESTION: {sample['question']}

GOLD ANSWER: {sample['gold_answer_text']}

MODEL RESPONSE: {sample['raw_response']}

Judge the response according to your rubric."""

                result = call_reference_judge(
                    judge_cfg["provider"],
                    judge_cfg["model_id"],
                    SYSTEMS[dim],
                    user_prompt,
                    dim,
                )

                writer.writerow({
                    "question_id": sample["question_id"],
                    "model_name": sample["model_name"],
                    "dimension": dim,
                    "human_score": result.score,
                    "human_notes": f"[{name} reference] {result.rationale[:200]}",
                    "is_correct": sample["is_correct"],
                    "timestamp": datetime.utcnow().isoformat(),
                })

                time.sleep(0.3)

            out_f.flush()

    print(f"  [OK] {name} annotations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Samples per reference judge (default: 100)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--only", choices=["gpt-4o", "claude-sonnet"],
                        help="Run only one reference judge")
    args = parser.parse_args()

    answers_path = Path("data/model_answers.jsonl")
    if not answers_path.exists():
        print("Run Day 1 pipeline first!")
        return

    # Use the SAME samples for both judges — critical for valid comparison
    samples = select_diverse_samples(answers_path, args.n, args.seed)
    print(f"Selected {len(samples)} diverse samples for reference judging")

    # Show sample composition
    by_model = {}
    for s in samples:
        by_model.setdefault(s["model_name"], {"correct": 0, "incorrect": 0})
        if s["is_correct"]:
            by_model[s["model_name"]]["correct"] += 1
        else:
            by_model[s["model_name"]]["incorrect"] += 1

    print("\nSample composition:")
    for m, counts in sorted(by_model.items()):
        print(f"  {m}: {counts['correct']} correct, {counts['incorrect']} incorrect")

    # Run reference judges
    for judge_cfg in REFERENCE_JUDGES:
        if args.only and judge_cfg["name"] != args.only:
            continue
        run_reference_judge(judge_cfg, samples)

    print(f"\n{'='*50}")
    print("REFERENCE ANNOTATION COMPLETE")
    print(f"{'='*50}")
    print("\nOutputs:")
    for j in REFERENCE_JUDGES:
        if args.only and j["name"] != args.only:
            continue
        print(f"  {j['output_file']}")
    print(f"\nNext: python scripts/calibrate.py")
    print(f"Then: python scripts/three_way_comparison.py")


if __name__ == "__main__":
    main()