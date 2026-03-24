import json
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── LLM Clients ──────────────────────────────────────────────

from openai import OpenAI
from anthropic import Anthropic

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ant = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a medical expert answering a USMLE-style clinical question.
Provide your answer in this exact format:

ANSWER: [letter]
EXPLANATION: [2-4 sentence clinical reasoning for your choice]

Be specific. Reference relevant pathophysiology, pharmacology, or clinical guidelines."""


def call_openai(model: str, question: str, options_str: str) -> dict:
    """Call OpenAI API with retry logic."""
    user_msg = f"Question: {question}\n\nOptions:\n{options_str}\n\nProvide your answer."
    for attempt in range(3):
        try:
            resp = oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=300,
                temperature=0.0,
            )
            return {
                "raw_response": resp.choices[0].message.content,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                },
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return {"raw_response": f"ERROR: {e}", "usage": {}}


def call_anthropic(question: str, options_str: str) -> dict:
    """Call Anthropic Claude API with retry logic."""
    user_msg = f"Question: {question}\n\nOptions:\n{options_str}\n\nProvide your answer."
    for attempt in range(3):
        try:
            resp = ant.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0.0,
            )
            text = resp.content[0].text
            return {
                "raw_response": text,
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            }
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
            else:
                return {"raw_response": f"ERROR: {e}", "usage": {}}


def parse_answer_letter(raw: str) -> str:
    """Extract the answer letter from model response."""
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            letter = line.split(":", 1)[1].strip().upper()
            # Extract just the letter
            for ch in letter:
                if ch in "ABCD":
                    return ch
    # Fallback: look for standalone letter
    for ch in raw[:50]:
        if ch in "ABCD":
            return ch
    return "UNKNOWN"


# ── Model configs ────────────────────────────────────────────

MODELS = [
    {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
    },
    {
        "name": "claude-sonnet",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
    },
    {
        "name": "gpt-3.5-turbo",
        "provider": "openai",
        "model_id": "gpt-3.5-turbo",
    },
]


def main():
    # Load questions
    data_path = Path("data/medqa_200.jsonl")
    questions = [json.loads(line) for line in open(data_path)]
    print(f"[INFO] Loaded {len(questions)} questions")

    output_path = Path("data/model_answers.jsonl")
    
    # Resume support: load existing answers
    existing = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing.add((r["question_id"], r["model_name"]))
        print(f"⏩ Resuming — {len(existing)} answers already collected")

    total_cost = 0.0
    with open(output_path, "a") as out_f:
        for model_cfg in MODELS:
            name = model_cfg["name"]
            print(f"\n[MODEL] Generating answers with {name}...")

            for q in tqdm(questions, desc=name):
                if (q["id"], name) in existing:
                    continue

                if model_cfg["provider"] == "openai":
                    result = call_openai(
                        model_cfg["model_id"],
                        q["question"],
                        q["options_formatted"],
                    )
                else:
                    result = call_anthropic(
                        q["question"],
                        q["options_formatted"],
                    )

                predicted_letter = parse_answer_letter(result["raw_response"])
                is_correct = predicted_letter == q["answer_key"]

                record = {
                    "question_id": q["id"],
                    "model_name": name,
                    "question": q["question"],
                    "options": q["options"],
                    "gold_answer_key": q["answer_key"],
                    "gold_answer_text": q["gold_answer"],
                    "predicted_key": predicted_letter,
                    "is_correct": is_correct,
                    "raw_response": result["raw_response"],
                    "usage": result.get("usage", {}),
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                # Rate limit buffer
                time.sleep(0.3)

    # Summary stats
    all_answers = [json.loads(l) for l in open(output_path)]
    print("\n" + "=" * 50)
    print("[STATS] ANSWER GENERATION SUMMARY")
    print("=" * 50)
    for m in MODELS:
        name = m["name"]
        model_answers = [a for a in all_answers if a["model_name"] == name]
        correct = sum(1 for a in model_answers if a["is_correct"])
        total = len(model_answers)
        acc = correct / total * 100 if total > 0 else 0
        print(f"  {name:20s}: {correct}/{total} correct ({acc:.1f}%)")
    
    print(f"\n[OK] Total answers saved: {len(all_answers)} → {output_path}")


if __name__ == "__main__":
    main()