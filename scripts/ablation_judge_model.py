import json
import csv
import os
import sys
import time
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from anthropic import Anthropic

oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ant = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Import judge prompts from our module
from judges import (
    FACTUALITY_SYSTEM, SAFETY_SYSTEM, HALLUCINATION_SYSTEM, COMPLETENESS_SYSTEM,
    parse_judge_output,
)

JUDGE_SYSTEMS = {
    "factuality": FACTUALITY_SYSTEM,
    "safety": SAFETY_SYSTEM,
    "hallucination": HALLUCINATION_SYSTEM,
    "completeness": COMPLETENESS_SYSTEM,
}


def call_judge_with_model(model: str, system: str, user: str, dim: str):
    """Call a specific model as judge."""
    for attempt in range(3):
        try:
            if model.startswith("claude"):
                resp = ant.messages.create(
                    model=model,
                    max_tokens=500,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=0.0,
                )
                raw = resp.content[0].text
            else:
                resp = oai.chat.completions.create(
                    model=model,
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


def build_user_prompt(question: str, gold: str, response: str) -> str:
    return f"""QUESTION: {question}

GOLD ANSWER: {gold}

MODEL RESPONSE: {response}

Judge the response according to your rubric."""


# ── Experiment 1: Judge Model Comparison ──────────────────

def run_judge_comparison(judge_model: str, annotations: list, answers_by_key: dict):
    """Re-run judge on annotated samples with alternative model."""
    output_path = Path("results/ablation_scores.jsonl")

    # Get unique (question_id, model_name) pairs from annotations
    sample_keys = set()
    for ann in annotations:
        sample_keys.add((ann["question_id"], ann["model_name"]))

    # Resume support
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("judge_model") == judge_model:
                    done.add((r["question_id"], r["model_name"], r["dimension"]))

    remaining = []
    for qid, mname in sample_keys:
        key = (qid, mname)
        if key in answers_by_key:
            for dim in ["factuality", "safety", "hallucination", "completeness"]:
                if (qid, mname, dim) not in done:
                    remaining.append((answers_by_key[key], dim))

    print(f"\nRunning {len(remaining)} judge calls with {judge_model}...")
    print(f"({len(done)} already completed)")

    with open(output_path, "a") as out_f:
        for answer, dim in tqdm(remaining, desc=f"Judging ({judge_model})"):
            user_prompt = build_user_prompt(
                answer["question"], answer["gold_answer_text"], answer["raw_response"]
            )
            result = call_judge_with_model(
                judge_model, JUDGE_SYSTEMS[dim], user_prompt, dim
            )

            record = {
                "question_id": answer["question_id"],
                "model_name": answer["model_name"],
                "dimension": dim,
                "score": result.score,
                "rationale": result.rationale,
                "judge_model": judge_model,
                "timestamp": datetime.utcnow().isoformat(),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            time.sleep(0.3)


# ── Experiment 2: Hallucination Injection ─────────────────

INJECTION_TEMPLATES = [
    "According to the landmark {year} study published in {journal}, {claim}.",
    "This is consistent with the {mechanism} pathway, which involves {detail}.",
    "As recommended by the {org} guidelines ({year}), {recommendation}.",
]

FAKE_CLAIMS = [
    {"year": "2019", "journal": "The Lancet", "claim": "early administration of corticosteroids reduces mortality by 40% in this condition",
     "mechanism": "TGF-beta mediated apoptotic", "detail": "upregulation of IL-27 receptors",
     "org": "WHO", "recommendation": "first-line treatment should include azithromycin prophylaxis"},
    {"year": "2021", "journal": "NEJM", "claim": "genetic predisposition accounts for 85% of cases in patients under 30",
     "mechanism": "CRISPR-Cas9 regulatory", "detail": "methylation of the BRCA3 promoter region",
     "org": "AMA", "recommendation": "screening should begin at age 15 for high-risk populations"},
    {"year": "2020", "journal": "BMJ", "claim": "combination therapy with metformin reduces recurrence by 60%",
     "mechanism": "mTOR-dependent autophagic", "detail": "inhibition of the PDL-3 checkpoint",
     "org": "NIH", "recommendation": "prophylactic anticoagulation is indicated for all patients"},
]


def inject_hallucinations(response: str, n_injections: int) -> str:
    """Inject n fabricated claims into a model response."""
    random.seed(42 + n_injections)
    sentences = response.split(". ")
    if len(sentences) < 2:
        return response

    for i in range(min(n_injections, len(FAKE_CLAIMS))):
        template = random.choice(INJECTION_TEMPLATES)
        fake = FAKE_CLAIMS[i]
        injected = template.format(**fake)

        # Insert after a random sentence
        insert_pos = random.randint(1, max(1, len(sentences) - 1))
        sentences.insert(insert_pos, injected)

    return ". ".join(sentences)


def run_injection_experiment(answers_by_key: dict):
    """Test judge sensitivity to injected hallucinations."""
    output_path = Path("results/hallucination_injection_results.json")

    # Take 30 correct answers (clean baseline)
    all_answers = list(answers_by_key.values())
    correct = [a for a in all_answers if a["is_correct"]]
    random.seed(42)
    random.shuffle(correct)
    samples = correct[:30]

    results = {"baseline": [], "1_injection": [], "2_injections": [], "3_injections": []}

    print(f"\nRunning hallucination injection experiment on {len(samples)} samples...")

    for sample in tqdm(samples, desc="Injection test"):
        for n_inj, key in [(0, "baseline"), (1, "1_injection"),
                           (2, "2_injections"), (3, "3_injections")]:
            if n_inj == 0:
                test_response = sample["raw_response"]
            else:
                test_response = inject_hallucinations(sample["raw_response"], n_inj)

            user_prompt = build_user_prompt(
                sample["question"], sample["gold_answer_text"], test_response
            )
            result = call_judge_with_model(
                "gpt-4o-mini", HALLUCINATION_SYSTEM, user_prompt, "hallucination"
            )
            results[key].append(result.score)
            time.sleep(0.2)

    # Compute stats
    summary = {}
    for key, scores in results.items():
        valid = [s for s in scores if s > 0]
        if valid:
            arr = np.array(valid)
            summary[key] = {
                "mean": round(float(arr.mean()), 2),
                "std": round(float(arr.std()), 2),
                "scores": valid,
                "detection_rate": round(float((arr <= 3).mean()), 3),
            }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nHallucination injection results:")
    print(f"  {'Condition':<20} {'Mean score':<12} {'Detection rate (<=3)'}")
    for key in ["baseline", "1_injection", "2_injections", "3_injections"]:
        if key in summary:
            s = summary[key]
            print(f"  {key:<20} {s['mean']:<12.2f} {s['detection_rate']:.1%}")

    return summary


def generate_ablation_report(
    judge_model: str, human_anns: list,
    day1_scores: dict, ablation_scores: dict,
    injection_results: dict
):
    """Generate the combined ablation report."""
    lines = [
        "# ClinicalBench — Day 2 Ablation Report",
        f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "\n## Experiment 1: Judge model comparison",
        f"\nComparing GPT-4o-mini (Day 1 judge) vs {judge_model} (ablation judge) "
        "against human annotations.",
    ]

    # Compute kappa for both judges per dimension
    lines.extend([
        "\n| Dimension | GPT-4o-mini kappa | GPT-4o-mini bias | "
        f"{judge_model} kappa | {judge_model} bias |",
        "|-----------|-------------------|------------------|"
        f"{'--' * len(judge_model)}---------|{'--' * len(judge_model)}--------|",
    ])

    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        h_scores, j1_scores, j2_scores = [], [], []
        for ann in human_anns:
            h = int(ann["human_score"])
            if h < 1:
                continue
            key = (ann["question_id"], ann["model_name"], dim)
            j1 = day1_scores.get(key, -1)
            j2 = ablation_scores.get(key, -1)
            if j1 > 0 and j2 > 0:
                h_scores.append(h)
                j1_scores.append(j1)
                j2_scores.append(j2)

        if len(h_scores) >= 5:
            k1 = cohen_kappa_score(h_scores, j1_scores, weights="quadratic")
            k2 = cohen_kappa_score(h_scores, j2_scores, weights="quadratic")
            b1 = np.mean(np.array(j1_scores) - np.array(h_scores))
            b2 = np.mean(np.array(j2_scores) - np.array(h_scores))
            lines.append(
                f"| {dim} | {k1:.3f} | {b1:+.2f} | {k2:.3f} | {b2:+.2f} |"
            )

    # Injection experiment
    if injection_results:
        lines.extend([
            "\n## Experiment 2: Hallucination injection sensitivity",
            "\nTests whether the hallucination judge detects fabricated claims "
            "injected into correct responses at varying rates.",
            "\n| Condition | Mean score | Std | Detection rate (score <= 3) |",
            "|-----------|-----------|-----|----------------------------|",
        ])
        for key in ["baseline", "1_injection", "2_injections", "3_injections"]:
            if key in injection_results:
                s = injection_results[key]
                lines.append(
                    f"| {key} | {s['mean']:.2f} | {s['std']:.2f} | {s['detection_rate']:.1%} |"
                )

        lines.extend([
            "\nA good hallucination judge should show:",
            "- High baseline scores (clean responses score 4-5)",
            "- Monotonically decreasing scores as injections increase",
            "- Detection rate increasing toward 100% with more injections",
        ])

    report_path = Path("results/ablation_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nAblation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", default="gpt-4o",
                        help="Alternative judge model (default: gpt-4o). "
                        "Use 'claude-sonnet-4-20250514' for Claude.")
    parser.add_argument("--skip-injection", action="store_true",
                        help="Skip hallucination injection experiment")
    args = parser.parse_args()

    # Load data
    answers_path = Path("data/model_answers.jsonl")
    human_path = Path("results/human_annotations.csv")

    if not human_path.exists():
        print("Run annotate.py first!")
        return

    answers_by_key = {}
    for line in open(answers_path):
        a = json.loads(line)
        answers_by_key[(a["question_id"], a["model_name"])] = a

    with open(human_path) as f:
        human_anns = [r for r in csv.DictReader(f) if int(r["human_score"]) > 0]

    print(f"Loaded {len(human_anns)} human annotations")

    # Load Day 1 judge scores
    day1_scores = {}
    for line in open("results/eval_scores.jsonl"):
        r = json.loads(line)
        day1_scores[(r["question_id"], r["model_name"], r["dimension"])] = r["score"]

    # Experiment 1: Judge model comparison
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1: Judge model comparison ({args.judge_model})")
    print(f"{'='*60}")
    run_judge_comparison(args.judge_model, human_anns, answers_by_key)

    # Load ablation scores
    ablation_scores = {}
    if Path("results/ablation_scores.jsonl").exists():
        for line in open("results/ablation_scores.jsonl"):
            r = json.loads(line)
            if r.get("judge_model") == args.judge_model:
                ablation_scores[(r["question_id"], r["model_name"], r["dimension"])] = r["score"]

    # Experiment 2: Hallucination injection
    injection_results = {}
    if not args.skip_injection:
        print(f"\n{'='*60}")
        print("EXPERIMENT 2: Hallucination injection sensitivity")
        print(f"{'='*60}")
        injection_results = run_injection_experiment(answers_by_key)
    else:
        inj_path = Path("results/hallucination_injection_results.json")
        if inj_path.exists():
            injection_results = json.loads(inj_path.read_text())

    # Generate combined report
    generate_ablation_report(
        args.judge_model, human_anns,
        day1_scores, ablation_scores,
        injection_results,
    )

    print(f"\n{'='*60}")
    print("DAY 2 COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  results/human_annotations.csv")
    print(f"  results/calibration_report.md")
    print(f"  results/calibration_stats.json")
    print(f"  results/ablation_report.md")
    print(f"  results/ablation_scores.jsonl")
    print(f"  results/hallucination_injection_results.json")
    print(f"\nNext: Day 3 — Write the README, package the repo, send the email")


if __name__ == "__main__":
    main()