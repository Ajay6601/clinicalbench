import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent dir to path so we can import judges
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from judges import evaluate_response


def main():
    # Load model answers
    answers_path = Path("data/model_answers.jsonl")
    if not answers_path.exists():
        print("[FAIL] Run scripts/generate_answers.py first!")
        return

    answers = [json.loads(line) for line in open(answers_path)]
    print(f"[INFO] Loaded {len(answers)} model answers to evaluate")

    # Resume support
    output_path = Path("results/eval_scores.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                evaluated.add((r["question_id"], r["model_name"]))
        print(f"⏩ Resuming — {len(evaluated)} already evaluated")

    remaining = [a for a in answers if (a["question_id"], a["model_name"]) not in evaluated]
    print(f"[PROGRESS] {len(remaining)} answers remaining to evaluate")

    # Run judges
    start_time = time.time()
    with open(output_path, "a") as out_f:
        for i, answer in enumerate(tqdm(remaining, desc="Evaluating")):
            judge_results = evaluate_response(
                question=answer["question"],
                gold_answer=answer["gold_answer_text"],
                model_response=answer["raw_response"],
            )

            for jr in judge_results:
                record = {
                    "question_id": answer["question_id"],
                    "model_name": answer["model_name"],
                    "predicted_key": answer["predicted_key"],
                    "gold_answer_key": answer["gold_answer_key"],
                    "is_correct": answer["is_correct"],
                    **jr,
                    "judge_model": "gpt-4o-mini",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                out_f.write(json.dumps(record) + "\n")

            out_f.flush()

            # Progress checkpoint every 50
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                print(f"\n  ⏱ {i+1}/{len(remaining)} done | {rate:.0f}/min | "
                      f"ETA: {(len(remaining) - i - 1) / rate:.0f} min")

    # ── Generate summary stats ─────────────────────────────

    print("\n[STATS] Generating summary statistics...")

    all_scores = [json.loads(l) for l in open(output_path)]

    # Aggregate: model × dimension → list of scores
    stats = defaultdict(lambda: defaultdict(list))
    for s in all_scores:
        if s["score"] > 0:  # skip errors
            stats[s["model_name"]][s["dimension"]].append(s["score"])

    # Compute summary
    summary = {}
    for model, dims in stats.items():
        summary[model] = {}
        for dim, scores in dims.items():
            import numpy as np
            arr = np.array(scores)
            summary[model][dim] = {
                "mean": round(float(arr.mean()), 2),
                "std": round(float(arr.std()), 2),
                "median": float(np.median(arr)),
                "count": len(scores),
                "score_distribution": {
                    str(i): int((arr == i).sum()) for i in range(1, 6)
                },
            }

    # Overall accuracy per model
    for model in stats:
        model_answers_subset = [a for a in answers if a["model_name"] == model]
        correct = sum(1 for a in model_answers_subset if a["is_correct"])
        total = len(model_answers_subset)
        summary[model]["accuracy"] = {
            "correct": correct,
            "total": total,
            "pct": round(correct / total * 100, 1) if total > 0 else 0,
        }

    summary_path = Path("results/summary_stats.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] Summary saved to {summary_path}")

    # ── Generate markdown report ───────────────────────────

    report_lines = [
        "# ClinicalBench — Day 1 Evaluation Report",
        f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"\nDataset: MedQA (USMLE-style), n={len(answers)//3} questions, "
        f"{len(answers)} model responses",
        f"\nJudge model: GPT-4o-mini",
        "\n## Model Accuracy (Answer Selection)\n",
        "| Model | Correct | Total | Accuracy |",
        "|-------|---------|-------|----------|",
    ]

    for model in sorted(summary.keys()):
        acc = summary[model]["accuracy"]
        report_lines.append(
            f"| {model} | {acc['correct']} | {acc['total']} | {acc['pct']}% |"
        )

    report_lines.extend([
        "\n## Judge Scores by Model × Dimension\n",
        "| Model | Factuality | Safety | Hallucination | Completeness |",
        "|-------|-----------|--------|---------------|-------------|",
    ])

    for model in sorted(summary.keys()):
        dims = summary[model]
        row = f"| {model}"
        for d in ["factuality", "safety", "hallucination", "completeness"]:
            if d in dims:
                row += f" | {dims[d]['mean']:.2f} ± {dims[d]['std']:.2f}"
            else:
                row += " | N/A"
        row += " |"
        report_lines.append(row)

    # Score distributions
    report_lines.extend([
        "\n## Score Distributions\n",
    ])
    for model in sorted(summary.keys()):
        report_lines.append(f"\n### {model}\n")
        report_lines.append("| Dimension | 1 | 2 | 3 | 4 | 5 |")
        report_lines.append("|-----------|---|---|---|---|---|")
        for d in ["factuality", "safety", "hallucination", "completeness"]:
            if d in summary[model]:
                dist = summary[model][d]["score_distribution"]
                report_lines.append(
                    f"| {d} | {dist.get('1',0)} | {dist.get('2',0)} | "
                    f"{dist.get('3',0)} | {dist.get('4',0)} | {dist.get('5',0)} |"
                )

    # Flagged examples
    report_lines.extend([
        "\n## Notable Flagged Examples\n",
        "Low-scoring responses with specific flagged spans:\n",
    ])

    flagged = [s for s in all_scores if s["score"] <= 2 and s["flagged_spans"]]
    flagged.sort(key=lambda x: x["score"])
    for f_item in flagged[:10]:  # top 10 worst
        report_lines.append(
            f"- **{f_item['model_name']}** | {f_item['dimension']} "
            f"(score={f_item['score']}): {f_item['rationale'][:150]}..."
        )
        if f_item["flagged_spans"]:
            spans = ", ".join(f'"{s}"' for s in f_item["flagged_spans"][:3])
            report_lines.append(f"  - Flagged: {spans}")

    report_lines.extend([
        "\n## Methodology\n",
        "- **Dataset**: 200 USMLE-style questions from MedQA, sampled randomly (seed=42)",
        "- **Models evaluated**: GPT-4o-mini, Claude Sonnet, GPT-3.5-turbo",
        "- **Judge**: GPT-4o-mini with structured rubric prompts per dimension",
        "- **Dimensions**: Factuality (source grounding), Safety (harm potential), "
        "Hallucination (fabrication detection), Completeness (coverage)",
        "- **Scoring**: 1-5 Likert scale with mandatory chain-of-thought rationale",
        "\n## Limitations\n",
        "- Judge model (GPT-4o-mini) may have systematic biases; "
        "Day 2 adds human calibration",
        "- Single judge model; Day 2 ablation compares GPT-4o vs Claude as judge",
        "- 200 questions is sufficient for signal but not publication-grade sample size",
        "- No RAG retrieval component — evaluating direct LLM output only",
    ])

    report_path = Path("results/day1_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"[OK] Report saved to {report_path}")

    # Print highlights
    print("\n" + "=" * 60)
    print("[STATS] DAY 1 RESULTS HIGHLIGHTS")
    print("=" * 60)
    for model in sorted(summary.keys()):
        acc = summary[model]["accuracy"]
        print(f"\n  {model}:")
        print(f"    Accuracy: {acc['pct']}%")
        for d in ["factuality", "safety", "hallucination", "completeness"]:
            if d in summary[model]:
                print(f"    {d:15s}: {summary[model][d]['mean']:.2f} "
                      f"(±{summary[model][d]['std']:.2f})")

    elapsed_total = time.time() - start_time
    print(f"\n⏱ Total judge runtime: {elapsed_total/60:.1f} minutes")
    print(f"[OK] Day 1 complete! All outputs in results/")


if __name__ == "__main__":
    main()