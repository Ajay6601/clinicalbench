import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import mannwhitneyu


def main():
    scores_path = Path("results/eval_scores.jsonl")
    if not scores_path.exists():
        print("Run Day 1 pipeline first!")
        return

    all_scores = [json.loads(l) for l in open(scores_path)]
    print(f"Loaded {len(all_scores)} judge evaluations")

    # Group by dimension
    by_dim = defaultdict(lambda: {"correct": [], "incorrect": []})

    for s in all_scores:
        if s["score"] < 1:
            continue
        bucket = "correct" if s["is_correct"] else "incorrect"
        by_dim[s["dimension"]][bucket].append(s["score"])

    results = {}

    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        correct = np.array(by_dim[dim]["correct"])
        incorrect = np.array(by_dim[dim]["incorrect"])

        if len(correct) < 5 or len(incorrect) < 5:
            continue

        # Mann-Whitney U test: are the distributions significantly different?
        u_stat, p_value = mannwhitneyu(correct, incorrect, alternative="greater")

        # Effect size (rank-biserial correlation)
        n1, n2 = len(correct), len(incorrect)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)

        # Threshold analysis: for each score threshold, compute
        # precision/recall for detecting INCORRECT answers
        threshold_analysis = []
        for threshold in [1, 2, 3, 4]:
            # "Flagged" = score <= threshold (judge thinks something is wrong)
            flagged_correct = int((correct <= threshold).sum())
            flagged_incorrect = int((incorrect <= threshold).sum())
            total_flagged = flagged_correct + flagged_incorrect
            total_incorrect = len(incorrect)

            precision = flagged_incorrect / total_flagged if total_flagged > 0 else 0
            recall = flagged_incorrect / total_incorrect if total_incorrect > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0)

            threshold_analysis.append({
                "threshold": f"score <= {threshold}",
                "flagged_correct": flagged_correct,
                "flagged_incorrect": flagged_incorrect,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
            })

        results[dim] = {
            "correct_mean": round(float(correct.mean()), 2),
            "correct_std": round(float(correct.std()), 2),
            "incorrect_mean": round(float(incorrect.mean()), 2),
            "incorrect_std": round(float(incorrect.std()), 2),
            "separation": round(float(correct.mean() - incorrect.mean()), 2),
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "mann_whitney_p": round(float(p_value), 6),
            "effect_size": round(float(effect_size), 3),
            "is_significant": p_value < 0.001,
            "threshold_analysis": threshold_analysis,
        }

    # Save results (convert numpy types for JSON)
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = Path("results/correctness_calibration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"[OK] Saved to {out_path}")

    # Generate report
    lines = [
        "# ClinicalBench — Correctness-Based Judge Calibration",
        f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "\n## Methodology",
        "\nSince MedQA provides gold-standard correct answers, we can test judge "
        "validity without human annotation: do judges assign significantly lower "
        "scores to responses where the model selected the WRONG answer?",
        "\n## Score separation: correct vs incorrect answers\n",
        "| Dimension | Correct (mean) | Incorrect (mean) | Separation | p-value | Significant |",
        "|-----------|---------------|-------------------|------------|---------|-------------|",
    ]

    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim not in results:
            continue
        d = results[dim]
        sig = "Yes (p<0.001)" if d["is_significant"] else f"No (p={d['mann_whitney_p']})"
        lines.append(
            f"| {dim} | {d['correct_mean']} (±{d['correct_std']}) | "
            f"{d['incorrect_mean']} (±{d['incorrect_std']}) | "
            f"{d['separation']:+.2f} | {d['mann_whitney_p']:.6f} | {sig} |"
        )

    lines.extend([
        "\n## Error detection: using judge scores as CI gates\n",
        "If we flag responses where judge score <= threshold, how well do we detect "
        "actually-incorrect answers?\n",
    ])

    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim not in results:
            continue
        lines.append(f"\n### {dim.capitalize()}\n")
        lines.append("| Threshold | Precision | Recall | F1 |")
        lines.append("|-----------|-----------|--------|-----|")
        for t in results[dim]["threshold_analysis"]:
            lines.append(
                f"| {t['threshold']} | {t['precision']:.1%} | "
                f"{t['recall']:.1%} | {t['f1']:.3f} |"
            )

    lines.extend([
        "\n## Interpretation for CI/CD gates",
        "\n- **High precision, low recall** (strict threshold like <= 1): "
        "Few false alarms, but misses many bad answers. Good for blocking "
        "only the most dangerous outputs.",
        "- **Lower precision, high recall** (loose threshold like <= 3): "
        "Catches more bad answers but also flags some correct ones. "
        "Good for flagging outputs for human review.",
        "- **Optimal threshold**: Pick the threshold where F1 peaks per dimension. "
        "This is the recommended CI gate setting.",
        "\n## Key finding",
        "\n_If separation is > 1.0 and p < 0.001, the judge dimension is a "
        "statistically valid signal for detecting incorrect clinical responses. "
        "Dimensions with separation < 0.5 need improved judge prompts or "
        "should not be used as automated gates._",
    ])

    report_path = Path("results/correctness_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[OK] Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("CORRECTNESS CALIBRATION RESULTS")
    print("=" * 60)
    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim not in results:
            continue
        d = results[dim]
        sig = "***" if d["is_significant"] else ""
        print(f"\n  {dim}:")
        print(f"    Correct answers:   {d['correct_mean']} ± {d['correct_std']}")
        print(f"    Incorrect answers: {d['incorrect_mean']} ± {d['incorrect_std']}")
        print(f"    Separation:        {d['separation']:+.2f} {sig}")

        # Find best F1 threshold
        best = max(d["threshold_analysis"], key=lambda t: t["f1"])
        print(f"    Best CI gate:      {best['threshold']} "
              f"(P={best['precision']:.0%}, R={best['recall']:.0%}, F1={best['f1']:.3f})")


if __name__ == "__main__":
    main()