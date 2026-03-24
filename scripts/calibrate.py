import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def load_judge_scores(path: Path) -> dict:
    """Load judge scores, keyed by (question_id, model_name, dimension)."""
    scores = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["question_id"], r["model_name"], r["dimension"])
        scores[key] = r["score"]
    return scores


def load_human_annotations(path: Path) -> list[dict]:
    """Load human annotations."""
    annotations = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["human_score"]) > 0:  # skip skipped items
                annotations.append(row)
    return annotations


def compute_calibration(judge_scores: dict, annotations: list[dict]) -> dict:
    """Compute calibration metrics per dimension and overall."""
    # Group by dimension
    by_dim = defaultdict(lambda: {"human": [], "judge": []})
    all_human = []
    all_judge = []

    for ann in annotations:
        key = (ann["question_id"], ann["model_name"], ann["dimension"])
        if key not in judge_scores:
            continue

        h_score = int(ann["human_score"])
        j_score = judge_scores[key]

        if j_score < 1:  # skip judge errors
            continue

        by_dim[ann["dimension"]]["human"].append(h_score)
        by_dim[ann["dimension"]]["judge"].append(j_score)
        all_human.append(h_score)
        all_judge.append(j_score)

    results = {}

    # Per-dimension metrics
    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim not in by_dim or len(by_dim[dim]["human"]) < 5:
            results[dim] = {"error": "insufficient data"}
            continue

        h = np.array(by_dim[dim]["human"])
        j = np.array(by_dim[dim]["judge"])

        # Cohen's kappa (weighted, for ordinal data)
        kappa = cohen_kappa_score(h, j, weights="quadratic")

        # Spearman rank correlation
        spear_r, spear_p = spearmanr(h, j)

        # Exact agreement rate
        exact_agree = float((h == j).mean())

        # Within-1 agreement (judge within 1 point of human)
        within_1 = float((np.abs(h - j) <= 1).mean())

        # Mean absolute error
        mae = float(np.abs(h - j).mean())

        # Directional bias (positive = judge scores higher than human)
        bias = float((j - h).mean())

        # Confusion matrix
        cm = confusion_matrix(h, j, labels=[1, 2, 3, 4, 5])

        results[dim] = {
            "n": len(h),
            "kappa_weighted": round(kappa, 3),
            "spearman_r": round(spear_r, 3),
            "spearman_p": round(spear_p, 4),
            "exact_agreement": round(exact_agree, 3),
            "within_1_agreement": round(within_1, 3),
            "mae": round(mae, 2),
            "bias": round(bias, 2),
            "human_mean": round(float(h.mean()), 2),
            "judge_mean": round(float(j.mean()), 2),
            "confusion_matrix": cm.tolist(),
        }

    # Overall metrics
    if len(all_human) >= 10:
        h_all = np.array(all_human)
        j_all = np.array(all_judge)
        kappa_all = cohen_kappa_score(h_all, j_all, weights="quadratic")
        spear_all, sp_all = spearmanr(h_all, j_all)

        results["overall"] = {
            "n": len(h_all),
            "kappa_weighted": round(kappa_all, 3),
            "spearman_r": round(spear_all, 3),
            "exact_agreement": round(float((h_all == j_all).mean()), 3),
            "within_1_agreement": round(float((np.abs(h_all - j_all) <= 1).mean()), 3),
            "mae": round(float(np.abs(h_all - j_all).mean()), 2),
            "bias": round(float((j_all - h_all).mean()), 2),
        }

    return results


def generate_report(results: dict) -> str:
    """Generate markdown calibration report."""
    lines = [
        "# ClinicalBench — Day 2 Calibration Report",
        f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "\n## Overview",
        "\nThis report measures how well the LLM judge (GPT-4o-mini) aligns with "
        "human clinical judgment across four evaluation dimensions.",
        "\n### Interpretation guide",
        "- **Cohen's kappa (weighted)**: >0.8 = excellent, 0.6-0.8 = substantial, "
        "0.4-0.6 = moderate, <0.4 = poor",
        "- **Spearman r**: rank correlation; >0.7 = strong, 0.5-0.7 = moderate",
        "- **Bias**: positive = judge scores higher than human, negative = stricter judge",
    ]

    # Overall summary
    if "overall" in results:
        o = results["overall"]
        lines.extend([
            "\n## Overall calibration",
            f"\n| Metric | Value |",
            f"|--------|-------|",
            f"| Samples | {o['n']} |",
            f"| Weighted kappa | {o['kappa_weighted']} |",
            f"| Spearman r | {o['spearman_r']} |",
            f"| Exact agreement | {o['exact_agreement']:.1%} |",
            f"| Within-1 agreement | {o['within_1_agreement']:.1%} |",
            f"| Mean absolute error | {o['mae']} |",
            f"| Judge bias | {'+' if o['bias'] > 0 else ''}{o['bias']} |",
        ])

    # Per-dimension
    lines.extend([
        "\n## Per-dimension calibration",
        "\n| Dimension | n | Kappa | Spearman | Exact agree | Within-1 | MAE | Bias |",
        "|-----------|---|-------|----------|-------------|----------|-----|------|",
    ])

    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim in results and "error" not in results[dim]:
            d = results[dim]
            bias_str = f"+{d['bias']}" if d['bias'] > 0 else str(d['bias'])
            lines.append(
                f"| {dim} | {d['n']} | {d['kappa_weighted']} | "
                f"{d['spearman_r']} | {d['exact_agreement']:.1%} | "
                f"{d['within_1_agreement']:.1%} | {d['mae']} | {bias_str} |"
            )

    # Confusion matrices
    lines.append("\n## Confusion matrices (human vs judge)")
    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim in results and "confusion_matrix" in results.get(dim, {}):
            cm = results[dim]["confusion_matrix"]
            lines.extend([
                f"\n### {dim.capitalize()}",
                f"Rows = human score, Columns = judge score\n",
                "| | J=1 | J=2 | J=3 | J=4 | J=5 |",
                "|------|-----|-----|-----|-----|-----|",
            ])
            for i, row in enumerate(cm):
                lines.append(
                    f"| H={i+1} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |"
                )

    # Key findings placeholder
    lines.extend([
        "\n## Key findings",
        "\n_Fill in after reviewing the numbers above:_",
        "\n1. Which dimension has the highest/lowest judge-human agreement?",
        "2. Is the judge systematically biased (over-scoring or under-scoring)?",
        "3. Where does the judge fail most — on low scores, mid scores, or high scores?",
        "4. Does the bimodal distribution from Day 1 align with human judgment or "
        "diverge from it?",
        "\n## Implications for Sully.ai",
        "\n- Dimensions with kappa > 0.6 can be used as automated CI gates with "
        "reasonable confidence",
        "- Dimensions with kappa < 0.4 need human-in-the-loop review or improved "
        "judge prompts",
        "- Positive bias means the judge is lenient — clinical eval should err toward "
        "strict, so negative bias is actually preferable for safety",
    ])

    return "\n".join(lines)


def save_confusion_matrices(results: dict, output_dir: Path):
    """Save confusion matrices as individual CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim in results and "confusion_matrix" in results.get(dim, {}):
            cm = results[dim]["confusion_matrix"]
            path = output_dir / f"{dim}_confusion.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["human\\judge", "1", "2", "3", "4", "5"])
                for i, row in enumerate(cm):
                    writer.writerow([str(i + 1)] + [str(v) for v in row])


def main():
    judge_path = Path("results/eval_scores.jsonl")
    human_path = Path("results/human_annotations.csv")

    if not judge_path.exists():
        print("Run Day 1 pipeline first")
        return
    if not human_path.exists():
        print("Run annotation tool first: python scripts/annotate.py")
        return

    print("Loading judge scores...")
    judge_scores = load_judge_scores(judge_path)

    print("Loading human annotations...")
    annotations = load_human_annotations(human_path)
    print(f"  {len(annotations)} human annotations loaded")

    print("Computing calibration metrics...")
    results = compute_calibration(judge_scores, annotations)

    # Save stats
    stats_path = Path("results/calibration_stats.json")
    # Convert numpy types for JSON serialization
    def clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = json.loads(json.dumps(results, default=clean))
    with open(stats_path, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"Stats saved to {stats_path}")

    # Save confusion matrices
    save_confusion_matrices(results, Path("results/confusion_matrices"))
    print("Confusion matrices saved to results/confusion_matrices/")

    # Generate report
    report = generate_report(results)
    report_path = Path("results/calibration_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Print highlights
    print("\n" + "=" * 60)
    print("CALIBRATION HIGHLIGHTS")
    print("=" * 60)
    if "overall" in results:
        o = results["overall"]
        print(f"\n  Overall weighted kappa: {o['kappa_weighted']}")
        print(f"  Overall Spearman r:     {o['spearman_r']}")
        print(f"  Exact agreement:        {o['exact_agreement']:.1%}")
        print(f"  Judge bias:             {'+' if o['bias']>0 else ''}{o['bias']}")

    print("\n  Per dimension:")
    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim in results and "kappa_weighted" in results.get(dim, {}):
            d = results[dim]
            print(f"    {dim:15s}  kappa={d['kappa_weighted']:.3f}  "
                  f"spearman={d['spearman_r']:.3f}  bias={d['bias']:+.2f}")

    print(f"\nNext: python scripts/ablation_judge_model.py")


if __name__ == "__main__":
    main()