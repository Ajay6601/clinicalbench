import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def load_scores_by_key(path: Path, source: str, is_csv: bool = False) -> dict:
    """Load scores keyed by (question_id, model_name, dimension)."""
    scores = {}
    if is_csv:
        with open(path) as f:
            for row in csv.DictReader(f):
                score = int(row["human_score"])
                if score < 1:
                    continue
                key = (row["question_id"], row["model_name"], row["dimension"])
                scores[key] = score
    else:
        for line in open(path):
            r = json.loads(line)
            if r["score"] < 1:
                continue
            key = (r["question_id"], r["model_name"], r["dimension"])
            scores[key] = r["score"]
    return scores


def pairwise_agreement(scores_a: dict, scores_b: dict, name_a: str, name_b: str):
    """Compute agreement between two judge score dicts."""
    common_keys = set(scores_a.keys()) & set(scores_b.keys())

    by_dim = defaultdict(lambda: {"a": [], "b": []})
    all_a, all_b = [], []

    for key in common_keys:
        dim = key[2]
        by_dim[dim]["a"].append(scores_a[key])
        by_dim[dim]["b"].append(scores_b[key])
        all_a.append(scores_a[key])
        all_b.append(scores_b[key])

    results = {"pair": f"{name_a} vs {name_b}", "n_common": len(common_keys)}

    # Overall
    if len(all_a) >= 10:
        a_arr, b_arr = np.array(all_a), np.array(all_b)
        results["overall"] = {
            "kappa": round(float(cohen_kappa_score(all_a, all_b, weights="quadratic")), 3),
            "spearman": round(float(spearmanr(all_a, all_b)[0]), 3),
            "exact_agree": round(float((a_arr == b_arr).mean()), 3),
            "within_1": round(float((np.abs(a_arr - b_arr) <= 1).mean()), 3),
            "bias": round(float((a_arr - b_arr).mean()), 2),
            "a_mean": round(float(a_arr.mean()), 2),
            "b_mean": round(float(b_arr.mean()), 2),
        }

    # Per dimension
    results["dimensions"] = {}
    for dim in ["factuality", "safety", "hallucination", "completeness"]:
        if dim not in by_dim or len(by_dim[dim]["a"]) < 5:
            continue
        a = np.array(by_dim[dim]["a"])
        b = np.array(by_dim[dim]["b"])

        results["dimensions"][dim] = {
            "n": len(a),
            "kappa": round(float(cohen_kappa_score(
                by_dim[dim]["a"], by_dim[dim]["b"], weights="quadratic"
            )), 3),
            "spearman": round(float(spearmanr(a, b)[0]), 3),
            "exact_agree": round(float((a == b).mean()), 3),
            "bias": round(float((a - b).mean()), 2),
            "a_mean": round(float(a.mean()), 2),
            "b_mean": round(float(b.mean()), 2),
        }

    return results


def self_judge_bias_analysis(
    mini_scores: dict, gpt4o_scores: dict, claude_scores: dict
):
    """
    Test: Does GPT-4o-mini judge OpenAI model outputs more leniently than
    Claude judges them?

    Compare GPT-4o-mini scores vs Claude scores specifically on:
      - GPT-4o-mini's OWN outputs (self-judging)
      - GPT-3.5 outputs (same family)
      - Claude Sonnet outputs (competitor)
    """
    results = {}

    for target_model in ["gpt-4o-mini", "gpt-3.5-turbo", "claude-sonnet"]:
        mini_on_target = []
        claude_on_target = []

        for key in set(mini_scores.keys()) & set(claude_scores.keys()):
            if key[1] == target_model:
                mini_on_target.append(mini_scores[key])
                claude_on_target.append(claude_scores[key])

        if len(mini_on_target) >= 5:
            m = np.array(mini_on_target)
            c = np.array(claude_on_target)
            results[target_model] = {
                "n": len(m),
                "mini_mean": round(float(m.mean()), 2),
                "claude_mean": round(float(c.mean()), 2),
                "bias": round(float((m - c).mean()), 2),
                "mini_higher_pct": round(float((m > c).mean()), 3),
            }

    return results


def generate_report(
    pairs: list, bias_results: dict, correctness_alignment: dict
) -> str:
    """Generate the three-way comparison report."""
    lines = [
        "# ClinicalBench — Three-Way Judge Comparison",
        f"\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "\n## Overview",
        "\nThree judges evaluated the same clinical QA samples:",
        "- **GPT-4o-mini**: Cheap production judge (Day 1 baseline)",
        "- **GPT-4o**: Stronger OpenAI reference judge",
        "- **Claude Sonnet**: Cross-provider independent judge",
        "\nAll judges used identical rubric prompts. The only variable is model capability.",
    ]

    # Pairwise agreement table
    lines.extend([
        "\n## Pairwise judge agreement (overall)\n",
        "| Pair | n | Kappa | Spearman | Exact agree | Within-1 | Bias |",
        "|------|---|-------|----------|-------------|----------|------|",
    ])
    for p in pairs:
        if "overall" in p:
            o = p["overall"]
            bias_str = f"+{o['bias']}" if o['bias'] > 0 else str(o['bias'])
            lines.append(
                f"| {p['pair']} | {p['n_common']} | {o['kappa']} | "
                f"{o['spearman']} | {o['exact_agree']:.1%} | "
                f"{o['within_1']:.1%} | {bias_str} |"
            )

    # Per-dimension breakdown
    lines.extend([
        "\n## Per-dimension agreement\n",
        "| Pair | Dimension | Kappa | Spearman | Bias |",
        "|------|-----------|-------|----------|------|",
    ])
    for p in pairs:
        for dim in ["factuality", "safety", "hallucination", "completeness"]:
            if dim in p.get("dimensions", {}):
                d = p["dimensions"][dim]
                bias_str = f"+{d['bias']}" if d['bias'] > 0 else str(d['bias'])
                lines.append(
                    f"| {p['pair']} | {dim} | {d['kappa']} | "
                    f"{d['spearman']} | {bias_str} |"
                )

    # Self-judging bias
    if bias_results:
        lines.extend([
            "\n## Self-judging bias analysis",
            "\nDoes GPT-4o-mini judge OpenAI outputs more favorably than Claude does?\n",
            "| Target model | GPT-4o-mini mean | Claude mean | Bias | Mini higher % |",
            "|-------------|------------------|-------------|------|---------------|",
        ])
        for target, r in bias_results.items():
            bias_str = f"+{r['bias']}" if r['bias'] > 0 else str(r['bias'])
            lines.append(
                f"| {target} | {r['mini_mean']} | {r['claude_mean']} | "
                f"{bias_str} | {r['mini_higher_pct']:.1%} |"
            )

        lines.extend([
            "\n**Interpretation**: If GPT-4o-mini scores its own outputs (and "
            "GPT-3.5 outputs) significantly higher than Claude does, this suggests "
            "self-judging bias. A bias > +0.3 on self-evaluation is concerning. "
            "A similar bias across all targets suggests the model is just generally "
            "more lenient, not specifically biased.",
        ])

    # Correctness alignment
    if correctness_alignment:
        lines.extend([
            "\n## Correctness alignment: which judge best detects wrong answers?\n",
            "| Judge | Correct mean | Incorrect mean | Separation | Best gate (F1) |",
            "|-------|-------------|----------------|------------|----------------|",
        ])
        for judge_name, data in correctness_alignment.items():
            lines.append(
                f"| {judge_name} | {data['correct_mean']} | "
                f"{data['incorrect_mean']} | {data['separation']:+.2f} | "
                f"{data.get('best_gate', 'N/A')} |"
            )

    lines.extend([
        "\n## Recommendations for production deployment",
        "\n1. **CI gate judge**: Use the judge with highest correctness separation "
        "and lowest self-bias",
        "2. **Ensemble approach**: Average scores from 2+ judges for dimensions "
        "where pairwise kappa < 0.6",
        "3. **Safety escalation**: For safety dimension, use the STRICTEST judge "
        "(lowest mean) as the gate — false positives are better than false negatives "
        "in clinical AI",
        "4. **Cost optimization**: If the cheap judge (GPT-4o-mini) has kappa > 0.7 "
        "with the reference judges, it's safe to use alone in CI; save expensive "
        "judges for periodic audits",
    ])

    return "\n".join(lines)


def main():
    # Load all three judge score sets
    mini_path = Path("results/eval_scores.jsonl")
    gpt4o_path = Path("results/human_annotations.csv")
    claude_path = Path("results/claude_reference_annotations.csv")

    if not mini_path.exists():
        print("Run Day 1 pipeline first!")
        return

    print("Loading judge scores...")
    mini_scores = load_scores_by_key(mini_path, "gpt-4o-mini")
    print(f"  GPT-4o-mini: {len(mini_scores)} scores")

    gpt4o_scores = {}
    if gpt4o_path.exists():
        gpt4o_scores = load_scores_by_key(gpt4o_path, "gpt-4o", is_csv=True)
        print(f"  GPT-4o:      {len(gpt4o_scores)} scores")
    else:
        print("  GPT-4o: not found (run proxy_annotate.py first)")

    claude_scores = {}
    if claude_path.exists():
        claude_scores = load_scores_by_key(claude_path, "claude", is_csv=True)
        print(f"  Claude:      {len(claude_scores)} scores")
    else:
        print("  Claude: not found (run proxy_annotate.py first)")

    # Pairwise comparisons
    pairs = []
    if gpt4o_scores:
        pairs.append(pairwise_agreement(mini_scores, gpt4o_scores, "GPT-4o-mini", "GPT-4o"))
    if claude_scores:
        pairs.append(pairwise_agreement(mini_scores, claude_scores, "GPT-4o-mini", "Claude"))
    if gpt4o_scores and claude_scores:
        pairs.append(pairwise_agreement(gpt4o_scores, claude_scores, "GPT-4o", "Claude"))

    # Self-judging bias
    bias_results = {}
    if claude_scores:
        print("\nAnalyzing self-judging bias...")
        bias_results = self_judge_bias_analysis(mini_scores, gpt4o_scores, claude_scores)

    # Correctness alignment per judge
    print("Computing correctness alignment...")
    correctness_alignment = {}

    all_answers = {
        json.loads(l)["question_id"] + "|" + json.loads(l)["model_name"]:
        json.loads(l)["is_correct"]
        for l in open("data/model_answers.jsonl")
    }

    for judge_name, scores in [
        ("GPT-4o-mini", mini_scores),
        ("GPT-4o", gpt4o_scores),
        ("Claude", claude_scores),
    ]:
        if not scores:
            continue
        correct_s, incorrect_s = [], []
        for key, score in scores.items():
            ans_key = f"{key[0]}|{key[1]}"
            if ans_key in all_answers:
                if all_answers[ans_key]:
                    correct_s.append(score)
                else:
                    incorrect_s.append(score)

        if len(correct_s) >= 5 and len(incorrect_s) >= 5:
            c, ic = np.array(correct_s), np.array(incorrect_s)
            correctness_alignment[judge_name] = {
                "correct_mean": round(float(c.mean()), 2),
                "incorrect_mean": round(float(ic.mean()), 2),
                "separation": round(float(c.mean() - ic.mean()), 2),
            }

    # Save stats
    stats = {
        "pairwise": [p for p in pairs],
        "self_judge_bias": bias_results,
        "correctness_alignment": correctness_alignment,
    }
    stats_path = Path("results/three_way_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n[OK] Stats saved to {stats_path}")

    # Generate report
    report = generate_report(pairs, bias_results, correctness_alignment)
    report_path = Path("results/three_way_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[OK] Report saved to {report_path}")

    # Print highlights
    print("\n" + "=" * 60)
    print("THREE-WAY JUDGE COMPARISON HIGHLIGHTS")
    print("=" * 60)

    for p in pairs:
        if "overall" in p:
            o = p["overall"]
            print(f"\n  {p['pair']}:")
            print(f"    Kappa: {o['kappa']}  Spearman: {o['spearman']}  "
                  f"Bias: {o['bias']:+.2f}")

    if bias_results:
        print("\n  Self-judging bias (GPT-4o-mini vs Claude on same targets):")
        for target, r in bias_results.items():
            print(f"    On {target}: mini={r['mini_mean']}, "
                  f"claude={r['claude_mean']}, bias={r['bias']:+.2f}")

    print(f"\n[OK] Day 2 analysis complete!")
    print(f"   Next: python scripts/ablation_judge_model.py")


if __name__ == "__main__":
    main()