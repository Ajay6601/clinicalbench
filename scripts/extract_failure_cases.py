"""
Extract and curate notable failure cases from Day 1 evaluation.
Creates results/failure_cases/ with human-readable markdown files
showing where models failed and judges caught (or missed) the error.

Usage: python scripts/extract_failure_cases.py
Time: ~2 seconds (no API calls)
"""

import json
from pathlib import Path
from collections import defaultdict


def main():
    scores_path = Path("results/eval_scores.jsonl")
    answers_path = Path("data/model_answers.jsonl")
    out_dir = Path("results/failure_cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load answers for context
    answers = {}
    for line in open(answers_path):
        a = json.loads(line)
        answers[(a["question_id"], a["model_name"])] = a

    # Load scores
    all_scores = [json.loads(l) for l in open(scores_path)]

    # Group scores by (question_id, model_name)
    grouped = defaultdict(list)
    for s in all_scores:
        grouped[(s["question_id"], s["model_name"])].append(s)

    # Find the most interesting failure cases
    cases = []

    for key, scores in grouped.items():
        if key not in answers:
            continue
        answer = answers[key]

        min_score = min(s["score"] for s in scores if s["score"] > 0)
        has_safety_1 = any(s["dimension"] == "safety" and s["score"] == 1 for s in scores)
        wrong_answer = not answer["is_correct"]

        # Priority: safety=1 failures, then wrong answers with low scores
        if has_safety_1:
            cases.append(("CRITICAL_SAFETY", key, scores, answer))
        elif wrong_answer and min_score <= 2:
            cases.append(("INCORRECT_DETECTED", key, scores, answer))
        elif wrong_answer and min_score >= 4:
            cases.append(("INCORRECT_MISSED", key, scores, answer))

    # Sort: critical safety first, then missed errors (most interesting)
    priority = {"CRITICAL_SAFETY": 0, "INCORRECT_MISSED": 1, "INCORRECT_DETECTED": 2}
    cases.sort(key=lambda x: (priority.get(x[0], 99), -min(
        s["score"] for s in x[2] if s["score"] > 0
    )))

    # Write top 15 cases
    summary_lines = [
        "# ClinicalBench — Failure Case Analysis\n",
        "Curated examples where models produced clinically problematic responses.\n",
        "## Case Index\n",
        "| # | Type | Model | Question | Min Score |",
        "|---|------|-------|----------|-----------|",
    ]

    for i, (case_type, key, scores, answer) in enumerate(cases[:15]):
        qid, model = key
        min_s = min(s["score"] for s in scores if s["score"] > 0)

        # Summary index
        type_label = {
            "CRITICAL_SAFETY": "Safety-critical error",
            "INCORRECT_DETECTED": "Error detected by judge",
            "INCORRECT_MISSED": "Error MISSED by judge",
        }[case_type]
        summary_lines.append(
            f"| {i+1} | {type_label} | {model} | {qid} | {min_s} |"
        )

        # Individual case file
        case_lines = [
            f"# Failure Case {i+1}: {type_label}\n",
            f"- **Model**: {model}",
            f"- **Question ID**: {qid}",
            f"- **Model answered correctly**: {'Yes' if answer['is_correct'] else 'No'}",
            f"- **Gold answer**: ({answer['gold_answer_key']}) {answer['gold_answer_text']}",
            f"- **Model predicted**: {answer['predicted_key']}\n",
            "## Question\n",
            answer["question"][:500],
            "\n## Model Response\n",
            answer["raw_response"][:600],
            "\n## Judge Scores\n",
            "| Dimension | Score | Rationale |",
            "|-----------|-------|-----------|",
        ]

        for s in sorted(scores, key=lambda x: x["score"]):
            if s["score"] > 0:
                rationale = s["rationale"][:150].replace("|", "/").replace("\n", " ")
                case_lines.append(
                    f"| {s['dimension']} | {s['score']}/5 | {rationale} |"
                )

        # Why this matters
        if case_type == "CRITICAL_SAFETY":
            case_lines.extend([
                "\n## Clinical Impact\n",
                "This response could cause direct patient harm if acted upon. "
                "The safety judge correctly identified this as a critical risk, "
                "which would trigger an automatic block in a production CI gate "
                "(config.yaml: safety.block_threshold = 2).",
            ])
        elif case_type == "INCORRECT_MISSED":
            case_lines.extend([
                "\n## Why This Matters\n",
                "The model selected the wrong answer, but the judge assigned high "
                "scores — a false negative. This is the most dangerous failure mode "
                "for automated eval: the system is wrong but confident. "
                "This case demonstrates why human-in-the-loop review remains "
                "necessary even with high overall kappa scores.",
            ])
        elif case_type == "INCORRECT_DETECTED":
            case_lines.extend([
                "\n## Why This Matters\n",
                "The model selected the wrong answer and the judge correctly "
                "flagged it with low scores. This demonstrates the eval pipeline "
                "working as intended — the CI gate would catch this before deployment.",
            ])

        case_path = out_dir / f"case_{i+1:02d}_{case_type.lower()}.md"
        with open(case_path, "w") as f:
            f.write("\n".join(case_lines))

    # Write summary
    summary_path = out_dir / "README.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"Extracted {min(15, len(cases))} failure cases to {out_dir}/")
    print(f"  Critical safety: {sum(1 for c in cases[:15] if c[0] == 'CRITICAL_SAFETY')}")
    print(f"  Errors detected: {sum(1 for c in cases[:15] if c[0] == 'INCORRECT_DETECTED')}")
    print(f"  Errors missed:   {sum(1 for c in cases[:15] if c[0] == 'INCORRECT_MISSED')}")


if __name__ == "__main__":
    main()