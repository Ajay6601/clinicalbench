#!/usr/bin/env python3
"""
ClinicalBench — LLM-as-Judge Evaluation Framework for Clinical AI
==================================================================

Usage:
  python main.py pull-data                     # Download MedQA dataset
  python main.py generate-answers              # Generate responses from eval models
  python main.py run-eval                      # Run all 4 judges on model responses
  python main.py run-eval --judge gpt-4o-mini  # Run with specific judge model
  python main.py calibrate                     # Correctness-based calibration
  python main.py reference-judge               # Run Claude as reference judge
  python main.py compare-judges                # Cross-model judge agreement
  python main.py inject-test                   # Hallucination injection sensitivity
  python main.py report                        # Generate all reports
  python main.py full-pipeline                 # Run everything end-to-end
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = {
    "pull-data": "scripts/pull_data.py",
    "generate-answers": "scripts/generate_answers.py",
    "run-eval": "scripts/run_judges.py",
    "calibrate": "scripts/correctness_analysis.py",
    "reference-judge": "scripts/proxy_annotate.py",
    "compare-judges": "scripts/three_way_comparison.py",
    "inject-test": "scripts/ablation_judge_model.py",
}

PIPELINE_ORDER = [
    "pull-data",
    "generate-answers",
    "run-eval",
    "calibrate",
    "reference-judge",
    "compare-judges",
]


def run_script(name: str, script: str, extra_args: list = None):
    """Run a pipeline step."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, script]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFailed: {name}. Fix and re-run — all steps have resume support.")
        sys.exit(1)


def cmd_report():
    """Generate all reports from existing data."""
    for script_name in ["calibrate", "compare-judges"]:
        script = SCRIPTS[script_name]
        if Path(script).exists():
            run_script(f"Report: {script_name}", script)

    print("\nReports generated in results/")
    for f in sorted(Path("results").glob("*.md")):
        print(f"  {f}")


def main():
    parser = argparse.ArgumentParser(
        description="ClinicalBench — LLM-as-Judge Eval for Clinical AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("pull-data", help="Download MedQA dataset")
    sub.add_parser("generate-answers", help="Generate model responses")

    eval_p = sub.add_parser("run-eval", help="Run LLM judges")
    eval_p.add_argument("--judge", default="gpt-4o-mini", help="Judge model")

    sub.add_parser("calibrate", help="Correctness-based calibration (no API calls)")

    ref_p = sub.add_parser("reference-judge", help="Run reference judges")
    ref_p.add_argument("--n", type=int, default=100, help="Number of samples")
    ref_p.add_argument("--only", choices=["gpt-4o", "claude-sonnet"])

    sub.add_parser("compare-judges", help="Three-way judge comparison")

    inj_p = sub.add_parser("inject-test", help="Hallucination injection test")
    inj_p.add_argument("--judge-model", default="gpt-4o-mini")

    sub.add_parser("report", help="Generate all reports")
    sub.add_parser("full-pipeline", help="Run everything end-to-end")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "full-pipeline":
        for step in PIPELINE_ORDER:
            run_script(step, SCRIPTS[step])
        cmd_report()
        return

    if args.command == "report":
        cmd_report()
        return

    if args.command == "run-eval":
        run_script(args.command, SCRIPTS[args.command])
        return

    if args.command == "reference-judge":
        extra = ["--n", str(args.n)]
        if args.only:
            extra.extend(["--only", args.only])
        run_script(args.command, SCRIPTS[args.command], extra)
        return

    if args.command == "inject-test":
        extra = ["--judge-model", args.judge_model]
        run_script(args.command, SCRIPTS[args.command], extra)
        return

    if args.command in SCRIPTS:
        run_script(args.command, SCRIPTS[args.command])


if __name__ == "__main__":
    main()