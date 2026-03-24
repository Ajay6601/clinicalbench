import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def run_step(name: str, script: str):
    """Run a pipeline step, exit on failure."""
    print(f"\n{'='*60}")
    print(f"[RUN] STEP: {name}")
    print(f"   Script: {script}")
    print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        [sys.executable, script],
        cwd=str(Path(__file__).parent),
    )

    if result.returncode != 0:
        print(f"\n[FAIL] FAILED: {name}")
        print(f"   Fix the error above and re-run: python run_day1.py")
        print(f"   (All steps have resume support — no work will be lost)")
        sys.exit(1)

    print(f"\n[OK] COMPLETED: {name}")


def check_env():
    """Verify environment is ready."""
    from dotenv import load_dotenv
    import os
    load_dotenv()

    missing = []
    if not os.getenv("OPENAI_API_KEY") or "your-key" in os.getenv("OPENAI_API_KEY", ""):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("ANTHROPIC_API_KEY") or "your-key" in os.getenv("ANTHROPIC_API_KEY", ""):
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print("[FAIL] Missing API keys in .env file:")
        for k in missing:
            print(f"   - {k}")
        print("\nEdit .env and add your keys, then re-run.")
        sys.exit(1)

    print("[OK] Environment check passed")


def main():
    start = time.time()

    print("=" * 60)
    print("  ClinicalBench — Day 1 Pipeline")
    print("  LLM-as-Judge Evaluation for Clinical QA")
    print("=" * 60)

    check_env()

    run_step(
        "1/3 — Pull MedQA dataset",
        "scripts/pull_data.py",
    )
    run_step(
        "2/3 — Generate model answers (GPT-4o-mini, Claude, GPT-3.5)",
        "scripts/generate_answers.py",
    )
    run_step(
        "3/3 — Run LLM judges (factuality, safety, hallucination, completeness)",
        "scripts/run_judges.py",
    )

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"[DONE] DAY 1 PIPELINE COMPLETE")
    print(f"   Total time: {elapsed/60:.0f} minutes")
    print(f"   Outputs:")
    print(f"     data/medqa_200.jsonl         — 200 clinical questions")
    print(f"     data/model_answers.jsonl      — 600 model responses")
    print(f"     results/eval_scores.jsonl     — 2400 judge evaluations")
    print(f"     results/summary_stats.json    — aggregate statistics")
    print(f"     results/day1_report.md        — human-readable report")
    print("=" * 60)
    print("\n[NEXT] Next: Day 2 — Human annotation + calibration")
    print("   Run: python scripts/annotate.py  (coming tomorrow)")


if __name__ == "__main__":
    main()