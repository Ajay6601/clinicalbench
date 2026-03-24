# ClinicalBench — Day 2 Ablation Report

Generated: 2026-03-24 17:04 UTC

## Experiment 1: Judge model comparison

Comparing GPT-4o-mini (Day 1 judge) vs gpt-4o (ablation judge) against human annotations.

| Dimension | GPT-4o-mini kappa | GPT-4o-mini bias | gpt-4o kappa | gpt-4o bias |
|-----------|-------------------|------------------|---------------------|--------------------|

## Experiment 2: Hallucination injection sensitivity

Tests whether the hallucination judge detects fabricated claims injected into correct responses at varying rates.

| Condition | Mean score | Std | Detection rate (score <= 3) |
|-----------|-----------|-----|----------------------------|
| baseline | 4.13 | 1.20 | 33.3% |
| 1_injection | 1.93 | 0.44 | 100.0% |
| 2_injections | 1.73 | 0.44 | 100.0% |
| 3_injections | 1.00 | 0.00 | 100.0% |

A good hallucination judge should show:
- High baseline scores (clean responses score 4-5)
- Monotonically decreasing scores as injections increase
- Detection rate increasing toward 100% with more injections