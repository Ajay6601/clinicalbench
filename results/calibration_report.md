# ClinicalBench — Day 2 Calibration Report

Generated: 2026-03-24 17:54 UTC

## Overview

This report measures how well the LLM judge (GPT-4o-mini) aligns with human clinical judgment across four evaluation dimensions.

### Interpretation guide
- **Cohen's kappa (weighted)**: >0.8 = excellent, 0.6-0.8 = substantial, 0.4-0.6 = moderate, <0.4 = poor
- **Spearman r**: rank correlation; >0.7 = strong, 0.5-0.7 = moderate
- **Bias**: positive = judge scores higher than human, negative = stricter judge

## Overall calibration

| Metric | Value |
|--------|-------|
| Samples | 380 |
| Weighted kappa | 0.788 |
| Spearman r | 0.786 |
| Exact agreement | 62.9% |
| Within-1 agreement | 90.5% |
| Mean absolute error | 0.51 |
| Judge bias | -0.17 |

## Per-dimension calibration

| Dimension | n | Kappa | Spearman | Exact agree | Within-1 | MAE | Bias |
|-----------|---|-------|----------|-------------|----------|-----|------|
| factuality | 95 | 0.841 | 0.839 | 67.4% | 90.5% | 0.44 | -0.17 |
| safety | 95 | 0.777 | 0.78 | 65.3% | 89.5% | 0.52 | -0.07 |
| hallucination | 95 | 0.699 | 0.747 | 69.5% | 83.2% | 0.55 | -0.46 |
| completeness | 95 | 0.823 | 0.852 | 49.5% | 98.9% | 0.52 | +0.03 |

## Confusion matrices (human vs judge)

### Factuality
Rows = human score, Columns = judge score

| | J=1 | J=2 | J=3 | J=4 | J=5 |
|------|-----|-----|-----|-----|-----|
| H=1 | 3 | 0 | 0 | 0 | 0 |
| H=2 | 7 | 18 | 7 | 0 | 1 |
| H=3 | 2 | 4 | 2 | 0 | 0 |
| H=4 | 0 | 4 | 0 | 2 | 3 |
| H=5 | 0 | 1 | 1 | 1 | 39 |

### Safety
Rows = human score, Columns = judge score

| | J=1 | J=2 | J=3 | J=4 | J=5 |
|------|-----|-----|-----|-----|-----|
| H=1 | 2 | 0 | 0 | 0 | 0 |
| H=2 | 6 | 20 | 11 | 2 | 2 |
| H=3 | 0 | 0 | 0 | 0 | 0 |
| H=4 | 0 | 1 | 0 | 0 | 0 |
| H=5 | 1 | 2 | 2 | 6 | 40 |

### Hallucination
Rows = human score, Columns = judge score

| | J=1 | J=2 | J=3 | J=4 | J=5 |
|------|-----|-----|-----|-----|-----|
| H=1 | 0 | 0 | 0 | 0 | 0 |
| H=2 | 2 | 20 | 2 | 1 | 0 |
| H=3 | 0 | 8 | 6 | 0 | 0 |
| H=4 | 0 | 1 | 0 | 1 | 0 |
| H=5 | 0 | 7 | 7 | 1 | 39 |

### Completeness
Rows = human score, Columns = judge score

| | J=1 | J=2 | J=3 | J=4 | J=5 |
|------|-----|-----|-----|-----|-----|
| H=1 | 2 | 3 | 0 | 0 | 0 |
| H=2 | 8 | 15 | 16 | 1 | 0 |
| H=3 | 0 | 1 | 1 | 3 | 0 |
| H=4 | 0 | 0 | 4 | 21 | 2 |
| H=5 | 0 | 0 | 0 | 10 | 8 |

## Key findings

_Fill in after reviewing the numbers above:_

1. Which dimension has the highest/lowest judge-human agreement?
2. Is the judge systematically biased (over-scoring or under-scoring)?
3. Where does the judge fail most — on low scores, mid scores, or high scores?
4. Does the bimodal distribution from Day 1 align with human judgment or diverge from it?

## Implications for Sully.ai

- Dimensions with kappa > 0.6 can be used as automated CI gates with reasonable confidence
- Dimensions with kappa < 0.4 need human-in-the-loop review or improved judge prompts
- Positive bias means the judge is lenient — clinical eval should err toward strict, so negative bias is actually preferable for safety