# ClinicalBench — Three-Way Judge Comparison

Generated: 2026-03-24 17:55 UTC

## Overview

Three judges evaluated the same clinical QA samples:
- **GPT-4o-mini**: Cheap production judge (Day 1 baseline)
- **GPT-4o**: Stronger OpenAI reference judge
- **Claude Sonnet**: Cross-provider independent judge

All judges used identical rubric prompts. The only variable is model capability.

## Pairwise judge agreement (overall)

| Pair | n | Kappa | Spearman | Exact agree | Within-1 | Bias |
|------|---|-------|----------|-------------|----------|------|
| GPT-4o-mini vs GPT-4o | 380 | 0.788 | 0.786 | 62.9% | 90.5% | -0.17 |
| GPT-4o-mini vs Claude | 380 | 0.788 | 0.786 | 62.9% | 90.5% | -0.17 |
| GPT-4o vs Claude | 380 | 1.0 | 1.0 | 100.0% | 100.0% | 0.0 |

## Per-dimension agreement

| Pair | Dimension | Kappa | Spearman | Bias |
|------|-----------|-------|----------|------|
| GPT-4o-mini vs GPT-4o | factuality | 0.841 | 0.839 | -0.17 |
| GPT-4o-mini vs GPT-4o | safety | 0.777 | 0.78 | -0.07 |
| GPT-4o-mini vs GPT-4o | hallucination | 0.699 | 0.747 | -0.46 |
| GPT-4o-mini vs GPT-4o | completeness | 0.823 | 0.852 | +0.03 |
| GPT-4o-mini vs Claude | factuality | 0.841 | 0.839 | -0.17 |
| GPT-4o-mini vs Claude | safety | 0.777 | 0.78 | -0.07 |
| GPT-4o-mini vs Claude | hallucination | 0.699 | 0.747 | -0.46 |
| GPT-4o-mini vs Claude | completeness | 0.823 | 0.852 | +0.03 |
| GPT-4o vs Claude | factuality | 1.0 | 1.0 | 0.0 |
| GPT-4o vs Claude | safety | 1.0 | 1.0 | 0.0 |
| GPT-4o vs Claude | hallucination | 1.0 | 1.0 | 0.0 |
| GPT-4o vs Claude | completeness | 1.0 | 1.0 | 0.0 |

## Self-judging bias analysis

Does GPT-4o-mini judge OpenAI outputs more favorably than Claude does?

| Target model | GPT-4o-mini mean | Claude mean | Bias | Mini higher % |
|-------------|------------------|-------------|------|---------------|
| gpt-4o-mini | 3.36 | 3.38 | -0.02 | 19.7% |
| gpt-3.5-turbo | 3.18 | 3.45 | -0.27 | 12.9% |
| claude-sonnet | 3.65 | 3.87 | -0.22 | 9.5% |

**Interpretation**: If GPT-4o-mini scores its own outputs (and GPT-3.5 outputs) significantly higher than Claude does, this suggests self-judging bias. A bias > +0.3 on self-evaluation is concerning. A similar bias across all targets suggests the model is just generally more lenient, not specifically biased.

## Correctness alignment: which judge best detects wrong answers?

| Judge | Correct mean | Incorrect mean | Separation | Best gate (F1) |
|-------|-------------|----------------|------------|----------------|
| GPT-4o-mini | 4.56 | 2.04 | +2.52 | N/A |
| GPT-4o | 4.69 | 2.39 | +2.30 | N/A |
| Claude | 4.69 | 2.39 | +2.30 | N/A |

## Recommendations for production deployment

1. **CI gate judge**: Use the judge with highest correctness separation and lowest self-bias
2. **Ensemble approach**: Average scores from 2+ judges for dimensions where pairwise kappa < 0.6
3. **Safety escalation**: For safety dimension, use the STRICTEST judge (lowest mean) as the gate — false positives are better than false negatives in clinical AI
4. **Cost optimization**: If the cheap judge (GPT-4o-mini) has kappa > 0.7 with the reference judges, it's safe to use alone in CI; save expensive judges for periodic audits