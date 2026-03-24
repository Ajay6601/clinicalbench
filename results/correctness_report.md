# ClinicalBench — Correctness-Based Judge Calibration

Generated: 2026-03-24 13:14 UTC

## Methodology

Since MedQA provides gold-standard correct answers, we can test judge validity without human annotation: do judges assign significantly lower scores to responses where the model selected the WRONG answer?

## Score separation: correct vs incorrect answers

| Dimension | Correct (mean) | Incorrect (mean) | Separation | p-value | Significant |
|-----------|---------------|-------------------|------------|---------|-------------|
| factuality | 4.78 (±0.68) | 1.88 (±0.71) | +2.90 | 0.000000 | Yes (p<0.001) |
| safety | 4.73 (±0.76) | 2.14 (±0.83) | +2.59 | 0.000000 | Yes (p<0.001) |
| hallucination | 4.67 (±0.84) | 2.14 (±0.65) | +2.52 | 0.000000 | Yes (p<0.001) |
| completeness | 4.07 (±0.54) | 2.01 (±0.8) | +2.07 | 0.000000 | Yes (p<0.001) |

## Error detection: using judge scores as CI gates

If we flag responses where judge score <= threshold, how well do we detect actually-incorrect answers?


### Factuality

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| score <= 1 | 95.3% | 28.1% | 0.434 |
| score <= 2 | 89.4% | 86.3% | 0.878 |
| score <= 3 | 82.7% | 97.9% | 0.897 |
| score <= 4 | 74.0% | 99.3% | 0.848 |

### Safety

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| score <= 1 | 80.0% | 19.2% | 0.309 |
| score <= 2 | 86.6% | 75.3% | 0.806 |
| score <= 3 | 81.4% | 93.2% | 0.869 |
| score <= 4 | 68.2% | 98.6% | 0.807 |

### Hallucination

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| score <= 1 | 93.8% | 10.3% | 0.185 |
| score <= 2 | 82.3% | 79.5% | 0.808 |
| score <= 3 | 71.2% | 96.6% | 0.820 |
| score <= 4 | 67.8% | 99.3% | 0.806 |

### Completeness

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| score <= 1 | 95.6% | 29.5% | 0.450 |
| score <= 2 | 94.6% | 71.9% | 0.817 |
| score <= 3 | 81.2% | 97.9% | 0.888 |
| score <= 4 | 27.8% | 100.0% | 0.435 |

## Interpretation for CI/CD gates

- **High precision, low recall** (strict threshold like <= 1): Few false alarms, but misses many bad answers. Good for blocking only the most dangerous outputs.
- **Lower precision, high recall** (loose threshold like <= 3): Catches more bad answers but also flags some correct ones. Good for flagging outputs for human review.
- **Optimal threshold**: Pick the threshold where F1 peaks per dimension. This is the recommended CI gate setting.

## Key finding

_If separation is > 1.0 and p < 0.001, the judge dimension is a statistically valid signal for detecting incorrect clinical responses. Dimensions with separation < 0.5 need improved judge prompts or should not be used as automated gates._