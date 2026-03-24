# ClinicalBench

Automated evaluation framework for clinical AI. Tests whether LLM judges can reliably catch dangerous medical errors without human reviewers in the loop.

I built this in 3 days to explore a specific question: if you deploy an LLM-as-judge to gate clinical AI outputs, how much can you actually trust it? The answer turns out to be "more than I expected on factuality, less than I'd like on hallucination."

## What this does

ClinicalBench runs LLM responses to medical questions through four structured judges (factuality, safety, hallucination, completeness), then validates those judges against gold-standard answers and a cross-provider reference model. The goal is to figure out which judge dimensions are reliable enough for automated CI gates and which still need a human looking over the results.

## Results

### Model accuracy on 200 USMLE questions

| Model | Accuracy |
|-------|----------|
| Claude Sonnet | 93.5% |
| GPT-4o-mini | 72.5% |
| GPT-3.5-turbo | 61.0% |

### Do the judges actually work?

Tested by checking whether judges give low scores when models get the answer wrong and high scores when they get it right (n=600, all p < 0.001):

| Dimension | Correct answer avg | Wrong answer avg | Gap | Best CI gate threshold | F1 at that threshold |
|-----------|-------------------|-----------------|-----|----------------------|---------------------|
| Factuality | 4.78 | 1.88 | 2.90 | score ≤ 3 | 0.897 |
| Safety | 4.73 | 2.14 | 2.59 | score ≤ 3 | 0.869 |
| Hallucination | 4.67 | 2.14 | 2.52 | score ≤ 2 | 0.808 |
| Completeness | 4.07 | 2.01 | 2.07 | score ≤ 3 | 0.888 |

The factuality judge is the strongest — F1 of 0.897 means it catches ~98% of wrong answers while only flagging ~17% of correct ones as false positives.

### Cross-provider judge agreement

GPT-4o-mini (cheap, used in CI) vs Claude Sonnet (independent reference), same rubric prompts, same samples (n=380):

| Dimension | Cohen's κ | Interpretation |
|-----------|----------|---------------|
| Factuality | 0.841 | Excellent — safe to automate |
| Completeness | 0.823 | Excellent |
| Safety | 0.777 | Substantial — automate with monitoring |
| Hallucination | 0.699 | Substantial — consider ensemble |

Overall κ = 0.788 with 90.5% within-1-point agreement. The cheap judge is slightly stricter than Claude across the board (bias = -0.17), which is actually what you want for clinical eval.

### Does GPT-4o-mini favor its own family's outputs?

No. Tested by comparing how GPT-4o-mini and Claude score the same responses:

| Whose output is being judged | GPT-4o-mini's score | Claude's score | Difference |
|------------------------------|-------------------|---------------|-----------|
| GPT-4o-mini responses | 3.36 | 3.38 | -0.02 |
| GPT-3.5 responses | 3.18 | 3.45 | -0.27 |
| Claude responses | 3.65 | 3.87 | -0.22 |

If anything, GPT-4o-mini is marginally harder on its own family. No self-favoritism detected.

### Hallucination injection test

Took 30 correct responses and injected fake medical claims to test judge sensitivity:

| What was injected | Avg judge score | Detection rate |
|-------------------|----------------|---------------|
| Nothing (clean baseline) | 4.13 | 33% flagged |
| 1 fabricated claim | 1.93 | 100% caught |
| 2 fabricated claims | 1.73 | 100% caught |
| 3 fabricated claims | 1.00 | 100% caught |

A single injected claim like "According to the 2019 Lancet study, early corticosteroids reduce mortality by 40%" drops the score from 4.13 to 1.93. The judge catches every single one.

## Usage

```bash
git clone https://github.com/YOUR_USERNAME/clinicalbench.git
cd clinicalbench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your API keys

# Full pipeline (~2-3 hours, API calls)
python main.py full-pipeline

# Or step by step
python main.py pull-data           # grab MedQA dataset
python main.py generate-answers    # 3 models answer 200 questions
python main.py run-eval            # 4 judges score all 600 responses
python main.py calibrate           # correctness analysis (instant, no API)
python main.py reference-judge     # Claude as reference evaluator
python main.py compare-judges      # cross-model agreement stats
python main.py inject-test         # hallucination sensitivity test
```

Every step saves progress and picks up where it left off if interrupted.

## How CI gates work

`config.yaml` defines score thresholds per dimension. The idea is that different clinical contexts need different strictness levels:

```yaml
ci_gates:
  safety:
    block_threshold: 2    # auto-reject anything scored ≤ 2
    flag_threshold: 3     # flag for human review if scored ≤ 3

profiles:
  emergency_medicine:
    safety:
      block_threshold: 3  # stricter in ED — block more aggressively
```

Thresholds are calibrated from the correctness analysis. At `safety ≤ 3`, the system catches 93.2% of actually-wrong responses with 81.4% precision.

## Project structure

```
clinicalbench/
├── main.py                  # CLI
├── config.yaml              # CI thresholds, tunable per deployment
├── judges/__init__.py       # 4 rubric-based judge prompts
├── scripts/
│   ├── pull_data.py         # MedQA loader
│   ├── generate_answers.py  # multi-model answer generation
│   ├── run_judges.py        # judge execution
│   ├── correctness_analysis.py
│   ├── proxy_annotate.py    # cross-provider reference judge
│   ├── three_way_comparison.py
│   ├── ablation_judge_model.py
│   └── extract_failure_cases.py
├── data/                    # questions + model answers
└── results/
    ├── eval_scores.jsonl    # 2,400 individual evaluations
    ├── failure_cases/       # curated worst-case examples
    └── *.md                 # reports
```

## What I'd do differently with more time

**Use messier data.** MedQA is multiple-choice — clean and structured. Real clinical AI deals with free-text notes, multi-turn conversations, and ambiguous presentations across dozens of specialties. The judge framework transfers, but the benchmarks need to match the deployment context.

**Get physician annotations.** Claude as a reference judge gives strong cross-provider signal (κ = 0.788), but there's no substitute for a clinician telling you "this response would have gotten my patient killed." Even 50 physician-annotated samples would meaningfully strengthen the calibration.

**Ensemble the hallucination judge.** At κ = 0.699, hallucination is the weakest dimension. Running two different judge models and taking the stricter score would likely push reliability above the 0.8 threshold needed for autonomous gating.

**Fine-tune a smaller judge.** GPT-4o-mini costs ~$0.002 per evaluation — cheap enough for CI, but a LoRA-tuned smaller model could bring that down 10x while maintaining κ > 0.7, based on what I've seen with similar distillation at ServiceNow.

## Cost

The whole pipeline runs for about $10-14 in API calls. The production judge (GPT-4o-mini) costs roughly $0.002 per evaluation, meaning you can evaluate 10,000 clinical responses across all 4 dimensions for ~$20.

## License

MIT
