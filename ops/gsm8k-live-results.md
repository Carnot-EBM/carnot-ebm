# Experiment 91: GSM8K Live Benchmark Results

**Date:** 2026-04-10T00:54:59Z
**Dataset:** GSM8K test (200 real, 0 synthetic)
**Questions:** 200
**Total time:** 1.8s

## Qwen3.5-0.8B (SIMULATED)

| Mode | Correct | Accuracy |
|------|---------|----------|
| Baseline | 130/200 | 65.0% |
| Verify-only | 130/200 | 65.0% |
| Verify+Repair | 160/200 | 80.0% |

**Δ accuracy (Repair vs Baseline):** +30 questions (+15.0%)

**Constraint coverage:** 162/200 questions had extractable arithmetic claims
**Total arithmetic steps found:** 162 (ad-hoc), 162 (pipeline)

**Hallucination detection (verify-only):**
- True positives: 32, True negatives: 130
- False positives: 0, False negatives: 38
- Detection accuracy: 81.0%, Precision: 100.0%, Recall: 45.7%

**Error categorization:**

| Type | Count | % of Errors |
|------|-------|-------------|
| arithmetic | 32 | 46% |
| logic | 30 | 43% |
| reading | 8 | 11% |

**Repair stats:** 30 questions repaired, avg 1.0 iterations

**Timing (per question):**
- Baseline: 0.000s avg
- Verify-only: 0.002s avg
- Verify+Repair: 0.000s avg

## Gemma4-E4B-it (SIMULATED)

| Mode | Correct | Accuracy |
|------|---------|----------|
| Baseline | 149/200 | 74.5% |
| Verify-only | 149/200 | 74.5% |
| Verify+Repair | 177/200 | 88.5% |

**Δ accuracy (Repair vs Baseline):** +28 questions (+14.0%)

**Constraint coverage:** 177/200 questions had extractable arithmetic claims
**Total arithmetic steps found:** 177 (ad-hoc), 177 (pipeline)

**Hallucination detection (verify-only):**
- True positives: 28, True negatives: 149
- False positives: 0, False negatives: 23
- Detection accuracy: 88.5%, Precision: 100.0%, Recall: 54.9%

**Error categorization:**

| Type | Count | % of Errors |
|------|-------|-------------|
| arithmetic | 28 | 55% |
| logic | 17 | 33% |
| reading | 6 | 12% |

**Repair stats:** 28 questions repaired, avg 1.0 iterations

**Timing (per question):**
- Baseline: 0.000s avg
- Verify-only: 0.000s avg
- Verify+Repair: 0.000s avg

## Cross-Model Comparison

| Model | Baseline | Verify | Repair | Δ |
|-------|----------|--------|--------|---|
| Qwen3.5-0.8B | 65.0% | 65.0% | 80.0% | +30 |
| Gemma4-E4B-it | 74.5% | 74.5% | 88.5% | +28 |
