# Carnot Experiment Log

Documenting all experiments — what worked, what failed, and what we learned.

## Summary Table

| # | Experiment | Result | Key Learning |
|---|-----------|--------|-------------|
| 1 | SAT verify-and-repair (random) | ✅ +91.6% energy reduction | Gradient repair works on continuous relaxation |
| 2 | Haiku SAT benchmark (20 inst) | ✅ 60% → 80% (+20%) | First real LLM hallucination → EBM repair |
| 3 | Multi-start repair on stubborn | ✅ Fixed instance single-start missed | P2 self-verification adds value on hard cases |
| 4 | Sonnet SAT (15v/50c) | ✅ 100% | Sonnet is too strong for SAT — need harder tasks or weaker model |
| 5 | Sonnet SAT (20v/85c) | ✅ 100% | Sonnet still perfect at phase transition |
| 6 | Sonnet SAT (30v/128c) | ⚠️ 1/3 parsed, correct | Parse failures on long output, not reasoning failures |
| 7 | Code verification (11 tasks) | ✅ 100% (both Sonnet and Haiku) | LLMs ace well-known algorithms — need novel tasks |
| 8 | Real hallucination detection | ✅ 64% detection accuracy | Activation-space direction IS real (energy gap +9.3) |
| 9 | Rejection: linear direction (25 cal) | ❌ 68% → 56% (-12%) | Prompt activations ≠ answer activations |
| 10 | Rejection: linear direction (93 cal) | ⚠️ 60% → 60% (net zero) | 4 fixes + 4 regressions cancel out |
| 11 | Rejection: Gibbs EBM (2048-dim) | ❌ 94% cal → 35% test | Classic overfitting: 42 examples in 2048-dim space |
| 12 | Rejection: PCA + Gibbs (4/8/16/32) | ❌ Best: PCA-8 at -5% | PCA reduces overfitting but mean-pooling is wrong feature |
| 13 | Rejection: logprob-based | ✅ 45% → 55% (+10%) | **Simplest approach wins.** Model's own confidence IS the energy. |

## Detailed Experiment Notes

### Experiment 9: Linear Hallucination Direction — Wrong Features
**Date:** 2026-04-05
**Hypothesis:** Mean-pooled activations from correct vs hallucinated answers define a linear separation direction. Use it to rank rejection sampling candidates.
**Result:** 68% → 56% (-12%)
**Root cause:** Extracted activations from the INPUT PROMPT (model(**inputs)), not from the GENERATED TOKENS. The prompt activations are identical across candidates since the prompt is the same.
**Fix applied:** Extract from generated tokens (model(full_sequence), slice to prompt_len:).

### Experiment 10: Scaled Calibration — Confidence ≠ Correctness
**Date:** 2026-04-05
**Hypothesis:** More calibration data (93 vs 25) will give a better direction.
**Result:** 60% → 60% (net zero: 4 fixes, 4 regressions)
**Learning:** The mean-difference direction captures confidence (uncertain vs certain), not correctness (right vs wrong). A confident hallucination looks identical to a confident correct answer. The direction can't distinguish them.

### Experiment 11: Nonlinear Gibbs EBM — Overfitting
**Date:** 2026-04-05
**Hypothesis:** A nonlinear Gibbs model [2048→256→64→1] trained via NCE can learn the curved boundary between correct and hallucinated activation patterns.
**Result:** 94% calibration accuracy, 35% test accuracy (-25%)
**Learning:** 42 examples in 2048-dim space → severe overfitting. The model memorizes calibration data perfectly but learns nothing generalizable. Need either: much more data (500+), dimensionality reduction, or regularization.

### Experiment 12: PCA + Gibbs — Mean-Pooling Is Wrong
**Date:** 2026-04-05
**Hypothesis:** PCA to 8-32 dims before Gibbs training prevents overfitting.
**Result:** PCA-8 best at -5%, all dims still regress
**Learning:** The problem isn't dimensionality — it's that mean-pooled activations lose token-level signal. The model's state while generating "Paris" vs "garbage" differs at individual token positions, but mean-pooling washes this out. Token-level features needed.

### Experiment 13: Logprob Selection — Simplicity Wins
**Date:** 2026-04-05
**Hypothesis:** Per-token log-probabilities (the model's own confidence per token) are a better energy signal than activation mean-pooling.
**Result:** 45% → 55% (+10%), 4 fixes, 2 regressions, net +2
**Learning:** The simplest approach works best. No calibration, no training, no PCA needed. The model's logprobs at each generated token ARE the right energy function. Higher total logprob = more confident = more likely correct. This validates the semantic energy paper (arxiv 2508.14496).

| 14 | Composite: logprob + structural (code) | ✅ 0% → 30% | Structural tests dominate for code; logprobs dominate for QA |
| 15 | Activation steering (in-generation) | ⚠️ 76% → 76% | Zero effect — mean-difference direction doesn't causally drive token choices |
| 16 | Concept steering (6 configs) | ⚠️ 75% → 75% | Zero effect across all layer/alpha combos — needs concept-specific prompting |

## Key Principles Learned

1. **Simpler is better for small-data regimes.** Linear projections > nonlinear models when you have <100 training examples.

2. **Token-level features > sequence-level.** Mean-pooling destroys the signal. Per-token logprobs preserve it.

3. **The model's own confidence is the best energy.** No external EBM needed for rejection sampling — the LLM's logprobs are already an energy function (as the ARM↔EBM bijection paper proved).

4. **Overfitting is the main enemy.** Every approach that trains on calibration data overfits when examples < dimensions.

5. **Extract features from the RIGHT part of the computation.** Prompt activations ≠ answer activations. The signal is in the GENERATED tokens, not the input.

6. **Different energy signals dominate in different domains.** Logprobs (LLM confidence) work best for QA/factual. Structural tests (EBM execution) work best for code. The composite combines both and is never worse than either alone.

7. **Statistical difference ≠ causal influence.** A direction that separates correct from hallucinated activations (experiment 8: 64% detection) does NOT necessarily steer the model when injected during generation (experiments 15-16: 0% effect). Effective steering requires concept-specific vectors found via targeted prompting (Anthropic's approach), not generic contrastive means.
