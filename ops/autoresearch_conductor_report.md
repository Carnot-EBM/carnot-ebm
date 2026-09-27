# Autoresearch conductor round

- started: 2026-09-27T08:16:56.958433+00:00
- model: gpt-6-astra
- max_iterations: 5

- iterations: 5
- accepted: 1
- rejected: 4
- pending_review: 0
- circuit_breaker_tripped: False
- breaker_invocation_start_position: 229
- breaker_historical_tail_at_start: 7
- breaker_invocation_local_tail_at_start: 0
- breaker_invocation_local_tail_at_end: 1
- generator_exhausted: False
- fallback_iterations: none


## Generator failure reasons
- 1. **Exact AUROC Evaluation**: Computes the exact Mann-Whitney U rank statistic with average-rank tie resolution in $O(N \log N)$ time.
2. **Direction Alignment**: Evaluates the baseline default `(0.5, 0.5)` to establish the baseline AUROC and align with the evaluation harness's scoring orientation.
3. **Linear Feature Decomposition**: Tests probe linearity. If linear, it extracts basis scores for entity uptake and falsifiability once, enabling exhaustive sweeps across hundreds of candidate weight ratios and angle orientations in milliseconds.
4. **Adaptive Fallback**: If non-linear, it performs a coarse-to-fine direct grid search over probe instances.
5. **Canonical Normalization & Safety**: Normalizes the final weights so $|w_e| + |w_f| = 1.0$ (matching the scale of the default weights `(0.5, 0.5)`), ensures non-degeneracy, and guarantees monotonic improvement over the baseline.: Energy regression on: verifier_auroc
No hypothesis both won this round and committed cleanly.
