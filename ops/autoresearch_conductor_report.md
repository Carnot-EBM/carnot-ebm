# Autoresearch conductor round

- started: 2026-11-16T20:53:26.153644+00:00
- model: gpt-6-astra
- max_iterations: 5

- iterations: 5
- accepted: 0
- rejected: 5
- pending_review: 0
- circuit_breaker_tripped: False
- breaker_invocation_start_position: 8273
- breaker_historical_tail_at_start: 30
- breaker_invocation_local_tail_at_start: 0
- breaker_invocation_local_tail_at_end: 5
- generator_exhausted: False
- fallback_iterations: none


## Generator failure reasons
- 1. **Exact AUROC & Fisher Separation Metric**: Measures exact pairwise rank-sum AUROC on the training rows. In case of discrete ties on AUROC, the normalized separation distance ($d'$) is used as a secondary criterion to prefer parameters that push the score distributions furthest apart.
2. **Degeneracy Protection**: Explicitly excludes $(0.0, 0.0)$ and detects degenerate score distributions ($\max - \min < 10^{-9}$), ensuring the final state is non-degenerate.
3. **Two-Stage Coarse-to-Fine Search**:
   - **Coarse Phase**: Sweeps simplex combinations $w \in [0, 1]$ with $(w, 1-w)$, a 2D grid over $[0, 1] \times [0, 1]$, and negative quadrants if supported by the probe class.
   - **Fine Phase**: Refines a neighborhood around the highest-scoring candidate with sub-grid increments ($\pm 0.005$ to $\pm 0.04$).
4. **Time-Bounded Execution**: Uses a strict wall-clock budget to guarantee fast completion without timeout risk.: Time budget exceeded
- ---: Energy regression on: verifier_auroc
- ---: Energy regression on: verifier_auroc, calibrated_decision
- ---: Energy regression on: verifier_auroc
- ---: Sandbox failed: ValueError: cannot reshape array of size 1 into shape (4,)
No hypothesis both won this round and committed cleanly.
