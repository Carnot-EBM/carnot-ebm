# Autoresearch conductor round

- started: 2026-11-19T15:54:29.262161+00:00
- model: gpt-6-astra
- max_iterations: 5

- iterations: 5
- accepted: 1
- rejected: 4
- pending_review: 0
- circuit_breaker_tripped: False
- breaker_invocation_start_position: 8332
- breaker_historical_tail_at_start: 29
- breaker_invocation_local_tail_at_start: 0
- breaker_invocation_local_tail_at_end: 2
- generator_exhausted: False
- fallback_iterations: none


## Generator failure reasons
- Code Implementation: Time budget exceeded
- Previous attempts failed due to **"Time budget exceeded"** because naive grid searches re-evaluate `.score(step_text, "")` across all corpus rows hundreds of times. We resolve this by precomputing the basis signals using two orthogonal probes—`Probe(1.0, 0.0)` and `Probe(0.0, 1.0)`—requiring only two total corpus passes ($\approx 0.15\text{s}$). The composite scores for any candidate mixture are then computed via vector broadcasting in NumPy, evaluating hundreds of candidates via exact Mann-Whitney U AUROC in milliseconds without invoking the probe again. We also include a degenerate check ($\sigma > 10^{-6}$) and an orientation check to ensure non-degenerate, strictly superior weights that finish within $\approx 0.35\text{s}$.: Energy regression on: verifier_auroc
No hypothesis both won this round and committed cleanly.
