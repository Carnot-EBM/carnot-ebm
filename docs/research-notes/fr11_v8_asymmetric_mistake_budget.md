# FR-11 V8 Asymmetric Mistake-Budget Audit

Run date: 20260507
Decision: Preserve narrow self-learning claim
Honest verdict: self_learning_claim_preserved_zero_soundness_mistakes

## Audit Framing

This is an asymmetric mistake-budget audit for Exp 1471 under the online
verifier learnability framing in arXiv:2603.03538. Soundness mistakes are
dangerous missed errors that can enter the feedback loop. Completeness mistakes
are conservative false flags that withhold usable cases.

## Evidence

- Source experiment: `experiment_1471_fr11_v8_verified_memory_growth_pivot`
- Source status: `complete`
- Source headline gate: `True`
- Source pivot preserved: `True`
- Soundness mistakes: `0`
- Completeness mistakes: `140`
- Cost weights: `soundness=10.0`,
  `completeness=1.0`
- Asymmetric cost score: `140.0`
- Pareto decision: `preserve_narrow_claim_on_soundness_frontier_with_completeness_caveat`

## Caveats

- Exp 1471 has aggregate soundness/completeness fields and a promoted/demoted
  memory ledger, but not full per-row semantic-state detail in the result JSON.
- The 140 conservative false flags in the live artifact are a completeness and
  candidate-supply limitation, not evidence of memory poisoning.
- The preserved claim, if preserved, is only the narrow Exp 1471 verified
  memory-growth claim. It is not a broad claim that online verifier learning is
  generally complete.
