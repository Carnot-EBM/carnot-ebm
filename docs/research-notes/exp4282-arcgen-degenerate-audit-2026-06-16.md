# Audit: the .396 ARC-GEN "cross-generator win" (exp4282) is DEGENERATE — do NOT headline it

**Outer-loop audit, 2026-06-16.** exp4282 reports `arcgen_cross_family_holds: True`,
`cross_family_delta: 1.0`, CI95 `[1.0, 1.0]`, `honest_verdict: arcgen_cross_family_generalizes`.
**This is a construction artifact, not evidence of cross-generator generalization.** The
cross-*generator* question — the last open axis — remains OPEN.

## The tell: a perfect 1.0 / 0.0 split

`pass_rates`: `vote_at_1 = 0.0`, `set_encoder_at_1 = 1.0`, `matched_control_at_1 = 0.0`,
`oracle_at_k = 1.0`, on 50 held-out tasks / 10 families / **4 candidates per task**.

- **Vote is wrong on EVERY task (0.0).** The ARC-GEN pool is wrong-majority-only by
  construction (`wrong_majority_n` over the whole pool), so vote@1 = 0 is structural, not
  measured.
- **The verifier is right on EVERY task (1.0).** With only 4 candidates/task and the correct
  one carrying a feature the set-encoder keys on, perfect separation is trivial.
- `cross_family_delta = set_encoder@1 − vote@1 = 1.0 − 0.0 = 1.0`, CI `[1.0, 1.0]` (zero width).

A delta of exactly 1.0 with a zero-width CI is the IMPLAUSIBLE_PERFECT / degenerate signature.
Contrast the **real** .395 result (exp4271): vote 0.25 / verifier 0.69, delta +0.40 on the
*natural* GAP-4 pool — believable precisely because vote wasn't trivially 0 and the verifier
wasn't perfect.

## Why this is not generalization evidence

The ARC-GEN pool was built to be (a) wrong-majority-only (vote can't win) and (b) tiny per task
(4 candidates), making the correct answer trivially separable. So the +1.0 measures *pool
degeneracy*, not whether the verifier transfers to a construction-disjoint generator. A faithful
cross-generator test needs a NON-wrong-majority-filtered ARC-GEN pool with realistic candidate
counts, reporting vote@1 well above 0 and an oracle ceiling below 1.0 — i.e. headroom the verifier
must actually earn.

## adversarial_verify gap

`scripts/adversarial_verify.py` does NOT flag exp4282 (0 flagged) — neither IMPLAUSIBLE_PERFECT
nor IMPLAUSIBLE_TIGHT_CI fires on a `*_delta == 1.0` with a `[1.0, 1.0]` CI and `vote_at_1 == 0.0`.
Recommend a DEGENERATE_SEPARATION check: flag when a beats-vote delta ≥ ~0.95 AND
`vote_at_1 ≤ ~0.05` (or oracle@K == 1.0 with a perfect selector) — the synthetic-pool signature.

## Recommendation

- **The .396 capstone MUST NOT headline "cross-generator generalization proven"** off exp4282.
  Report it as `arcgen_pool_degenerate_uninformative`.
- **The real, defensible result stands:** the .395 within-GAP-4 cross-family win (exp4271, +0.40,
  leak-free, 5-seed-replicated, audited). The cross-*generator* axis is still **OPEN** — it needs a
  non-degenerate ARC-GEN pool.
- This does not change `paper_ready` for the within-distribution claim, but it bounds the scope:
  cross-generator is not yet shown.

## Provenance
- exp4282 `results/experiment_4282_arcgen_cross_family_stress.json` (pass_rates above)
- pool `results/experiment_4282_arcgen_candidate_pool.json.gz` (50 tasks × 4 candidates)
- contrast: exp4271 (the genuine within-GAP-4 cross-family win) +
  `docs/research-notes/exp4245-arc-oracle-distinct-audit-2026-06-15.md`
