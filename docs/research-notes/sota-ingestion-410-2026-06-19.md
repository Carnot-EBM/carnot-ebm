# SOTA ingestion 2026-06-19: .410 example-corpus solver map for .411

reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv API / arXiv abs-page checks, and low-concurrency WebSearch/WebFetch. The
Semantic Scholar focused queries returned HTTP 429, so no S2-only source was
promoted. `/deep-research` was not invoked. No leaderboard submission was made.

## .410 outcome conditioning

- Exp 4432: leave-one-out generic transfer solved 2 of 7 reproduction-gated targets.
- Exp 4433: example-conditioned win induction reproduced `g50t` L1.
- Exp 4434: example-conditioned world-model synthesis improved accuracy from
  0.714286 to 1.0, but added zero reproduced levels.
- Exp 4435: generic first contact on `dc22` still logged an open verifier gap.
- Exp 4436: `tu93` deepened to L5 and consolidated reusable primitives.

## Verified SOTA methods

- LILO documented library induction, arXiv:2310.19791.
- DreamCoder wake-sleep library learning, arXiv:2006.08381.
- Stitch top-down synthesis, arXiv:2211.16605.
- HYSYNTH context-free LLM approximation, arXiv:2405.15880.
- CodeARC differential-query program induction, arXiv:2503.23145.
- Executable ARC-AGI-3 world models, arXiv:2605.05138.
- Loop-OWM composable world models, arXiv:2606.12316.
- ARC-TGI generator-backed task families, arXiv:2603.05099.

## SOTA->experiment mapping

The `.411` planner should build a documented primitive-library induction pass:
compress solved predicates, executable world models, and the primitive ledger;
name and document each primitive; retrieve those primitives for first-contact
games; and require held-out reproduction gates before any primitive is counted.

flagged_for_v411: LILO-style documented library induction over the ARC solver/example corpus (arXiv:2310.19791)
