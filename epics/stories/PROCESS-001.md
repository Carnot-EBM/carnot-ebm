# PROCESS-001 — Complete Apple Adversarial Benchmark (Deferred from 2026.04.21)

**Status:** Open
**Origin:** Operational Retrospective 2026.04.21, action item RETRO-2026-04-20-C
**Carry-over from:** 2026.04.20 retro (original origin: 2026.04.19)
**Consecutive milestones deferred:** 2

## Problem

Experiments 282 and 283 ran in blocked/partial mode because live GPU inference
was unavailable in the CI environment.  As a result, Exp 284 returned
INCONCLUSIVE — the Apple adversarial benchmark has never been completed
end-to-end.

## Acceptance Criteria

- [ ] Exp 282 re-runs with a live GPU (or a simulated-GPU fallback that produces
  meaningful logits) and writes `results/experiment_282_results.json` with
  `classification` field set.
- [ ] Exp 283 re-runs on the same corpus and writes
  `results/experiment_283_results.json` with `classification` field set.
- [ ] Exp 284 re-runs with both inputs present and produces a CONFIRMED or
  REFUTED verdict in `results/experiment_284_results.json`.
- [ ] All three result files are committed and the Exp 284 `classification`
  field is not INCONCLUSIVE.

## Why This Matters

The Apple adversarial dataset (Exp 281, 400 rows) is the primary evaluation
harness for verify-repair's robustness against number-swap and irrelevant-sentence
attacks.  Until this benchmark is complete, we cannot claim verify-repair
improves on the adversarial distribution.

## Suggested Next Steps

1. Add a `--simulated-gpu` flag to Exp 282/283 scripts so they can run in CI.
2. Schedule a conductor turn that exports `LIVE_GPU=1` and re-runs Exp 282→283→284
   in sequence.
3. Update `ops/status.md` and `_bmad/traceability.md` once CONFIRMED or REFUTED.
