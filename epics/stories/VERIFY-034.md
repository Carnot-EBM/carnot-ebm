# Epic: VERIFY-034 - Exp 235 Live GSM8K Semantic Benchmark V2

**Status:** Completed 2026-04-13
**Goal:** Re-run the live GSM8K semantic benchmark on the shared Exp 219 cohort
with the refreshed Exp 233 output policy and the calibrated semantic-verifier-v2
path, then write `results/experiment_235_results.json` with direct comparison
against Exp 219.
**Rationale:** Exp 219 proved Carnot could surface real semantic failures on
live GSM8K, but verify-only harmed both target models because false positives
were still too expensive. Exp 235 needs the honest paired rerun: same cohort,
same model pair, new policy plus calibrated verifier, explicit comparison, and
no invented live results if a cell cannot be completed.

## Stories
- [x] Add `REQ-VERIFY-048`, `REQ-VERIFY-049`, `SCENARIO-VERIFY-050`, and
  `SCENARIO-VERIFY-051` to the `verifiable-reasoning` spec before
  implementation changes
- [x] Write tests first for Exp 235 cohort reuse, artifact schema
  compatibility, semantic-v2 summary fields, and direct Exp 219 comparison
- [x] Implement the Exp 235 wrapper around the shared live harness without
  breaking the existing Exp 218-221 artifact schema
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Execute the live Exp 235 benchmark and write
  `results/experiment_235_results.json`
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and any required metrics metadata
