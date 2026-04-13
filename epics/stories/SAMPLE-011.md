# Epic: SAMPLE-011 - Sampler-Backed Repair Reranking Replay Benchmark

**Status:** Completed 2026-04-13
**Goal:** Build a deterministic Exp 243 replay benchmark over the checked-in
Exp 235 semantic and Exp 238 code repair histories, then rerank those saved
candidates through the CPU sampler path and the KV260 path when available.
**Rationale:** Exp 242 measured the control-plane round trip honestly, but it
did not show whether the sampler path changes a real Carnot task. Exp 243 is
the smallest honest bridge from hardware plumbing to energy-guided inference:
reuse saved repair attempts, replay candidate selection deterministically, and
report whether sampler-backed reranking changes quality or latency.

## Stories
- [x] Add `REQ-SAMPLE-008` and `SCENARIO-SAMPLE-015` through
  `SCENARIO-SAMPLE-017` to the `training-inference` spec before
  implementation changes
- [x] Write tests first for candidate-set replay, scorer integration,
  honest backend labeling, and result-summary reporting
- [x] Implement the Exp 243 replay benchmark and artifact writer
- [x] Run targeted 100% coverage for the new code plus the required Python
  suite, spec-coverage, lint, type-check, and applicable E2E/integration
  checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
