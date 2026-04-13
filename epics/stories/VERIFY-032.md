# Epic: VERIFY-032 - Exp 233 Output Policy Refresh

**Status:** Completed 2026-04-13
**Goal:** Refresh Carnot's small-model output-mode policy with a larger mixed
slice that compares terse output against minimal and grammar-gated JSON, then
publish a machine-readable policy artifact later experiments can consume
directly.
**Rationale:** Exp 213 established that forcing structure everywhere is a
mistake, but it used an 11-example slice and one structured mode. Exp 233
updates the policy around minimal schema choice, retry budget, and task-gated
mode routing without touching `scripts/research_conductor.py`.

## Stories
- [x] Add `REQ-VERIFY-044`, `REQ-VERIFY-045`, `SCENARIO-VERIFY-045`, and
  `SCENARIO-VERIFY-046` to the `verifiable-reasoning` spec before
  implementation changes
- [x] Write tests first for Exp 233 mode routing, metric aggregation, and
  deterministic artifact generation
- [x] Implement the Exp 233 benchmark script, refreshed policy artifact, and
  any direct policy-consumer routing updates needed for later experiments
- [x] Run the required Python suite, targeted 100% coverage checks for new
  code, spec coverage, lint/type checks, and the applicable workflow-level
  end-to-end validation
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
