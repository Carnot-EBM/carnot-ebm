# Epic: VERIFY-040 - Chronological Self-Learning Replay V2

**Status:** Completed 2026-04-13
**Goal:** Extend the replay workflow so Carnot can evaluate continuous
self-learning on newer semantic and code artifacts, compare
`no_learning` / `tracker_only` / `case_memory` /
`case_memory_plus_policy`, and write `results/experiment_241_results.json`.
**Rationale:** Exp 223 showed that live-only replay can cut held-out false
positives without producing held-out task gain. Exp 239 and Exp 240 added the
missing ingredients: richer case retrieval and provenance-bearing policy
updates. Exp 241 is the milestone check for whether those additions produce a
real held-out improvement on future runs without breaking the false-positive
budget.

## Stories
- [x] Add `REQ-VERIFY-054`, `REQ-VERIFY-055`, and
  `SCENARIO-VERIFY-060` through `SCENARIO-VERIFY-062` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for four-way replay branching, metric summaries, and
  compatibility across the Exp 235 and Exp 238 artifact schemas
- [x] Implement the Exp 241 replay path and CLI without touching
  `scripts/research_conductor.py`
- [x] Generate `results/experiment_241_results.json` with fixed run-date
  metadata `20260413`
- [x] Run the required targeted command, targeted 100% coverage for the new or
  changed replay code, the full Python suite, spec coverage, and the
  applicable E2E and reconciliation checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
