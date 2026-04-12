# Epic: VERIFY-030 - Self-Learning From Code Verification Traces

**Status:** Complete
**Goal:** Add `python/carnot/pipeline/code_learning.py` so Carnot can learn from
the checked-in Exp 225 / Exp 226 verification artifacts, rank the most useful
PBT properties, and recommend repair strategies from accumulated code-repair
history.
**Rationale:** Exp 226 now captures rich per-problem code-verification traces,
but the pipeline has no checked-in learner that turns those traces into
actionable property priorities or repair guidance. This story closes that gap
without changing `scripts/research_conductor.py`.

## Stories
- [x] Add `REQ-CODE-016`, `REQ-CODE-017`, `REQ-CODE-018`,
  `SCENARIO-CODE-014`, and `SCENARIO-CODE-015` to the
  `code-verification` spec before implementation changes
- [x] Write tests first for artifact parsing, property ranking, repair-strategy
  learning, and cumulative-learning behavior
- [x] Implement `python/carnot/pipeline/code_learning.py` and export it through
  `carnot.pipeline`
- [x] Run targeted 100% coverage for the new module plus the required suite,
  spec-coverage, and applicable E2E/integration validation
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
