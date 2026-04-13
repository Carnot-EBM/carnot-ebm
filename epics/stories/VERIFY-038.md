# Epic: VERIFY-038 - Additive Case-Based Memory For Live Replay

**Status:** Completed 2026-04-13
**Goal:** Add `python/carnot/pipeline/case_memory.py` so Carnot can reuse
specific live cases keyed by model, benchmark slice, violation family, prompt
sketch, property names, and repair outcome without replacing the existing Exp
222 / Exp 223 pattern-memory path.
**Rationale:** Exp 222 and Exp 223 showed that raw pattern reuse is too coarse:
domain-wide `error_type` buckets do not distinguish which prompt shape, code
property, or repair history actually transfers. Carnot needs a cheap,
deterministic case-memory layer that improves targeting while staying additive
to the checked-in live replay artifacts.

## Stories
- [x] Add `REQ-VERIFY-050`, `REQ-VERIFY-051`, and
  `SCENARIO-VERIFY-052` through `SCENARIO-VERIFY-055` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for case normalization, retrieval ranking,
  serialization, and additive replay integration
- [x] Implement `python/carnot/pipeline/case_memory.py` plus the additive
  `self_learning_replay.py` hook without touching `scripts/research_conductor.py`
- [x] Run the required targeted command, targeted 100% coverage for the new
  module, the full Python suite, spec coverage, and the applicable E2E and
  reconciliation checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
