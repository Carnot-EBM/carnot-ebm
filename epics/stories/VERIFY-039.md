# Epic: VERIFY-039 - Learned Self-Learning Policy Compiler

**Status:** Completed 2026-04-13
**Goal:** Add `python/carnot/pipeline/self_learning_policy.py` so Carnot can
compile accepted repairs and high-precision case-memory evidence into
deterministic, provenance-bearing policy updates for later replay and runtime
use.
**Rationale:** Exp 239 added richer case retrieval, but retrieval alone does
not change behavior. Carnot needs the narrow Tier 1 / Tier 2 bridge that turns
repeated, trusted evidence into concrete threshold updates, property budgets,
repair-prompt patches, and routing hints without replacing the current tracker
or memory paths.

## Stories
- [x] Add `REQ-VERIFY-052`, `REQ-VERIFY-053`, and
  `SCENARIO-VERIFY-056` through `SCENARIO-VERIFY-059` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for deterministic policy compilation, provenance,
  serialization, and additive integration with tracker and case-memory inputs
- [x] Implement `python/carnot/pipeline/self_learning_policy.py` plus any
  minimal public exports needed, without touching `scripts/research_conductor.py`
- [x] Run the required targeted test command, targeted 100% coverage on the
  new module, the full Python suite, spec coverage, and the applicable E2E and
  reconciliation checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
