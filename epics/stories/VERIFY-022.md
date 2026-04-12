# Epic: VERIFY-022 - Exp 222 Live Trace Memory And Repair Guidance

**Status:** Completed
**Goal:** Build a provenance-aware Exp 222 workflow that ingests the checked-in
live Exp 219 through Exp 221 artifacts, converts high-confidence verifier
outcomes into reusable `ConstraintMemory` entries, derives repair snippets and
prompt patches from live repair histories, and writes
`results/experiment_222_results.json` plus
`results/constraint_memory_live_222.json`.
**Rationale:** Tier 1 and Tier 2 learning primitives already exist, but Carnot
is still not learning from the exact live failures it now understands best.
Exp 222 needs to turn current live verifier traces into reusable memory while
preserving provenance so false positives and ambiguous traces do not pollute
the learned state.

## Stories
- [x] Add `REQ-VERIFY-030` through `REQ-VERIFY-032` and
  `SCENARIO-VERIFY-030` through `SCENARIO-VERIFY-032` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for trace normalization, provenance gating, memory
  growth, repair-snippet extraction, policy-update derivation, and artifact
  refresh behavior
- [x] Implement the Exp 222 ingestion module and
  `scripts/experiment_222_live_trace_memory.py`
- [x] Execute the workflow and write `results/experiment_222_results.json`
  plus `results/constraint_memory_live_222.json`
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
