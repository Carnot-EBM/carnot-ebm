# Epic: VERIFY-017 - Structured Reasoning Emission Path For Monitorable Outputs

**Status:** Completed
**Goal:** Add `python/carnot/pipeline/structured_reasoning.py` so Carnot can
request monitorable structured outputs from supported small models, validate
them into typed reasoning IR, and fall back safely when the structured
emission path fails.
**Rationale:** Exp 213 established that structured output is only worth the
extra tokens on specific task slices, while Exp 212 and Exp 215 already gave
Carnot typed and semantic verifier layers that benefit from monitorable
intermediate state. This work closes the loop by adding the actual
policy-gated structured emission path instead of relying on raw prose alone.

## Stories
- [x] Add `REQ-VERIFY-022`, `REQ-VERIFY-023`, `REQ-VERIFY-024`,
  `SCENARIO-VERIFY-022`, `SCENARIO-VERIFY-023`, and
  `SCENARIO-VERIFY-024` to the verifiable-reasoning spec
- [x] Write tests first for clean structured emissions, malformed-output
  retries, policy gating, and the additive pipeline entry point
- [x] Implement `python/carnot/pipeline/structured_reasoning.py`
- [x] Expose the structured emission path through an additive
  `VerifyRepairPipeline` entry point without breaking existing flows
- [x] Run unit tests, the full Python suite, targeted 100% coverage on the
  new module, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
