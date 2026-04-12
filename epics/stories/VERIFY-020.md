# Epic: VERIFY-020 - Exp 219 Live GSM8K Semantic Artifact

**Status:** Completed
**Goal:** Extend `scripts/experiment_218_live_dual_model_suite.py` so the
`gsm8k_semantic` run can write `results/experiment_219_results.json` with the
correct experiment metadata, semantic benchmark metrics, and per-question trace
artifacts needed for follow-on learning.
**Rationale:** Exp 218 delivered the shared harness, but the follow-on Exp 219
deliverable needs its own artifact contract. The current harness still labels
the output as experiment 218 and summarizes GSM8K too narrowly for the
semantic-verifier milestone.

## Stories
- [x] Add `REQ-VERIFY-027` and `SCENARIO-VERIFY-027` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for Exp 219 experiment metadata, GSM8K summary fields,
  and preserved semantic trace artifacts
- [x] Implement the shared-harness changes needed for
  `results/experiment_219_results.json`
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Execute the live Exp 219 benchmark and write
  `results/experiment_219_results.json`
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
