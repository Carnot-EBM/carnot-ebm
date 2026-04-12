# Epic: VERIFY-021 - Exp 220 Live HumanEval Property Artifact

**Status:** Completed
**Goal:** Extend `scripts/experiment_218_live_dual_model_suite.py` so the
`humaneval_property` run can write `results/experiment_220_results.json` with
paired execution-only vs execution-plus-property verification metrics,
repair outcomes, and per-problem traces for later self-learning.
**Rationale:** Exp 217 added the prompt-derived property verifier, but the
shared Exp 218 harness still summarizes HumanEval too narrowly for the live
dual-model milestone. Exp 220 needs an honest artifact that distinguishes what
execution-only checks catch from what the new property-derived verifier adds on
the same 50-problem paired cohort.

## Stories
- [x] Add `REQ-VERIFY-028` and `SCENARIO-VERIFY-028` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for Exp 220 experiment metadata, HumanEval summary
  fields, and preserved execution/repair trace artifacts
- [x] Implement the shared-harness changes needed for
  `results/experiment_220_results.json`
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Execute the live Exp 220 benchmark and write
  `results/experiment_220_results.json`
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
