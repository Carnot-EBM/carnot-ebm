# Epic: VERIFY-011 - Live HumanEval Verify-Repair on Gemma4-E4B-it

**Status:** Completed
**Goal:** Run 30 live HumanEval problems through Carnot's code verification
stack on Gemma4-E4B-it, then publish the paired baseline vs verify-repair
artifact at `results/experiment_208_results.json`.
**Rationale:** Code generation is the domain most likely to benefit from
structural verification because Carnot can combine AST-based `CodeExtractor`
signals, Exp 53 runtime instrumentation, and the official HumanEval tests in a
single repair loop.

## Stories
- [x] Confirm the existing verifiable-reasoning spec already covers this work via `REQ-VERIFY-001`, `REQ-VERIFY-002`, `REQ-VERIFY-003`, and `SCENARIO-VERIFY-006`
- [x] Write tests first for the reusable HumanEval live benchmark helper
- [x] Implement `python/carnot/pipeline/humaneval_live_benchmark.py`
- [x] Implement `scripts/experiment_208_humaneval_live_it.py`
- [x] Run unit tests, the full Python suite, targeted 100% coverage on new helper code, and the applicable integration/E2E checks
- [x] Execute the live Gemma4-E4B-it benchmark and publish `results/experiment_208_results.json`
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`
