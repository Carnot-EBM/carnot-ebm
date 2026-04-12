# Epic: VERIFY-012 - Constraint IR Benchmark for Semantic Grounding

**Status:** Completed
**Goal:** Publish a deterministic Exp 211 benchmark at
`data/research/constraint_ir_benchmark_211.jsonl` plus a summary artifact at
`results/experiment_211_results.json` that defines the prompt-side constraint
IR Carnot should target next.
**Rationale:** Exp 203, Exp 206, and Exp 207 showed that live wrong answers on
Gemma4-E4B-it are mostly semantic/question-grounding failures. Before building
another verifier, Carnot needs a benchmark that states which atomic
constraints, verifier paths, and answer schemas should be extracted in the
first place.

## Stories
- [x] Add `REQ-VERIFY-011`, `REQ-VERIFY-012`, `SCENARIO-VERIFY-011`, and `SCENARIO-VERIFY-012` to the verifiable-reasoning spec
- [x] Write tests first for the deterministic Exp 211 benchmark generator and artifact schema
- [x] Implement `scripts/experiment_211_constraint_ir_benchmark.py`
- [x] Generate `data/research/constraint_ir_benchmark_211.jsonl` and `results/experiment_211_results.json`
- [x] Run unit tests, full Python test suite, targeted 100% coverage on the new script, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`
