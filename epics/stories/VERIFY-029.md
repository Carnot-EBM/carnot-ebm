# Epic: VERIFY-029 - Seeded Qwen HumanEval PBT Benchmark On The Exp 208 Cohort

**Status:** Complete
**Goal:** Add `scripts/experiment_227_qwen_pbt.py` so Carnot can run a live
30-problem HumanEval PBT benchmark on `Qwen/Qwen3.5-0.8B`, reuse the exact
ordered cohort recorded in `results/experiment_208_results.json`, and emit an
explicit Qwen-vs-Gemma comparison artifact.
**Rationale:** Exp 208 provides the checked-in 30-problem Gemma cohort and
Exp 226 proves the Hypothesis-backed PBT workflow on Gemma at full scale. The
repository now needs the same seeded cohort exercised on Qwen3.5-0.8B to check
whether the code-verification approach transfers across model families.

## Stories
- [x] Add `REQ-CODE-015` and `SCENARIO-CODE-013` to the `code-verification`
  spec before implementation changes
- [x] Write tests first for cohort reuse from Exp 208, Qwen live-model wiring,
  PBT verify-repair flow, Gemma comparison summaries, artifact writing, and the
  CLI entrypoint
- [x] Implement `scripts/experiment_227_qwen_pbt.py`
- [x] Run targeted 100% coverage for the new script plus the required suite,
  spec-coverage, lint, integration, and reconciliation checks
- [x] Run the live 30-problem Qwen benchmark and record the honest artifact
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
