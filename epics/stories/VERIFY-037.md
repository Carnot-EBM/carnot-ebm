# Epic: VERIFY-037 - Exp 238 Identical-Stack Dual-Model HumanEval Benchmark

**Status:** Completed 2026-04-13
**Goal:** Add `scripts/experiment_238_dual_model_spec_code.py` so Carnot can
run `Qwen/Qwen3.5-0.8B` and `google/gemma-4-E4B-it` on the same seeded
HumanEval cohort with the same official-tests + PBT + explicit-spec verifier
stack and the same repair budget, then publish `results/experiment_238_results.json`.
**Rationale:** Exp 226 proved the PBT path on full HumanEval for Gemma, Exp 227
showed the seeded Qwen cohort story but compared against a pre-Hypothesis
Gemma reference, and Exp 236/237 now make explicit specs available in-tree. The
next honest comparison is one same-stack, same-cohort, same-budget rerun across
both target models.

## Stories
- [x] Add `REQ-CODE-028` through `REQ-CODE-030` and
  `SCENARIO-CODE-026` through `SCENARIO-CODE-028` to the
  `code-verification` spec before implementation changes
- [x] Write tests first for shared cohort reuse, comparison summaries,
  blocker-aware partial artifacts, and the final artifact schema
- [x] Implement `scripts/experiment_238_dual_model_spec_code.py` without
  touching `scripts/research_conductor.py`
- [x] Run targeted 100% coverage for the new code plus spec coverage, the
  applicable workflow-level E2E validation, and the live benchmark command.
  The required full-suite command `.venv/bin/pytest tests/python -q` was
  executed, but it still exits nonzero in the current worktree because the
  repo-wide coverage gate is **99%** and the measured total is **93.31%**
  from pre-existing unrelated coverage debt.
- [x] Run the live benchmark command and record the honest Exp 238 artifact
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, and
  `ops/changelog.md`
