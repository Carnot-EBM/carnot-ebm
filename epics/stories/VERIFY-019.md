# Epic: VERIFY-019 - Shared Dual-Model Live Benchmark Harness

**Status:** Completed
**Goal:** Add `scripts/experiment_218_live_dual_model_suite.py` so Carnot can
run checkpointed paired live benchmarks on `Qwen/Qwen3.5-0.8B` and
`google/gemma-4-E4B-it` with shared prompts, shared prompt seeds, and stable
output schemas for Exp 219 through Exp 221.
**Rationale:** Exp 206 through Exp 208 produced useful live evidence, but each
benchmark had its own script, checkpoint pattern, and artifact shape. The next
milestone needs one reusable harness that preserves pairing across models and
modes, resumes cleanly after long runs, and gives later live experiments a
single schema contract instead of one-off adapters.

## Stories
- [x] Add `REQ-VERIFY-025`, `REQ-VERIFY-026`,
  `SCENARIO-VERIFY-025`, and `SCENARIO-VERIFY-026` to the
  `verifiable-reasoning` spec
- [x] Write tests first for the unified CLI surface, deterministic cohort and
  prompt-seed manifest, checkpoint resume behavior, and the stable paired
  payload schema
- [x] Implement `scripts/experiment_218_live_dual_model_suite.py`
- [x] Keep the harness scoped to exactly the three requested benchmarks and
  the two requested target models
- [x] Run targeted tests, targeted 100% coverage on the new script, the full
  Python suite, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
