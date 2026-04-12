# Epic: VERIFY-028 - Full HumanEval PBT Benchmark On Gemma4-E4B-it

**Status:** Complete
**Goal:** Add `scripts/experiment_226_pbt_humaneval_full.py` so Carnot can run
the full official 164-problem HumanEval benchmark on live
`google/gemma-4-E4B-it`, use `PBTCodeVerifier` inside the verify-repair loop,
checkpoint every 10 problems, and emit a publishable summary with bootstrap
confidence intervals plus published-baseline comparison.
**Rationale:** Exp 224 proved the Hypothesis-backed verifier on a deterministic
five-problem slice, and Exp 225 showed the local host can reduce long-run wall
time enough to justify the full benchmark. The repository now needs the
publishable 164-problem result with honest resume support and direct comparison
to the model's published HumanEval baseline.

## Stories
- [x] Add `REQ-CODE-012` through `REQ-CODE-014` and
  `SCENARIO-CODE-011` through `SCENARIO-CODE-012` to the
  `code-verification` spec before implementation changes
- [x] Write tests first for the full-run checkpoint contract, per-case
  PBT-guided verify-repair flow, bootstrap summary output, published baseline
  comparison, and CLI wiring
- [x] Implement `scripts/experiment_226_pbt_humaneval_full.py`
- [x] Run targeted coverage, the full Python suite, spec coverage, and the
  applicable integration/E2E checks
- [x] Run the live 164-problem benchmark and record the honest artifact
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
