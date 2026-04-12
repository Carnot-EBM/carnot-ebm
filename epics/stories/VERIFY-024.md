# Epic: VERIFY-024 - Hypothesis-Backed PBT Verifier For Generated Code

**Status:** Completed
**Goal:** Add `python/carnot/pipeline/pbt_code_verifier.py` and wire it into
`VerifyRepairPipeline` as an additive generated-code verification path that
uses Hypothesis-backed property checks for HumanEval-style prompts.
**Rationale:** Exp 208 showed that execution-only verification has live signal,
and Exp 217 added lightweight prompt-derived invariants. The next increment is
to replace purely fixed probes with real property-based generation so Carnot can
search for counterexamples on edge cases and prompt-implied invariants that the
official tests miss.

## Stories
- [x] Add `REQ-CODE-009` through `REQ-CODE-011` and
  `SCENARIO-CODE-008` through `SCENARIO-CODE-010` to the
  `code-verification` spec before implementation changes
- [x] Write tests first for Hypothesis-backed property derivation,
  structured `ConstraintResult` failures, `VerifyRepairPipeline`
  integration, and a deterministic five-problem HumanEval-style
  execution-vs-PBT comparison
- [x] Implement `python/carnot/pipeline/pbt_code_verifier.py`
- [x] Wire the new verifier into `VerifyRepairPipeline` as an additive
  generated-code verification path without breaking existing `verify()`
  callers
- [x] Run targeted coverage, the full Python suite, spec coverage, and
  the applicable integration/E2E checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
