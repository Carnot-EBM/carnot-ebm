# Epic: VERIFY-033 - Claim-Isolated Semantic Verifier V2

**Status:** Completed 2026-04-13
**Goal:** Turn the Exp 232 calibration corpus and Exp 233 output policy into a
claim-isolated semantic verifier with calibrated confidence and an explicit
abstain path for weak-evidence semantic cases.
**Rationale:** Exp 215 proved Carnot can catch real semantic failures, but the
first-layer verifier was still too coarse and too eager. The next step is a
claim-level verifier that uses Exp 232's calibration evidence and Exp 233's
monitorability policy to fail only high-confidence semantic mismatches while
leaving ambiguous cases inspectable without automatically spending false-
positive budget.

## Stories
- [x] Add `REQ-VERIFY-046`, `REQ-VERIFY-047`, `SCENARIO-VERIFY-047`,
  `SCENARIO-VERIFY-048`, and `SCENARIO-VERIFY-049` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for calibration thresholds, abstain behavior,
  deterministic serialization, and additive `VerifyRepairPipeline`
  integration
- [x] Implement `python/carnot/pipeline/semantic_verifier_v2.py` so it reuses
  typed reasoning and semantic grounding, scores claim-level target coverage
  plus premise support, and emits `supported` / `violated` / `abstain`
  verdicts with calibrated confidence
- [x] Integrate the new verifier into `VerifyRepairPipeline` behind a
  dedicated entry point and structured result field without breaking existing
  callers that ignore the new data
- [x] Run the required targeted and full Python suite checks, achieve 100%
  targeted coverage for the new module, run spec coverage plus the applicable
  E2E/integration checks, and reconcile `_bmad/traceability.md`,
  `ops/status.md`, `ops/changelog.md`, and `ops/metrics.md`
