# Epic: VERIFY-016 - Semantic Grounding Verifier for Wrong-Problem Answers

**Status:** Completed
**Goal:** Add `python/carnot/pipeline/semantic_grounding.py` and wire it into
`VerifyRepairPipeline` so Carnot can catch Exp 214-style question-grounding
and omitted-premise failures even when the arithmetic inside the response is
internally consistent.
**Rationale:** Exp 203 / 206 / 207 showed that Carnot's live misses are often
not arithmetic contradictions but answers to the wrong target or incomplete
uses of the prompt. Exp 214 then turned those misses into a deterministic
failure corpus. Exp 215 closes the loop by adding a semantic-grounding layer
that decomposes claims, aligns them to prompt entities and quantities, and
surfaces structured violations suitable for repair.

## Stories
- [x] Add `REQ-VERIFY-020`, `REQ-VERIFY-021`,
  `SCENARIO-VERIFY-020`, and `SCENARIO-VERIFY-021` to the
  verifiable-reasoning spec
- [x] Write tests first for Exp 214-grounded semantic failures, conservative
  false-positive behavior, the optional refinement hook, and pipeline
  integration
- [x] Implement `python/carnot/pipeline/semantic_grounding.py`
- [x] Wire semantic-grounding violations into `VerifyRepairPipeline` without
  replacing the existing extractor path
- [x] Run unit tests, the full Python suite, targeted 100% coverage on the
  new module, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
