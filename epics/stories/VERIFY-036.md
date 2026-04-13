# Epic: VERIFY-036 - Spec-Aware Code Verification And Trace-Ranked Repair Hints

**Status:** Completed 2026-04-13
**Goal:** Add an opt-in spec-aware generated-code verifier that combines
official tests, Hypothesis-backed PBT, and the checked-in explicit code-spec
corpus while ranking repair hints with the existing trace-learning signals.
**Rationale:** PBT is currently Carnot's strongest live code-verification path,
but the accepted repairs still skew toward syntax-heavy fixes and the verifier
still relies partly on prompt heuristics. The checked-in Exp 236 code-spec
corpus plus the Exp 225 / 226 / 227 trace-learning path should make the
verifier more explicit and the repair guidance more selective without breaking
the current packaged path.

## Stories
- [x] Add `REQ-CODE-025` through `REQ-CODE-028` and
  `SCENARIO-CODE-022` through `SCENARIO-CODE-025` to the
  `code-verification` spec before implementation changes
- [x] Write tests first for corpus ingestion, aggregated verification,
  trace-ranked repair guidance, and the additive pipeline opt-in path
- [x] Implement `python/carnot/pipeline/spec_code_verifier.py` and the
  additive integration without touching `scripts/research_conductor.py`
- [x] Run the required targeted command, targeted 100% coverage for the new
  code, the full Python suite, spec coverage, lint/type checks, and the
  applicable code-verification E2E validation
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  `ops/metrics.md`, and any affected E2E result notes
