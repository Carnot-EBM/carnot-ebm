# Epic: VERIFY-009 - SMT-Backed Arithmetic Extraction

**Status:** Completed
**Goal:** Add a Z3-backed arithmetic extractor that can verify explicit
equations, verbal arithmetic steps, chained reasoning, and approximate
calculations without introducing false positives on correct reasoning traces.
**Rationale:** Exp 203 showed that regex-only extraction misses instruction-
tuned reasoning patterns and can emit violations on correct answers. The next
step is a solver-backed extractor that formalizes arithmetic constraints
instead of matching only surface syntax.

## Stories
- [x] Add `REQ-VERIFY-009` and `SCENARIO-VERIFY-009` to the verifiable-reasoning spec
- [x] Write tests for SMT-backed arithmetic extraction, including Exp 203 regression cases
- [x] Implement `python/carnot/pipeline/z3_extractor.py`
- [x] Run unit tests, full Python test suite, spec coverage, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`
