# Epic: VERIFY-010 - LLM-Assisted Arithmetic Claim Extraction

**Status:** Completed
**Goal:** Add an auxiliary LLM-backed arithmetic extractor that can convert
free-form reasoning traces into canonical arithmetic claims consumable by the
existing verification pipeline.
**Rationale:** Exp 203 showed that regex extraction misses natural-language
arithmetic in live Gemma4-E4B-it GSM8K traces. An auxiliary small model can
normalize those steps into `CLAIM: a OP b = c` form before Carnot verifies
them.

## Stories
- [x] Add `REQ-VERIFY-010` and `SCENARIO-VERIFY-010` to the verifiable-reasoning spec
- [x] Write tests for LLM-assisted arithmetic extraction, including Exp 203 regression coverage
- [x] Implement `python/carnot/pipeline/llm_extractor.py`
- [x] Benchmark the LLM extractor head-to-head against the shared live Exp 206 Z3 cohort and publish the paired Exp 207 artifact
- [x] Run unit tests, full Python test suite, spec coverage, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`
