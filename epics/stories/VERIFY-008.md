# Epic: VERIFY-008 - Live Extraction Autopsy for GSM8K

**Status:** Completed
**Goal:** Capture full live Gemma4-E4B-it GSM8K responses, show exactly what
the regex ArithmeticExtractor did or did not match, and preserve a structured
autopsy for future extractor redesign work.
**Rationale:** Exp 203 is the first credibility check after the simulation
collapse. We need a ground-truth artifact that explains why wrong live answers
produce zero arithmetic violations.

## Stories
- [x] Add `REQ-VERIFY-008` and `SCENARIO-VERIFY-008` to the verifiable-reasoning spec
- [x] Write tests for extraction-autopsy categorization, extractor match capture, and JSON serialization
- [x] Implement the Exp 203 helper logic and live GPU script
- [x] Run unit tests, full Python test suite, spec coverage, and the applicable E2E checks
- [x] Run the live Gemma4-E4B-it autopsy on 20 GSM8K questions
- [x] Save `results/experiment_203_results.json`
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, and `ops/changelog.md`
