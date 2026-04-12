# Epic: VERIFY-014 - Typed Reasoning IR for Verifier-Friendly Extraction

**Status:** Completed
**Goal:** Add a typed reasoning intermediate representation at
`python/carnot/pipeline/typed_reasoning.py` so Carnot can carry prompt
constraints, reasoning steps, atomic claims, final answers, and extraction
provenance in a deterministic verifier-friendly form between Exp 211's
benchmark contract and Exp 213's monitorability policy.
**Rationale:** Exp 211 defined what Carnot needs to extract, and Exp 213
showed that different response modes expose different amounts of usable state.
Carnot now needs a stable representation that can accept direct structured
JSON when the model emits it but still salvage plain-text reasoning when it
does not.

## Stories
- [x] Add `REQ-VERIFY-015`, `REQ-VERIFY-016`, `REQ-VERIFY-017`,
  `SCENARIO-VERIFY-015`, `SCENARIO-VERIFY-016`, and
  `SCENARIO-VERIFY-017` to the verifiable-reasoning spec
- [x] Write tests first for direct-JSON parsing, fallback parsing,
  validation/serialization, and backward-compatible pipeline integration
- [x] Implement `python/carnot/pipeline/typed_reasoning.py`
- [x] Wire the typed reasoning IR into `VerifyRepairPipeline` without
  changing existing extractor behavior
- [x] Run unit tests, the full Python suite, targeted 100% coverage on the
  new module, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
