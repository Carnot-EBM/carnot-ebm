# Epic: VERIFY-041 - Formal Claim Corpus From Live Traces

**Status:** Completed 2026-04-13
**Goal:** Build `data/research/formal_claim_corpus_244.jsonl` plus
`results/experiment_244_results.json` from checked-in live semantic and
prompt-side traces so later solver-routed verification work starts from real
Qwen and Gemma evidence instead of a new synthetic benchmark.
**Rationale:** Exp 235 showed that calibrated semantic scoring alone still
hurts verify-only behavior on both target small models. The next step is not a
new scalar judge but a typed claim-routing substrate, and that substrate needs
a checked-in claim corpus with explicit provenance, localization, and abstain
labels before Exp 245 can implement the solver-routed verifier.

## Stories
- [x] Add `REQ-VERIFY-056`, `REQ-VERIFY-057`, and
  `SCENARIO-VERIFY-063` / `SCENARIO-VERIFY-064` to the
  `verifiable-reasoning` spec before implementation changes
- [x] Write tests first for schema shape, deterministic regeneration,
  provenance coverage, and explicit `abstain` / `not_formalizable` handling
- [x] Implement `scripts/experiment_244_formal_claim_corpus.py` without
  touching `scripts/research_conductor.py`
- [x] Generate `data/research/formal_claim_corpus_244.jsonl` and
  `results/experiment_244_results.json` with fixed run-date metadata
  `20260413`
- [x] Run the required command, targeted 100% coverage for the new script, the
  full Python suite, spec coverage, and the applicable E2E and reconciliation
  checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
