# Epic: VERIFY-035 - Exp 236 Explicit Code Spec Corpus

**Status:** Completed 2026-04-13
**Goal:** Build a deterministic explicit code-spec corpus from the checked-in
Exp 226 and Exp 227 HumanEval benchmark traces, write the checked-in
`data/research/code_spec_corpus_236.jsonl` artifact plus a compact summary, and
keep provenance back to the source traces explicit.
**Rationale:** Exp 226 showed that additive PBT can help on full HumanEval, and
Exp 227 showed the same verifier stack does not transfer automatically to Qwen.
The next bridge is to make prompt intent explicit in a verifier-friendly schema
that is grounded in real failure and repair traces rather than prompt reading
alone.

## Stories
- [x] Add `REQ-CODE-023`, `REQ-CODE-024`, `SCENARIO-CODE-020`, and
  `SCENARIO-CODE-021` to the `code-verification` spec before implementation
  changes
- [x] Write tests first for schema shape, deterministic generation, and
  provenance links back to Exp 226 / Exp 227
- [x] Implement the code-spec corpus generator module and the checked-in
  `scripts/experiment_236_code_spec_corpus.py` workflow without touching
  `scripts/research_conductor.py`
- [x] Generate `data/research/code_spec_corpus_236.jsonl` plus the summary
  artifact with fixed run-date metadata `20260413`
- [x] Run targeted 100% coverage for the new code plus the required suite,
  spec-coverage, and the applicable workflow-level E2E validation
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
