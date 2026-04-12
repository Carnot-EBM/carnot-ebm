# Epic: VERIFY-015 - Semantic Failure Corpus for Verifier Training

**Status:** Completed
**Goal:** Publish a deterministic Exp 214 corpus at
`data/research/semantic_failure_corpus_214.jsonl` plus a summary artifact at
`results/experiment_214_results.json` so Carnot has a labeled failure set for
semantic/question-grounding verifier work rather than guessing at why live
answers failed.
**Rationale:** Exp 203 / 206 / 207 showed that live GSM8K misses are usually
semantic or grounding errors, while Exp 208 exposed the need to keep
code-specific verifier signals explicit as well. Exp 214 turns those lessons
into a unit-test-friendly corpus with diagnoses, expected verifier signals,
and structured-reasoning guidance.

## Stories
- [x] Add `REQ-VERIFY-018`, `REQ-VERIFY-019`,
  `SCENARIO-VERIFY-018`, and `SCENARIO-VERIFY-019` to the
  verifiable-reasoning spec
- [x] Write tests first for corpus construction, summary generation, and
  idempotent artifact writing
- [x] Implement `scripts/experiment_214_semantic_failure_corpus.py`
- [x] Generate `data/research/semantic_failure_corpus_214.jsonl` and
  `results/experiment_214_results.json`
- [x] Run unit tests, the full Python suite, targeted 100% coverage on the
  new script, and the applicable E2E/integration checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`,
  and `ops/metrics.md`
