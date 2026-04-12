# Epic: VERIFY-018 - Prompt-Derived Property Verifier For HumanEval Code Paths

**Status:** Completed
**Goal:** Add `python/carnot/pipeline/property_code_verifier.py` and wire it
into the existing execution-based HumanEval path so Carnot can derive extra
properties from the prompt, signature, docstrings/examples, and official
tests.
**Rationale:** Exp 208 showed the code path still has live signal, but the
current verifier mostly relies on `CodeExtractor`, Exp 53 runtime probes, and
the official `check()` harness. The next increment is a lightweight additive
verifier that can synthesize stronger invariants, surface structured repair
feedback, and catch some bugs the official tests alone miss.

## Stories
- [x] Add `REQ-CODE-006`, `REQ-CODE-007`, `REQ-CODE-008`,
  `SCENARIO-CODE-006`, and `SCENARIO-CODE-007` to the
  `code-verification` spec
- [x] Write tests first for prompt/example/test parsing, missed-bug property
  detection, pipeline-compatible repair feedback, and HumanEval integration
- [x] Implement `python/carnot/pipeline/property_code_verifier.py`
- [x] Wire the property verifier into the existing execution-based HumanEval
  path without replacing static/runtime/official-test checks
- [x] Run targeted tests, targeted 100% coverage on the new module, the full
  Python suite, and the applicable integration/E2E checks
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`,
  `ops/changelog.md`, and `ops/metrics.md`
