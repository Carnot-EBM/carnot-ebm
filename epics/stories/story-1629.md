# Story 1629: Validate EBRM optimizations against local SOTA models

**Epic:** Verifiable Reasoning
**Status:** In Progress
**Owner:** Gemini CLI

## Description
The EBRM trajectory optimization must be validated against mandated local SOTA models. Use `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`. Ensure caching handles the SOTA models properly using the `cached_sota_pair()` pattern. Output to the deliverable JSON `results/experiment_1629_ebrm_sota.json`.

## Requirements
- REQ-EBRM-1629
- SCENARIO-EBRM-1629

## Tasks
- [x] Create spec entries
- [ ] Create tests `tests/python/eval/test_experiment_1629.py`
- [ ] Create implementation `scripts/experiment_1629_ebrm_sota.py`
- [ ] Run tests and ensure 100% coverage
