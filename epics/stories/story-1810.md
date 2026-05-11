# Story 1810: Capstone E2E Pipeline with Gemma4-26B

## Context

Phase 16 capstone pipeline execution requires evaluating the MoE model using `unsloth/gemma-4-26B-A4B-it-GGUF`.

## Spec Traces

- REQ-E2E-1810
- SCENARIO-E2E-1810

## Acceptance Criteria

- [ ] Script `scripts/experiment_1810_capstone_gemma26.py` is implemented.
- [ ] Outputs JSON artifact to `results/experiment_1810_gemma26.json`.
- [ ] Tracks `accuracy` and `energy` metrics.
- [ ] 100% test coverage.

## Status

In Progress
