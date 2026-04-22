# Publication Readiness Capability Specification

**Capability:** publication
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-PUB

## Overview

Defines the criteria that must be satisfied before any Carnot headline result
can be published, shared publicly, or cited in external communications.

The core invariant: every number in a headline claim must trace to a live-GPU
inference run recorded in a result JSON. Simulated, mocked, or CPU-only results
may appear in the repo as reference points but must be excluded from headline
claims.

## Requirements

### REQ-PUBLISH-001: Live-GPU Provenance for Headline Numbers

All headline VR numbers (signed_improvement, cross_model_delta, grammar_recall,
safety AUROC) MUST have `inference_mode == "live_gpu"` in the source result
file. A result with `inference_mode != "live_gpu"` (e.g. "blocked", "cpu",
"simulated") MUST be excluded from headline claims and marked as a caveat or
negative result in the publication.

Why: The project's credibility rests on the energy function being ground truth.
Allowing simulated numbers into headlines would undermine the anti-hallucination
claim that motivates the project (see _bmad/prd.md Phase 1).

### REQ-PUBLISH-002: Negative Results Section Required

The model card and technical report MUST include a dedicated "Negative Results"
section documenting all significant failures alongside successes. Specifically:

- JEPA v15 OOD AUC = 0.4751 (below random, GSM8K 500-699)
- JEPA v16 OOD AUC = 0.4759 (InfoNCE contrastive loss did not fix root cause)
- Adversarial VR: blocked / not yet measured on live GPU
- Cross-model Gemma-4-E4B-it: signed_improvement = -0.8 (regression)

Why: Publishing only positive results is selection bias. Negative results help
the community understand where constraint-based EBMs succeed and where they
currently fail. They also protect the project's long-term credibility: if
failures are discovered by third parties they should already be documented.

### REQ-PUBLISH-003: Gate Check Before Publication Actions

A publication readiness audit script MUST load all gate result files and compute
a boolean `publication_ready` flag before any publication action is taken. The
script MUST emit an `honest_verdict` field in the result artifact with one of:

- `"publication_ready"` — all provenance valid AND cross-model result exists
- `"publication_ready_with_caveats"` — provenance valid but cross-model blocked
- `"publication_blocked_no_primary_result"` — Exp 679 gate fails

## Scenarios

### SCENARIO-PUBLISH-001: All Headline Numbers Have Live-GPU Provenance

**Given** all source result files have `inference_mode == "live_gpu"`
**When** the provenance audit runs
**Then** every entry in the provenance table has `provenance_valid = True`
 AND `publication_ready = True`

### SCENARIO-PUBLISH-002: Negative Results Documented in Model Card

**Given** JEPA v15 OOD AUC = 0.4751 is in the source result
**When** the model card draft is written
**Then** the model card contains a "Negative Results" section that includes
 the JEPA v15 OOD AUC value AND the adversarial VR blocked status

### SCENARIO-PUBLISH-003: Primary Result Gate Fails

**Given** Exp 679 result file is absent or `vr_200q_validated` is False
**When** the publication readiness script runs
**Then** `honest_verdict == "publication_blocked_no_primary_result"`
 AND `publication_ready == False`

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-PUBLISH-001 | Implemented | Exp 700 provenance audit |
| REQ-PUBLISH-002 | Implemented | Exp 700 model card draft |
| REQ-PUBLISH-003 | Implemented | Exp 700 gate logic |
