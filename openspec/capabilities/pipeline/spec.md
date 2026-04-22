# Pipeline Capability Specification

**Capability:** pipeline
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-08

## Overview

Defines the multi-tier cascade pipeline that routes LLM-generated reasoning steps through
progressively more expensive verifiers (Tier 1: fast energy gate, Tier 2: JEPA ranking,
Tier 3: full formal verification).  The JEPA predictor tier must be loaded from a
version-tagged checkpoint to prevent silent rollbacks to sub-threshold models.

## Requirements

### REQ-INFRA-043: JEPA Cascade Version-Pinned Checkpoint Loading

The Tier 2 JEPA predictor in the cascade MUST load its model via a version-tagged
checkpoint path.  The active version is recorded in the conductor exclusion manifest
under the `jepa_v18_active` flag.  Loading an older version (v15/v16/v17) is
prohibited while the manifest marks them excluded; attempting to load an excluded
version MUST raise `ValueError`.

**Rationale:** JEPA v15/v16/v17 all produced OOD AUC below random chance (0.47–0.48),
meaning the cascade would actively harm verification quality.  The manifest is the
authoritative gate that prevents accidental rollback to a below-threshold model.

**Acceptance criteria:**
- `tier2_jepa.load_v18_from_manifest()` returns a `JEPALambdaRankV18` instance.
- Passing `version="v17"` raises `ValueError` with a message naming the blocked version.
- The loaded model's `predict_score()` returns a float for any string input.

### REQ-INFRA-044: Cascade AUC Gating

The cascade integration smoke test (Exp 718) MUST achieve cascade_auc >= 0.70 on a
50-question held-out GSM8K validation set before the JEPA v18 deployment is considered
successful.  If cascade_auc < 0.70, the honest_verdict MUST be "cascade_deploy_auc_fail"
and the gate file for Exp 719 MUST record `gate: "fail"`.

### REQ-INFRA-045: Cascade Latency Budget

The per-question latency overhead added by the JEPA v18 Tier 2 scorer MUST be less than
5 ms (latency_delta_ms < 5).  Exceeding this budget causes honest_verdict
"cascade_deploy_latency_fail" regardless of AUC.

## Scenarios

### SCENARIO-INFRA-052: Version-Blocked Model Raises Error

**Given** the exclusion manifest marks `jepa_v17_blocked: true`
**When** code calls `tier2_jepa.load_v18_from_manifest(version="v17")`
**Then** a `ValueError` is raised with message containing "blocked"

**Spec traces:** REQ-INFRA-043

### SCENARIO-INFRA-053: v18 Loads and Scores Successfully

**Given** a trained `JEPALambdaRankV18` instance
**When** `predict_score("Step 1: 3 + 5 = 8.")` is called
**Then** a scalar float is returned without raising any exception

**Spec traces:** REQ-INFRA-043

### SCENARIO-INFRA-054: Cascade AUC on Held-Out Groups

**Given** the cascade is loaded with JEPA v18 as Tier 2
**When** `evaluate_auc(eval_groups)` is called on 50 held-out GSM8K groups
**Then** the returned float is in [0, 1]

**Spec traces:** REQ-INFRA-044

### REQ-INFRA-046: EORM Confidence Gate for Tier 3 Ising Skip

The cascade router MUST support an EORM confidence gate that skips Tier 3 Ising
sampling when the EORM confidence score exceeds a configurable threshold (default 0.92).
When EORM confidence > eorm_ising_skip_threshold, the result is marked "verified_fast"
and Tier 3 Ising is not run.  The threshold is configurable at CascadeRouter
construction time.  Each query MUST log ising_skip (bool) and eorm_confidence (float).

**Acceptance criteria:**
- `CascadeRouter(eorm_ising_skip_threshold=0.92)` skips Ising when EORM confidence > 0.92.
- `CascadeRouter(eorm_ising_skip_threshold=0.92)` runs Ising when EORM confidence <= 0.92.
- Per-query logs contain ising_skip and eorm_confidence fields.

### REQ-INFRA-047: EORM Gate False-Negative Delta

The EORM confidence gate MUST NOT increase the false-negative rate by more than 5
percentage points versus the full cascade (no gate) on a representative test set.
Formally: fn_delta = false_negative_rate_gated - false_negative_rate_baseline < 0.05.

**Acceptance criteria:**
- Exp 727 measures fn_delta < 0.05 on 200-question test set at threshold=0.92.

## Scenarios

### SCENARIO-INFRA-055: EORM Gate Skips Ising Above Threshold

**Given** a CascadeRouter with eorm_ising_skip_threshold=0.92
**When** EORM returns confidence=0.95 for a query
**Then** Tier 3 Ising is NOT invoked and result is marked "verified_fast"

**Spec traces:** REQ-INFRA-046

### SCENARIO-INFRA-056: EORM Gate Does Not Skip Ising Below Threshold

**Given** a CascadeRouter with eorm_ising_skip_threshold=0.92
**When** EORM returns confidence=0.80 for a query
**Then** Tier 3 Ising IS invoked as normal

**Spec traces:** REQ-INFRA-046

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-INFRA-043 | Implemented | Exp 718 — tier2_jepa.py |
| REQ-INFRA-044 | Implemented | Exp 718 smoke test |
| REQ-INFRA-045 | Implemented | Exp 718 latency measurement |
| REQ-INFRA-046 | Implemented | Exp 727 — cascade_router.py |
| REQ-INFRA-047 | Implemented | Exp 727 fn_delta measurement |
