# Phase 4 Active Inference

## Implementation Status
- Phase 4 Active Inference (Simulation): Implemented

## Requirements

### REQ-PHASE4-001: MLD Simulation
**Description:** The system SHALL simulate MLD steps on an Ising substrate, computing `mu_P` and `alpha_t` per step.

### REQ-PHASE4-002: Verifier Ensemble Stability
**Description:** The verifier-as-free-energy ensemble SHALL maintain `inf_t alpha_t > 0.10` with `k=6` verifiers, and collapse to `< 0.05` with `k=1`.

## Scenarios

### SCENARIO-PHASE4-1
**Description:** Running 100 MLD steps with `k=6` and `k=1` on `n=8` substrate yields a `delta_alpha > 0.05`.

### REQ-PHASE4-003: MLD Substrate Scaling
**Description:** The system SHALL simulate MLD steps on scaling Ising substrates (n=8, n=16, n=32) and track `delta_alpha` scaling behavior to identify potential substrate size collapse points.

### SCENARIO-PHASE4-2
**Description:** Running scaling simulation on n=8, 16, 32 yields recorded delta_alpha scaling points, correctly identifying if and where delta_alpha collapses below 0.05.

### REQ-PHASE4-004: Verifier Ablation Audit
**Description:** The system SHALL support running a random-verifier injection ablation audit to test if `delta_alpha` genuinely depends on verifier content.

### SCENARIO-PHASE4-3
**Description:** Running a 4-cell random-verifier ablation audit correctly computes `delta_alpha` and bootstrap CIs for each fraction, and sets the `monotonic_decay_observed` and `artifact_detected` flags based on the results.

### REQ-PHASE4-005: Maximum-Caliber Alpha_t Replacement
**Description:** The system SHALL implement `alpha_t'` derived from the maximum-caliber formulation (prediction error), where `alpha_t'` monotonically decays as random verifiers replace real verifiers in the ensemble.

### SCENARIO-PHASE4-4
**Description:** Running a 4-cell random-verifier ablation audit using the maximum-caliber `alpha_t'` results in monotonic decay of `delta_alpha` (`monotonic_decay_observed=true`) and absence of the bijection-invariance artifact (`artifact_detected=false`).
