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
