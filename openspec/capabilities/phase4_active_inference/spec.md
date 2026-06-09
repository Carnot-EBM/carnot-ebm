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

### REQ-PHASE4-CANONICAL-DECISION
**Description:** The system SHALL produce a canonical decision artifact documenting the transition from alpha_t metrics to the Fast-Slow Variant empirical metric.

### SCENARIO-DECISION-ARTIFACT-GENERATION
**Description:** Running the decision script outputs a valid carnot.phase4_canonical_decision.v2 JSON artifact confirming the retirement of thermodynamic metrics.

### REQ-PHASE4-006: ARC-AGI-3 Synthetic Harness Scaffold
**Description:** The system SHALL provide an importable ARC-AGI-3 harness scaffold that runs without GPU or live model dependencies, checks `import carnot.verify` before declaring readiness, exposes a tiny synthetic grid environment, encodes observations into verifier inputs, uses the cheap energy verifier as a verifier-as-router score for candidate actions, prunes low-scoring actions, escalates only when all candidates score poorly, and records that the scaffold is synthetic rather than a real ARC-AGI-3 benchmark result.

### SCENARIO-PHASE4-006
**Description:** Running the scaffold on the tiny synthetic grid task selects the verifier-preferred action, prunes at least one low-scoring candidate, solves the task, writes the Exp 3919 readiness artifact with bare scalar readiness fields, and reports no ARC-AGI-3 benchmark performance claim.

### REQ-PHASE4-007: ARC-AGI-3 Synthetic Action-Efficiency Measurement
**Description:** The system SHALL measure verifier-as-pruner action efficiency on a richer synthetic ARC-AGI-3-style environment with deterministic transitions, a larger discrete action space, a known goal, and at least 30 episodes, comparing cheap energy-verifier action selection against a no-verifier random/greedy baseline without claiming real ARC-AGI-3 benchmark performance.

### SCENARIO-PHASE4-007
**Description:** Running Exp 3929 records mean actions-to-solve for verifier and baseline arms, bootstrap CI for the baseline/verifier action-efficiency ratio, solve rates for both arms, methodology fields, real ARC-AGI-3 access preflight reachability, and an honest verdict that reports either verifier-router help only when the CI lower bound is above 1.0 or a no-advantage synthetic finding otherwise.

### REQ-PHASE4-008: Active Data Codex Nonspatial Sweep
**Description:** The system SHALL test whether the "active data -> codex program synthesis -> consistency-energy verification" pipeline generalizes across the 6 non-spatial games. It MUST collect active transitions, run codex program synthesis with the consistency-energy refactor-from-best loop (bounded to <=3 iterations), and grade the best program on a common held-out test. It MUST report a per-game table including best held-out consistency energy, whether it is trustworthy (<=0.15), and the vc33 baseline diff.

### SCENARIO-PHASE4-008
**Description:** Running Exp 3947 produces a valid artifact with fields `n_trustworthy_at_0.15`, `per_game_best_energy`, `total_codex_calls`, `total_codex_seconds`, and an honest verdict.
