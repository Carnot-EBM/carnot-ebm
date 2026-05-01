# KAN Distillation Capability Specification

**Capability:** kan
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-KAN-001, FR-KAN-002, FR-KAN-003, FR-KAN-004

## Overview

Defines how Carnot distills a large teacher safety model (gpt-oss-safeguard-20b)
into a compact KAN (Kolmogorov-Arnold Network) classifier for real-time prompt
injection detection.  KAN spline activations are interpretable: each control
point maps directly to a named injection pattern feature, so a human auditor
can read the spline to understand why a prompt was flagged.

The distillation pipeline trains the KAN on teacher-labeled examples, measures
AUROC on a held-out split, and gates deployment on reaching a minimum quality
threshold (REQ-KAN-003/004).

## Requirements

### REQ-KAN-003: Distillation Training Dataset >= 3000 Examples

The KAN distillation training dataset MUST contain at least 3000 labeled
examples with a balanced positive/negative ratio (1500 injection + 1500 benign).

**Rationale:**
    Exp 710 trained on 1091 examples and achieved AUROC=0.8747, just below the
    0.90 Tier 0b deployment gate.  Variance analysis shows the gap is driven by
    dataset size: with fewer than 1500 examples per class the contrastive loss
    surface is noisy and the optimizer settles in a shallow basin.  3000 balanced
    examples provide enough signal for the 16-knot splines to fit the teacher's
    classification boundary.

**Acceptance criteria:**
    - `len(dataset) >= 3000`
    - `count(label="injection") == count(label="benign")` (within ±1)
    - Dataset persisted at `results/kan_distill_v3_dataset.json` with schema field.

### SCENARIO-KAN-003: Dataset balance check

Given a generated dataset,
When the loader reads `results/kan_distill_v3_dataset.json`,
Then `len(examples) >= 3000` AND `abs(n_positive - n_negative) <= 1`.

### REQ-KAN-004: KAN Architecture >= 16 Knots Per Activation

The KAN distillation model used for Tier 0b quality gating MUST use at least
16 spline knots per activation.

**Rationale:**
    8 knots (v2) could not capture the fine decision boundary near the teacher's
    0.90 AUROC threshold.  16 knots double the spline resolution, allowing the
    energy landscape to model sharper transitions between benign and injection
    regions.  The interpretability property is preserved: each knot corresponds
    to a piecewise-linear segment of the activation, readable as a breakpoint
    in the injection-feature sensitivity curve.

    Parameter count with 16 knots, n_hidden=8, n_features=32, degree=3:
        edge_params  = 8 * 32 * (16+3) = 8 * 32 * 19 = 4864
        output_params = 8 * 19          = 152
        total         = 5016

**Acceptance criteria:**
    - `PromptInjectionEnergyCheckerV3._N_KNOTS == 16`
    - `checker.n_params() == 5016` with default constructor arguments.
    - Training for 100 epochs on 3000 examples completes without error.

### SCENARIO-KAN-004: 16-knot architecture sanity check

Given `PromptInjectionEnergyCheckerV3()`,
When we call `checker.n_params()`,
Then the result is 5016 (= 8 * 32 * 19 + 8 * 19).

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-KAN-003 | Proposed | Exp 724 target: 3000-example balanced dataset |
| REQ-KAN-004 | Proposed | Exp 724 target: 16 knots/spline in v3 KAN |
| REQ-KAN-020 | Implemented | Exp 866: LUT analysis complete. N=8 MLP-bound=14400 > 7680 (over budget). ISING_PRIORITY. |
| REQ-MODEL-SOS-001 | Implemented | Exp 1047: SOSKANEnergy confirmed 0 violations on 16000 test points. |

## REQ-KAN-020: KAEMEnergy FPGA LUT Budget Analyzability

KAEMEnergy MUST be analyzable for FPGA LUT budget using arXiv 2604.03345
per-knot estimates. Target: N=8, within iCE40 HX8K 7680 LUT budget.

**Rationale:**
    The iCE40 HX8K has 7,680 LUTs.  Before committing synthesis effort, a
    conservative LUT estimate is required.  arXiv 2604.03345
    (Hardware-Oriented KAN Inference Complexity) provides the formula:
        LUTs = fan_in * fan_out * n_knots * luts_per_segment
    where luts_per_segment = 8–12 (we use 10 as conservative midpoint).

    Exp 866 result (2026-04-25):
      - KAN N=8 MLP upper bound: 14,400 LUTs (over 7,680 budget)
      - Ising N=8 actual (Exp 859): 134 LUTs
      - Synthesis priority: ISING_PRIORITY
      - Note: actual KAEMEnergy is sparse graph-based (edge_density=0.1),
        so the real LUT cost is ~10% of the MLP estimate (~1,440 LUTs),
        which would fit within budget.  A sparse-aware analysis is needed
        to confirm feasibility definitively.

**Acceptance criteria:**
    - `KANHardwareAnalyzer` in `python/carnot/analysis/kan_hw_analysis.py`
      implements `lut_estimate_layer()`, `total_lut_estimate()`,
      `synthesis_priority()`, and `sensitivity_analysis()`.
    - `total_lut_estimate()` returns dict with keys:
      layer1_luts, layer2_luts, total_luts, ice40_hx8k_budget, within_budget.
    - `synthesis_priority()` correctly returns one of:
      KAN_PRIORITY / ISING_PRIORITY / BOTH_FEASIBLE.

### SCENARIO-KAN-030: LUT estimate for N=8 KAEMEnergy on iCE40 HX8K

Given KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10),
When total_lut_estimate() is called,
Then layer1_luts=12800, layer2_luts=1600, total_luts=14400,
     within_budget=False (14400 > 7680).

## REQ-MODEL-SOS-001: SOSKANEnergy MUST guarantee ψ'(x) >= 0 as a type-level property

SOSKANEnergy MUST guarantee ψ'(x) >= 0 (monotonicity) as a type-level property,
with zero violations possible regardless of parameter values V and c.

**Rationale:**
    KAEMEnergy's post-hoc isotonic projection (enforce_monotonicity) fixes violations
    AFTER they occur during training. This is the wrong framing: violations can
    accumulate within an epoch, the MILP verifier must re-run after each epoch, and
    the projection changes the energy landscape in a non-gradient direction.

    SOSKANEnergy uses the SOS (Sum-of-Squares) parameterization:
        ψ'(x) = ||V^T B(x)||² = B(x)^T (V V^T) B(x) >= 0
    for any unconstrained V. No projection, no constraint, no post-hoc repair.
    The verifier can be run ONCE and the invariant holds forever.

    Exp 1047 confirmed: 0 monotonicity violations on 16,000 random test points
    (1,000 × 16 features) with adversarially large random V matrices.

**Acceptance criteria:**
    - `SOSKANEnergy(n_sos_basis=2, ...)` instantiates without error.
    - `verify_invariants(n_samples=1000)` returns `n_monotone_violations == 0`
      for any V (including V set to large random values post-construction).
    - `forward(x) >= 0` for any x in [-1, 1]^n_features.
    - AUROC >= 0.50 on FoVer corpus (no regression from KAEMEnergy baseline).
    - Training is 5x faster than KAEMEnergy (no projection overhead).

### SCENARIO-MODEL-SOS-001: zero violations on adversarial V

Given SOSKANEnergy with V overwritten to random N(0, 10) values,
When verify_invariants(n_samples=1000) is called,
Then n_monotone_violations == 0 and invariants_hold == True.
