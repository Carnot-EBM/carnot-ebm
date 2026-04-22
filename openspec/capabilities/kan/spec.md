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
