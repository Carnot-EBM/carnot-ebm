# Safety Capability Specification

**Capability:** safety
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-SAFE-001, FR-SAFE-002, FR-SAFE-003

## Overview

Defines how Carnot detects safety violations and compliance infractions using
Energy-Based Models. Safety constraints are STRUCTURAL: "this response gives
specific investment advice" is a structural claim about language patterns, not
arithmetic. KAN spline activations learn to assign high energy to responses with
violating patterns the same way arithmetic EBMs assign high energy to invalid
computations.

The compliance energy is auditable: because spline weights map directly to
keyword co-occurrence features, a human auditor can read the spline control
points to understand why the model flagged a response. This is the core
advantage over black-box classifiers for regulated industries.

## Requirements

### REQ-SAFE-004: ComplianceEnergyChecker — Low Energy = Compliant

The system shall provide a `ComplianceEnergyChecker` that assigns scalar energy
to text inputs such that:
- **Compliant text** (properly hedged, advisory language) receives LOW energy.
- **Violation text** (imperative advice, specific treatment recommendations,
  binding legal commitments) receives HIGH energy.

The checker uses KAN spline activations over bag-of-words domain keyword
features. Training is supervised: compliant examples are pushed to low energy,
violation examples are pushed to high energy via contrastive loss.

**Acceptance criteria:**
- `checker.energy(text) -> float` returns a scalar.
- `checker.is_compliant(text, threshold) -> bool` returns True when energy
  is below the threshold.
- After training on labeled examples, compliant texts reliably have lower energy
  than violation texts.
- AUC-ROC > 0.50 after training on balanced labeled data.

### REQ-SAFE-005: Multi-Domain Compliance Support

The system shall support at least three regulated-industry compliance domains:
- **financial**: Detects unauthorized investment advice (imperative buy/sell
  recommendations, guaranteed returns, specific profit promises).
- **medical**: Detects unauthorized treatment recommendations (specific dosing
  instructions, diagnose/cure claims).
- **legal**: Detects binding legal commitments made without authority (contract
  guarantees, liability waivers, indemnification claims).
- **general**: Union of all domain keywords; for cross-domain screening.

Each domain has a distinct keyword vocabulary. The `encode_compliance_text()`
function maps text to a fixed-size [0,1] feature vector by counting domain
keyword occurrences normalized by text length.

**Acceptance criteria:**
- `ComplianceDomain` type alias restricts to the four valid values.
- `encode_compliance_text(text, domain, max_features)` returns a JAX array of
  shape `(max_features,)` with values in [0, 1].
- Features are deterministic: same text + domain always yields same vector.

### REQ-SAFE-006: KAN Spline Inspection for Compliance Auditing

The system shall expose the learned spline control points so auditors can
determine WHY the compliance energy is high for a given text.

Because the input features are human-interpretable keyword counts (e.g.,
feature 0 = count of "buy" in financial domain), and the KAN spline maps each
feature through a learnable 1D function, the control points directly reveal
the learned relationship between each keyword and the energy output. A positive
slope in the spline for "guarantee" means the model learned that "guarantee"
raises energy — exactly the auditable explanation a regulated-industry user
needs.

**Acceptance criteria:**
- `checker.inspect_spline(hidden_unit, feature_idx) -> np.ndarray` returns
  the control point array for spline (hidden_unit, feature_idx).
- The return value shape is `(n_knots + degree,)` (same as BSpline control
  points in kan.py).
- Positive control points indicate the feature raises energy for that hidden
  unit; negative indicates it lowers energy.

## Scenarios

### SCENARIO-SAFE-004: Financial Compliance Classification

**Given** a ComplianceEnergyChecker trained on financial domain examples where
violations say "You should buy XYZ stock now, it will guarantee 20% returns"
and compliant examples say "XYZ stock has historically returned 8% annually;
past performance does not guarantee future results."

**When** the checker evaluates both texts,

**Then** the violation text energy is higher than the compliant text energy after
training.

### SCENARIO-SAFE-005: Medical Compliance Cross-Domain Isolation

**Given** a ComplianceEnergyChecker trained on the medical domain,

**When** the checker evaluates "take 500mg aspirin twice daily to cure your
headache" (violation) and "aspirin may help with headaches; consult your
doctor for proper dosage" (compliant),

**Then** the violation receives higher energy, and training on medical data
does not require financial or legal keywords.

### SCENARIO-SAFE-006: Spline Inspection Reveals Keyword Contribution

**Given** a trained ComplianceEnergyChecker on the financial domain,

**When** `inspect_spline(hidden_unit=0, feature_idx=0)` is called (feature 0
= "buy" keyword count),

**Then** the returned control point array has shape `(n_knots + degree,)` and
the auditor can inspect whether the "buy" keyword contributes positively or
negatively to energy at that hidden unit.

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-SAFE-004 | Implemented | ComplianceEnergyChecker in compliance_checker.py |
| REQ-SAFE-005 | Implemented | financial, medical, legal, general domains |
| REQ-SAFE-006 | Implemented | inspect_spline() exposes control points |
