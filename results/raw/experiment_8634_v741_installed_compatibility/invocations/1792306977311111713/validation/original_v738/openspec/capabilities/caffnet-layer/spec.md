# CAffNet Layer

**Capability:** caffnet-layer
**Status:** In Progress

## Requirements

### REQ-CAFFNET-3385: Differentiable Affine Constraint Layer
Carnot MUST provide a differentiable constraint layer that guarantees affine constraints for Tier-1 EBM filtering in `python/carnot/models/caffnet_layer.py`.

Sub-requirements:
- REQ-CAFFNET-3385-1: The layer SHALL project unconstrained logits onto a constrained affine subspace $Ax = b$ and/or $Ax \le b$.
- REQ-CAFFNET-3385-2: The projection SHALL be fully JAX-differentiable.
- REQ-CAFFNET-3385-3: The layer SHALL assert 100% hard constraint satisfaction at inference time.

### REQ-CAFFNET-3398: Robustness to Out-of-Distribution Constraint Configurations
The CAffNet layer MUST maintain strict affine constraint satisfaction ($Ax = b$) or handle gracefully when provided with adversarial or out-of-distribution (OOD) constraint configurations.

Sub-requirements:
- REQ-CAFFNET-3398-1: The layer SHALL project inputs securely under extreme condition numbers or scaling of the constraint matrices $A$ and $b$.
- REQ-CAFFNET-3398-2: The projected output SHALL consistently satisfy $Ax = b$ within numerical tolerance, even for OOD inputs.

## Scenarios

### SCENARIO-CAFFNET-3398: Robustness against adversarial constraints
**Given** a set of adversarial OOD constraint configurations (e.g., poorly conditioned matrices, extreme magnitudes)
**When** the constraints are fed into the CAffNet layer
**Then** the projection maintains numerical stability and the output satisfies the affine constraints.

### SCENARIO-CAFFNET-3385: Exact affine constraint satisfaction
**Given** a set of unconstrained logits and linear equality/inequality constraints
**When** the CAffNet layer is applied
**Then** the output perfectly satisfies the affine constraints while gradients propagate correctly through the projection.
