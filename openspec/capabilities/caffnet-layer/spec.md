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

## Scenarios

### SCENARIO-CAFFNET-3385: Exact affine constraint satisfaction
**Given** a set of unconstrained logits and linear equality/inequality constraints
**When** the CAffNet layer is applied
**Then** the output perfectly satisfies the affine constraints while gradients propagate correctly through the projection.
