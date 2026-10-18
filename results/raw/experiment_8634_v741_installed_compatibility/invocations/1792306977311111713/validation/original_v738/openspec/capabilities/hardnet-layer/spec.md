# HardNet Layer

**Capability:** hardnet-layer
**Status:** Implemented

## Requirements

### REQ-HARDNET-2086: Differentiable Enforcement Layer

Carnot MUST provide a HardNet-style differentiable enforcement layer in `python/carnot/models/hardnet_layer.py`.

Sub-requirements:
- REQ-HARDNET-2086-1: The layer SHALL project continuous latent outputs to feasible logical bounds (inequality constraints).
- REQ-HARDNET-2086-2: The projection SHALL be fully JAX-differentiable.
- REQ-HARDNET-2086-3: The layer SHALL be compatible with Optax optimizers.

## Scenarios

### SCENARIO-HARDNET-2086: Gradient-preserving hard bounds

**Given** a continuous output tensor and defined lower/upper bounds
**When** the HardNet layer is applied
**Then** the output is clamped within the bounds, and gradients propagate correctly through the projection.
