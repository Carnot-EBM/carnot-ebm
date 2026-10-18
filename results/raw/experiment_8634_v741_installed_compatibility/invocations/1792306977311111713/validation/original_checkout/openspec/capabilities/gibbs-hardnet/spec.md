# Gibbs Sampler with HardNet++

**Capability:** gibbs-sampler-hardnet
**Status:** Implemented

## Requirements

### REQ-SAMPLE-9999: Gibbs Sampler with HardNet++

Carnot MUST provide a Gibbs sampler utilizing HardNet++ projection to prune infeasible states in `python/carnot/models/gibbs/sampler.py`.

Sub-requirements:
- REQ-SAMPLE-9999-1: The Gibbs sampler SHALL route through `hardnet.py` to enforce strict boundaries.
- REQ-SAMPLE-9999-2: The final sampled states SHALL never violate the provided strict boundaries.

## Scenarios

### SCENARIO-SAMPLE-9999: Gibbs sampler never violates constraints

**Given** a Gibbs model and a constraint function
**When** sampling is performed via Langevin dynamics with HardNet++ projection
**Then** sampled outputs never violate strict boundaries, projecting infeasible states.
