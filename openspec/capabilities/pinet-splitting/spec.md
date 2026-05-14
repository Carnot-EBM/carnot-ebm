# PiNet Douglas-Rachford Splitting Capability Spec

## Overview
This capability implements PiNet's Douglas-Rachford operator splitting for convex constraint projection in JAX.

## Requirements

### REQ-PINET-001 - Douglas-Rachford Forward Pass
The system shall provide a Douglas-Rachford splitting loop that enforces linear equality constraints in the forward pass.

### REQ-PINET-002 - PySAT Validation
The PiNet continuous splitting solution MUST align with exact Boolean solutions provided by PySAT for simple SAT problems represented as linear constraints.

### REQ-HARDNET-001 - Nonlinear Projection
The system shall provide a HardNet++ damped local linearization projection loop for nonlinear inequalities.

## Scenarios

### SCENARIO-PINET-001 - Exact one constraint
Given a constraint x1 + x2 = 1 and x2 + x3 = 1 and x1 = 1, the splitting loop finds x=(1, 0, 1), which matches the PySAT result.

### SCENARIO-HARDNET-001 - Nonlinear inequality projection
Given random nonlinear constraint configurations, the projection correctly satisfies the constraints with damped step updates.

## Implementation Status
| REQ | Status | Experiment |
|-----|--------|------------|
| REQ-PINET-001 | IMPLEMENTED | python/carnot/solvers/pinet_prototype.py |
| REQ-PINET-002 | IMPLEMENTED | python/carnot/solvers/pinet_prototype.py |
| REQ-HARDNET-001 | IMPLEMENTED | python/carnot/solvers/hardnet_projection.py |