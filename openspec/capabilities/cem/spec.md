# Compositional Energy Minimization (CEM)

## Context
Complex multi-constraint optimization degrades. We need Compositional Energy Minimization (CEM) to decompose and aggregate energy landscapes. This capability implements a logic decomposition engine that splits a monolithic constraint graph into localized energy landscapes.

## Requirements
- REQ-CEM-001: The CEM decomposition engine shall accept a constraint graph and split it into multiple subsets (localized energy landscapes).
- REQ-CEM-002: The decomposed constraint graphs shall be validated against existing CCTU benchmark cases.
- REQ-CEM-003: The decomposition artifact shall include a `schema` field with value `carnot.cem.decomposition.v1` and a `num_subsets` integer field.
- REQ-CEM-004: The CompositionalEnergyMinimizer shall sum multiple independent IsingEBM instances.

## Scenarios
- SCENARIO-CEM-001: Decompose a multi-constraint CCTU trace into localized landscapes and report the number of generated subsets.
- SCENARIO-CEM-002: Test CompositionalEnergyMinimizer on a 3-SAT toy problem (n=16).
