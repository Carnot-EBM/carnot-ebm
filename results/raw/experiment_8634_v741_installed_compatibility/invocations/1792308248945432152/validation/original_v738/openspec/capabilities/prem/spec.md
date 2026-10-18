# Process-Reward Energy Model (PREM)

## Overview
The Process-Reward Energy Model (PREM) base architecture provides step-wise energy evaluation by subclassing standard EBMs. It is designed to evaluate intermediate steps in a reasoning or generation process, rather than just the final state.

## Requirements

- **REQ-PREM-001**: `PREMConfig` must define the configuration for the model, extending or wrapping standard EBM configurations (e.g. `GibbsConfig`).
- **REQ-PREM-002**: `PREMModel` must subclass standard EBMs and provide a `step_energy` method for step-wise evaluation.
- **REQ-PREM-003**: The model must support sequence-level energy calculation by aggregating step-wise energies.

## Scenarios

- **SCENARIO-PREM-001**: A sequence of intermediate steps is provided. The model computes the energy at each step using `step_energy` and returns the sequence of energies.
- **SCENARIO-PREM-002**: A complete trajectory is evaluated to compute the total process reward energy.
- **SCENARIO-PREM-003**: Given a low PREM energy variance, `TTCController` returns a baseline budget.
- **SCENARIO-PREM-004**: Given a high PREM energy variance, `TTCController` returns an expanded budget to increase test-time compute.

## Dynamic Budget Controller (TTC)

- **REQ-PREM-004**: The system shall provide a `TTCController` that monitors PREM energy variance and dynamically expands search budgets (e.g. beam width, MCMC steps).
- **REQ-PREM-005**: `TTCController.get_budget(energy_history)` must compute variance of recent energies and scale the base budget based on the variance.
