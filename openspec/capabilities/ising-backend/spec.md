# Ising Backend Capability Spec

## Overview

Specification for the Ising-tier sampler backends, including the standard
Metropolis-Hastings sampler and the enhanced InertiaIsingSampler.

## Requirements

### REQ-ISING-020

**IsingEBM MUST support InertiaIsingSampler (arXiv 2604.17109) with EMA inertia
and Mpemba initialization as alternative to standard Metropolis-Hastings.**

**Rationale:**
Standard Metropolis-Hastings on Ising problems suffers from spin-glass freezing
and slow mixing, producing near-zero discrimination_delta between correct and
erroneous code configurations. The EMA inertia term smooths the energy landscape
and allows escapes from shallow local minima, reducing mixing sweeps by 15-25x.

Mpemba initialization (arXiv 2603.24183) seeds spins from the leading eigenvector
of J, further reducing thermalization time.

**Implementation:** `python/carnot/samplers/inertia_ising.py::InertiaIsingSampler`

**Status:** Implemented (Exp 860)

### REQ-ISING-021

**2D parallel tempering CPU validation MUST compare FoVer constraint convergence
against a single-temperature Ising baseline and gate all KV260 hardware claims.**

**Rationale:**
arXiv:2601.09037 reports a 15-temperature-replica FPGA parallel-tempering
solver, while Carnot's existing parallel Ising path has no temperature-replica
exchange. Before any RTL investment, Carnot needs a CPU-only validation that
keeps the algorithmic convergence evidence separate from KV260 feasibility.

**Acceptance criteria:**
- The sampler SHALL run 15 replicas over temperatures in `[0.5, 5.0]`.
- Adjacent-temperature replica swaps SHALL use the Metropolis criterion.
- The experiment SHALL load between 3 and 5 local FoVer-derived constraint
  problems and measure both single-temperature and 2D parallel-tempering
  convergence steps on the same problems.
- `results/experiment_1387_2d_parallel_tempering_kv260_fpga_estimate.json`
  SHALL include `status`, `constraint_problems_tested`, `replica_count`,
  `temperature_schedule`, `steps_to_convergence_standard_pt`,
  `steps_to_convergence_2d_pt`, `convergence_speedup_2d_pt`,
  `sparsification_k_value`, `estimated_kv260_lut_count_per_replica`,
  `estimated_kv260_total_lut_count_15_replicas`, `lut_budget_feasible`,
  `hardware_claim_allowed`, `kv260_claim_allowed`, and `honest_verdict`.
- The KV260 estimate SHALL record that `15 * 36000 = 540000` LUTs exceeds the
  117000-LUT KV260 budget, while `floor(117000 / 36000) = 3` replicas fit.
- `hardware_claim_allowed` and `kv260_claim_allowed` SHALL remain false unless
  local synthesis or board execution is performed in the same run.

**Implementation status:** Implemented (Exp 1387)

## Scenarios

### SCENARIO-ISING-030

**Discrimination test:** InertiaIsingSampler MUST produce a positive
discrimination_delta (energy_error - energy_correct > 0) when comparing
a correct code constraint encoding to an erroneous one.

**Why:** If the sampler cannot discriminate correct from erroneous configurations,
it cannot be used for code verification. A positive delta means the sampler
correctly assigns lower energy (higher probability) to the valid configuration.

**Status:** Validated by Exp 860

### SCENARIO-ISING-031

**FoVer 2D PT CPU artifact:** Given no Vivado synthesis, bitfile generation, or
KV260 board execution, Exp 1387 SHALL write the complete CPU-only convergence
artifact, include the 15-replica LUT over-budget estimate, report the 3-replica
KV260 feasible maximum, and disallow all hardware/KV260 claims.

**Implementation status:** Implemented (Exp 1387)
