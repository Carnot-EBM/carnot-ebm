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

### REQ-ISING-022

**Discrete Simulated Bifurcation CPU validation MUST compare FoVer constraint
convergence against a Gibbs Ising baseline and report KV260 BRAM/LUT budget
gates.**

**Rationale:**
Exp 1387 found that a 15-replica 2D parallel-tempering implementation is
LUT-limited on KV260. Discrete simulated bifurcation has a different resource
profile: the dense coupling matrix is stored in BRAM, while LUT consumption is
set mainly by the number of update units instantiated. Carnot needs a CPU-only
probe that keeps convergence evidence, BRAM arithmetic, and KV260 claim gating
explicit before any RTL work.

**Acceptance criteria:**
- The experiment SHALL load between 3 and 5 local FoVer-derived Ising/QUBO
  problems with variable counts spanning 64 through 256 in the default run.
- The dSB update SHALL use a linearly increasing pressure schedule from 0 to 1
  and the sign update `x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) -
  pressure(t))`.
- The experiment SHALL compare dSB convergence steps against a standard Gibbs
  Ising baseline on the same problems and seeds.
- `results/experiment_1399_discrete_sb_kv260_cpu_simulation.json` SHALL include
  `status`, `algorithm`, `constraint_problems_tested`, `n_variables`,
  `steps_to_convergence_ising_baseline`, `steps_to_convergence_discrete_sb`,
  `convergence_speedup_discrete_sb`, `bram_estimate_kb_for_256var`,
  `kv260_bram_budget_kb`, `bram_budget_feasible`,
  `lut_estimate_per_update_unit`, `kv260_lut_budget_fits`,
  `hardware_claim_allowed`, `kv260_claim_allowed`, and `honest_verdict`.
- The KV260 estimate SHALL record a 256 x 256 int8 J matrix as 65536 bytes
  (64 KB) and the KV260 BRAM budget as 144 BRAM_36 blocks = 648 KB.
- `hardware_claim_allowed` and `kv260_claim_allowed` SHALL be true only when
  both the BRAM estimate and the single-update-unit LUT estimate fit the KV260
  budget.

**Implementation status:** Implemented (Exp 1399)

### REQ-ISING-023

**Discrete Simulated Bifurcation KV260 RTL specification MUST document the
datapath, memory layout, update schedule, host interface, and honest synthesis
claim boundary before RTL implementation or board claims.**

**Rationale:**
Exp 1399 found that a dense N=256 int8 dSB coupling matrix and one update unit
fit the KV260 BRAM/LUT arithmetic budget, but it did not produce RTL and did
not run Vivado synthesis or the KV260 board. Exp 1422 must turn that feasibility
evidence into an RTL-reviewable specification and synthesis plan without
upgrading an estimate into a hardware execution claim.

**Acceptance criteria:**
- `hardware/kv260/discrete_sb_rtl_spec.md` SHALL describe the dSB datapath,
  dense int8 coupling memory layout, random/noise source assumptions, pressure
  update schedule, and AXI-Lite host interface.
- The spec SHALL derive the N=256 dense int8 matrix storage from Exp 1399 as
  256 x 256 x 8 bits = 65536 bytes = 64 KB and compare it with the KV260 BRAM
  budget of 144 BRAM_36 blocks = 648 KB.
- The spec SHALL carry forward the Exp 1399 one-update-unit LUT estimate of
  2000 LUTs and compare it with the 117000-LUT KV260 estimate budget.
- The spec SHALL document Vivado synthesis commands for a later
  synthesis-capable host, but the Exp 1422 run SHALL NOT claim synthesis,
  bitfile generation, or KV260 board execution unless those steps are actually
  performed in the current run.
- `results/experiment_1422_discrete_sb_kv260_rtl_spec.json` SHALL include
  `status`, `rtl_spec_complete`, `rtl_spec_path`, `estimated_lut`,
  `estimated_bram`, `kv260_budget_fits`, `synthesis_command_documented`,
  `hardware_execution_performed`, `hardware_claim_allowed`, and
  `honest_verdict`.
- `hardware_execution_performed=false` and `hardware_claim_allowed=false`
  unless the artifact records actual synthesis or KV260 board validation in
  the current run metadata.

**Implementation status:** Implemented (Exp 1422)

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

### SCENARIO-ISING-032

**FoVer dSB CPU artifact:** Given local FoVer rows and no RTL synthesis, Exp 1399
SHALL write the complete CPU convergence artifact, include the 64 KB int8 J
matrix BRAM estimate for N=256, report whether a single update unit fits the
117000-LUT KV260 budget, and gate the hardware/KV260 claim fields from those
two estimates.

**Implementation status:** Implemented (Exp 1399)

### SCENARIO-ISING-033

**Discrete SB RTL spec artifact:** Given Exp 1399's CPU-only feasibility artifact
and no Vivado synthesis, bitfile generation, or KV260 board execution in the
current run, Exp 1422 SHALL write a complete RTL specification plus a JSON
artifact that reports the BRAM/LUT estimates, documents the synthesis command,
sets `hardware_execution_performed=false`, and disallows hardware claims.

**Implementation status:** Implemented (Exp 1422)
