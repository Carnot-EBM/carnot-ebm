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

### REQ-ISING-024

**Discrete Simulated Bifurcation KV260 RTL lint/simulation attempt MUST record
actual source/tool availability and preserve hardware claim gating.**

**Rationale:**
Exp 1422 produced a reviewable RTL specification and synthesis plan for the
Discrete SB KV260 path, but it did not produce a Verilog source, synthesis
report, bitfile, simulation transcript, or KV260 board execution evidence.  Exp
1437 must take the next bounded local step by inspecting the expected RTL
source locations, probing local RTL tools without installing anything, and
running the cheapest available lint/syntax/simulation command only when source
and tools are actually present.

**Acceptance criteria:**
- The experiment SHALL create
  `results/experiment_1437_discrete_sb_kv260_rtl_lint_sim.json` with
  `status="in_progress"` before the source/tool inspection is completed.
- The experiment SHALL inspect Exp 1422 and the local `hardware/kv260/` RTL
  tree for the expected Discrete SB RTL source, including the planned
  `hardware/kv260/discrete_sb_256.v` source named by the Exp 1422 spec.
- The experiment SHALL probe local `yosys`, `verilator`, `iverilog`, and
  `vivado` availability without installing new toolchains.
- If Discrete SB RTL sources and an available lint or simulation tool exist,
  the experiment SHALL run the cheapest bounded syntax/lint/simulation command
  available and record the exact command plus a short stdout/stderr summary.
- If sources or tools are missing, the artifact SHALL record a precise blocker
  and `next_bitfile_step` instead of fabricating a pass.
- `results/experiment_1437_discrete_sb_kv260_rtl_lint_sim.json` SHALL include
  `status`, `rtl_sources_checked`, `rtl_lint_complete`, `simulation_complete`,
  `synthesis_attempted`, `yosys_available`, `verilator_available`,
  `vivado_available`, `hardware_execution_performed`,
  `hardware_claim_allowed`, `next_bitfile_step`, and `honest_verdict`.
- `hardware_execution_performed=false` and `hardware_claim_allowed=false`
  unless the same run records actual KV260 board evidence.

**Implementation status:** Implemented (Exp 1437; current local run blocked by
missing `hardware/kv260/discrete_sb_256.v` source)

### REQ-ISING-025

**Discrete Simulated Bifurcation KV260 RTL source implementation MUST provide a
minimal deterministic 256-node update core and a matching testbench before
lint/simulation reruns.**

**Rationale:**
Exp 1437 found Verilator, Icarus Verilog, and Yosys locally available, but it
could not lint or simulate because the planned
`hardware/kv260/discrete_sb_256.v` source did not exist.  Exp 1441 must close
that source-level blocker without upgrading the result into a synthesis,
bitfile, or KV260 board execution claim.

**Acceptance criteria:**
- `hardware/kv260/discrete_sb_256.v` SHALL exist and define a synthesizable
  `discrete_sb_256` module for `N_VARIABLES=256`.
- The module SHALL implement the row-serial deterministic update rule from
  `hardware/kv260/discrete_sb_rtl_spec.md`:
  `x_i(t+1) = sign(x_i(t) + eta * sum_j J_ij * x_j(t) - pressure(t))`.
- The source SHALL expose host-loadable packed initial spins and row-major
  signed int8 coupling writes so a lint/simulation rerun can drive a bounded
  deterministic case without requiring KV260 board hardware.
- A matching testbench SHALL exist under `hardware/kv260/` or `tests/hardware/`
  and SHALL drive reset, coupling/init inputs, start, and at least one complete
  update step.
- `results/experiment_1441_discrete_sb_rtl_source_implementation.json` SHALL
  include `status`, `rtl_source_created`, `rtl_source_path`,
  `testbench_created`, `testbench_path`, `spec_requirements_covered`,
  `syntax_probe_command`, `commands_run`, and `honest_verdict`.
- The artifact SHALL identify the next lint/simulation command and SHALL NOT
  claim KV260 board execution unless an actual board command runs in the same
  experiment.

**Implementation status:** Implemented (Exp 1441)

### REQ-ISING-026

**Discrete Simulated Bifurcation KV260 RTL lint/simulation rerun MUST execute
only local source-level checks, record tool failures precisely, and preserve the
no-board-claim boundary.**

**Rationale:**
Exp 1437 proved the local HDL tools were available but was blocked by the
missing `hardware/kv260/discrete_sb_256.v` source.  Exp 1441 created the source
and testbench.  Exp 1451 is therefore a gated rerun: it must verify that the
Exp 1441 source artifact is complete, run the narrowest available local RTL
lint and simulation commands, and report exactly what passed or failed without
upgrading the result into a KV260 hardware claim.

**Acceptance criteria:**
- The experiment SHALL create
  `results/experiment_1451_discrete_sb_rtl_lint_sim_rerun.json` with
  `status="in_progress"` before source/tool inspection completes.
- The experiment SHALL verify that `hardware/kv260/discrete_sb_256.v` exists
  and that `results/experiment_1441_discrete_sb_rtl_source_implementation.json`
  reports `rtl_source_created=true`.
- The experiment SHALL probe local `verilator`, `iverilog`, `yosys`, and
  `vivado` availability without installing new toolchains.
- When source exists, the experiment SHALL run the narrowest available local
  lint command and SHALL run a testbench simulation when a local simulator is
  available.
- The terminal artifact SHALL include `status`, `rtl_source_present`,
  `rtl_lint_complete`, `simulation_complete`, `tools_available`,
  `lint_command`, `simulation_command`, `lint_errors`, `simulation_errors`,
  `hardware_claim_allowed`, `commands_run`, and `honest_verdict`.
- `hardware_claim_allowed=false` unless the same run records an actual KV260
  board execution command and evidence.

**Implementation status:** Implemented (Exp 1451)

### REQ-ISING-027

**Discrete Simulated Bifurcation KV260 RTL regression packaging MUST preserve
the Exp 1451/1460 no-board-claim boundary while emitting a repeatable manifest
and terminal evidence artifact.**

**Rationale:**
Exp 1451 proved that the Discrete SB RTL source and testbench can pass local
Verilator lint and Icarus simulation, while Exp 1460 narrowed KV260 activity to
source-level lint/simulation evidence only. Exp 1476 must package that evidence
into a repeatable regression manifest without upgrading the result into a
Vivado, bitfile, KV260 board, or latency claim.

**Acceptance criteria:**
- The experiment SHALL create
  `results/experiment_1476_kv260_discrete_sb_rtl_regression_pack.json` with
  `status="in_progress"` before regression packaging completes.
- The experiment SHALL read
  `results/experiment_1451_discrete_sb_rtl_lint_sim_rerun.json` and
  `results/experiment_1460_hardware_portfolio_narrowing.json` to preserve the
  no-board-claim boundary.
- The experiment SHALL locate `hardware/kv260/discrete_sb_256.v` and its
  testbench, or document any missing source/testbench blocker without
  fabricating tool passes.
- The experiment SHALL run available local source-level Verilator lint,
  Icarus compile/simulation, and Yosys availability/probe commands without
  installing new tools.
- The experiment SHALL write
  `hardware/kv260/discrete_sb_regression_manifest.md` with exact commands, tool
  availability, expected outputs, and the board/bitfile/latency claim boundary.
- The terminal artifact SHALL include `status`, `rtl_files`,
  `testbench_files`, `rtl_regression_complete`, `verilator_lint_passed`,
  `icarus_sim_passed`, `yosys_available`, `board_execution_performed`,
  `bitfile_produced`, `latency_claimed`, `regression_manifest_path`, and
  `honest_verdict`.
- `board_execution_performed=false`, `bitfile_produced=false`, and
  `latency_claimed=false` unless the same run records actual board execution,
  bitfile generation, or latency measurement evidence.

**Implementation status:** Implemented (Exp 1476)

### REQ-ISING-028

**Discrete Simulated Bifurcation KV260 RTL property packaging MUST define
source-level bounded-behavior, reset, deterministic-ordering, and width/shape
properties while preserving the no-board and no-bitstream claim boundary.**

**Rationale:**
Exp 1476 packaged repeatable source-level lint and simulation evidence for
`hardware/kv260/discrete_sb_256.v`, but it did not name the behavioral
properties that future RTL changes must preserve. Exp 1517 must add that
property pack at the source/simulator layer only, because Exp 1460 keeps the
active KV260 track limited to lint/simulation evidence until Vivado synthesis,
bitfile flashing, and board commands are captured.

**Acceptance criteria:**
- The experiment SHALL create
  `results/experiment_1517_kv260_discrete_sb_rtl_property_pack_v2.json` with
  `status="in_progress"` before source, prior-artifact, or tool inspection
  completes.
- The experiment SHALL verify that Exp 1506 reports
  `prior_kv260_source_track_active=true`; if absent, it SHALL write a terminal
  gated artifact instead of running source checks.
- The experiment SHALL inventory the actual Discrete SB RTL, testbench,
  supplemental property source, existing regression manifest, and local HDL
  helper scripts; if a requested path differs from the actual path, the
  terminal manifest SHALL record the path mismatch.
- The property pack SHALL define source-level properties covering bounded
  Discrete SB behavior, reset behavior, deterministic update ordering, and
  shape/width assumptions for the 256-variable, int8-coupling source.
- The experiment SHALL run only local source-level lint, parse, and simulation
  commands such as Verilator, Icarus Verilog, and Yosys probes; it SHALL NOT run
  Vivado bitfile generation, board programming, PYNQ, SSH, or hardware latency
  commands.
- The experiment SHALL write
  `results/kv260_discrete_sb_property_manifest_1517.json` with checked files,
  property definitions, exact commands, pass/fail results, claim boundaries,
  blockers, and the actual Exp 1460 path used.
- The terminal artifact SHALL include `status`, `kv260_property_pack_ready`,
  `gated_inputs_present`, `source_level_only`, `no_board_execution`,
  `no_bitstream_claim`, `rtl_files_checked`, `properties_defined`,
  `simulations_run`, `lint_or_parse_results`, `property_manifest_path`,
  `blockers`, and `honest_verdict`.
- `source_level_only=true`, `no_board_execution=true`, and
  `no_bitstream_claim=true` SHALL be true in every terminal artifact.
- `kv260_property_pack_ready=true` only when the property manifest is written
  and source-level lint/parse/simulation results are reported.

**Implementation status:** Implemented (Exp 1517)

### REQ-ISING-029

**Discrete Simulated Bifurcation inertial CPU ablation MUST compare an
inertia-augmented simulator against Carnot sequential Gibbs on the same
FoVer-derived Ising problems without making hardware claims.**

**Rationale:**
Exp 1399 compared the base dSB sign-pressure simulator against sequential
Gibbs, while Exp 860 validated EMA inertia for a separate Ising sampler. Exp
1597 must isolate whether an explicit inertial update term is useful in the dSB
CPU simulator before any RTL, KV260, or accelerator work is considered.

**Acceptance criteria:**
- The simulator SHALL expose an inertial dSB update with a configurable inertia
  coefficient and deterministic seeded execution.
- Setting the inertia coefficient to `0.0` SHALL reduce to the existing
  non-inertial dSB update on the same problem, seed, pressure schedule, and
  `eta`.
- The experiment SHALL load between 3 and 5 local FoVer-derived Ising/QUBO
  problems and compare inertial dSB convergence against Carnot's sequential
  Gibbs baseline on the same problems and seeds.
- `results/experiment_1597_inertial_ising.json` SHALL include `status`,
  `experiment_id`, `algorithm`, `baseline_algorithm`,
  `constraint_problems_tested`, `n_variables`, `seeds`,
  `steps_to_convergence_gibbs_baseline`,
  `steps_to_convergence_inertial_ising`,
  `convergence_speedup_inertial_ising`, `inertia_coefficient`,
  `pressure_schedule`, `eta`, `cpu_only`, `simulator_only`,
  `hardware_execution_performed`, `hardware_claim_allowed`,
  `kv260_claim_allowed`, `honest_verdict`, and `per_problem_results`.
- `hardware_execution_performed`, `hardware_claim_allowed`, and
  `kv260_claim_allowed` SHALL remain false because Exp 1597 is a CPU/simulator
  ablation only.

**Implementation status:** Implemented (Exp 1597)

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

### SCENARIO-ISING-034

**Discrete SB RTL lint/sim artifact:** Given Exp 1422's RTL specification and
the local hardware tree, Exp 1437 SHALL probe local RTL tools, inspect the
expected Discrete SB source paths, run bounded lint/simulation only when source
and tools exist, and otherwise write a blocked artifact that disallows hardware
claims.

**Implementation status:** Implemented (Exp 1437; current local run blocked by
missing `hardware/kv260/discrete_sb_256.v` source)

### SCENARIO-ISING-035

**Discrete SB RTL source scaffold artifact:** Given Exp 1437's
`blocked_missing_discrete_sb_rtl_source` verdict and no KV260 board command in
the current run, Exp 1441 SHALL create `hardware/kv260/discrete_sb_256.v`, a
matching testbench, run only local syntax/simulation probes when tools are
present, and write a complete terminal artifact that names the next lint/sim
command while preserving the no-board-claim boundary.

**Implementation status:** Implemented (Exp 1441)

### SCENARIO-ISING-036

**Discrete SB RTL lint/sim rerun artifact:** Given Exp 1441 reports
`rtl_source_created=true` and the Discrete SB source exists locally, Exp 1451
SHALL probe HDL tool availability, run bounded lint and testbench simulation
commands when available, capture failure classes and output summaries, and
write a complete terminal artifact that keeps `hardware_claim_allowed=false`
unless real KV260 board evidence exists in the same run.

**Implementation status:** Implemented (Exp 1451)

### SCENARIO-ISING-037

**Discrete SB RTL regression pack artifact:** Given Exp 1451 reports local
source-level lint/simulation success and Exp 1460 keeps KV260 active only at the
RTL lint/simulation layer, Exp 1476 SHALL rerun the available local RTL
commands, write a repeatable regression manifest, set
`rtl_regression_complete=true` only from source-level evidence, and keep
`board_execution_performed`, `bitfile_produced`, and `latency_claimed` false
unless real board, bitfile, or latency commands run in the same experiment.

**Implementation status:** Implemented (Exp 1476)

### SCENARIO-ISING-038

**Discrete SB source-level property pack artifact:** Given Exp 1506 keeps the
KV260 source track active and Exp 1460 limits KV260 work to RTL lint/simulation
evidence, Exp 1517 SHALL define the bounded-behavior, reset, deterministic
ordering, and shape/width property set for `hardware/kv260/discrete_sb_256.v`,
run only local source-level lint/parse/simulation commands, write
`results/kv260_discrete_sb_property_manifest_1517.json`, and keep board
execution, bitstream, and hardware latency claims disabled.

**Implementation status:** Implemented (Exp 1517)

### SCENARIO-ISING-039

**Inertial dSB CPU ablation artifact:** Given local FoVer rows and no hardware
execution in the current run, Exp 1597 SHALL write
`results/experiment_1597_inertial_ising.json`, compare inertial dSB against
Carnot sequential Gibbs on matched problems and seeds, report convergence steps
and speedup, and keep all hardware/KV260 claim fields false.

**Implementation status:** Implemented (Exp 1597)

### REQ-ISING-040

**THRML/Carnot Parity Independent-RNG Audit MUST confirm independent PRNG lineages.**

**Rationale:**
Previous audits produced byte-identical histograms across 10,240-sample distributions, indicating shared random number generation rather than independent validation. This test ensures that Carnot and THRML utilize completely disjoint root seeds and distinct PRNG lineages for parity sweeps (e.g., n=32, n=64) to confirm legitimate independent sampler parity and observe valid, non-zero stochastic deltas.

**Implementation:** `scripts/experiment_1673_thrml_rng_audit.py`

### SCENARIO-ISING-040

**Independent RNG artifact:** Given exact parity sweeps on n=32 and n=64, the test SHALL output `results/experiment_1673_rng_audit.json` with field `simulator_only_no_hardware_claim: true`, confirm disjoint root seeds, and observe non-zero stochastic deltas and distinct sample-path hashes.

### REQ-ISING-041

**PIPIM dense-problem inertial-update ablation MUST compare CPU-only
inertial p-bit Ising dynamics against Carnot sequential Gibbs.**

**Rationale:**
Dense Ising/QUBO problems expose the oscillation and slow-mixing failure modes
that motivate p-bit inertia. Exp 1674 must isolate the algorithmic effect of
the inertial p-bit update in a simulator-only setting before any hardware or
accelerator claim is considered.

**Acceptance criteria:**
- The experiment SHALL define `scripts/experiment_1674_pipim.py`.
- The experiment SHALL derive between 3 and 5 deterministic dense Ising
  problems from local FoVer rows.
- The PIPIM simulator SHALL use synchronous p-bit updates with a configurable
  EMA inertia term over local fields and deterministic seeded execution.
- The baseline SHALL be Carnot's sequential bipolar Gibbs Ising baseline run on
  the same problems and seeds.
- The artifact SHALL report both time-to-energy deltas and sample-quality
  deltas between PIPIM and sequential Gibbs.
- `results/experiment_1674_pipim.json` SHALL include `status`,
  `experiment_id`, `spec_refs`, `algorithm`, `baseline_algorithm`,
  `dense_problems_tested`, `n_variables`, `seeds`,
  `time_to_energy_gibbs_baseline`, `time_to_energy_pipim`,
  `time_to_energy_delta_steps`, `time_to_energy_speedup`,
  `sample_quality_gibbs_baseline`, `sample_quality_pipim`,
  `sample_quality_delta`, `cpu_only`, `simulator_only`,
  `hardware_execution_performed`, `hardware_claim_allowed`, and
  `honest_verdict`.
- `hardware_execution_performed` and `hardware_claim_allowed` SHALL remain
  false because Exp 1674 is a CPU/simulator-only ablation.

**Implementation:** `scripts/experiment_1674_pipim.py`

### SCENARIO-ISING-041

**PIPIM CPU ablation artifact:** Given local FoVer rows and no hardware
execution in the current run, Exp 1674 SHALL write
`results/experiment_1674_pipim.json`, compare inertial PIPIM against Carnot
sequential Gibbs on matched dense problems and seeds, report time-to-energy and
sample-quality deltas, and keep `hardware_claim_allowed=false`.

### REQ-ISING-042

**LagONN toy Max-3-SAT prototype MUST compare Lagrange multiplier oscillation
against a fixed Ising soft-penalty baseline.**

**Rationale:**
LagONN applies Lagrange multipliers to oscillatory neural networks so violated
constraints grow their own penalties over time instead of relying on one static
soft-penalty weight. Exp 1675 must isolate that mechanism on a tiny,
deterministic Max-3-SAT instance before applying it to larger FoVer-derived or
hardware-bound workloads.

**Acceptance criteria:**
- The experiment SHALL define `scripts/experiment_1675_lagonn.py`.
- The toy problem SHALL contain a deterministic Max-3-SAT instance with
  three-literal clauses and an initially infeasible all-true assignment.
- The LagONN solver SHALL maintain one non-negative Lagrange multiplier per
  clause, update multipliers from current clause violations, and use the
  augmented energy to choose local binary flips.
- The baseline SHALL use a fixed-weight Ising-style soft penalty on the same
  clauses, initial assignment, bias, and step budget.
- `results/experiment_1675_lagonn.json` SHALL include `status`,
  `experiment_id`, `spec_refs`, `algorithm`, `baseline_algorithm`,
  `toy_problem`, `initial_assignment`, `steps_to_convergence_lagonn`,
  `steps_to_convergence_soft_penalty`, `lagonn_converged`,
  `soft_penalty_converged`, `final_violations_lagonn`,
  `final_violations_soft_penalty`, `convergence_speedup_lagonn_over_soft_penalty`,
  `lagrange_multiplier_trace`, `soft_penalty_trace`, `cpu_only`,
  `simulator_only`, `hardware_execution_performed`, `hardware_claim_allowed`,
  and `honest_verdict`.
- `hardware_execution_performed` and `hardware_claim_allowed` SHALL remain
  false because Exp 1675 is a CPU-only prototype.

**Implementation:** `scripts/experiment_1675_lagonn.py`

### SCENARIO-ISING-042

**LagONN Max-3-SAT artifact:** Given the deterministic toy Max-3-SAT instance,
Exp 1675 SHALL write `results/experiment_1675_lagonn.json`, show the LagONN
multiplier solver reaching zero violated clauses within the configured step
budget, compare it against the fixed soft-penalty baseline, and keep
`hardware_claim_allowed=false`.

### REQ-ISING-043

**Ising Consensus Protocol among multiple agent hypotheses.**

**Rationale:**
Scaling constraint checking across multiple reasoning paths. Implement a consensus protocol among multiple agent hypotheses that minimizes an Ising loss function.

**Acceptance criteria:**
- The protocol SHALL generate 5 diverse SOTA answers.
- The protocol SHALL encode their conflicts as an Ising graph.
- The protocol SHALL solve the graph to find the minimum-energy consensus.
- The protocol SHALL write the output to `results/experiment_1872_ising_consensus.json`.

**Implementation:** `python/carnot/pipeline/ising_consensus.py`

### SCENARIO-ISING-043

**Ising Consensus artifact:** Given 5 diverse SOTA answers, their conflicts are encoded as an Ising graph, solved for minimum-energy consensus, and results are written to `results/experiment_1872_ising_consensus.json`.

### REQ-ISING-044

**Hard CSP neural-solver reality check MUST report true constraint satisfaction under a strict time budget.**

**Rationale:**
Neural and Ising-style solvers can report low surrogate energy while still
violating hard CSP constraints. Exp 1927 must make that failure mode visible by
evaluating a neural-style solver on a deterministic hard 3-SAT instance and
scoring the actual clauses directly, not a proxy loss.

**Acceptance criteria:**
- The experiment SHALL define `scripts/experiment_1927_hard_csp_neural.py`.
- The hard CSP SHALL be a deterministic 3-SAT instance with three-literal
  clauses and a documented planted satisfying assignment used only for instance
  construction and validation.
- The evaluator SHALL enforce a configurable wall-clock time budget and stop
  launching solver work once the budget is exhausted.
- The reported constraint satisfaction rate SHALL be computed by directly
  checking the true 3-SAT clauses for each candidate assignment.
- `results/experiment_1927_hard_csp_neural.json` SHALL include `status`,
  `experiment_id`, `spec_refs`, `run_date`, `solver_name`, `csp_family`,
  `problem`, `config`, `time_budget_s`, `wall_time_s`, `timeout_exceeded`,
  `assignments_evaluated`, `true_constraint_satisfaction_rate`,
  `best_satisfied_constraints`, `total_constraints`, `best_assignment`,
  `attempts`, `cpu_only`, `hardware_execution_performed`,
  `hardware_claim_allowed`, and `honest_verdict`.
- `hardware_execution_performed` and `hardware_claim_allowed` SHALL remain
  false because Exp 1927 is a CPU-only reality check.

**Implementation:** `scripts/experiment_1927_hard_csp_neural.py`

### SCENARIO-ISING-044

**Hard CSP neural reality-check artifact:** Given the deterministic hard 3-SAT
instance and a bounded CPU solver configuration, Exp 1927 SHALL write
`results/experiment_1927_hard_csp_neural.json`, report the direct
clause-satisfaction rate achieved within the allotted wall-clock budget, and
avoid any hardware execution claim.

### REQ-ISING-045

**p-bit/p-dit Ising Sampler Accounting v3 MUST extend the CPU Ising sampler to support hardware-accurate p-bit/p-dit states and measure latency/accuracy.**

**Rationale:**
To accurately account for hardware p-bit and p-dit implementations before running on actual hardware, the CPU Ising sampler must simulate p-bit states. This allows tracking the latency overhead and accuracy changes against the standard Ising sampler.

**Acceptance criteria:**
- Extend the Ising sampler to support hardware-accurate p-bit/p-dit states.
- Measure the latency and accuracy changes compared to the baseline sampler.
- Write the output artifact to `results/experiment_1929_p_bit_ising_v3.json`.
- The artifact SHALL contain `status`, `experiment_id`, `latency_change_ms`, `accuracy_change`, `p_bit_supported`, `hardware_execution_performed` (false), `honest_verdict`.

**Implementation:** `python/carnot/samplers/parallel_ising.py`

### SCENARIO-ISING-045

**p-bit Ising Sampler v3 artifact:** Given the implementation of p-bit states in the Ising sampler, Exp 1929 SHALL measure latency and accuracy changes, write `results/experiment_1929_p_bit_ising_v3.json`, and keep `hardware_execution_performed=false`.

### REQ-ISING-046

**SAT/SMT to Ising model translator.**

**Rationale:**
To fully utilize thermodynamic sampling, we must map discrete logical constraints into continuous Ising spin Hamiltonians.

**Acceptance criteria:**
- The experiment SHALL define `python/carnot/inference/ising_translator.py`.
- Map basic AND/OR/NOT clauses to quadratic energy penalties.
- `results/experiment_2147_ising_translation.json` SHALL include `status`, `experiment_id`, and `honest_verdict`.

**Implementation:** `python/carnot/inference/ising_translator.py`

### SCENARIO-ISING-046

**SAT/SMT to Ising model translator artifact:** Given basic AND/OR/NOT clauses, Exp 2147 SHALL translate them to Ising models, write `results/experiment_2147_ising_translation.json`.

### REQ-ISING-7190

**The V633 board-placement receipt MUST preserve each attached board's latest
receipt-backed disposition and compute sparse-graph compatibility on the host
without issuing a hardware command.**

**Rationale:**
KV260 has transcript-backed graduation evidence. GateMate has no operator
physical-state receipt newer than Exp6559. PolarFire has prior SSH CPU-dispatch
evidence but no retained raw transcript hash. A single host aggregation keeps
these different states visible without turning reachability, board CPU work,
or programmable-logic sampling into the same claim. The placement check also
prevents a degree limit from being reported as a proven Z1 topology fit.

**Acceptance criteria:**
- The entrypoint SHALL be
  `scripts/experiments/experiment_7190_v633_board_placement_receipt.py`.
- The entrypoint SHALL print and flush a phase-start line before precondition
  checks. It SHALL print flushed boundaries around every numbered phase,
  validation subprocess, and final atomic write.
- Preconditions SHALL record expected and observed values for the driving
  specification, required source bytes, the exact V633 task contract, Python
  tools, writable output directories, and source hashes.
- An unreadable required input or failed task contract SHALL produce a terminal
  blocked artifact. Its `gate_check_summary` SHALL name the failed check,
  upstream path, field, expected value, and observed value.
- `board_rows` SHALL contain exactly one row for KV260, GateMate, and PolarFire.
  Each row SHALL record its terminal criterion, evidence path, recorded date,
  raw transcript hash or an explicit missing-hash value, last observed value,
  evidence kind, disposition, and exact next prerequisite.
- The KV260 row SHALL preserve Exp3721's graduation and its Exp3709 transcript
  hash without claiming a new performance result.
- The GateMate row SHALL compare a dated operator physical-state receipt with
  Exp6559. If none is newer, it SHALL inherit Exp7146's block and record zero
  JTAG, reset, flash, cable, and power commands. A newer receipt SHALL only
  name the next action for a later task.
- The PolarFire row SHALL keep SSH reachability and board CPU work separate
  from programmable-logic sampling. Missing raw dispatch transcript evidence
  SHALL remain unresolved.
- `placement_rows` SHALL record maximum degree, edge count, coefficient field
  width, and host correction cost for each checked graph contract. Existing
  Exp7187 and Exp7188 contracts SHALL be included when readable and ready.
- If the Exp7187/7188 branch is unavailable, one deterministic synthetic
  `n=16` sparse graph SHALL be checked and labeled `compatibility_only`.
- A row MAY pass the necessary `maximum_degree <= 16` check. It SHALL still use
  `topology_unknown` unless an explicit node-and-edge mapping into the
  published parent graph is present. The experiment SHALL NOT remove edges to
  force a fit.
- `board_placement_receipt_complete_score` SHALL equal `1` only when all three
  board dispositions and all available host placement checks are recorded.
  Blocked board rows do not make this aggregation incomplete.
- `hardware_execution_claimed` SHALL remain false. The result SHALL not claim
  Z1 execution, latency, power, speed, FPGA integration, or new board
  performance.
- A successful `honest_verdict` SHALL use the shared terminal `positive:`
  prefix so downstream artifact-readiness checks classify the receipt as
  terminal.
- The artifact SHALL include `field_principles`, `status`,
  `preconditions_checked`, `run_date`, `inference_substrate`,
  `execution_venue`, `duration_s`, `source_artifact_hashes`, `rows`,
  `random_seed`, `reproducibility_checksum`, `gate_check_summary`,
  `verifier_is_oracle`, `verdict_class`, `honest_verdict`,
  `inference_substrate_class`, `board_placement_receipt_complete_score`,
  `board_rows`, `placement_rows`, `operator_state_receipt`, and
  `hardware_execution_claimed`.
- The terminal artifact SHALL be written atomically to
  `results/experiment_7190_v633_board_placement_receipt.json` and SHALL pass
  its independent validator.

**Implementation status:** Implemented (Exp 7190)

### SCENARIO-ISING-7190-PREFLIGHT

**Precondition failure:** Given an unreadable required source, a missing
`REQ-*` driving spec, a changed V633 task contract, a missing Python tool, or an
unwritable output directory, Exp7190 SHALL write a terminal blocked artifact
before host placement computation and identify the exact failed check.

### SCENARIO-ISING-7190-BOARDS

**Three distinct board states:** Given the current checked-in receipts,
Exp7190 SHALL retain KV260's transcript-backed graduation, GateMate's inherited
post-Exp6559 physical-state block, and PolarFire's unresolved raw dispatch
transcript. The receipt SHALL not convert SSH or board CPU evidence into
programmable-logic sampling evidence.

### SCENARIO-ISING-7190-GATEMATE

**Changed-state authorization boundary:** Given no operator receipt newer than
Exp6559, Exp7190 SHALL record zero physical or programming commands. Given a
newer valid receipt, it SHALL record the newly authorized next action for a
later task and still run no hardware command.

### SCENARIO-ISING-7190-PLACEMENT

**Necessary but insufficient topology check:** Given Exp7187 and Exp7188, each
host row SHALL preserve the full graph, compute its degree and edge count,
record its field-width and correction-cost contract, and return
`topology_unknown` without an explicit parent-graph mapping. Given unavailable
upstream artifacts, the same rule SHALL apply to the fixed synthetic `n=16`
fallback graph.

### SCENARIO-ISING-7190-ARTIFACT

**Complete host-only receipt:** Given all required sources and three board
dispositions, Exp7190 SHALL emit a validated atomic artifact with
`board_placement_receipt_complete_score=1`, `inference_substrate_class` set to
`aggregation`, and `hardware_execution_claimed=false`, even when one or more
board rows remain externally blocked.

### REQ-ISING-7203

**The V634 hardware-correction study MUST preserve each attached board's
receipt-backed disposition and measure delayed-acceptance correction cost on
the host without issuing a hardware command.**

**Rationale:**
Exp7188 established the corrected transition law on small CPU fixtures.
Exp7190 retained distinct KV260, GateMate, and PolarFire evidence states.
Deployment planning also needs measured host correction costs at larger graph
sizes. These costs do not supply device timing, topology mapping, power, or
mixing evidence.

**Acceptance criteria:**
- The executable entrypoint SHALL be
  `scripts/experiments/experiment_7203_v634_hardware_correction.py`.
- The entrypoint SHALL print and flush a boundary for every numbered phase.
  It SHALL print before and after each long benchmark or validation call.
- Preconditions SHALL bind nonempty source bytes, the exact V634 roadmap task,
  Python tools, writable result and checkpoint directories, upstream gate
  fields, source hashes, and upstream quarantine state.
- A missing required resource or rejected upstream SHALL produce one terminal
  blocked artifact. Its `gate_check_summary` SHALL name the failed check,
  upstream, field, expected value, and observed value.
- Quarantined upstream data SHALL be rejected before its structured readiness
  field can authorize consumption. A known failed upstream value SHALL remain
  visible and SHALL NOT become positive evidence.
- `board_rows` SHALL contain exactly one row for KV260, GateMate, and PolarFire.
  Each row SHALL retain the terminal criterion, receipt date, source hash,
  evidence type, disposition, and unresolved prerequisite.
- KV260 graduation SHALL remain only when the cited receipt and raw transcript
  hash support it. PolarFire SSH and board CPU work SHALL remain distinct from
  programmable-logic sampling and its missing raw dispatch transcript SHALL
  remain unresolved.
- The latest valid operator-authored GateMate physical-state receipt SHALL be
  compared with Exp6559. Without a newer receipt, all JTAG, reset, flash,
  cable, and power command counts SHALL remain zero. A newer receipt MAY name
  one action for a later task but SHALL NOT authorize this host study to run it.
- The host benchmark SHALL use
  `make_frustrated_instance(n, seed)` from Exp7187 with `n` in `{16,32,64}`,
  `k=2`, `beta=1`, seeds `7203001..7203010`, and precisions `{4,8,16}`.
  It SHALL retain every edge and nonzero field from each generated instance.
- For each size, seed, precision, and arm, the equal-work panel SHALL run 1024
  proposals. The equal-wall panel SHALL run for at least 100 milliseconds per
  chain. Both panels SHALL compare full-precision Metropolis with Exp7188's
  two-stage delayed-acceptance law.
- Every cost row SHALL report proposal count, accepted moves, acceptance rate,
  cheap-stage rejects, full-energy calls, measured host latency, and retained
  rejected states. No host benchmark may continue after 900 seconds total.
- Each non-control precision condition SHALL change at least one coefficient
  before its distortion result is interpreted. Exact-grid fixtures SHALL stay
  explicit negative controls.
- A separate `n=8`, `k=2`, `beta=1` exact-law panel SHALL recompute full-target
  stationarity for the corrected kernel before any correction-cost claim.
- `break_even_rows` SHALL combine measured host costs only with labeled,
  hypothetical device-compute and transfer-latency axes. Missing measured
  device timing SHALL remain `unknown`. Each row SHALL state that the envelope
  is per proposed transition and does not establish effective-sample throughput
  or mixing speed.
- Every graph row SHALL record actual edge count and maximum degree with zero
  dropped edges. `maximum_degree <= 16` is necessary only. `topology_fit` SHALL
  remain `topology_unknown` without an explicit parent-graph mapping.
- `hardware_envelope_complete_score` SHALL equal `1` only after all three board
  dispositions, the exact-law panel, both cost panels, and the break-even
  envelope finish. Blocked board dispositions do not erase completed host data.
- `MODEL_SPECS` SHALL be empty and `model_invoked` SHALL be false. The result
  SHALL not claim TSU, FPGA, soft-spin, Z1, board, power, speed, or mixing
  execution.
- The artifact SHALL include every field named by the V634 task contract,
  per-unit `rows`, an explicit `sample_size_budget`, and a checksum over source
  hashes, code, seeds, contracts, and raw rows.
- The terminal artifact SHALL be written atomically to
  `results/experiment_7203_v634_hardware_correction.json` and SHALL pass its
  independent validator.

**Implementation status:** Implemented (Exp 7203)

### SCENARIO-ISING-7203-PREFLIGHT

**Fail-closed intake:** Given a missing source, changed roadmap contract,
failed readiness field, or quarantined Exp7188 or Exp7190 artifact, Exp7203
SHALL publish a diagnosed terminal block before running the host benchmark.

### SCENARIO-ISING-7203-BOARDS

**Distinct board evidence:** Given the current checked-in receipts, Exp7203
SHALL preserve KV260 graduation, GateMate's post-Exp6559 physical-state block,
and PolarFire's unresolved raw dispatch transcript with zero hardware commands.

### SCENARIO-ISING-7203-SMALL-LAW

**Correction authority:** Given the fixed `n=8` exact-law fixtures, the
delayed-acceptance matrix SHALL preserve the full-precision target within the
declared numerical tolerance. An exact-grid negative control SHALL report no
coefficient change and no target distortion.

### SCENARIO-ISING-7203-COST

**Matched host costs:** Given every fixed size, seed, and precision, Exp7203
SHALL retain one full and corrected row for equal work and one full and
corrected row for equal wall. Rejections SHALL keep the current chain state.

### SCENARIO-ISING-7203-BREAK-EVEN

**Hypothetical device envelope:** Given measured host costs and declared
hypothetical transfer and device latencies, Exp7203 SHALL compute a
per-transition break-even condition without claiming measured device timing,
mixing speed, or effective-sample throughput.

### SCENARIO-ISING-7203-ARTIFACT

**Complete CPU evidence:** Given clean upstream inputs and completed bounded
CPU panels, Exp7203 SHALL emit a validated atomic artifact with
`hardware_envelope_complete_score=1`,
`inference_substrate_class=cpu_exact_solver_or_simulator`,
`hardware_execution_claimed=false`, and a terminal `complete:` verdict.

### REQ-ISING-7215

**The V635 sampler prototype MUST implement and certify the target-weighted
down-up transition on finite fixed-cardinality Ising slices.**

**Rationale:**
The prior sampler study used pair-swap Metropolis. Algorithm 1 of
arXiv:2609.08873v1 uses a different transition. It first removes a uniform
member. It then samples a replacement from the full conditional law. The
removed member remains a candidate, so the transition can stay at the same
state. A finite check of this kernel does not prove the paper's mixing theorem.

**Acceptance criteria:**
- The opt-in kernel SHALL live in
  `python/carnot/samplers/experiment_7215_down_up.py`. Existing sampler defaults
  SHALL remain unchanged.
- The executable entrypoint SHALL be
  `scripts/experiments/experiment_7215_v635_down_up_prototype.py`.
- The implementation SHALL represent each state as a `k`-subset. It SHALL
  remove one member uniformly. It SHALL sample the replacement from every site
  outside the reduced set, including the removed site.
- Replacement weights SHALL be proportional to `exp(-beta * E)`. The energy
  SHALL use the Exp7187 edge-once convention. The normalization SHALL use a
  log-sum-exp shift.
- The sampled step SHALL accept optional caller-owned uniform tapes for the
  down choice and the up categorical choice.
- The kernel SHALL retain self-transitions. It SHALL return the only state for
  `k=0` and `k=n`. It SHALL reject invalid `k`, invalid subsets, nonfinite or
  negative beta, zero fields, and asymmetric or duplicate edge input.
- An independently derived transition matrix SHALL retain all down paths that
  reach the same target. Its rows SHALL be nonnegative and stochastic.
- The exact target law SHALL use independent scalar energies. The law SHALL
  satisfy detailed balance and stationarity to at most `1e-10`.
- The experiment SHALL retain 90 transition cells for `n=8`,
  `k in {1,2,4}`, `beta in {0,1,2}`, and seeds `7215001..7215010`.
- Representative empirical one-step categorical draws SHALL be compared with
  exact transition rows. The artifact SHALL retain the draw count and error.
- Mutation checks SHALL reject kernels that omit the removed site, reverse the
  energy sign, or drop self-transitions. All three controls SHALL use one
  nondegenerate fixture.
- The cost contract SHALL charge each replacement-candidate energy and the
  normalization work. It SHALL make no hardware speed claim.
- Preconditions SHALL bind required bytes, the exact roadmap task, imports,
  tools, output directories, the paper version and excerpt locations, and the
  contextual Exp7202 quarantine and authentication state. A quarantine signal
  SHALL prevent any structured value from authorizing this task.
- Exp7202's known failed quality and speed values SHALL remain failed. This task
  SHALL not promote either value.
- A required external precondition failure SHALL produce a terminal blocked
  artifact. Its gate summary SHALL name the check, upstream, field, expected
  value, and observed value.
- The complete artifact SHALL set `MODEL_SPECS=[]`, `model_invoked=false`,
  `execution_venue=host`, and
  `inference_substrate_class=cpu_exact_solver_or_simulator`.
- `down_up_kernel_ready_score` SHALL equal one only when all finite-law checks
  and all mutation checks pass. A successful exact certification SHALL use
  `verdict_class=circular_positive` because the verifier is not an independent
  scientific oracle.
- The artifact SHALL make `paper_replication_claimed=false`,
  `general_mixing_theorem_claimed=false`, and
  `hardware_speed_claimed=false`.
- The terminal artifact SHALL be written atomically to
  `results/experiment_7215_v635_down_up_prototype.json`. Its validator SHALL
  recompute row coverage, scientific gates, claim limits, and checksums.

**Implementation status:** Prototype implementation in progress (Exp 7215)

### SCENARIO-ISING-7215-KERNEL

**Conditional replacement:** Given a valid non-boundary subset and two uniform
tape values, the kernel SHALL remove the indexed member and sample from all
sites outside the reduced set. The exact transition matrix SHALL match this
conditional construction and retain the self-transition probability.

### SCENARIO-ISING-7215-BOUNDARIES

**Boundary and input handling:** Given `k=0` or `k=n`, the kernel SHALL return
the sole state with probability one. Given invalid cardinality, fields, edges,
or a malformed subset, it SHALL fail before sampling.

### SCENARIO-ISING-7215-FINITE-LAW

**Complete small roster:** Given the fixed 90-cell roster, every transition
matrix SHALL be stochastic, nonnegative, cardinality preserving, reversible,
and stationary within `1e-10` under the independently calculated target law.

### SCENARIO-ISING-7215-MUTATIONS

**Effective negative controls:** Given the fixed nondegenerate fixture, the
law tests SHALL reject omission of the removed candidate, reversal of the
energy sign, and deletion of self-transitions.

### SCENARIO-ISING-7215-ARTIFACT

**Bounded CPU claim:** Given clean required inputs and complete finite checks,
the experiment SHALL emit one validated atomic host artifact. It SHALL report
kernel readiness as circular evidence. It SHALL not report a general mixing
theorem, paper reproduction, hardware execution, or hardware speed.

### REQ-ISING-7216

**The V635 quality study MUST compare the shipped target-weighted down-up
kernel with pair-swap Metropolis under frozen work, wall-time, and independent
long-chain quality protocols.**

**Rationale:**
Exp7215 certified the finite transition law, but finite-law correctness alone
does not establish useful mixing or performance. The down-up conditional is
more expensive than one pair-swap proposal. The comparison therefore charges
every target-energy evaluation and normalization, qualifies both arms against
an independently enumerated target law, and keeps completion separate from
scientific value.

**Acceptance criteria:**
- The executable entrypoint SHALL be
  `scripts/experiments/experiment_7216_v635_down_up_quality.py`, backed by an
  opt-in experiment module. Existing sampler defaults SHALL remain unchanged.
- Preconditions SHALL print before each check and bind nonempty named source
  bytes, the exact V635 roadmap task, required imports and tools, writable
  result and checkpoint directories, and the exact Exp7215 producer gate.
  Artifact and exclusion-manifest quarantine signals SHALL be checked before
  any structured gate value. Only a two-key `value`/`principle` wrapper SHALL
  be unwrapped. The Exp7215 artifact SHALL authenticate with its shipped
  validator, and `down_up_kernel_ready_score` SHALL equal one.
- An external precondition failure SHALL produce a terminal blocked artifact
  with no invented computation rows. `gate_check_summary` SHALL name the
  failed check, upstream, field, expected value, and observed value.
- Before sampling, each graph/cell target SHALL be enumerated independently
  for `n in {16,32}`, `k=2`, and the frozen cells `(32,1)`, `(32,2)`, and
  `(16,1)`, where each tuple is `(n,beta)`. The primary cell SHALL be
  `n=32,k=2,beta=1`; the other two cells are sensitivities.
- The ten graph seeds SHALL be `7216001..7216010`. Every graph SHALL use
  Exp7187's nonzero fields, frustrated couplings, and edge-once Hamiltonian
  `E(s)=-sum J_ij s_i s_j-sum h_i s_i`. No field or coupling SHALL be tuned
  after mixing results are observed.
- Each graph/cell SHALL run four overdispersed independently seeded chains per
  arm. Arm random streams SHALL be distinct and arm order SHALL be randomized
  from frozen seeds; identical explicit RNG tapes SHALL not be used as parity
  evidence.
- Matched-work rows SHALL charge exactly 100000 target-energy evaluations per
  chain, including initialization and any unfinished final conditional work.
  Matched-wall rows SHALL charge initialization, every completed conditional
  normalization, and all target-energy calls within a two-second window.
  Rejections and self-transitions SHALL remain in each sampled trace.
- The quality panel SHALL use 4096 burn-in transitions and 16384 retained
  transitions per chain, with a 1800-second total measurement cap. A truncated
  panel SHALL be terminal complete-but-insufficient evidence and SHALL never
  set the value score.
- Raw matched-budget and quality state-index traces and every stream seed SHALL
  be written as a compressed archive under `results/checkpoints/`. The
  terminal artifact SHALL retain its path, byte count, and SHA-256 digest.
- Exact rows SHALL retain state count, probability normalization, energy
  parity, exact energy moments, all site marginals, and prespecified probe
  variances. Quality rows SHALL track energy and sites `0`, `floor(n/3)`, and
  `floor(2n/3)`, plus empirical total variation where estimable.
- Every probe SHALL use the same initial-positive-sequence ESS estimator across
  arms and SHALL report estimated mean error with Monte Carlo uncertainty.
  Split R-hat SHALL use the four independently seeded chains. Zero target
  variance SHALL be structurally degenerate; positive target variance with a
  constant observed trace SHALL be unqualified, never infinite ESS.
- The frozen primary gate SHALL require complete panels, finite-law checks,
  zero sector violations, ESS at least 200 in every chain for every
  nondegenerate probe, split R-hat at most 1.05, pooled absolute occupancy mean
  error at most 0.02, and pooled standardized energy mean error at most 0.05
  for both arms. Only after both arms meet the exact-mean tolerances SHALL it
  compare minimum-probe ESS per second.
- The primary throughput clause SHALL use ten paired graph ratios and a 95%
  paired graph-bootstrap interval with seed `7216002` and 10000 resamples. Its
  lower endpoint SHALL exceed one for down-up to pass.
- `down_up_comparison_complete_score` SHALL equal one for a complete
  measurement receipt, independent of quality. `down_up_value_score` SHALL
  equal one only when every frozen primary clause passes.
- The artifact SHALL preserve V634's `nfr_01_10x_met=false` without promotion.
  It SHALL set `MODEL_SPECS=[]`, `model_invoked=false`,
  `execution_venue=host`, and use
  `inference_substrate_class=cpu_exact_solver_or_simulator` after qualifying
  CPU work or `blocked_no_run` only when no qualifying work ran.
- A successful exact-authority claim SHALL use
  `verdict_class=circular_positive`. A completed failed or insufficient gate
  SHALL use `verdict_class=null`. The study SHALL set
  `paper_replication_claimed=false` and SHALL make no Rust-speed, TSU,
  sparse-SK theorem, or hardware-power claim.
- The terminal artifact SHALL be written atomically to
  `results/experiment_7216_v635_down_up_quality.json`. Its validator SHALL
  recompute row coverage, scientific gates, claim limits, row hashes, and the
  artifact checksum.

**Implementation status:** Quality study in progress (Exp7216)

### SCENARIO-ISING-7216-PREFLIGHT

**Authenticated producer gate:** Given a clean, producer-valid Exp7215
artifact whose exact readiness gate is one, the study SHALL proceed. Given an
artifact or manifest quarantine flag, a failed producer validator, a malformed
wrapper, or a changed gate value, it SHALL fail closed before sampling and
retain the exact observation in `gate_check_summary`.

### SCENARIO-ISING-7216-LAW-TRACE

**Independent law to sampled trace:** Given one frozen frustrated graph, the
study SHALL enumerate the finite target through the independent scalar energy
path, run each actual kernel with independent streams, retain every transition,
and score energy, fixed occupancy probes, sector safety, and empirical total
variation against that law.

### SCENARIO-ISING-7216-QUALITY

**Constant-trace and split-chain discipline:** Given positive exact variance
and a constant observed probe, the quality calculation SHALL return null ESS
and an unqualified result. Given four nonconstant chains, it SHALL compute
per-chain ESS and one split R-hat with the same estimators for both arms.

### SCENARIO-ISING-7216-MATCHED-BUDGETS

**Charged conditional cost:** Given the equal-work protocol, both arms SHALL
consume exactly 100000 target-energy evaluations while only completed
transitions enter traces. Given the equal-wall protocol, both timers SHALL
start before initialization and retain the measured overshoot and completed
work. All normalization and unfinished conditional costs SHALL remain visible.

### SCENARIO-ISING-7216-GATE

**Quality before throughput:** Given complete primary rows, the evaluator SHALL
first require both arms' exact-law mean tolerances, chain ESS, split R-hat, and
sector clauses. It SHALL compute the paired graph-bootstrap throughput interval
only for qualified comparators and set value to one only when its lower
endpoint exceeds one.

### SCENARIO-ISING-7216-ARTIFACT

**Bounded host conclusion:** Given a complete panel, the experiment SHALL emit
one validated atomic host artifact and a compressed trace receipt. Completion
SHALL not imply value. The artifact SHALL preserve the V634 NFR null and SHALL
not claim a paper theorem, Rust or TSU speed, or hardware power savings.

### REQ-ISING-7217

**The V635 ABI and board-readiness study MUST bind the shipped PyO3 extension
to the executing virtual-environment interpreter and preserve independent
attached-board dispositions without issuing a hardware operation.**

**Rationale:**
The 2026-09-11 reproduction failed while loading the native extension with an
undefined `Py_GetConstantBorrowed` symbol. A fresh interpreter-bound import,
explicit transition replay, and cross-process state round trip can establish
host deployment readiness. They do not reopen the failed 10x throughput claim
or establish board performance.

**Acceptance criteria:**
- The executable entrypoint SHALL be
  `scripts/experiments/experiment_7217_v635_abi_board_readiness.py`.
- The entrypoint SHALL print and flush every numbered phase boundary. It SHALL
  stream long subprocess output and emit truthful elapsed-time heartbeats.
  Native import calls SHALL have bounded deadlines. A build SHALL use a
  900-second deadline.
- Preconditions SHALL bind nonempty source bytes, `REQ-ISING-7217`, the exact
  V635 roadmap task, required imports and tools, writable result and checkpoint
  directories, exact producer gate fields, producer authentication, source
  hashes, and quarantine state.
- Quarantine signals from an artifact or the exclusion manifest SHALL reject
  an upstream before its structured gate value is read. Only an exact two-key
  `principle`/`value` wrapper SHALL be unwrapped. A failed upstream value SHALL
  remain failed and SHALL NOT authorize a positive claim.
- A missing unchanged external prerequisite SHALL produce one terminal blocked
  artifact. Its `gate_check_summary` SHALL name the failed check, upstream,
  field, expected value, and observed value. Checkpoints SHALL remain under
  `results/checkpoints/`.
- The study SHALL print and record the selected `.venv` interpreter, Python
  version, SOABI, extension suffix, Python library configuration, and PyO3/Cargo
  features. It SHALL reproduce the historical extension import in a bounded
  fresh process and retain its path, exit code, stdout, stderr, and undefined
  symbol observation. A working existing binary SHALL take a verified fast
  path without recreating the historical failure.
- When the existing binary does not import, the study SHALL build
  `carnot-python` in a task-specific Cargo target directory with `PYO3_PYTHON`
  set to the executing `.venv` interpreter. It SHALL NOT set
  `PYO3_USE_ABI3_FORWARD_COMPATIBILITY` as a workaround. The new binary SHALL be
  copied to a task-specific load directory without deleting shared outputs.
- A fresh process SHALL import the selected binary and run the shipped
  `RustFixedCardinalitySampler` with explicit transition tapes. The exact
  outputs SHALL match the Exp7187 Python energy and transition authority.
- The native sampler SHALL serialize its restart state. A second fresh process
  SHALL restore that state and continue the stream. The receipt SHALL retain
  the binary hash, module `__file__`, interpreter, linked libraries, exact
  outputs, and exit codes.
- `native_abi_ready_score` SHALL equal one only after genuine compiled import,
  explicit replay parity, and cross-process state restoration all pass. The
  task SHALL NOT rerun a throughput sweep or overwrite Exp7201 or Exp7202.
- `board_rows` SHALL contain one terminal row for KV260, GateMate, and PolarFire.
  Each row SHALL retain its dated source, evidence hash, terminal criterion,
  disposition, and exact next prerequisite.
- The study SHALL preserve transcript-supported KV260 graduation. It SHALL
  preserve GateMate's need for an operator-authored physical-state change after
  Exp6559 and PolarFire's missing raw dispatch transcript. It SHALL issue zero
  JTAG, reset, flash, cable, power, storage-discovery, or board-probe operations.
  A future KV260 action SHALL use SSH.
- The operation map SHALL label down-up replacement normalization as host work
  until an explicit mapping exists. It MAY list potential CPU, GPU, FPGA, and
  TSU operations but SHALL claim no measured device performance. Z1 degree at
  most 16 SHALL remain necessary but insufficient for fixed-parent-graph
  placement. Topology fit, device latency, and device power SHALL remain
  unknown.
- `abi_board_receipt_complete_score` SHALL equal one after the host ABI outcome
  and all three board dispositions are terminal. A blocked board row SHALL not
  erase a successful native-host result.
- The artifact SHALL set `MODEL_SPECS=[]`, `model_invoked=false`,
  `execution_venue=host`, and record the hostname in `execution_host`. Genuine
  native replay SHALL use
  `inference_substrate_class=cpu_exact_solver_or_simulator`.
- The artifact SHALL include every field in the V635 task contract, per-unit
  numeric `rows`, an explicit `sample_size_budget`, source hashes, ABI rows,
  E2E receipts, board rows, hardware-operation receipts, and checksums.
- The terminal artifact SHALL be written atomically to
  `results/experiment_7217_v635_abi_board_readiness.json`. Its validator SHALL
  recompute terminal state, native provenance, parity, board continuity, claim
  limits, row hashes, and the artifact checksum.

**Implementation status:** Implemented and verified (Exp7217)

### SCENARIO-ISING-7217-PREFLIGHT

**Authenticated and quarantine-safe intake:** Given clean producer artifacts
whose shipped validators pass and whose exact gate fields match, the study SHALL
continue. Given any quarantine signal, malformed wrapper, authentication
failure, or changed gate, it SHALL block before native execution and retain the
exact failed observation.

### SCENARIO-ISING-7217-ABI

**Fresh native execution:** Given the selected interpreter and an importable
interpreter-bound extension, two fresh processes SHALL replay explicit
transitions and continue one serialized Rust stream. The outputs SHALL match the
independent Python authority before native readiness becomes one.

### SCENARIO-ISING-7217-REBUILD

**Scoped recovery:** Given a failed historical-binary import, the study SHALL
build in its task-specific target with the exact executing interpreter. It SHALL
load the new binary without changing global Python, dependency versions, or
shared build outputs.

### SCENARIO-ISING-7217-BOARDS

**Read-only continuity:** Given the newest authenticated receipts, the study
SHALL preserve KV260 graduation, GateMate's unchanged physical-state block, and
PolarFire dispatch uncertainty. It SHALL issue no board operation.

### SCENARIO-ISING-7217-ARTIFACT

**Separate host and board readiness:** Given successful native execution and
terminal board dispositions, the study SHALL emit a complete atomic host
artifact with both readiness scores equal to one. Per-board blocks SHALL remain
visible and SHALL not change the native-host score.

### REQ-ISING-7231

**The V636 board-continuity audit MUST preserve authenticated KV260 and
GateMate state and recover one bounded PolarFire CPU-dispatch transcript when
the deployed workload is available.**

**Rationale:**
Exp7217 preserved KV260 graduation, GateMate's unchanged physical-state block,
and PolarFire CPU-dispatch uncertainty. The PolarFire result retained output
values but not the raw dispatch bytes. A read-only audit can close that evidence
gap without treating SSH CPU work as programmable-logic sampling. The compact
controller contract also supplies a small deployment footprint. It does not
supply topology, power, or speed evidence.

**Acceptance criteria:**
- The production implementation SHALL be
  `python/carnot/experiment_7231_v636_board_continuity.py`. The executable
  entrypoint SHALL be
  `scripts/experiments/experiment_7231_v636_board_continuity.py`.
- The entrypoint SHALL print and flush before preconditions and at every
  numbered phase boundary. It SHALL print before and after the bounded SSH
  subprocess, final validation, and atomic terminal write.
- Preconditions SHALL bind nonempty cited source bytes, `REQ-ISING-7231`, the
  exact V636 roadmap task, required imports, SSH availability, and writable
  raw, checkpoint, and result destinations.
- The Exp7217 board receipt and available compact-controller contract SHALL
  pass their shipped validators. Artifact or exclusion-manifest quarantine
  signals SHALL reject an upstream before any gate value is consumed. Only a
  dictionary with exactly `principle` and `value` keys MAY be unwrapped.
- A failed unchanged repository precondition SHALL produce one terminal
  blocked artifact. Its `gate_check_summary` SHALL name the failed check,
  upstream, field, expected value, and observed value. It SHALL contain no
  board command or fabricated row.
- `board_rows` SHALL contain one terminal row for KV260, GateMate, and
  PolarFire. Each row SHALL record the independently selected latest
  authenticated receipt, receipt date and hash, exact observation, terminal
  criterion, disposition, execution venue, processor class, and next
  prerequisite.
- The KV260 row SHALL preserve the transcript-supported graduation and the
  exact criterion `board-level programmable-logic latency transcript and
  successful KV260 synthesis`. This task SHALL issue no KV260 command. Any
  future KV260 access SHALL use `ssh kria`; host storage discovery is forbidden.
- GateMate access SHALL require an operator-authored physical-state receipt
  newer than Exp6559. Without it, the row SHALL use
  `blocked_inherited_no_new_physical_state` and record zero JTAG, reset, flash,
  cable, and power commands. A valid new receipt SHALL authorize only a named
  later action; this task SHALL still issue no GateMate command.
- The PolarFire audit SHALL issue at most one SSH command. It SHALL use a
  10-second connection timeout and a 60-second remote workload timeout. The
  remote shell SHALL execute only the already deployed `/usr/bin/carnot`
  binary when it is executable. It SHALL not install, upload, flash, or change
  persistent configuration.
- A PolarFire attempt SHALL retain local raw stdout and stderr, transport exit
  code and timing, executed binary hash, fixed input hash, returned output
  hash, and local hash verification. A missing transport or deployed workload
  SHALL produce an exact blocked PolarFire row. It SHALL not invent a dispatch
  result.
- The top-level `execution_venue` SHALL be `host`. A completed remote smoke
  SHALL identify `polarfire` as its child venue and `cpu` as its processor.
  It SHALL set programmable-logic sampling false.
- `operation_map` SHALL record the current compact controller's memory-table
  footprint and separate host, board CPU, FPGA, and TSU placement. Topology
  fit, device power, and device speed SHALL remain unknown.
- `board_continuity_complete_score` SHALL equal one when all three terminal
  board dispositions are recorded. A blocked board row SHALL not reduce this
  receipt-completeness score.
- The artifact SHALL set `MODEL_SPECS=[]` and `model_invoked=false`. A
  completed PolarFire CPU smoke SHALL use `cpu_exact_solver_or_simulator` for
  both substrate fields. Historical aggregation without a new smoke SHALL use
  `aggregation`. A precondition block before qualifying work SHALL use
  `blocked_no_run`.
- Every row SHALL contain `unit_id`, `arm`, `seed`, `metric`, `error`, and
  `abstention`. `sample_size_budget` SHALL retain planned, attempted,
  completed, censored, and independent-unit counts.
- The terminal artifact SHALL contain every V636 task-contract field. It SHALL
  be validated independently and written atomically to
  `results/experiment_7231_v636_board_continuity.json`. Raw dispatch evidence
  SHALL stay under `results/raw/experiment_7231/`. Running state SHALL stay
  under `results/checkpoints/`.

**Implementation status:** Implemented and verified (Exp7231)

### SCENARIO-ISING-7231-PREFLIGHT

**Authenticated fail-closed intake:** Given a missing cited source, changed
roadmap identity, failed producer validation, quarantine signal, malformed
wrapper, or unwritable output, Exp7231 SHALL stop before board access and emit
the exact failed gate.

### SCENARIO-ISING-7231-BOARDS

**Independent board continuity:** Given the newest authenticated receipt for
each board, Exp7231 SHALL preserve KV260's exact graduation criterion, compare
GateMate with Exp6559, and keep PolarFire CPU evidence separate from fabric
sampling.

### SCENARIO-ISING-7231-GATEMATE

**Physical-state boundary:** Given no operator-authored GateMate state change
after Exp6559, Exp7231 SHALL issue no GateMate command. Given a newer valid
receipt, it SHALL only record the next action for a later task.

### SCENARIO-ISING-7231-POLARFIRE

**Bounded deployed-workload smoke:** Given reachable `ssh polarfire` and an
executable `/usr/bin/carnot`, Exp7231 SHALL run the existing binary once within
60 seconds and retain verified raw bytes and hashes. Given either prerequisite
is missing, the PolarFire row SHALL remain blocked without upload or install.

### SCENARIO-ISING-7231-PLACEMENT

**Separated deployment map:** Given an authenticated compact-controller
contract, Exp7231 SHALL retain its byte and bit footprint. It SHALL label
measured CPU work and unmeasured FPGA or TSU paths separately. It SHALL leave
topology, power, and speed unknown.

### SCENARIO-ISING-7231-ARTIFACT

**Complete disposition receipt:** Given three terminal board rows, Exp7231
SHALL emit `board_continuity_complete_score=1` even when GateMate or PolarFire
is blocked. Completion SHALL not imply hardware performance or fabric sampling.

### REQ-ISING-7244

**The V637 board-disposition review MUST authenticate the latest three-board
evidence without issuing a board operation.**

**Rationale:**
Exp7231 retained a successful PolarFire CPU dispatch transcript and preserved
KV260 graduation. GateMate still has no operator-authored physical-state change
after Exp6559. Repeating either completed smoke would add no evidence. The next
receipt must identify each board's exact state and next prerequisite while it
keeps board CPU execution separate from FPGA fabric execution.

**Acceptance criteria:**
- The production implementation SHALL be
  `python/carnot/experiment_7244_v637_board_disposition.py`. The executable
  entrypoint SHALL be
  `scripts/experiments/experiment_7244_v637_board_disposition.py`.
- Preconditions SHALL bind nonempty source bytes, `REQ-ISING-7244`, the exact
  V637 roadmap task, required imports, and writable checkpoint and result
  paths. They SHALL reject quarantined inputs before they read acceptance
  values. Only a dictionary with exactly `principle` and `value` keys MAY be
  unwrapped.
- The review SHALL use aggregation only. It SHALL set `MODEL_SPECS=[]`,
  `model_invoked=false`, `inference_substrate=aggregation_from_upstream_artifacts`,
  `inference_substrate_class=aggregation`, and `execution_venue=host`.
- `board_rows` SHALL contain one terminal row for KV260, GateMate, and
  PolarFire. Each row SHALL record its authenticated source path, date, hash,
  criterion, observed state, disposition, and exact next prerequisite.
- The KV260 row SHALL require both successful synthesis and its board-level
  programmable-logic latency transcript. It SHALL preserve graduation. A later
  KV260 task SHALL use only `ssh kria` as its access precondition.
- The PolarFire row SHALL authenticate the Exp7231 raw transcript, binary hash,
  input hash, output hash, and successful CPU dispatch. It SHALL state that this
  receipt is not FPGA sampling. The review SHALL not repeat the smoke.
- The GateMate search SHALL accept only an operator-authored physical-state
  receipt after Exp6559. It SHALL record the timestamp, changed cable, port, or
  power condition, and evidence hash. If no receipt exists, it SHALL record the
  explicit absence and `blocked_inherited_no_new_physical_state` at board-row
  scope.
- `hardware_operations_issued` SHALL be empty. The review SHALL issue no JTAG,
  USB reset, flash, power, device-storage, or board-access command.
- The review SHALL use the V637 memory footprint when it is available and
  authenticated. Missing footprint data SHALL not block board dispositions.
  `operation_map` SHALL separate measured CPU/Rust archive-mask, validation,
  and lookup work from prospective FPGA BRAM and TSU roles. Device topology,
  bandwidth, power, and latency SHALL remain unknown. Degree-16 connectivity
  SHALL not establish device fit.
- `board_disposition_complete_score` SHALL equal one when all three board rows
  and exact next conditions are present. A GateMate board-row block SHALL not
  block the completed review.
- Every board row SHALL contain `unit_id`, `arm`, `seed`, `metric`, `error`,
  and `abstention`. The sample budget SHALL preserve planned, attempted,
  completed, censored, and independent-unit counts.
- The producer SHALL write provisional state only under `results/checkpoints/`.
  It SHALL validate and atomically write the terminal artifact to
  `results/experiment_7244_v637_board_disposition.json`.

**Implementation status:** Planned (Exp7244)

### SCENARIO-ISING-7244-PREFLIGHT

**Authenticated fail-closed intake:** Given a missing source, changed task
contract, quarantine signal, invalid upstream hash, malformed wrapper, or
unwritable output, Exp7244 SHALL stop before disposition aggregation. It SHALL
emit the exact failed gate and no board row or hardware operation.

### SCENARIO-ISING-7244-BOARDS

**Independent board evidence:** Given authenticated Exp3709 and Exp7231 bytes,
Exp7244 SHALL preserve KV260 graduation and PolarFire CPU dispatch separately.
It SHALL not claim that the PolarFire transcript sampled programmable logic.

### SCENARIO-ISING-7244-GATEMATE

**Physical-state boundary:** Given no operator-authored GateMate physical-state
receipt after Exp6559, Exp7244 SHALL record explicit absence and zero hardware
commands. A later receipt SHALL only authorize a named future task.

### SCENARIO-ISING-7244-PLACEMENT

**Measured and prospective placement:** Given an authenticated V637 memory
receipt, Exp7244 SHALL retain its measured sizes. It SHALL leave FPGA and TSU
topology, bandwidth, power, and latency unknown.

### SCENARIO-ISING-7244-ARTIFACT

**Terminal read-only receipt:** Given three terminal board rows, Exp7244 SHALL
emit `board_disposition_complete_score=1` with an empty hardware-operation
list. The GateMate row MAY remain blocked while the top-level review is
complete.
