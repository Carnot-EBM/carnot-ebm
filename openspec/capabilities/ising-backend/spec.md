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
