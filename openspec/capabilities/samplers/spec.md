# Samplers Capability Specification

**Capability:** samplers
**Status:** Draft

## Requirements

### REQ-SAMPLE-1727: Equilibrium Matching Sampler

Carnot MUST provide a Python/JAX Equilibrium Matching sampler that uses online
equilibrium-gradient learning instead of Langevin diffusion noise when searching
for low-energy, constraint-satisfying states.

Sub-requirements:
- REQ-SAMPLE-1727-1: The sampler SHALL accept any Python `EnergyFunction` with
  `energy` and `grad_energy` methods.
- REQ-SAMPLE-1727-2: The sampler SHALL maintain a learned equilibrium gradient
  as an exponential moving average of clipped energy gradients.
- REQ-SAMPLE-1727-3: `sample()` SHALL return the final finite state after
  `n_steps`, and `sample_chain()` SHALL return an array of all intermediate
  states with shape `(n_steps, *init.shape)`.
- REQ-SAMPLE-1727-4: Optional gradient clipping SHALL preserve gradient
  direction while bounding the L2 norm.

### REQ-SAMPLE-1728: Experiment 1727 EqM Comparison Artifact

Experiment 1727 MUST compare Equilibrium Matching convergence against the
standard Langevin sampler on a deterministic soft-constraint energy problem and
write `results/experiment_1727_eqm.json`.

Sub-requirements:
- REQ-SAMPLE-1728-1: The artifact SHALL include the experiment id, spec refs,
  problem metadata, per-sampler convergence metrics, a speedup value, and an
  honest verdict.
- REQ-SAMPLE-1728-2: Per-sampler metrics SHALL include initial energy, final
  energy, first step at or below the configured threshold, convergence status,
  and finite-chain status.
- REQ-SAMPLE-1728-3: The comparison SHALL be deterministic for a fixed seed and
  SHALL not require GPU, network, or hardware backends.

### REQ-SAMPLE-1740: EqM GPU-Accelerated Backend

Carnot MUST provide an EqM backend path that places EqM state and gradients on a
selected JAX device and JIT-compiles the EqM update loop so live inference can
use CUDA/ROCm acceleration when a GPU backend is available.

Sub-requirements:
- REQ-SAMPLE-1740-1: The sampler SHALL expose `backend="auto" | "cpu" | "gpu"`
  and select a GPU device for `"auto"` only when JAX reports one.
- REQ-SAMPLE-1740-2: `backend="gpu"` SHALL place the EqM state on the selected
  GPU device and report an unavailable GPU as a clear runtime error.
- REQ-SAMPLE-1740-3: The accelerated update rule SHALL preserve the existing EqM
  CPU numerical behavior within floating-point tolerance.
- REQ-SAMPLE-1740-4: Experiment 1740 SHALL benchmark CPU versus GPU EqM update
  latency when a GPU is available, and SHALL write
  `results/experiment_1740_eqm_gpu.json` with honest backend availability and
  latency fields.

## Scenarios

### SCENARIO-SAMPLE-1727: EqM Converges On A Soft Constraint Bowl

**Given** a convex soft-constraint energy function with a known satisfying
target state
**When** Equilibrium Matching runs from a high-energy initialization
**Then** the chain remains finite
**And** the final energy is lower than the initial energy
**And** the returned final sample matches the last chain state.

### SCENARIO-SAMPLE-1728: Exp 1727 Writes A Terminal JSON Artifact

**Given** the Exp 1727 comparison runner
**When** it is executed on the deterministic benchmark
**Then** it writes a JSON artifact with EqM and Langevin convergence metrics
**And** records whether EqM reached the threshold in fewer steps.

### SCENARIO-SAMPLE-1740: EqM GPU Benchmark Writes Backend-Aware Artifact

**Given** the Exp 1740 EqM GPU benchmark runner
**When** it is executed on the local JAX installation
**Then** it writes a JSON artifact with CPU latency, GPU availability, selected
device metadata, and a backend verdict
**And** it records GPU latency and parity only when a GPU backend is available.

### REQ-SAMPLE-5611: Bounded cDLS Matched CPU/CUDA Crossover Benchmark

Carnot MUST provide Exp 5611 at
`python/carnot/experiment_5611_cdls_matched_sampler_crossover.py` and write
`results/experiment_5611_cdls_matched_sampler_crossover.json` without modifying
`scripts/research_conductor.py`. The experiment SHALL preserve the existing
discrete Langevin/heat-bath baseline from Exp 5573 unchanged and add a separately
identified bounded continuous-intermediate discrete Langevin sampler (cDLS).

The cDLS proposal SHALL use deterministic seeds, form a bounded continuous
intermediate for each discrete Ising state, project that intermediate back to
`{-1, +1}` spins, and apply a documented Metropolis-Hastings correction using
the exact discrete Ising energy and proposal probability so the projection does
not replace the exact discrete target distribution.

Exp 5611 SHALL use descriptor-derived Ising instances already available in
Carnot, with target descriptors recorded for `n=128,256,512,1024` where memory
and the bounded runtime permit execution. Discrete DLS and cDLS SHALL share the
same target energies, descriptor checksums, seeds, warm-up count, retained sample
count, temperature, thinning, precision, and timing boundaries across CPU and
CUDA. Successful matched rows SHALL retain at least 10000 post-warmup samples.
OOM, timeout, CUDA-blocked, descriptor-blocked, or runtime-budget-blocked rows
SHALL remain explicit and SHALL NOT enter speedup summaries.

The artifact SHALL define quality equivalence before inspecting timing. A
crossover claim SHALL be allowed only when successful matched cDLS/DLS CPU/CUDA
pairs pass the quality non-inferiority gate and the relevant timing interval
excludes 1.0 in the favorable direction. `crossover_size` MAY be null, and
`crossover_claim_allowed` SHALL be false when the quality or timing gate is not
conjunctively satisfied. No KV260, PolarFire, GateMate, or TSU speedup claim is
in scope, and `board_speedup_claimed` SHALL be false.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `target_descriptors`, `instance_sizes`, `methods`, `seeds`,
`samples_per_pair`, `cpu_device_receipt`, `cuda_device_receipt`, `timing_rows`,
`energy_quality_metrics`, `mixing_metrics`, `successful_matched_pairs`,
`speedup_by_pair`, `crossover_size`, `crossover_claim_allowed`,
`board_speedup_claimed`, `inference_substrate`, and `honest_verdict`.
`inference_substrate` SHALL equal `matched_cpu_cuda_exact_ising_sampling`.

Required field principles:

- `field_principles`: principle "Explains why every headline and gate field exists before a reviewer trusts the JSON shape."
- `target_descriptors`: principle "Proves both methods sample identical descriptor-derived Ising problems."
- `instance_sizes`: principle "Makes the tested crossover range explicit instead of implying unmeasured scale."
- `methods`: principle "Keeps the unchanged discrete DLS baseline and bounded cDLS proposal separately identifiable."
- `seeds`: principle "Records paired seeds so every CPU/CUDA and method comparison is reproducible."
- `samples_per_pair`: principle "Guards the >=10000 retained-sample floor required for mixing estimates."
- `cpu_device_receipt`: principle "Authenticates CPU identity, runtime, and free memory for the local benchmark."
- `cuda_device_receipt`: principle "Authenticates CUDA identity, driver/runtime, and free memory or records a precise blocker."
- `timing_rows`: principle "Preserves raw matched timing evidence, including failed rows, before any summary ratio."
- `energy_quality_metrics`: principle "Prevents speed from hiding wrong samples by reporting energy and exact constraint quality."
- `mixing_metrics`: principle "Reports ESS and autocorrelation to test whether the cDLS mechanism actually mixes."
- `successful_matched_pairs`: principle "Counts only complete method/device rows that satisfy the same schedule; quality gates are recorded before claims."
- `speedup_by_pair`: principle "Reports only matched ratios; failed or quality-inferior rows cannot enter speedups."
- `crossover_size`: principle "Records the smallest gated crossover size, or null when no crossover is proven."
- `crossover_claim_allowed`: principle "Requires quality and timing gates to pass together before any crossover claim."
- `board_speedup_claimed`: principle "Bare false prevents this CPU/CUDA sampler study from reopening board continuity."
- `inference_substrate`: principle "Declares matched CPU/CUDA exact Ising sampling, not LLM inference or board timing."
- `honest_verdict`: principle "Terminal complete: or blocked: verdict states whether no-crossover evidence is final."

### SCENARIO-SAMPLE-5611: cDLS Benchmark Emits Matched Evidence Or Blockers

**Given** the Exp 5556 descriptor output is available
**When** Exp 5611 runs on a host with CPU and CUDA available
**Then** it builds descriptor-derived Ising targets for the requested sizes,
runs unchanged discrete DLS and bounded cDLS under the same schedule and seeds,
writes raw timing rows for CPU and CUDA, reports acceptance, ESS, integrated
autocorrelation, energy distribution, best/mean energy, exact constraint
satisfaction, memory, compile/warmup time, sampling time, end-to-end wall time,
matched speedups, and a terminal artifact at
`results/experiment_5611_cdls_matched_sampler_crossover.json`
**And** sets `crossover_claim_allowed=true` only when the quality and timing
gates both pass, otherwise preserving the evidence with `crossover_size=null`
or a false crossover claim.

**If** CUDA, descriptors, memory, or bounded runtime are unavailable
**Then** Exp 5611 SHALL still write the same result path with explicit blocker
rows, `successful_matched_pairs` excluding failed rows,
`crossover_claim_allowed=false`, `board_speedup_claimed=false`, and an
`honest_verdict` starting with `blocked:` or `complete:` as appropriate for the
available matched evidence.

## Implementation Status (REQ-SAMPLE-5611)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5611 | Planned (`python/carnot/experiment_5611_cdls_matched_sampler_crossover.py`, `results/experiment_5611_cdls_matched_sampler_crossover.json`) | Planned (`tests/python/test_experiment_5611_cdls_matched_sampler_crossover.py`) |

### REQ-SAMPLE-5622: cDLS Exact Small-Ising Kernel Audit

Carnot MUST provide Exp 5622 at
`python/carnot/experiment_5622_cdls_exact_kernel_audit.py` and write
`results/experiment_5622_cdls_exact_kernel_audit.json` without modifying
`scripts/research_conductor.py`. The experiment SHALL audit the continuous-
intermediate discrete Langevin sampler against exactly enumerable finite-
temperature Ising targets before any additional CPU/CUDA timing run is allowed.

The audit SHALL preserve three separately named kernels: the existing discrete
Langevin/heat-bath baseline as `discrete_dls_heat_bath`, the deliberately
uncorrected continuous-intermediate projection as
`uncorrected_cdls_projection`, and the final corrected kernel as
`corrected_cdls_projection_mh`. For each exact small Ising system the audit
SHALL enumerate normalized target probabilities, build exact transition
matrices when feasible, and otherwise record a separately verified
high-precision estimate. The exact systems SHALL cover several sizes and
topologies, use finite temperature, and include nontrivial couplings and fields.

The verifier SHALL test row normalization, nonnegative probabilities,
transition support and irreducibility, reversibility or a declared invariant
condition, detailed-balance residual, stationary distribution total variation,
energy-histogram total variation, and deterministic seed replay. If the
uncorrected projection introduces target bias, the corrected cDLS kernel SHALL
use a mathematically justified Metropolis-Hastings projection correction based
only on the exact discrete Ising energy and exact projected proposal
probabilities; that correction SHALL NOT be tuned on any later large-n timing
set.

Empirical checks SHALL use at least five independent seeds and report
exact-versus-empirical total variation with uncertainty. Deliberately biased
and broken-proposal positive controls SHALL be included and SHALL be rejected
by the audit. The audit SHALL predeclare at least three large-n quality-
equivalence requirements covering target or energy behavior, mixing behavior,
and constraint validity. These preregistered fields, not a manually set success
boolean, SHALL gate Exp 5623.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `target_descriptors`, `models_tested`,
`state_space_sizes`, `transition_row_sum_error_max`,
`detailed_balance_residual_max`, `exact_distribution_tv_max`,
`empirical_distribution_intervals`, `broken_kernel_controls_rejected`,
`correction_applied`, `correction_spec`, `quality_gate_specification`,
`quality_gate_specified_count`, `kernel_audit_ready_score`,
`inference_substrate`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL equal
`deterministic_verifier`. `kernel_audit_ready_score` SHALL equal exactly `1.0`
only when the exactness gates pass for the corrected kernel and the biased and
broken controls are rejected; a biased final kernel SHALL block timing.

Required field principles:

- `field_principles`: principle "Explains why each audit field exists before Exp5623 can consume it."
- `target_descriptors`: principle "Makes every exact Ising system explicit enough to recompute the target distribution."
- `models_tested`: principle "Keeps the discrete baseline, biased cDLS control, and corrected cDLS kernel separately named."
- `state_space_sizes`: principle "Shows the exact enumeration limit and the number of states audited per topology."
- `transition_row_sum_error_max`: principle "Confirms every accepted transition matrix is normalized."
- `detailed_balance_residual_max`: principle "Measures reversibility or invariant-condition residual instead of assuming it."
- `exact_distribution_tv_max`: principle "Quantifies stationary-target parity for the final corrected kernel."
- `empirical_distribution_intervals`: principle "Bounds finite-sample exact-versus-empirical total variation across independent seeds."
- `broken_kernel_controls_rejected`: principle "Proves the audit rejects biased and broken positive controls."
- `correction_applied`: principle "Declares whether projection bias was corrected before any timing run."
- `correction_spec`: principle "Documents the exact acceptance rule and proposal probability used by the corrected kernel."
- `quality_gate_specification`: principle "Predeclares the large-n Exp5623 quality gates before timing evidence exists."
- `quality_gate_specified_count`: principle "Makes downstream gate coverage numeric rather than prose-only."
- `kernel_audit_ready_score`: principle "Equals 1.0 only when exactness and control-rejection gates pass."
- `inference_substrate`: principle "Declares that the artifact came from deterministic enumeration and simulation, not LLM judgment."
- `random_seeds`: principle "Records replay seeds for empirical checks."
- `reproducibility_checksum`: principle "Content-addresses the audit artifact for deterministic replay."
- `honest_verdict`: principle "States whether a biased kernel blocks timing."

### SCENARIO-SAMPLE-5622: Exact cDLS Kernel Audit Emits Gated Evidence

**Given** exactly enumerable finite-temperature Ising systems with nontrivial
couplings
**When** Exp 5622 runs
**Then** it enumerates each target distribution, audits exact transition
matrices for `discrete_dls_heat_bath`, `uncorrected_cdls_projection`, and
`corrected_cdls_projection_mh`, rejects deliberately biased and broken controls,
reports exact and empirical total variation, writes
`results/experiment_5622_cdls_exact_kernel_audit.json`, and sets
`kernel_audit_ready_score=1.0` only when the corrected kernel is target-exact
and all control gates pass.

## Implementation Status (REQ-SAMPLE-5622)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5622 | Planned (`python/carnot/experiment_5622_cdls_exact_kernel_audit.py`, `results/experiment_5622_cdls_exact_kernel_audit.json`) | Planned (`tests/python/test_experiment_5622_cdls_exact_kernel_audit.py`) |

### REQ-SAMPLE-5623: Quality-Matched Multi-Seed Corrected-cDLS CPU/CUDA Crossover

Carnot MUST provide Exp 5623 at
`python/carnot/experiment_5623_cdls_multiseed_cpu_cuda_crossover.py` and write
`results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json` without
modifying `scripts/research_conductor.py`. The experiment SHALL run only after
Exp 5622's corrected-kernel exactness gate is authenticated from
`results/experiment_5622_cdls_exact_kernel_audit.json`; if that upstream gate is
missing, stale, unparseable, checksum-invalid, or not ready, Exp 5623 SHALL emit
blocked rows rather than allocating CUDA work.

The benchmark SHALL preserve the unchanged discrete Langevin/heat-bath baseline
and compare it only with Exp 5622's corrected final kernel named
`corrected_cdls_projection_mh`. It SHALL NOT substitute
`uncorrected_cdls_projection` or any biased control kernel. The tested targets
SHALL be deterministic exact Ising Hamiltonians at `n=128,256,512,1024` where
memory permits. Both methods and both devices SHALL share target descriptors,
seeds, temperature, warmup, retained sample count, thinning, schedules,
precision, and measurement boundaries. At least five paired seeds SHALL be
recorded, and successful matched rows SHALL retain at least 10000 post-warmup
samples. Each successful retained row therefore has at least 10000 post-warmup
samples before it can enter quality or speed summaries. OOM, timeout,
CUDA-blocked, upstream-gate-blocked, memory-blocked, and failed-quality rows
SHALL remain explicit timing rows and SHALL NOT enter speedup summaries.
In short, no successful matched row may enter a claim without at least 10000 post-warmup samples.

Before timing can support any crossover claim, Exp 5623 SHALL apply Exp 5622's
preregistered quality-equivalence rules to acceptance, effective sample size,
integrated autocorrelation, energy-distribution distance, best and mean energy,
exact constraint satisfaction, and uncertainty. Each size/seed pair SHALL have
an inclusion or exclusion explanation. The experiment SHALL compute paired
CPU/CUDA speedups and corrected-cDLS/discrete-DLS speedups with intervals across
paired seeds. `crossover_claim_allowed=true` SHALL be allowed only when a
successful quality-matched pair set has a timing interval excluding 1.0 in the
favorable direction. `crossover_size` MAY be null. No physical board or TSU
speedup claim is in scope, and `board_speedup_claimed` SHALL be false.

The experiment SHALL preserve raw samples or stable sufficient-statistic files
so ESS, integrated autocorrelation, energy summaries, exact constraint
satisfaction, and timing summaries can be recomputed independently. The terminal
artifact SHALL include at minimum these top-level fields: `field_principles`,
`upstream_gate_receipt`, `target_descriptors`, `instance_sizes`,
`models_tested`, `seeds`, `samples_per_pair`, `cpu_device_receipt`,
`cuda_device_receipt`, `timing_rows`, `energy_quality_metrics`,
`mixing_metrics`, `quality_gate_results_by_pair`,
`successful_matched_pairs`, `speedup_by_pair`, `timing_intervals_by_size`,
`sufficient_statistics_path`, `sufficient_statistics_sha256`,
`crossover_size`, `crossover_claim_allowed`, `board_speedup_claimed`,
`inference_substrate`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`.
`inference_substrate` SHALL equal `matched_cpu_cuda_exact_ising_sampling`.

Required field principles:

- `field_principles`: principle "Explains why every headline and gate field exists before a reviewer trusts the JSON shape."
- `upstream_gate_receipt`: principle "Proves Exp5622 exactness prerequisites are fixed before GPU work is allocated."
- `target_descriptors`: principle "Proves both methods sample identical exact Ising Hamiltonians."
- `instance_sizes`: principle "Makes the tested crossover range explicit instead of implying unmeasured scale."
- `models_tested`: principle "Names sampler methods only, keeping the unchanged discrete baseline and corrected cDLS kernel separate with no LLM implied."
- `seeds`: principle "Records at least five paired seeds so every CPU/CUDA and method comparison is reproducible."
- `samples_per_pair`: principle "Guards the >=10000 retained-sample floor required for mixing estimates."
- `cpu_device_receipt`: principle "Authenticates CPU identity, runtime, and free memory for the local benchmark."
- `cuda_device_receipt`: principle "Authenticates CUDA identity, driver/runtime, and free memory or records a precise blocker."
- `timing_rows`: principle "Preserves raw matched timing evidence, including failed rows, before any summary ratio."
- `energy_quality_metrics`: principle "Prevents speed from hiding wrong samples by reporting energy and exact constraint quality."
- `mixing_metrics`: principle "Reports ESS and autocorrelation to test whether the corrected cDLS mechanism actually mixes."
- `quality_gate_results_by_pair`: principle "Explains every pair's inclusion or exclusion before timing enters a speedup claim."
- `successful_matched_pairs`: principle "Lists only quality-matched complete method/device pairs; failed rows do not enter speedups."
- `speedup_by_pair`: principle "Reports only matched ratios with intervals; failed or quality-inferior rows cannot enter speedups."
- `timing_intervals_by_size`: principle "Aggregates paired-seed ratios into intervals so a crossover cannot rest on one lucky row."
- `sufficient_statistics_path`: principle "Points to recomputable energy, constraint, and timing evidence instead of trusting summaries."
- `sufficient_statistics_sha256`: principle "Pins the sufficient-statistic file used to recompute ESS, autocorrelation, quality, and timing."
- `crossover_size`: principle "Records the smallest gated crossover size, or null when no crossover is proven."
- `crossover_claim_allowed`: principle "Requires quality and timing gates to pass together before any crossover claim."
- `board_speedup_claimed`: principle "Bare false prevents this CPU/CUDA sampler study from reopening board or TSU scope."
- `inference_substrate`: principle "Declares matched CPU/CUDA exact Ising sampling, not LLM inference or board timing."
- `random_seeds`: principle "Duplicates the paired seeds in methodology form for verifier compatibility."
- `reproducibility_checksum`: principle "Content-addresses the benchmark artifact and sufficient-statistic receipt."
- `honest_verdict`: principle "Terminal complete: or blocked: verdict states whether no-crossover evidence is final."

### SCENARIO-SAMPLE-5623: Corrected-cDLS Multi-Seed Crossover Emits Gated Evidence Or Blockers

**Given** Exp 5622's corrected-kernel exact audit is ready
**When** Exp 5623 runs on a host with CPU and CUDA available
**Then** it builds deterministic exact Ising targets for `n=128,256,512,1024`,
runs unchanged discrete DLS and `corrected_cdls_projection_mh` under the same
schedule and at least five paired seeds, writes raw timing rows and recomputable
sufficient-statistic evidence, applies quality gates before speedups, and writes
`results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json`
**And** sets `crossover_claim_allowed=true` only when the quality and timing
interval gates both pass.

**If** Exp 5622 readiness, CUDA, memory, or runtime is unavailable
**Then** Exp 5623 SHALL still write the same result path with explicit blocker
rows, `successful_matched_pairs` excluding failed rows,
`crossover_claim_allowed=false`, `board_speedup_claimed=false`, and an
`honest_verdict` starting with `blocked:` or `complete:` as appropriate for the
available matched evidence.

## Implementation Status (REQ-SAMPLE-5623)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5623 | Planned (`python/carnot/experiment_5623_cdls_multiseed_cpu_cuda_crossover.py`, `results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json`) | Planned (`tests/python/test_experiment_5623_cdls_multiseed_cpu_cuda_crossover.py`) |

### REQ-SAMPLE-5633: Exact Fixed-Ladder Temperature-Label Exchange cDLS Audit

Carnot MUST provide Exp 5633 at
`python/carnot/experiment_5633_temperature_exchange_cdls_exact_audit.py` and
write `results/experiment_5633_temperature_exchange_cdls_exact_audit.json`
without modifying `scripts/research_conductor.py`. The experiment SHALL add a
fixed beta-ladder temperature-label exchange layer around Exp 5622's final
`corrected_cdls_projection_mh` kernel only. The within-replica cDLS acceptance
rule, proposal probability, drift scale, proposal standard deviation, and
source module/hash receipt SHALL be inherited from Exp 5622 unchanged; Exp 5633
SHALL NOT retune or replace the corrected within-replica kernel.

The experiment SHALL preregister a fixed beta ladder, a within-replica
transition schedule, an adjacent-pair exchange schedule, and the exact exchange
acceptance rule. Exchange moves SHALL swap temperature labels attached to
physical replicas while leaving all replica states unchanged. For states
`x_a,x_b` currently wearing adjacent beta labels `beta_a,beta_b`, the accepted
label swap probability SHALL be
`min(1, exp((beta_a - beta_b) * (E(x_a) - E(x_b))))`. The artifact SHALL also
include equal-transition single-chain and independent-replica no-exchange
baselines so any later quality trial can separate the exchange layer from the
unchanged corrected cDLS substrate.

On exactly enumerable frustrated Ising targets, Exp 5633 SHALL enumerate exact
target marginals and energy distributions for every beta in the fixed ladder.
It SHALL verify within-kernel normalization, swap detailed balance,
product-target stationarity under the composed fixed schedule, cold-label
target parity, deterministic replay, and round-trip label accounting. The audit
SHALL include broken controls for missing beta factors, wrong energy sign,
state-copy swap, asynchronous stale energy, one-way exchange, and biased proposal schedules. Every broken control SHALL be detected before
`replica_exchange_kernel_ready_score` can equal `1.0`.

`replica_exchange_kernel_ready_score` SHALL equal exactly `1.0` only when
`exact_distribution_tv_max <= 0.02`,
`swap_detailed_balance_residual_max <= 1e-6`, transition normalization and
deterministic replay pass, every broken control is rejected, round-trip label
accounting is valid, and no validity regression appears. Exp 5633 SHALL collect
no timing, CUDA, board, SNN, TSU, or crossover result. `timing_claimed` SHALL be false and `hardware_speedup_claimed` SHALL be false.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `corrected_kernel_receipt`, `beta_ladder`,
`within_replica_schedule`, `exchange_schedule`, `swap_rule`,
`enumerable_targets`, `transition_normalization_error_max`,
`swap_detailed_balance_residual_max`, `exact_distribution_tv_max`,
`cold_replica_energy_error`, `round_trip_accounting_error`,
`broken_controls`, `deterministic_replay_pass`, `timing_claimed`,
`hardware_speedup_claimed`, `replica_exchange_kernel_ready_score`,
`inference_substrate`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL equal
`exact_corrected_cdls_with_replica_temperature_exchange`.

Required field principles:

- `field_principles`: principle "Explains why each audit and gate field exists before any exchange quality trial can consume it."
- `corrected_kernel_receipt`: principle "Proves the exact Exp5622 corrected within-replica substrate is unchanged and source-hash pinned."
- `beta_ladder`: principle "Freezes the temperature scope so later quality differences cannot come from an unreported ladder search."
- `within_replica_schedule`: principle "Records the exact corrected-cDLS transition accounting applied at each physical replica."
- `exchange_schedule`: principle "Records which adjacent beta-label pairs are proposed and in what fixed order."
- `swap_rule`: principle "Makes the exact label-exchange acceptance ratio inspectable."
- `enumerable_targets`: principle "Preserves deterministic Ising ground truth, marginals, and energy distributions for replay."
- `transition_normalization_error_max`: principle "Confirms the within-replica and exchange kernels are stochastic before stationarity is trusted."
- `swap_detailed_balance_residual_max`: principle "Measures reversibility of the temperature-label exchange instead of assuming it."
- `exact_distribution_tv_max`: principle "Measures exact product-target and cold-label parity for the composed exchange sampler."
- `cold_replica_energy_error`: principle "Reports target-temperature energy parity explicitly instead of hiding it in aggregate TV."
- `round_trip_accounting_error`: principle "Checks that label movement is accounted for without copying or losing temperature labels."
- `broken_controls`: principle "Demonstrates the audit rejects biased exchange variants that would otherwise look superficially normalized."
- `deterministic_replay_pass`: principle "Proves fixed seeds reproduce the same label-exchange trace."
- `timing_claimed`: principle "Bare false keeps Exp5633 an exactness audit and does not reopen retired crossover timing."
- `hardware_speedup_claimed`: principle "Bare false prevents CUDA, board, SNN, TSU, or hardware inference from entering this artifact."
- `replica_exchange_kernel_ready_score`: principle "Provides a mechanical downstream gate that is 1.0 only when exactness and control-rejection gates pass."
- `inference_substrate`: principle "Declares exact corrected cDLS plus replica temperature-label exchange, not LLM inference or hardware timing."
- `random_seeds`: principle "Records replay seeds for deterministic trace and accounting checks."
- `reproducibility_checksum`: principle "Content-addresses the artifact so future reruns can detect silent drift."
- `honest_verdict`: principle "States whether invariant failure blocks later quality trials."

### SCENARIO-SAMPLE-5633: Temperature-Label Exchange Emits Exact Audit Evidence

**Given** Exp 5622's corrected `corrected_cdls_projection_mh` kernel and
exactly enumerable frustrated Ising systems
**When** Exp 5633 runs
**Then** it builds fixed-ladder corrected-cDLS within-replica kernels, composes
them with fixed adjacent temperature-label exchanges, verifies normalization,
swap detailed balance, product-target stationarity, cold-label energy parity,
deterministic replay, round-trip accounting, and all broken-control rejections,
writes `results/experiment_5633_temperature_exchange_cdls_exact_audit.json`,
and sets `replica_exchange_kernel_ready_score=1.0` only when every exactness
gate passes.

**If** the corrected kernel receipt is missing, stale, changed, or any invariant
or broken-control gate fails
**Then** Exp 5633 SHALL still emit the same result path with terminal evidence,
`timing_claimed=false`, `hardware_speedup_claimed=false`,
`replica_exchange_kernel_ready_score=0.0`, and an `honest_verdict` starting
with `blocked:`.

## Implementation Status (REQ-SAMPLE-5633)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5633 | Implemented (`python/carnot/experiment_5633_temperature_exchange_cdls_exact_audit.py`, `results/experiment_5633_temperature_exchange_cdls_exact_audit.json`) | Implemented (`tests/python/test_experiment_5633_temperature_exchange_cdls_exact_audit.py`) |

### REQ-SAMPLE-5634: Paired Equal-Transition Temperature-Exchange cDLS Quality Trial

Carnot MUST provide Exp 5634 at
`python/carnot/experiment_5634_temperature_exchange_cdls_quality.py` and write
`results/experiment_5634_temperature_exchange_cdls_quality.json` without
modifying `scripts/research_conductor.py`. The experiment SHALL run only after
Exp 5622 proves the corrected cDLS kernel exact and Exp 5633 proves the
fixed-ladder temperature-label exchange kernel exact. Exp 5623 timing evidence
SHALL NOT be imported as a speedup claim; wall time MAY be reported only as
provenance.

The trial SHALL preregister a frozen panel of frustrated Ising instances and
exact-verifier CSP encodings with stated energy barriers or metastability,
cover at least two size strata, use at least five paired seeds, and use a fixed
total corrected-kernel transition budget before observing any treatment result.
This SHALL be recorded as the fixed total corrected-kernel transition budget.
The compared arms SHALL be `temperature_exchange_cdls`,
`independent_corrected_cdls_replicas`, and `single_corrected_cold_chain`.
Initialization, beta ladder where applicable, cold-target retained samples,
corrected-kernel transition proposals, exact-validation calls, and stopping
rules SHALL be matched by instance and seed. Exchange proposals MAY be counted
separately, but they SHALL NOT replace the equal corrected-kernel transition
budget.

The artifact SHALL report round trips, accepted exchanges, barrier crossings,
effective sample size, integrated autocorrelation, energy distribution
diagnostics, best energy, mean energy, exact constraint satisfaction, solve
probability, invalid rate, and paired confidence intervals by instance and size.
It SHALL include a beta-ladder ablation, disabled-exchange control,
label-shuffle diagnostic, transition-budget audit, seed-order permutation, and
exact-verifier replay. Instances, seeds, burn-in, and endpoints SHALL remain
fixed after preregistration.

`quality_mixing_ready` SHALL equal true only when at least one preregistered
quality or mixing endpoint improves for `temperature_exchange_cdls` versus both
baselines with a paired interval excluding zero, no primary endpoint materially
worsens, exact validity does not regress, and target diagnostics remain within
Exp 5633 bounds. `hardware_speedup_claimed` SHALL be false, `timing_claimed`
SHALL be false, and `inference_substrate` SHALL equal
`paired_exact_corrected_cdls_replica_sampling`.
`timing_claimed` SHALL be false.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `upstream_gate_receipts`, `instance_panel`,
`paired_seed_schedule`, `transition_budget_receipt`, `method_arms`,
`round_trip_stats`, `barrier_crossing_stats`, `ess_by_arm`,
`autocorrelation_by_arm`, `energy_distribution_diagnostics`,
`best_energy_by_arm`, `mean_energy_by_arm`, `solve_probability_by_arm`,
`exact_valid_rate_by_arm`, `paired_deltas_and_intervals`,
`wall_time_provenance_only`, `hardware_speedup_claimed`, `timing_claimed`,
`quality_mixing_ready`, `inference_substrate`, `random_seeds`,
`reproducibility_checksum`, and `honest_verdict`.

Required field principles:

- `field_principles`: principle "Explains why every required artifact field exists before reviewer or conductor consumption."
- `upstream_gate_receipts`: principle "Proves Exp5622 corrected-kernel exactness and Exp5633 exchange exactness are fixed prerequisites."
- `instance_panel`: principle "Freezes the hard-instance difficulty scope before any arm result can select the panel."
- `paired_seed_schedule`: principle "Matches every comparison by seed so deltas are paired rather than cherry-picked."
- `transition_budget_receipt`: principle "Proves algorithmic work is equal under the fixed corrected-kernel transition budget."
- `method_arms`: principle "Names the exchange treatment and both corrected-cDLS baselines explicitly."
- `round_trip_stats`: principle "Measures whether temperature labels actually move through the ladder."
- `barrier_crossing_stats`: principle "Measures the proposed mechanism directly on metastable basin changes."
- `ess_by_arm`: principle "Quantifies usable cold-target sample count instead of raw sample volume."
- `autocorrelation_by_arm`: principle "Quantifies serial dependence so mixing claims cannot hide slow chains."
- `energy_distribution_diagnostics`: principle "Makes quality matching visible through comparable energy sufficient statistics."
- `best_energy_by_arm`: principle "Reports optimization quality explicitly."
- `mean_energy_by_arm`: principle "Reports target-quality central tendency explicitly."
- `solve_probability_by_arm`: principle "Measures exact-constraint utility on CSP and Ising verifier targets."
- `exact_valid_rate_by_arm`: principle "Prevents invalid samples from winning a quality gate."
- `paired_deltas_and_intervals`: principle "Reports uncertainty from paired seeds by instance and size."
- `wall_time_provenance_only`: principle "Records elapsed time only as provenance and blocks any speedup interpretation."
- `hardware_speedup_claimed`: principle "Bare false keeps hardware and board timing scopes closed."
- `timing_claimed`: principle "Bare false keeps Exp5623 timing mismatch from being laundered into this trial."
- `quality_mixing_ready`: principle "Makes capstone promotion mechanical instead of manually declared."
- `inference_substrate`: principle "Declares paired exact corrected cDLS replica sampling with no LLM inference."
- `random_seeds`: principle "Records the paired seeds needed to replay the trial."
- `reproducibility_checksum`: principle "Content-addresses the preregistered panel, seeds, budget, controls, and summaries."
- `honest_verdict`: principle "States whether the repeated quality/mixing mismatch retires the hybrid."

### SCENARIO-SAMPLE-5634: Temperature-Exchange Quality Trial Emits Paired Evidence

**Given** Exp 5622 and Exp 5633 exactness gates are ready
**When** Exp 5634 runs on the frozen hard-instance panel
**Then** it compares temperature exchange, independent corrected replicas, and
a corrected cold chain under matched paired seeds and equal corrected-kernel
transition budgets, writes
`results/experiment_5634_temperature_exchange_cdls_quality.json`, reports the
required quality, mixing, validity, control, and uncertainty fields, and sets
`quality_mixing_ready=true` only by the mechanical promotion gate.

**If** upstream exactness gates fail, budgets are unequal, controls fail, exact
validity regresses, or no preregistered endpoint improves with an interval
excluding zero
**Then** Exp 5634 SHALL still emit the same result path with
`quality_mixing_ready=false`, `hardware_speedup_claimed=false`,
`timing_claimed=false`, and an `honest_verdict` starting with `blocked:` or
`complete:` as appropriate for terminal evidence.

## Implementation Status (REQ-SAMPLE-5634)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5634 | Planned (`python/carnot/experiment_5634_temperature_exchange_cdls_quality.py`, `results/experiment_5634_temperature_exchange_cdls_quality.json`) | Planned (`tests/python/test_experiment_5634_temperature_exchange_cdls_quality.py`) |

### REQ-SAMPLE-5644: Exact Two-Axis Temperature-By-Penalty Label-Exchange Audit

Carnot MUST provide Exp 5644 at
`python/carnot/experiment_5644_two_axis_parallel_tempering_exact_audit.py` and
write `results/experiment_5644_two_axis_parallel_tempering_exact_audit.json`
without modifying `scripts/research_conductor.py`. The experiment SHALL add a
rectangular inverse-temperature by nonnegative constraint-penalty label-exchange
layer around Exp 5622's final `corrected_cdls_projection_mh` kernel only. Exp
5644 SHALL reuse Exp 5622's corrected within-replica cDLS proposal and Exp
5633's exact label-swap semantics unchanged: exchange moves swap parameter labels
attached to physical replicas and SHALL NOT copy or swap replica states.

The target for slot `(beta, lambda)` SHALL be proportional to
`exp(-beta * (E(x) + lambda * C(x)))`, where `E(x)` is the exact Ising energy
and `C(x)` is a nonnegative exact constraint-violation penalty. The audit SHALL
freeze a small rectangular ladder over beta and lambda before running. For a
horizontal adjacent temperature-label exchange at common lambda, states
`x_a,x_b` currently wearing beta labels `beta_a,beta_b` SHALL accept with
`min(1, exp((beta_a - beta_b) * ((E(x_a) + lambda*C(x_a)) - (E(x_b) + lambda*C(x_b)))))`.
For a vertical adjacent penalty-label exchange at common beta, states `x_a,x_b`
currently wearing penalty labels `lambda_a,lambda_b` SHALL accept with
`min(1, exp(beta * (lambda_a - lambda_b) * (C(x_a) - C(x_b))))`.

On at least three exactly enumerable constrained Ising fixtures with multiple
energy and feasibility barriers, Exp 5644 SHALL enumerate the exact joint label
and state target and the target-replica distribution at the strongest beta and
lambda. It SHALL verify transition row normalization, nonnegative transition
probabilities, horizontal and vertical detailed balance, fixed-schedule
stationary parity, target-replica total variation, target feasibility marginal
parity, and deterministic replay. It SHALL compare exactness only against the
promoted one-axis exchange and equal-transition independent corrected chains.
It SHALL include broken controls for missing penalty terms, sign reversal,
state swapping, asymmetric scheduling, extreme lambda, and disabled swaps.
Every broken control SHALL be detected before `two_axis_invariant_ready_score`
can equal `1.0`.

`two_axis_invariant_ready_score` SHALL equal exactly `1.0` only when all
enumerated target total-variation and detailed-balance errors meet
preregistered tolerances, the target-replica feasibility marginal matches the
exact value, deterministic replay passes, every broken control is rejected, and
no timing or hardware speedup is claimed. `timing_claimed` SHALL be false,
`hardware_speedup_claimed` SHALL be false, and `inference_substrate` SHALL
equal `cpu_exact_enumeration_and_corrected_cdls`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `upstream_kernel_receipts`,
`openspec_requirement_ids`, `fixture_definitions`, `temperature_ladder`,
`penalty_ladder`, `horizontal_swap_rule`, `vertical_swap_rule`, `scheduler`,
`transition_row_error_max`, `horizontal_detailed_balance_error_max`,
`vertical_detailed_balance_error_max`, `exact_joint_target_tv`,
`exact_target_replica_tv`, `target_feasibility_marginal_error`,
`deterministic_replay_pass`, `broken_controls`,
`broken_control_rejected`, `timing_claimed`, `hardware_speedup_claimed`,
`two_axis_invariant_ready_score`, `inference_substrate`, `random_seeds`,
`reproducibility_checksum`, and `honest_verdict`.

Required field principles:

- `field_principles`: principle "Explains why every required audit field exists before any two-axis quality trial can consume it."
- `upstream_kernel_receipts`: principle "Pins Exp5622 corrected cDLS and Exp5633 exact label semantics so Exp5644 cannot silently retune substrates."
- `openspec_requirement_ids`: principle "Keeps the implementation and tests anchored to REQ-SAMPLE-5644 and SCENARIO-SAMPLE-5644."
- `fixture_definitions`: principle "Makes every constrained Ising target reconstructable, including energies, constraints, and feasibility barriers."
- `temperature_ladder`: principle "Freezes the beta axis before exactness is measured."
- `penalty_ladder`: principle "Freezes the nonnegative constraint-penalty axis before exactness is measured."
- `horizontal_swap_rule`: principle "Makes the temperature-label acceptance ratio inspectable at fixed penalty."
- `vertical_swap_rule`: principle "Makes the penalty-label acceptance ratio inspectable at fixed temperature."
- `scheduler`: principle "Records the fixed within, horizontal, and vertical update order required for deterministic replay."
- `transition_row_error_max`: principle "Confirms every composed transition is stochastic before stationary parity is trusted."
- `horizontal_detailed_balance_error_max`: principle "Measures exact reversibility along the temperature axis."
- `vertical_detailed_balance_error_max`: principle "Measures exact reversibility along the penalty axis."
- `exact_joint_target_tv`: principle "Measures full joint state and label stationary parity instead of checking only marginals."
- `exact_target_replica_tv`: principle "Measures the strongest-beta strongest-penalty target replica that downstream constrained optimization would consume."
- `target_feasibility_marginal_error`: principle "Checks that exact constraint feasibility is preserved by the two-axis invariant."
- `deterministic_replay_pass`: principle "Proves fixed seeds reproduce the same schedule, labels, and RNG trace."
- `broken_controls`: principle "Documents invalid two-axis kernels that the audit must reject."
- `broken_control_rejected`: principle "Provides a single mechanical gate proving every broken control was detected."
- `timing_claimed`: principle "Bare false keeps this exactness audit from becoming a timing or crossover claim."
- `hardware_speedup_claimed`: principle "Bare false prevents CPU enumeration from being read as CUDA, board, SNN, or TSU evidence."
- `two_axis_invariant_ready_score`: principle "Equals 1.0 only when exactness, feasibility, replay, and control-rejection gates all pass."
- `inference_substrate`: principle "Declares CPU exact enumeration plus corrected cDLS, not LLM inference or board execution."
- `random_seeds`: principle "Records deterministic replay seeds."
- `reproducibility_checksum`: principle "Content-addresses the audit so future reruns can detect silent drift."
- `honest_verdict`: principle "Starts complete: or blocked: and treats exactness failure as terminal."

### SCENARIO-SAMPLE-5644: Two-Axis Label Exchange Emits Exact Audit Evidence

**Given** Exp 5622's corrected `corrected_cdls_projection_mh` kernel, Exp 5633's
label-only exchange semantics, and exactly enumerable constrained Ising fixtures
**When** Exp 5644 runs a fixed rectangular beta/lambda label-exchange audit
**Then** it enumerates exact joint and target-replica distributions, verifies
normalization, nonnegativity, horizontal and vertical detailed balance,
stationary parity, target feasibility marginals, deterministic replay, promoted
one-axis and independent-chain exactness baselines, and all broken-control
rejections, writes
`results/experiment_5644_two_axis_parallel_tempering_exact_audit.json`, and
sets `two_axis_invariant_ready_score=1.0` only when every exactness gate passes.

**If** any substrate receipt, invariant, feasibility marginal, replay, or
broken-control gate fails
**Then** Exp 5644 SHALL still emit the same result path with
`timing_claimed=false`, `hardware_speedup_claimed=false`,
`two_axis_invariant_ready_score=0.0`, and an `honest_verdict` starting with
`blocked:`.

## Implementation Status (REQ-SAMPLE-5644)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5644 | Planned (`python/carnot/experiment_5644_two_axis_parallel_tempering_exact_audit.py`, `results/experiment_5644_two_axis_parallel_tempering_exact_audit.json`) | Planned (`tests/python/test_experiment_5644_two_axis_parallel_tempering_exact_audit.py`) |

### REQ-SAMPLE-5645: Two-Axis Tempering Hard-Constraint Quality Trial

Carnot MUST provide Exp 5645 at
`python/carnot/experiment_5645_two_axis_tempering_hard_constraint_quality.py`
and write
`results/experiment_5645_two_axis_tempering_hard_constraint_quality.json`
without modifying `scripts/research_conductor.py`. The trial SHALL run only
after Exp 5644 proves the exact two-axis invariant, Exp 5634 promotes the
one-axis temperature-exchange quality gate, and Exp 5622 proves the corrected
cDLS kernel exact. The experiment is a matched quality and feasibility trial,
not a speed benchmark; `timing_claimed` SHALL be false,
`hardware_speedup_claimed` SHALL be false, and `inference_substrate` SHALL
equal `cpu_two_axis_corrected_cdls_replica_exchange`.

Before execution, Exp 5645 SHALL preregister at least two constrained
Ising/CSP families, immutable instance hashes, sizes, exact violation
penalty definitions, beta and penalty ladders, burn-in, sampling schedule,
total corrected-kernel transition budget, at least five paired seeds, and
promotion thresholds. The three primary arms SHALL be
`two_axis_tempering`, `one_axis_temperature_exchange`, and
`independent_corrected_cdls`. The total within-replica corrected-kernel
proposal budget SHALL match by arm, instance, and seed; swap proposals MAY be
reported separately but SHALL NOT replace within-kernel work. Outcome-driven
tuning after inspecting test outcomes SHALL be excluded by the preregistered
protocol.

The artifact SHALL report exact constraint-validity checks, feasible-hit rate,
violation magnitude, time-to-first-feasible in corrected-kernel transition
counts, best and mean feasible energy, temperature and penalty round trips,
barrier crossings, effective sample size, integrated autocorrelation,
energy/violation distributions, solve probability, and paired bootstrap or
randomization intervals. It SHALL include disabled-penalty-swap,
collapsed-ladder, shuffled-label, fixed-weak-penalty, fixed-strong-penalty,
and invalid-state controls. Zero-solve instances SHALL be separated from
invalid execution and all denominators SHALL be published through
`successful_seed_count` and `failed_seed_reasons`.

`two_axis_quality_ready_score` SHALL equal exactly `1.0` only when the
two-axis arm improves a preregistered feasibility or mixing primary metric
over one-axis exchange with an interval excluding zero, exact validity does
not materially regress, target diagnostics and feasible energy do not regress,
all required seeds and families succeed, controls pass, transition budgets
match, upstream exactness eligibility is explicit, and no timing or hardware
claim is made. If any gate fails, Exp 5645 SHALL still emit the same result
path with terminal evidence and an `honest_verdict` starting with `complete:`
or `blocked:`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `upstream_gate_receipts`, `preregistered_protocol`,
`instance_manifest`, `instance_hashes`, `sampler_configs`,
`transition_budget_parity`, `successful_seed_count`, `failed_seed_reasons`,
`constraint_validity_by_arm`, `feasible_hit_rate_by_arm`,
`violation_distribution_by_arm`, `first_feasible_transition_by_arm`,
`temperature_round_trips`, `penalty_round_trips`, `barrier_crossings_by_arm`,
`ess_by_arm`, `autocorrelation_by_arm`, `feasible_energy_by_arm`,
`solve_probability_by_arm`, `paired_intervals`,
`material_quality_regression_count`, `timing_claimed`,
`hardware_speedup_claimed`, `two_axis_quality_ready_score`,
`inference_substrate`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`.

Required field principles:

- `field_principles`: principle "Explains why every required quality field exists before the artifact can promote a two-axis sampler."
- `upstream_gate_receipts`: principle "Makes exactness eligibility explicit by pinning Exp5622, Exp5634, and Exp5644 readiness before quality evidence is interpreted."
- `preregistered_protocol`: principle "Freezes families, ladders, schedules, budgets, seeds, metrics, controls, and promotion thresholds before outcomes are inspected."
- `instance_manifest`: principle "Publishes the immutable constrained workloads, sizes, families, and exact penalty definitions used by every arm."
- `instance_hashes`: principle "Content-addresses each workload so later reruns cannot silently change the hard-instance panel."
- `sampler_configs`: principle "Makes every arm reconstructable, including fixed ladders, swaps, burn-in, sampling, and corrected-kernel settings."
- `transition_budget_parity`: principle "Proves within-replica corrected-kernel proposals are matched while swap work is accounted for separately."
- `successful_seed_count`: principle "Publishes the denominator of paired seeds that produced valid execution rows."
- `failed_seed_reasons`: principle "Separates zero solves from invalid execution instead of hiding failed rows."
- `constraint_validity_by_arm`: principle "Treats exact feasibility checks as authoritative for constrained quality."
- `feasible_hit_rate_by_arm`: principle "Uses constraint feasibility as the primary utility metric for hard constrained instances."
- `violation_distribution_by_arm`: principle "Keeps near-feasible violations visible instead of compressing them to pass/fail."
- `first_feasible_transition_by_arm`: principle "Measures discovery cost in transition counts and avoids wall-time laundering."
- `temperature_round_trips`: principle "Shows whether the temperature axis actually mixes through its ladder."
- `penalty_round_trips`: principle "Shows whether the constraint-penalty axis actually mixes through its ladder."
- `barrier_crossings_by_arm`: principle "Measures metastable basin movement as the proposed mechanism rather than inferring it from final energy."
- `ess_by_arm`: principle "Reports usable sample count for the retained cold constrained target."
- `autocorrelation_by_arm`: principle "Reports serial dependence so mixing gains are explicit."
- `feasible_energy_by_arm`: principle "Checks that feasibility gains do not degrade feasible optimization quality."
- `solve_probability_by_arm`: principle "Bounds success probability over paired hard instances and seeds."
- `paired_intervals`: principle "Reports uncertainty for preregistered two-axis versus baseline comparisons."
- `material_quality_regression_count`: principle "Requires zero material validity, diagnostic, or feasible-energy regressions before promotion."
- `timing_claimed`: principle "Bare false keeps the trial scoped to quality rather than speed."
- `hardware_speedup_claimed`: principle "Bare false prevents CPU evidence from becoming a board or hardware claim."
- `two_axis_quality_ready_score`: principle "Provides a scalar downstream gate that is 1.0 only under the preregistered quality promotion rule."
- `inference_substrate`: principle "Declares CPU corrected-cDLS two-axis replica exchange with no LLM or board participation."
- `random_seeds`: principle "Records paired seeds for replay."
- `reproducibility_checksum`: principle "Content-addresses the full trial payload after the self-checksum field is blanked."
- `honest_verdict`: principle "Starts complete: or blocked: and a null verdict retires the extension."

### SCENARIO-SAMPLE-5645: Two-Axis Hard-Constraint Quality Artifact

**Given** Exp 5622 corrected-kernel exactness, Exp 5634 one-axis quality
promotion, and Exp 5644 exact two-axis eligibility are ready
**When** Exp 5645 runs the preregistered hard constrained-instance panel
**Then** it compares two-axis tempering, one-axis exchange, and independent
corrected cDLS under matched corrected-kernel transition budgets, records the
required feasibility, violation, transition-to-feasible, round-trip, barrier,
ESS, autocorrelation, feasible-energy, solve-probability, control, and
uncertainty fields, writes
`results/experiment_5645_two_axis_tempering_hard_constraint_quality.json`,
and sets `two_axis_quality_ready_score=1.0` only by the mechanical promotion
gate.

**If** upstream exactness eligibility, budget parity, controls, exact validity,
target diagnostics, feasible energy, seeds/families, or preregistered
interval-improvement gates fail
**Then** Exp 5645 SHALL still emit terminal evidence with
`timing_claimed=false`, `hardware_speedup_claimed=false`,
`two_axis_quality_ready_score=0.0`, and an `honest_verdict` starting with
`complete:` or `blocked:`.

## Implementation Status (REQ-SAMPLE-5645)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5645 | Implemented (`python/carnot/experiment_5645_two_axis_tempering_hard_constraint_quality.py`, `results/experiment_5645_two_axis_tempering_hard_constraint_quality.json`) | Implemented (`tests/python/test_experiment_5645_two_axis_tempering_hard_constraint_quality.py`) |

### REQ-SAMPLE-5714: One-Axis Corrected-cDLS Rust/Python Replica-Exchange Parity

Carnot MUST provide the smallest Rust/PyO3 energy core for the promoted
one-axis corrected-cDLS temperature-label replica-exchange method and write
`results/experiment_5714_one_axis_tempering_rust_parity.json` without
modifying `scripts/research_conductor.py`. The port SHALL be derived only from
Exp 5633's exact temperature-label exchange audit and Exp 5634's paired
mixing promotion, with Exp 5645/5646 treated as retired two-axis scope. The
implementation SHALL claim semantic portability only: `timing_claimed=false`,
`hardware_speedup_claimed=false`, and `two_axis_code_added=false`.

The Rust core SHALL implement the exact Ising energy
`E(x) = -0.5 x^T J x - h^T x`, Exp 5622's corrected
`corrected_cdls_projection_mh` proposal probability and Metropolis-Hastings
decision, Exp 5633's fixed beta-label ladder, label-only adjacent swap
log-ratio and decision, fixed scheduler, cold-target extraction, and a
serializable seeded state/checkpoint. The PyO3 surface SHALL expose only the
minimal config, state, one-step/checkpoint, and diagnostic methods required to
compare deterministic behavior and restart behavior. Existing pure-Python use SHALL continue unchanged when the Rust extension is absent.

The parity audit SHALL freeze fixtures and tolerances before comparison. It
SHALL compare Python and Rust deterministic energies, violations where present,
proposal log probabilities, corrected decisions, swap log-ratios, label
updates, scheduler traces, cold-target states, malformed inputs, checkpoint
round-trips, and cross-language restarts. On exactly enumerable small Ising
systems it SHALL compare both implementations with the exact Boltzmann product
target using transition normalization, detailed-balance residuals, total
variation, target marginals, and fixed-seed replay. Broken controls for
stale-label exchange, wrong-sign swap, uncorrected-kernel proposals,
collapsed ladders, and corrupt state SHALL be preregistered and rejected.
Penalty-axis exchange SHALL NOT be implemented.

`one_axis_rust_parity_ready_score` SHALL equal exactly `1.0` only when
deterministic decisions match, parity errors are within frozen tolerances,
Python and Rust pass exact-target bounds, every broken control is rejected,
checkpoints and restarts reproduce, malformed ABI input fails safely,
fallback equivalence is proven, and no timing or hardware speedup claim is
present. `broken_control_rejected_score` SHALL equal exactly `1.0` only when
every preregistered broken port is rejected; otherwise it SHALL equal `0.0`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `source_promotion_receipts`, `source_artifact_hashes`,
`source_algorithm_hash`, `openspec_requirement_ids`, `rust_source_paths`,
`python_binding_paths`, `python_reference_paths`, `compiler_and_toolchain`,
`pyo3_version`, `build_features`, `abi_receipt`, `fixture_manifest`,
`frozen_tolerances`, `energy_error_max`, `proposal_probability_error_max`,
`swap_log_ratio_error_max`, `deterministic_decision_parity`,
`scheduler_parity`, `exact_target_tv_python`, `exact_target_tv_rust`,
`target_marginal_delta`, `detailed_balance_error_by_impl`,
`checkpoint_roundtrip_pass`, `cross_language_restart_pass`,
`malformed_input_controls`, `broken_control_results`,
`broken_control_rejected`, `broken_control_rejected_score`,
`python_fallback_equivalence`, `two_axis_code_added`, `timing_claimed`,
`hardware_speedup_claimed`, `one_axis_rust_parity_ready_score`,
`inference_substrate`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL equal
`rust_python_exact_one_axis_sampler_parity`.

Required field principles:

- `field_principles`: principle "Explains why every required parity field exists before the artifact can promote a Rust port."
- `source_promotion_receipts`: principle "Pins Exp5633 exactness, Exp5634 quality promotion, and Exp5645/5646 retirement scope before port identity is trusted."
- `source_artifact_hashes`: principle "Content-addresses upstream evidence so the port cannot silently drift from promoted one-axis provenance."
- `source_algorithm_hash`: principle "Hashes the frozen one-axis algorithm recipe independently of implementation language."
- `openspec_requirement_ids`: principle "Keeps implementation, tests, and artifact anchored to REQ-SAMPLE-5714 and SCENARIO-SAMPLE-5714."
- `rust_source_paths`: principle "Lists the Rust files that implement the portable one-axis core."
- `python_binding_paths`: principle "Lists the PyO3/fallback files that expose the Rust core to Python."
- `python_reference_paths`: principle "Lists the Python reference and test files used for exact parity."
- `compiler_and_toolchain`: principle "Records the local Rust, Python, Cargo, and platform versions needed to reconstruct the build."
- `pyo3_version`: principle "Records the binding ABI dependency version explicitly."
- `build_features`: principle "Shows that the build used the ordinary one-axis Rust/PyO3 feature surface."
- `abi_receipt`: principle "Proves malformed inputs fail safely at the Python/Rust boundary."
- `fixture_manifest`: principle "Freezes the exact enumerable Ising workloads, ladders, schedules, and seeds."
- `frozen_tolerances`: principle "Predeclares numerical and exact-target tolerances before parity is interpreted."
- `energy_error_max`: principle "Quantifies deterministic energy parity between Python and Rust."
- `proposal_probability_error_max`: principle "Quantifies corrected cDLS proposal-probability parity."
- `swap_log_ratio_error_max`: principle "Quantifies temperature-label swap-ratio parity."
- `deterministic_decision_parity`: principle "Proves seeded within-replica and swap decisions match exactly."
- `scheduler_parity`: principle "Proves both implementations apply the same fixed within-replica and exchange schedule."
- `exact_target_tv_python`: principle "Measures Python exact-target stationarity against the enumerable Boltzmann product target."
- `exact_target_tv_rust`: principle "Measures Rust exact-target stationarity against the same enumerable Boltzmann product target."
- `target_marginal_delta`: principle "Measures cold-target marginal agreement across implementations and exact target."
- `detailed_balance_error_by_impl`: principle "Reports within-kernel and swap detailed-balance residuals by implementation."
- `checkpoint_roundtrip_pass`: principle "Proves Rust state serialization preserves labels, states, seed, and sweep."
- `cross_language_restart_pass`: principle "Proves a Python checkpoint can resume through Rust and reproduce the trace."
- `malformed_input_controls`: principle "Documents invalid ABI inputs and confirms they fail closed."
- `broken_control_results`: principle "Documents stale-label, wrong-sign, uncorrected-kernel, collapsed-ladder, and corrupt-state controls."
- `broken_control_rejected`: principle "Provides the boolean gate that every broken control was rejected."
- `broken_control_rejected_score`: principle "Provides the scalar gate that is 1.0 only when all broken controls reject."
- `python_fallback_equivalence`: principle "Proves the pure-Python path remains available when Rust is absent."
- `two_axis_code_added`: principle "Bare false closes the retired penalty-axis scope."
- `timing_claimed`: principle "Bare false prevents semantic parity from becoming a speed claim."
- `hardware_speedup_claimed`: principle "Bare false prevents portability evidence from becoming board or hardware evidence."
- `one_axis_rust_parity_ready_score`: principle "Provides a scalar downstream gate that is 1.0 only under the full parity contract."
- `inference_substrate`: principle "Declares Rust/Python exact one-axis sampler parity with no LLM or board participation."
- `random_seeds`: principle "Records replay seeds for deterministic decisions and restart."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether parity is final."

### SCENARIO-SAMPLE-5714: One-Axis Rust Port Emits Exact Parity Evidence

**Given** Exp 5633 exact one-axis temperature-label exchange evidence and
Exp 5634 paired one-axis quality promotion are hash-pinned
**When** the Rust/PyO3 one-axis core and Python reference are compared on
frozen enumerable Ising fixtures
**Then** deterministic energies, corrected proposal probabilities,
within-replica accept decisions, swap log-ratios, label schedules,
cold-target states, checkpoints, cross-language restarts, exact-target
stationarity, detailed balance, total variation, and marginals match within
frozen tolerances
**And** `results/experiment_5714_one_axis_tempering_rust_parity.json` is
written with `one_axis_rust_parity_ready_score=1.0`,
`broken_control_rejected_score=1.0`, `two_axis_code_added=false`,
`timing_claimed=false`, and `hardware_speedup_claimed=false`.

**If** source provenance cannot reconstruct the promoted one-axis algorithm,
Rust/Python parity fails, exact-target bounds fail, malformed input does not
fail safely, fallback equivalence is lost, any broken control is accepted, or
any timing/hardware claim appears
**Then** the artifact SHALL still emit terminal evidence with
`one_axis_rust_parity_ready_score=0.0`,
`broken_control_rejected_score=0.0` when applicable, and an
`honest_verdict` starting with `blocked:`.

## Implementation Status (REQ-SAMPLE-5714)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5714 | Planned (`crates/carnot-samplers/src/one_axis_tempering.rs`, `crates/carnot-python/src/one_axis_tempering.rs`, `python/carnot/experiment_5714_one_axis_tempering_rust_parity.py`, `results/experiment_5714_one_axis_tempering_rust_parity.json`) | Planned (`tests/python/test_experiment_5714_one_axis_tempering_rust_parity.py`) |

### REQ-SAMPLE-5715: One-Axis Rust/Python Hard-Instance Quality And Restart Parity

Carnot MUST provide Exp 5715 at
`python/carnot/experiment_5715_one_axis_tempering_rust_quality_restart.py` and
write `results/experiment_5715_one_axis_tempering_rust_quality_restart.json`
without modifying `scripts/research_conductor.py`. Exp 5715 SHALL compare the
promoted one-axis corrected-cDLS temperature-label exchange algorithm across
the Python reference and Rust/PyO3 boundary on the Exp 5634 hard-instance
families and configuration without tuning. The experiment SHALL use Exp 5634
eligibility as a source-quality gate and Exp 5714 exact Rust/Python parity as
the implementation gate.

The protocol SHALL freeze instances, instance hashes, seeds, burn-in, sample
sweeps, corrected-transition budget, beta ladder, swap schedule, checkpoint
midpoints, tolerances, margins, and paired-interval analysis before any arm
quality result is interpreted. Python and Rust arms SHALL receive identical
corrected proposals, ladder, seeds, transitions, swaps, and cold-target
collection. Setup and swap work SHALL be accounted for, but wall time SHALL not
be compared: `timing_claimed=false` and `hardware_speedup_claimed=false`.
No two-axis arm or penalty-axis exchange code SHALL be used in any arm, and
`two_axis_arm_count` SHALL equal `0`.

The quality report SHALL include uninterrupted Python and Rust arms,
Python-to-Rust and Rust-to-Python midpoint restarts, corrupt checkpoint, wrong
ladder, stale label, and independent-cDLS diagnostic controls. Checkpoints
SHALL carry a schema version, instance identity, ladder hash, implementation
label, state payload, and payload checksum. Corrupt, truncated, wrong-version,
wrong-endianness, wrong-ladder, and stale-label checkpoints SHALL fail closed
without producing a resumed sample. At fixed midpoints, deterministic suffixes
SHALL match exactly where the shared seeded state permits; independent restart
seed suffixes SHALL be compared distributionally otherwise.

Exp 5715 SHALL report exact validity, best and mean energy, feasible hits
where applicable, effective sample size, integrated autocorrelation, barrier
crossings, temperature round trips, solve probability, target distributions,
paired intervals, failures, and denominators for each quality and restart arm.
`one_axis_rust_quality_ready_score` SHALL equal exactly `1.0` only when Rust
has no material validity, energy, ESS, autocorrelation, barrier, or solve
regression against Python; all frozen paired intervals stay within margins;
both cross-language restarts pass; corrupt/unsafe state fails closed; no
two-axis arm is used; and no timing or hardware-speedup claim is present.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `upstream_gate_receipts`, `source_quality_receipt`,
`preregistered_protocol`, `instance_manifest`, `instance_hashes`,
`implementation_hashes`, `sampler_configs`, `transition_budget_parity`,
`swap_schedule_parity`, `successful_seed_count`, `failed_seed_reasons`,
`exact_validity_by_arm`, `energy_by_arm`, `feasible_hit_rate_by_arm`,
`ess_by_arm`, `autocorrelation_by_arm`, `barrier_crossings_by_arm`,
`temperature_round_trips_by_arm`, `solve_probability_by_arm`,
`target_distributions_by_arm`, `paired_intervals`,
`material_regression_count`, `checkpoint_matrix`,
`checkpoint_schema_version`, `python_to_rust_restart_pass`,
`rust_to_python_restart_pass`, `restart_suffix_metrics`,
`corrupted_checkpoint_controls`, `two_axis_arm_count`, `timing_claimed`,
`hardware_speedup_claimed`, `one_axis_rust_quality_ready_score`,
`inference_substrate`, `random_seeds`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL equal
`matched_rust_python_one_axis_sampler_cpu`.

Required field principles:

- `field_principles`: principle "Explains why every required hard-instance quality and restart field exists before promotion."
- `upstream_gate_receipts`: principle "Pins Exp5634 quality eligibility and Exp5714 Rust/Python exact parity before Exp5715 is interpreted."
- `source_quality_receipt`: principle "Binds hard-instance eligibility to Exp5634 instead of selecting a new tuned quality panel."
- `preregistered_protocol`: principle "Freezes instances, seeds, schedule, budget, margins, checkpoints, and analysis before results."
- `instance_manifest`: principle "Lists the exact Exp5634 hard instances and families used by every arm."
- `instance_hashes`: principle "Content-addresses each hard instance so tuning or silent panel drift is detectable."
- `implementation_hashes`: principle "Hashes the Python reference, Rust core, PyO3 binding, tests, and upstream artifacts needed to reconstruct arms."
- `sampler_configs`: principle "Makes Python, Rust, restart, and diagnostic arms reconstructable without reading prose."
- `transition_budget_parity`: principle "Proves corrected proposals and cold-target collections are matched across languages and restarts."
- `swap_schedule_parity`: principle "Proves label-only adjacent swaps and exchange attempts are matched."
- `successful_seed_count`: principle "Reports the denominator that actually produced paired quality evidence."
- `failed_seed_reasons`: principle "Preserves failed-seed blockers instead of silently shrinking denominators."
- `exact_validity_by_arm`: principle "Prevents invalid states from appearing as quality wins."
- `energy_by_arm`: principle "Reports best and mean cold-target energy quality."
- `feasible_hit_rate_by_arm`: principle "Reports feasible hits where exact verifier or CSP utility applies."
- `ess_by_arm`: principle "Reports usable sample count under serial dependence."
- `autocorrelation_by_arm`: principle "Reports integrated autocorrelation so mixing regressions are visible."
- `barrier_crossings_by_arm`: principle "Measures metastable basin transitions directly."
- `temperature_round_trips_by_arm`: principle "Measures whether the one-axis temperature labels traverse the ladder."
- `solve_probability_by_arm`: principle "Reports exact-solve utility by arm."
- `target_distributions_by_arm`: principle "Records cold-target energy and validity distributions for distributional parity."
- `paired_intervals`: principle "Bounds Python-vs-Rust and restart deltas using frozen paired intervals and margins."
- `material_regression_count`: principle "Counts only frozen-margin failures that would block promotion."
- `checkpoint_matrix`: principle "Shows uninterrupted and cross-language midpoint restart arms per instance and seed."
- `checkpoint_schema_version`: principle "Pins the portable checkpoint schema version."
- `python_to_rust_restart_pass`: principle "Gates Python checkpoint resumed by Rust."
- `rust_to_python_restart_pass`: principle "Gates Rust checkpoint resumed by Python."
- `restart_suffix_metrics`: principle "Compares deterministic and independent-seed suffixes after restart."
- `corrupted_checkpoint_controls`: principle "Proves corrupt, truncated, wrong-version, wrong-endianness, wrong-ladder, and stale-label states fail closed."
- `two_axis_arm_count`: principle "Bare zero keeps retired two-axis scope closed."
- `timing_claimed`: principle "Bare false prevents quality parity from becoming a wall-time benchmark."
- `hardware_speedup_claimed`: principle "Bare false prevents CPU portability evidence from becoming a board or hardware claim."
- `one_axis_rust_quality_ready_score`: principle "Equals 1.0 only when all quality, interval, restart, corruption, and no-speed gates pass."
- `inference_substrate`: principle "Declares matched Rust/Python one-axis CPU sampling with no LLM or board involvement."
- `random_seeds`: principle "Records replay seeds for every paired row and restart suffix."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether hard-instance Rust quality and restart parity is final."

### SCENARIO-SAMPLE-5715: Exp5715 Emits Hard-Instance Rust Quality And Restart Evidence

**Given** Exp 5634 quality eligibility and Exp 5714 exact Rust/Python parity
are hash-pinned
**When** Exp 5715 runs the frozen Exp5634 one-axis hard-instance panel with
matched Python, Rust, Python-to-Rust restart, Rust-to-Python restart, and
diagnostic control arms
**Then** it writes
`results/experiment_5715_one_axis_tempering_rust_quality_restart.json` with all
required schema fields, matched transition and swap budgets, honest
successful/failed seed denominators, per-arm validity/energy/ESS/autorrelation/
barrier/round-trip/solve/target-distribution metrics, paired intervals, and
restart suffix metrics
**And** `one_axis_rust_quality_ready_score=1.0`, `two_axis_arm_count=0`,
`timing_claimed=false`, and `hardware_speedup_claimed=false` only when no
material Rust or restart regression appears and corrupt state fails closed.

**If** Exp5634 or Exp5714 provenance is stale, budgets differ, any material
quality interval fails, either cross-language restart fails, corrupt state does
not fail closed, a two-axis arm appears, or any speed/hardware claim appears
**Then** the artifact SHALL still emit terminal evidence with
`one_axis_rust_quality_ready_score=0.0` and an `honest_verdict` starting with
`blocked:`.

## Implementation Status (REQ-SAMPLE-5715)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5715 | Planned (`python/carnot/experiment_5715_one_axis_tempering_rust_quality_restart.py`, `results/experiment_5715_one_axis_tempering_rust_quality_restart.json`) | Planned (`tests/python/test_experiment_5715_one_axis_tempering_rust_quality_restart.py`) |

### REQ-SAMPLE-5723: One-Axis Rust SamplerBackend Production Integration

Carnot MUST expose the promoted one-axis corrected-cDLS temperature-label
replica-exchange Rust/PyO3 kernel through the production sampler backend
boundary without changing the default backend. The implementation SHALL provide
a `OneAxisRustBackend` that satisfies `SamplerBackend`, register it under the
explicit factory name `one_axis_rust`, and leave the default factory selection
as `cpu` unless a caller explicitly requests the one-axis backend.

The backend SHALL accept only the frozen one-axis descriptor
`one_axis_corrected_cdls_temperature_label_exchange` with topology
`one_axis_temperature_label_exchange`, Exp5714's source algorithm hash,
Exp5714/5715's beta ladder, proposal standard deviation, drift scale, label-only
adjacent swap schedule, corrected transition budget, LCG seed semantics, and
Ising energy convention `E(x) = -0.5 x^T J x - h^T x`. It SHALL route
Rust-supported descriptors through the Rust/PyO3 config/core/state symbols and
SHALL use an explicit exact Python fallback when the extension is missing,
broken, missing symbols, the input is a declared Python-compatibility case, or
the input dtype/layout is outside the adapter's Rust-supported array surface.
Fallback use SHALL be recorded with `active_backend="python_exact_fallback"` and
a non-empty `fallback_reason`; malformed descriptors, unsupported topology,
invalid shape, corrupt checkpoints, seed mismatches, and two-axis requests SHALL
fail closed.

The backend SHALL emit stable input, output, and checkpoint receipts for each
run. Checkpoints SHALL include a schema version, source algorithm hash,
descriptor hash, input hash, seed, beta-ladder hash, portable sampler state,
active backend, fallback reason, and payload checksum. Duplicate save/load,
corruption rejection, Python-to-Rust restart, and Rust-to-Python restart SHALL
be verified through the production adapter rather than only through experiment
helpers.

Experiment 5723 SHALL write
`results/experiment_5723_one_axis_rust_samplerbackend_integration.json` after
verifying Exp5714/5715 hashes, ready scores, algorithm hash, checkpoint schema,
and two-axis-closed receipts. `one_axis_samplerbackend_ready_score` SHALL equal
exactly `1.0` only when protocol/factory conformance, exact Rust/Python parity,
checkpoint/restart, fallback equivalence, and broken controls pass. The
artifact SHALL make no speed or hardware claim:
`two_axis_code_added=false`, `timing_claimed=false`, and
`hardware_speedup_claimed=false`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `upstream_gate_receipts`, `source_algorithm_hash`,
`rust_source_hash`, `pyo3_binding_hash`, `python_adapter_hash`,
`sampler_backend_protocol`, `factory_registration_receipt`,
`supported_descriptor_schema`, `unsupported_case_policy`, `seed_semantics`,
`temperature_ladder_receipt`, `swap_schedule_receipt`,
`transition_budget_receipt`, `energy_parity_max_error`,
`proposal_parity_max_error`, `swap_parity_max_error`, `decision_log_parity`,
`checkpoint_schema_version`, `checkpoint_roundtrip_pass`,
`python_to_rust_restart_pass`, `rust_to_python_restart_pass`,
`fallback_cases`, `fallback_equivalence_pass`,
`exact_fallback_equivalence_score`, `broken_control_results`,
`one_axis_samplerbackend_ready_score`, `two_axis_code_added`,
`timing_claimed`, `hardware_speedup_claimed`, `inference_substrate`,
`random_seeds`, `reproducibility_checksum`, and `honest_verdict`.
`inference_substrate` SHALL equal
`production_python_samplerbackend_plus_rust_pyo3_one_axis`.

Required field principles:

- `field_principles`: principle "Explains why every required production integration field exists before the sampler backend can be marked ready."
- `upstream_gate_receipts`: principle "Pins Exp5714 exact parity, Exp5715 hard-instance quality/restart, source hashes, algorithm identity, checkpoint schema, and two-axis-closed receipts."
- `source_algorithm_hash`: principle "Binds the adapter to the frozen one-axis algorithm recipe rather than a renamed or tuned variant."
- `rust_source_hash`: principle "Content-addresses the Rust kernel used by the adapter."
- `pyo3_binding_hash`: principle "Content-addresses the PyO3 binding symbols used by the adapter."
- `python_adapter_hash`: principle "Content-addresses the production Python adapter that implements fallback and receipts."
- `sampler_backend_protocol`: principle "Proves the adapter satisfies the production SamplerBackend boundary."
- `factory_registration_receipt`: principle "Proves explicit factory registration while preserving the cpu default."
- `supported_descriptor_schema`: principle "Declares the only descriptor accepted for the promoted one-axis path."
- `unsupported_case_policy`: principle "Separates exact fallback cases from fail-closed malformed or out-of-scope cases."
- `seed_semantics`: principle "Records the portable LCG state and seed mismatch rule used for replay and restart."
- `temperature_ladder_receipt`: principle "Proves the Exp5714/5715 one-axis beta ladder was not changed."
- `swap_schedule_receipt`: principle "Proves exchange remains label-only adjacent one-axis swapping."
- `transition_budget_receipt`: principle "Accounts for corrected transitions and swap attempts without adding work silently."
- `energy_parity_max_error`: principle "Quantifies production-adapter energy parity against the exact Python fallback."
- `proposal_parity_max_error`: principle "Quantifies production-adapter corrected proposal parity."
- `swap_parity_max_error`: principle "Quantifies production-adapter swap-ratio parity."
- `decision_log_parity`: principle "Proves the recorded production decision log matches across Rust and exact fallback."
- `checkpoint_schema_version`: principle "Pins the production adapter checkpoint schema."
- `checkpoint_roundtrip_pass`: principle "Proves duplicate save/load preserves state and receipts."
- `python_to_rust_restart_pass`: principle "Proves a Python fallback checkpoint can resume through the Rust adapter."
- `rust_to_python_restart_pass`: principle "Proves a Rust adapter checkpoint can resume through exact Python fallback."
- `fallback_cases`: principle "Lists every exercised exact fallback case and the recorded reason."
- `fallback_equivalence_pass`: principle "Gates fallback only when samples, checkpoints, and decision logs match exactly."
- `exact_fallback_equivalence_score`: principle "Provides a scalar gate equal to 1.0 only when all fallback cases are exact."
- `broken_control_results`: principle "Documents broken-extension, wrong-symbol, malformed descriptor, unsupported topology, corrupt checkpoint, seed mismatch, and energy-sign controls."
- `one_axis_samplerbackend_ready_score`: principle "Equals 1.0 only when upstream gates, protocol/factory exposure, parity, restart, fallback, and controls all pass."
- `two_axis_code_added`: principle "Bare false keeps retired penalty-axis exchange closed."
- `timing_claimed`: principle "Bare false prevents integration readiness from becoming a timing claim."
- `hardware_speedup_claimed`: principle "Bare false prevents PyO3 routing from becoming a hardware claim."
- `inference_substrate`: principle "Declares production Python SamplerBackend plus Rust/PyO3 one-axis execution with no LLM or board participation."
- `random_seeds`: principle "Records replay seeds for production adapter parity, fallback, and restart checks."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether production SamplerBackend integration is ready."

### SCENARIO-SAMPLE-5723: One-Axis Rust Backend Emits Production Integration Evidence

**Given** Exp5714 exact Rust/Python one-axis parity and Exp5715 hard-instance
quality/restart readiness are hash-pinned
**When** the production `OneAxisRustBackend` is selected explicitly through
`get_backend("one_axis_rust")` and run on the frozen one-axis descriptor
**Then** Rust/PyO3 and exact Python fallback samples, energies, proposal
diagnostics, swap diagnostics, decision logs, transition budgets, output
receipts, checkpoints, duplicate save/load, and cross-language restarts match
within frozen tolerances
**And** missing/broken/wrong-symbol extension and declared compatibility cases
take recorded exact fallback
**And** malformed descriptors, unsupported topology, corrupt checkpoints, seed
mismatches, and energy-sign controls fail closed or are rejected
**And** `results/experiment_5723_one_axis_rust_samplerbackend_integration.json`
is written with `one_axis_samplerbackend_ready_score=1.0`,
`two_axis_code_added=false`, `timing_claimed=false`, and
`hardware_speedup_claimed=false`.

**If** upstream Exp5714/5715 gates are stale, the factory changes the default,
Rust/Python adapter parity fails, checkpoint or restart receipts fail, fallback
equivalence is not exact, any out-of-scope control is accepted, or any
timing/hardware claim appears
**Then** the artifact SHALL still emit terminal evidence with
`one_axis_samplerbackend_ready_score=0.0` and an `honest_verdict` starting with
`blocked:`.

## Implementation Status (REQ-SAMPLE-5723)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5723 | Implemented (`python/carnot/samplers/one_axis_rust_backend.py`, `python/carnot/samplers/backend.py`, `python/carnot/experiment_5723_one_axis_rust_samplerbackend_integration.py`, `results/experiment_5723_one_axis_rust_samplerbackend_integration.json`) | Implemented (`tests/python/samplers/test_one_axis_rust_backend.py`, `tests/python/test_experiment_5723_one_axis_rust_samplerbackend_integration.py`) |

### REQ-SAMPLE-5724: One-Axis Rust/Python Production-Backend Matched Crossover

Carnot MUST provide Exp 5724 at
`python/carnot/experiment_5724_one_axis_rust_python_matched_crossover.py` and
write `results/experiment_5724_one_axis_rust_python_matched_crossover.json`
without modifying `scripts/research_conductor.py`. Exp 5724 SHALL ask only
where, if anywhere, the production Rust/PyO3 one-axis backend beats the exact
Python one-axis fallback under identical CPU work and quality. A null crossover
is valid and SHALL be terminal for this CPU Rust/Python substrate. No CPU/CUDA,
GPU, FPGA, board, or TSU speedup claim is in scope.

The protocol SHALL freeze build profile, hardware/OS/compiler/Python receipts,
CPU affinity, observable frequency/governor, thread settings, workloads,
problem sizes, random seeds, beta ladders, corrected-transition budgets,
restarts, warmups, timing repetitions, quality margins, and crossover
thresholds before timing. The study SHALL use at least six size strata spanning
PyO3-overhead-dominated through kernel-dominated work, at least three
topology/hardness families, and at least ten paired seeds through the same
production `SamplerBackend` API. Rust and Python arms SHALL match transitions,
energy evaluations, replicas, swaps, restarts, stopping rules, checkpoints,
initial states, validation, and work counters.

Both arms SHALL be warmed before measurement. Timing order SHALL be randomized
and interleaved with a deterministic benchmark-order seed. Each qualified
size/family SHALL record at least thirty measured paired repetitions. The
artifact SHALL report affinity, observable frequency/governor, peak RSS, build
mode, and timing noise. It SHALL separate kernel, setup, PyO3 boundary,
serialization/checkpoint, validation, and end-to-end timing while keeping the
primary speedup based on end-to-end time without subtracting overhead.

Quality SHALL be matched before speedups are interpreted. Matching SHALL cover
state feasibility, best and mean energy, within-replica acceptance, swap
acceptance, exact target-distribution parity where enumerable, restart parity,
and work-counter equality. Excluded pairs SHALL remain in the denominator with
preregistered reasons. Speedups SHALL be paired Rust-speedup ratios computed as
`python_end_to_end_time / rust_end_to_end_time` with bootstrap intervals.
`qualified_crossover_n` SHALL be the first size whose consecutive larger-size
suffix remains quality-matched and has a Rust end-to-end interval entirely
above `1.0`; otherwise it SHALL be null. `rust_crossover_ready_score` SHALL be
`1.0` only when that crossover gate passes with timing claimed and all
hardware/GPU/FPGA/TSU claims false; otherwise it SHALL be `0.0`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `upstream_gate_receipts`, `hardware_receipt`,
`software_receipt`, `build_profile`, `cpu_affinity`, `thread_receipts`,
`preregistered_protocol`, `workload_manifest`, `problem_sizes`,
`topology_families`, `random_seeds`, `warmup_count`,
`measured_repetition_count`, `arm_configs`, `matched_work_receipts`,
`quality_metrics_by_pair`, `quality_margins`, `quality_matched_pair_count`,
`excluded_pair_reasons`, `kernel_times`, `pyo3_overhead_times`,
`serialization_times`, `validation_times`, `end_to_end_times`,
`peak_rss_by_arm`, `paired_speedup_ratios`, `paired_speedup_intervals`,
`qualified_crossover_n`, `rust_crossover_ready_score`,
`software_speedup_claimed`, `timing_claimed`, `hardware_speedup_claimed`,
`gpu_speedup_claimed`, `fpga_or_tsu_used`, `inference_substrate`,
`reproducibility_checksum`, and `honest_verdict`. `timing_claimed` SHALL be
true. `hardware_speedup_claimed`, `gpu_speedup_claimed`, and
`fpga_or_tsu_used` SHALL be false. `inference_substrate` SHALL equal
`matched_cpu_python_vs_rust_pyo3_production_samplerbackend`.

Required field principles:

- `field_principles`: principle "Explains why every crossover field exists before a reviewer trusts the JSON shape."
- `upstream_gate_receipts`: principle "Pins Exp5611/5623 CPU/CUDA terminal scope and Exp5714/5715/5723 one-axis Rust/Python readiness before timing is interpreted."
- `hardware_receipt`: principle "Authenticates CPU, OS, memory, affinity, and observable frequency without implying board or accelerator use."
- `software_receipt`: principle "Records Python, NumPy, Rust extension, compiler, and source hashes needed to replay the production arms."
- `build_profile`: principle "Freezes release/debug, PyO3 ABI, and feature settings before timing."
- `cpu_affinity`: principle "Shows the CPU set used for timing instead of leaving scheduler placement implicit."
- `thread_receipts`: principle "Records thread environment and observed thread pools so Python and Rust work are compared under the same CPU policy."
- `preregistered_protocol`: principle "Freezes workloads, sizes, seeds, repetitions, warmups, budgets, margins, thresholds, and null rules before timing."
- `workload_manifest`: principle "Lists every size/family Hamiltonian and descriptor hash used by both arms."
- `problem_sizes`: principle "Makes the PyO3-overhead to kernel-dominated size range explicit."
- `topology_families`: principle "Shows all hardness/topology families entering the denominator."
- `random_seeds`: principle "Records the ten paired seeds and benchmark-order seed for replay."
- `warmup_count`: principle "Prevents cold-start effects from entering primary measurements."
- `measured_repetition_count`: principle "Proves each qualified size/family has at least thirty timing repetitions."
- `arm_configs`: principle "Names the Rust/PyO3 and exact Python fallback SamplerBackend configurations under one contract."
- `matched_work_receipts`: principle "Proves transitions, energy evaluations, replicas, swaps, restarts, checkpoints, stopping, and initial states are identical."
- `quality_metrics_by_pair`: principle "Reports quality before any speedup denominator is formed."
- `quality_margins`: principle "Predeclares feasibility, energy, acceptance, swap, target, restart, and work-counter margins."
- `quality_matched_pair_count`: principle "Counts only pairs that passed quality gates before timing intervals."
- `excluded_pair_reasons`: principle "Keeps failed or unmatched pairs in the denominator instead of silently shrinking evidence."
- `kernel_times`: principle "Reports the measured SamplerBackend sampling call cost separately from setup and validation."
- `pyo3_overhead_times`: principle "Records PyO3 boundary probe cost without subtracting it from primary end-to-end timing."
- `serialization_times`: principle "Reports checkpoint/JSON serialization cost separately while keeping it in end-to-end timing."
- `validation_times`: principle "Reports per-run validation cost separately while keeping it in end-to-end timing."
- `end_to_end_times`: principle "Preserves the primary timing evidence that includes all production overhead."
- `peak_rss_by_arm`: principle "Records memory pressure by arm so speed is not traded against unreported RSS growth."
- `paired_speedup_ratios`: principle "Reports paired Python/Rust end-to-end ratios rather than unrelated aggregate means."
- `paired_speedup_intervals`: principle "Uses bootstrap intervals so a crossover cannot rest on one noisy repetition."
- `qualified_crossover_n`: principle "Records the first gated consecutive crossover size or null for a terminal no-crossover result."
- `rust_crossover_ready_score`: principle "Equals 1.0 only when matched quality and end-to-end timing prove a Rust CPU crossover with no hardware claim."
- `software_speedup_claimed`: principle "Bare boolean separates an allowed Rust/Python software timing claim from a null result."
- `timing_claimed`: principle "Bare true declares this is a CPU timing study, unlike Exp5714/5715/5723 parity-only artifacts."
- `hardware_speedup_claimed`: principle "Bare false prevents Rust/PyO3 CPU timing from becoming a board or TSU claim."
- `gpu_speedup_claimed`: principle "Bare false prevents the retired CPU/CUDA substrate from being reopened."
- `fpga_or_tsu_used`: principle "Bare false records that no FPGA, board, or TSU participated."
- `inference_substrate`: principle "Declares matched CPU Python versus Rust/PyO3 production SamplerBackend sampling, not LLM, GPU, or hardware timing."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether the Rust/Python crossover is proven or terminal null."

### SCENARIO-SAMPLE-5724: Exp5724 Emits Matched CPU Rust/Python Crossover Evidence

**Given** Exp5714 exact parity, Exp5715 hard-instance quality/restart parity,
and Exp5723 production SamplerBackend integration are hash-pinned
**When** Exp5724 runs the frozen one-axis workload panel through the same
production SamplerBackend API with Rust/PyO3 and exact Python fallback arms
**Then** it records matched work counters, quality metrics, timing components,
paired speedup ratios, bootstrap intervals, environment receipts, and writes
`results/experiment_5724_one_axis_rust_python_matched_crossover.json`
**And** any `qualified_crossover_n` is allowed only when quality matches and
the consecutive larger-size end-to-end Rust-speedup intervals are entirely
above `1.0`.

**If** upstream gates fail, Rust/Python quality diverges, work counters differ,
timing repetitions are insufficient, Rust speed intervals do not prove a
consecutive larger-size crossover, or any GPU/FPGA/TSU/hardware claim appears
**Then** the artifact SHALL still emit terminal evidence with
`qualified_crossover_n=null`, `rust_crossover_ready_score=0.0`,
`software_speedup_claimed=false`, `timing_claimed=true`,
`hardware_speedup_claimed=false`, `gpu_speedup_claimed=false`,
`fpga_or_tsu_used=false`, and an `honest_verdict` starting with `complete:` or
`blocked:`.

## Implementation Status (REQ-SAMPLE-5724)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5724 | Implemented (`python/carnot/experiment_5724_one_axis_rust_python_matched_crossover.py`, `results/experiment_5724_one_axis_rust_python_matched_crossover.json`) | Implemented (`tests/python/test_experiment_5724_one_axis_rust_python_matched_crossover.py`) |

### REQ-SAMPLE-5738: One-Axis Batched Rust SamplerBackend Boundary

Carnot MUST provide Exp 5738 at
`python/carnot/experiment_5738_one_axis_rust_batched_backend.py` and write
`results/experiment_5738_one_axis_rust_batched_backend.json` without modifying
`scripts/research_conductor.py`. Exp 5738 SHALL reproduce the Exp5724 large
size reversal cells at `n=48` and `n=96` with identical workload descriptors,
seeds, checkpoint budgets, and production `SamplerBackend` semantics, then
attribute time and memory proxies by phase before deciding whether a batched
boundary is justified. The phase receipts SHALL cover serialization, Python
allocation, PyO3 crossing, Rust allocation, energy update, proposal, exchange,
validation, checkpoint, restart, and end-to-end work. If the measured dominant
phase is not the scalar Python/PyO3 call boundary or another batch-removable
boundary cost, Exp5738 SHALL emit an honest null or blocked artifact and SHALL
NOT add speculative optimization code.

When justified, the production one-axis backend SHALL add a backward-compatible
`sample_batch` path for independent workloads. The scalar `sample()` API SHALL
remain unchanged, the exact Python fallback SHALL remain byte-for-byte
equivalent to scalar fallback replay, result ordering SHALL be deterministic,
and the production factory SHALL expose the batched-capable
`OneAxisRustBackend` under the existing explicit `one_axis_rust` name. The
batch API SHALL handle normal, empty, singleton, mixed-size,
corrupted-checkpoint, broken-binding, and exception controls without changing
the retired two-axis exchange topology.

Batch parity SHALL match Python/Rust energy traces, proposal diagnostics,
accepted transitions, temperature-label exchanges, scheduler positions,
checkpoints, restart suffixes, and result order. Distributional parity SHALL
run at least 10000 retained samples for `n>=64` and apply a multiple-comparison
correction before any readiness gate is set. `batch_backend_ready_score` SHALL
equal exactly `1.0` only when semantic/distributional/restart parity pass,
fallback equivalence passes, ordering is deterministic, production factory
wiring is real, and all timing/software/hardware speedup claims remain false.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `preconditions_checked`, `upstream_artifact_hashes`,
`software_receipt`, `build_profile`, `cpu_topology`, `cpu_affinity`,
`thread_receipts`, `reproduction_workloads`, `phase_timing_receipts`,
`memory_phase_receipts`, `large_size_reversal_reproduced`, `dominant_phase`,
`optimization_hypothesis`, `batch_api_contract`, `batch_factory_receipts`,
`scalar_api_unchanged`, `python_fallback_receipts`, `parity_manifest`,
`energy_trace_mismatch_count`, `proposal_mismatch_count`,
`exchange_mismatch_count`, `checkpoint_mismatch_count`,
`restart_mismatch_count`, `result_order_mismatch_count`,
`distributional_parity_receipts`, `batch_backend_ready_score`,
`timing_claimed`, `software_speedup_claimed`, `hardware_speedup_claimed`,
`fpga_or_tsu_used`, `inference_substrate`, `random_seeds`,
`reproducibility_checksum`, and `honest_verdict`. `timing_claimed`,
`software_speedup_claimed`, and `hardware_speedup_claimed` SHALL be false,
`fpga_or_tsu_used` SHALL be false, and `inference_substrate` SHALL equal
`local_cpu_rust_pyo3_one_axis_batched_sampler`.

Required field principles:

- `field_principles`: principle "Explains why every Exp5738 artifact field exists before batch readiness can be trusted."
- `preconditions_checked`: principle "Records release Rust/PyO3 build, CPU topology, affinity, RAM, compiler, and upstream-hash checks before measurement or optimization."
- `upstream_artifact_hashes`: principle "Pins Exp5723 and Exp5724 inputs so reversal attribution cannot drift with stale artifacts."
- `software_receipt`: principle "Records Python, NumPy, Rust extension, compiler, and source hashes needed to replay batch semantics."
- `build_profile`: principle "Freezes release/debug and PyO3 ABI state before interpreting boundary evidence."
- `cpu_topology`: principle "Separates physical/logical core topology from any speed or hardware claim."
- `cpu_affinity`: principle "Shows scheduler placement controls used during reproduction and profiling."
- `thread_receipts`: principle "Records thread environment so scalar and batch paths share the same CPU policy."
- `reproduction_workloads`: principle "Lists the exact n=48 and n=96 workload cells used to reproduce the reversal."
- `phase_timing_receipts`: principle "Attributes observed time to measurable phases before any batch-boundary edit is justified."
- `memory_phase_receipts`: principle "Reports peak RSS and memory-traffic proxies so allocation pressure is not hidden by aggregate time."
- `large_size_reversal_reproduced`: principle "States whether the Exp5724 Rust loss at n=48 and n=96 was reproduced under this run's preconditions."
- `dominant_phase`: principle "Names the measured phase that controls the optimization hypothesis."
- `optimization_hypothesis`: principle "States a falsifiable batch-boundary hypothesis or an honest null before implementation."
- `batch_api_contract`: principle "Defines result ordering, independent workload semantics, controls, and compatibility for sample_batch."
- `batch_factory_receipts`: principle "Proves the production factory returns the batched-capable explicit one-axis backend."
- `scalar_api_unchanged`: principle "Prevents batch support from mutating the existing sample() behavior."
- `python_fallback_receipts`: principle "Proves the exact Python fallback remains equivalent for scalar and batch calls."
- `parity_manifest`: principle "Lists semantic, checkpoint, restart, fallback, ordering, and adversarial controls in one replayable manifest."
- `energy_trace_mismatch_count`: principle "Counts energy-trace mismatches instead of burying them in a pass boolean."
- `proposal_mismatch_count`: principle "Counts proposal diagnostic mismatches across scalar, batch, Rust, and fallback paths."
- `exchange_mismatch_count`: principle "Counts temperature-label exchange mismatches so retired two-axis behavior cannot reappear silently."
- `checkpoint_mismatch_count`: principle "Counts checkpoint mismatches across scalar and batch runs."
- `restart_mismatch_count`: principle "Counts restart suffix mismatches after checkpoint handoff."
- `result_order_mismatch_count`: principle "Counts deterministic result-order mismatches for independent workloads."
- `distributional_parity_receipts`: principle "Records >=10000-sample n>=64 parity and multiple-comparison correction evidence."
- `batch_backend_ready_score`: principle "Equals 1.0 only when semantic, distributional, restart, fallback, ordering, and factory gates pass with no speed claims."
- `timing_claimed`: principle "Bare false prevents phase attribution from becoming a timing promotion."
- `software_speedup_claimed`: principle "Bare false prevents the batched boundary from claiming Rust/Python software speedup."
- `hardware_speedup_claimed`: principle "Bare false prevents local CPU work from becoming a board or TSU claim."
- `fpga_or_tsu_used`: principle "Bare false records that no FPGA or TSU participated."
- `inference_substrate`: principle "Declares local CPU Rust/PyO3 one-axis batched sampling, not LLM, GPU, FPGA, or TSU inference."
- `random_seeds`: principle "Records replay seeds for reproduction, parity, restart, fallback, and distributional checks."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether batched backend readiness is proven or honestly null."

### SCENARIO-SAMPLE-5738: Exp5738 Emits Batched Boundary Evidence Or Honest Null

**Given** Exp5723 production one-axis SamplerBackend readiness and Exp5724
matched crossover evidence are hash-pinned
**When** Exp5738 reproduces the n=48 and n=96 cells, profiles the required
phases, and identifies a batch-removable dominant boundary phase
**Then** `OneAxisRustBackend.sample_batch` returns deterministic independent
workload results through the existing explicit factory path while preserving
scalar `sample()` and exact Python fallback behavior
**And** scalar/batch Rust/fallback runs match energy traces, proposals,
accepted transitions, temperature-label exchanges, scheduler positions,
checkpoints, restart suffixes, result order, and >=10000-sample n>=64
distributional parity with multiple-comparison correction
**And** `results/experiment_5738_one_axis_rust_batched_backend.json` is written
with `batch_backend_ready_score=1.0`, all mismatch counts equal zero,
`timing_claimed=false`, `software_speedup_claimed=false`,
`hardware_speedup_claimed=false`, and `fpga_or_tsu_used=false`.

**If** preconditions fail, the reversal is not reproduced, phase evidence does
not justify a batch-boundary fix, or any semantic/distributional/restart/
fallback/ordering/factory gate fails
**Then** the artifact SHALL still emit terminal evidence with
`batch_backend_ready_score=0.0`, all overclaim booleans false, and an
`honest_verdict` starting with `complete:` or `blocked:`.

## Implementation Status (REQ-SAMPLE-5738)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5738 | Planned (`python/carnot/samplers/one_axis_rust_backend.py`, `python/carnot/experiment_5738_one_axis_rust_batched_backend.py`, `results/experiment_5738_one_axis_rust_batched_backend.json`) | Planned (`tests/python/samplers/test_one_axis_rust_backend.py`, `tests/python/test_experiment_5738_one_axis_rust_batched_backend.py`) |

### REQ-SAMPLE-5739: One-Axis Batched Rust/Python 10x Crossover

Carnot MUST provide Exp 5739 at
`python/carnot/experiment_5739_one_axis_batched_10x_crossover.py` and write
`results/experiment_5739_one_axis_batched_10x_crossover.json` without modifying
`scripts/research_conductor.py`. Exp 5739 SHALL benchmark the Exp5738
production-reachable, parity-qualified batched Rust/PyO3 one-axis backend
against the exact Python one-axis fallback under matched CPU work and quality.
The claim is only an end-to-end CPU software throughput claim. A null result is
terminal and honest when matched quality or the 10x interval gate fails.

The protocol SHALL freeze upstream gates, Exp5738 backend and build hashes,
workloads, topology families, problem sizes including `48`, `96`, and `192`
where feasible, batch sizes, seeds, beta ladders, transition budgets,
checkpoints, warmups, measured batch count, quality margins, thread regimes,
CPU affinity, and the 10x claim rule before timing. It SHALL compare two
thread regimes: one physical-core placement and a fixed recorded physical-core
allocation. Both arms SHALL run with the same thread policy in each regime and
the artifact SHALL disclose any implementation asymmetry, including the fact
that Rust uses the PyO3 batched path while Python uses the exact fallback path.
The baseline SHALL NOT be artificially serialized, slept, throttled, or
otherwise weakened outside the preregistered one-core regime requested by this
experiment.

Each `(thread_regime, size, topology_family, batch_size)` cell SHALL execute at
least thirty independent measured batches per arm on identical batch inputs.
The primary latency and throughput SHALL be end-to-end and SHALL NOT subtract
serialization, validation, PyO3 crossing, allocation, checkpoint, or restart
costs. Phase receipts and peak RSS SHALL be preserved as evidence, not used to
subtract overhead from the primary speedup.

Quality SHALL be matched before speedups are interpreted. Matching SHALL require
identical Hamiltonians, transition budgets, energy accounting, sample counts,
seed schedules, checkpoint/restart behavior, result ordering, and
preregistered quality metrics. Mismatched pairs SHALL be excluded visibly with
predeclared reasons and SHALL NOT enter speed distributions. Speedups SHALL be
paired Python/Rust end-to-end ratios with bootstrap confidence intervals and a
multiple-comparison correction across thread-regime and size comparisons.
`rust_batched_10x_ready_score` SHALL equal exactly `1.0` only when matched
quality holds and the adjusted lower confidence bound is at least `10.0` for
two consecutive larger sizes in the same thread regime. Otherwise it SHALL
equal `0.0`, `software_speedup_claimed` SHALL be false, and the artifact SHALL
record a terminal null plus bottleneck evidence.

The terminal artifact SHALL include at minimum these top-level fields:
`schema`, `experiment`, `experiment_id`, `milestone`, `run_date`, `spec_refs`,
`duration_s`, `field_principles`, `preconditions_checked`,
`upstream_gate_receipts`, `software_receipt`, `build_profile`, `cpu_topology`,
`thread_receipts`, `cpu_affinity`, `preregistered_protocol`,
`workload_manifest`, `arm_configs`, `batch_sizes`, `problem_sizes`,
`random_seeds`, `warmup_count`, `measured_batch_count`,
`matched_work_receipts`, `quality_metrics_by_pair`,
`quality_matched_pair_count`, `excluded_pair_reasons`, `end_to_end_times`,
`throughput_distributions`, `phase_times`, `peak_rss_by_arm`,
`paired_speedup_ratios`, `paired_speedup_intervals`, `qualified_10x_sizes`,
`qualified_10x_thread_regime`, `rust_batched_10x_ready_score`,
`timing_claimed`, `software_speedup_claimed`, `gpu_speedup_claimed`,
`hardware_speedup_claimed`, `fpga_or_tsu_used`, `inference_substrate`,
`reproducibility_checksum`, and `honest_verdict`. `timing_claimed` SHALL be
true. `gpu_speedup_claimed`, `hardware_speedup_claimed`, and
`fpga_or_tsu_used` SHALL be false. `inference_substrate` SHALL equal
`matched_cpu_python_rust_batched_sampler_benchmark`.

Required field principles:

- `schema`: principle "Names the artifact contract so downstream validators do not infer fields from experiment number alone."
- `experiment`: principle "Keeps the numeric experiment identifier explicit for corpus indexing without turning it into a metric."
- `experiment_id`: principle "Provides a stable human-readable artifact identity for cross-run provenance."
- `milestone`: principle "Records the conductor milestone without letting milestone status substitute for benchmark evidence."
- `run_date`: principle "Anchors the local CPU software measurement to the requested 2026-07-20 run date."
- `spec_refs`: principle "Binds the artifact to REQ-SAMPLE-5739 and SCENARIO-SAMPLE-5739."
- `duration_s`: principle "Records real wall-clock artifact construction time for fabrication and reproducibility review."
- `field_principles`: principle "Explains why every Exp5739 field exists before a reviewer trusts the JSON shape."
- `preconditions_checked`: principle "Shows every upstream, build, topology, affinity, thread, and workload gate checked before timing interpretation."
- `upstream_gate_receipts`: principle "Pins Exp5724 null timing evidence and Exp5738 batched backend readiness plus source hashes."
- `software_receipt`: principle "Records Python, NumPy, Rust extension, compiler, and source hashes needed to replay the two production arms."
- `build_profile`: principle "Freezes release/debug, PyO3 ABI, bulk-run symbol, and feature state before any 10x timing claim."
- `cpu_topology`: principle "Separates physical and logical CPU evidence from accelerator or board claims."
- `thread_receipts`: principle "Records matched thread policy for both arms in each regime and discloses implementation asymmetry."
- `cpu_affinity`: principle "Shows the one-core and fixed-core placements used for timing."
- `preregistered_protocol`: principle "Freezes workloads, sizes, batches, seeds, ladders, budgets, checkpoints, warmups, repetitions, quality margins, and 10x null rule."
- `workload_manifest`: principle "Lists every size and topology Hamiltonian shared by Python and Rust batch arms."
- `arm_configs`: principle "Names the Rust PyO3 batch and exact Python fallback batch configurations under one production API."
- `batch_sizes`: principle "Makes the amount of independent work per batch explicit instead of implying unmeasured amortization."
- `problem_sizes`: principle "Makes the 48/96/192 large-size panel explicit, including any infeasible-size blocker."
- `random_seeds`: principle "Records the independent batch seed schedule for exact replay."
- `warmup_count`: principle "Prevents cold-start effects from entering primary measurements."
- `measured_batch_count`: principle "Proves every qualified cell has at least thirty independent measured batches."
- `matched_work_receipts`: principle "Proves identical work, energy accounting, samples, transitions, checkpoints, and restart behavior."
- `quality_metrics_by_pair`: principle "Reports matched quality before any speedup ratio enters a distribution."
- `quality_matched_pair_count`: principle "Counts only pairs that passed quality gates before timing intervals."
- `excluded_pair_reasons`: principle "Keeps mismatches visible instead of silently relabeling them as speed evidence."
- `end_to_end_times`: principle "Preserves primary latency evidence including serialization, validation, PyO3, allocation, checkpoint, and restart costs."
- `throughput_distributions`: principle "Reports samples-per-second distributions from the same end-to-end measurements."
- `phase_times`: principle "Preserves bottleneck evidence without subtracting phases from primary timing."
- `peak_rss_by_arm`: principle "Records memory pressure by arm and thread regime so speed is not traded for hidden RSS growth."
- `paired_speedup_ratios`: principle "Reports paired Python/Rust ratios on identical batches rather than unrelated aggregate means."
- `paired_speedup_intervals`: principle "Uses adjusted confidence intervals so a 10x claim cannot rest on noisy multiple comparisons."
- `qualified_10x_sizes`: principle "Lists only consecutive larger sizes whose adjusted lower confidence bound is at least 10.0."
- `qualified_10x_thread_regime`: principle "Names the single thread regime, if any, where the strict consecutive-size 10x rule passed."
- `rust_batched_10x_ready_score`: principle "Equals 1.0 only when matched quality and adjusted intervals prove the strict 10x CPU software rule."
- `timing_claimed`: principle "Bare true declares this is a timing benchmark, unlike Exp5738 readiness-only evidence."
- `software_speedup_claimed`: principle "Bare true is allowed only when the strict 10x CPU software rule passes."
- `gpu_speedup_claimed`: principle "Bare false prevents this CPU benchmark from reopening GPU claims."
- `hardware_speedup_claimed`: principle "Bare false prevents local CPU software timing from becoming a board or TSU claim."
- `fpga_or_tsu_used`: principle "Bare false records that no FPGA, TSU, or board participated."
- `inference_substrate`: principle "Declares matched CPU Python/Rust batched one-axis sampler benchmarking, not LLM inference or accelerator timing."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether the strict 10x rule passed or terminally failed."

### SCENARIO-SAMPLE-5739: Exp5739 Emits Matched Batched 10x Evidence Or Terminal Null

**Given** Exp5738 batched one-axis backend readiness and Exp5724 terminal
Rust/Python CPU crossover null are hash-pinned
**When** Exp5739 runs identical batched workloads through the Rust PyO3
`sample_batch` path and exact Python fallback `sample_batch` path in the
one-core and fixed-core regimes
**Then** it records matched work counters, quality metrics, phase receipts,
peak RSS, end-to-end latency, throughput, paired speedup distributions,
adjusted intervals, and writes
`results/experiment_5739_one_axis_batched_10x_crossover.json`
**And** `rust_batched_10x_ready_score=1.0` only when the adjusted lower
confidence bound is at least `10.0` at two consecutive larger sizes in the same
thread regime with matched quality.

**If** upstream gates fail, work or quality mismatches, fewer than thirty
independent measured batches exist in any qualified cell, the adjusted 10x
consecutive-size rule fails, or any GPU/FPGA/TSU/hardware claim appears
**Then** the artifact SHALL still emit terminal evidence with
`rust_batched_10x_ready_score=0.0`, `software_speedup_claimed=false`,
`timing_claimed=true`, `gpu_speedup_claimed=false`,
`hardware_speedup_claimed=false`, `fpga_or_tsu_used=false`, and an
`honest_verdict` starting with `complete:` or `blocked:`.

## Implementation Status (REQ-SAMPLE-5739)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5739 | Planned (`python/carnot/experiment_5739_one_axis_batched_10x_crossover.py`, `results/experiment_5739_one_axis_batched_10x_crossover.json`) | Planned (`tests/python/test_experiment_5739_one_axis_batched_10x_crossover.py`) |

### REQ-SAMPLE-5751: One-Axis Rust Restart Parity Repair

Carnot MUST repair the production-reachable one-axis Rust/PyO3 batch restart
path when Exp5739 restart receipts diverge despite identical resumed sampler
semantics. The repair SHALL first reproduce at least one Exp5739
`restart_match` exclusion at `n=96` or `n=192`, identify the first divergent
checkpoint-boundary field, and patch the narrowest production adapter path
without adding a second sampler API or modifying `scripts/research_conductor.py`.

Checkpoint-boundary receipts SHALL canonicalize diagnostic floats so Python and
Rust logs that are semantically equal, including signed-zero cases, hash
identically. The sampler state schema SHALL remain backward compatible unless a
versioned migration is explicitly required. Corrupted checkpoints, wrong
schema, wrong input, wrong seed, wrong ladder, and malformed checkpoint payloads
SHALL continue to fail closed.

The repair SHALL inject interruption/resume checks before and after restart and
checkpoint transitions at `n=48`, `n=96`, and `n=192`. It SHALL compare
uninterrupted versus resumed Python and Rust traces exactly for energies,
proposals, scheduler decisions, restart counts, sample counts, RNG state and
schema, checkpoint hashes, final samples, exact fallback equivalence, and
corrupted-checkpoint rejection. Distributional parity SHALL be reported for the
large non-enumerable cases using the existing one-axis quality summaries. No
timing promotion is in scope: `timing_claimed=false` and
`hardware_speedup_claimed=false`.

Experiment 5751 SHALL write
`results/experiment_5751_rust_restart_parity_repair.json` with at least these
top-level fields: `field_principles`, `preconditions_checked`, `spec_refs`,
`upstream_artifact_hashes`, `rust_toolchain`, `python_version`, `pyo3_version`,
`release_build_receipt`, `reproduced_failure_receipts`,
`first_divergence_receipt`, `root_cause`, `changed_files`,
`checkpoint_schema_before`, `checkpoint_schema_after`, `migration_receipt`,
`interruption_injection_manifest`, `energy_parity`, `proposal_parity`,
`scheduler_parity`, `rng_parity`, `restart_parity`, `checkpoint_parity`,
`sample_count_parity`, `distributional_parity`, `fallback_equivalence`,
`production_backend_reachable`, `restart_parity_ready_score`,
`timing_claimed`, `hardware_speedup_claimed`, `random_seeds`, `test_commands`,
`test_exit_codes`, `reproducibility_checksum`, and `honest_verdict`.

Required field principles:

- `field_principles`: principle "Explains why every restart-repair field exists before downstream timing work can consume the artifact."
- `preconditions_checked`: principle "Records local Rust, PyO3, Python, release-build, affinity, memory, disk, upstream-hash, and mismatch-reproduction gates before declaring repair."
- `spec_refs`: principle "Binds the artifact to REQ-SAMPLE-5751 and SCENARIO-SAMPLE-5751."
- `upstream_artifact_hashes`: principle "Pins Exp5738 and Exp5739 evidence so the repair addresses the observed production divergence."
- `rust_toolchain`: principle "Records rustc and cargo versions used to build and test the PyO3 path."
- `python_version`: principle "Records the Python runtime used by the production adapter and tests."
- `pyo3_version`: principle "Records the binding ABI dependency version relevant to restart receipts."
- `release_build_receipt`: principle "Shows whether a release PyO3 build was attempted and whether it completed."
- `reproduced_failure_receipts`: principle "Preserves at least one original Exp5739 restart_match failure before patch interpretation."
- `first_divergence_receipt`: principle "Documents the first divergent field, event, and hash boundary instead of patching by guess."
- `root_cause`: principle "States the causal bug in production-reachable terms."
- `changed_files`: principle "Lists the exact implementation, spec, test, and artifact files changed."
- `checkpoint_schema_before`: principle "Pins the checkpoint schema accepted before the repair."
- `checkpoint_schema_after`: principle "Pins the checkpoint schema accepted after the repair."
- `migration_receipt`: principle "Declares whether migration was required or why compatibility is preserved."
- `interruption_injection_manifest`: principle "Lists every before/after restart and checkpoint transition case at n=48, n=96, and n=192."
- `energy_parity`: principle "Proves resumed Rust and Python energy diagnostics remain equal."
- `proposal_parity`: principle "Proves resumed Rust and Python proposal diagnostics remain equal."
- `scheduler_parity`: principle "Proves within and swap scheduler decisions remain equal."
- `rng_parity`: principle "Proves RNG state and consumption schema remain equal across uninterrupted and resumed paths."
- `restart_parity`: principle "Gates the repaired suffix hashes and restart counts."
- `checkpoint_parity`: principle "Gates checkpoint hashes, schema, and corrupted-checkpoint rejection."
- `sample_count_parity`: principle "Proves resumed and uninterrupted paths keep the same retained sample counts."
- `distributional_parity`: principle "Records large-case non-enumerable parity evidence without promoting timing."
- `fallback_equivalence`: principle "Proves exact Python fallback remains equivalent to the Rust production path."
- `production_backend_reachable`: principle "Proves the existing OneAxisRustBackend.sample_batch path is the repaired path."
- `restart_parity_ready_score`: principle "Equals 1.0 only when reproduction, repair, interruption, fallback, corruption, and no-speed gates pass."
- `timing_claimed`: principle "Bare false prevents this repair from becoming a speed claim."
- `hardware_speedup_claimed`: principle "Bare false prevents CPU restart evidence from becoming a board or hardware claim."
- `random_seeds`: principle "Records replay seeds for failure reproduction and interruption injection."
- `test_commands`: principle "Lists the verification commands run for the repair."
- `test_exit_codes`: principle "Records command outcomes honestly, including any bounded or blocked checks."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether restart parity is repaired."

### SCENARIO-SAMPLE-5751: Exp5751 Emits Restart Parity Repair Evidence

**Given** Exp5738 qualified the production `sample_batch` backend and Exp5739
excluded larger-size rows for `restart_match`
**When** the first larger-size restart divergence is reproduced and repaired in
the production one-axis adapter
**Then** Rust and Python uninterrupted and resumed traces at `n=48`, `n=96`,
and `n=192` have matching semantic diagnostics and repaired checkpoint-boundary
hashes
**And** fallback and corrupted-checkpoint controls still pass
**And** `results/experiment_5751_rust_restart_parity_repair.json` is written
with `restart_parity_ready_score=1.0`, `timing_claimed=false`, and
`hardware_speedup_claimed=false` only when every restart repair gate passes.

**If** the original Exp5739 mismatch cannot be reproduced, root cause is not
identified, checkpoint compatibility is broken without migration, interruption
parity fails, fallback equivalence regresses, corrupted checkpoints are
accepted, or any speed/hardware claim appears
**Then** the artifact SHALL still emit terminal evidence with
`restart_parity_ready_score=0.0` and an `honest_verdict` starting with
`blocked:`.

## Implementation Status (REQ-SAMPLE-5751)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5751 | Implemented (`python/carnot/samplers/one_axis_rust_backend.py`, `results/experiment_5751_rust_restart_parity_repair.json`) | Implemented (`tests/python/samplers/test_one_axis_rust_backend.py`, `tests/python/test_experiment_5751_rust_restart_parity_repair.py`) |

### REQ-SAMPLE-5758: One-Axis Rust Parity Scalar Bridge

Carnot MUST provide Exp5758 at
`scripts/experiment_5758_rust_parity_scalar_bridge.py` and write
`results/experiment_5758_rust_parity_scalar_bridge.json` without modifying
`python/carnot/samplers/one_axis_rust_backend.py`, without modifying
`scripts/research_conductor.py`, without rerunning Exp5752 timing work, and
without changing the Exp5751 structured parity receipts.

The bridge SHALL derive `distributional_parity_score=1.0` only when
Exp5751's `distributional_parity.passed` is explicitly true and every
distributional case at `n=48`, `n=96`, and `n=192` has zero
`energy_histogram_tv`, zero `mean_energy_delta_abs`, and zero
`best_energy_delta_abs`. It SHALL derive `fallback_equivalence_score=1.0`
only when `fallback_equivalence.exact_fallback_equivalence` is explicitly true
and every fallback case records `rust_python_samples_match=true`. It SHALL
derive `production_backend_reachable_score=1.0` only when
`production_backend_reachable.passed`, `sample_batch_callable`, and
`scalar_api_unchanged` are explicitly true and
`second_sampler_api_added=false`.

The bridge SHALL verify that every semantic parity family shares the same
`n=48`, `n=96`, and `n=192` case denominator, that restart suffix hashes match,
that interruption manifests match uninterrupted Rust and Python suffixes, that
checkpoint corruption remains rejected, that the release PyO3 build receipt
completed with exit code zero, and that all Exp5751 source hashes still match
local files. If any predicate is missing, object-wrapped, ambiguous,
contradicts its row evidence, or depends on stale source bytes, every affected
derived score SHALL be `0.0` and the bridge SHALL block rather than promoting
`rust_benchmark_gate_ready_score`.

The downstream readiness field `rust_benchmark_gate_ready_score` SHALL equal
`1.0` only when `restart_parity_ready_score`, `distributional_parity_score`,
`fallback_equivalence_score`, and `production_backend_reachable_score` are all
`1.0`, all source and receipt hashes match, unsafe synthesis is zero, and the
planned Exp5764 conductor gate replay passes through
`scripts/conductor_gates.evaluate_gates()` against bare numeric fields. No
timing promotion is in scope: `timing_claimed=false` and
`hardware_speedup_claimed=false`.

### SCENARIO-SAMPLE-5758: Exp5758 Bridges Repaired One-Axis Rust Parity To Bare Gates

**Given** Exp5751 records repaired restart parity at `n=48`, `n=96`, and
`n=192`
**When** Exp5758 reads the artifact and derives scalar bridge predicates
**Then** it preserves the structured receipts by hash, emits bare numeric
scores for downstream gates, passes the planned Exp5764 gate replay, leaves the
sampler implementation untouched, and makes no timing or hardware speedup
claim.

**If** any distributional, fallback, production-reachability, interruption,
checkpoint, source-hash, or gate-replay predicate is absent, object-wrapped,
ambiguous, contradictory, or stale
**Then** the bridge SHALL emit terminal evidence with
`rust_benchmark_gate_ready_score=0.0` and an `honest_verdict` starting with
`blocked:`.

## Implementation Status (REQ-SAMPLE-5758)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5758 | Planned (`scripts/experiment_5758_rust_parity_scalar_bridge.py`, `results/experiment_5758_rust_parity_scalar_bridge.json`) | Planned (`tests/python/test_experiment_5758_rust_parity_scalar_bridge.py`) |

### REQ-SAMPLE-5764: One-Axis Profiled Allocation-Free Hot Path

Carnot MUST provide Exp5764 at
`python/carnot/experiment_5764_one_axis_profiled_allocation_free_hot_path.py`
and write
`results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json`
without modifying `scripts/research_conductor.py`. Exp5764 SHALL profile the
production-reachable one-axis Rust/PyO3 `SamplerBackend` batch path before any
speed interpretation and SHALL optimize only the measured dominant
end-to-end phase.

Before profiling, Exp5764 SHALL verify local Rust, Python, and PyO3 toolchains,
release-build capability, upstream artifact hashes, source hashes, CPU
affinity, CPU governor observability or blocker, fixed thread counts, free RAM,
free disk, absence of competing benchmark processes, and replay of Exp5751
restart parity. If any provenance or host-stability gate fails, profiling SHALL
stop and the artifact SHALL report `status="blocked"`.

The profiling ledger SHALL freeze Python/Rust semantics, beta ladder, seeds,
batch shapes, restart schedule, checkpoint schema, quality metrics, and the
production `SamplerBackend` entrypoint. It SHALL measure serialization, Python
preparation, PyO3 crossing, Rust batch allocation, worker scheduling,
within/swap kernel work, validation, result conversion, checkpoint, and restart
phases for warm steady-state and restart-containing batches at `n=48`, `n=96`,
`n=192`, and one larger feasible size. The dominant phase SHALL be selected
from preregistered median phase share with a confidence interval before
optimization.

The optimized path SHALL preserve the one-axis target distribution and SHALL
NOT add a second sampler. When the measured dominant phase is result conversion
or boundary allocation, the production backend MAY use a backward-compatible
compact Rust/PyO3 path that reuses contiguous buffers, returns slice/view-based
sample storage, avoids per-sample Rust heap allocation in the steady-state hot
path, and records a fixed-worker receipt. Full diagnostic decision logs SHALL
remain available for semantic, scheduler, RNG, checkpoint, restart, fallback,
and corrupted-checkpoint tests.

Allocation counters SHALL report before and after counts for Python and Rust
hot-path allocation boundaries. Any remaining allocation SHALL be explicitly
documented and SHALL not be hidden from `optimized_path_ready_score`. Exp5764
SHALL rerun exact semantic, scheduler, RNG, checkpoint, restart, sample-count,
fallback, production-entrypoint, corrupted-checkpoint, and distributional
diagnostics across at least ten seeds per measured cell. Exp5764 MAY profile
local CPU timing but SHALL NOT claim a 10x result:
`timing_promotion_claimed=false`, `hardware_speedup_claimed=false`, and
`two_axis_exchange_reopened=false`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `status`, `preconditions_checked`, `spec_refs`,
`upstream_artifact_hashes`, `source_hashes_before`, `source_hashes_after`,
`rust_toolchain`, `python_version`, `pyo3_version`, `host_receipts`,
`release_build_receipt`, `affinity_receipt`, `phase_definitions`,
`phase_timing_receipts`, `phase_share_by_size`, `dominant_phase`,
`dominant_phase_selection_receipt`, `allocation_counts_before`,
`allocation_counts_after`, `buffer_reuse_receipts`, `worker_pool_receipts`,
`changed_files`, `checkpoint_compatibility`, `restart_parity_receipts`,
`fallback_equivalence_receipts`, `semantic_parity_score`,
`distributional_parity_score`, `production_backend_reachable_score`,
`producer_gate_fields`, `optimized_path_ready_score`, `timing_promotion_claimed`,
`hardware_speedup_claimed`, `two_axis_exchange_reopened`,
`inference_substrate`, `random_seeds`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and `honest_verdict`.
`semantic_parity_score`, `distributional_parity_score`,
`production_backend_reachable_score`, and `optimized_path_ready_score` SHALL be
bare numeric `0.0` or `1.0` fields and SHALL also be listed by
`producer_gate_fields`. `inference_substrate` SHALL equal
`local_rust_pyo3_cpu_release_profile_and_parity`.

Required field principles:

- `field_principles`: principle "Explains why every Exp5764 field exists before downstream allocation or timing work can consume the artifact."
- `status`: principle "Bare terminal state lets gates distinguish complete from blocked without parsing prose."
- `preconditions_checked`: principle "Records provenance, build, host-stability, resource, and Exp5751 replay gates before profiling starts."
- `spec_refs`: principle "Binds the artifact to REQ-SAMPLE-5764 and SCENARIO-SAMPLE-5764."
- `upstream_artifact_hashes`: principle "Pins Exp5751, Exp5758, and Exp5739 evidence so parity and prior timing context cannot drift."
- `source_hashes_before`: principle "Hashes the source bytes that were profiled before the compact hot-path edit."
- `source_hashes_after`: principle "Hashes the source bytes after optimization so reviewers can separate measured before/after code."
- `rust_toolchain`: principle "Records rustc and cargo versions used for release PyO3 profiling."
- `python_version`: principle "Records the CPython runtime used for SamplerBackend execution."
- `pyo3_version`: principle "Records the binding dependency version relevant to PyO3 boundary allocation."
- `host_receipts`: principle "Authenticates CPU, governor, memory, disk, thread, and competing-process observations."
- `release_build_receipt`: principle "Shows the release PyO3 build command and exit code."
- `affinity_receipt`: principle "Records fixed CPU placement used for profiling and parity replay."
- `phase_definitions`: principle "Defines every measured phase before selecting a dominant phase."
- `phase_timing_receipts`: principle "Preserves raw warm and restart-containing phase samples by size."
- `phase_share_by_size`: principle "Shows median phase share and confidence interval for each measured size."
- `dominant_phase`: principle "Names the preregistered dominant end-to-end phase selected from phase-share evidence."
- `dominant_phase_selection_receipt`: principle "Explains why only the selected measured phase was optimized."
- `allocation_counts_before`: principle "Reports pre-optimization allocation boundaries rather than inferring them from time."
- `allocation_counts_after`: principle "Reports post-optimization allocation boundaries and any unavoidable remaining allocation."
- `buffer_reuse_receipts`: principle "Proves the compact path uses contiguous reused work buffers instead of per-sample heap buffers."
- `worker_pool_receipts`: principle "Documents fixed worker policy and that no dynamic per-sample worker scheduling is introduced."
- `changed_files`: principle "Lists the exact spec, implementation, test, and artifact files changed."
- `checkpoint_compatibility`: principle "Proves compact execution preserves the v1 checkpoint schema and restartable state."
- `restart_parity_receipts`: principle "Proves optimized checkpoints resume with the same samples and suffix hashes."
- `fallback_equivalence_receipts`: principle "Proves exact Python fallback remains equivalent for the profiled cells."
- `semantic_parity_score`: principle "Bare scalar equals 1.0 only when semantic, scheduler, RNG, checkpoint, restart, fallback, and sample-count checks pass."
- `distributional_parity_score`: principle "Bare scalar equals 1.0 only when matched distribution diagnostics pass across the required seeds and cells."
- `production_backend_reachable_score`: principle "Bare scalar equals 1.0 only when the existing explicit one_axis_rust production entrypoint reaches the optimized path."
- `producer_gate_fields`: principle "Lists the bare scalar downstream gates without wrapping their values in objects."
- `optimized_path_ready_score`: principle "Bare scalar equals 1.0 only when profiling, allocation, buffer reuse, parity, and no-speed gates all pass."
- `timing_promotion_claimed`: principle "Bare false prevents profiling from becoming a speed promotion."
- `hardware_speedup_claimed`: principle "Bare false prevents local CPU profiling from becoming an accelerator or board claim."
- `two_axis_exchange_reopened`: principle "Bare false keeps retired two-axis exchange out of scope."
- `inference_substrate`: principle "Declares local Rust/PyO3 CPU release profiling and parity, not LLM inference or accelerator timing."
- `random_seeds`: principle "Records replay seeds for profiling, restart, fallback, and distributional diagnostics."
- `test_commands`: principle "Lists the commands used to verify the implementation and artifact."
- `test_exit_codes`: principle "Records verification command outcomes honestly."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether the profiled allocation-free hot path is ready."

### SCENARIO-SAMPLE-5764: Exp5764 Emits Profiled Allocation-Free Hot-Path Evidence

**Given** Exp5751 restart parity and Exp5758 scalar parity gates are
hash-pinned
**When** Exp5764 profiles the production one-axis batch path at `n=48`,
`n=96`, `n=192`, and one larger feasible size
**Then** it records a campaign-style phase ledger, selects the dominant phase
from median phase share with confidence intervals, optimizes only that phase,
proves compact-path parity and checkpoint compatibility, emits bare producer
gate scores, and writes
`results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json`
with `timing_promotion_claimed=false`,
`hardware_speedup_claimed=false`, and `two_axis_exchange_reopened=false`.

**If** provenance, release build, host stability, Exp5751 replay, phase
selection, allocation counts, buffer reuse, fallback equivalence, production
reachability, checkpoint compatibility, or distributional parity fails
**Then** the artifact SHALL still emit all required fields with
`optimized_path_ready_score=0.0` and an `honest_verdict` starting with
`blocked:`.

## Implementation Status (REQ-SAMPLE-5764)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5764 | Implemented (`python/carnot/samplers/one_axis_rust_backend.py`, `crates/carnot-python/src/one_axis_tempering.rs`, `crates/carnot-samplers/src/one_axis_tempering.rs`, `python/carnot/experiment_5764_one_axis_profiled_allocation_free_hot_path.py`, `results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json`) | Implemented (`tests/python/samplers/test_one_axis_rust_backend.py`, `tests/python/test_experiment_5764_one_axis_profiled_allocation_free_hot_path.py`, `crates/carnot-samplers/tests/one_axis_tempering.rs`) |

### REQ-SAMPLE-5765: One-Axis Final Matched-Quality 10x Crossover

Carnot MUST provide Exp5765 at
`python/carnot/experiment_5765_one_axis_final_10x_crossover.py` and write
`results/experiment_5765_one_axis_final_10x_crossover.json` without modifying
`scripts/research_conductor.py`. Exp5765 is the final preregistered attempt to
qualify the PRD NFR-01 10x software claim for the Exp5764 profiled
allocation-reduced production one-axis Rust/PyO3 technique after the Exp5751
restart repair. A repeated null SHALL retire this allocation-free one-axis
PyO3 technique only; it SHALL NOT retire all future Rust work, hardware work,
or reopen two-axis exchange.

Before timing, Exp5765 SHALL verify optimized source hashes, release binary
hashes, Rust/Python/PyO3 versions, upstream Exp5751/Exp5739/Exp5764 hashes,
CPU affinity, governor observability or blocker, fixed thread counts, absence
of competing benchmark load, minimum free RAM and disk, stable warmup,
restart/checkpoint fixtures, and baseline manifests. If the host or evidence
chain is unstable, Exp5765 SHALL bail before timed science rows and emit a
`status="blocked"` artifact.

The benchmark manifest SHALL be frozen before timed rows are collected. It
SHALL include `n=48`, `n=96`, `n=192`, and one larger feasible size, one
production Rust/PyO3 release arm, one exact Python fallback arm, identical
Hamiltonians, batch counts, seeds, warmup budget, retained-sample budget,
restart schedule, checkpoint schema, quality tolerances, outlier policy,
paired bootstrap method, and the final consecutive larger-size 10x rule. Each
cell SHALL collect at least thirty paired measured batches after warmup,
interleave arm order, pin or record CPU affinity, and preserve raw per-batch
end-to-end timing and phase receipts.

For every pair, Exp5765 SHALL require semantic parity, restart/checkpoint
parity, exact fallback equivalence, matched energy, feasibility, acceptance,
retained-sample count, effective sample size or bounded proxy,
autocorrelation or bounded proxy, and distributional diagnostics before the
pair can enter speed summaries. Exclusions SHALL be allowed only for
preregistered reasons and every exclusion SHALL be reported. Speedups SHALL be
paired Python/Rust end-to-end ratios. Exp5765 SHALL compute bootstrap 95%
confidence intervals by size and for an end-to-end aggregate, without mixing
setup-only, kernel-only, simulated, debug, or unmatched rows into the NFR-01
headline.

Bare `rust_10x_claimed` SHALL be true only when all matched-quality and parity
gates pass and the paired lower confidence bound is at least `10.0` at two
consecutive larger sizes. Otherwise `rust_10x_claimed=false`,
`rust_10x_retired=true`, the honest measured speedup SHALL be preserved, and
`remaining_bottleneck` SHALL name the bottleneck that prevented the 10x claim.
`hardware_speedup_claimed=false`, `two_axis_exchange_reopened=false`, and
`inference_substrate` SHALL equal
`local_rust_pyo3_cpu_release_matched_benchmark`.

The terminal artifact SHALL include at minimum these top-level fields:
`field_principles`, `status`, `preconditions_checked`, `spec_refs`,
`upstream_artifact_hashes`, `source_hashes`, `release_binary_hashes`,
`host_receipts`, `affinity_receipt`, `benchmark_manifest`,
`benchmark_manifest_hash`, `cell_sizes`, `paired_batches_per_cell`,
`raw_timing_receipts`, `warmup_receipts`, `quality_metrics_by_cell`,
`semantic_parity_by_cell`, `restart_parity_by_cell`,
`distributional_parity_by_cell`, `fallback_equivalence`,
`production_backend_reachable`, `exclusion_manifest`,
`speedup_median_by_size`, `speedup_lcb_by_size`, `speedup_ucb_by_size`,
`consecutive_larger_size_rule_passed`, `matched_quality_gate_passed`,
`rust_10x_claimed`, `rust_10x_retired`, `remaining_bottleneck`,
`nfr01_status`, `hardware_speedup_claimed`, `two_axis_exchange_reopened`,
`inference_substrate`, `random_seeds`, `duration_s`, `test_commands`,
`test_exit_codes`, `reproducibility_checksum`, and `honest_verdict`.

Required field principles:

- `field_principles`: principle "Explains why every Exp5765 field exists before a reviewer trusts the final NFR-01 claim or retirement."
- `status`: principle "Bare terminal state lets gates distinguish complete from blocked without parsing prose."
- `preconditions_checked`: principle "Records provenance, release-build, host-stability, warmup, restart, and manifest gates before timing starts."
- `spec_refs`: principle "Binds the artifact to REQ-SAMPLE-5765 and SCENARIO-SAMPLE-5765."
- `upstream_artifact_hashes`: principle "Pins Exp5751, Exp5739, and Exp5764 so final timing cannot drift from repaired and optimized provenance."
- `source_hashes`: principle "Hashes optimized source bytes used by the final release benchmark."
- `release_binary_hashes`: principle "Hashes the local release extension artifacts used by the Rust/PyO3 path."
- `host_receipts`: principle "Authenticates CPU, governor, thread, memory, disk, and competing-load observations."
- `affinity_receipt`: principle "Records fixed CPU placement used for warmup and paired timing."
- `benchmark_manifest`: principle "Freezes sizes, batches, seeds, budgets, paths, tolerances, outlier policy, bootstrap method, and claim rule before timing."
- `benchmark_manifest_hash`: principle "Content-addresses the preregistered manifest independently of timing results."
- `cell_sizes`: principle "Makes the 48/96/192 plus larger feasible final panel explicit."
- `paired_batches_per_cell`: principle "Proves each qualified size has at least thirty paired measured batches after warmup."
- `raw_timing_receipts`: principle "Preserves raw interleaved per-batch end-to-end timing and phase receipts before summaries."
- `warmup_receipts`: principle "Shows warmup stability before science rows enter the benchmark."
- `quality_metrics_by_cell`: principle "Reports energy, feasibility, acceptance, retained count, ESS, and autocorrelation diagnostics before speed claims."
- `semantic_parity_by_cell`: principle "Proves exact semantic parity where exactness applies."
- `restart_parity_by_cell`: principle "Proves checkpoint and restart parity remained repaired during timing."
- `distributional_parity_by_cell`: principle "Reports distributional diagnostics so speed cannot hide wrong samples."
- `fallback_equivalence`: principle "Proves the Python exact fallback and Rust production path remain matched."
- `production_backend_reachable`: principle "Proves the explicit one_axis_rust SamplerBackend reached the optimized release path."
- `exclusion_manifest`: principle "Reports every excluded pair with a preregistered reason instead of silently filtering timing."
- `speedup_median_by_size`: principle "Preserves honest measured paired speedup medians by size."
- `speedup_lcb_by_size`: principle "Applies the paired bootstrap lower bound used by the 10x claim rule."
- `speedup_ucb_by_size`: principle "Preserves the paired bootstrap upper bound for uncertainty review."
- `consecutive_larger_size_rule_passed`: principle "Bare gate proves two consecutive larger sizes have lower confidence bound >=10.0."
- `matched_quality_gate_passed`: principle "Bare gate proves all parity and quality checks passed before any NFR-01 claim."
- `rust_10x_claimed`: principle "Bare true is allowed only under the final consecutive larger-size confidence rule."
- `rust_10x_retired`: principle "Bare true retires only this allocation-free one-axis PyO3 technique after a repeated null."
- `remaining_bottleneck`: principle "Names the measured bottleneck left after allocation reduction when the 10x rule fails."
- `nfr01_status`: principle "States whether PRD NFR-01 is qualified, retired for this technique, or blocked."
- `hardware_speedup_claimed`: principle "Bare false prevents local CPU software timing from becoming a hardware claim."
- `two_axis_exchange_reopened`: principle "Bare false keeps retired two-axis exchange out of scope."
- `inference_substrate`: principle "Declares local Rust/PyO3 CPU release matched benchmarking, not LLM, GPU, FPGA, or TSU inference."
- `random_seeds`: principle "Records the paired seed schedule for replay."
- `duration_s`: principle "Records real wall-clock artifact construction time for fabrication review."
- `test_commands`: principle "Lists the verification commands run for the final benchmark."
- `test_exit_codes`: principle "Records command outcomes honestly."
- `reproducibility_checksum`: principle "Content-addresses the complete artifact after blanking the self-checksum field."
- `honest_verdict`: principle "Starts complete: or blocked: and states whether NFR-01 was claimed, retired for this technique, or blocked."

### SCENARIO-SAMPLE-5765: Exp5765 Emits Final 10x Claim Or Technique Retirement

**Given** Exp5751 restart parity repair, Exp5739 terminal 10x null context, and
Exp5764 profiled allocation-reduced production path are hash-pinned
**When** Exp5765 runs the preregistered matched-quality release benchmark at
`n=48`, `n=96`, `n=192`, and one larger feasible size with at least thirty
paired batches per cell
**Then** it records preconditions, raw timing, warmup, quality, semantic,
restart/checkpoint, fallback, distributional, exclusion, paired speedup, and
bootstrap confidence evidence and writes
`results/experiment_5765_one_axis_final_10x_crossover.json`
**And** `rust_10x_claimed=true` only when the paired lower confidence bound is
at least `10.0` at two consecutive larger sizes and all quality/parity gates
pass.

**If** any precondition blocks, fewer than thirty paired measured batches exist
in a qualified cell, matched-quality gates fail, the consecutive larger-size
10x confidence rule fails, or any hardware/two-axis acceleration claim appears
**Then** the artifact SHALL still emit all required fields with
`rust_10x_claimed=false`, `hardware_speedup_claimed=false`,
`two_axis_exchange_reopened=false`, and an `honest_verdict` starting with
`complete:` or `blocked:`; a repeated timing null SHALL set
`rust_10x_retired=true` only for this allocation-free one-axis PyO3 technique.

## Implementation Status (REQ-SAMPLE-5765)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-5765 | Planned (`python/carnot/experiment_5765_one_axis_final_10x_crossover.py`, `results/experiment_5765_one_axis_final_10x_crossover.json`) | Planned (`tests/python/test_experiment_5765_one_axis_final_10x_crossover.py`) |

### REQ-SAMPLE-1746: EqM Sampler Profiling Experiment

Carnot MUST provide an experiment script `scripts/experiment_1746_eqm_profile.py`
that profiles the EqM test-time sampler using `torch.profiler` to identify
the latency bottleneck during gradient updates.

Sub-requirements:
- REQ-SAMPLE-1746-1: The script SHALL configure `MODEL_SPECS` to use
  `unsloth/Qwen3.6-35B-A3B-GGUF`.
- REQ-SAMPLE-1746-2: The script SHALL implement PyTorch/CUDA profiling around
  the EqM sampling process to capture timing metrics.
- REQ-SAMPLE-1746-3: The script SHALL write `results/experiment_1746_profile.json`
  with the captured latency and profiler metrics.

### SCENARIO-SAMPLE-1746: Exp 1746 Profiler Extracts Timing Metrics

**Given** the Exp 1746 EqM profile script
**When** it is executed
**Then** it successfully runs the EqM sampler under `torch.profiler`
**And** it writes a valid JSON artifact with timing metrics.

### REQ-SAMPLE-1748: Sparse EqM Hardware Benchmark

Carnot MUST provide an experiment script `scripts/experiment_1748_benchmark.py`
that benchmarks the hardware latency of the sparse EqM sampler.

Sub-requirements:
- REQ-SAMPLE-1748-1: The script SHALL benchmark the sparse EqM latency across a batch of 100 samples.
- REQ-SAMPLE-1748-2: The script SHALL output a JSON artifact to `results/experiment_1748_benchmark.json`.
- REQ-SAMPLE-1748-3: The artifact SHALL include the experiment id, spec refs, and latency metrics.

### SCENARIO-SAMPLE-1748: Exp 1748 Computes Hardware Benchmark Latency

**Given** the Exp 1748 benchmark script
**When** it is executed
**Then** it successfully runs the sparse EqM sampler
**And** it writes a valid JSON artifact detailing latency statistics.

### REQ-SAMPLE-1834: THRML Multi-Period Turnover Constraints

Carnot MUST support multi-period sampling with turnover constraints via the
THRML/Extropic adapter.

Sub-requirements:
- REQ-SAMPLE-1834-1: The adapter SHALL flatten multi-period biases and couplings
  into a single space, augmenting them with the required turnover penalty.
- REQ-SAMPLE-1834-2: Turnover constraints SHALL map such that variables at t and
  t+1 are encouraged to be equal.

### SCENARIO-SAMPLE-1834: Exp 1834 Writes Terminal JSON Artifact

**Given** the multi-period constraint testing mechanism
**When** the multi-period sampling API is invoked
**Then** turnover tests pass and the adapter scales correctly
**And** it writes `results/experiment_1834_thrml_turnover.json`.

### REQ-SAMPLE-1686: Exp 1682 Beta-Dependent Bias Correction

Carnot MUST expose a bias-correction function and a corrected-mean estimator for
the parallel Ising Gibbs sampler, calibrated from Exp 1682's finding that the
systematic per-spin underestimate is linear in beta.

Sub-requirements:
- REQ-SAMPLE-1686-1: `ising_mean_bias_correction(beta)` SHALL return the additive
  scalar correction derived from the OLS linear fit to Exp 1682 sweep_b data
  (n_spins=128, J=1/(n-1), Curie-Weiss). Calibration domain: beta in [1.05, 1.5].
- REQ-SAMPLE-1686-2: `corrected_magnetization_mean(samples, beta)` SHALL compute
  the per-spin sample mean and add the scalar correction uniformly across all spins.
- REQ-SAMPLE-1686-3: The correction SHALL be implemented in
  `python/carnot/samplers/parallel_ising.py` without modifying the existing
  `sample()` API (backward-compatible).
- REQ-SAMPLE-1686-4: A results artifact SHALL be written to
  `results/experiment_1686_thrml_bias_fix.json` with schema, bias_correction_applied,
  old_bias, new_bias, and acceptance_gate_passed fields.

### SCENARIO-SAMPLE-1686: Corrected Mean Reduces Systematic Underestimate

**Given** a ParallelIsingSampler run at a calibration beta in [1.05, 1.5]
**When** `corrected_magnetization_mean()` is called on the resulting samples
**Then** the returned mean differs from the raw mean by the expected linear correction
**And** the corrected bias is strictly smaller in magnitude than the raw bias.

### REQ-SAMPLE-1845: Simulated HILED Interface for KV260

Carnot MUST provide a simulated Hardware-In-The-Loop Energy Decoding (HILED) boundary inside `python/carnot/samplers/hiled.py`. This interface acts as a simulator target for Potts integration targeting the KV260 execution pipeline.

Sub-requirements:
- REQ-SAMPLE-1845-1: The module SHALL provide a `HiledSimulator` class capable of simulating offloaded energy decoding.
- REQ-SAMPLE-1845-2: The simulator SHALL support executing a mock execution pipeline for the KV260 target.
- REQ-SAMPLE-1845-3: An experiment deliverable SHALL be written to `results/experiment_1845_hiled.json` to prove the interface is functional.

### SCENARIO-SAMPLE-1845: HILED Interface Execution

**Given** a KV260 target configuration
**When** the simulated HILED execution pipeline is invoked
**Then** it successfully mimics offloaded decoding and records metrics
**And** it writes `results/experiment_1845_hiled.json`.

### REQ-SAMPLE-1935: Continuous-Latent FAR-Inspired Constraint Sampler

Carnot MUST provide a continuous-latent sampling layer inspired by Fast
Autoregressive (FAR) models that enables high-speed surrogate shortcut
constraint checking, implemented in
`python/carnot/samplers/continuous_latent.py`.

This sampler wraps the standard Langevin or EqM step with a learned low-rank
surrogate head that predicts whether the current state satisfies a set of soft
constraints *before* committing a full energy evaluation — allowing the inner
loop to skip expensive energy calls on clearly-violating candidate states.

Sub-requirements:
- REQ-SAMPLE-1935-1: The module SHALL provide a `FARSurrogateHead` dataclass
  that holds a low-rank linear projection (W shape `(latent_dim, n_constraints)`)
  and a bias vector, and exposes a `predict(z)` method returning a
  `(n_constraints,)`-shaped JAX array of soft constraint scores in `[0, 1]`.
- REQ-SAMPLE-1935-2: The module SHALL provide a `ContinuousLatentSampler`
  dataclass that wraps any `EnergyFunction` and a `FARSurrogateHead`, and
  exposes a `sample(key, init, n_steps)` method returning the final latent state.
- REQ-SAMPLE-1935-3: The sampler SHALL skip the full energy gradient call when
  the surrogate head predicts that ALL constraints are satisfied below a
  configurable `skip_threshold` (a value in `(0, 1]`), using a cheaper scaled
  gradient of the surrogate scores instead.
- REQ-SAMPLE-1935-4: Experiment 1935 SHALL benchmark `ContinuousLatentSampler`
  against `LangevinSampler` and `ParallelIsingsampler` on an identical quadratic
  soft-constraint energy, measuring wall-clock steps/second.
- REQ-SAMPLE-1935-5: The benchmark SHALL run entirely on CPU with a fixed JAX
  seed, require no GPU or network access, and write
  `results/experiment_1935_continuous_latent_sampler.json`.

### SCENARIO-SAMPLE-1935: FAR-Inspired Sampler Benchmark

**Given** a quadratic soft-constraint energy (Ising-equivalent) of dimension 32
**When** `ContinuousLatentSampler`, `LangevinSampler`, and a reference discrete
Ising sweep are each run for 200 steps from the same random initial state
**Then** all three produce finite final states with non-NaN energies
**And** the benchmark artifact records steps/second for each sampler
**And** the surrogate skip rate (fraction of steps where full energy was bypassed)
is recorded and is strictly positive (> 0) when `skip_threshold` < 1.0.

### REQ-SAMPLE-2359: Self-Adaptive Lagrangian Ising Sampler

Carnot MUST provide a pure NumPy `SelfAdaptiveIsingSampler` that updates
constraint Lagrange multipliers from observed violations so constrained Ising
sampling does not depend only on manually tuned fixed penalties.

Sub-requirements:
- REQ-SAMPLE-2359-1: The sampler SHALL accept a coupling matrix `J`, bias vector
  `h`, and `constraint_fn(state)` callable, initialize binary state locally, and
  initialize one Lagrange multiplier per reported constraint violation to zero.
- REQ-SAMPLE-2359-2: `sample_step()` SHALL run a Gibbs sweep over all spins using
  the current Ising energy plus the current Lagrange-weighted constraint
  violation energy.
- REQ-SAMPLE-2359-3: `lagrange_update()` SHALL apply
  `lambdas += lr * constraint_violations` and return the updated multipliers.
- REQ-SAMPLE-2359-4: `solve(n_outer=50, n_inner=10)` SHALL alternate inner Gibbs
  sweeps with outer Lagrange updates, record violation history, and report the
  first outer iteration that reaches the configured feasibility threshold.
- REQ-SAMPLE-2359-5: Experiment 2359 SHALL benchmark the adaptive sampler against
  a fixed-penalty baseline with `fixed_penalty=1.0` on exactly two CPU-only
  constrained Ising problems using `random_seed=42`.
- REQ-SAMPLE-2359-6: The benchmark artifact SHALL be written to
  `results/experiment_2359_self_adaptive_ising.json` and include
  `honest_verdict`, `adaptive_ising_validated`, `adaptive_speedup`,
  `final_constraint_violation`, `n_problems`, and `random_seed`.

### SCENARIO-SAMPLE-2359: Adaptive Lagrangian Benchmark Validates On One Problem

**Given** the Exp 2359 CPU benchmark problems
**When** the adaptive sampler and fixed-penalty baseline run with seed 42
**Then** both problems report final constraint violation and iterations to
feasibility
**And** `adaptive_ising_validated` is true only when at least one adaptive speedup
is at least 1.5x.

### REQ-SAMPLE-2044: AIA Continuous Gumbel Sampler Simulator

Carnot MUST provide a `ContinuousGumbelSampler` in
`python/carnot/samplers/continuous_gumbel.py` that uses Gumbel-perturbed
categorical moves to simulate AIA-style sampling on continuous energy
landscapes.

Sub-requirements:
- REQ-SAMPLE-2044-1: The sampler SHALL accept any Python `EnergyFunction` with
  `energy` and `grad_energy` methods.
- REQ-SAMPLE-2044-2: Each step SHALL evaluate a categorical support of
  continuous move values, perturb the move logits with Gumbel noise, and update
  the continuous state using the selected or relaxed move.
- REQ-SAMPLE-2044-3: `sample()` SHALL return the final finite state after
  `n_steps`, and `sample_chain()` SHALL return all intermediate states with
  shape `(n_steps, *init.shape)`.
- REQ-SAMPLE-2044-4: Experiment 2044 SHALL apply the sampler to the Exp 2041
  EqM gradient landscape, compare convergence speed against a
  Metropolis-Hastings random-walk baseline, and write
  `results/experiment_2044_aia_gumbel.json`.
- REQ-SAMPLE-2044-5: The experiment SHALL run deterministically for a fixed
  seed on CPU and require no network, GPU, or external hardware backend.

### SCENARIO-SAMPLE-2044: Exp 2044 Gumbel Simulator Writes Comparison Artifact

**Given** the Exp 2041 EqM quadratic hidden-state landscape
**When** `ContinuousGumbelSampler` and a Metropolis-Hastings baseline are run
from the same initial continuous state
**Then** both chains remain finite and reduce energy
**And** the artifact records per-sampler initial energy, final energy, best
energy, first threshold step, convergence flag, and a Gumbel speedup value
**And** the artifact is saved to `results/experiment_2044_aia_gumbel.json`.

### REQ-SAMPLE-1960: Flow Sampling Density Estimator Prototype

Carnot MUST provide a Flow Sampling density estimator prototype inside `python/carnot/samplers/flow_sampling.py` that allows learning to sample from unnormalized densities via conditional denoising processes.

Sub-requirements:
- REQ-SAMPLE-1960-1: The sampler SHALL implement conditional denoising forward and reverse steps.
- REQ-SAMPLE-1960-2: A benchmark script `scripts/experiment_1960_flow_sampling.py` SHALL define an unnormalized continuous target density using existing EBMs (e.g. Gibbs or Ising).
- REQ-SAMPLE-1960-3: The script SHALL measure the KL divergence against exact block Gibbs baselines.
- REQ-SAMPLE-1960-4: The script SHALL write the results to `results/experiment_1960_flow_sampling_unnormalized.json`.

### SCENARIO-SAMPLE-1960: Flow Sampling Prototype Execution

**Given** the Exp 1960 Flow Sampling script
**When** it is executed
**Then** it successfully runs the conditional denoising forward and reverse steps
**And** it writes a valid JSON artifact with the KL divergence metric.

### REQ-SAMPLE-1962: Neural Indicator Sampling (NI Sampling)

Carnot MUST provide a Neural Indicator (NI) Sampling approach inside `python/carnot/inference/samplers.py` that accelerates discrete diffusion by optimizing token resolution order based on an energy-based indicator step.

Sub-requirements:
- REQ-SAMPLE-1962-1: The module SHALL provide an `NISampler` class that incorporates an energy-based indicator step to dynamically determine which tokens to denoise first.
- REQ-SAMPLE-1962-2: A benchmark script `scripts/experiment_1962_ni_sampling_token_order.py` SHALL measure the wall-clock acceleration of NI Sampling against a random-order discrete diffusion baseline.
- REQ-SAMPLE-1962-3: The script SHALL verify structural semantic retention over benchmark traces by ensuring both samplers converge to valid denoised sequences.
- REQ-SAMPLE-1962-4: The script SHALL write the results artifact to `results/experiment_1962_ni_sampling_token_order.json`.

### SCENARIO-SAMPLE-1962: Neural Indicator Sampling Wall-Clock Acceleration

**Given** the Exp 1962 NI Sampling script
**When** it is executed
**Then** it successfully runs both `NISampler` and the random baseline
**And** it verifies semantic retention
**And** it writes a valid JSON artifact with acceleration metrics, showing NI Sampling completes its task.

### REQ-SAMPLE-1972: Unsupervised RUN-CSP Network Solver

Carnot MUST provide a CPU-only RUN-CSP-style solver for binary constraint
satisfaction problems that maps validator graphs into bipartite
variable-constraint message passing, trains without labels using Carnot's
energy as the loss, and evaluates scale generalization.

Sub-requirements:
- REQ-SAMPLE-1972-1: The solver SHALL expose a graph representation for binary
  variables, pairwise binary constraints, and an energy callback compatible with
  Carnot validator energy functions.
- REQ-SAMPLE-1972-2: The solver SHALL perform iterative message passing from
  constraints to variables and train unsupervised by minimizing the supplied
  energy function over continuous Bernoulli probabilities.
- REQ-SAMPLE-1972-3: Experiment 1972 SHALL train on a 40-variable problem graph,
  evaluate the same learned message-passing parameters on 40-variable and
  1000-variable graphs, and write
  `results/experiment_1972_run_csp_unsupervised.json`.
- REQ-SAMPLE-1972-4: The artifact SHALL include graph metadata, training
  history, per-scale energy and satisfaction metrics, CPU-only execution flags,
  and an honest verdict.

### SCENARIO-SAMPLE-1972: RUN-CSP Unsupervised Generalization Artifact

**Given** the Exp 1972 RUN-CSP experiment runner
**When** it trains on the deterministic 40-variable binary-CSP graph and
evaluates on both 40-variable and 1000-variable graphs
**Then** it writes a terminal JSON artifact with the learned configuration,
training history, per-scale energy metrics, and satisfaction rates
**And** the run uses no labels, GPU, network, or hardware backends.

### REQ-SAMPLE-1985: ConsFormer-style Refinement Loop Sampler

Carnot MUST provide an iterative refinement sampler `ConsFormerRefinementSampler`
in `python/carnot/inference/samplers.py` that utilizes the ConsFormer model
to iteratively refine variable assignments for constraint satisfaction problems.

Sub-requirements:
- REQ-SAMPLE-1985-1: The sampler SHALL initialize the ConsFormer model parameters
  if not provided, or accept pre-trained parameters.
- REQ-SAMPLE-1985-2: The `sample()` method SHALL take an initial sequence and an
  adjacency matrix, then perform the refinement loop to return a final state.
- REQ-SAMPLE-1985-3: An experiment deliverable SHALL be written to
  `results/experiment_1985_consformer_refinement_loop.json` that compares the
  refinement loop's convergence to deterministic baselines like Z3 or PySAT.

### SCENARIO-SAMPLE-1985: ConsFormer Sampler Convergence Comparison

**Given** a simple constraint satisfaction problem encoded as an adjacency matrix
**When** `ConsFormerRefinementSampler` is used alongside a deterministic solver
**Then** the sampler produces a refined state sequence
**And** it writes `results/experiment_1985_consformer_refinement_loop.json`
with convergence metrics and a comparison.

### REQ-SAMPLE-1991: Corrected Curie-Weiss Parity

Carnot MUST provide an experiment script `python/carnot/samplers/thrml_carnot_parity_curie_weiss_1991.py`
that performs a corrected three-way Curie-Weiss parity test comparing Carnot, THRML, and analytic ground truth.

Sub-requirements:
- REQ-SAMPLE-1991-1: The script SHALL execute a Curie-Weiss n=128 model in Carnot and THRML.
- REQ-SAMPLE-1991-2: The script SHALL calculate analytic ground truth for the mean energy.
- REQ-SAMPLE-1991-3: The script SHALL compare Carnot, THRML, and analytic means using calibrated KL gates.
- REQ-SAMPLE-1991-4: The script SHALL explicitly state `hardware_execution_claim=false`.
- REQ-SAMPLE-1991-5: The script SHALL output a JSON artifact to `results/experiment_1991_curie_weiss_parity_correction.json`.

### SCENARIO-SAMPLE-1991: Curie-Weiss Parity Evaluation

**Given** the Exp 1991 Curie-Weiss parity script
**When** it is executed
**Then** it successfully calculates the analytic ground truth
**And** it runs both Carnot and THRML samplers
**And** it writes a valid JSON artifact with the comparison metrics.

### REQ-SAMPLE-2043: AIA Knuth-Yao Categorical Sampler Simulator

Carnot MUST provide a CPU-only software simulator for Knuth-Yao categorical
sampling in `python/carnot/samplers/knuth_yao.py`, representing an Approximate
Inference Accelerator (AIA) sampler boundary without claiming FPGA or hardware
execution.

Sub-requirements:
- REQ-SAMPLE-2043-1: The module SHALL provide a `KnuthYaoSampler` class that
  accepts a finite discrete probability distribution, builds a dyadic
  discrete-distribution-generating (DDG) bit matrix, and samples category
  indices using unbiased random bits.
- REQ-SAMPLE-2043-2: The sampler SHALL expose RNG-bit accounting so experiments
  can report average bits consumed per emitted sample and compare that value to
  a fixed-width categorical sampler baseline.
- REQ-SAMPLE-2043-3: Experiment 2043 SHALL compare `KnuthYaoSampler` against a
  standard RNG categorical sampler on 10,000 samples from the same discrete
  distribution and record empirical frequencies, count deltas, and pass/fail
  parity thresholds.
- REQ-SAMPLE-2043-4: Experiment 2043 SHALL write
  `results/experiment_2043_aia_knuth_yao.json` with required experiment schema
  fields, spec refs, statistical parity metrics, bit-accounting metrics, and
  `hardware_execution_claim=false`.

### SCENARIO-SAMPLE-2043: AIA Knuth-Yao Statistical Parity Artifact

**Given** a dyadic non-uniform categorical distribution
**When** `KnuthYaoSampler` and a standard RNG categorical sampler each emit
10,000 samples with fixed independent seeds
**Then** their empirical category frequencies remain within the configured
parity thresholds
**And** the artifact records average RNG bits consumed per Knuth-Yao sample
**And** it writes `results/experiment_2043_aia_knuth_yao.json` without a
hardware execution claim.

### REQ-SAMPLE-3105: CPU cLUT Logistic Bernoulli Microbench

Carnot MUST provide a CPU-only compressed lookup-table random-variate sampler
for Bernoulli draws whose probabilities are logistic transforms of Ising-style
local fields. The implementation MUST remain a software microbench and MUST
not claim FPGA, GateMate, KV260, TSU, or other hardware speedup without
authenticated hardware execution evidence.

Sub-requirements:
- REQ-SAMPLE-3105-1: `python/carnot/samplers/clut_random_variate.py` SHALL
  provide a compressed fixed-point logistic lookup-table sampler that maps
  local-field logits to Bernoulli samples with deterministic NumPy seeds.
- REQ-SAMPLE-3105-2: The sampler SHALL expose exact-logistic baseline sampling
  and distribution-error helpers so tests can compare cLUT empirical
  frequencies and table approximation error against exact sigmoid
  probabilities.
- REQ-SAMPLE-3105-3: Experiment 3105 SHALL benchmark the cLUT sampler against a
  simple exact-logistic baseline on CPU using deterministic seeds and SHALL
  report speed only as CPU wall-clock timing.
- REQ-SAMPLE-3105-4: Experiment 3105 SHALL write
  `results/experiment_3105_clut_random_variate_sampler_microbench_v1.json`
  with `clut_microbench_ready`, `implementation_path`, `benchmark_commands`,
  `distribution_error`, `speedup_vs_baseline`, `fpga_mapping_notes_path`,
  `hardware_claim_made`, `hardware_commands_run`, `source_artifacts`,
  `inference_substrate`, and a terminal `honest_verdict`.
- REQ-SAMPLE-3105-5: FPGA/GateMate mapping notes SHALL be written as
  architecture context only, separated from the executable benchmark, and
  SHALL state that no hardware commands were run.

### SCENARIO-SAMPLE-3105: CPU cLUT Microbench Writes Claim-Bounded Artifact

**Given** deterministic Ising-style local-field logits and a fixed random seed
**When** the cLUT sampler and exact-logistic baseline each emit Bernoulli
samples on CPU
**Then** the cLUT table approximation and empirical distribution errors are
recorded against exact sigmoid probabilities
**And** CPU timing records a scoped speedup versus the simple baseline
**And** `results/experiment_3105_clut_random_variate_sampler_microbench_v1.json`
is written with `hardware_claim_made=false` and an empty
`hardware_commands_run` list.

### REQ-SAMPLE-3118: Optional CPU cLUT Backend Integration Boundary

Carnot MUST expose the cLUT logistic Bernoulli sampler as an optional CPU
backend adapter without changing the default sampler backend. The adapter MUST
remain a software-only path and MUST NOT claim FPGA, GateMate, KV260, TSU, or
other hardware speedup without authenticated hardware execution evidence.

Sub-requirements:
- REQ-SAMPLE-3118-1: `python/carnot/samplers/clut_backend.py` SHALL provide a
  `ClutCpuBackend` adapter that satisfies the existing sampler backend call
  shape for `sample` and `minimize_energy`.
- REQ-SAMPLE-3118-2: The adapter SHALL use deterministic NumPy seeds so backend
  selection tests can reproduce identical samples for identical seeds.
- REQ-SAMPLE-3118-3: Registering the adapter SHALL preserve the existing
  default backend when no explicit backend name or environment override is
  supplied.
- REQ-SAMPLE-3118-4: Tests SHALL check backend selection, reproducibility,
  zero-coupling Bernoulli distribution error, and default-backend preservation.
- REQ-SAMPLE-3118-5: Experiment 3118 SHALL write
  `results/experiment_3118_clut_sampler_backend_integration_boundary_v2.json`
  with `clut_backend_integration_boundary_v2_ready`, `implementation_paths`,
  `default_backend_preserved`, `distribution_checks_passed`,
  `cpu_timing_summary`, `hardware_claim_made`, `hardware_commands_run`,
  `tests_run`, `source_artifacts`, `inference_substrate`, and a terminal
  `honest_verdict`.

### SCENARIO-SAMPLE-3118: cLUT Backend Is Optional And Claim-Bounded

**Given** deterministic zero-coupling Ising logits and the default sampler
configuration
**When** `clut_cpu` is requested explicitly through the backend factory
**Then** the returned adapter emits reproducible Bernoulli samples using the
cLUT random-variate path
**And** empirical distribution error is recorded against exact sigmoid
probabilities
**And** the default backend remains `cpu` when no backend override is supplied
**And** `results/experiment_3118_clut_sampler_backend_integration_boundary_v2.json`
records CPU-only integration status with `hardware_claim_made=false` and an
empty `hardware_commands_run` list.

### REQ-SAMPLE-2059: TSUSampler thrml API Integration Stub

Carnot MUST provide a `TSUSampler` interface inside `python/carnot/samplers/tsu_sampler.py` that mocks the `thrml` SDK API.

Sub-requirements:
- REQ-SAMPLE-2059-1: The module SHALL provide a `TSUSampler` class.
- REQ-SAMPLE-2059-2: The `TSUSampler` SHALL expose a mock `thrml` SDK API.
- REQ-SAMPLE-2059-3: The `TSUSampler` SHALL be wired into the Carnot sampler registry in `backend.py` as `"thrml_tsu"`.
- REQ-SAMPLE-2059-4: An experiment deliverable SHALL be written to `results/experiment_2059_thrml_integration.json` to prove the interface is functional.

### SCENARIO-SAMPLE-2059: TSUSampler Registration and Execution

**Given** the Carnot sampler registry
**When** the `"thrml_tsu"` backend is requested
**Then** it returns an instance of `TSUSampler`
**And** it writes `results/experiment_2059_thrml_integration.json`.

### REQ-SAMPLE-1682: Falsifiable hypothesis test of joint underestimate in Curie-Weiss sampling

Carnot MUST provide an experiment script `scripts/experiment_1682_thrml_bias.py` to sweep Curie-Weiss parameters.

Sub-requirements:
- REQ-SAMPLE-1682-1: The script SHALL sweep sample size N in [10000, 30000, 100000] at beta=1.2*beta_c on n=128.
- REQ-SAMPLE-1682-2: The script SHALL sweep beta in [1.05*beta_c, 1.2*beta_c, 1.5*beta_c] at N=10000 on n=128.
- REQ-SAMPLE-1682-3: The script SHALL compute bias and fit bias[N] vs 1/sqrt(N) to output a verdict of finite_n, systematic, or mixed.
- REQ-SAMPLE-1682-4: The script SHALL write `results/experiment_1682_thrml_bias.json`.

### SCENARIO-SAMPLE-1682: Bias sweep test produces JSON

**Given** the bias investigation script
**When** it executes
**Then** it outputs the JSON artifact with required sweeps and a fit verdict.

### REQ-SAMPLE-1688: CASAL Sampler Implementation

Carnot MUST provide a `casal_sample(energy_fn, constraint_fn, init_state, steps)` implementation in JAX that uses Split Augmented Langevin Sampling (arXiv:2505.18017) for strictly constrained generative modeling.

Sub-requirements:
- REQ-SAMPLE-1688-1: The `constraint_fn` SHALL return 0 when satisfied and >0 when violated.
- REQ-SAMPLE-1688-2: `casal_sample` SHALL enforce hard constraints such that the output state never violates them (violation rate = 0.0).
- REQ-SAMPLE-1688-3: The experiment script SHALL write `results/experiment_1688_casal_sampler.json` containing the fields: `schema`, `constraint_violation_rate`, `execution_time_ms`, and `acceptance_gate_passed`.

### SCENARIO-SAMPLE-1688: CASAL Sampler respects constraints

**Given** an energy function and a set of hard constraints
**When** the CASAL sampler is executed
**Then** the constraint violation rate is 0.0
**And** the required JSON artifact is produced.

### REQ-SAMPLE-1689: CASAL vs Langevin Scaling Comparison

Carnot MUST compare the CASAL primal-dual sampler against the baseline Langevin
sampler on n=16 and n=32 random continuous EBMs and write
`results/experiment_1689_casal_scaling.json`.

Sub-requirements:
- REQ-SAMPLE-1689-1: Both samplers SHALL run for 1000 steps on the same
  random n-variable ContinuousEBM (random symmetric J, random h, same seed).
- REQ-SAMPLE-1689-2: The artifact SHALL include `schema`, `casal_energy`,
  `langevin_energy`, `speedup_ratio`, and `acceptance_gate_passed` fields.
- REQ-SAMPLE-1689-3: `speedup_ratio` SHALL be the ratio of CASAL's energy
  improvement to Langevin's energy improvement over 1000 steps.
- REQ-SAMPLE-1689-4: `acceptance_gate_passed` SHALL be True when both samplers
  produce finite results on both n=16 and n=32.

### SCENARIO-SAMPLE-1689: CASAL vs Langevin comparison writes terminal artifact

**Given** a random n=16 and n=32 ContinuousEBM
**When** both CASAL and Langevin run for 1000 steps
**Then** both produce finite final states
**And** the required JSON artifact is written with all required fields
**And** `honest_verdict` begins with a terminal prefix (complete:/success:).

### REQ-SAMPLE-2110: CASAL + PiNet Projection Integration

Carnot MUST integrate the Douglas-Rachford PiNet projection layer into the CASAL
sampler primal step to enforce hard linear constraints with geometric guarantees.

Sub-requirements:
- REQ-SAMPLE-2110-1: `casal_sample` SHALL accept an optional `pinet_layer` parameter
  of type `DouglasRachfordPiNetLayer` (or None for the existing gradient-descent path).
- REQ-SAMPLE-2110-2: When `pinet_layer` is provided, the primal projection step SHALL
  call `pinet_layer.project_vector(proposed_state)` instead of the gradient-descent loop.
- REQ-SAMPLE-2110-3: The acceptance gate (constraint_fn <= 1e-5) SHALL still apply
  after PiNet projection to guarantee zero violation in the returned state.
- REQ-SAMPLE-2110-4: Experiment 2110 SHALL measure the constraint violation rate over
  100 independent trials and write `results/experiment_2110_casal_pinet.json` with
  `violation_rate` and `honest_verdict` fields using a terminal prefix.
- REQ-SAMPLE-2110-5: The integration SHALL be backward-compatible: existing calls with
  no `pinet_layer` argument must produce identical results.

### SCENARIO-SAMPLE-2110: CASAL + PiNet Achieves Zero Violation Rate

**Given** a CASAL sampler configured with a DouglasRachfordPiNetLayer encoding
the same constraint as the constraint_fn
**When** `casal_sample` runs for 200 steps from a feasible initial state
**Then** the final state satisfies the constraint (violation <= 1e-5)
**And** the violation rate across 100 independent trials is 0.0
**And** the artifact is written to `results/experiment_2110_casal_pinet.json`.

### REQ-SAMPLE-2245: CASAL Primal-Dual Equality Sampler

Carnot MUST provide a `CASALSampler` class in `python/carnot/samplers/casal.py`
that runs primal-dual Split Augmented Langevin sampling for hard equality
constraints during inference.

Sub-requirements:
- REQ-SAMPLE-2245-1: `CASALSampler(constraints, step_size, dual_step_size, n_steps)`
  SHALL accept equality constraints as residual callables, where zero residual
  means satisfied.
- REQ-SAMPLE-2245-2: `sample(x_init, energy_fn)` SHALL run Langevin primal
  updates, project each primal state back onto the equality manifold, and return
  a finite final sample.
- REQ-SAMPLE-2245-3: The sampler SHALL update Lagrange multipliers from
  post-projection residuals and expose diagnostics for mean violation and dual
  update convergence.
- REQ-SAMPLE-2245-4: The Exp 2245 unit test SHALL verify mean equality
  violation below `1e-4` on a deterministic 2D quadratic-energy problem with one
  equality constraint.
- REQ-SAMPLE-2245-5: Exp 2245 SHALL write
  `results/experiment_2245_casal_impl.json` with `casal_impl_ready=true` only
  when the unit test demonstrates mean violation below `1e-4`.

### SCENARIO-SAMPLE-2245: CASAL Enforces A 2D Equality Constraint

**Given** a 2D quadratic energy and the equality constraint `x0 + x1 = 1`
**When** `CASALSampler.sample()` runs from an infeasible initial state
**Then** the returned sample satisfies the equality constraint with mean
violation below `1e-4`
**And** the dual update convergence diagnostic is true
**And** `results/experiment_2245_casal_impl.json` records the implementation
gate outcome.

### REQ-SAMPLE-2246: CASAL vs AdamFLIP Constraint Violation Benchmark

Carnot MUST compare CASAL inference-time hard equality sampling against an
AdamFLIP-trained soft-penalty MCMC baseline on the same 3D continuous EBM
energy landscape.

Sub-requirements:
- REQ-SAMPLE-2246-1: The benchmark SHALL import
  `python/carnot/samplers/casal.py` and `python/carnot/training/adamflip.py`
  before sampling, and SHALL write `blocked_casal_missing` or
  `blocked_adamflip_missing` if either import precondition fails.
- REQ-SAMPLE-2246-2: The benchmark SHALL define a 3D `ContinuousEBM` with two
  hard equality constraints and SHALL draw at least 100 samples from each
  regime using the same random seed and initial states.
- REQ-SAMPLE-2246-3: Regime A SHALL run MCMC with AdamFLIP-trained EBM
  parameters and a post-training soft constraint penalty.
- REQ-SAMPLE-2246-4: Regime B SHALL run `CASALSampler` primal-dual sampling on
  the same trained energy parameters and equality constraints.
- REQ-SAMPLE-2246-5: The artifact SHALL report
  `casal_violation_mean`, `adamflip_violation_mean`, maximum constraint
  violation for each regime, energy mean for each regime, `n_samples`, and
  `random_seed`.
- REQ-SAMPLE-2246-6: `casal_validated` SHALL be true exactly when
  `casal_violation_mean <= adamflip_violation_mean / 2`.
- REQ-SAMPLE-2246-7: Exp 2246 SHALL write
  `results/experiment_2246_casal_vs_adamflip.json` with a terminal
  `honest_verdict` beginning with `complete:` only when the validation gate
  passes.

### SCENARIO-SAMPLE-2246: CASAL Beats AdamFLIP Soft-Penalty MCMC On Constraint Violation

**Given** a deterministic 3D continuous EBM with two equality constraints
**And** AdamFLIP has trained the EBM parameters against those constraints
**When** 100 shared-seed initial states are sampled by soft-penalty MCMC and
CASAL
**Then** the artifact records both mean and maximum constraint violations
**And** `casal_validated` equals the half-violation acceptance gate
**And** `results/experiment_2246_casal_vs_adamflip.json` contains the required
field-principle metadata for downstream capstone gating.

### REQ-SAMPLE-2355: Projected-Langevin Constraint Sampler

Carnot MUST provide a CPU-only Projected-Langevin sampler in
`python/carnot/samplers/projected_langevin.py` that takes a Langevin step and
then projects the state back onto explicit hard constraints.

Sub-requirements:
- REQ-SAMPLE-2355-1: `ProjectedLangevinSampler.sample()` SHALL accept
  `sample(energy_fn, constraints, init, n_steps=100, step_size=0.01,
  temperature=1.0)` and return the final NumPy state.
- REQ-SAMPLE-2355-2: Box constraints of the form `x[i] in [lo, hi]` SHALL be
  enforced by clamping after every Langevin update.
- REQ-SAMPLE-2355-3: Linear equality constraints SHALL be enforced by gradient
  projection so the 3-problem benchmark can cover linear and sum constraints.
- REQ-SAMPLE-2355-4: Exp 2355 SHALL benchmark Projected-Langevin against CASAL
  when CASAL is importable on exactly three CPU-only problems for 100 steps and
  write `results/experiment_2355_projected_langevin.json`.
- REQ-SAMPLE-2355-5: The artifact SHALL include `honest_verdict`,
  `projected_langevin_competitive`, `constraint_satisfaction_rate`,
  `langevin_vs_casal_delta`, `n_problems`, and `random_seed`.

### SCENARIO-SAMPLE-2355: Projected-Langevin Benchmark Writes Terminal Artifact

**Given** the Projected-Langevin sampler, three deterministic constrained
energy problems, and random seed 42
**When** Projected-Langevin and CASAL are each run for 100 steps where CASAL is
available
**Then** the artifact records final constraint satisfaction and energy for each
sampler
**And** `projected_langevin_competitive` is true when Projected-Langevin
matches or exceeds CASAL satisfaction
**And** `results/experiment_2355_projected_langevin.json` contains the required
field-principle metadata.

### REQ-SAMPLE-2250: CASAL SamplerBackend Protocol Adapter

Carnot MUST expose the CASAL primal-dual sampler through the sampler backend
boundary without forcing `CASALSampler` to structurally match the Ising-only
backend call shape directly.

Sub-requirements:
- REQ-SAMPLE-2250-1: `SamplerBackend` SHALL include optional
  `set_constraints(constraints)` and `dual_update_step(dual_lr)` hooks with
  no-op behavior for non-CASAL backends.
- REQ-SAMPLE-2250-2: `CASALBackend` SHALL wrap `CASALSampler` and satisfy the
  runtime-checkable `SamplerBackend` protocol.
- REQ-SAMPLE-2250-3: Exp 2250 SHALL write
  `results/experiment_2250_thrml_casal.json` with terminal
  `honest_verdict`, `protocol_methods_added`, `casal_backend_wraps_protocol`,
  and `preconditions_checked` fields.

### SCENARIO-SAMPLE-2250: CASAL Adapter Satisfies Backend Protocol

**Given** the CASAL module imports successfully
**When** a `CASALBackend` instance is checked against `SamplerBackend`
**Then** the runtime protocol conformance check passes
**And** non-CASAL backends retain no-op primal-dual hooks
**And** `results/experiment_2250_thrml_casal.json` records the protocol
evolution audit fields.

### REQ-SAMPLE-1807: Lyapunov Control Barrier Functions in Langevin Sampler

Carnot MUST provide a way to integrate Lyapunov Control Barrier Functions (CBFs) as
additive penalty terms in the Langevin sampler to enforce hard safety constraints.

Sub-requirements:
- REQ-SAMPLE-1807-1: The Langevin sampler SHALL optionally accept a strict logical constraint and a corresponding Control Barrier Function.
- REQ-SAMPLE-1807-2: The Control Barrier Function SHALL artificially raise the energy near the constraint violation.
- REQ-SAMPLE-1807-3: The experiment script SHALL write `results/experiment_1807_lyapunov_cbf.json` containing the fields: `schema`, `violation_rate`, and `acceptance_gate_passed`.

### SCENARIO-SAMPLE-1807: CBF reduces constraint violations

**Given** an energy function and a Control Barrier Function defining a hard constraint
**When** the modified Langevin sampler is executed
**Then** the constraint violation rate is logged
**And** the required JSON artifact is produced to `results/experiment_1807_lyapunov_cbf.json`.

### REQ-SAMPLE-2605: Mpemba-inspired Initialization

Carnot MUST provide Mpemba-inspired initialization for Gibbs and Langevin samplers in `python/carnot/samplers/thermo_init.py` to accelerate convergence to the target Boltzmann distribution (arXiv:2605.13883).

Sub-requirements:
- REQ-SAMPLE-2605-1: The module SHALL provide a function `mpemba_init(key, energy_fn, shape, hot_beta, target_beta)` that generates a non-equilibrium initial state.
- REQ-SAMPLE-2605-2: The initial state SHALL be drawn from a modified distribution that accelerates convergence compared to a standard infinite-temperature initialization.
- REQ-SAMPLE-2605-3: Tests SHALL verify that `mpemba_init` provides a convergence speedup on a toy 2D energy landscape.

### SCENARIO-SAMPLE-2605: Mpemba Initialization Accelerates Convergence

**Given** a toy 2D energy landscape and a target beta
**When** a sampler is initialized using `mpemba_init` and compared to a standard random initialization
**Then** the Mpemba-initialized sampler reaches a target energy threshold in fewer steps.

### REQ-SAMPLE-2605-12782: Langevin Clock Rescaling

Carnot MUST provide a modified Langevin sampler in `python/carnot/samplers/langevin_clock.py` that speeds up the Langevin clock by scaling deterministic forces and adding specific noise structures (arXiv:2605.12782).

Sub-requirements:
- REQ-SAMPLE-2605-12782-1: The module SHALL provide a `langevin_clock_step(state, grad, step_size, force_scale, noise_scale, key)` function.
- REQ-SAMPLE-2605-12782-2: The update step SHALL compute the next state applying `force_scale` to the deterministic gradient and `noise_scale` to the random noise.
- REQ-SAMPLE-2605-12782-3: Tests SHALL verify that the function returns a valid next state with the expected shapes.

### SCENARIO-SAMPLE-2605-12782: Langevin Clock Rescaling Step Execution

**Given** a state, gradient, step size, and scaling factors
**When** `langevin_clock_step` is executed
**Then** it successfully computes a modified Langevin step
**And** returns a state matching the input shape.

### REQ-SAMPLE-051: Vectorized Langevin dynamics on BoltzmannModel

Carnot MUST provide a vectorized Langevin step for the `BoltzmannModel` to support GPU-accelerated parallel sampling.

Sub-requirements:
- REQ-SAMPLE-051-1: The `BoltzmannModel` SHALL expose a `langevin_step_vectorized(x_batch, step_size, key)` method.
- REQ-SAMPLE-051-2: The method SHALL compute the Langevin step simultaneously across a batch of states via `jax.vmap`.
- REQ-SAMPLE-051-3: Experiment 2010 SHALL compare CPU vs GPU mocked performance of the gradient step and output `results/experiment_2010_langevin.json`.

### SCENARIO-SAMPLE-051: Vectorized Langevin Step Batch Execution

**Given** a `BoltzmannModel` and a batch of states
**When** `langevin_step_vectorized` is executed
**Then** it successfully computes the next states for the entire batch simultaneously.

### REQ-SAMPLE-2011: ROCm Langevin Sampler Fidelity

Carnot MUST provide an experiment script `scripts/experiment_2011_fidelity.py` to calculate the KL divergence (or divergence metric) between the CPU and mock GPU Langevin sampler exact counts.

Sub-requirements:
- REQ-SAMPLE-2011-1: The script SHALL evaluate divergence between CPU and mock GPU Langevin sampler.
- REQ-SAMPLE-2011-2: The KL divergence and fidelity validation SHALL be saved to `results/experiment_2011_fidelity.json`.

### SCENARIO-SAMPLE-2011: ROCm Langevin Fidelity Validation

**Given** the ROCm Langevin sampler fidelity script is executed
**When** it calculates divergence between CPU and mock GPU samplers
**Then** it writes the result to `results/experiment_2011_fidelity.json`

### REQ-SAMPLE-2067: DTM Simulator

Carnot MUST provide a Denoising Thermodynamic Model (DTM) CPU simulator in `python/carnot/samplers/dtm_sampler.py`.
The DTM simulator integrates Langevin thermal noise with Carnot's energy function (arXiv:2510.23972).

Sub-requirements:
- REQ-SAMPLE-2067-1: The module SHALL provide a `DTMSampler` class mapping state updates to thermal noise profiles.
- REQ-SAMPLE-2067-2: The sampler SHALL use Langevin thermal noise in its updates.
- REQ-SAMPLE-2067-3: The experiment artifact SHALL be written to `results/experiment_2067_dtm_simulation.json` with `dtm_sim_ready=true`.

### SCENARIO-SAMPLE-2067: DTM Simulator writes terminal artifact

**Given** the DTM Simulator and an energy function
**When** the DTM sampler runs
**Then** it successfully outputs a valid sample and updates
**And** it writes `results/experiment_2067_dtm_simulation.json` with `dtm_sim_ready=true`.

### REQ-SAMPLE-2081: SGLRW Dual RTX GSM8K Benchmark

Carnot MUST provide an experiment script `scripts/experiment_2081_dual_rtx_gsm8k.py` to benchmark the SGLRW sampler on a dual-GPU setup using a GSM8K subset.

Sub-requirements:
- REQ-SAMPLE-2081-1: The script SHALL use model 'unsloth/Qwen3.6-35B-A3B-GGUF'.
- REQ-SAMPLE-2081-2: The script SHALL probe for ROCm/CUDA availability on multiple devices and distribute the EBM sampler.
- REQ-SAMPLE-2081-3: The script SHALL record hardware latency and generative accuracy on 100 GSM8K problems.
- REQ-SAMPLE-2081-4: The script SHALL write an initial bootstrap artifact to `results/experiment_2081_dual_rtx_benchmark.json` before running the benchmark, and update it with the final results.

### SCENARIO-SAMPLE-2081: SGLRW Dual RTX Benchmark Writes Artifact

**Given** the Exp 2081 Dual RTX benchmark script
**When** it is executed
**Then** it successfully checks for multi-GPU availability
**And** it runs the SGLRW sampler natively against a 100-sample GSM8K partition
**And** it writes `results/experiment_2081_dual_rtx_benchmark.json` containing latency and accuracy metrics.

### REQ-SAMPLE-2109: ALPS Sampler Implementation

Carnot MUST provide an Annealed Langevin Posterior Sampling (ALPS) sampler in `python/carnot/samplers/alps.py`.
ALPS provides faster convergence for EBMs by annealing static posterior distributions.

Sub-requirements:
- REQ-SAMPLE-2109-1: The module SHALL provide an `AlpsSampler` class that composes directly with `ContinuousEBM`.
- REQ-SAMPLE-2109-2: The sampler SHALL implement an ALPS schedule for faster convergence than standard Langevin.
- REQ-SAMPLE-2109-3: Tests SHALL verify ALPS convergence is significantly faster than standard Langevin on a multi-modal toy landscape.
- REQ-SAMPLE-2109-4: An experiment deliverable SHALL be written to `results/experiment_2109_alps_module.json`.

### SCENARIO-SAMPLE-2109: ALPS Sampler Faster Convergence

**Given** a multi-modal toy landscape represented as a `ContinuousEBM`
**When** both ALPS and standard Langevin samplers are run
**Then** ALPS reaches the target minimum in significantly fewer steps
**And** the experiment artifact is written to `results/experiment_2109_alps_module.json`.

### REQ-SAMPLE-5116: HUBO/p-Spin 2D Parallel Tempering CPU Reference

Carnot MUST provide a deterministic CPU reference sampler for direct
HUBO/p-spin constraint energies that runs replicas over both inverse
temperature and constraint-penalty dimensions without making any hardware
speedup claim.

Sub-requirements:
- REQ-SAMPLE-5116-1: The implementation SHALL expose direct HUBO/p-spin energy
  evaluation over binary states and exact enumeration helpers for tiny
  instances.
- REQ-SAMPLE-5116-2: The CPU sampler SHALL support deterministic-seeded
  unguided Gibbs, one-dimensional beta parallel tempering, and two-dimensional
  beta/penalty parallel tempering runs over declared grids.
- REQ-SAMPLE-5116-3: Adjacent replica swaps SHALL use the Metropolis ratio for
  the fixed parameter slots being exchanged, and SHALL record beta-axis and
  penalty-axis swap attempts, acceptances, and acceptance rates.
- REQ-SAMPLE-5116-4: Exp 5116 SHALL validate tiny instances by exact
  enumeration of optima and energy distributions before reporting sampler
  utility metrics.
- REQ-SAMPLE-5116-5: Exp 5116 SHALL compare unguided Gibbs, one-dimensional
  beta parallel tempering, and two-dimensional beta/penalty parallel
  tempering on at least two synthetic constraint families.
- REQ-SAMPLE-5116-6: The terminal artifact
  `results/experiment_5116_hubo_2dpt_sampling_reference_v469.json` SHALL
  include `experiment_id`, `milestone`, `honest_verdict`,
  `inference_substrate`, `duration_s`, `preconditions_checked`,
  `exact_enumeration_checked`, `instance_families`, `beta_grid`,
  `penalty_grid`, `swap_acceptance_rates`,
  `best_energy_delta_vs_baselines`, `optimum_hit_rate`,
  `hubo_2dpt_reference_ready`, `hardware_speedup_claimed`,
  `seeds_or_checksums`, `flagged_adversarial`, and `tests_run`.
- REQ-SAMPLE-5116-7: `hubo_2dpt_reference_ready` SHALL be true only when exact
  enumeration checks pass and the two-dimensional beta/penalty PT arm improves
  on or honestly ties both CPU baselines without claiming hardware execution.

### SCENARIO-SAMPLE-5116: Exact-Checked HUBO 2D PT Artifact

**Given** tiny direct HUBO/p-spin constraint instances from at least two
synthetic families
**When** Exp 5116 runs on CPU with fixed seeds
**Then** exact enumeration verifies optima and energy distributions
**And** unguided Gibbs, one-dimensional beta PT, and two-dimensional beta/penalty
PT are compared on the same instances
**And** beta-axis and penalty-axis swap bookkeeping is recorded
**And** `results/experiment_5116_hubo_2dpt_sampling_reference_v469.json` is
written with `hardware_speedup_claimed=false` and a terminal honest verdict.

### REQ-SAMPLE-5129: Adaptive HUBO/p-Spin 2D PT Temperature Ladder

Carnot MUST extend the exact-checked CPU HUBO/p-spin 2D parallel-tempering
reference with adaptive inverse-temperature ladder optimization while
preserving exact-label authority and detailed-balance diagnostics.

Sub-requirements:
- REQ-SAMPLE-5129-1: The adaptive ladder update SHALL keep inverse
  temperatures positive and monotonically ordered, with fixed endpoints unless
  explicitly configured otherwise.
- REQ-SAMPLE-5129-2: The adaptive sampler SHALL record beta-axis and
  penalty-axis swap acceptance rates, per-sweep residual energy against exact
  enumeration, round-trip proxies, and local reversibility or detailed-balance
  sanity checks.
- REQ-SAMPLE-5129-3: Exp 5129 SHALL load or reproduce the Exp 5116 fixed-grid
  baseline on the same tiny exact-enumerable instance families.
- REQ-SAMPLE-5129-4: Exp 5129 SHALL run exact enumeration for every tiny
  instance and preserve optimum labels before reporting adaptive readiness.
- REQ-SAMPLE-5129-5: The terminal artifact
  `results/experiment_5129_hubo_adaptive_2dpt_v470.json` SHALL include
  `experiment_id`, `milestone`, `honest_verdict`, `inference_substrate`,
  `duration_s`, `exp5116_baseline_loaded`, `instance_families`,
  `exact_enumeration_checked`, `adaptive_temperature_config`,
  `swap_acceptance_rates`, `residual_energy_by_sweep`, `optimum_hit_rate`,
  `detailed_balance_sanity`, `best_energy_delta_vs_baselines`,
  `adaptive_2dpt_ready`, `hardware_speedup_claimed`, `flagged_adversarial`,
  `conductor_modified`, and `tests_run`.
- REQ-SAMPLE-5129-6: `adaptive_2dpt_ready` SHALL be true only when exact labels
  are preserved, no hardware speedup is claimed, and at least one mixing or
  energy metric improves over the fixed-grid baseline without harmful energy or
  optimum-hit regressions.

### SCENARIO-SAMPLE-5129: Adaptive HUBO 2D PT Artifact

**Given** the Exp 5116 tiny exact-checkable HUBO/p-spin instance families
**When** Exp 5129 runs adaptive inverse-temperature 2D PT on CPU
**Then** every instance is exact-enumerated before adaptive metrics are gated
**And** the adaptive beta ladder remains positive and monotonically ordered
**And** residual-energy, swap-acceptance, round-trip, and detailed-balance
telemetry are recorded
**And** `results/experiment_5129_hubo_adaptive_2dpt_v470.json` is written with
`hardware_speedup_claimed=false`, `conductor_modified=false`, and a terminal
honest verdict.

### REQ-SAMPLE-5141: Partitioned HUBO/Ising 2D PT Residual Telemetry

Carnot MUST extend the exact-checked CPU HUBO/p-spin 2D PT reference toward
hardware-compatible telemetry by measuring partitioned update schedules,
boundary-refresh ratios, residual-energy decay exponents, and board-ready
workload descriptors without executing hardware or claiming speedup.

Sub-requirements:
- REQ-SAMPLE-5141-1: Exp 5141 SHALL load the Exp 5129 adaptive HUBO/2D-PT
  baseline when available and SHALL record whether the upstream baseline was
  loaded before comparing partitioned CPU telemetry.
- REQ-SAMPLE-5141-2: Exp 5141 SHALL generate tiny exact-enumerable HUBO and
  Ising-compatible instances, run exact enumeration for every reported
  instance, and preserve exact optimum labels before reporting sampler quality.
- REQ-SAMPLE-5141-3: Exp 5141 SHALL compare unguided Gibbs, a monolithic CPU
  2D-PT reference, and partitioned 2D-PT variants across declared partition
  layouts, boundary-refresh ratios, inverse-temperature ladders, and
  residual-energy measurement windows.
- REQ-SAMPLE-5141-4: Exp 5141 SHALL verify detailed balance for variants whose
  physical-state kernel can satisfy it, and SHALL record an exact blocker for
  stale-boundary variants whose augmented boundary-cache state prevents a
  physical-state-only detailed-balance proof.
- REQ-SAMPLE-5141-5: Exp 5141 SHALL report optimum-hit rate, residual-energy
  exponent, boundary mismatch rate, and effective sample quality relative to
  the monolithic CPU reference and unguided baselines.
- REQ-SAMPLE-5141-6: The terminal artifact
  `results/experiment_5141_hubo_partition_residual_exponent_v471.json` SHALL
  include `experiment_id`, `milestone`, `honest_verdict`,
  `inference_substrate`, `duration_s`, `exp5129_baseline_loaded`,
  `partition_configs`, `boundary_refresh_ratios`,
  `residual_energy_exponents`, `exact_enumeration_checked`,
  `detailed_balance_evidence`, `monolithic_reference`,
  `board_ready_workload_descriptors`, `partition_telemetry_ready`,
  `hardware_speedup_claimed`, `conductor_modified`, and `tests_run`.
- REQ-SAMPLE-5141-7: `partition_telemetry_ready` SHALL be true only when exact
  enumeration checks pass, all unblocked detailed-balance checks pass,
  telemetry metrics are finite across the declared sweep, board descriptors are
  hash-addressed, and no hardware speedup or conductor mutation is claimed.

### SCENARIO-SAMPLE-5141: Partitioned Residual Telemetry Artifact

**Given** tiny exact-checkable HUBO/p-spin and Ising-compatible instances
**When** Exp 5141 runs CPU partitioned 2D PT telemetry sweeps
**Then** exact enumeration verifies optima and energy distributions before
sample-quality metrics are reported
**And** monolithic and partitioned update variants are compared over declared
partition layouts, boundary-refresh ratios, beta ladders, and residual windows
**And** detailed-balance evidence or an exact blocker is recorded for every
variant
**And** board-ready KV260, GateMate, and PolarFire workload descriptors include
stable hashes while `hardware_speedup_claimed=false`
**And** `results/experiment_5141_hubo_partition_residual_exponent_v471.json` is
written with `conductor_modified=false` and a terminal honest verdict.

### REQ-SAMPLE-5130: TACO Sampler Held-Out CSP Trace Suite

Carnot MUST scale the TACO-style adaptive CSP solver-help path to held-out
exact-checkable CSP families while preserving complete exact-solver or exact
enumerator correctness authority for every reported label.

Sub-requirements:
- REQ-SAMPLE-5130-1: Exp 5130 SHALL hard-block when
  `results/experiment_5129_hubo_adaptive_2dpt_v470.json` does not report
  `adaptive_2dpt_ready=true`.
- REQ-SAMPLE-5130-2: Exp 5130 SHALL build a deterministic held-out CSP suite
  disjoint from any Exp 5117 tuning instance and SHALL record a stable SHA-256
  hash for every held-out instance.
- REQ-SAMPLE-5130-3: Exp 5130 SHALL compare the baseline exact solver,
  unguarded adaptive heuristic, harm-gated adaptive heuristic, and an
  adaptive-sampler feature variant on the same held-out instances.
- REQ-SAMPLE-5130-4: A heuristic answer SHALL NOT be counted correct unless
  the complete exact solver or exact enumerator agrees with the reported label.
- REQ-SAMPLE-5130-5: The terminal artifact with
  `inference_substrate="cpu_exact_solver_with_adaptive_heuristic"`
  `results/experiment_5130_taco_sampler_heldout_scale_v470.json` SHALL include
  `experiment_id` with value `exp5130-taco-sampler-heldout-scale-v470`,
  `milestone`, `honest_verdict`, `inference_substrate`, `duration_s`,
  `exp5117_baseline_loaded`,
  `exp5129_sampler_features_loaded`, `exact_solver_backend`,
  `heldout_instance_hashes`, `instance_count`,
  `average_effort_reduction_ratio_guarded`,
  `harmful_instance_count_guarded`, `harmful_instance_count_unguarded`,
  `wrong_label_count`, `timeout_rate`, `per_family_results`,
  `heldout_csp_trace_suite_ready`, `flagged_adversarial`,
  `conductor_modified`, and `tests_run`.
- REQ-SAMPLE-5130-6: `heldout_csp_trace_suite_ready` SHALL be true only when
  Exp 5117 and Exp 5129 dependencies load, all held-out labels are exact
  checked, no heuristic-only answer is counted correct, and per-family traces
  are sufficient for the FR-11 case-policy task.

### SCENARIO-SAMPLE-5130: Held-Out CSP Trace Artifact

**Given** Exp 5117 exact-label TACO harm-gate evidence and Exp 5129 adaptive
sampler telemetry with `adaptive_2dpt_ready=true`
**When** Exp 5130 runs on CPU against disjoint held-out CSP families
**Then** every baseline, unguarded, guarded, and sampler-feature label agrees
with exact authority
**And** effort reduction, harmful-instance counts, wrong-label count, timeout
rate, held-out hashes, and per-family results are recorded
**And** `results/experiment_5130_taco_sampler_heldout_scale_v470.json` is
written with `conductor_modified=false`, `flagged_adversarial=false`, and a
terminal honest verdict.

### REQ-SAMPLE-5142: TACO Harm Root-Cause Scale Gate

Carnot MUST scale the exact-checked TACO/CSP trace suite before self-learning
uses sampler-feature traces, and MUST explain and gate harmful guarded regimes
without changing any exact labels or treating sampler features as a correctness
authority.

Sub-requirements:
- REQ-SAMPLE-5142-1: Exp 5142 SHALL load Exp 5130 evidence and reproduce its
  baseline, guarded, and sampler-feature effort measurements before reporting a
  scaled V471 trace suite.
- REQ-SAMPLE-5142-2: Exp 5142 SHALL evaluate at least 80 held-out
  exact-checkable CSP instances across multiple task families and difficulty
  bands, measuring baseline effort, original guarded effort, sampler-feature
  effort, repaired guarded effort, exact labels, and label disagreements.
- REQ-SAMPLE-5142-3: Exp 5142 SHALL cluster harmful guarded instances by
  auditable structural and sampler features including degree, constraint
  density, near-tie score, sampler entropy, and branching factor.
- REQ-SAMPLE-5142-4: Exp 5142 SHALL implement a conservative repaired harm gate
  or abstention rule that rejects sampler-feature guidance in identified
  harmful regimes while preserving exact-solver label authority.
- REQ-SAMPLE-5142-5: The terminal artifact with
  `inference_substrate="exact_checked_taco_csp_trace_suite"`
  `results/experiment_5142_taco_harm_rootcause_scale_v471.json` SHALL include
  `experiment_id` with value `exp5142-taco-harm-rootcause-scale-v471`,
  `milestone`, `honest_verdict`, `inference_substrate`, `duration_s`,
  `exp5130_baseline_loaded`, `instance_count`, `task_families`,
  `harmful_instance_root_causes`,
  `average_effort_reduction_ratio_guarded`,
  `harmful_instance_count_guarded`, `wrong_label_count`,
  `repaired_harm_gate`, `ablation_results`, `trace_suite_v2_ready`,
  `conductor_modified`, and `tests_run`.
- REQ-SAMPLE-5142-6: `trace_suite_v2_ready` SHALL be true only when Exp 5130
  reproduction succeeds, `wrong_label_count` is zero, the repaired guard
  reduces or eliminates harmful guarded cases, useful effort reduction remains,
  and `conductor_modified=false`.

### SCENARIO-SAMPLE-5142: Harm-Rooted Trace Suite V2 Artifact

**Given** Exp 5130 exact-checked held-out TACO/CSP traces
**When** Exp 5142 scales the suite and evaluates conservative repaired gating
**Then** every reported label is backed by exact solver and enumerator authority
**And** harmful guarded regimes are clustered by structural and sampler features
**And** sampler-feature guidance is abstained in the identified harmful regimes
**And** `results/experiment_5142_taco_harm_rootcause_scale_v471.json` is
written with `conductor_modified=false` and a terminal honest verdict.

### REQ-SAMPLE-6152: Typed Stochastic Constraint IR For Exp6145

Carnot MUST provide Exp6152 at
`python/carnot/experiment_6152_typed_stochastic_constraint_ir.py` and write
`results/experiment_6152_typed_stochastic_constraint_ir.json` without modifying
`scripts/research_conductor.py`. The experiment SHALL represent one bounded
Exp6145 finite-domain constraint workflow as a dependency-light typed
Parametrized Stochastic Circuit / Directed Factor Graph style IR with explicit
binary and categorical wires, local deterministic and stochastic kernels, exact
finite semantics, stable serialization, and an optional adapter to the real
pinned Torx package when it is importable.

Sub-requirements:
- REQ-SAMPLE-6152-1: The IR SHALL declare a schema version, binary and
  categorical wire types, category order, batch shape contracts, parameter
  schemas, factor/channel edges, and deterministic/stochastic local-kernel
  kinds.
- REQ-SAMPLE-6152-2: Graph validation SHALL reject unsupported cycles,
  dangling wires, type mismatches, invalid or unnormalized probability mass,
  duplicate wire or kernel identifiers, invalid category indexes, and ambiguous
  seeds.
- REQ-SAMPLE-6152-3: The Carnot executor SHALL compute exact support,
  conditional probabilities, joint probabilities, normalization, marginals,
  deterministic factors, impossible states, batch execution, and seed replay by
  exhaustive finite enumeration rather than sampling.
- REQ-SAMPLE-6152-4: The compiler SHALL map one bounded Exp6145 workflow into
  the IR and compare it against an independent exact reference that would catch
  wire-order and category-index bugs.
- REQ-SAMPLE-6152-5: Serialization SHALL round trip through canonical JSON
  without changing semantics, state order, checksums, or executable receipts.
- REQ-SAMPLE-6152-6: The Torx adapter SHALL name the real package version,
  import namespace, package metadata, pinned repository commit, and exercised
  API when Torx is available. If Torx is absent or the API cannot be exercised,
  compatibility evidence SHALL be blocked honestly and
  `typed_stochastic_ir_ready_score` SHALL remain `0.0`.
- REQ-SAMPLE-6152-7: The terminal artifact SHALL include `status`,
  `preconditions_checked`, `structured_gate_receipt`,
  `source_workflow_validator_sampler_and_exclusion_hashes`,
  `torx_package_version_commit_import_and_api_receipts`,
  `ir_schema_version_types_kernels_and_graph_contract`,
  `compiler_executor_adapter_and_test_paths`,
  `exact_enumeration_case_counts`,
  `support_conditional_joint_normalization_and_marginal_deltas`,
  `deterministic_impossible_batch_seed_and_serialization_controls`,
  `wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls`,
  `torx_compatibility_scope`, `deterministic_rebuild_checksum`,
  `typed_stochastic_ir_ready_score`, `protected_files_unchanged`,
  `duration_s`, `inference_substrate`, `verifier_is_oracle`,
  `missing_verifier_gaps`, `field_provenance`, `test_commands`,
  `test_exit_codes`, `reproducibility_checksum`, and `honest_verdict`.
  `inference_substrate` SHALL be `jax_cpu_exact_stochastic_program` only when
  the exact Carnot semantics and real pinned Torx compatibility are exercised;
  otherwise it SHALL be `carnot_only_blocked_torx_compatibility`.

Required field principles:

- `status`: A terminal state distinguishes ready, Carnot-only, null, and blocked compiler boundaries.
- `preconditions_checked`: Hashes JAX CPU mode, Exp6145 workflow evidence, validators, sampler APIs, Torx metadata, exclusions, outputs, and protected files before construction.
- `structured_gate_receipt`: Exp6145 readiness and sidecar replay must pass before the stochastic workflow is compiled.
- `source_workflow_validator_sampler_and_exclusion_hashes`: Content hashes bind the source workflow, exact validators, sampler APIs, Torx package metadata or commit, exclusions, output paths, and protected files.
- `torx_package_version_commit_import_and_api_receipts`: Real importable Torx package/version/commit/API evidence is recorded; resemblance to public docs is not compatibility.
- `ir_schema_version_types_kernels_and_graph_contract`: The versioned IR contract names wire types, kernel schemas, edge rules, state shapes, and normalization gates.
- `compiler_executor_adapter_and_test_paths`: Paths identify the compiler, exact executor, optional adapter, tests, and artifact boundary.
- `exact_enumeration_case_counts`: Finite state counts prove bounded exhaustive enumeration rather than sampled evidence.
- `support_conditional_joint_normalization_and_marginal_deltas`: Exact enumeration, not sampling, is authoritative on bounded cases.
- `deterministic_impossible_batch_seed_and_serialization_controls`: Deterministic factors, impossible states, batch shape, seed replay, and JSON round trips are checked together.
- `wire_order_category_type_cycle_dangling_and_invalid_mass_negative_controls`: Negative controls prove validators and the independent reference catch subtle graph and indexing failures.
- `torx_compatibility_scope`: Names the real exercised version/API; source resemblance is not compatibility.
- `deterministic_rebuild_checksum`: A second construction must reproduce the same program and exact-semantics commitment.
- `typed_stochastic_ir_ready_score`: Exactly one only when exact semantics, controls, serialization, protected files, tests, and real Torx compatibility pass.
- `protected_files_unchanged`: Conductor and reconciler-owned files remain byte-identical during artifact construction.
- `duration_s`: Measured deterministic construction time is reported without padding.
- `inference_substrate`: Use `jax_cpu_exact_stochastic_program` or `carnot_only_blocked_torx_compatibility` honestly.
- `verifier_is_oracle`: The independent exact finite enumerator is the probability oracle for bounded cases.
- `missing_verifier_gaps`: Any absent Torx or non-enumerated semantic evidence is explicit.
- `field_provenance`: Every field traces to the prompt, spec, workflow, source, tests, command receipts, or package metadata.
- `test_commands`: Commands document focused unit/spec coverage, structured gate, exact probability, Torx adapter, serialization, negative controls, protected-file, E2E, global pytest, and root-clutter checks.
- `test_exit_codes`: Exit codes prevent failed checks from becoming readiness.
- `reproducibility_checksum`: The artifact hash detects source, schema, probability, adapter, test, or protected-file drift.
- `honest_verdict`: Use `complete_ready:`, `complete_carnot_only:`, `complete_null:`, or `blocked:` and state the exact compiler boundary.

### SCENARIO-SAMPLE-6152-VALIDATION: Typed Graphs Fail Closed

**Given** an Exp6152 PSC/DFG payload with typed binary and categorical wires
**When** validation inspects cycles, edges, wire types, category indexes,
probability masses, and seeds
**Then** valid acyclic graphs are accepted
**And** unsupported cycles, dangling wires, type mismatches, invalid mass,
category-index bugs, and ambiguous seeds are rejected deterministically.

### SCENARIO-SAMPLE-6152-EXACT: Enumeration Owns Probabilities

**Given** the bounded Exp6145 stochastic constraint workflow
**When** the Carnot executor enumerates every finite state
**Then** support, conditional and joint probabilities, normalization,
marginals, deterministic factors, impossible states, batches, and seed replay
match an independent exact reference at the preregistered tolerance.

### SCENARIO-SAMPLE-6152-SERIALIZATION-TORX: Stable JSON And Real Adapter

**Given** the validated Exp6152 IR and pinned Torx package metadata
**When** the IR is serialized, deserialized, and passed through the Torx adapter
**Then** serialization preserves exact semantics and checksums
**And** the adapter reports a real exercised Torx version/API or blocks
compatibility without claiming readiness.

## Implementation Status (REQ-SAMPLE-6152)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6152 | Planned (`python/carnot/experiment_6152_typed_stochastic_constraint_ir.py`, `results/experiment_6152_typed_stochastic_constraint_ir.json`) | Planned (`tests/python/test_experiment_6152_typed_stochastic_constraint_ir.py`) |

### REQ-SAMPLE-6153: Thermalized Program Error Audit

Carnot MUST provide Exp6153 at
`python/carnot/experiment_6153_thermalized_program_error_audit.py` and write
`results/experiment_6153_thermalized_program_error_audit.json` without
modifying `scripts/research_conductor.py`. Exp6153 SHALL use Exp6152's exact
typed PSC/DFG program as the semantic oracle, compile every eligible Exp6152
factor into a software EBM kernel through pinned Torx and THRML-compatible
interfaces, precommit a factor-to-joint error bound before joint evaluation,
and compare isolated factor fitting with context-matched fitting under
resource-matched software-only conditions.

Sub-requirements:
- REQ-SAMPLE-6153-1: Preconditions SHALL force `JAX_PLATFORMS=cpu`, recompute
  the structured gate, authenticate Exp6152 IR/executor/exact-reference hashes,
  record Torx and THRML-compatible version/API receipts, hash exact references,
  retired lineage, output paths, and protected files, and block readiness if
  the real software interfaces required by the claim are unavailable.
- REQ-SAMPLE-6153-2: Factor eligibility SHALL be explicit and deterministic.
  Eligible finite-domain factors SHALL preserve input/output wire types,
  category order, support masks, and output normalization when compiled to the
  local EBM parameterization.
- REQ-SAMPLE-6153-3: Isolated and context-matched arms SHALL freeze the same
  Exp6152 program, factors, calibration-only training data, seeds, update
  count, sampler schedule, metrics, and evaluation groups before any joint
  divergence is read.
- REQ-SAMPLE-6153-4: The preregistered per-factor-to-joint bound SHALL be
  derived and SHA-256 hashed from measured local conditional errors before
  exact or sampled joint outputs are evaluated. A post-hoc bound SHALL NOT
  count as evidence.
- REQ-SAMPLE-6153-5: Exact enumerable cases SHALL report per-factor and joint
  total-variation/KL divergence exactly. Sampled bounded cases SHALL use
  matched seeds/sample counts and report uncertainty, lag autocorrelation,
  effective sample size, nonconvergence, and support violations.
- REQ-SAMPLE-6153-6: The audit SHALL compare measured joint divergence with
  the preregistered compositional bound and SHALL report whether context
  matching improves or noninferiorly preserves the primary divergence relative
  to isolated fitting.
- REQ-SAMPLE-6153-7: Identity, deliberately bad factor, permuted-wire,
  unsupported-state, and overly-loose-bound controls SHALL fire under the audit
  interface.
- REQ-SAMPLE-6153-8: Exp6153 SHALL preserve the retired Exp1526-Exp1544
  boundary by producing no THRML/Carnot size sweep, no vendored-THRML parity
  table, and no device, latency, power, energy, or speedup claim.
- REQ-SAMPLE-6153-9: The terminal artifact SHALL include `status`,
  `preconditions_checked`, `structured_gate_receipt`,
  `prior_failure_and_operator_override_receipts`,
  `upstream_ir_executor_and_exact_reference_hashes`,
  `torx_thrml_versions_commits_import_and_api_receipts`,
  `factor_eligibility_and_compilation_manifest`,
  `isolated_and_context_matched_training_config`,
  `preregistered_per_factor_to_joint_error_bound`,
  `exact_and_sampled_case_counts`,
  `per_factor_and_joint_distribution_divergences`,
  `bound_slack_and_violation_counts`,
  `context_matched_minus_isolated_intervals`,
  `autocorrelation_effective_sample_size_and_convergence`,
  `identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls`,
  `retired_parity_scaling_nonreuse_receipt`, `hardware_execution_claimed`,
  `latency_power_energy_and_speedup_claimed`,
  `thermalized_program_ready_score`, `retirement_triggered`,
  `protected_files_unchanged`, `duration_s`, `inference_substrate`,
  `verifier_is_oracle`, `missing_verifier_gaps`, `field_provenance`,
  `test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
  `honest_verdict`. `inference_substrate` SHALL equal
  `jax_cpu_software_thermalization`. `hardware_execution_claimed` and
  `latency_power_energy_and_speedup_claimed` SHALL be bare `false`.
  `honest_verdict` SHALL start with `complete_ready:`,
  `complete_bound_violated:`, `retired:`, or `blocked:` and state whether
  program-level error composition held.

Required field principles:

- `status`: Terminal state separates ready, bound-violated, retired, and blocked Exp6153 outcomes.
- `preconditions_checked`: Records CPU JAX, upstream gates, interface availability, output paths, exact references, retired lineage, and protected files before evaluation.
- `structured_gate_receipt`: Recomputes Exp6145 and Exp6152 readiness instead of trusting stale downstream claims.
- `prior_failure_and_operator_override_receipts`: Preserves the operator override that distinguishes Exp6153 from retired THRML parity scaling.
- `upstream_ir_executor_and_exact_reference_hashes`: Binds Exp6152 IR, executor, exact reference, artifact, and tests before factor replacement.
- `torx_thrml_versions_commits_import_and_api_receipts`: Proves the pinned Torx and THRML-compatible software interfaces actually imported and executed.
- `factor_eligibility_and_compilation_manifest`: Shows exactly which Exp6152 factors were eligible, compiled, support-preserving, and normalized.
- `isolated_and_context_matched_training_config`: Freezes calibration data, seeds, steps, schedules, arms, and metrics before outcomes.
- `preregistered_per_factor_to_joint_error_bound`: Hashes the local-error-derived joint bound before joint distribution results are read.
- `exact_and_sampled_case_counts`: Separates exhaustive finite cases from matched sampled bounded cases.
- `per_factor_and_joint_distribution_divergences`: Reports local and end-to-end divergence for isolated and context-matched arms.
- `bound_slack_and_violation_counts`: Compares measured joint divergence with the precommitted bound and counts violations.
- `context_matched_minus_isolated_intervals`: States whether context matching improves or noninferiorly preserves primary divergence.
- `autocorrelation_effective_sample_size_and_convergence`: Reports sampling uncertainty, lag correlation, ESS, support violations, and nonconvergence.
- `identity_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls`: Proves positive and negative controls detect exact, bad, rewired, support-breaking, and uninformative-bound cases.
- `retired_parity_scaling_nonreuse_receipt`: Proves no size sweep or Carnot-versus-vendored-THRML parity table is produced.
- `hardware_execution_claimed`: Bare false prevents software thermalization from becoming a hardware claim.
- `latency_power_energy_and_speedup_claimed`: Bare false prevents software quality evidence from becoming a performance claim.
- `thermalized_program_ready_score`: Equals 1.0 only when interfaces execute, support is preserved, the bound holds, context matching is noninferior, controls fire, and no forbidden claim appears.
- `retirement_triggered`: Records whether this run crossed a retired-lineage rule.
- `protected_files_unchanged`: Confirms conductor and reconciliation files were byte-identical during artifact construction.
- `duration_s`: Reports real wall time without padding.
- `inference_substrate`: Declares `jax_cpu_software_thermalization`, not hardware, GPU, or LLM inference.
- `verifier_is_oracle`: Declares Exp6152 exact enumeration as the oracle for bounded cases.
- `missing_verifier_gaps`: Lists absent evidence rather than silently granting readiness.
- `field_provenance`: Maps each required field to prompt, spec, source, tests, artifacts, or package receipts.
- `test_commands`: Records focused unit/spec coverage, structured gate, interfaces, bounds, exact/sample divergence, controls, no-hardware, E2E, full pytest, and root-clutter checks.
- `test_exit_codes`: Prevents failed commands from becoming readiness evidence.
- `reproducibility_checksum`: Content-addresses the artifact with volatile duration and self-checksum blanked.
- `honest_verdict`: Uses a required terminal prefix and states whether program-level error composition held.

### SCENARIO-SAMPLE-6153-BOUND-PRECOMMIT: Bound Is Frozen Before Joint Results

**Given** Exp6152's exact finite stochastic program and compiled local EBM
factors
**When** Exp6153 computes per-factor conditional divergence
**Then** it derives and hashes a compositional TV/KL joint bound before exact
or sampled joint distribution outputs are evaluated
**And** any later joint result records the precommit hash it consumed.

### SCENARIO-SAMPLE-6153-CONTEXT-MATCHING: Matched Arms Share Work

**Given** isolated and context-matched factor-fitting arms
**When** both arms train and evaluate on Exp6152's fixed factors, calibration
data, seeds, steps, sampler schedule, and exact/sampled cases
**Then** context matching either improves or noninferiorly preserves the
primary joint divergence relative to isolated fitting
**And** all per-factor and joint divergence metrics remain tied to the same
precommitted bound family.

### SCENARIO-SAMPLE-6153-CONTROLS-SCOPE: Controls And Scope Boundaries Fire

**Given** the identity, bad-factor, permuted-wire, unsupported-state, and
overly-loose-bound controls
**When** Exp6153 runs the audit
**Then** the identity control is zero-error, the bad/re-wired/support-breaking
controls are detected, the loose bound is rejected as uninformative, and no
retired THRML/Carnot parity-scaling or hardware/performance claim is emitted.

## Implementation Status (REQ-SAMPLE-6153)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6153 | Planned (`python/carnot/experiment_6153_thermalized_program_error_audit.py`, `results/experiment_6153_thermalized_program_error_audit.json`) | Planned (`tests/python/test_experiment_6153_thermalized_program_error_audit.py`) |

### REQ-SAMPLE-6166: Mode-Jumping Factor Thermalization

Carnot MUST provide Exp6166 at
`python/carnot/experiment_6166_mode_jumping_factor_thermalization.py` and write
`results/experiment_6166_mode_jumping_factor_thermalization.json` without
modifying `scripts/research_conductor.py`. Exp6166 SHALL use Exp6152's typed
PSC/DFG IR and Exp6153's software factor-replacement interfaces to test one
exactly enumerable multimodal categorical factor with Conditional NCE style
mode-jumping noise. The experiment is software simulation only.

Sub-requirements:
- REQ-SAMPLE-6166-MULTIMODAL-FACTOR: The factor SHALL have at least two
  separated modes, frozen labels, support, exact probabilities, mode masses,
  a known relative mode-mass ratio, and at least one unsupported state for
  support-violation controls.
- REQ-SAMPLE-6166-EXACT-ENUMERATION: Exact factor and joint distributions SHALL
  be enumerated through Exp6152's typed program executor, and all exact,
  approximate, identity, bad, and permuted arms SHALL lower through the same
  Exp6153 software-kernel execution interface.
- REQ-SAMPLE-6166-LOCAL-CNCE: A local-only CNCE arm SHALL train only from
  fixed local noise pairs and SHALL not copy exact log probabilities or the
  exact table into the approximate arm.
- REQ-SAMPLE-6166-CROSS-MODE-NOISE: A local-plus-cross-mode-jump CNCE arm
  SHALL add deliberate cross-mode pairs that compare separated modes directly.
- REQ-SAMPLE-6166-MATCHED-TRAINING: The local-only and mode-jump arms SHALL
  predeclare matching seeds, total pair budget, optimizer, step count, support
  mask, local noise, cross-mode noise, and primary TV/KL endpoint before
  fitting.
- REQ-SAMPLE-6166-NONZERO-ERROR: Approximate factor and/or joint divergence
  SHALL be finite and strictly positive, while the exact-table identity arm
  SHALL remain zero-error.
- REQ-SAMPLE-6166-RELATIVE-MODE-MASS: The artifact SHALL report relative
  mode-mass ratio error for local-only and mode-jump arms and use mode-jump
  improvement as a readiness gate.
- REQ-SAMPLE-6166-FACTOR-JOINT-DIVERGENCE: The artifact SHALL report exact
  factor and joint TV, KL, support, normalization, and mode-mass errors for
  identity, local-only, mode-jump, bad, and permuted controls.
- REQ-SAMPLE-6166-COMPOSITION-BOUND: The factor-to-joint bound SHALL be
  derived and SHA-256 hashed before joint evaluation, and measured joint TV
  SHALL be compared with the precommitted bound.
- REQ-SAMPLE-6166-CONTROLS: Identity, no-jump, wrong-jump, bad-factor,
  permuted-wire, unsupported-state, and loose-bound controls SHALL fire under
  the same audit interface.
- REQ-SAMPLE-6166-RETIRED-NONREUSE: Exp6166 SHALL preserve the retired
  THRML/Carnot parity-scaling lineage by producing no size sweep, no parity
  table, and no retired THRML scaling claim.
- REQ-SAMPLE-6166-SOFTWARE-ONLY: `hardware_execution_claimed` and
  `latency_power_energy_and_speedup_claimed` SHALL be bare `false`, and
  `inference_substrate` SHALL equal
  `jax_cpu_software_multimodal_factor_thermalization`.

The terminal artifact SHALL include `status`, `preconditions_checked`,
`prior_failure_and_operator_override_receipts`,
`upstream_ir_executor_bound_and_software_version_hashes`,
`cnce_source_and_algorithm_receipt`,
`exact_multimodal_factor_support_distribution_and_mode_masses`,
`frozen_local_and_cross_mode_noise_distributions`,
`matched_training_configs_seeds_samples_and_parameters`,
`exact_local_only_mode_jump_bad_and_permuted_arm_receipts`,
`factor_and_joint_tv_kl_and_mode_mass_ratio_errors`,
`deliberately_nonzero_error_receipt`,
`preregistered_factor_to_joint_bound`, `bound_slack_and_violation_counts`,
`seed_intervals_and_convergence`,
`identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls`,
`retired_parity_scaling_nonreuse_receipt`, `hardware_execution_claimed`,
`latency_power_energy_and_speedup_claimed`,
`mode_jumping_factor_thermalization_ready_score`, `retirement_triggered`,
`protected_files_unchanged`, `duration_s`, `inference_substrate`,
`verifier_is_oracle`, `missing_verifier_gaps`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`. `honest_verdict` SHALL start with `complete_positive:`,
`complete_null:`, `retired:`, or `blocked:` and state whether mode jumping
improved composition.

Required field principles:

- `status`: Terminal state separates positive, null, retired, and blocked Exp6166 outcomes.
- `preconditions_checked`: Hashes CPU mode, upstream typed IR, software interfaces, CNCE source reachability, retired lineage, output paths, and protected files before fitting.
- `prior_failure_and_operator_override_receipts`: Records why Exp6153's exact replacement was vacuous and why this nonzero-error software approximation is allowed without reopening retired parity scaling.
- `upstream_ir_executor_bound_and_software_version_hashes`: Binds Exp6152 IR/executor, Exp6153 bound/executor code, Torx, vendored THRML, sampler paths, outputs, and protected files.
- `cnce_source_and_algorithm_receipt`: Names OpenReview `07OWUWmUHp`, records reachable source metadata, and states the local conditional pairwise objective used.
- `exact_multimodal_factor_support_distribution_and_mode_masses`: Freezes labels, support mask, exact probabilities, separated modes, mode masses, and relative mode-mass ratio before fitting.
- `frozen_local_and_cross_mode_noise_distributions`: Freezes local pair noise and cross-mode jump noise before training outcomes are read.
- `matched_training_configs_seeds_samples_and_parameters`: Freezes seeds, samples, optimizer, step count, learning rate, support mask, arm budgets, and primary endpoints before fitting.
- `exact_local_only_mode_jump_bad_and_permuted_arm_receipts`: Keeps identity, local-only, mode-jump, deliberately bad, and permuted arms separately named and lowered through the same software-kernel interface.
- `factor_and_joint_tv_kl_and_mode_mass_ratio_errors`: Reports exact factor/joint divergence, normalization, support, and mode-mass ratio error for every arm.
- `deliberately_nonzero_error_receipt`: Proves approximate divergence is finite and strictly positive while the identity arm remains zero.
- `preregistered_factor_to_joint_bound`: Hashes the factor-derived joint TV/KL bound before joint evaluation.
- `bound_slack_and_violation_counts`: Compares measured joint TV with the precommitted bound and counts violations.
- `seed_intervals_and_convergence`: Reports deterministic seed intervals, loss trends, finite parameters, and convergence diagnostics.
- `identity_no_jump_wrong_jump_bad_factor_permuted_wire_unsupported_state_and_loose_bound_controls`: Proves exact identity, no-jump/local-only weakness, wrong-jump degradation, bad factor, permuted category wiring, support violation, and loose-bound rejection controls fire.
- `retired_parity_scaling_nonreuse_receipt`: Proves no THRML/Carnot scaling sweep or parity table is produced.
- `hardware_execution_claimed`: Bare false prevents this software semantic result from becoming a hardware claim.
- `latency_power_energy_and_speedup_claimed`: Bare false prevents quality evidence from becoming a performance claim.
- `mode_jumping_factor_thermalization_ready_score`: Equals 1.0 only when preconditions pass, nonzero finite error exists, mode jumping improves local-only, the bound holds, controls fire, protected files are unchanged, tests pass, and no forbidden claim appears.
- `retirement_triggered`: Records whether this run crossed a retired-lineage rule.
- `protected_files_unchanged`: Confirms conductor and reconciliation files were byte-identical during artifact construction.
- `duration_s`: Reports real wall time without padding.
- `inference_substrate`: Declares `jax_cpu_software_multimodal_factor_thermalization`, not hardware, GPU, or LLM inference.
- `verifier_is_oracle`: Declares exact finite enumeration as the oracle.
- `missing_verifier_gaps`: Lists absent evidence rather than silently granting readiness.
- `field_provenance`: Maps each required field to prompt, spec, source, tests, artifacts, or package receipts.
- `test_commands`: Records focused unit/spec coverage, version/API, multimodal exact reference, CNCE arms, nonzero-error, bound, controls, no-hardware, JAX CPU, schema, adversarial, protected-file, E2E, full pytest, and root-clutter checks.
- `test_exit_codes`: Prevents failed commands from becoming readiness evidence.
- `reproducibility_checksum`: Content-addresses the artifact with volatile duration and self-checksum blanked.
- `honest_verdict`: Uses a required terminal prefix and states whether mode jumping improved composition.

### SCENARIO-SAMPLE-6166-MULTIMODAL-CNCE: Mode Jumps Improve Relative Mass

**Given** the frozen multimodal categorical factor and matched CNCE training
budgets
**When** local-only and local-plus-cross-mode-jump arms fit approximate
support-masked software kernels
**Then** the identity arm has zero error
**And** both approximate arms have finite strictly positive error
**And** the mode-jump arm has lower joint TV and lower relative mode-mass ratio
error than the local-only arm.

### SCENARIO-SAMPLE-6166-BOUND-CONTROLS: Bound And Controls Fail Closed

**Given** the precommitted factor-to-joint bound and identity/no-jump/
wrong-jump/bad/permuted/unsupported/loose controls
**When** Exp6166 evaluates exact joint outputs
**Then** joint TV respects the precommitted bound for the approximate arms
**And** each control fires under the same typed software execution boundary
**And** no retired parity-scaling, hardware, latency, power, energy, or
speedup claim is emitted.

## Implementation Status (REQ-SAMPLE-6166)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6166 | Planned (`python/carnot/experiment_6166_mode_jumping_factor_thermalization.py`, `results/experiment_6166_mode_jumping_factor_thermalization.json`) | Planned (`tests/python/test_experiment_6166_mode_jumping_factor_thermalization.py`) |

### REQ-SAMPLE-6180: Exp6166 Reproducibility Adjudication

Carnot MUST provide Exp6180 at
`python/carnot/experiment_6180_exp6166_reproducibility_adjudication.py` and
write
`results/experiment_6180_exp6166_reproducibility_adjudication.json` without
modifying `scripts/research_conductor.py`, without rewriting any Exp6166
source, test, or result artifact, and without promoting any hardware,
latency, power, energy, or speedup claim.

Sub-requirements:
- REQ-SAMPLE-6180-IMMUTABLE-SOURCE: Exp6180 SHALL snapshot Exp6166 source,
  test, result, sampler spec, exclusion manifest, protected files, and git
  status before adjudication and compare the same byte hashes after
  adjudication.
- REQ-SAMPLE-6180-NO-REFIT: Exp6180 SHALL NOT invoke Exp6166 training or
  fitting functions. It SHALL recompute declared metrics only from the
  immutable Exp6166 artifact probabilities and existing exact software
  execution code.
- REQ-SAMPLE-6180-STOCHASTIC-REPLAY: Exp6180 SHALL replay frozen stochastic
  seeds and conditional-pair hashes from Exp6166 training receipts without
  changing the historical artifact.
- REQ-SAMPLE-6180-ISOLATED-TESTS: Exp6180 SHALL record focused task-owned
  reproducibility checks and classify nonzero suite-level receipts separately
  from the Exp6166 scientific result.
- REQ-SAMPLE-6180-OLD-FAILURE-CLASSIFICATION: Exp6180 SHALL preserve
  Exp6166's historical `blocked` state exactly while classifying the old
  full-suite exit 2 using focused Exp6166 receipts and Exp6170 task-isolation
  evidence.
- REQ-SAMPLE-6180-NO-HARDWARE-PROMOTION: Exp6180 SHALL emit bare false for
  hardware execution and performance-claim fields and SHALL use a
  software-only inference substrate:
  `jax_cpu_software_exp6166_artifact_replay`.
- REQ-SAMPLE-6180-PROTECTED-FILE: Exp6180 SHALL prove
  `scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, and
  `_bmad/traceability.md` remain byte-identical.
- REQ-SAMPLE-6180-CHECKSUM: Exp6180 SHALL content-address its artifact with a
  checksum that excludes only volatile duration and the checksum field itself.
  Focused tests SHALL live at
  `tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py`.

The terminal artifact SHALL include `status`, `experiment_id`, `run_date`,
`preconditions_checked`, `immutable_exp6166_byte_snapshot`,
`no_refit_receipt`, `stochastic_replay_receipt`,
`recomputed_metric_determination`, `old_full_suite_failure_classification`,
`companion_determination`, `no_hardware_promotion_receipt`,
`protected_files_unchanged`, `before_after_byte_comparison`,
`field_provenance`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and `honest_verdict`.

Required field principles:

- `status`: Terminal state for the companion adjudication, not a rewrite of Exp6166.
- `experiment_id`: Names only the Exp6180 companion artifact.
- `run_date`: Pins the operator-declared adjudication date.
- `preconditions_checked`: Snapshots Exp6166 bytes, stochastic seeds, tests, exclusion state, protected files, and git status before work.
- `immutable_exp6166_byte_snapshot`: Records byte hashes for the Exp6166 result, source, tests, sampler spec, and exclusion manifest.
- `no_refit_receipt`: States that Exp6166 fitting functions were not invoked and metrics came from historical artifact probabilities.
- `stochastic_replay_receipt`: Replays deterministic pair-sample hashes for the frozen Exp6166 seeds without changing Exp6166.
- `recomputed_metric_determination`: Recomputes declared TV, KL, support, normalization, and mode-ratio metrics from immutable probabilities.
- `old_full_suite_failure_classification`: Classifies the historical full-suite exit 2 while preserving Exp6166's blocked state.
- `companion_determination`: Separates the positive software-only stochastic determination from the historical blocked artifact.
- `no_hardware_promotion_receipt`: Prevents stochastic software evidence from becoming hardware, latency, power, energy, or speedup evidence.
- `protected_files_unchanged`: Confirms conductor and reconciler-owned files stayed byte-identical.
- `before_after_byte_comparison`: Compares immutable Exp6166 and protected-file bytes before and after adjudication.
- `field_provenance`: Maps every required field to the prompt, spec, immutable artifacts, tests, and command receipts.
- `test_commands`: Records focused task-owned tests, new-code coverage, and the single classified full-suite receipt.
- `test_exit_codes`: Stores command exit codes so failures cannot become silent readiness evidence.
- `reproducibility_checksum`: Hashes artifact content with only duration and self-checksum blanked.
- `honest_verdict`: Uses a terminal prefix and states both the reproduced software result and preserved historical block.

### SCENARIO-SAMPLE-6180-READ-ONLY-ADJUDICATION: Existing Evidence Is Replayed Without Refit

**Given** the committed Exp6166 result artifact and module constants
**When** Exp6180 recomputes metric receipts
**Then** it does not call Exp6166 training
**And** it matches the historical arm metrics, stochastic seed hashes, and
software-only claim fields.

### SCENARIO-SAMPLE-6180-HISTORICAL-BLOCK-PRESERVED: Full-Suite Exit 2 Is Classified

**Given** Exp6166 recorded focused checks at zero and the full Python suite at
exit 2
**When** Exp6180 builds its companion determination
**Then** the historical Exp6166 status remains `blocked`
**And** the full-suite failure is classified as unrelated pre-existing
repository-suite state rather than a failed stochastic reproducibility check.

## Implementation Status (REQ-SAMPLE-6180)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6180 | Planned (`python/carnot/experiment_6180_exp6166_reproducibility_adjudication.py`, `results/experiment_6180_exp6166_reproducibility_adjudication.json`) | Planned (`tests/python/test_experiment_6180_exp6166_reproducibility_adjudication.py`) |

### REQ-SAMPLE-6194: Fixed Mode-Jump Rust/PyO3 Sampler Parity

Carnot MUST provide Exp6194 at
`python/carnot/experiment_6194_mode_jump_rust_pyo3_parity.py` and write
`results/experiment_6194_mode_jump_rust_pyo3_parity.json` without modifying
Exp6166, Exp6180, `scripts/research_conductor.py`, THRML scaling artifacts,
two-axis tempering code, or hardware-speed evidence. Exp6194 SHALL port only
the frozen Exp6166/Exp6180 software mode-jump categorical mechanism into the
existing Rust sampler crate and expose that mechanism through PyO3.

Sub-requirements:
- REQ-SAMPLE-6194-PROPOSAL: The Rust sampler SHALL implement the frozen
  Exp6166 local-plus-cross-mode proposal table over the supported categorical
  labels and SHALL reject malformed, non-normalized, asymmetric-support, or
  unsupported-state proposal configurations without fallback.
- REQ-SAMPLE-6194-ACCEPTANCE: The transition SHALL use the exact
  Metropolis-Hastings log acceptance
  `log pi(y)-log pi(x)+log q(x|y)-log q(y|x)`, with
  `E(label)=-log pi(label)`, and SHALL record proposed label, proposal
  uniform, acceptance uniform, energies, proposal log probabilities, log
  acceptance, acceptance probability, accept decision, and state transition.
- REQ-SAMPLE-6194-RNG-DETERMINISM: The sampler SHALL use a seedable,
  serializable RNG state and SHALL replay identical short chains for identical
  `(configuration, initial state, seed)` across Rust and Python.
- REQ-SAMPLE-6194-SERIALIZATION: The Rust and PyO3 state surfaces SHALL expose
  typed state snapshot, restore, and serialization/deserialization operations
  that validate label membership, counters, and schema before use.
- REQ-SAMPLE-6194-PYO3-BINDING: The PyO3 binding SHALL expose typed
  construction, one-step, multi-step, energy/proposal queries, snapshot,
  restore, and serialization operations with explicit `ValueError` failures
  for invalid inputs.
- REQ-SAMPLE-6194-EXACT-PARITY: Exp6194 SHALL derive immutable Python fixtures
  from the committed Exp6166 result and Exp6180 adjudication evidence, then
  compare exact short-chain energies, proposals, uniforms, accept decisions,
  states, counters, and final serialized state against Rust/PyO3.
- REQ-SAMPLE-6194-DISTRIBUTION-PARITY: Exp6194 SHALL predeclare long-run
  sample count, burn-in, total-variation, KL, acceptance, autocorrelation, and
  ESS tolerances before inspecting Rust outcomes, then compare Python and
  Rust/PyO3 empirical frequencies against the exact Exp6166 target.
- REQ-SAMPLE-6194-ERROR-HANDLING: Broken configurations, bad labels, corrupt
  snapshots, corrupt serialized states, zero-probability targets, invalid
  proposal rows, and invalid step counts SHALL fail closed.
- REQ-SAMPLE-6194-NO-HARDWARE: `hardware_or_speedup_claimed` SHALL be the bare
  boolean `false`, timing SHALL be diagnostic only, and the inference substrate
  SHALL equal `local_cpu_rust_pyo3_cross_runtime_sampler_parity`.
- REQ-SAMPLE-6194-DETERMINATION-PRESERVATION: Exp6194 SHALL snapshot the
  immutable Exp6166 and Exp6180 hashes and verdicts before and after the run,
  preserving Exp6166's historical blocked determination and Exp6180's positive
  companion determination without relabeling either artifact.

The terminal artifact SHALL include `status`, `preconditions_checked`,
`immutable_exp6166_exp6180_hash_and_verdict_receipt`,
`fixed_algorithm_equations_config_and_seed`,
`rust_module_and_pyo3_binding_paths`, `rust_python_api_contract`,
`exact_transition_fixture_hash_and_parity_matrix`,
`distribution_frequency_tv_kl_metrics`,
`acceptance_autocorrelation_and_ess_metrics`,
`serialization_snapshot_restore_and_error_receipts`,
`deterministic_seed_replay_receipt`,
`task_owned_rust_python_test_commands_and_exit_codes`,
`nonzero_command_classification`, `timing_diagnostic_only`,
`hardware_or_speedup_claimed`, `historical_artifacts_unchanged`,
`mode_jump_rust_pyo3_ready_score`, `protected_files_unchanged`,
`duration_s`, `inference_substrate`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`. `mode_jump_rust_pyo3_ready_score` SHALL equal exactly `1.0`
only when exact short-chain parity, preregistered distribution and quality
tolerances, serialization/error coverage, historical-artifact preservation,
protected-file preservation, and zero task-owned test failures all hold.
`honest_verdict` SHALL start with `complete_ready:`, `complete_partial:`,
`retired:`, or `blocked:` and state exact parity plus every classified nonzero
command.

Required field principles:

- `status`: Terminal state separates ready, partial, retired, and blocked Exp6194 outcomes.
- `preconditions_checked`: Records Exp6184 preflight, immutable upstream hashes, toolchain, build surface, output root, exclusions, git status, protected files, and root clutter before implementation evidence is interpreted.
- `immutable_exp6166_exp6180_hash_and_verdict_receipt`: Preserves the historical blocked Exp6166 determination and positive Exp6180 companion determination by byte hash.
- `fixed_algorithm_equations_config_and_seed`: Freezes labels, target probabilities, proposal table, MH acceptance equation, RNG, seed, initial state, and tolerances before Rust outcomes are read.
- `rust_module_and_pyo3_binding_paths`: Lists the Rust kernel, crate exports, PyO3 binding, compatibility import, Python fixture, tests, and artifact paths touched by the port.
- `rust_python_api_contract`: Names typed construction, one-step, multi-step, energy/proposal queries, snapshot, restore, serialization, and error behavior.
- `exact_transition_fixture_hash_and_parity_matrix`: Content-addresses the immutable short-chain fixture and records field-by-field Python/Rust parity.
- `distribution_frequency_tv_kl_metrics`: Measures long-run empirical frequencies and TV/KL against the exact target without timing or hardware claims.
- `acceptance_autocorrelation_and_ess_metrics`: Reports quality diagnostics so a correct stationary distribution is not inferred from frequencies alone.
- `serialization_snapshot_restore_and_error_receipts`: Proves snapshot/restore/serialization round trips and corrupt inputs fail closed.
- `deterministic_seed_replay_receipt`: Proves repeated Python and Rust/PyO3 runs with the same seed replay exactly.
- `task_owned_rust_python_test_commands_and_exit_codes`: Stores task-owned command receipts so failed local checks cannot become readiness evidence.
- `nonzero_command_classification`: Classifies every nonzero command separately from exact parity and readiness.
- `timing_diagnostic_only`: Bare true prevents diagnostic timing from becoming a speed claim.
- `hardware_or_speedup_claimed`: Bare false prevents Rust/PyO3 parity from becoming FPGA, TSU, CUDA, THRML, latency, power, energy, or speedup evidence.
- `historical_artifacts_unchanged`: Proves Exp6166 and Exp6180 artifacts were not rewritten.
- `mode_jump_rust_pyo3_ready_score`: Equals 1.0 only when exact parity, distribution/quality tolerances, serialization/error coverage, preservation, and task-owned tests all pass.
- `protected_files_unchanged`: Confirms conductor and reconciler-owned files stayed byte-identical.
- `duration_s`: Reports real wall time without padding.
- `inference_substrate`: Declares `local_cpu_rust_pyo3_cross_runtime_sampler_parity`, not LLM, GPU, FPGA, TSU, or THRML scaling.
- `field_provenance`: Maps every required field to prompt, spec, immutable artifacts, source, tests, commands, or computed fixtures.
- `test_commands`: Records focused cargo, PyO3, Python, spec, artifact, preservation, adversarial, protected-file, root-clutter, and suite receipts.
- `test_exit_codes`: Stores exit codes for every recorded command.
- `reproducibility_checksum`: Content-addresses the artifact after blanking only duration and the checksum field.
- `honest_verdict`: Uses a required terminal prefix and states exact parity plus every classified nonzero command.

### SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY: Rust Replays Frozen Mode-Jump Steps

**Given** immutable Exp6166/Exp6180 hashes, the frozen categorical target, the
cross-mode proposal table, a seed, and a short-chain Python fixture
**When** the Rust/PyO3 mode-jump sampler runs the same initial state
**Then** proposal uniforms, proposed labels, acceptance uniforms, energies,
proposal log probabilities, log acceptance, acceptance probabilities,
accept decisions, current labels, counters, RNG state, snapshots, and serialized
state match the Python fixture exactly within the preregistered tolerance.

### SCENARIO-SAMPLE-6194-DISTRIBUTION-QUALITY-PARITY: Long-Run Diagnostics Match Target

**Given** preregistered long-run sample count, burn-in, TV/KL, acceptance,
autocorrelation, and ESS tolerances
**When** Python and Rust/PyO3 run from the same frozen configuration and seed
**Then** empirical frequencies remain within tolerance of the exact target,
Python/Rust frequency deltas stay within tolerance, acceptance and quality
diagnostics are recorded, and timing remains diagnostic only.

### SCENARIO-SAMPLE-6194-SERIALIZATION-ERROR-PRESERVATION: Boundaries Fail Closed

**Given** valid snapshots, serialized states, and malformed controls
**When** Rust and PyO3 restore states, deserialize payloads, or receive invalid
labels, probabilities, proposal rows, snapshots, serialized strings, or step
counts
**Then** valid states round-trip exactly, corrupt inputs raise explicit errors,
Exp6166/Exp6180 historical artifacts stay byte-identical, protected files stay
byte-identical, and no hardware or speedup claim is emitted.

## Implementation Status (REQ-SAMPLE-6194)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6194 | Planned (`crates/carnot-samplers/src/mode_jump.rs`, `crates/carnot-python/src/mode_jump.rs`, `python/carnot/experiment_6194_mode_jump_rust_pyo3_parity.py`, `results/experiment_6194_mode_jump_rust_pyo3_parity.json`) | Planned (`crates/carnot-samplers/tests/mode_jump.rs`, `tests/python/test_experiment_6194_mode_jump_rust_pyo3_parity.py`) |

### REQ-SAMPLE-6208: Mode-Jump Runtime Adapter Integration

Carnot MUST expose the qualified fixed mode-jump Rust/PyO3 kernel through the
existing runtime sampler selection boundary as an explicit, default-off
backend named `mode_jump_rust`. The default runtime sampler selection SHALL
remain `cpu` unless a caller explicitly selects `mode_jump_rust`, and the
mode-jump adapter SHALL still require an explicit runtime feature/config flag
before routing a supported run through Rust.

Sub-requirements:
- REQ-SAMPLE-6208-DEFAULT-OFF: `get_backend()` and
  `get_sampler_backend()` SHALL continue to select the current CPU sampler by
  default. `mode_jump_rust` SHALL be selectable only by explicit backend name,
  and Rust execution SHALL be disabled unless
  `enable_mode_jump_runtime=true` or `CARNOT_ENABLE_MODE_JUMP_RUNTIME=1`.
- REQ-SAMPLE-6208-FIXED-KERNEL: The adapter SHALL consume Exp6194's ready
  artifact hash, Rust kernel hash, PyO3 binding hash, fixed algorithm hash,
  labels, target probabilities, proposal table, RNG semantics, and state
  schema without changing the qualified transition mathematics.
- REQ-SAMPLE-6208-SHAPE-CONTRACT: The adapter SHALL accept only the frozen
  six-label categorical target as `biases.shape == (6,)` and proposal table as
  `couplings.shape == (6, 6)` with finite normalized probabilities and the
  frozen label order. Float64 C-contiguous inputs MAY use Rust when the feature
  flag and PyO3 symbols are available; declared compatibility cases such as
  float32 or non-C-contiguous valid inputs SHALL use exact Python fallback;
  malformed or unsupported shapes SHALL fail closed.
- REQ-SAMPLE-6208-EXACT-FALLBACK: Missing, broken, or incomplete Rust/PyO3
  bindings, disabled feature flags, declared Python compatibility, and
  supported-value but Rust-unsupported dtype/layout SHALL record
  `active_backend="python_exact_fallback"` with a non-empty fallback reason and
  SHALL produce samples, state, quality metrics, and checkpoints identical to
  the exact Python fallback.
- REQ-SAMPLE-6208-RUNTIME-ACCOUNTING: Runtime receipts SHALL preserve seeded
  quality, empirical distribution TV/KL and interval diagnostics,
  autocorrelation, ESS, accepted/attempted counts, serialization round trips,
  corrupt-state rejection, cancellation, timeout, invalid configuration, and
  unsupported-shape errors at the adapter boundary.
- REQ-SAMPLE-6208-ARTIFACT: Exp6208 SHALL write
  `results/experiment_6208_mode_jump_runtime_integration.json` after running
  matched CPU fixtures with preregistered seeds and adequate chain lengths.
  Timing SHALL be diagnostic only, SHALL NOT enter readiness, and
  `hardware_or_speed_power_energy_claimed` SHALL be the bare boolean `false`.

The terminal artifact SHALL include at minimum these top-level fields:
`status`, `exp6194_artifact_and_kernel_hashes`,
`runtime_adapter_paths_and_hashes`, `default_off_receipt`,
`config_and_feature_flag_contract`,
`supported_and_unsupported_shape_matrix`, `seeded_quality_parity`,
`distribution_tv_kl_and_interval_receipts`,
`autocorrelation_and_effective_sample_size`,
`serialization_roundtrip`, `cancellation_timeout_and_error_receipts`,
`exact_fallback_receipts`, `task_owned_test_commands_and_exit_codes`,
`unrelated_nonzero_command_classifications`,
`hardware_or_speed_power_energy_claimed`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`duration_s`, `reproducibility_checksum`, and `honest_verdict`.
`inference_substrate` SHALL equal
`production_python_samplerbackend_plus_rust_pyo3_mode_jump_cpu`.
`honest_verdict` SHALL start with `complete_ready:`,
`complete_partial:`, or `blocked:` and state default-off, fallback, parity,
and every classified nonzero command.

Required field principles:

- `status`: Distinguishes ready, partial, and blocked runtime integration outcomes.
- `exp6194_artifact_and_kernel_hashes`: Pins Exp6194 readiness and the qualified Rust/PyO3 kernel before adapter evidence is trusted.
- `runtime_adapter_paths_and_hashes`: Content-addresses the runtime adapter, factory registration, spec, tests, and terminal artifact path.
- `default_off_receipt`: Proves current CPU sampler selection remains the default and mode-jump Rust execution needs explicit opt-in.
- `config_and_feature_flag_contract`: Records the only accepted backend name, config flag, environment flag, seed, state, and fallback controls.
- `supported_and_unsupported_shape_matrix`: Separates Rust-supported, exact-fallback, and fail-closed input surfaces.
- `seeded_quality_parity`: Shows Rust and exact fallback replay the same seeded samples and final state.
- `distribution_tv_kl_and_interval_receipts`: Quantifies empirical distribution quality against the frozen target with intervals.
- `autocorrelation_and_effective_sample_size`: Reports mixing diagnostics rather than inferring quality from frequencies alone.
- `serialization_roundtrip`: Proves adapter checkpoints and serialized kernel state restore exactly and reject corruption.
- `cancellation_timeout_and_error_receipts`: Proves runtime interruption and bad inputs fail closed with explicit errors.
- `exact_fallback_receipts`: Lists every exercised fallback reason and its exact equivalence result.
- `task_owned_test_commands_and_exit_codes`: Stores focused Rust/PyO3/Python/spec/coverage command receipts.
- `unrelated_nonzero_command_classifications`: Prevents unrelated nonzero commands from masquerading as integration failure or success evidence.
- `hardware_or_speed_power_energy_claimed`: Bare false prevents this software adapter from claiming hardware, speed, power, or energy results.
- `inference_substrate`: Declares local CPU production sampler backend plus Rust/PyO3 mode-jump integration.
- `verifier_is_oracle`: States whether the verifier is the exact finite categorical target and fixed transition, not LLM judgment.
- `field_provenance`: Maps each artifact field to prompt, spec, source hashes, tests, commands, or deterministic computation.
- `field_principles`: Explains why each required field exists before the JSON shape is trusted.
- `duration_s`: Reports real wall time without converting it into a speed claim.
- `reproducibility_checksum`: Content-addresses the artifact after blanking duration and the checksum field.
- `honest_verdict`: Terminal verdict states readiness and all nonzero command classifications.

### SCENARIO-SAMPLE-6208-DEFAULT-OFF-FALLBACK: Runtime Selection Is Explicit

**Given** the production sampler factory, no mode-jump feature flag, and the
fixed Exp6194 categorical target
**When** callers request the default sampler, explicitly request
`mode_jump_rust`, or force exact fallback
**Then** the default remains `cpu`, the mode-jump backend records disabled or
declared fallback reasons until explicitly enabled, and fallback samples,
checkpoints, diagnostics, and serialized state match the exact Python path.

### SCENARIO-SAMPLE-6208-RUNTIME-PARITY: Enabled Adapter Matches Exact Fallback

**Given** Exp6194 ready evidence, the frozen target/proposal arrays, seed,
initial label, burn-in, and retained sample count
**When** `mode_jump_rust` runs with `enable_mode_jump_runtime=true`
**Then** Rust/PyO3 and exact fallback seeded samples, final state,
acceptance accounting, TV/KL, intervals, autocorrelation, ESS, checkpoint
round trip, and resumed run remain identical within preregistered tolerances.

### SCENARIO-SAMPLE-6208-BOUNDARY-ERRORS: Runtime Adapter Fails Closed

**Given** unsupported shapes, malformed probabilities, bad labels, corrupt
checkpoints, disabled or broken Rust bindings, cancellation, and timeout
controls
**When** the runtime adapter validates or advances a run
**Then** declared compatibility cases take exact fallback with receipts, while
unsupported shapes and corrupt inputs raise explicit errors, and the terminal
artifact still records `hardware_or_speed_power_energy_claimed=false`.

## Implementation Status (REQ-SAMPLE-6208)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6208 | Planned (`python/carnot/samplers/mode_jump_rust_backend.py`, `python/carnot/samplers/backend.py`, `python/carnot/experiment_6208_mode_jump_runtime_integration.py`, `results/experiment_6208_mode_jump_runtime_integration.json`) | Planned (`tests/python/samplers/test_mode_jump_rust_backend.py`, `tests/python/test_experiment_6208_mode_jump_runtime_integration.py`, `crates/carnot-samplers/tests/mode_jump.rs`) |

### REQ-SAMPLE-6220: Mode-Jump Runtime A/B Measurement

Carnot MUST provide Exp6220 at
`python/carnot/experiment_6220_mode_jump_runtime_ab.py` and write
`results/experiment_6220_mode_jump_runtime_ab.json`. The experiment SHALL
compare the Exp6208 exact Python fallback arm and the default-off
`mode_jump_rust` runtime arm on preregistered software fixtures, seeds,
schedules, burn-in, sample counts, observables, tolerances, and timing rules.
It SHALL keep the runtime feature default off and SHALL NOT make FPGA, TSU,
power, hardware, latency-speedup, or energy-efficiency claims.

Sub-requirements:
- REQ-SAMPLE-6220-SPEC-FIRST: Exp6220 SHALL freeze the fixture, seed,
  schedule, burn-in, sample count, observable, tolerance, and timing matrix
  before measuring either arm.
- REQ-SAMPLE-6220-SUPPORT: Each fixture/arm pair SHALL record whether the
  Exp6208 adapter accepted the frozen inputs, selected Rust/PyO3, selected exact
  Python fallback, or failed closed. Unsupported fixtures SHALL remain evidence
  for the boundary and SHALL NOT be forced through by changing the Exp6208
  target, proposal table, feature flag, or stopping rule.
- REQ-SAMPLE-6220-QUALITY: For every accepted fixture/arm pair, Exp6220 SHALL
  validate sample support, observable error, energy-moment error, TV/KL, ESS,
  autocorrelation, transition counts, and mode occupancy before interpreting
  timing.
- REQ-SAMPLE-6220-STATE: Exp6220 SHALL test checkpoint serialization, restart
  equivalence, backend-unavailable fallback, exact fallback, malformed state
  rejection, and Python/Rust/PyO3 consistency.
- REQ-SAMPLE-6220-TIMING: CPU wall time SHALL be diagnostic unless all quality
  gates pass for all required fixture arms and the preregistered uncertainty
  rule excludes parity. Timing SHALL NOT alter sampler settings.
- REQ-SAMPLE-6220-ARTIFACT: The terminal artifact SHALL classify task-owned
  command failures separately from pre-existing repository-wide nonzero checks,
  preserve default-off behavior, and emit bare
  `fpga_tsu_power_hardware_claim_count=0`.

The terminal artifact SHALL include these top-level fields:
`status`, `upstream_exp6194_and_exp6208_paths_hashes`,
`backend_paths_build_and_abi_receipts`,
`preregistered_fixture_seed_schedule_matrix`, `matched_arm_configuration`,
`support_validity_by_fixture_arm`,
`observable_and_energy_error_by_fixture_arm`,
`ess_and_autocorrelation_by_fixture_arm`,
`transition_and_mode_occupancy_counts`,
`serialization_and_restart_receipts`, `fallback_trigger_and_exactness`,
`cpu_thread_and_wall_time_receipts`, `quality_gate_passed`,
`timing_claim_allowed`, `default_off_preserved`,
`fpga_tsu_power_hardware_claim_count`,
`task_owned_and_preexisting_test_classification`,
`sampler_runtime_ready_score`, `protected_files_unchanged`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

Required field principles:

- `status`: Terminal state separates ready, partial, and blocked Exp6220 outcomes.
- `upstream_exp6194_and_exp6208_paths_hashes`: Pins the qualified parity and runtime-integration evidence before this A/B run is trusted.
- `backend_paths_build_and_abi_receipts`: Records adapter, factory, Rust kernel, PyO3 binding, import, and ABI availability.
- `preregistered_fixture_seed_schedule_matrix`: Freezes fixtures, seeds, schedules, burn-in, counts, observables, tolerances, and timing rule before outcomes.
- `matched_arm_configuration`: Proves the fallback and runtime arms use matched seeds, target, proposal, burn-in, and sample counts.
- `support_validity_by_fixture_arm`: Separates accepted support, exact fallback, Rust execution, and fail-closed unsupported fixtures.
- `observable_and_energy_error_by_fixture_arm`: Reports observable, mode-mass, and energy-moment errors before timing.
- `ess_and_autocorrelation_by_fixture_arm`: Reports mixing quality before any runtime interpretation.
- `transition_and_mode_occupancy_counts`: Shows accepted/proposed transitions and mode occupancy for supported runs.
- `serialization_and_restart_receipts`: Proves checkpoint round trip, restart equivalence, and malformed state rejection.
- `fallback_trigger_and_exactness`: Proves disabled, forced, unavailable, and dtype/layout fallback paths remain exact.
- `cpu_thread_and_wall_time_receipts`: Stores CPU/thread context and diagnostic wall-clock measurements only.
- `quality_gate_passed`: Bare boolean gates timing interpretation after support and quality checks.
- `timing_claim_allowed`: Bare boolean stays false unless quality passes and uncertainty excludes parity.
- `default_off_preserved`: Bare true only when the production default remains CPU and Rust execution needs explicit opt-in.
- `fpga_tsu_power_hardware_claim_count`: Bare integer must be `0` for this software-only task.
- `task_owned_and_preexisting_test_classification`: Keeps task-owned failures separate from pre-existing repository-wide nonzero checks.
- `sampler_runtime_ready_score`: Summarizes readiness from default-off, support, quality, state, fallback, tests, and no-claim gates.
- `protected_files_unchanged`: Confirms conductor and reconciliation files were not modified.
- `inference_substrate`: Declares local CPU software A/B measurement, not hardware, TSU, GPU, or LLM inference.
- `verifier_is_oracle`: States that the verifier is exact finite enumeration plus fixed transition receipts.
- `field_provenance`: Maps every required field to prompt, spec, source hashes, tests, commands, or computed fixtures.
- `field_principles`: Explains why each required field exists.
- `test_commands`: Records focused tests, coverage, artifact command, required suite, and applicable E2E receipts.
- `test_exit_codes`: Stores exit codes for every recorded command.
- `duration_s`: Reports real wall time without padding.
- `reproducibility_checksum`: Content-addresses the artifact with volatile duration and checksum blanked.
- `honest_verdict`: Uses a terminal prefix and states support, quality, timing, fallback, default-off, and nonzero-command classifications.

### SCENARIO-SAMPLE-6220-MATCHED-RUNTIME-QUALITY: Supported Fixture Compares Exactly

**Given** the frozen Exp6194 multimodal fixture, matched seed, proposal table,
burn-in, retained sample count, and timing rule
**When** Exp6220 runs exact Python fallback and enabled `mode_jump_rust`
**Then** support stays valid, samples and state match exactly, observable and
energy errors remain within tolerance, ESS and autocorrelation are recorded,
transition and mode occupancy counts are recorded, and timing remains
diagnostic until all gates permit it.

### SCENARIO-SAMPLE-6220-UNSUPPORTED-FIXTURE-BOUNDARY: Unsupported Fixture Fails Closed

**Given** a preregistered unimodal software fixture that is outside the frozen
Exp6208 target contract
**When** fallback and runtime arms receive the fixture without changing the
proposal, temperature, feature flag, or stopping rule
**Then** both arms fail closed with explicit support errors, the unsupported
fixture blocks broad timing claims, and the artifact records the boundary
instead of changing the backend contract.

### SCENARIO-SAMPLE-6220-STATE-FALLBACK-NOCLAIM: State And Fallback Gates Are Audited

**Given** valid checkpoints, corrupt checkpoints, disabled feature flags,
forced fallback, backend-unavailable controls, and malformed state payloads
**When** Exp6220 builds the A/B artifact
**Then** valid restarts match, corrupt states are rejected, fallback triggers
are exact, default-off behavior is preserved, protected files are unchanged,
and `fpga_tsu_power_hardware_claim_count` is the bare integer `0`.

## Implementation Status (REQ-SAMPLE-6220)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLE-6220 | Planned (`python/carnot/experiment_6220_mode_jump_runtime_ab.py`, `results/experiment_6220_mode_jump_runtime_ab.json`) | Planned (`tests/python/test_experiment_6220_mode_jump_runtime_ab.py`) |

### REQ-SAMPLER-6237: Activated Mode-Jump Sampler A/B

Carnot MUST provide Exp6237 at
`python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py` and write
`results/experiment_6237_activated_mode_jump_sampler_ab.json` without
modifying `scripts/research_conductor.py`. The experiment SHALL compare the
seeded exact Python fallback and enabled Rust/PyO3 mode-jump runtime arms on a
bounded preregistered fixture, seed, and budget matrix. The runtime feature
default SHALL remain off. No hardware, speedup, power, or energy-efficiency
claim is in scope.

Sub-requirements:
- REQ-SAMPLER-6237-PRECONDITIONS: Exp6237 SHALL freeze the fixture, seeds,
  transition budget, wall budget, metrics, equivalence margins, positive
  control, focused commands, protected-file hashes, and upstream hashes before
  sampler chains are measured.
- REQ-SAMPLER-6237-ACTIVATION: The mode-jump arm SHALL prove activation before
  any quality conclusion. Activation SHALL require Rust/PyO3 selection,
  nonzero mode-jump proposals, nonzero mode-jump acceptances, and a multimodal
  positive control. An inactive treatment SHALL produce `instrument_failure`.
  It SHALL NOT produce a null sampler verdict.
- REQ-SAMPLER-6237-MATCHED-AB: Each preregistered fixture/seed/budget cell
  SHALL run both arms with matched target, proposal table, initial state,
  burn-in, retained sample count, transition budget, and wall budget. Every
  preregistered main fixture arm SHALL be supported. Unsupported controls SHALL
  remain in `unsupported_or_failed_cells`.
- REQ-SAMPLER-6237-CHAIN-RECEIPTS: Exp6237 SHALL persist chain-level sample
  labels, transition counts, accepted counts, proposal counts, and mode-jump
  proposal and acceptance counts before aggregation.
- REQ-SAMPLER-6237-QUALITY: For each supported arm, Exp6237 SHALL report exact
  distribution error, energy moments, mode coverage, ESS, autocorrelation, and
  cost before paired decisions.
- REQ-SAMPLER-6237-EQUIVALENCE: Paired intervals SHALL compare mode jumping
  against the seeded fallback. The artifact SHALL distinguish positive,
  negative, equivalence-supported, inconclusive, and instrument-failure
  outcomes using preregistered equivalence bounds.
- REQ-SAMPLER-6237-CONTROLS: Focused tests SHALL cover unsupported fixtures,
  zero activation, degenerate chains, interruption/restart, Rust/Python parity,
  and default-off fallback behavior.
- REQ-SAMPLER-6237-NO-RECURSIVE-SUITE: The experiment CLI SHALL NOT invoke a
  repository-wide test suite. Repository validation MAY be recorded only as a
  separately run command receipt.

The terminal artifact SHALL include these top-level fields:
`status`, `upstream_paths_hashes_and_determinations`,
`literature_control_path_and_hash`,
`preregistered_fixture_seed_budget_matrix`,
`exact_reference_distribution_receipts`, `arm_support_matrix`,
`matched_arm_configuration`,
`jump_proposal_acceptance_and_transition_counts`,
`treatment_activation_score`, `multimodal_positive_control`,
`fallback_parity_control`, `distribution_quality_by_fixture_arm`,
`mode_coverage_ess_autocorrelation_by_fixture_arm`,
`wall_and_transition_costs`, `paired_intervals`,
`equivalence_bounds_and_decision`, `unsupported_or_failed_cells`,
`task_owned_and_preexisting_nonzero_command_ledger`,
`default_off_preserved`, `hardware_claim_count`,
`sampler_runtime_ready_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`. `hardware_claim_count` SHALL be the bare integer `0`.
`inference_substrate` SHALL equal
`local_cpu_software_activated_mode_jump_sampler_ab`.

Required field principles:

- `status`: Separates supported equivalence evidence from inconclusive, blocked, and instrument-failure outcomes.
- `upstream_paths_hashes_and_determinations`: Pins Exp6166, Exp6208, Exp6220, and sampler source evidence before this A/B is trusted.
- `literature_control_path_and_hash`: Pins the local arXiv:2608.05025 warning that inactive treatments cannot support sampler conclusions.
- `preregistered_fixture_seed_budget_matrix`: Freezes fixtures, seeds, transition budgets, wall budgets, metrics, margins, and positive controls before compute.
- `exact_reference_distribution_receipts`: Records the exact finite target, modes, support, and hashes used as the oracle.
- `arm_support_matrix`: Shows that every preregistered main fixture arm is supported and that controls fail closed.
- `matched_arm_configuration`: Proves the fallback and mode-jump arms share target, proposal, seed, initial state, burn-in, retained samples, and wall budget.
- `jump_proposal_acceptance_and_transition_counts`: Stores chain-level samples and transition counts before aggregation.
- `treatment_activation_score`: Equals 1.0 only when Rust/PyO3 ran and nonzero mode-jump proposals and acceptances occurred.
- `multimodal_positive_control`: Proves the fixture can exercise cross-mode moves before quality is interpreted.
- `fallback_parity_control`: Proves seeded exact fallback and active Rust/PyO3 replay the same transition stream.
- `distribution_quality_by_fixture_arm`: Reports exact-distribution error and energy moments for each supported fixture arm.
- `mode_coverage_ess_autocorrelation_by_fixture_arm`: Reports mode coverage, ESS, and autocorrelation instead of using frequency alone.
- `wall_and_transition_costs`: Reports matched transition budgets and diagnostic wall costs without making a speed claim.
- `paired_intervals`: Stores paired quality and cost intervals across seeds.
- `equivalence_bounds_and_decision`: Applies preregistered margins and emits positive, negative, equivalence-supported, inconclusive, or instrument-failure decisions.
- `unsupported_or_failed_cells`: Keeps unsupported fixtures, degenerate chains, interruption, and activation failures visible.
- `task_owned_and_preexisting_nonzero_command_ledger`: Separates task-owned command failures from separately classified repository-wide nonzero commands.
- `default_off_preserved`: Bare true only when the production default remains CPU and mode-jump Rust execution requires explicit opt-in.
- `hardware_claim_count`: Bare integer `0` prevents this software A/B from becoming a hardware claim.
- `sampler_runtime_ready_score`: Summarizes activation, support, parity, equivalence, default-off, protected-file, and command gates.
- `protected_files_unchanged`: Confirms conductor and reconciler-owned files stayed unchanged.
- `preconditions_checked`: Records that preregistration and protected hashes were captured before sampler chains ran.
- `inference_substrate`: Declares local CPU software mode-jump sampling, not hardware, GPU, TSU, or LLM inference.
- `verifier_is_oracle`: States that exact finite enumeration and transition receipts are the verifier.
- `field_provenance`: Maps each required field to prompt, spec, local source, upstream artifact, command receipt, or computed chain evidence.
- `field_principles`: Explains why each required field exists before a reviewer trusts the artifact.
- `test_commands`: Records focused tests, coverage, experiment, E2E, adversarial, and separately run suite receipts.
- `test_exit_codes`: Stores exit codes so failed commands cannot become readiness evidence.
- `duration_s`: Reports real wall time without padding.
- `reproducibility_checksum`: Content-addresses the artifact after blanking volatile duration and the checksum field.
- `honest_verdict`: Uses a terminal prefix and states activation, equivalence, unsupported controls, commands, default-off, and no hardware claim.

### SCENARIO-SAMPLER-6237-ACTIVATED-EQUIVALENCE: Active Mode Jumps Gate Quality

**Given** the frozen Exp6166 multimodal categorical target, Exp6208 runtime
adapter, preregistered seeds, and matched budgets
**When** Exp6237 runs the seeded fallback and enabled mode-jump arms
**Then** every main fixture arm is supported
**And** the mode-jump arm records nonzero cross-mode proposals and acceptances
**And** the multimodal positive control passes before quality conclusions
**And** paired quality intervals are compared against preregistered
equivalence bounds.

### SCENARIO-SAMPLER-6237-CONTROLS-FAIL-CLOSED: Controls Stay Out Of Main Claims

**Given** unsupported fixture, zero-activation, degenerate-chain,
interruption, restart, Rust/Python parity, and default-off fallback controls
**When** Exp6237 builds the artifact
**Then** unsupported or failed controls are recorded in
`unsupported_or_failed_cells`
**And** default-off and fallback parity stay true
**And** inactive treatment evidence returns `instrument_failure`, never a null
sampler verdict.

## Implementation Status (REQ-SAMPLER-6237)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLER-6237 | Planned (`python/carnot/experiment_6237_activated_mode_jump_sampler_ab.py`, `results/experiment_6237_activated_mode_jump_sampler_ab.json`) | Planned (`tests/python/test_experiment_6237_activated_mode_jump_sampler_ab.py`) |

### REQ-SAMPLER-6268: Frozen Bounded Exact Sampler Fixture Suite

Carnot MUST provide Exp6268 at
`python/carnot/experiment_6268_multimodal_sampler_fixture_suite.py` and write
`results/experiment_6268_multimodal_sampler_fixture_suite.json` without
modifying `scripts/research_conductor.py`. The experiment SHALL freeze a
bounded exact sampler fixture suite before any later A/B run. The suite SHALL
cover binary Ising, multistate Potts, and Carnot typed-factor targets. It SHALL
not run a headline sampler comparison. It SHALL not make timing, CUDA, cDLS,
FPGA, TSU, power, or speedup claims.

Sub-requirements:
- REQ-SAMPLER-6268-PRECONDITIONS: Exp6268 SHALL freeze fixture families, size
  bounds, exact tolerance, seeds, source hashes, protected files, and artifact
  paths before exact enumeration is interpreted.
- REQ-SAMPLER-6268-FAMILIES: The preregistered suite SHALL include the original
  six-state Exp6237 treatment-positive control, at least two multimodal Ising
  targets, at least two Potts targets, and at least two Carnot typed-factor
  targets.
- REQ-SAMPLER-6268-EXACT: Every bounded state space SHALL be enumerated by code
  independent of the sampler under test. The artifact SHALL store normalized
  probabilities, energies, modes, basins, barriers, hashes, and normalization
  errors for every fixture.
- REQ-SAMPLER-6268-CONTROLS: The suite SHALL include unimodal,
  inactive-treatment, and unsupported-shape controls. Valid controls SHALL be
  required before readiness can equal one.
- REQ-SAMPLER-6268-NEGATIVE-TESTS: Focused tests SHALL cover normalization,
  label permutation, Potts cardinality, typed-factor arity, duplicate fixtures,
  and energy-sign errors.
- REQ-SAMPLER-6268-READY-GATE: `sampler_fixture_suite_ready_score` SHALL equal
  exactly `1.0` only when every preregistered family is present, exact
  normalization errors are within tolerance, fixture hashes are unique, controls
  are valid, protected files are unchanged, and source mutation count is zero.

The terminal artifact SHALL include these top-level fields:
`status`, `fixture_manifest_path_and_hash`, `fixture_family_counts`,
`source_paths_and_hashes`, `state_space_sizes`,
`exact_enumeration_receipts`, `normalized_target_probability_hashes`,
`energy_and_factor_definitions`, `basin_labels_and_barrier_metadata`,
`mode_jump_support_by_fixture`, `original_six_state_positive_control`,
`unimodal_control`, `inactive_treatment_control`,
`unsupported_shape_control`, `random_seeds_and_schedule_defaults`,
`duplicate_fixture_count`,
`exact_probability_normalization_error_by_fixture`, `source_mutation_count`,
`sampler_fixture_suite_ready_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`. `duplicate_fixture_count` and `source_mutation_count` SHALL
be the bare integer `0`. `inference_substrate` SHALL equal
`local_cpu_exact_fixture_construction`.

Required field principles:

- `status`: Separates ready exact-suite evidence from blocked fixture, control, and command states.
- `fixture_manifest_path_and_hash`: Pins the generated fixture manifest so later sampler runs consume the same suite.
- `fixture_family_counts`: Proves the preregistered binary Ising, Potts, typed-factor, and control families are all present.
- `source_paths_and_hashes`: Pins source, spec, test, upstream artifact, sampler, and protected paths before the JSON is trusted.
- `state_space_sizes`: Makes every bounded enumeration size explicit.
- `exact_enumeration_receipts`: Stores independently enumerated probabilities, energies, modes, basins, and normalization receipts.
- `normalized_target_probability_hashes`: Content-addresses normalized target probabilities per fixture.
- `energy_and_factor_definitions`: Records the Ising Hamiltonians, Potts couplings, and typed factor kernels used to enumerate targets.
- `basin_labels_and_barrier_metadata`: Records basin assignment and minimum barrier evidence instead of inferring modality from names.
- `mode_jump_support_by_fixture`: Separates the six-state supported mode-jump target from unsupported suite fixtures and inactive controls.
- `original_six_state_positive_control`: Proves the Exp6237 treatment-positive fixture is reproduced in the suite.
- `unimodal_control`: Proves a valid exact target can be unimodal and therefore cannot support a multimodal treatment claim.
- `inactive_treatment_control`: Proves inactive treatment evidence is classified as an instrument failure, not a null result.
- `unsupported_shape_control`: Proves unsupported fixture shapes fail closed at the mode-jump boundary.
- `random_seeds_and_schedule_defaults`: Freezes replay seeds and default construction schedules without making timing claims.
- `duplicate_fixture_count`: Bare zero proves the suite did not register duplicate exact targets.
- `exact_probability_normalization_error_by_fixture`: Makes normalization tolerance evidence mechanical for each fixture.
- `source_mutation_count`: Bare zero proves protected and preregistered source hashes did not change during construction.
- `sampler_fixture_suite_ready_score`: Equals one only when family, exactness, control, duplicate, source, and protection gates pass.
- `protected_files_unchanged`: Confirms conductor and reconciler-owned files stayed byte-identical.
- `preconditions_checked`: Records frozen families, bounds, tolerances, seeds, hashes, and protected files before enumeration.
- `inference_substrate`: Declares local CPU exact fixture construction, not LLM inference, CUDA, FPGA, TSU, cDLS, or timing.
- `verifier_is_oracle`: States that exact finite enumeration is the oracle for this suite.
- `field_provenance`: Maps every required field to prompt, spec, source, fixture manifest, command receipts, or computed exact evidence.
- `field_principles`: Explains why each required field exists before a reviewer trusts the artifact.
- `test_commands`: Records focused Python, coverage, Rust, E2E, artifact, adversarial, and suite command receipts.
- `test_exit_codes`: Stores exit codes so failed checks stay visible and cannot be reported as passing.
- `duration_s`: Reports real wall time without padding or timing interpretation.
- `reproducibility_checksum`: Content-addresses the artifact after blanking volatile duration and the checksum field.
- `honest_verdict`: Uses a terminal prefix and states fixture readiness, exactness, controls, and no performance claim.

### SCENARIO-SAMPLER-6268-EXACT-SUITE: Fixture Families Are Frozen And Enumerated

**Given** the preregistered Exp6268 fixture family manifest, size bounds,
tolerance, and seeds
**When** Exp6268 builds the suite
**Then** it enumerates the original six-state control, at least two
multimodal Ising targets, at least two Potts targets, and at least two
typed-factor targets independent of sampler execution
**And** every target has normalized probabilities, energies, modes, basins,
barriers, hashes, and state-space size receipts.

### SCENARIO-SAMPLER-6268-CONTROLS-FAIL-CLOSED: Controls Gate Readiness

**Given** unimodal, inactive-treatment, unsupported-shape, duplicate,
label-permutation, Potts-cardinality, typed-arity, and energy-sign controls
**When** Exp6268 validates the fixture suite
**Then** valid controls are recorded, invalid mutations are rejected, duplicate
fixture count is the bare integer `0`, and inactive treatment cannot produce a
null sampler verdict.

### SCENARIO-SAMPLER-6268-NO-PERFORMANCE-CLAIM: Exact Construction Stays Claim-Bounded

**Given** the frozen exact fixture suite
**When** Exp6268 writes the terminal artifact
**Then** `inference_substrate` is `local_cpu_exact_fixture_construction`,
source mutation count is the bare integer `0`, no timing or hardware claim is
made, and `sampler_fixture_suite_ready_score` is `1.0` only when every
preregistered family, exactness, and control gate passes.

## Implementation Status (REQ-SAMPLER-6268)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLER-6268 | Planned (`python/carnot/experiment_6268_multimodal_sampler_fixture_suite.py`, `results/experiment_6268_multimodal_sampler_fixture_suite.json`) | Planned (`tests/python/test_experiment_6268_multimodal_sampler_fixture_suite.py`) |

### REQ-SAMPLER-6269: Mode-Jump Multifamily A/B

Carnot MUST provide Exp6269 at
`python/carnot/experiment_6269_mode_jump_multifamily_ab.py` and write
`results/experiment_6269_mode_jump_multifamily_ab.json` without modifying
`scripts/research_conductor.py`. The experiment SHALL run a matched
fallback-versus-mode-jump A/B across every Exp6268 fixture supported by the
existing Rust/PyO3 mode-jump backend. Unsupported Exp6268 fixtures SHALL be
preserved as unsupported cells. They SHALL NOT be replaced with fallback
output. No cDLS, hardware, timing speedup, power, or efficiency claim is in
scope.

Sub-requirements:
- REQ-SAMPLER-6269-PRECONDITIONS: Exp6269 SHALL validate the Exp6268 exact
  suite, freeze the fixture/seed/arm matrix, fixed budgets, equivalence
  margins, value gate, protected hashes, and upstream fixture hash before
  sampler outcomes are compared.
- REQ-SAMPLER-6269-MATCHED-CELLS: Each supported fixture/seed/arm cell SHALL
  share the same target, initial state, seed, burn-in, retained sample count,
  proposal budget, and schedule. Each successful cell SHALL record an
  independent chain receipt and sample hash.
- REQ-SAMPLER-6269-ACTIVATION: Treatment activation SHALL be proven before
  outcome comparison. The positive control SHALL require Rust/PyO3 selection,
  nonzero treatment attempts, nonzero treatment accepts, and nonzero treatment
  fires. Inactive treatment evidence SHALL produce an instrument failure.
- REQ-SAMPLER-6269-EXACTNESS: For each supported fixture and arm, Exp6269
  SHALL compare empirical samples with the exact Exp6268 target. It SHALL
  report exact distribution error, energy error, basin occupancy, barrier
  crossings, autocorrelation, ESS, and acceptance before pooled summaries.
- REQ-SAMPLER-6269-INTERVALS: Exp6269 SHALL compute paired intervals with
  sample sizes and preregistered equivalence margins. It SHALL separate
  distribution safety, mixing value, and descriptive wall time.
- REQ-SAMPLER-6269-VALUE-GATE: Workload value SHALL require distribution
  safety plus a preregistered positive mixing improvement on at least two
  non-toy fixture families with no exactness regression. Otherwise
  `mode_jump_workload_value_ready_score` SHALL be `0.0`.
- REQ-SAMPLER-6269-NEGATIVE-TESTS: Focused tests SHALL cover inactive
  treatment, unsupported shape, seed mismatch, sample-count mismatch,
  acceptance accounting, and exactness regression.

The terminal artifact SHALL include these top-level fields:
`status`, `upstream_fixture_path_and_hash`,
`preregistered_fixture_seed_arm_matrix`, `matched_arm_configuration`,
`rust_pyo3_backend_receipts`,
`treatment_attempt_accept_and_fire_counts_by_fixture`,
`positive_and_inactive_control_results`, `chain_sample_hashes`,
`exact_distribution_error_by_arm_fixture`,
`energy_error_by_arm_fixture`,
`basin_occupancy_and_barrier_crossings_by_arm_fixture`,
`autocorrelation_ess_and_acceptance_by_arm_fixture`,
`paired_intervals_equivalence_margins_and_sample_sizes`,
`harmful_regressions`, `descriptive_wall_time_by_arm_fixture`,
`unsupported_or_failed_cells`, `source_mutation_count`,
`hardware_claim_count`, `timing_speedup_claimed`,
`mode_jump_safety_ready_score`, `mode_jump_workload_value_ready_score`,
`protected_files_unchanged`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.
`source_mutation_count` and `hardware_claim_count` SHALL be the bare integer
`0`. `timing_speedup_claimed` SHALL be the bare boolean `false`.
`inference_substrate` SHALL equal
`local_cpu_exact_multifamily_mode_jump_ab`.

Required field principles:

- `status`: Separates safety evidence, workload-value evidence, blockers, and instrument failures.
- `upstream_fixture_path_and_hash`: Pins the Exp6268 exact suite consumed by the A/B.
- `preregistered_fixture_seed_arm_matrix`: Freezes fixtures, seeds, arms, budgets, margins, and the value gate before sampling.
- `matched_arm_configuration`: Proves each compared arm uses the same target, seed, initial state, burn-in, retained sample count, proposal budget, and schedule.
- `rust_pyo3_backend_receipts`: Authenticates the backend, descriptor, input hash, final state, and transition budget for each chain.
- `treatment_attempt_accept_and_fire_counts_by_fixture`: Proves treatment activity before outcome comparison.
- `positive_and_inactive_control_results`: Records both the activation-positive control and the inactive-treatment fail-closed control.
- `chain_sample_hashes`: Content-addresses retained samples without making the artifact depend on raw sample arrays.
- `exact_distribution_error_by_arm_fixture`: Reports empirical-versus-exact distribution error per fixture and arm.
- `energy_error_by_arm_fixture`: Reports empirical-versus-exact energy error per fixture and arm.
- `basin_occupancy_and_barrier_crossings_by_arm_fixture`: Reports basin mass and cross-basin transitions per fixture and arm.
- `autocorrelation_ess_and_acceptance_by_arm_fixture`: Reports autocorrelation, ESS, and acceptance per fixture and arm.
- `paired_intervals_equivalence_margins_and_sample_sizes`: Stores paired intervals, equivalence margins, and n before any value decision.
- `harmful_regressions`: Lists exactness or mixing regressions that block safety.
- `descriptive_wall_time_by_arm_fixture`: Records wall time as cost evidence only.
- `unsupported_or_failed_cells`: Preserves unsupported or failed Exp6268 cells without fallback substitution.
- `source_mutation_count`: Bare zero proves this experiment did not mutate preregistered source during compute.
- `hardware_claim_count`: Bare zero prevents a software sampler A/B from becoming a hardware claim.
- `timing_speedup_claimed`: Bare false prevents descriptive wall time from becoming a speedup claim.
- `mode_jump_safety_ready_score`: Equals one only when exactness, activation, controls, protected files, source, and command gates pass.
- `mode_jump_workload_value_ready_score`: Equals one only when safety and the non-toy positive mixing value gate both pass.
- `protected_files_unchanged`: Confirms conductor-owned and reconciler-owned files stayed byte-identical.
- `preconditions_checked`: Records exact-suite validation, frozen budgets, frozen margins, value gate, and protected hashes.
- `inference_substrate`: Declares local CPU exact multifamily mode-jump sampling, not hardware, cDLS, or LLM inference.
- `verifier_is_oracle`: States that Exp6268 exact finite distributions are the oracle.
- `field_provenance`: Maps every required field to prompt, spec, source, upstream fixture, command, or chain evidence.
- `field_principles`: Explains why each artifact field exists before a reviewer trusts the JSON shape.
- `test_commands`: Records focused Python, coverage, Rust, E2E, artifact, adversarial, and suite command receipts.
- `test_exit_codes`: Stores exit codes so failed checks cannot become readiness evidence.
- `duration_s`: Reports real wall time without padding.
- `reproducibility_checksum`: Content-addresses the artifact after blanking volatile duration and checksum fields.
- `honest_verdict`: Uses a terminal prefix and states safety, workload value, unsupported fixtures, and no hardware or speedup claim.

### SCENARIO-SAMPLER-6269-MATCHED-SUPPORTED-CELLS: Supported Fixtures Get Matched Chain Receipts

**Given** the frozen Exp6268 exact fixture suite, fixed seeds, and fixed
budgets
**When** Exp6269 runs
**Then** each Exp6268 fixture supported by the existing Rust/PyO3 backend has
matched fallback and mode-jump cells
**And** each successful chain records backend receipts, treatment counts,
sample hashes, distribution error, energy error, basin occupancy, barrier
crossings, autocorrelation, ESS, acceptance, and descriptive wall time.

### SCENARIO-SAMPLER-6269-UNSUPPORTED-CELLS-FAIL-CLOSED: Unsupported Fixtures Are Preserved

**Given** Exp6268 Ising, Potts, typed-factor, and control fixtures that the
fixed mode-jump backend does not support
**When** Exp6269 builds the A/B artifact
**Then** unsupported fixtures are recorded in `unsupported_or_failed_cells`
**And** no unsupported fixture contributes fallback samples, pooled quality,
or workload-value evidence.

### SCENARIO-SAMPLER-6269-SAFETY-VALUE-SEPARATION: Safety Does Not Imply Workload Value

**Given** paired exactness intervals, mixing intervals, activation controls,
and the preregistered value gate
**When** Exp6269 computes readiness scores
**Then** `mode_jump_safety_ready_score` can pass only with no exactness
regression
**And** `mode_jump_workload_value_ready_score` can pass only with positive
mixing improvement on at least two non-toy fixture families.

## Implementation Status (REQ-SAMPLER-6269)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLER-6269 | Planned (`python/carnot/experiment_6269_mode_jump_multifamily_ab.py`, `results/experiment_6269_mode_jump_multifamily_ab.json`) | Planned (`tests/python/test_experiment_6269_mode_jump_multifamily_ab.py`) |

### REQ-SAMPLER-6280: Variable-Cardinality Mode-Jump Backend ABI

Carnot MUST replace the fixed six-state-only mode-jump runtime contract with an
explicit typed state metadata contract while preserving the original six-state
behavior. The backend SHALL support the Exp6268 binary Ising, multistate
Potts, original six-state categorical, typed multimodal factor, and bounded
typed access-control fixtures through one metadata schema. The work SHALL NOT
run the Exp6269 scientific A/B and SHALL NOT make sampler-value, timing, CUDA,
cDLS, FPGA, TSU, power, or speedup claims.

Sub-requirements:
- REQ-SAMPLER-6280-SPEC-FIRST: The metadata schema SHALL declare rank-1 state
  shape, per-variable cardinality, encoding, labels or categories, valid
  proposal domain, and round-trip rules before implementation evidence is
  interpreted.
- REQ-SAMPLER-6280-METADATA: The Python adapter, Rust core, and PyO3 binding
  SHALL reject ambiguous, inconsistent, duplicate-label, out-of-domain,
  mismatched-cardinality, unsupported-rank, or non-square proposal metadata
  before sampling.
- REQ-SAMPLER-6280-COMPATIBILITY: The original Exp6194/Exp6208 six-label call
  SHALL still run through the compatibility path with the same samples, decision
  log, checkpoint state, and treatment counts for fixed seeds.
- REQ-SAMPLER-6280-PARITY: Rust/PyO3 and exact Python fallback SHALL match
  encode-decode round trips, proposal probabilities, one-step decisions,
  retained samples, decision logs, checkpoint states, and deterministic replay
  for every supported Exp6268 fixture.
- REQ-SAMPLER-6280-ACTIVATION: Each positive-control fixture SHALL record
  nonzero treatment attempts, accepted jumps, and fires before any backend
  readiness score can equal `1.0`.
- REQ-SAMPLER-6280-CONTROLS: Malformed cardinality, malformed shape,
  out-of-domain proposal, and label-permutation controls SHALL fail closed.
  Unsupported non-rank-1 shapes SHALL remain explicit errors and SHALL NOT be
  replaced with fallback samples.
- REQ-SAMPLER-6280-ARTIFACT: Exp6280 SHALL write
  `results/experiment_6280_variable_cardinality_mode_jump_backend.json` after
  focused Rust/Python parity tests, deterministic replay checks, activation
  checks, fail-closed controls, and adversarial verification of the artifact.

The terminal artifact SHALL include these top-level fields: `status`,
`exp6268_fixture_path_hash_and_terminal_class`,
`exp6269_failure_path_hash_and_root_cause`,
`rust_and_python_source_paths_hashes_before_after`,
`typed_state_metadata_schema`, `supported_fixture_families_and_shapes`,
`unsupported_shapes_and_fail_closed_behavior`,
`original_six_state_regression_receipt`,
`rust_python_encode_decode_roundtrip_by_fixture`,
`rust_python_proposal_parity_by_fixture`,
`deterministic_seed_replay_by_fixture`,
`treatment_attempt_accept_and_fire_counts_by_fixture`,
`malformed_cardinality_controls`, `malformed_shape_controls`,
`out_of_domain_proposal_controls`, `label_permutation_controls`,
`focused_python_test_results`, `focused_rust_test_results`,
`source_mutation_count`, `variable_cardinality_backend_ready_score`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`, `random_seed`,
`reproducibility_checksum`, and `honest_verdict`.
`source_mutation_count` SHALL be the bare integer `0`.
`variable_cardinality_backend_ready_score` SHALL equal exactly `1.0` only when
all preregistered fixture families are supported, Rust/Python round trips and
proposals are exact, deterministic seed replay holds, treatment fires on
positive controls, malformed controls fail closed, protected files are
unchanged, and focused command exits are clean. `inference_substrate` SHALL
equal `local_cpu_rust_python_variable_cardinality_sampler_abi`.

Required field principles:

- `status`: Separates ready ABI evidence from blocked metadata, parity, or control states.
- `exp6268_fixture_path_hash_and_terminal_class`: Pins the exact fixture suite and its terminal class before support claims are trusted.
- `exp6269_failure_path_hash_and_root_cause`: Pins the prior failure and names the fixed ABI root cause instead of rerunning the A/B.
- `rust_and_python_source_paths_hashes_before_after`: Shows which backend and binding files changed, by hash.
- `typed_state_metadata_schema`: Defines shape, cardinality, encoding, proposal domain, and round-trip rules in one place.
- `supported_fixture_families_and_shapes`: Lists the Exp6268 families and rank-1 shapes accepted by the new ABI.
- `unsupported_shapes_and_fail_closed_behavior`: Records unsupported ranks or malformed surfaces as errors, not fallback samples.
- `original_six_state_regression_receipt`: Proves the fixed six-state compatibility path still replays.
- `rust_python_encode_decode_roundtrip_by_fixture`: Proves both languages map state labels to indices and back exactly.
- `rust_python_proposal_parity_by_fixture`: Proves both languages read the same proposal domain and probability table.
- `deterministic_seed_replay_by_fixture`: Proves fixed seeds replay identical traces and checkpoints.
- `treatment_attempt_accept_and_fire_counts_by_fixture`: Proves treatment activation before readiness.
- `malformed_cardinality_controls`: Shows bad cardinality metadata fails closed.
- `malformed_shape_controls`: Shows rank and shape mismatches fail closed.
- `out_of_domain_proposal_controls`: Shows proposal labels or indices outside the metadata domain fail closed.
- `label_permutation_controls`: Shows label order drift is detected instead of silently relabeling states.
- `focused_python_test_results`: Records focused Python test and coverage results for the changed code.
- `focused_rust_test_results`: Records focused Rust test results for the changed core.
- `source_mutation_count`: Bare zero proves no protected or preregistered source drift occurred during verification.
- `variable_cardinality_backend_ready_score`: Equals one only when support, parity, replay, activation, controls, and commands all pass.
- `protected_files_unchanged`: Confirms conductor-owned and reconciler-owned files stayed byte-identical.
- `preconditions_checked`: Records git status, source hashes, fixture hashes, tool versions, ABI hashes, supported shapes, seed, and protected hashes before edits.
- `inference_substrate`: Declares local CPU Rust/Python variable-cardinality ABI verification, not timing, LLM, GPU, cDLS, or hardware work.
- `verifier_is_oracle`: States that exact finite metadata, proposal tables, and deterministic replay are the verifier.
- `field_provenance`: Maps every required field to prompt, spec, source, upstream artifact, command, or computed ABI evidence.
- `field_principles`: Explains why each required field exists before a reviewer trusts the JSON shape.
- `test_commands`: Lists focused Python, coverage, Rust, E2E, experiment, adversarial, and suite commands.
- `test_exit_codes`: Stores command exit codes so failed checks cannot become readiness evidence.
- `duration_s`: Reports real wall time without timing interpretation.
- `random_seed`: Records the replay seed used for deterministic controls.
- `reproducibility_checksum`: Content-addresses the artifact after blanking volatile duration and checksum fields.
- `honest_verdict`: Uses a terminal prefix and states ABI readiness without claiming sampler value.

### SCENARIO-SAMPLER-6280-METADATA-ROUNDTRIP: Typed Metadata Encodes Every Fixture

**Given** the frozen Exp6268 exact fixture suite and the typed state metadata
schema
**When** the Python adapter and Rust/PyO3 core encode and decode each supported
fixture state
**Then** binary Ising, q-state Potts, original categorical, and typed-factor
states round-trip exactly
**And** malformed cardinality, malformed shape, duplicate labels, label
permutations, and out-of-domain proposal controls raise explicit errors.

### SCENARIO-SAMPLER-6280-PROPOSAL-PARITY: Rust And Python Proposals Match

**Given** typed metadata, target probabilities, proposal tables, fixed seeds,
and initial states for every supported Exp6268 fixture
**When** the Rust/PyO3 path and exact Python fallback each advance chains
**Then** proposal probabilities, decision logs, retained sample labels,
checkpoints, accepted counts, attempted counts, and treatment fire counts match
exactly for every supported fixture.

### SCENARIO-SAMPLER-6280-NO-AB-VALUE-CLAIM: ABI Readiness Does Not Rerun A/B

**Given** Exp6269 failed before a scientific comparison because only shape
`(6,)` was accepted
**When** Exp6280 writes its terminal artifact
**Then** it records the fixed root cause, declares the variable-cardinality ABI
ready only under the parity and fail-closed gates, and does not run or claim
the scientific A/B value result.

## Implementation Status (REQ-SAMPLER-6280)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLER-6280 | Planned (`python/carnot/samplers/mode_jump_rust_backend.py`, `crates/carnot-samplers/src/mode_jump.rs`, `crates/carnot-python/src/mode_jump.rs`, `python/carnot/experiment_6280_variable_cardinality_mode_jump_backend.py`, `results/experiment_6280_variable_cardinality_mode_jump_backend.json`) | Planned (`tests/python/samplers/test_mode_jump_rust_backend.py`, `tests/python/test_experiment_6280_variable_cardinality_mode_jump_backend.py`, `crates/carnot-samplers/tests/mode_jump.rs`) |

### REQ-SAMPLER-6281: Typed Mode-Jump Multifamily A/B Rerun

Carnot MUST provide Exp6281 at
`python/carnot/experiment_6281_mode_jump_multifamily_rerun.py` and write
`results/experiment_6281_mode_jump_multifamily_rerun.json` without modifying
`scripts/research_conductor.py`. The experiment SHALL rerun the matched
fallback-versus-mode-jump A/B on the frozen Exp6268 fixture suite only after
the Exp6280 variable-cardinality backend is ready. The rerun SHALL hold target
hashes, initial states, seeds, burn-in, retained samples, proposal budgets, and
schedules fixed between arms. It SHALL prove treatment activation before any
outcome comparison. No cDLS, hardware, speedup, power, or efficiency claim is
in scope.

Sub-requirements:
- REQ-SAMPLER-6281-PRECONDITIONS: Exp6281 SHALL validate the Exp6280 backend
  readiness artifact, the exact Exp6268 fixture hash, the Exp6269 failure root
  cause, the fixture/seed/arm matrix, budgets, margins, protected hashes, and
  source hashes before sampler outcomes are compared.
- REQ-SAMPLER-6281-MATCHED-CELLS: Every Exp6268 fixture supported by the
  Exp6280 typed backend SHALL run matched seeded-fallback and Rust/PyO3
  mode-jump arms for each preregistered seed. Failed cells SHALL be preserved
  as failed cells and SHALL NOT be replaced with fallback output.
- REQ-SAMPLER-6281-ACTIVATION: Treatment activation SHALL be proven before
  outcome comparison. Positive controls SHALL require Rust/PyO3 selection,
  nonzero treatment attempts, nonzero accepts, and nonzero fires. Inactive and
  unimodal controls SHALL stay separate from value evidence.
- REQ-SAMPLER-6281-EXACTNESS: For each fixture and arm, Exp6281 SHALL compare
  empirical samples with the exact Exp6268 target and report distribution
  error, energy error, basin occupancy, barrier crossings, autocorrelation,
  ESS, and acceptance before pooling.
- REQ-SAMPLER-6281-INTERVALS: Exp6281 SHALL compute paired and
  equivalence-aware intervals with sample sizes. It SHALL separate distribution
  safety, family-level mixing value, and descriptive wall time.
- REQ-SAMPLER-6281-RETIREMENT: If full typed support still yields a blocked or
  null workload-value conclusion, Exp6281 SHALL recommend permanent retirement
  of this mode-jump lane.
- REQ-SAMPLER-6281-NEGATIVE-TESTS: Focused tests SHALL cover backend
  preconditions, treatment inactivity, matched seed mismatch, sample-count
  mismatch, acceptance accounting, exactness regression, forbidden claims, and
  retirement recommendation logic.

The terminal artifact SHALL include these top-level fields:
`status`, `upstream_backend_path_hash_and_terminal_class`,
`frozen_fixture_path_and_hash`, `preregistered_fixture_seed_arm_matrix`,
`matched_arm_configuration`, `rust_pyo3_backend_receipts`,
`treatment_attempt_accept_and_fire_counts_by_fixture`,
`positive_inactive_and_unimodal_control_results`, `chain_sample_hashes`,
`exact_distribution_error_by_arm_fixture`, `energy_error_by_arm_fixture`,
`basin_occupancy_and_barrier_crossings_by_arm_fixture`,
`autocorrelation_ess_and_acceptance_by_arm_fixture`,
`paired_intervals_equivalence_margins_and_sample_sizes`,
`family_level_safety_results`, `family_level_mixing_value_results`,
`harmful_regressions`, `descriptive_wall_time_by_arm_fixture`,
`unsupported_or_failed_cells`, `source_mutation_count`,
`hardware_claim_count`, `timing_speedup_claimed`,
`mode_jump_safety_ready_score`, `mode_jump_workload_value_ready_score`,
`retire_mechanism_recommendation`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`, `test_exit_codes`,
`duration_s`, `random_seed`, `reproducibility_checksum`, and
`honest_verdict`. `source_mutation_count` and `hardware_claim_count` SHALL be
the bare integer `0`. `timing_speedup_claimed` SHALL be the bare boolean
`false`. `inference_substrate` SHALL equal
`local_cpu_exact_typed_multifamily_mode_jump_ab`.

Required field principles:

- `status`: Separates safety, value, blocker, and retirement states.
- `upstream_backend_path_hash_and_terminal_class`: Pins Exp6280 before the rerun trusts typed backend support.
- `frozen_fixture_path_and_hash`: Pins the Exp6268 exact suite consumed by the rerun.
- `preregistered_fixture_seed_arm_matrix`: Freezes fixtures, seeds, arms, budgets, margins, and gates before sampling.
- `matched_arm_configuration`: Proves compared arms share targets, seeds, initial states, burn-in, samples, budgets, and schedules.
- `rust_pyo3_backend_receipts`: Authenticates backend selection, descriptors, inputs, states, and transition budgets per chain.
- `treatment_attempt_accept_and_fire_counts_by_fixture`: Proves treatment activity before outcome comparison.
- `positive_inactive_and_unimodal_control_results`: Keeps activation-positive, inactive-treatment, and unimodal controls separate.
- `chain_sample_hashes`: Content-addresses retained chains without storing raw arrays as gate inputs.
- `exact_distribution_error_by_arm_fixture`: Reports empirical-versus-exact distribution error per fixture and arm.
- `energy_error_by_arm_fixture`: Reports empirical-versus-exact energy error per fixture and arm.
- `basin_occupancy_and_barrier_crossings_by_arm_fixture`: Reports basin mass and cross-basin movement per fixture and arm.
- `autocorrelation_ess_and_acceptance_by_arm_fixture`: Reports autocorrelation, ESS, and acceptance per fixture and arm.
- `paired_intervals_equivalence_margins_and_sample_sizes`: Stores paired intervals, margins, and n before value decisions.
- `family_level_safety_results`: Reports distribution safety by family before any pooled conclusion.
- `family_level_mixing_value_results`: Reports family-level mixing value without converting cost into speedup.
- `harmful_regressions`: Lists exactness or mixing regressions that block safety.
- `descriptive_wall_time_by_arm_fixture`: Records wall time as descriptive cost only.
- `unsupported_or_failed_cells`: Preserves failed or unsupported cells without fallback substitution.
- `source_mutation_count`: Bare zero proves no preregistered source drift during compute.
- `hardware_claim_count`: Bare zero prevents software evidence from becoming hardware evidence.
- `timing_speedup_claimed`: Bare false prevents descriptive time from becoming a speedup claim.
- `mode_jump_safety_ready_score`: Equals one only when exactness, activation, controls, commands, and protection pass.
- `mode_jump_workload_value_ready_score`: Equals one only when safety and family-level mixing value pass.
- `retire_mechanism_recommendation`: States whether this lane should be permanently retired after the rerun.
- `protected_files_unchanged`: Confirms conductor and ops-owned files stayed byte-identical.
- `preconditions_checked`: Records backend readiness, fixture hash, budgets, margins, seeds, and protected hashes.
- `inference_substrate`: Declares local CPU exact typed mode-jump sampling, not hardware, cDLS, LLM, or speedup work.
- `verifier_is_oracle`: States that Exp6268 exact finite distributions are the oracle.
- `field_provenance`: Maps every field to prompt, spec, source, upstream, command, or chain evidence.
- `field_principles`: Explains why each artifact field exists before a reviewer trusts the JSON shape.
- `test_commands`: Records focused Python, coverage, Rust, E2E, full suite, experiment, and adversarial commands.
- `test_exit_codes`: Stores command exit codes so failed checks cannot become readiness evidence.
- `duration_s`: Reports real wall time without padding.
- `random_seed`: Records the root seed for deterministic matrix construction.
- `reproducibility_checksum`: Content-addresses the artifact after blanking volatile duration and checksum fields.
- `honest_verdict`: Uses a terminal prefix and states safety, value, retirement, and forbidden-claim counts.

### SCENARIO-SAMPLER-6281-TYPED-MATCHED-CELLS: Typed Fixtures Get Matched Chain Receipts

**Given** Exp6280 is complete-ready and the Exp6268 suite hash matches the
frozen precondition
**When** Exp6281 runs
**Then** each Exp6268 typed-backend-supported fixture has matched fallback and
mode-jump cells
**And** each successful chain records backend receipts, treatment counts,
sample hashes, distribution error, energy error, basin occupancy, barrier
crossings, autocorrelation, ESS, acceptance, and descriptive wall time.

### SCENARIO-SAMPLER-6281-CONTROLS-SEPARATE-VALUE: Controls Gate Interpretation

**Given** activation-positive, inactive-treatment, and unimodal controls
**When** Exp6281 computes treatment evidence
**Then** treatment activation is proven before outcome comparison
**And** inactive or unimodal evidence cannot by itself support a workload-value
claim.

### SCENARIO-SAMPLER-6281-RETIREMENT-DECISION: Null Or Blocked Value Retires The Lane

**Given** paired intervals, family-level safety, family-level mixing value,
and forbidden-claim counts
**When** Exp6281 computes terminal fields
**Then** safety and workload value remain separate
**And** a blocked or null workload-value conclusion recommends permanent
retirement of this mode-jump lane.

## Implementation Status (REQ-SAMPLER-6281)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLER-6281 | Planned (`python/carnot/experiment_6281_mode_jump_multifamily_rerun.py`, `results/experiment_6281_mode_jump_multifamily_rerun.json`) | Planned (`tests/python/test_experiment_6281_mode_jump_multifamily_rerun.py`) |

### REQ-SAMPLER-6597: Spectral k-Block Ising Canary

Carnot SHALL provide a CPU-only finite-state canary at
`python/carnot/experiment_6597_spectral_k_block_ising_canary.py`. It SHALL
write `results/experiment_6597_spectral_k_block_ising_canary.json` atomically.

- REQ-SAMPLER-6597-PARTITION: The canary SHALL build each partition from the
  bottom nonconstant eigenspace of the squared random-site heat-bath operator.
  It SHALL use stationary-weighted rounding and exact objective rescoring.
  It SHALL record the eigensolver, ordering, tied-eigenvalue rule, blocks,
  setup time, and failures.
- REQ-SAMPLER-6597-MATCHED: Sequential single-spin Gibbs and each spectral
  `G_O P` arm SHALL share fixture, temperature, seed, initial state, random-site
  stream, burn-in, retained count, and kernel-transition budget where valid.
  The artifact SHALL keep component work and charged wall time separate.
- REQ-SAMPLER-6597-EXACT: A separate exact enumerator SHALL evaluate retained
  counts. Each row SHALL report total variation, mean error, covariance error,
  autocorrelation, ESS, ESS per transition, costs, transitions, failure, and
  peak memory. The sampler SHALL not receive the evaluator probability vector.
- REQ-SAMPLER-6597-FLOOR: The frozen matrix SHALL contain independent,
  ferromagnetic, and frustrated exact-enumerable Ising families. It SHALL use
  at least five seeds and at least 10,000 retained samples after explicit
  burn-in. It SHALL preregister all tested block counts before sampling.
- REQ-SAMPLER-6597-PAIRED: A positive verdict SHALL require stationary-error
  noninferiority and a positive paired ESS-per-transition or charged-time
  effect with a nonnegative lower bound. Family rows SHALL remain separate.
  A frustrated-family regression SHALL block a pooled positive verdict.
- REQ-SAMPLER-6597-ATTACKS: The canary SHALL fail closed on evaluator use in
  sampling, unmatched transitions, omitted setup, sample-floor failure, seed
  loss, favorable block selection, synchronous-update bias, PIMI or hardware
  rebranding, and aggregate-only evidence.
- REQ-SAMPLER-6597-BOUNDARY: The result SHALL apply only to charged CPU
  software fixtures. It SHALL make no FPGA, TSU, PIMI, or general hardware
  claim. `inference_substrate` SHALL equal
  `cpu_spectral_k_block_ising_exact_enumeration`.
  `verifier_is_oracle` SHALL be the bare boolean `false`.
- REQ-SAMPLER-6597-ATOMIC: The terminal artifact SHALL contain the required
  principle-annotated fields, per-seed rows, protected-file hashes, command
  receipts, field provenance, duration, and a reproducibility checksum.
  A blocked verdict SHALL name the failed method, fixture, sample, resource,
  or numerical check and its observed value in `gate_check_summary`.

### SCENARIO-SAMPLER-6597-PAPER-FAITHFUL-PARTITION

**Given** a finite reversible heat-bath operator and an exact-enumerable Ising
fixture
**When** the canary constructs a spectral partition
**Then** it rounds bottom modes of the squared operator
**And** it records deterministic handling for ordering and degeneracy
**And** it selects only from the preregistered block counts.

### SCENARIO-SAMPLER-6597-MATCHED-EXACT-EVIDENCE

**Given** the three frozen fixture families and five frozen seeds
**When** sequential Gibbs and all spectral arms run
**Then** every successful row retains at least 10,000 post-burn-in samples
**And** all arms use matched transition budgets and matched initial states
**And** the separate evaluator recomputes distribution and moment errors.

### SCENARIO-SAMPLER-6597-FAIL-CLOSED-VERDICT

**Given** per-seed distribution, mixing, cost, failure, and memory rows
**When** the paired reducer and adversarial checks run
**Then** a positive verdict requires both stationary and efficiency gates
**And** every family remains visible
**And** failed, missing, unmatched, under-sized, or rebranded evidence cannot
support a positive verdict.

## Implementation Status (REQ-SAMPLER-6597)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-SAMPLER-6597 | Implemented (`python/carnot/experiment_6597_spectral_k_block_ising_canary.py`, `results/experiment_6597_spectral_k_block_ising_canary.json`) | Implemented (`tests/python/test_experiment_6597_spectral_k_block_ising_canary.py`) |
