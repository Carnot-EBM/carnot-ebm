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
