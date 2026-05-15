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

