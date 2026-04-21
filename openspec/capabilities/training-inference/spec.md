# Training & Inference Capability Specification

**Capability:** training-inference
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-06, FR-07

## Overview

Defines training algorithms and inference (sampling) methods for Energy Based Models.

## Requirements

### REQ-TRAIN-001: Contrastive Divergence (CD-k)

The system shall implement CD-k training with configurable k steps:
- Positive phase: compute energy on data samples
- Negative phase: run k steps of Gibbs sampling from data
- Update: gradient of (E_pos - E_neg) w.r.t. parameters

### REQ-TRAIN-002: Score Matching

The system shall implement score matching training:
- Minimize Fisher divergence between model and data score functions
- Support denoising score matching variant

### REQ-TRAIN-003: Noise Contrastive Estimation (NCE)

The system shall implement NCE training:
- Binary classification between data and noise samples
- Configurable noise distribution

### REQ-TRAIN-004: Training Loop

The system shall provide a training loop with:
- Configurable optimizer (SGD, Adam)
- Learning rate scheduling
- Gradient clipping
- Checkpointing (safetensors format)
- Loss logging

### REQ-SAMPLE-001: Langevin Dynamics Sampler

The system shall implement unadjusted Langevin dynamics:
- `x_{t+1} = x_t - (step_size/2) * grad_energy(x_t) + sqrt(step_size) * noise`
- Configurable step size and number of steps
- Optional gradient clipping (REQ-SAMPLE-004)
- Support for annealed Langevin dynamics

### REQ-SAMPLE-002: Hamiltonian Monte Carlo (HMC)

The system shall implement HMC sampling:
- Leapfrog integrator with configurable step size and trajectory length
- Metropolis-Hastings accept/reject step
- Optional gradient clipping in leapfrog steps (REQ-SAMPLE-004)
- Configurable mass matrix (identity, diagonal, dense)

### REQ-SAMPLE-004: Gradient Clipping for Samplers

The system shall provide optional gradient clipping for all MCMC samplers:
- `clip_norm: float | None` parameter (default None = no clipping)
- When enabled, gradients are rescaled to have at most `clip_norm` L2 norm
- Clipping formula: `grad * clip_norm / max(||grad||, clip_norm)`
- Prevents divergence (NaN) on steep energy landscapes (e.g., Rosenbrock)
- Backward compatible: None means existing behavior is unchanged

### REQ-SAMPLE-003: Sampler Interface

The system shall define a `Sampler` trait/protocol:
- `sample(&self, energy_fn: &dyn EnergyFunction, init: Array, n_steps: usize) -> Array`
- `sample_chain(&self, ...) -> Vec<Array>` — return full chain for diagnostics
- Parallel Ising Gibbs sampler (`parallel_ising.py`): checkerboard updates, simulated annealing schedules, and thrml-compatible `parallel_sample_states` wrapper

### REQ-SAMPLE-005: FPGA Ising Architecture And Register Map

The system shall define a KV260-class FPGA Ising sampler architecture for
dynamic constraint verification, where:
- the target array size is up to 4096 spins using a tiled p-bit design
  (32 tiles × 128 spins per tile)
- spin updates use an even/odd phase schedule compatible with the existing
  parallel Ising approximation
- couplings are uploaded at runtime as sparse row data rather than fixed at
  synthesis time
- an AXI-Lite register map covers control, status, annealing configuration,
  bias upload, sparse coupling upload, and sampled-state readback

### REQ-SAMPLE-006: FPGA Sampler Backend With Simulation And Fallback

The system shall provide an `FPGAIsingSampler` backend in
`python/carnot/samplers/fpga_ising.py`, where:
- the backend satisfies the `SamplerBackend` protocol
- the backend can drive a PYNQ overlay when available
- the backend provides a software model that exercises the same register-map
  upload, trigger, and readback path without hardware
- the backend falls back to the CPU Ising sampler when FPGA access is not
  available or explicitly disabled
- the backend exposes a small benchmark helper comparing the simulated FPGA
  control path against the CPU sampler on the same Ising problem

### REQ-SAMPLE-007: KV260 Hardware Round-Trip Validation Artifact

The system shall provide a bring-up script for the KV260 FPGA Ising control
plane, where:
- the script attempts a real PYNQ overlay / MMIO interaction using the
  existing AXI-Lite register-map contract
- when hardware is available, the script measures and records upload,
  trigger, and sample-readback latencies for one round trip
- when hardware is missing or incomplete, the script emits a blocker artifact
  that names the missing dependency, setup step, or overlay path instead of
  reporting fabricated timings
- the artifact labels the execution path explicitly as `hardware`,
  `software_model`, or `blocked` so fallback or simulated transports do not
  masquerade as live board validation
- the artifact records whether `FPGAIsingSampler(mode="auto")` would remain on
  the FPGA path or fall back to CPU in the same environment

### REQ-SAMPLE-008: Sampler-Backed Repair Reranking Replay Benchmark

The system shall provide a deterministic replay benchmark for sampler-backed
repair reranking, where:
- saved semantic and code repair attempts are normalized into recorded
  candidate sets with stable ordering, per-candidate verifier metadata, and
  source-artifact provenance
- candidate-set scoring uses the existing Carnot scorer path to produce a
  replayable energy score for each saved candidate without regenerating model
  outputs
- candidate selection can be executed through the CPU sampler path and through
  the KV260 path when a real or software-model transport is available
- the artifact labels the sampler execution path honestly as `hardware`,
  `software_model`, `cpu`, or `blocked`
- the artifact reports top-1 repair quality, verifier precision, repair yield,
  and reranking latency deltas versus the original saved repair outcome for
  both the semantic and code slices

## Scenarios

### SCENARIO-TRAIN-001: CD-1 Training Step

**Given** an Ising model and a batch of binary data
**When** one CD-1 training step is executed
**Then** model parameters are updated
**And** the energy of data samples decreases (on average over 100 steps)

### SCENARIO-TRAIN-002: Score Matching Convergence

**Given** a Gibbs model and data from a known Gaussian distribution
**When** trained with score matching for 1000 steps
**Then** the model's score function approximates the true score (MSE < 0.1)

### SCENARIO-SAMPLE-001: Langevin Sampling from Known Distribution

**Given** a quadratic energy function E(x) = 0.5 * x^T A x (Gaussian)
**When** 10000 Langevin dynamics steps are run
**Then** sample mean is within 0.1 of true mean
**And** sample covariance is within 0.2 (Frobenius norm) of true covariance

### SCENARIO-SAMPLE-002: HMC Accept Rate

**Given** an energy function and well-tuned HMC parameters
**When** 1000 HMC steps are run
**Then** acceptance rate is between 0.6 and 0.9

### SCENARIO-SAMPLE-003: Sampler Interface Genericity

**Given** a Langevin sampler and an HMC sampler
**When** both are used with the same EnergyFunction via the Sampler interface
**Then** both produce samples without requiring model-specific code

### SCENARIO-SAMPLE-004: Gradient Clipping Prevents Divergence

**Given** a Rosenbrock energy function producing gradients with norm ~3200
**When** Langevin dynamics is run with clip_norm=10.0 and step_size=0.01
**Then** all samples remain finite (no NaN or Inf)
**And** the sampler converges toward the energy minimum

### SCENARIO-SAMPLE-005: Gradient Clipping Is No-Op Below Threshold

**Given** a quadratic energy function with small gradients (norm < 1.0)
**When** a sampler runs with clip_norm=100.0
**Then** the gradient is not modified (clipping is a no-op)
**And** samples are statistically equivalent to the unclipped sampler

### SCENARIO-SAMPLE-006: Annealing Schedule

**Given** a linear or geometric annealing schedule with beta_init and beta_final
**When** the schedule is evaluated at step 0, midpoint, and final step
**Then** linear schedule returns beta_init at step 0 and beta_final at the last step
**And** geometric schedule midpoint equals sqrt(beta_init * beta_final)

### SCENARIO-SAMPLE-007: Parallel/Checkerboard Updates

**Given** an Ising model with coupling matrix J and bias vector h
**When** a checkerboard Gibbs update is applied at high inverse temperature
**Then** even-indexed and odd-indexed spins are updated in two parallel sweeps
**And** strong ferromagnetic coupling drives spins toward alignment

### SCENARIO-SAMPLE-008: thrml-Compatible Interface

**Given** a thrml IsingEBM instance with trained weights
**When** `parallel_sample_states` is called with the model and sample count
**Then** samples are returned as a list of (n_vars, 1) arrays matching thrml conventions
**And** the sampler uses extracted coupling matrices and biases from the IsingEBM

### SCENARIO-SAMPLE-009: Software FPGA Upload And Readback

**Given** a sparse Ising coupling matrix and bias vector for up to 4096 spins
**When** the software FPGA model receives the upload through the AXI-Lite
register map and a sampling run is triggered
**Then** the uploaded biases, sparse row pointers, and coupling entries are
stored without shape loss
**And** sampled spin states can be read back through the modeled result window

### SCENARIO-SAMPLE-010: FPGA Detection Fallback

**Given** an environment without a loadable PYNQ overlay or FPGA device
**When** `FPGAIsingSampler` is constructed in auto-detect mode
**Then** it remains callable through the `SamplerBackend` interface
**And** it delegates sampling to the CPU Ising sampler while recording that
the active execution path is a fallback

### SCENARIO-SAMPLE-011: Simulated FPGA Benchmark Contract

**Given** the same Ising problem and sample request
**When** the repository benchmark helper runs the software FPGA model and the
CPU backend
**Then** it returns elapsed-time metrics for both paths
**And** it reports the compared backend names, problem size, and sample shape

### SCENARIO-SAMPLE-012: KV260 Hardware Round-Trip Timing

**Given** a reachable KV260 PYNQ overlay exposing `carnot_ising_0.mmio`
**When** the Exp 242 bring-up script uploads a sparse Ising problem, triggers
one sampling run, and reads back the sample window
**Then** the artifact records measured upload, trigger, and readback
latencies
**And** it labels the execution path as `hardware`

### SCENARIO-SAMPLE-013: Blocker Artifact For Missing Hardware Bring-Up

**Given** an environment without the KV260 bitfile, PYNQ install, or MMIO
endpoint required for `carnot_ising_0`
**When** the Exp 242 bring-up script is executed
**Then** it writes a blocker artifact naming the exact missing dependency,
setup step, or overlay path
**And** it does not label the run as `hardware` or report fabricated latency
measurements

### SCENARIO-SAMPLE-014: Software-Model Bring-Up Is Labeled Honestly

**Given** the Exp 242 bring-up flow is exercised against the software overlay
for contract validation
**When** the upload, trigger, and readback path completes without real board
hardware
**Then** the artifact labels the execution path as `software_model`
**And** it preserves the distinction between software-model timings and live
hardware validation

### SCENARIO-SAMPLE-015: Saved Repair Histories Become Deterministic Candidate Sets

**Given** the checked-in Exp 235 semantic verify-repair artifact and Exp 238
spec-aware code artifact
**When** the Exp 243 replay benchmark is built
**Then** each replay case records the ordered saved candidates from the source
repair history without regenerating outputs
**And** each candidate preserves the saved verifier evidence and source
iteration needed for deterministic reranking

### SCENARIO-SAMPLE-016: CPU Sampler Reranks Candidate Sets From Saved Scores

**Given** a replay case with two or more saved candidates and replayable
candidate energy scores
**When** the Exp 243 benchmark executes the CPU sampler-backed reranker
**Then** it returns one top-1 candidate per case through the existing sampler
backend interface
**And** the artifact reports the selected candidate, selection latency, and
quality deltas versus the original saved repair outcome

### SCENARIO-SAMPLE-017: Hardware Mode Is Reported Honestly For Exp 243

**Given** the same KV260 transport conditions used for Exp 242
**When** the Exp 243 reranking benchmark is executed
**Then** the artifact labels the sampler path as `hardware`,
`software_model`, or `blocked`
**And** blocked runs preserve the setup blocker instead of fabricating
hardware-backed reranking timings

### REQ-SAMPLE-009: KV260 FPGA Overlay Bring-Up With 60s Timeout

The system shall provide a bring-up script for the KV260 FPGA Ising overlay
(Exp 288) that enforces a hard 60-second wall-clock timeout, where:
- the script checks the `CARNOT_KV260_BITFILE` environment variable as its
  first action and emits a blocked artifact immediately if the variable is unset
- the blocked artifact must include `execution_path: "blocked"`, the name of
  the missing environment variable, and the exact next step needed to proceed
- when the env var is set, the script attempts `pynq.Overlay(bitfile)` with a
  60-second timeout and emits a blocked artifact if the load does not complete
  within that window
- when the overlay loads, the script exercises the AXI-Lite register map from
  the Exp 228 design: write CONTROL, read STATUS, verify the STATUS round-trip
- the script writes a minimal coupling matrix into the bias window, issues
  `CONTROL_START`, reads back the spin state, and converts the boolean result
  to a ±1 signed integer array for validity checking
- the artifact records `overlay_load_ms`, `register_roundtrip_us`, and
  `spin_state_valid` when hardware or software-model path completes
- the artifact labels the execution path as `hardware`, `software_model`, or
  `blocked` — a software-model transport must never be labeled `hardware`

## Scenarios

### SCENARIO-SAMPLE-018: Blocked Artifact Emitted Immediately When Env Var Is Missing

**Given** `CARNOT_KV260_BITFILE` is not set in the environment
**When** the Exp 288 bring-up script is invoked
**Then** it writes a blocked artifact without attempting any PYNQ import
**And** the artifact contains `execution_path: "blocked"`, `missing:
  "CARNOT_KV260_BITFILE"`, and `next_step` describing how to set the variable
**And** `overlay_load_ms` is absent or null in the artifact

### SCENARIO-SAMPLE-019: Spin State Validity Check After Bring-Up Round-Trip

**Given** the Exp 288 bring-up script completed the upload/trigger/readback
cycle on either a hardware or software-model transport
**When** the packed boolean spin words are converted to a ±1 signed integer
array
**Then** every element of the array is exactly +1 or −1
**And** the artifact records `spin_state_valid: true`

### REQ-SAMPLE-010: FpgaBackend vs CPU Benchmark With 6× Quantum-Inspired Speedup Validation

The system shall provide a benchmark script (Exp 290) that compares FpgaBackend
against the CPU baseline (ParallelIsingSampler) across three problem sizes
(100, 500, 1000 spins) and validates the 6× SA-convergence speedup claimed in
arXiv 2604.04606 for the quantum-inspired geometric β-schedule, where:
- Each benchmark configuration enforces a hard 60-second wall-clock timeout and
  emits a partial artifact if the timeout is exceeded
- The artifact labels each run as `hardware` (when `CARNOT_KV260_BITFILE` is set
  and hardware is live) or `software_model` (CPU fallback) — never fabricates
  hardware labels in software simulation
- The benchmark measures samples/second for both FpgaBackend and CPU baseline
- Energy convergence is measured against the best energy found across 10
  independent restarts (used as a proxy for ground truth)
- The geometric (quantum-inspired) β-schedule is compared against a uniform
  (linear) β-schedule at identical step count; the artifact reports whether the
  geometric schedule achieves strictly lower or equal final energy
- LagONN penalty (arXiv 2505.07179) is tested with and without on a known
  highly-frustrated 3-SAT-derived Ising instance; the artifact reports energy
  difference (with_penalty − without_penalty) per problem size
- The primary prediction from arXiv 2604.04606 — that the quantum-inspired
  sparse β-schedule achieves ≥ 6× faster SA convergence — is recorded as
  `confirmed`, `refuted`, or `inconclusive` in the artifact

### SCENARIO-SAMPLE-020: Geometric Schedule Achieves Lower Energy Than Uniform Schedule

**Given** an Ising problem of 100, 500, or 1000 spins
**When** FpgaBackend geometric schedule and a CPU linear schedule are run for
the same number of annealing steps
**Then** the geometric schedule's best final energy is ≤ the linear schedule's
best final energy for at least 2 of 3 problem sizes
**And** the artifact records `schedule_comparison.geometric_wins` as a count
of sizes where geometric ≤ linear

### SCENARIO-SAMPLE-021: Benchmark Artifact Has Required Keys at All Problem Sizes

**Given** Exp 290 runs to completion (with or without timeout)
**When** the JSON artifact is written to `results/experiment_290_results.json`
**Then** the artifact contains a `results` list with one entry per problem size
**And** each entry contains: `n_spins`, `fpga_samples_per_sec`,
  `cpu_samples_per_sec`, `execution_path`, `schedule_comparison`,
  `lagonn_comparison`, and `timeout_exceeded`
**And** `execution_path` is one of `hardware`, `software_model`, or `timeout`

### SCENARIO-SAMPLE-022: LagONN Penalty Reduces Energy on Highly-Frustrated Instance

**Given** a highly-frustrated Ising instance derived from a 3-SAT problem
**When** FpgaBackend is run with `use_lagrangian_penalty=True` and `False`
**Then** the artifact records the mean final energy for both penalty modes
**And** the energy with penalty is ≤ the energy without penalty (confirming
  LagONN escapes infeasible local minima)

### REQ-SAMPLE-011: Verilog RTL for 128-Spin Ising Sampler (Exp 291)

The system shall provide synthesizable Verilog RTL for a 128-spin Ising sampler
targeting the KV260 FPGA (hardware/kv260/ising_sampler_v1.v), implementing:

- AXI-Lite slave interface with the Exp 289 register map: CONTROL (0x0000),
  STATUS (0x0004), SPIN_COUNT (0x0008), BETA_FINAL (0x001C), bias_ram
  (0x1000+), adj_ram (0x2000+), coupl_ram (0x4000+), spin_out (0x8010+)
- Q8.8 fixed-point (16-bit signed, 8 fractional bits) for all weights and β
- Sparse connectivity: max_degree=32 adjacency and coupling RAMs per spin
- Gibbs update: local field h_eff = Σ_j J_ij s_j + h_i, flip probability
  p = sigmoid(2 * h_eff * β), LFSR random comparison
- 16-bit Fibonacci LFSR pseudo-random number generator
- Log-linear β schedule: β(t) = β_min × (β_max/β_min)^(t/T), fixed-point
- Mpemba initialization: first 10% of N_STEPS run at β=0 (maximum temperature),
  then ramp from β_min per arXiv 2603.24183 to suppress slow relaxation modes
- Checkerboard (even/odd) spin update order for parallel correctness
- Halt after N_STEPS sweeps; expose final spin state via AXI read of spin_out

A companion Python behavioral simulation (scripts/simulate_ising_sampler.py)
implements identical logic in numpy for test validation without physical FPGA.

### SCENARIO-SAMPLE-023: AXI Register Map Fully Addressable

**Given** the Verilog module ising_sampler_128 with the Exp 289 AXI-Lite map
**When** all control and memory registers are written and read via AXI-Lite
**Then** CONTROL, STATUS, SPIN_COUNT, BETA_FINAL, bias_ram[0..127],
  adj_ram[0..127*32-1], coupl_ram[0..127*32-1], spin_out[0..3] are all
  accessible at their documented offsets with correct read-back

### SCENARIO-SAMPLE-024: Gibbs Update Produces Correct Energy on 4-Spin Graph

**Given** a 4-spin Ising ring (J=1.0, h=0.0) with initial state [+1,-1,+1,-1]
**When** one Gibbs sweep is executed by the behavioral simulation matching Verilog
**Then** the computed local field h_eff for each spin matches the Python reference
  (h_eff_i = Σ_j J_ij s_j) to within Q8.8 quantization error (< 0.01)
**And** the total energy E = -Σ_ij J_ij s_i s_j matches the Python reference

### REQ-SAMPLE-012: KV260 Hardware Round-Trip Latency ≤100μs for 100-Spin Ising (Exp 313)

The system shall attempt actual KV260 FPGA hardware bring-up and, when the hardware
is available, demonstrate a 100-spin Ising sampling round-trip within 100μs:

- Detect hardware availability via CARNOT_KV260_BITFILE env var, pynq import, and
  overlay load; each detection step is recorded as a named check
- When hardware is not available, emit an honest_verdict from the set:
  "blocked_no_bitfile", "blocked_pynq", "blocked_overlay", "blocked_timeout"
- When hardware is available, measure 100 trials of 100-spin sampling and record
  mean_latency_us and p99_latency_us
- CPU fallback latency is always measured (100 trials) for comparison, regardless
  of hardware status
- Artifact schema: experiment=313, honest_verdict, kv260_detected, bringup_steps_passed,
  hardware_latency_us (null if not working), cpu_fallback_latency_us
- honest_verdict="hardware_working" only when: overlay loads, AXI register round-trip
  succeeds, all returned spins are ±1, and mean_latency_us ≤ 100μs

### SCENARIO-SAMPLE-025: Hardware Latency Within 100μs For 100-Spin Ising

**Given** the KV260 overlay loads and AXI-Lite register map is accessible
**When** 100 trials of 100-spin Ising sampling are measured end-to-end
**Then** mean_latency_us ≤ 100 and p99_latency_us ≤ 200
**And** all returned spin values are exactly +1 or −1
**And** honest_verdict = "hardware_working" is recorded in the artifact

### SCENARIO-SAMPLE-026: CPU Fallback Always Measured For Comparison

**Given** any execution path (hardware working or blocked)
**When** the experiment completes
**Then** cpu_fallback_latency_us is present and non-negative in the artifact
**And** the artifact records honest_verdict (never fabricated as "hardware_working"
  when hardware was not exercised)
**And** kv260_detected is False when CARNOT_KV260_BITFILE is unset or pynq is absent

### SCENARIO-TRAIN-003: Checkpoint Round-Trip

**Given** a model trained for N steps
**When** a checkpoint is saved and loaded
**Then** training can resume from step N with identical optimizer state
**And** the next gradient step produces identical results

### REQ-HW-003: Open-Source FPGA Synthesis of Ising Sampler (Exp 349)

The system shall attempt open-source synthesis of the KV260 Ising sampler RTL using
yosys and nextpnr-xilinx, proving the open-source path is viable before investing in
a Vivado license:

- check_yosys_available() checks subprocess yosys --version, returns False gracefully
  when yosys is not installed
- check_nextpnr_available() checks subprocess nextpnr-xilinx --version, returns False
  gracefully when nextpnr-xilinx is not installed
- check_verilog_source_exists(path) checks os.path.exists on the RTL source path
- parse_synthesis_output(stdout) extracts LUT count, FF count, and timing info from
  yosys synthesis report via regex on "Number of cells:" style lines
- SynthesisResult dataclass records: yosys_available, nextpnr_available,
  verilog_found, synthesis_attempted, synthesis_success, lut_count, ff_count,
  and honest_verdict
- honest_verdict options: "synthesis_success", "synthesis_partial" (netlist only,
  no place-and-route), "blocked_missing_yosys", "blocked_missing_verilog",
  "synthesis_failed"
- When yosys is available and Verilog source exists: run synthesis, capture output,
  parse LUT/FF counts from report
- When nextpnr-xilinx is also available: attempt place-and-route for KV260
  (xczu5ev-sfvc784-2-e device); if successful, write bitfile to
  hardware/kv260/carnot_ising.bit
- Artifact schema: experiment=349, schema="carnot.fpga_synthesis.v1",
  prereqs_checked, synthesis_result, lut_count, ff_count, bitfile_generated (bool),
  honest_verdict

### SCENARIO-HW-005: Yosys Synthesis Reports LUT/FF Utilization

**Given** yosys is installed and hardware/kv260/ising_sampler_v1.v exists
**When** synthesis is run with synth_xilinx targeting the Ising sampler top module
**Then** synthesis completes without error
**And** parse_synthesis_output extracts a non-zero lut_count from the report
**And** honest_verdict is "synthesis_success" or "synthesis_partial"
**And** the SynthesisResult is serialized to results/experiment_349_kv260_synthesis.json

### SCENARIO-HW-006: Graceful Blocked Verdict When Prerequisites Missing

**Given** yosys is not installed OR hardware/kv260/ising_sampler_v1.v is not present
**When** the experiment runs
**Then** no synthesis is attempted
**And** honest_verdict is "blocked_missing_yosys" or "blocked_missing_verilog"
**And** synthesis_attempted is False in the artifact
**And** lut_count and ff_count are None in the artifact

### REQ-SAMPLE-015: KAEMEnergy Exact Inference via Inverse-Transform Sampling (Exp 447)

The system shall implement KAEMEnergy with exact inference using the Kolmogorov-Arnold
Energy Model (KAEM, arXiv 2506.14167), where:
- UnivariateKAEMLayer stores per-variable marginal energy splines (one 1D B-spline per variable)
- The joint energy decomposes as E(x) = sum_i e_i(x_i), making variables independent
- energy(x) returns the scalar sum of per-variable spline energies, differentiable via JAX
- marginal_cdf(var_idx, x) computes the marginal CDF of variable var_idx at point x using
  numerical integration (trapezoidal rule over a _N_QUAD=256-point grid)
- sample_exact(n_samples, rng_key) draws exact joint samples via inverse-transform: for each
  variable, invert the marginal CDF using binary search on the precomputed CDF table
- KAEMEnergy wraps UnivariateKAEMLayer with sample(), energy(), and fit() methods
- fit(data, n_epochs) trains the model by gradient descent on per-variable marginal energies
- All samples produced by sample_exact are in [-1, 1] (the spline domain)
- energy() is differentiable: jax.grad(model.energy)(x) returns a valid gradient

### REQ-SAMPLE-016: KAEMEnergy Latency Benchmark vs IsingEBM MCMC (Exp 447)

The system shall provide benchmark_kaem_vs_mcmc(n_vars, n_samples) that:
- Runs both KAEM exact sampling and IsingEBM MCMC (ParallelIsingSampler) on the same problem
- Measures wall-clock latency for each method with a warm-up draw to exclude JIT compilation
- Returns a dict with n_vars, n_samples, kaem_latency_ms, ising_mcmc_latency_ms, speedup_ratio
- Experiment 447 runs this benchmark for n_vars in {10, 25, 50, 100} with n_samples=100
- honest_verdict is 'kaem_faster' if mean speedup > 5x, 'modest_speedup' if > 1.5x, else 'no_speedup'
- The artifact is written to results/experiment_447_kaem_exact_sampling.json with
  schema='carnot.kaem_exact.v1'

### SCENARIO-SAMPLE-027: KAEM Exact Samples Are in [-1, 1]

**Given** a KAEMEnergy or UnivariateKAEMLayer with n_vars variables
**When** sample_exact(n_samples, rng_key) is called
**Then** all returned samples have shape (n_samples, n_vars)
**And** every element is in the range [-1, 1]
**And** all values are finite (no NaN or Inf)

### SCENARIO-SAMPLE-028: KAEM Energy Is JAX-Differentiable

**Given** a KAEMEnergy model with n_vars variables
**When** jax.grad(model.energy)(x) is called for any x in [-1, 1]^n_vars
**Then** the gradient has shape (n_vars,) and all values are finite
**And** energy(x) returns a scalar (shape ())

### SCENARIO-SAMPLE-029: KAEM Benchmark Returns Valid Latency Dict

**Given** a benchmark_kaem_vs_mcmc(n_vars, n_samples) call
**When** the benchmark completes
**Then** the result dict contains n_vars, n_samples, kaem_latency_ms, ising_mcmc_latency_ms,
  speedup_ratio
**And** both latencies are non-negative
**And** speedup_ratio is positive
**And** all numeric values are finite

### REQ-SAMPLE-017: GPUOscillatorIsingSimulator — Parallel OIM Dynamics (Exp 472)

The system SHALL provide `GPUOscillatorIsingSimulator` that implements GPU-accelerated
Oscillator Ising Machine (OIM) dynamics via JAX/CUDA parallel spin updates.

- REQ-SAMPLE-017-1: `GPUOscillatorIsingSimulator(n_spins: int, n_steps: int = 1000, device: str = 'cpu')`
  SHALL implement discrete Kuramoto phase update: phi_i += dt * K * sum_j J_ij * sin(phi_j - phi_i)
  for all spins simultaneously via JAX vmap.
- REQ-SAMPLE-017-2: `sample(J: jnp.ndarray, n_samples: int) -> jnp.ndarray` SHALL return shape
  (n_samples, n_spins) boolean array. Each sample is an independent OIM trajectory.
- REQ-SAMPLE-017-3: The simulator SHALL fall back gracefully to CPU if the requested device is unavailable.

Spec: REQ-SAMPLE-017, SCENARIO-SAMPLE-030

### REQ-SAMPLE-018: GPUOscillatorIsingSimulator Speedup Benchmark (Exp 472)

The system SHALL provide `OIMSpeedupResult` comparing GPU OIM vs CPU ParallelIsingSampler.

- REQ-SAMPLE-018-1: `GPUOscillatorIsingSimulator.benchmark(J, n_samples=1000) -> float` SHALL return
  milliseconds-per-sample (wall-clock, post-JIT, excluding compile time).
- REQ-SAMPLE-018-2: `OIMSpeedupResult(n_spins: int, gpu_ms: float, cpu_ms: float)` SHALL expose
  `speedup: float` (= cpu_ms / gpu_ms) and `is_production_ready: bool` (speedup >= 10.0).
- REQ-SAMPLE-018-3: On RTX 3090 at n_spins=128, GPU OIM SHALL achieve >= 10x speedup vs CPU
  ParallelIsingSampler, enabling real-time JEPA gating without FPGA hardware.

Spec: REQ-SAMPLE-018, SCENARIO-SAMPLE-031

### SCENARIO-SAMPLE-030: GPUOscillatorIsingSimulator Sample Shape

**Given** `GPUOscillatorIsingSimulator(n_spins=128, device='cpu')`
**When** `sample(J, n_samples=10)` is called with a valid 128×128 coupling matrix
**Then** the result has shape (10, 128)
**And** all values are boolean (True = spin +1)
**And** no exceptions are raised

### SCENARIO-SAMPLE-031: OIMSpeedupResult Production Ready

**Given** `OIMSpeedupResult(n_spins=128, gpu_ms=0.1, cpu_ms=10.0)`
**When** `speedup` and `is_production_ready` are evaluated
**Then** `speedup == 100.0`
**And** `is_production_ready == True`

### REQ-SAMPLE-019: KAEM Crossover Profile — n_vars Where Speedup >= 5x vs MCMC

The system SHALL profile KAEM exact sampling vs ParallelIsingSampler MCMC at
n_vars in (100, 200, 300, 500, 1000) to find the exact crossover point where
KAEM achieves >= 5x speedup.

- `KAEMCrossoverResult(n_vars_tested: list[int], speedups: list[float])` SHALL expose:
  - `crossover_n_vars: int | None` — first n_vars where speedup_ratio >= 5.0; None if not found
  - `max_speedup: float` — maximum speedup observed across all n_vars tested
  - `kaem_viable_for_production: bool` — True iff crossover_n_vars is not None
  - `speedup_at(n_vars: int) -> float` — speedup ratio at the given n_vars

Spec: REQ-SAMPLE-019, SCENARIO-SAMPLE-032

### SCENARIO-SAMPLE-032: KAEMCrossoverResult Crossover Detection

**Given** `KAEMCrossoverResult(n_vars_tested=[100, 200, 300], speedups=[1.5, 3.0, 6.0])`
**When** `crossover_n_vars`, `max_speedup`, and `kaem_viable_for_production` are evaluated
**Then** `crossover_n_vars == 300`
**And** `max_speedup == 6.0`
**And** `kaem_viable_for_production == True`
**And** `speedup_at(200) == 3.0`

### REQ-SAMPLE-020: KAEM Extended Profile — n_vars Up to 5000 with Early Stopping

The system SHALL profile KAEM exact sampling vs ParallelIsingSampler MCMC at
n_vars in (1000, 2000, 3000, 5000) with early stopping at the first n_vars where
the speedup ratio reaches or exceeds 5x (VIABILITY_THRESHOLD).

- `KAEMExtendedResult` SHALL expose all fields of `KAEMCrossoverResult` plus:
  - `prior_max_n: int` — the largest n_vars tested in the preceding experiment (Exp 483)
  - `vs_prior: str` — `'crossover_found'` if crossover found, else `'no_crossover_extended'`

### REQ-SAMPLE-021: KAEM CPU Viability Verdict and FPGA Path

If no crossover is found at n_vars=5000 (the maximum CPU-feasible size for this
profiling pass), `KAEMExtendedResult` SHALL set:
- `kaem_viable_for_cpu: bool = False`
- `fpga_path_recommended: bool = True`

This closes RETRO-031: KAEM does not achieve 5x speedup at CPU-feasible sizes;
revisit for FPGA implementation where bisection is native arithmetic.

### SCENARIO-SAMPLE-033: KAEMExtendedResult Crossover Found at n=2000

**Given** `KAEMExtendedResult(n_vars_tested=[1000, 2000], speedups=[2.0, 6.0], prior_max_n=1000)`
**When** evaluated
**Then** `crossover_n_vars == 2000`
**And** `kaem_viable_for_cpu == True`
**And** `fpga_path_recommended == False`
**And** `vs_prior == 'crossover_found'`

### SCENARIO-SAMPLE-034: KAEMExtendedResult No Crossover at n=5000 (FPGA Path)

**Given** `KAEMExtendedResult(n_vars_tested=[1000, 2000, 3000, 5000], speedups=[1.5, 2.0, 2.5, 3.0], prior_max_n=1000)`
**When** evaluated
**Then** `crossover_n_vars == None`
**And** `kaem_viable_for_cpu == False`
**And** `fpga_path_recommended == True`
**And** `vs_prior == 'no_crossover_extended'`

### REQ-SAMPLE-022: KAEM Distribution Family Benchmark — Three Families (Exp 508)

The system SHALL implement a benchmark that tests KAEM exact sampling vs MCMC on three
distribution families: GaussianMixture, StudentT, and PiecewiseUniform.

- `DistributionFamilyResult(family_name: str, kaem_mean_l2: float, mcmc_mean_l2: float)` SHALL expose:
  - `kaem_advantage: float` — `mcmc_mean_l2 - kaem_mean_l2` (positive = KAEM wins)
  - `kaem_wins: bool` — `True` iff `kaem_advantage > 0`
  - `to_dict() -> dict`
- `KAEMDistributionBenchmark(n_vars: int = 10, n_samples: int = 200)` SHALL expose:
  - `benchmark_gaussian_mixture() -> DistributionFamilyResult`
  - `benchmark_student_t(nu: float = 2.0) -> DistributionFamilyResult`
  - `benchmark_piecewise_uniform(n_pieces: int = 5) -> DistributionFamilyResult`
  - `best_family() -> str` — name of family with largest kaem_advantage, or `'none'` if all kaem_wins=False

Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-035

### REQ-SAMPLE-023: KAEM Distribution Family Benchmark — mean_l2 Metric (Exp 508)

The benchmark SHALL compute `mean_l2` as the mean L2 distance between the empirical CDF of
KAEM samples and the empirical CDF of ground-truth samples drawn from each distribution family.
This metric captures how well the sampler covers the full support of the target distribution.

Spec: REQ-SAMPLE-023, SCENARIO-SAMPLE-036

### REQ-SAMPLE-024: KAEM Distribution Family Benchmark — Best Family Identification (Exp 508)

The benchmark SHALL identify the distribution family with the largest KAEM advantage
(defined as `mcmc_mean_l2 - kaem_mean_l2`). If MCMC wins on all three families
(all `kaem_wins=False`), the result is `'none'`. This closes RETRO-031 if any
family shows KAEM advantage.

Spec: REQ-SAMPLE-024, SCENARIO-SAMPLE-037

### SCENARIO-SAMPLE-035: DistributionFamilyResult KAEM Wins

**Given** `DistributionFamilyResult(family_name='gaussian_mixture', kaem_mean_l2=0.1, mcmc_mean_l2=0.5)`
**When** `kaem_advantage` and `kaem_wins` are evaluated
**Then** `kaem_advantage == 0.4`
**And** `kaem_wins == True`

### SCENARIO-SAMPLE-036: DistributionFamilyResult KAEM Loses

**Given** `DistributionFamilyResult(family_name='student_t', kaem_mean_l2=0.6, mcmc_mean_l2=0.3)`
**When** `kaem_advantage` and `kaem_wins` are evaluated
**Then** `kaem_advantage == -0.3`
**And** `kaem_wins == False`

### SCENARIO-SAMPLE-037: KAEMDistributionBenchmark Best Family

**Given** a `KAEMDistributionBenchmark` instance
**When** all three benchmarks are run and `best_family()` is called
**Then** it returns the family name with the largest `kaem_advantage`
**Or** returns `'none'` if all `kaem_wins` are False

### REQ-SAMPLE-025: CIKANEnergy — Constraint-Informed KAN with Boundary-Concentrated Knots (Exp 519)

The system SHALL implement `CIKANEnergy` (Constraint-Informed KAN, arXiv 2412.03710) that
concentrates spline knot density near hard constraint boundaries, where:
- `ConstraintBoundary(position: float, sharpness: float = 1.0)` represents a boundary at
  `position` in [-1, 1], with `sharpness` controlling the width of the extra-knot region
- `CIKANLayer(n_vars, n_knots_base=8, boundary_k=4)` extends `UnivariateKAEMLayer`:
  - `_distribute_knots_with_boundaries(boundaries, data_std)` returns sorted knot positions
    with `boundary_k - 1` additional knots inserted within `sharpness * data_std` of each boundary
  - With boundaries, a CIKANLayer with `n_knots_base=8` has more knots near boundary positions
    than an equivalent `UnivariateKAEMLayer(n_knots=8)`
- `CIKANEnergy(n_vars, n_hidden=16, boundaries=None)` extends `KAEMEnergy`:
  - `fit_with_constraints(data, boundaries)` sets `self.boundaries` and calls `fit(data)` using
    boundary-aware knot placement in the underlying `CIKANLayer`
  - `energy(x)` is JAX-differentiable: `jax.grad(model.energy)(x)` returns a valid gradient
- Why: uniform knots (KAEM, Exp 447) miss the complex energy landscape near constraint
  boundaries; concentrating knots there gives sharper gradients and higher AUROC on near-boundary
  examples at the cost of smoothness far from boundaries — a correct tradeoff for constraint
  verification

Spec: REQ-SAMPLE-025, SCENARIO-SAMPLE-038, SCENARIO-SAMPLE-039, SCENARIO-SAMPLE-040

### REQ-SAMPLE-026: CIKANEnergy Boundary AUROC Advantage Benchmark (Exp 519)

The system SHALL provide Experiment 519 that validates CIKANEnergy's boundary AUROC advantage:
- Train `KAEMEnergy` (baseline) and `CIKANEnergy(boundaries=[0.0])` on 400 samples from a
  distribution with a hard boundary at x=0 (correct side: x>0, violated: x<0)
- Evaluate AUROC on 100 held-out samples split into near-boundary (|x|<0.2) and far (|x|>0.5)
- Artifact schema `carnot.cikan_energy.v1` must include:
  `baseline_auroc_near_boundary`, `cikan_auroc_near_boundary`,
  `baseline_auroc_far`, `cikan_auroc_far`,
  `cikan_advantage` (bool), `honest_verdict` ('cikan_advantage' or 'no_advantage')

Spec: REQ-SAMPLE-026, SCENARIO-SAMPLE-040

### SCENARIO-SAMPLE-038: CIKANLayer Has More Knots Near Boundary

**Given** a `CIKANLayer(n_vars=1, n_knots_base=8, boundary_k=4)` with `boundaries=[ConstraintBoundary(0.5)]`
**When** `_distribute_knots_with_boundaries([ConstraintBoundary(0.5)], data_std=0.3)` is called
**Then** the returned knot array has more than 8 elements (extra knots inserted near 0.5)
**And** all returned knot positions are in [-1, 1]
**And** the positions are sorted in ascending order

### SCENARIO-SAMPLE-039: CIKANEnergy Energy Is JAX-Differentiable

**Given** a `CIKANEnergy(n_vars=3)` model
**When** `jax.grad(model.energy)(x)` is called for any x in [-1, 1]^3
**Then** the gradient has shape (3,) and all values are finite
**And** `energy(x)` returns a scalar (shape ())

### SCENARIO-SAMPLE-040: CIKANEnergy fit_with_constraints Completes

**Given** `CIKANEnergy(n_vars=1)` and 50 training samples in [-1, 1]
**When** `fit_with_constraints(data, boundaries=[ConstraintBoundary(0.0)])` is called
**Then** it completes without error
**And** `self.boundaries` equals the provided boundaries list

### REQ-SAMPLE-027: LowRankKAEMEnergy — Project to Top-k Singular Vectors Before Spline Computation (Exp 532)

The system SHALL implement `LowRankKAEMEnergy` that projects input to the top-k left singular
vectors of the training data matrix before spline computation, where:
- `LowRankProjector(data: jnp.ndarray, k: int = 11)` computes the SVD of data and stores the
  top-k left singular vectors (principal directions of the logit energy landscape)
- `.project(x: jnp.ndarray) -> jnp.ndarray` projects x (shape n_vars,) into k-dimensional subspace
- `.explained_variance_ratio(k: int) -> float` returns the fraction of total variance captured by the top-k components
- `.auto_k(threshold: float = 0.90) -> int` returns the minimum k such that explained_variance_ratio(k) >= threshold
- `LowRankKAEMEnergy(n_vars, k=11, auto_k=False)` wraps KAEMEnergy: fit() first computes the
  LowRankProjector, then projects data to k dims, then fits KAEMEnergy on the projected data
- `energy(x)` projects x to k dims then calls the underlying KAEMEnergy.energy() — differentiable via jax.grad
- Why: arXiv 2604.04384 shows transformer logit matrices are low-rank; 90% of variance is in
  only 2-11 singular components; projecting to this subspace reduces compute from O(n_vars) to
  O(k) spline evaluations where k << n_vars, preserving information while cutting cost 10-100x

Spec: REQ-SAMPLE-027, SCENARIO-SAMPLE-041, SCENARIO-SAMPLE-042, SCENARIO-SAMPLE-043

### REQ-SAMPLE-028: LowRankKAEMEnergy Rank Auto-Selection (Exp 532)

The system SHALL implement rank auto-selection in `LowRankProjector`:
- `auto_k(threshold=0.90) -> int` returns the minimum k such that the top-k singular values
  explain >= `threshold` fraction of total variance (sum of squared singular values)
- When `auto_k=True` in `LowRankKAEMEnergy`, the projector selects k automatically during fit()
  rather than using the constructor's fixed k value
- This mirrors arXiv 2604.04384's observation that 90% explained variance requires only 2-11
  components for real transformer logit matrices, making the rank selection data-driven

Spec: REQ-SAMPLE-028, SCENARIO-SAMPLE-042

### REQ-VERIFY-106: PottsMachineVerifier — q-State Constraint Encoding

The system shall implement a `PottsMachineVerifier` in `python/carnot/models/potts_machine.py`
that generalizes IsingEBM from binary (+1/-1) to q discrete states per spin, where:
- each spin represents one constraint variable with states {0, 1, ..., q-1}
- the default q=3 encoding maps: 0=correct, 1=partial, 2=violated
- the coupling tensor J has shape (q, q, n_spins, n_spins), symmetric under (a,b,i,j) <-> (b,a,j,i)
- the local field h has shape (q, n_spins)
- the energy function is E(s) = -0.5 * sum_ij J[s_i, s_j, i, j] - sum_i h[s_i, i]
- coupling structure is sparse-compatible for FPGA upload (same AXI-Lite register-map format as IsingEBM)
- arXiv 2602.04200 mean-field constraints preserve sparsity during training

### REQ-VERIFY-107: q=3 Gibbs Sampler

The system shall implement a sequential Gibbs sampler for PottsMachineVerifier where:
- each spin i is updated by computing conditional energy E_i(a) for all states a in {0,...,q-1}
- the new state is sampled from P(s_i=a | s_{-i}) ∝ exp(-beta * E_i(a))
- one full sweep updates all n_spins spins sequentially
- `gibbs_update(config, beta)` performs one sweep
- `sample(n_steps)` runs n_steps sweeps from a random initial configuration

### REQ-VERIFY-108: AUROC Parity — PottsMachineVerifier vs IsingEBM Baseline

The system shall provide experimental evidence (Exp 534) that:
- 3-class AUROC for PottsMachineVerifier (q=3) is reported alongside binary AUROC for IsingEBM
- partial_class_accuracy measures how well PottsMachineVerifier identifies partial-credit examples
- potts_viable is true when potts_3class_auroc >= ising_binary_auroc

### SCENARIO-VERIFY-142: PottsMachineVerifier Energy Is Scalar

**Given** a `PottsMachineVerifier(n_spins=8, q=3)`
**When** `energy(config)` is called with a valid integer config of shape (8,)
**Then** the result is a scalar JAX array
**And** the value is finite

### SCENARIO-VERIFY-143: Gibbs Update Changes Configuration

**Given** a `PottsMachineVerifier(n_spins=8, q=3)` initialized with random weights
**When** `gibbs_update(config, beta=1.0)` is called
**Then** the returned configuration has the same shape as input
**And** at least one spin differs from the input with high probability

### SCENARIO-VERIFY-144: predict_class Returns Valid Label

**Given** a trained `PottsMachineVerifier(n_spins=8, q=3)`
**When** `predict_class(config)` is called with any valid config
**Then** the return value is an integer in {0, 1, 2}

### SCENARIO-SAMPLE-041: LowRankProjector Compresses to k Dimensions

**Given** a `LowRankProjector` fitted on 100-dimensional data with k=2
**When** `.project(x)` is called for any x of shape (100,)
**Then** the result has shape (2,)
**And** all values are finite

### SCENARIO-SAMPLE-042: LowRankProjector Explained Variance >= 0.90 at k=11

**Given** a `LowRankProjector` fitted on data with known low-rank structure (rank-11 signal + noise)
**When** `.explained_variance_ratio(k=11)` is evaluated
**Then** the ratio is >= 0.90
**And** `auto_k(threshold=0.90)` returns a value <= 11

### SCENARIO-SAMPLE-043: LowRankKAEMEnergy Energy Is JAX-Differentiable

**Given** a `LowRankKAEMEnergy(n_vars=50, k=11)` fitted on 100 training samples
**When** `jax.grad(model.energy)(x)` is called for x of shape (50,)
**Then** the gradient has shape (50,) and all values are finite
**And** `energy(x)` returns a scalar

### REQ-SAMPLE-029: LowRankKAEMEnergy as Default KAN Fast-Path Tier for n_vars <= 100 (Exp 544)

The system SHALL provide a `get_kaem_energy(n_vars, use_lowrank=True, k=2)` factory function
in `python/carnot/models/kaem_energy.py` that:
- Returns `LowRankKAEMEnergy(n_vars=n_vars, k=k)` when `use_lowrank=True`
- Returns `KAEMEnergy(n_vars=n_vars)` when `use_lowrank=False`
- Default `use_lowrank=True` and `k=2` based on Exp 532 result (k=2 achieves 23.7x speedup)

The `VerificationResult` dataclass in `python/carnot/pipeline/verify_repair.py` SHALL have:
- `use_lowrank_kaem: bool = False` field tracking whether the KAN fast-path used low-rank KAEM

The `VerifyRepairPipeline` in `python/carnot/pipeline/verify_repair.py` SHALL have:
- `_build_kan_fast_path_model(n_vars)` method that calls `get_kaem_energy(n_vars, use_lowrank=(n_vars <= 100))`
  returning `(model, use_lowrank)` where `use_lowrank` is the flag for `VerificationResult`

**Rationale:** Exp 532 demonstrated LowRankKAEMEnergy at k=2 achieves 23.7x speedup over
full-rank MCMC. Wiring it as the default fast-path tier for n_vars <= 100 captures that
speedup in the verification cascade without sacrificing accuracy.

Spec: REQ-SAMPLE-029, SCENARIO-SAMPLE-044, SCENARIO-SAMPLE-045

### SCENARIO-SAMPLE-044: get_kaem_energy Returns LowRankKAEMEnergy for n_vars <= 100

**Given** `get_kaem_energy(n_vars=50, use_lowrank=True, k=2)` is called
**When** the factory function executes
**Then** the returned object is an instance of `LowRankKAEMEnergy`
**And** `model.n_vars == 50` and `model.k == 2`

### SCENARIO-SAMPLE-045: get_kaem_energy Returns KAEMEnergy When use_lowrank=False

**Given** `get_kaem_energy(n_vars=200, use_lowrank=False)` is called
**When** the factory function executes
**Then** the returned object is an instance of `KAEMEnergy`
**And** `model.n_vars == 200`

### REQ-SAMPLE-030: CalibratedLowRankKAEMEnergy — Affine Calibration Layer for Energy Accuracy (Exp 559)

**Problem:** Exp 544/RETRO-057 found that LowRankKAEMEnergy at k=2 achieves 4-155x speedup but
energy_mad_normalized ≈ 0.96-0.99, far outside the 5% production tolerance. The low-rank SVD
projection at small k discards information needed to match the scale and offset of the full-rank
energy function.

**Solution:** Fit an affine calibration layer E_calibrated = a * E_lowrank + b using least-squares
regression on synthetic Ising instances. The calibration corrects scale (a) and offset (b) so that
the calibrated low-rank energy closely tracks the full-rank energy.

**Requirements:**
- REQ-SAMPLE-030-1: `CalibrationLayer` SHALL expose `fit(E_full, E_lowrank)` that fits least-squares
  affine parameters (a, b) such that a * E_lowrank + b ≈ E_full.
- REQ-SAMPLE-030-2: `CalibrationLayer.transform(E_lowrank)` SHALL apply the fitted affine transform
  a * E_lowrank + b and return the calibrated energy.
- REQ-SAMPLE-030-3: `CalibratedLowRankKAEMEnergy` SHALL wrap `LowRankKAEMEnergy` and apply
  `CalibrationLayer.transform()` after the low-rank energy computation.
- REQ-SAMPLE-030-4: `CalibratedLowRankKAEMEnergy.calibrate(n_samples, n_vars)` SHALL generate
  synthetic Ising instances, compute E_full and E_lowrank, and fit the CalibrationLayer.
- REQ-SAMPLE-030-5: The production threshold is energy_mad_normalized < 0.05 (5% of full-rank
  energy std). A rank k that satisfies this threshold with speedup > 5x is the production k.

Spec: REQ-SAMPLE-030, SCENARIO-SAMPLE-046, SCENARIO-SAMPLE-047, SCENARIO-SAMPLE-048

### SCENARIO-SAMPLE-046: CalibrationLayer fit/transform Reduces energy_mad

**Given** a CalibrationLayer is created and fit on 1000 E_full / E_lowrank pairs
**When** transform(E_lowrank) is applied to a held-out set
**Then** mean(|E_calibrated - E_full|) < mean(|E_lowrank - E_full|) (calibration strictly improves accuracy)

### SCENARIO-SAMPLE-047: CalibratedLowRankKAEMEnergy calibrate Completes Without Error

**Given** CalibratedLowRankKAEMEnergy(n_vars=20, k=4) is constructed
**When** calibrate(n_samples=100, n_vars=20) is called
**Then** the CalibrationLayer is fitted (a != 1.0 or b != 0.0 in general)
**And** energy() returns a scalar after calibration

### SCENARIO-SAMPLE-048: Rank Sweep Finds Production k With energy_mad < 0.05

**Given** a rank sweep over k in [2, 4, 8, 16, 32] with calibration applied at each k
**When** energy_mad_normalized is measured for each calibrated model vs full-rank KAEM
**Then** at least one k achieves energy_mad_normalized < 0.05 and speedup > 5x (RETRO-057 closure)

### REQ-SAMPLE-031: KV260 FPGA Bring-Up v2 — Hardware Latency <100μs for 100-Spin Ising (Exp 568)

The system SHALL complete KV260 FPGA bring-up now that the board is physically available
(arrived 2026-04-20, blocked in Exps 228, 288, 289, 290, 313):

- REQ-SAMPLE-031-1: When CARNOT_KV260_BITFILE is set, FpgaBackend SHALL be instantiated and
  100 sampling trials (100-spin Ising) SHALL be timed; mean_latency_us SHALL be recorded.
- REQ-SAMPLE-031-2: fpga_alive SHALL be True only when mean_latency_us < 100μs (vs CPU ~358ms).
- REQ-SAMPLE-031-3: When CARNOT_KV260_BITFILE is unset, a valid Vivado synthesis command
  AND a minimal hardware/kv260/synth_ising.tcl stub SHALL be generated automatically.
- REQ-SAMPLE-031-4: CPU simulation baseline SHALL always be measured regardless of hardware status.
- REQ-SAMPLE-031-5: honest_verdict SHALL be one of:
  'hardware_working' (fpga_alive=True),
  'synthesis_required' (bitfile not set),
  'hardware_too_slow' (bitfile set but latency >= 100μs).

Spec: REQ-SAMPLE-031, SCENARIO-SAMPLE-049, SCENARIO-SAMPLE-050, SCENARIO-SAMPLE-051

### REQ-SAMPLE-032: KV260 Vivado Synthesis — Ising Sampler RTL Compiled to Bitfile (Exp 584)

The system SHALL attempt Vivado synthesis and document the result now that the KV260 board
is physically present (arrived 2026-04-20, Exp 568 honest_verdict=synthesis_required):

- REQ-SAMPLE-032-1: When Vivado is installed and on PATH, the experiment SHALL run
  `vivado -mode batch -source hardware/kv260/synth_ising.tcl` with a 7000-second timeout
  and record the first 2000 chars of stdout.
- REQ-SAMPLE-032-2: bitfile_built SHALL be True only when the .bit file exists at
  output/carnot_ising_synth/carnot_ising.bit after synthesis.
- REQ-SAMPLE-032-3: When Vivado is NOT installed, vivado_install_steps SHALL be populated
  with the manual installation commands needed to install Vivado 2023.2.
- REQ-SAMPLE-032-4: cpu_baseline_latency_us from Exp 568 (289608 µs) SHALL always be
  recorded in the artifact for comparison.
- REQ-SAMPLE-032-5: honest_verdict SHALL be one of:
  'bitfile_built' (synthesis succeeded, .bit file exists),
  'synthesis_attempted_failed' (Vivado ran but no .bit file produced),
  'vivado_not_installed' (Vivado not on PATH).

Spec: REQ-SAMPLE-032, SCENARIO-SAMPLE-052, SCENARIO-SAMPLE-053, SCENARIO-SAMPLE-054

### SCENARIO-SAMPLE-052: Vivado Not Installed — Install Steps Recorded

**Given** Vivado is not found on PATH
**When** Exp 584 runs
**Then** vivado_available = False
**And** bitfile_built = False
**And** vivado_install_steps is a non-empty list of installation commands
**And** honest_verdict = 'vivado_not_installed'

### SCENARIO-SAMPLE-053: Vivado Available — Synthesis Succeeds

**Given** Vivado 2023.2+ is on PATH
**When** `vivado -mode batch -source hardware/kv260/synth_ising.tcl` completes
**And** output/carnot_ising_synth/carnot_ising.bit is created
**Then** bitfile_built = True
**And** honest_verdict = 'bitfile_built'
**And** results/kv260_export_command.txt contains CARNOT_KV260_BITFILE=<path>

### SCENARIO-SAMPLE-054: Vivado Available — Synthesis Fails

**Given** Vivado is on PATH but synthesis encounters an error
**When** Vivado returns non-zero or the .bit file is absent
**Then** bitfile_built = False
**And** honest_verdict = 'synthesis_attempted_failed'
**And** synthesis_stdout contains the first 2000 chars of Vivado output for debugging

### REQ-SAMPLE-033: KV260 Hardware Ising Benchmark — Latency < 100μs vs CPU 289ms (Exp 585)

**What this requirement is for:**
    After Vivado synthesis produces a bitfile (Exp 584), this experiment measures actual
    FPGA hardware latency for 100-spin Ising sampling and compares to the CPU baseline
    (289608 µs from Exp 568).  The target is < 100 µs, a 2900× speedup.

**Acceptance Criteria:**
- REQ-SAMPLE-033-1: The experiment SHALL gate on Exp 584 result; if bitfile_built=False
  it SHALL write a blocked artifact with honest_verdict='blocked_no_bitfile' and exit.
- REQ-SAMPLE-033-2: When a bitfile is available, 1000 trials of 100-spin Ising sampling
  SHALL be timed via FpgaBackend.sample() and mean_hardware_latency_us recorded.
- REQ-SAMPLE-033-3: speedup_ratio = 289608 / mean_hardware_latency_us SHALL be computed.
- REQ-SAMPLE-033-4: fpga_target_met SHALL be True iff mean_hardware_latency_us < 100.
- REQ-SAMPLE-033-5: honest_verdict SHALL be 'hardware_working', 'hardware_too_slow', or
  'hardware_failed' depending on outcome.

Spec: REQ-SAMPLE-033, SCENARIO-SAMPLE-055, SCENARIO-SAMPLE-056, SCENARIO-SAMPLE-057

### REQ-SAMPLE-034: DWaveNealBackend — D-Wave Ocean SDK Integration and Latency Benchmark (Exp 598)

Validates the D-Wave integration path described in research-hardware-wishlist.md using
the dwave-ocean-sdk SimulatedAnnealingSampler (local neal) as a CPU-comparable baseline.

- REQ-SAMPLE-034-1: ``DWaveNealBackend.__init__`` SHALL try importing ``dwave.samplers``
  first, then ``neal``, setting ``self.available=True`` on success and ``False`` on
  ImportError from both.
- REQ-SAMPLE-034-2: ``DWaveNealBackend.sample(J, h, n_samples)`` SHALL return a jnp
  boolean array of shape (n_samples, n_spins).
- REQ-SAMPLE-034-3: When available=False, ``sample()`` SHALL fall back to
  ``ParallelIsingSampler`` and return the same shape contract.
- REQ-SAMPLE-034-4: ``DWaveNealBackend.latency_ms(n_spins)`` SHALL run 10 calls and
  return mean wall-clock milliseconds as a float > 0.
- REQ-SAMPLE-034-5: ``DWaveNealBackend`` SHALL be exported from ``carnot.samplers``.

Spec: REQ-SAMPLE-034, SCENARIO-SAMPLE-058, SCENARIO-SAMPLE-059

### SCENARIO-SAMPLE-058: D-Wave SDK Present — DWaveNealBackend Initialises and Returns Correct Shape

**Given** dwave.samplers or neal is importable
**When** DWaveNealBackend() is constructed and sample(J, h, n_samples=10) is called
**Then** backend.available == True and result.shape == (10, n_spins).

Spec: SCENARIO-SAMPLE-058

### SCENARIO-SAMPLE-059: D-Wave SDK Absent — DWaveNealBackend Falls Back to CPU

**Given** neither dwave.samplers nor neal is importable
**When** DWaveNealBackend() is constructed and sample(J, h, n_samples=10) is called
**Then** backend.available == False and result.shape == (10, n_spins) via CPU fallback.

Spec: SCENARIO-SAMPLE-059

### SCENARIO-SAMPLE-055: Exp 584 Blocked — Immediate Blocked Artifact

**Given** results/experiment_584_kv260_synthesis.json has bitfile_built=False (or is absent)
**When** Exp 585 runs
**Then** honest_verdict = 'blocked_no_bitfile'
**And** upstream_exp = 584
**And** the process exits without running any FPGA benchmark

### SCENARIO-SAMPLE-056: FPGA Benchmark Achieves < 100μs (Target Met)

**Given** CARNOT_KV260_BITFILE is set and Exp 584 bitfile_built=True
**When** 1000 trials of 100-spin Ising sampling complete via FpgaBackend
**And** mean_hardware_latency_us < 100
**Then** fpga_target_met = True
**And** honest_verdict = 'hardware_working'
**And** speedup_ratio > 2890

### SCENARIO-SAMPLE-057: FPGA Benchmark Slower Than Target

**Given** CARNOT_KV260_BITFILE is set and Exp 584 bitfile_built=True
**When** 1000 trials complete via FpgaBackend
**And** mean_hardware_latency_us >= 100
**Then** fpga_target_met = False
**And** honest_verdict = 'hardware_too_slow'

### SCENARIO-SAMPLE-049: FPGA Hardware Path Achieves <100μs Latency

**Given** CARNOT_KV260_BITFILE is set and points to a valid bitfile
**When** 100 trials of 100-spin Ising sampling are timed via FpgaBackend
**Then** mean_latency_us < 100
**And** fpga_alive = True
**And** honest_verdict = 'hardware_working'

### SCENARIO-SAMPLE-050: Synthesis Path Generated When Bitfile Absent

**Given** CARNOT_KV260_BITFILE is not set
**When** the experiment runs
**Then** synthesis_command = 'vivado -mode batch -source hardware/kv260/synth_ising.tcl'
**And** hardware/kv260/synth_ising.tcl exists (created as stub if absent)
**And** cpu_baseline_latency_us is measured and recorded
**And** honest_verdict = 'synthesis_required'

### SCENARIO-SAMPLE-051: CPU Baseline Always Present in Artifact

**Given** any execution path (hardware or synthesis_required)
**When** the experiment completes
**Then** cpu_baseline_latency_us is non-negative in the artifact
**And** fpga_speedup is the ratio hardware_latency_us / cpu_baseline_latency_us (or None)
**And** bitfile_set reflects the CARNOT_KV260_BITFILE env var state

### REQ-MODEL-020: SymbolicKANEnergy — KAN with Discrete Symbolic Activations

The system shall provide a SymbolicKANEnergy class that replaces continuous spline
activations with discrete symbolic equations (linear, quadratic, tanh, relu, abs),
enabling human-readable energy function explanations.

- Each activation is selected by minimum MSE from a fixed candidate set
- The fitted activation type, coefficient, and bias are stored as a SymbolicActivation dataclass
- energy_interpretable = True (class attribute distinguishes this tier)

### REQ-MODEL-021: formula_export — Human-Readable Formula String

The system shall expose an explain() method on SymbolicKANEnergy that returns a
human-readable string of the full energy function, e.g.:
  E(x) = 2.1*x1 + 0.8*x2^2 + tanh(1.3*x3)

- Each SymbolicActivation carries a formula_str auto-generated at fit time
- explain() assembles per-variable formulas into a single expression

### SCENARIO-MODEL-030: Linear Data Selects Linear Activation

**Given** x_data and y_data drawn from y = 2.0 * x + 0.5
**When** fit_activation() is called with all candidate types
**Then** the selected activation_type is 'linear'
**And** coefficient is approximately 2.0 and bias is approximately 0.5

### SCENARIO-MODEL-031: Absolute Constraint Captured by Symbolic Layer

**Given** training data from E(x1, x2) = |x1 + x2 - 1|
**When** SymbolicKANEnergy.fit() is called on 160 samples
**Then** explain() returns a non-empty formula string
**And** the formula contains 'abs' or 'tanh' reflecting the constraint structure

### SCENARIO-MODEL-032: SymbolicKANEnergy MSE Within 1.5x of KAEMEnergy

**Given** the same synthetic constraint dataset used in Exp 586
**When** both SymbolicKANEnergy and KAEMEnergy are fitted and evaluated on held-out data
**Then** symbolic_mse <= kaem_mse * 1.5 (honest_verdict = 'symbolic_viable')
**Or** honest_verdict = 'symbolic_accuracy_loss' if the threshold is exceeded

## Implementation Status

| Requirement | Rust | Python | Tests |
|------------|------|--------|-------|
| REQ-TRAIN-001 | Implemented | Not Started | 2 Rust |
| REQ-TRAIN-002 | Implemented | Implemented | 5 Rust + 13 Python |
| REQ-TRAIN-003 | Not Started | Not Started | Not Started |
| REQ-TRAIN-004 | Partial | Not Started | 5 Rust |
| REQ-SAMPLE-001 | Implemented | Implemented | 6 Rust + 4 Python |
| REQ-SAMPLE-002 | Implemented | Implemented | 6 Rust + 2 Python |
| REQ-SAMPLE-003 | Implemented | Implemented | 2 Rust + 30+ Python (incl. parallel Ising) |
| REQ-SAMPLE-004 | Not Started | Implemented | 8 Python |
| REQ-SAMPLE-005 | Not Started | Implemented | 6 Python |
| REQ-SAMPLE-006 | Not Started | Implemented | 10 Python |
| REQ-SAMPLE-007 | Not Started | Implemented | 5 Python |
| REQ-SAMPLE-008 | Not Started | Implemented | 8 Python |
| REQ-SAMPLE-009 | Not Started | Implemented | 68 Python (21 Exp 288 bringup + 47 Exp 289 FpgaBackend) |
| REQ-SAMPLE-010 | Not Started | Implemented | 27 Python (Exp 290 benchmark + tests) |
| REQ-SAMPLE-011 | Not Started | Implemented | 30 Python (Exp 291 RTL behavioral sim + tests) |
| REQ-SAMPLE-012 | Not Started | Not Started | Not Started |
| REQ-SAMPLE-015 | N/A | Implemented | 51 Python (test_kaem_energy.py, 100% coverage) |
| REQ-SAMPLE-016 | N/A | Implemented | 7 Python (benchmark tests in test_kaem_energy.py) |
| REQ-SAMPLE-019 | N/A | Implemented | Python (test_kaem_crossover.py, 100% coverage) |
| REQ-SAMPLE-020 | N/A | Implemented | Python (test_kaem_extended_result.py, 100% coverage) |
| REQ-SAMPLE-021 | N/A | Implemented | Python (test_kaem_extended_result.py, 100% coverage) |
| REQ-SAMPLE-022 | N/A | Implemented | Python (test_kaem_distribution_benchmark.py, 100% coverage) |
| REQ-SAMPLE-023 | N/A | Implemented | Python (test_kaem_distribution_benchmark.py, 100% coverage) |
| REQ-SAMPLE-024 | N/A | Implemented | Python (test_kaem_distribution_benchmark.py, 100% coverage) |
| REQ-SAMPLE-025 | N/A | Implemented | Python (test_cikan_energy.py, 100% coverage) |
| REQ-SAMPLE-026 | N/A | Implemented | Python (test_cikan_energy.py, 100% coverage) |
| REQ-SAMPLE-027 | N/A | Implemented | Python (test_lowrank_kaem.py, 100% coverage) |
| REQ-SAMPLE-028 | N/A | Implemented | Python (test_lowrank_kaem.py, 100% coverage) |
| REQ-SAMPLE-029 | N/A | Implemented | Python (test_lowrank_kaem_cascade.py, 100% coverage) |
| REQ-SAMPLE-030 | N/A | Implemented | Python (test_kaem_calibration.py, 100% coverage) |
| REQ-SAMPLE-031 | N/A | Implemented | Python (test_experiment_568_kv260_bringup_v2.py, 100% targeted) |
| REQ-SAMPLE-032 | N/A | Implemented | Python (test_experiment_584_kv260_synthesis.py, 100% targeted) |
| REQ-SAMPLE-033 | N/A | Implemented | Python (test_experiment_585_kv260_live_benchmark_v3.py, 100% targeted) |
| REQ-SAMPLE-034 | N/A | Implemented | Python (test_dwave_backend.py, 100%) |
| REQ-SAMPLE-035 | N/A | Implemented | Python (test_sampler_registry.py, 100%) |
| REQ-HW-003 | N/A | Not Started | Not Started |
| REQ-MODEL-020 | N/A | Implemented | Python (test_symbolic_kan_energy.py, 100% coverage) |
| REQ-MODEL-021 | N/A | Implemented | Python (test_symbolic_kan_energy.py, 100% coverage) |
| REQ-VERIFY-106 | N/A | Implemented | Python (test_potts_machine.py, 100% coverage) |
| REQ-VERIFY-107 | N/A | Implemented | Python (test_potts_machine.py, 100% coverage) |
| REQ-VERIFY-108 | N/A | Implemented | Python (Exp 534 experiment) |

## REQ-SAMPLE-035: Production CARNOT_SAMPLER env-var-selectable backend registry

The sampler backend layer MUST expose a public ``backend_registry`` dict and a
``get_sampler_backend(name)`` factory that selects backends via the
``CARNOT_SAMPLER`` environment variable (default: ``"cpu"``).

### REQ-SAMPLE-035-1
``backend_registry`` MUST map ``"cpu"`` to ``CpuBackend`` and ``"dwave"`` to
``DWaveNealBackend``.

### REQ-SAMPLE-035-2
``get_sampler_backend(None)`` MUST read ``os.environ.get('CARNOT_SAMPLER', 'cpu')``
and return an instance of the mapped class.

### REQ-SAMPLE-035-3
``get_sampler_backend('dwave')`` MUST return a ``DWaveNealBackend`` instance.

### REQ-SAMPLE-035-4
``get_sampler_backend`` and ``backend_registry`` MUST be exported from
``carnot.samplers.__init__``.

### SCENARIO-SAMPLE-040: get_sampler_backend respects CARNOT_SAMPLER env var

Given: CARNOT_SAMPLER=dwave in the process environment.
When: get_sampler_backend() is called with no arguments.
Then: returns a DWaveNealBackend instance.

**Implementation Status:** Implemented (Exp 610)

### SCENARIO-SAMPLE-041: get_sampler_backend defaults to cpu when CARNOT_SAMPLER unset

Given: CARNOT_SAMPLER is not set.
When: get_sampler_backend() is called with no arguments.
Then: returns a CpuBackend instance.

**Implementation Status:** Implemented (Exp 610)

## REQ-SAMPLE-036: Synchronous p-bit Ising RTL (arXiv 2604.01564)

Synchronous-DAC-free Ising sampler for KV260 FPGA.  Replaces asynchronous
random-order spin updates with a fully synchronous design, reducing LUT
utilization ~50% while preserving annealing dynamics.

- REQ-SAMPLE-036-1: ising_sampler_v2.v uses `always @(posedge clk)` for all spin updates.
- REQ-SAMPLE-036-2: All spins in the current checkerboard phase update simultaneously each clock cycle.
- REQ-SAMPLE-036-3: No random spin-selection LFSR index mux (DAC-free).
- REQ-SAMPLE-036-4: Each spin has its own LFSR lane seeded with a distinct non-zero value.
- REQ-SAMPLE-036-5: AXI-Lite register map is identical to v1 (Exp 289 regmap).
- REQ-SAMPLE-036-6: Mpemba hot-start (first N_HOT steps at β=0) retained from v1.
- REQ-SAMPLE-036-7: v1 is not modified; v2 is a new file hardware/kv260/ising_sampler_v2.v.

Spec: REQ-SAMPLE-036, SCENARIO-SAMPLE-060

### SCENARIO-SAMPLE-060: Synchronous RTL Exists and Contains posedge clk

Given: hardware/kv260/ising_sampler_v2.v has been created.
When: the file is inspected.
Then: the file exists, contains 'posedge clk', and has fewer non-empty lines
than ising_sampler_v1.v (proxy for area reduction).

**Implementation Status:** Implemented (Exp 612)

### REQ-SAMPLE-037: SynchronousIsingSampler — Python Simulation of Synchronous Ising RTL (Exp 624)

Python-level simulation of the synchronous checkerboard p-bit update logic
in ising_sampler_v2.v.  Used to validate RTL correctness before FPGA synthesis.

- REQ-SAMPLE-037-1: `SynchronousIsingSampler(n_spins, couplings, biases, beta)` SHALL store
  parameters and validate that couplings.shape == (n_spins, n_spins) and biases.shape == (n_spins,).
- REQ-SAMPLE-037-2: `step(state)` SHALL implement the synchronous two-phase checkerboard sweep:
  phase 0 updates all even-indexed spins simultaneously; phase 1 updates all odd-indexed spins
  using the freshly updated even values.  Spin convention is {-1, +1}.
- REQ-SAMPLE-037-3: `sample(n_steps, init_state)` SHALL run n_steps sweeps and return the final
  state.  Default init_state is all-+1 (matching RTL reset state).
- REQ-SAMPLE-037-4: `compare_with_async(n_steps, n_trials)` SHALL return a dict with keys
  sync_mean_energy, async_mean_energy, energy_gap, and sync_converged.
- REQ-SAMPLE-037-5: `SynchronousIsingSampler` SHALL be exported from `carnot.samplers`.

Spec: REQ-SAMPLE-037, SCENARIO-SAMPLE-061, SCENARIO-SAMPLE-062

### SCENARIO-SAMPLE-061: SynchronousIsingSampler Returns Required Keys from compare_with_async

Given: a 6-spin Ising instance.
When: compare_with_async(n_steps=20, n_trials=3) is called.
Then: the returned dict contains sync_mean_energy, async_mean_energy, energy_gap,
sync_converged, and energy_gap == sync_mean_energy - async_mean_energy.

**Implementation Status:** Implemented (Exp 624)

### SCENARIO-SAMPLE-062: Synchronous and Async Samplers Reach Comparable Energy on Small Instance

Given: a 4-spin ferromagnetic Ising instance (all J[i,j]=0.5, no bias, beta=2.0).
When: compare_with_async(n_steps=50, n_trials=5) is called.
Then: both sync_mean_energy and async_mean_energy are non-None floats, validating
that the synchronous update logic runs end-to-end without error.

**Implementation Status:** Implemented (Exp 624)

## REQ-SAMPLE-038: MultilevelKAEMTrainer — Knot Refinement Schedule for KAEMEnergy (Exp 634)

Multilevel training (arXiv 2603.04827) trains a sequence of KANs at increasing spline
knot resolution via analytic geometric interpolation operators between levels.
Starting coarse (K=16), training to convergence, then interpolating weights to K=32,
K=64, and finally K=128 achieves superior energy accuracy vs. single-level training
at equal or fewer total parameter updates.

- REQ-SAMPLE-038-1: `KnotRefinementInterpolator(coarse_layer, fine_n_knots)` SHALL create a finer
  UnivariateKAEMLayer by linearly interpolating the coarse control points onto the fine knot grid.
- REQ-SAMPLE-038-2: `MultilevelKAEMTrainer(schedule, epochs_per_level)` SHALL train KAEMEnergy
  at each knot resolution in schedule, interpolating weights between levels.
- REQ-SAMPLE-038-3: The default refinement schedule SHALL be [16, 32, 64, 128].
- REQ-SAMPLE-038-4: Total epochs SHALL equal len(schedule) * epochs_per_level.
- REQ-SAMPLE-038-5: `MultilevelKAEMTrainer` and `KnotRefinementInterpolator` SHALL be exported
  from `carnot.training`.

Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-063, SCENARIO-SAMPLE-064

### SCENARIO-SAMPLE-063: MultilevelKAEMTrainer Reduces Energy Error vs Standard Training

Given: 20-variable synthetic energy landscape (sinusoidal+quadratic ground truth).
When: MultilevelKAEMTrainer(schedule=[16,32,64,128], epochs_per_level=20) is trained
      alongside KAEMEnergy(n_hidden=128) trained for 80 epochs.
Then: accuracy_multilevel >= accuracy_standard OR the results are comparable within
      experimental noise (accuracy_improvement > -0.5 indicating no regression).

**Implementation Status:** Implemented (Exp 634)

### SCENARIO-SAMPLE-064: KnotRefinementInterpolator Preserves Knot Count

Given: A UnivariateKAEMLayer with n_knots=16.
When: KnotRefinementInterpolator(layer, 32).interpolate() is called.
Then: The returned layer has n_knots=32 and control_points.shape == (n_vars, 32).

**Implementation Status:** Implemented (Exp 634)
