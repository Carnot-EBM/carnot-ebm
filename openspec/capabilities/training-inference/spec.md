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
| REQ-HW-003 | N/A | Not Started | Not Started |
