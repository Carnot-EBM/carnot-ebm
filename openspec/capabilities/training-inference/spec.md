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

### SCENARIO-TRAIN-003: Checkpoint Round-Trip

**Given** a model trained for N steps
**When** a checkpoint is saved and loaded
**Then** training can resume from step N with identical optimizer state
**And** the next gradient step produces identical results

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
| REQ-SAMPLE-009 | Not Started | Implemented | 6 Python |
