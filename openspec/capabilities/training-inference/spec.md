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
