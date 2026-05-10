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
