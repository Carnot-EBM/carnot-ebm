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

### REQ-HASHI-001: WOPR Hashi Ising Cartridge

The system shall expose a WOPR Hashi cartridge where:
- each possible horizontal or vertical bridge between adjacent visible islands is encoded as one spin
- spin +1 means the bridge is present and spin -1 means the bridge is absent
- island degree penalties are zero exactly when every island has its requested number of bridges
- crossing penalties are positive when orthogonal bridges share an empty grid cell
- the canonical 5x5 puzzle converges to a connected zero-energy solution

### REQ-SLITHERLINK-001: WOPR Slitherlink Ising Cartridge

The system shall expose a WOPR Slitherlink cartridge where:
- each horizontal or vertical dot-grid edge is encoded as one spin
- spin +1 means the edge is in the loop and spin -1 means the edge is absent
- clue penalties are zero exactly when every clued cell has its requested number of adjacent loop edges
- dot degree penalties are zero exactly when every dot has degree 0 or degree 2
- the all-off state receives a positive empty-loop penalty
- the canonical 3x3 diamond puzzle has 24 spins and converges to a connected zero-energy loop

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

### REQ-HW-035: DualGPU GPU1 Compute Utilization Confirmed via pynvml (Exp 684)

The system shall verify true parallel GPU compute by measuring GPU1 compute utilization
directly via pynvml during simultaneous inference on both GPUs:

- pynvml is installed (via pip if absent) and nvmlDeviceGetUtilizationRates() is called
  every 2 seconds on GPU1 handle during the parallel inference window
- max_gpu1_util_pct records the peak GPU1 compute utilization observed across all samples
- honest_verdict is "dualgpu_confirmed" if and only if max_gpu1_util_pct > 0 AND
  GPU1 inference completed (proves real parallel compute, not just memory allocation)
- honest_verdict is "dualgpu_partial_no_pynvml" if GPU1 inference completed but
  pynvml installation failed (throughput evidence only, not utilization proof)
- honest_verdict is "dualgpu_blocked" if fewer than 2 GPUs or CARNOT_FORCE_LIVE not set
- retro_071_resolved is True if and only if honest_verdict == "dualgpu_confirmed"
- Artifact schema: experiment=684, pynvml_installed, max_gpu0_util_pct, max_gpu1_util_pct,
  throughput_ratio, retro_071_resolved, honest_verdict

### SCENARIO-HW-035: GPU1 Utilization Greater Than Zero During Parallel Inference

**Given** two NVIDIA GPUs are present and CARNOT_FORCE_LIVE=1 is set
**And** pynvml is installed and nvmlDeviceGetUtilizationRates() is callable
**When** Qwen3.5-0.8B runs simultaneously on cuda:0 and cuda:1 via ThreadPoolExecutor
**And** GPU1 utilization is polled every 2 seconds during inference
**Then** max_gpu1_util_pct is greater than 0
**And** honest_verdict is "dualgpu_confirmed"
**And** retro_071_resolved is True in the artifact

### REQ-HW-036: DualGPU EORM+JEPA Parallel Retrain Completes in Under 40 Minutes (Exp 685)

The system shall retrain EORMModel on cuda:0 and ContextPredictionEnergy (JEPA v15)
on cuda:1 simultaneously via ThreadPoolExecutor(max_workers=2), completing the combined
parallel retrain in under 40 minutes wall-clock time:

- EORM trains for 50 epochs on the fover_labeled_steps_live.json contrastive pairs
- JEPA retrains for 100 epochs on violation pairs derived from the same data using JEPARetrainer
- Sequential baseline is measured first (EORM then JEPA on the same device) to compute speedup
- speedup = sequential_total_s / parallel_total_s; honest_verdict reflects the measured speedup
- Retrained weights are saved to results/eorm_v2_dualgpu.safetensors and
  results/jepa_predictor_v15_1_dualgpu.safetensors
- GATE: experiment reads results/experiment_684_dualgpu_pynvml.json and checks
  retro_071_resolved == True before proceeding; writes a blocked artifact if the gate fails

**Implementation Status:** Implemented (Exp 685)

### REQ-LEARN-044: JEPA v15.1 OOD AUC Maintained After Parallel Retrain (Exp 685)

The ContextPredictionEnergy model retrained in the DualGPU parallel experiment SHALL
produce a finite training loss after 100 retraining epochs on the live violation pairs,
confirming the model continued to learn without divergence:

- jepa_loss after parallel retrain shall be a finite float (not NaN or inf)
- The retrain shall complete within the parallel training window without timeout
- Weights shall be persisted to results/jepa_predictor_v15_1_dualgpu.safetensors

**Implementation Status:** Implemented (Exp 685)

### SCENARIO-HW-036: DualGPU Parallel Retrain Produces Valid Weights and Speedup >= 1.3

**Given** Exp 684 has retro_071_resolved=True (two GPUs confirmed usable)
**And** fover_labeled_steps_live.json has at least 10 labeled step records
**When** EORMTrainer runs 50 epochs on cuda:0 and JEPARetrainer runs 100 epochs on cuda:1
  simultaneously via ThreadPoolExecutor(max_workers=2)
**Then** eorm_v2_dualgpu.safetensors is written to the results directory
**And** jepa_predictor_v15_1_dualgpu.safetensors is written to the results directory
**And** speedup >= 1.3 on a DualGPU host (honest_verdict == "dualgpu_retrain_success")
**And** both eorm_loss and jepa_loss are finite floats

### SCENARIO-LEARN-074: JEPA v15.1 Loss Is Finite After Parallel Retrain

**Given** the JEPA v15 ContextPredictionEnergy model is loaded and fine-tuned for 100 epochs
**When** training runs on violation pairs derived from fover_labeled_steps_live.json
**Then** the final jepa_loss is a finite float (math.isfinite(jepa_loss) == True)
**And** no NaN gradients or divergence occurs during training

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
| REQ-HASHI-001 | N/A | Implemented | 4 Python |
| REQ-SLITHERLINK-001 | N/A | Implemented | 5 Python |
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
| REQ-HW-035 | N/A | Implemented | Python (test_experiment_684_dualgpu_pynvml.py, 100% targeted) |
| REQ-HW-036 | N/A | Implemented | Python (test_experiment_685_dualgpu_eorm_jepa.py, 100% targeted) |
| REQ-LEARN-044 | N/A | Implemented | Python (test_experiment_685_dualgpu_eorm_jepa.py, 100% targeted) |
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

## REQ-SAMPLE-039: synth_ising_v2.tcl — Vivado Synthesis Script Targeting ising_sampler_v2.v (Exp 636)

hardware/kv260/synth_ising_v2.tcl SHALL be a Vivado batch-mode synthesis script that
targets ising_sampler_v2.v (module ising_sampler_128_sync, synchronous p-bit design)
rather than ising_sampler_v1.v.  The script is derived from synth_ising.tcl with all
references updated for v2.

- REQ-SAMPLE-039-1: rtl_files SHALL list hardware/kv260/ising_sampler_v2.v (not v1).
- REQ-SAMPLE-039-2: top_module SHALL be ising_sampler_128_sync (v2 module name).
- REQ-SAMPLE-039-3: output_dir SHALL be output/carnot_ising_synth_v2 (separate from v1).
- REQ-SAMPLE-039-4: The bitfile path SHALL be output/carnot_ising_synth_v2/carnot_ising_v2.bit.
- REQ-SAMPLE-039-5: A comment block at the top SHALL identify the file as targeting v2 and
  reference Exp 636.

**Implementation Status:** Implemented (Exp 636)

### SCENARIO-SAMPLE-065: synth_ising_v2.tcl References v2 Module and File

Given: The file hardware/kv260/synth_ising_v2.tcl exists.
When: The file content is inspected for module name and RTL source path.
Then: top_module contains "ising_sampler_128_sync", rtl_files contains "ising_sampler_v2.v",
and output_dir contains "carnot_ising_synth_v2".

**Implementation Status:** Implemented (Exp 636)

## REQ-SAMPLE-021: SparseKAEMEnergy — Full-Rank KAEM with Top-K Sparse Coupling Graph (Exp 637)

SparseKAEMEnergy SHALL implement a full-rank KAEM energy model where each variable retains
its own univariate spline term (same as KAEMEnergy) PLUS a sparse pairwise coupling matrix
that retains only the top-K interactions per variable by coupling strength.

- REQ-SAMPLE-021-1: energy(x) SHALL return univariate_energy + 0.5 * x^T * C_sparse * x where C_sparse is the sparsified coupling matrix.
- REQ-SAMPLE-021-2: sparsify(couplings) SHALL zero out all but the top_k largest-magnitude entries per row of the coupling matrix.
- REQ-SAMPLE-021-3: fit(data, n_epochs) SHALL train both univariate splines and the coupling matrix, applying sparsification after each coupling update.
- REQ-SAMPLE-021-4: top_k = max(1, int(n_vars * top_k_fraction)); default top_k_fraction=0.1.
- REQ-SAMPLE-021-5: SparseKAEMEnergy SHALL be exported from carnot.models.

**Implementation Status:** Implemented (Exp 637)

## REQ-SAMPLE-022: SparseKAEMEnergy — Energy Accuracy Within 5% vs Dense Baseline (Exp 637)

When trained on the same synthetic 20-variable energy landscape (sin(3*x_i) + x_i^2),
SparseKAEMEnergy with top_k_fraction=0.1 SHALL achieve energy MAE within 5% relative
error compared to the dense KAEMEnergy baseline (|sparse_mae - dense_mae| / dense_mae < 0.05
or equivalently sparse_vs_dense_error < 0.05).

**Implementation Status:** Implemented (Exp 637)

### SCENARIO-SAMPLE-035: SparseKAEMEnergy Energy Within 5% vs Dense at Default top_k_fraction

Given: 20-variable synthetic energy landscape (ground truth: sum_i sin(3*x_i) + x_i^2).
When: SparseKAEMEnergy(n_vars=20, n_knots=64, top_k_fraction=0.1) is trained with
      MultilevelKAEMTrainer for the same epoch budget as the dense baseline.
Then: sparse_vs_dense_error < 0.05 (retro_057_resolved=True).

**Implementation Status:** Implemented (Exp 637)

### SCENARIO-SAMPLE-036: SparseKAEMEnergy Sparsify Retains top_k Entries Per Row

Given: A coupling matrix of shape (n_vars, n_vars) with distinct absolute values.
When: sparsify(coupling) is called with top_k=2.
Then: Each row has at most top_k non-zero entries, corresponding to the largest magnitudes.

**Implementation Status:** Implemented (Exp 637)

## REQ-SAMPLE-040: Extropic Z1/XTR-0 THRML integration packet and backend stub

Carnot SHALL provide an Extropic early-access integration packet and a
`ThrmlSamplerBackend` adapter stub that preserve the existing `SamplerBackend`
shape while Z1/XTR-0 hardware and SDK access are unavailable.

Acceptance criteria:
- `docs/hardware/extropic_integration_packet.md` documents the minimal Carnot
  EBM workload that Z1 must support: Ising sampling, spin state read/write, and
  energy evaluation.
- The packet documents `ThermoSamplerBackend` interface requirements,
  acceptance gates `KL(Z1 || CPU_Gibbs) < 0.05` and latency `< 1ms` for 128
  spins, and a fallback plan using THRML CPU simulation when Z1 is unavailable.
- `python/carnot/samplers/thrml_backend.py` exposes `ThrmlSamplerBackend` as a
  `SamplerBackend` implementation. When `CARNOT_TSU_DEVICE` is unset it SHALL
  route `sample()` and `minimize_energy()` to the CPU Gibbs backend. When
  `CARNOT_TSU_DEVICE` is set it SHALL raise `NotImplementedError` rather than
  pretending to access hardware.
- `scripts/experiment_1150_extropic_integration_packet.py` SHALL probe THRML
  availability, run a 100-spin THRML parity benchmark only when import succeeds,
  and emit `results/experiment_1150_extropic_integration_packet.json` with the
  roadmap-required schema and honest verdict vocabulary.

**Implementation Status:** Implemented (Exp 1150)

### SCENARIO-SAMPLE-066: THRML backend falls back honestly without TSU hardware

Given: `CARNOT_TSU_DEVICE` is unset.
When: `ThrmlSamplerBackend.sample()` or `minimize_energy()` is called with a
small Ising problem.
Then: The backend returns boolean samples from the CPU Gibbs fallback and reports
the fallback execution path honestly.

**Implementation Status:** Implemented (Exp 1150)

### SCENARIO-SAMPLE-067: TSU hardware path is explicit until SDK access exists

Given: `CARNOT_TSU_DEVICE` is set to a requested Z1/XTR-0 device label.
When: `ThrmlSamplerBackend.sample()` or `minimize_energy()` is called.
Then: The backend raises `NotImplementedError` with a message naming the missing
Extropic TSU hardware/SDK integration.

**Implementation Status:** Implemented (Exp 1150)

### SCENARIO-SAMPLE-068: Extropic packet artifact records THRML availability

Given: Experiment 1150 runs in an environment with or without importable THRML.
When: `scripts/experiment_1150_extropic_integration_packet.py` completes.
Then: The deliverable JSON records THRML availability, packet/backend paths,
whether the interface is documented, optional THRML latency, and an honest
verdict from the approved vocabulary.

**Implementation Status:** Implemented (Exp 1150)

## REQ-SAMPLE-041: THRML compatibility parity audit for tiny Ising/KAN cases

Carnot SHALL provide a local, no-install THRML compatibility audit for the tiny
Ising and KAN energy cases used by the hardware portability track. The audit
MUST probe whether a local THRML Python import is available without network
access or package installation, attempt parity mapping only when the import and
the needed local THRML APIs are present, and refuse TSU hardware claims unless a
local THRML-backed parity run actually executes and measures energy parity.

Acceptance criteria:
- `results/experiment_1347_thrml_compatibility_parity_audit.json` includes
  `status`, `thrml_import_available`, `cases_attempted`,
  `energy_parity_max_abs_error`, `sample_quality_proxy`,
  `missing_api_or_dependency`, `tsu_mapping_notes`, `hardware_claim_allowed`,
  and `honest_verdict`.
- The audit records `run_date="20260505"` and the project root used for the
  task in artifact metadata.
- When THRML is not importable, the artifact records the missing dependency,
  leaves parity metrics unset, records mapping notes only, sets
  `hardware_claim_allowed=false`, and completes with an honest blocked verdict.
- When THRML is importable and exposes the required Ising mapping APIs, the
  audit compares the local tiny Ising energy against the THRML-represented
  energy and records `energy_parity_max_abs_error`.
- `hardware_claim_allowed=true` is valid only when a local THRML-backed run
  executes and parity is measured; THRML CPU/JAX simulation alone is not a TSU
  hardware execution claim.

**Implementation Status:** Implemented (Exp 1347)

### SCENARIO-SAMPLE-069: Exp 1347 writes honest THRML compatibility audit

Given: THRML may or may not be importable on the local host.
When: the Exp 1347 compatibility audit runs without installing packages.
Then: the artifact records import availability, tiny Ising/KAN case attempts,
parity metrics only when measured, missing APIs or dependencies otherwise, and
an honest verdict that disallows hardware claims unless local THRML-backed
parity execution occurred.

**Implementation Status:** Implemented (Exp 1347)

## REQ-SAMPLE-042: Exp 1477 THRML/NPIM simulator-only Ising microprobe

Carnot SHALL provide a bounded simulator-only microprobe that checks local
THRML import availability, compares Carnot sampler output against exact CPU
reference energy on two or three tiny Ising cases, attempts THRML energy parity
only when an already-available local THRML API imports cleanly, and evaluates a
compact NPIM-style learned-update-rule heuristic on the same toy cases without
claiming Extropic hardware access.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1477_thrml_npim_simulator_parity_microprobe.json` with
  `status="in_progress"` before probe completion.
- The terminal artifact SHALL include `status`, `thrml_available`,
  `carnot_sampler_cases`, `thrml_parity_cases`, `parity_metric`,
  `npim_probe_attempted`, `npim_energy_delta`,
  `npim_time_to_energy_delta`, `hardware_claim_allowed`, `simulator_only`,
  `blockers`, and `honest_verdict`.
- The THRML probe SHALL use local imports only and SHALL NOT install packages
  or infer hardware access from a CPU/JAX simulator.
- If THRML is not importable, the artifact SHALL record the import blocker,
  leave `thrml_parity_cases` empty, keep the parity metric unset or blocked,
  and still preserve `hardware_claim_allowed=false`.
- Carnot sampler rows SHALL use tiny Ising cases with exact enumerated CPU
  reference energy available.
- The NPIM-style probe, when attempted, SHALL report energy and time-to-energy
  deltas against a fixed local update baseline and label the result as a
  simulator heuristic only.
- `hardware_claim_allowed=false` and `simulator_only=true` SHALL remain set
  unless the same run records authenticated Extropic TSU hardware execution,
  which Exp 1477 is not expected to perform.

**Implementation Status:** Planned (Exp 1477)

### SCENARIO-SAMPLE-070: Exp 1477 writes honest simulator parity microprobe

Given: THRML may or may not be importable on the local host.
When: the Exp 1477 microprobe runs on the tiny Ising cases.
Then: the artifact records local Carnot sampler/reference rows, records THRML
parity rows only when THRML imports and exposes the required Ising APIs, records
the NPIM-style simulator heuristic deltas when attempted, and disallows all
hardware claims.

**Implementation Status:** Planned (Exp 1477)

## REQ-SAMPLE-043: Exp 1488 THRML installability/import terminal preflight

Carnot SHALL provide a bounded terminal preflight for the THRML simulator lane
that probes whether the active Python environment can import `thrml`, records
the observed version or import failure, and attempts only a non-mutating
installability check when THRML is not already importable.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1488_thrml_installability_import_preflight.json` with
  `status="in_progress"` before probe completion.
- The terminal artifact SHALL include `status`, `thrml_preflight_complete`,
  `thrml_import_ready`, `thrml_version`, `import_error`,
  `install_probe_attempted`, `install_probe_result`,
  `simulator_lane_allowed`, `hardware_claim_allowed`,
  `next_task_gate_value`, and `honest_verdict`.
- The import probe SHALL execute `python3 -c "import thrml"` or an equivalent
  active-environment Python import command without masking the return code,
  stderr, or exception text.
- If THRML is not importable, the installability probe SHALL be bounded and
  non-mutating, such as a pip dry-run/no-deps check, and SHALL NOT install or
  upgrade project dependencies.
- `thrml_import_ready=true` is valid only when the active-environment Python
  import succeeds.
- `hardware_claim_allowed=false` SHALL be set regardless of import result
  because THRML importability is simulator readiness, not Extropic TSU hardware
  access.
- `simulator_lane_allowed=true` is valid after the preflight completes because
  the simulator lane may continue with honest import-readiness gating and no
  hardware claim.

**Implementation Status:** Implemented (Exp 1488)

### SCENARIO-SAMPLE-071: Exp 1488 writes honest terminal readiness artifact

Given: THRML may or may not be importable in the active Python environment.
When: the Exp 1488 preflight runs from the Carnot project root for run date
20260507.
Then: the artifact records import readiness, version or import error,
bounded installability probe result when needed, simulator-lane allowance, and
disallows hardware claims regardless of THRML import result.

**Implementation Status:** Implemented (Exp 1488)

## REQ-SAMPLE-044: Exp 1503 THRML import readiness repair gate

Carnot SHALL provide a bounded THRML readiness repair gate that reproduces the
Exp 1488 import check, optionally repairs only the local project Python
environment when the blocker is a missing `thrml` package, records THRML
version/path/provenance on success, and never upgrades import readiness into a
TSU hardware or simulator parity claim.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1503_thrml_import_readiness_repair_gate.json` with
  `status="in_progress"` before probing THRML.
- The terminal artifact SHALL include `status`, `thrml_import_ready`,
  `import_error`, `repair_attempted`, `repair_actions`, `thrml_version`,
  `thrml_import_path`, `compatibility_probe_passed`,
  `parity_followup_allowed`, `hardware_claim_allowed`, `blockers`, and
  `honest_verdict`.
- The artifact metadata SHALL preserve the exact Exp 1488-style import command
  result, including stderr traceback on failure or success details on success.
- A mutating repair SHALL be attempted only when the active project virtualenv
  cannot import `thrml` because the package is missing; the repair SHALL use
  the virtualenv Python executable and SHALL NOT mutate global Python
  toolchains.
- If THRML remains unavailable after the bounded repair path, the artifact
  SHALL set `thrml_import_ready=false`, record a terminal blocker, and keep
  `parity_followup_allowed=false`.
- If THRML imports, the experiment SHALL run only a minimal local compatibility
  probe that inspects package metadata/path and public module surfaces without
  running Carnot/THRML parity or TSU hardware behavior.
- `parity_followup_allowed` SHALL equal `thrml_import_ready`, and
  `hardware_claim_allowed=false` SHALL be set regardless of import result.

**Implementation Status:** Implemented (Exp 1503)

### SCENARIO-SAMPLE-072: Exp 1503 writes terminal THRML readiness gate

Given: Exp 1488 previously reported `thrml_import_ready=false` for run date
20260507.
When: the Exp 1503 readiness repair gate runs from the Carnot project root.
Then: the artifact records the initial import result, any local virtualenv
repair action, THRML version/path/provenance when available, compatibility
probe status, parity follow-up gate, and a terminal no-hardware-claim verdict.

**Implementation Status:** Implemented (Exp 1503)

## REQ-SAMPLE-045: Exp 1504 gated THRML/Carnot simulator parity v3

Carnot SHALL provide a gated simulator-only THRML/Carnot parity audit that runs
only when Exp 1503 reports `thrml_import_ready=true`, compares the same tiny
discrete Ising energy/stochastic case through Carnot's local simulator and the
available THRML software API, records explicit numerical tolerances, and never
upgrades software parity into an Extropic TSU hardware claim.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1504_thrml_carnot_simulator_parity_v3.json` with
  `status="in_progress"` before gate inspection or parity execution completes.
- The experiment SHALL read
  `results/experiment_1503_thrml_import_readiness_repair_gate.json` and SHALL
  run parity only when `thrml_import_ready=true`; otherwise it SHALL write a
  terminal gated artifact with `parity_experiment_ran=false`.
- The terminal artifact SHALL include `status`, `parity_experiment_ran`,
  `thrml_import_ready`, `simulator_only`, `cases_compared`,
  `parity_pass_count`, `parity_fail_count`, `tolerance`,
  `max_observed_delta`, `hardware_claim_allowed`, `blockers`, and
  `honest_verdict`.
- The parity run SHALL use fixed seeds and explicit tolerances for both exact
  energy parity and stochastic sample-summary parity on the same tiny Ising
  case.
- The artifact SHALL record case-level Carnot outputs, THRML outputs, observed
  deltas, tolerance values, pass/fail status, and any THRML API limitation that
  prevents execution.
- `simulator_only=true` and `hardware_claim_allowed=false` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1504)

### SCENARIO-SAMPLE-073: Exp 1504 writes honest gated simulator parity artifact

Given: Exp 1503 may or may not report `thrml_import_ready=true`.
When: the Exp 1504 simulator parity audit runs from the Carnot project root for
run date 20260507.
Then: it either writes a terminal gated artifact without running parity, or it
compares the tiny Carnot and THRML software simulator paths with fixed seeds,
records tolerance-bounded case deltas, and disallows all hardware claims.

**Implementation Status:** Implemented (Exp 1504)

## REQ-SAMPLE-046: Exp 1515 THRML SamplerBackend simulator conformance pack

Carnot SHALL provide a simulator-only THRML SamplerBackend conformance pack
that is gated on Exp 1506 `prior_thrml_parity_ready=true`, verifies THRML
import readiness only in the active local Python environment, records adapter
API boundaries, seed/reproducibility behavior, sample-shape expectations, and
Exp 1504 Carnot/THRML parity vectors, and never claims Extropic TSU hardware
execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1515_thrml_samplerbackend_conformance_pack.json` with
  `status="in_progress"` before gate inspection, THRML import checks, or
  conformance execution completes.
- The experiment SHALL read
  `results/experiment_1506_115_completion_archive_116_activation.json` and
  SHALL write a terminal gated artifact if
  `prior_thrml_parity_ready=true` is absent.
- The experiment SHALL import `thrml` only from the active local Python
  environment; if import fails, the terminal artifact SHALL report a simulator
  dependency blocker instead of installing packages or claiming hardware.
- The terminal artifact SHALL include `status`,
  `thrml_samplerbackend_conformance_ready`, `gated_inputs_present`,
  `thrml_import_ready`, `simulator_only`, `no_tsu_hardware_claim`,
  `conformance_cases`, `parity_cases_passed`, `sample_shape_contracts`,
  `seed_reproducibility_checked`, `conformance_manifest_path`, `blockers`, and
  `honest_verdict`.
- The conformance manifest at
  `results/thrml_samplerbackend_conformance_1515.jsonl` SHALL contain one JSON
  object per bounded conformance case covering accepted model shape,
  schedule/seed behavior, returned sample shapes, energy/parity fields, and
  provenance.
- `thrml_samplerbackend_conformance_ready=true` is valid only when conformance
  rows are written, sample-shape checks are reported, and at least one Exp 1504
  parity vector is carried forward as passed.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1515)

### SCENARIO-SAMPLE-074: Exp 1515 writes simulator-only conformance manifest

Given: Exp 1506 reports `prior_thrml_parity_ready=true` and local THRML import
is available.
When: the Exp 1515 conformance pack runs from the Carnot project root for run
date 20260508.
Then: it writes a JSONL manifest with adapter contract, seed reproducibility,
sample-shape, and Exp 1504 parity-vector rows, writes a terminal JSON artifact
with all required fields, marks the conformance pack ready only when the rows
and checks exist, and disallows all TSU hardware claims.

**Implementation Status:** Implemented (Exp 1515)

## REQ-SAMPLE-047: Exp 1526 exact n=8 THRML/Carnot simulator parity

Carnot SHALL provide an exact n=8 THRML/Carnot Ising parity run that stays
software-only, uses a deterministic signed ring-chord topology, enumerates all
256 states for partition/distribution comparisons when the local THRML API is
available, records fixed-seed sampling as a secondary check, and never claims
Extropic TSU hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1526_thrml_carnot_parity_n8.json` with
  `status="in_progress"` before THRML import probing or parity execution
  completes.
- The experiment SHALL verify local THRML import readiness from the active
  Python environment and SHALL write a simulator-only terminal blocker if the
  import or required software energy API is unavailable.
- The parity case SHALL use eight spins, explicit deterministic biases,
  symmetric zero-diagonal couplings, beta, topology, and fixed seed.
- Exact comparison SHALL enumerate 256 states and report Carnot and THRML
  partition functions, partition relative error, exact mean-energy delta, and
  KL divergence between the exact Boltzmann distributions.
- Fixed-seed sampling SHALL be recorded as secondary evidence in
  `results/thrml_carnot_parity_n8_1526.jsonl` without being treated as TSU
  hardware evidence.
- `thrml_parity_n8_passed=true` is valid only when the exact partition relative
  error, exact mean-energy delta, and exact KL divergence are all within the
  documented thresholds in the terminal artifact.
- The terminal artifact SHALL include `status`, `thrml_parity_n8_passed`,
  `simulator_only`, `no_tsu_hardware_claim`, `n_spins`,
  `exact_states_enumerated`, `topology`, `seed`,
  `carnot_partition_function`, `thrml_partition_function`,
  `partition_relative_error`, `mean_energy_delta`, `kl_divergence`,
  `parity_manifest_path`, `blockers`, and `honest_verdict`.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1526)

### SCENARIO-SAMPLE-075: Exp 1526 writes exact n=8 parity evidence

Given: Exp 1515 reports local THRML import readiness and no TSU hardware claim.
When: the Exp 1526 parity run executes from the Carnot project root for run
date 20260508.
Then: it writes JSONL rows for exact n=8 distribution metrics and fixed-seed
sampling metrics, writes a terminal artifact with all required fields, marks
the run passed only when exact metrics meet the documented thresholds, and
disallows all TSU hardware claims.

**Implementation Status:** Implemented (Exp 1526)

## REQ-SAMPLE-048: Exp 1527 exact n=16 THRML/Carnot simulator parity

Carnot SHALL provide an exact n=16 THRML/Carnot Ising parity run that reuses
the Exp 1526 simulator-parity helper pattern, stays software-only, uses a
deterministic signed ring-chord topology, enumerates all 65,536 states for
partition/distribution comparisons when the local THRML API is available,
records fixed-seed sampling as a secondary check, and never claims Extropic
TSU hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1527_thrml_carnot_parity_n16.json` with
  `status="in_progress"` before THRML import probing or parity execution
  completes.
- The experiment SHALL verify Exp 1526 n=8 parity passed before running the
  n=16 scale step and SHALL write a simulator-only terminal blocker if the
  upstream parity evidence is missing, malformed, or not passed.
- The parity case SHALL use 16 spins, explicit deterministic biases,
  symmetric zero-diagonal couplings, beta, topology, and fixed seed.
- Exact comparison SHALL enumerate 65,536 states and report Carnot and THRML
  partition functions, partition relative error, exact mean-energy delta, and
  KL divergence between the exact Boltzmann distributions.
- Fixed-seed sampling SHALL be recorded as secondary evidence in
  `results/thrml_carnot_parity_n16_1527.jsonl` without being treated as TSU
  hardware evidence.
- `thrml_parity_n16_passed=true` is valid only when the exact partition
  relative error, exact mean-energy delta, exact KL divergence, and secondary
  sample mean-energy delta are within the documented thresholds in the
  terminal artifact/manifest.
- The terminal artifact SHALL include `status`, `thrml_parity_n16_passed`,
  `simulator_only`, `no_tsu_hardware_claim`, `n_spins`,
  `exact_states_enumerated`, `topology`, `seed`,
  `partition_relative_error`, `mean_energy_delta`, `kl_divergence`,
  `parity_manifest_path`, `blockers`, and `honest_verdict`.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1527)

### SCENARIO-SAMPLE-076: Exp 1527 writes exact n=16 parity evidence

Given: Exp 1526 reports exact n=8 THRML/Carnot simulator parity passed and no
TSU hardware claim.
When: the Exp 1527 parity run executes from the Carnot project root for run
date 20260508.
Then: it writes JSONL rows for exact n=16 distribution metrics and fixed-seed
sampling metrics, writes a terminal artifact with all required fields, marks
the run passed only when exact and secondary sampling metrics meet the
documented thresholds, and disallows all TSU hardware claims.

**Implementation Status:** Implemented (Exp 1527)

## REQ-SAMPLE-049: Exp 1528 sampled n=32 THRML/Carnot simulator parity

Carnot SHALL provide a sampled n=32 THRML/Carnot Ising parity run that reuses
the Exp 1526/1527 simulator-parity helper pattern, stays software-only, uses a
deterministic signed ring-chord topology, replaces exact enumeration with
fixed-seed repeated-chain sampling, reports energy, magnetization,
autocorrelation, and distributional summaries, and never claims Extropic TSU
hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1528_thrml_carnot_parity_n32_sample.json` with
  `status="in_progress"` before upstream gate inspection, THRML import probing,
  or parity execution completes.
- The experiment SHALL verify Exp 1527 n=16 parity passed before running the
  n=32 scale step and SHALL write a simulator-only terminal blocker if the
  upstream parity evidence is missing, malformed, or not passed.
- The parity case SHALL use 32 spins, explicit deterministic biases, symmetric
  zero-diagonal couplings, beta, topology, repeated fixed seeds, warmup, sample
  count, and thinning parameters.
- The sampled comparison SHALL run Carnot and THRML software samplers with
  identical model parameters and comparable fixed-temperature schedules for
  each seed.
- The manifest at `results/thrml_carnot_parity_n32_1528.jsonl` SHALL contain
  one JSON object per seed/backend plus a summary row.
- The terminal artifact SHALL include `status`,
  `thrml_parity_n32_passed`, `simulator_only`, `no_tsu_hardware_claim`,
  `n_spins`, `topology`, `seeds`, `n_samples_per_backend`,
  `mean_energy_delta`, `magnetization_delta`, `autocorrelation_summary`,
  `kl_divergence`, `parity_manifest_path`, `blockers`, and `honest_verdict`.
- `thrml_parity_n32_passed=true` is valid only when the sampled mean-energy
  delta, magnetization delta, and KL divergence are all within the documented
  thresholds in the terminal artifact/manifest; KL gating SHALL be used only as
  an empirical sampled estimate, not an exact distribution claim.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1528)

### SCENARIO-SAMPLE-077: Exp 1528 writes sampled n=32 parity evidence

Given: Exp 1527 reports exact n=16 THRML/Carnot simulator parity passed and no
TSU hardware claim.
When: the Exp 1528 sampled parity run executes from the Carnot project root for
run date 20260508.
Then: it writes JSONL rows for each seed/backend and a summary row, writes a
terminal artifact with all required fields, marks the run passed only when
sampled energy, magnetization, and KL thresholds pass, and disallows all TSU
hardware claims.

**Implementation Status:** Implemented (Exp 1528)

## REQ-SAMPLE-050: Exp 1529 sampled n=64 THRML/Carnot simulator parity

Carnot SHALL provide a sampled n=64 THRML/Carnot Ising parity run that reuses
the Exp 1528 sampled-metrics helper pattern, stays software/simulator-only,
uses a deterministic signed ring-chord topology, reports repeated fixed-seed
energy, magnetization, autocorrelation, and sampled energy-histogram KL
metrics, and never claims Extropic TSU hardware execution, board execution,
synthesis, or bitstream evidence.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1529_thrml_carnot_parity_n64_sample.json` with
  `status="in_progress"` before upstream gate inspection, THRML import probing,
  or parity execution completes.
- The experiment SHALL verify Exp 1528 n=32 sampled parity passed before
  running the n=64 scale step and SHALL write a simulator-only terminal blocker
  if the upstream parity evidence is missing, malformed, or not passed.
- The parity case SHALL use 64 spins, explicit deterministic biases,
  symmetric zero-diagonal couplings, beta, topology, repeated fixed seeds,
  warmup, sample count, and thinning parameters.
- The sampled comparison SHALL run Carnot and THRML software samplers with
  identical model parameters and comparable fixed-temperature schedules for
  each seed.
- The manifest at `results/thrml_carnot_parity_n64_1529.jsonl` SHALL contain
  one JSON object per seed/backend plus a summary row.
- The terminal artifact SHALL include `status`,
  `thrml_parity_n64_passed`, `simulator_only`, `no_tsu_hardware_claim`,
  `n_spins`, `topology`, `seeds`, `n_samples_per_backend`,
  `mean_energy_delta`, `magnetization_delta`, `autocorrelation_summary`,
  `kl_divergence`, `parity_manifest_path`, `blockers`, and
  `honest_verdict`.
- `thrml_parity_n64_passed=true` is valid only when the sampled mean-energy
  delta is within the documented absolute or percent threshold, the sampled KL
  divergence is below the documented threshold, magnetization and stability
  diagnostics are present, and the terminal artifact records the thresholds
  used for that decision.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1529)

### SCENARIO-SAMPLE-078: Exp 1529 writes sampled n=64 parity evidence

Given: Exp 1528 reports sampled n=32 THRML/Carnot simulator parity passed and
no TSU hardware claim.
When: the Exp 1529 sampled parity run executes from the Carnot project root for
run date 20260508.
Then: it writes JSONL rows for each seed/backend and a summary row, writes a
terminal artifact with all required fields, marks the run passed only when
sampled energy, magnetization, KL, and stability gates pass, and disallows all
TSU hardware, board, synthesis, and bitstream claims.

**Implementation Status:** Implemented (Exp 1529)

## REQ-SAMPLE-051: Exp 1530 sampled n=128 THRML/Carnot production-scale simulator parity

Carnot SHALL provide a sampled n=128 THRML/Carnot Ising parity run that reuses
the Exp 1528/1529 sampled-metrics helper pattern, stays software/simulator-only,
uses a deterministic signed ring-chord topology at the production-scale spin
count, records repeated fixed-seed energy, magnetization, autocorrelation,
sampled energy-histogram KL, runtime, and memory diagnostics, and never claims
Extropic TSU hardware execution, acceleration, board execution, synthesis, or
bitstream evidence.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1530_thrml_carnot_parity_n128_production_scale.json` with
  `status="in_progress"` before upstream gate inspection, THRML import probing,
  or parity execution completes.
- The experiment SHALL verify Exp 1529 n=64 sampled parity passed before
  running the n=128 production-scale step and SHALL write a simulator-only
  terminal blocker if the upstream parity evidence is missing, malformed, or
  not passed.
- The parity case SHALL use 128 spins, explicit deterministic biases,
  symmetric zero-diagonal couplings, beta, topology, repeated fixed seeds,
  warmup, sample count, and thinning parameters chosen to keep conductor
  execution bounded.
- The sampled comparison SHALL run Carnot and THRML software samplers with
  identical model parameters and comparable fixed-temperature schedules for
  each seed.
- The manifest at `results/thrml_carnot_parity_n128_1530.jsonl` SHALL contain
  one JSON object per seed/backend plus a summary row.
- The terminal artifact SHALL include `status`,
  `thrml_parity_n128_passed`, `simulator_only`, `no_tsu_hardware_claim`,
  `n_spins`, `topology`, `seeds`, `n_samples_per_backend`,
  `mean_energy_delta`, `magnetization_delta`, `autocorrelation_summary`,
  `kl_divergence`, `runtime_seconds_by_backend`, `memory_summary`,
  `parity_manifest_path`, `blockers`, and `honest_verdict`.
- `thrml_parity_n128_passed=true` is valid only when the sampled mean-energy
  delta is within the documented absolute or 10 percent threshold, the sampled
  KL divergence is below the documented threshold, magnetization, stability,
  runtime, and memory diagnostics are present, and the terminal artifact
  records the thresholds used for that decision.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Planned (Exp 1530)

### SCENARIO-SAMPLE-079: Exp 1530 writes sampled n=128 production-scale parity evidence

Given: Exp 1529 reports sampled n=64 THRML/Carnot simulator parity passed and
no TSU hardware claim.
When: the Exp 1530 sampled parity run executes from the Carnot project root for
run date 20260508.
Then: it writes JSONL rows for each seed/backend and a summary row, writes a
terminal artifact with all required fields, marks the run passed only when
sampled energy, magnetization, KL, stability, runtime, and memory gates pass,
and disallows all TSU hardware, board, synthesis, acceleration, and bitstream
claims.

**Implementation Status:** Planned (Exp 1530)

## REQ-SAMPLE-052: Exp 1531 sampled n=32 diverse-topology THRML/Carnot parity

Carnot SHALL provide a sampled n=32 THRML/Carnot Ising parity run across
complete, sparse random, lattice, and scale-free graph families, reusing the
Exp 1528 sampled-metrics helper pattern, staying software/simulator-only,
recording per-topology pass/fail metrics, and never claiming Extropic TSU
hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1531_thrml_diverse_topology_parity_n32.json` with
  `status="in_progress"` before THRML import probing or parity execution
  completes.
- The experiment SHALL verify Exp 1528 n=32 sampled parity passed before
  running the diverse-topology step and SHALL write a simulator-only terminal
  blocker if that upstream parity evidence is missing, malformed, or not
  passed.
- The topology cases SHALL use 32 spins, fixed seeds where randomness is part
  of graph construction, deterministic biases, symmetric zero-diagonal
  couplings, beta, warmup, sample count, and thinning parameters.
- The sampled comparison SHALL run Carnot and THRML software samplers with
  identical model parameters and comparable fixed-temperature schedules for
  each topology and seed.
- The manifest at `results/thrml_diverse_topology_parity_n32_1531.jsonl`
  SHALL contain one JSON object per topology/backend/seed plus a summary row.
- The terminal artifact SHALL include `status`,
  `diverse_topology_parity_ready`, `simulator_only`,
  `no_tsu_hardware_claim`, `n_spins`, `topologies_tested`,
  `topologies_passed`, `topology_results`,
  `mean_energy_delta_by_topology`, `kl_divergence_by_topology`,
  `parity_manifest_path`, `blockers`, and `honest_verdict`.
- `diverse_topology_parity_ready=true` is valid only when at least three of
  the four documented topology families pass the sampled mean-energy,
  magnetization, KL, and stability gates recorded in the artifact.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome.

**Implementation Status:** Implemented (Exp 1531)

### SCENARIO-SAMPLE-080: Exp 1531 writes diverse-topology n=32 parity evidence

Given: Exp 1528 reports sampled n=32 THRML/Carnot simulator parity passed and
no TSU hardware claim.
When: the Exp 1531 diverse-topology parity run executes from the Carnot project
root for run date 20260508.
Then: it writes JSONL rows for each topology/backend/seed and a summary row,
writes a terminal artifact with all required fields, marks the run ready only
when at least three topology families pass documented thresholds, records any
negative topology result honestly, and disallows all TSU hardware claims.

**Implementation Status:** Implemented (Exp 1531)

## REQ-SAMPLE-053: Exp 1543 n=256 THRML/Carnot schedule-stress simulator parity

Carnot SHALL provide a sampled n=256 THRML/Carnot Ising parity stress run
across multiple fixed-temperature schedule variants, reusing the Exp 1530 n=128
and Exp 1531 diverse-topology sampled-metric tolerances unless a stronger
local reason is recorded, staying software/simulator-only, and never claiming
Extropic TSU, Z1, XTR-0, board, synthesis, bitstream, or hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1543_thrml_carnot_parity_n256_schedule_stress.json` with
  `status="in_progress"` before schedule generation, THRML import probing, or
  parity execution completes.
- The experiment SHALL load the Exp 1530 n=128 production-scale parity artifact
  and Exp 1531 diverse-topology parity artifact, SHALL reuse their documented
  sampled tolerances where applicable, and SHALL record those tolerance sources
  in the terminal artifact.
- The n=256 case SHALL use explicit deterministic biases, symmetric
  zero-diagonal couplings, sampled states, and at least three schedule variants
  that differ in beta, warmup, thinning, or checkerboard setting.
- The sampled comparison SHALL run Carnot and THRML software simulator lanes
  with identical model parameters and comparable schedules for each schedule
  variant. If the THRML lane is the local CPU fallback, the artifact SHALL
  label it as simulator fallback rather than hardware execution.
- The schedule-stress manifest SHALL contain one JSON object per
  schedule/backend plus a summary row, and each schedule row SHALL record its
  schedule identifier, sample count, mean energy, magnetization, lag-one energy
  autocorrelation, KL inputs, and simulator-only/no-TSU claim boundary.
- The terminal artifact SHALL include `status`, `milestone`,
  `thrml_parity_n256_schedule_ready`, `n_spins`, `schedules_tested`,
  `samples_per_schedule`, `mean_energy_delta`, `max_energy_delta`,
  `kl_divergence`, `autocorrelation_delta`, `parity_passed`,
  `simulator_only`, `no_tsu_hardware_claim`, `parity_report_path`,
  `focused_tests_passed`, and `honest_verdict`.
- `thrml_parity_n256_schedule_ready=true` and `parity_passed=true` are valid
  only when every schedule variant has both Carnot and THRML software rows, the
  aggregate sampled mean-energy, max-energy, KL, and autocorrelation deltas pass
  the recorded thresholds, the artifact records the manifest path, and
  `simulator_only=true` plus `no_tsu_hardware_claim=true`.

**Implementation Status:** Planned (Exp 1543)

### SCENARIO-SAMPLE-081: Exp 1543 writes n=256 schedule-stress parity evidence

Given: Exp 1530 reports sampled n=128 THRML/Carnot simulator parity passed,
Exp 1531 reports diverse-topology simulator parity ready, and no TSU hardware
claim is available.
When: the Exp 1543 n=256 schedule-stress run executes from the Carnot project
root for run date 20260508.
Then: it writes schedule/backend JSONL evidence plus a terminal artifact with
all required fields, marks schedule readiness true only when documented sampled
metric gates pass across the schedule variants, labels all execution as
software/simulator-only, and disallows all TSU, Z1, XTR-0, board, synthesis,
bitstream, and hardware claims.

**Implementation Status:** Planned (Exp 1543)

## REQ-SAMPLE-054: Exp 1544 sampled n=64 diverse-topology THRML/Carnot parity

Carnot SHALL provide a sampled n=64 THRML/Carnot Ising parity run across
complete, sparse random, lattice, and scale-free graph families after the Exp
1543 n=256 schedule-stress gate, staying software/simulator-only, recording
per-topology pass/fail metrics, and never claiming Extropic TSU, Z1, XTR-0,
board, synthesis, bitstream, or hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1544_thrml_diverse_topology_parity_n64.json` with
  `status="in_progress"` before graph generation, prior-artifact inspection,
  THRML import probing, or parity execution completes.
- The experiment SHALL load the Exp 1531 n=32 diverse-topology parity artifact
  and Exp 1543 n=256 schedule-stress artifact, SHALL require both to be
  complete, simulator-only, and no-TSU-claim, and SHALL write a simulator-only
  terminal blocker if either prerequisite is missing, malformed, or not ready.
- The topology cases SHALL use exactly 64 spins, complete, sparse random,
  lattice, and scale-free graph families, deterministic seeds where randomness
  is part of graph construction, deterministic biases, symmetric zero-diagonal
  couplings, beta, warmup, sample count, and thinning parameters.
- The sampled comparison SHALL run Carnot and THRML software simulator lanes
  with identical model parameters and comparable schedules for each topology
  and seed. If the THRML lane is the local CPU fallback, the artifact SHALL
  label it as simulator fallback rather than hardware execution.
- The manifest at `results/thrml_diverse_topology_parity_n64_1544.jsonl`
  SHALL contain one JSON object per topology/backend/seed plus a summary row,
  and each row SHALL record topology, sample count, mean energy,
  magnetization, KL inputs, and simulator-only/no-TSU claim boundary.
- The terminal artifact SHALL include `status`, `milestone`,
  `diverse_topology_parity_n64_ready`, `n_spins`, `topologies_tested`,
  `per_topology_results`, `mean_energy_delta`, `max_energy_delta`,
  `kl_divergence`, `parity_passed`, `simulator_only`,
  `no_tsu_hardware_claim`, `parity_report_path`, `focused_tests_passed`, and
  `honest_verdict`.
- `diverse_topology_parity_n64_ready=true` and `parity_passed=true` are valid
  only when all four topology families have Carnot and THRML software rows, at
  least three topology families pass the recorded sampled metric thresholds,
  aggregate mean-energy, max-energy, and KL deltas pass the recorded
  thresholds, the artifact records the manifest path, and
  `simulator_only=true` plus `no_tsu_hardware_claim=true`.

**Implementation Status:** Implemented (Exp 1544)

### SCENARIO-SAMPLE-082: Exp 1544 writes n=64 diverse-topology parity evidence

Given: Exp 1531 reports n=32 diverse-topology THRML/Carnot simulator parity
ready, Exp 1543 reports n=256 schedule-stress simulator parity ready, and no
TSU hardware claim is available.
When: the Exp 1544 n=64 diverse-topology parity run executes from the Carnot
project root for run date 20260508.
Then: it writes topology/backend JSONL evidence plus a terminal artifact with
all required fields, marks n=64 diverse-topology readiness true only when the
documented sampled metric gates pass across the topology families, labels all
execution as software/simulator-only, and disallows all TSU, Z1, XTR-0, board,
synthesis, bitstream, and hardware claims.

**Implementation Status:** Implemented (Exp 1544)

## REQ-SAMPLE-055: Exp 1545 Extropic Z1 access-readiness packet

Carnot SHALL provide an Extropic Z1/XTR-0 access-readiness packet after the
Exp 1543 n=256 schedule-stress and Exp 1544 n=64 diverse-topology simulator
parity artifacts exist. The packet SHALL transform software-only THRML/Carnot
parity evidence into benchmark manifests, transcript requirements, evidence
boundaries, and rollback criteria for a future authenticated device run, while
never claiming Extropic TSU, Z1, XTR-0, board, synthesis, bitstream, latency,
or hardware execution in the current run.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1545_extropic_z1_access_readiness_packet.json` with
  `status="in_progress"` before source-artifact inspection or packet rendering
  completes.
- The experiment SHALL read
  `results/experiment_1543_thrml_carnot_parity_n256_schedule_stress.json`,
  `results/experiment_1544_thrml_diverse_topology_parity_n64.json`,
  `research-hardware-wishlist.md`, `research-references.md`, and
  `ops/known-issues.md` before declaring readiness.
- The readiness packet at `ops/extropic_z1_readiness_packet.md` SHALL include
  a benchmark case list, seed/schedule/topology manifests, required device
  metadata, transcript schema, expected output checksum or metric fields,
  no-hardware-claim boundary language, explicit access blockers, and rollback
  criteria.
- A machine-readable transcript schema SHALL be written under `ops/` or
  `results/` and SHALL require device identity, SDK/THRML versions,
  authenticated access proof, benchmark case identifiers, latency fields,
  sample-shape fields, output checksum or metric fields, and claim-boundary
  fields.
- The terminal artifact SHALL include `status`, `milestone`,
  `extropic_z1_readiness_packet_ready`, `readiness_packet_path`,
  `benchmark_cases_included`, `transcript_schema_path`,
  `required_device_evidence_fields`, `no_hardware_execution_claim`,
  `simulator_artifacts_referenced`, `access_blockers`,
  `focused_checks_passed`, and `honest_verdict`.
- `extropic_z1_readiness_packet_ready=true` is valid only when the packet and
  transcript schema exist, the simulator artifacts are complete and remain
  simulator/no-TSU-claim only, access blockers are listed explicitly, focused
  schema checks pass, and `no_hardware_execution_claim=true`.

**Implementation Status:** Planned (Exp 1545)

### SCENARIO-SAMPLE-083: Exp 1545 writes Z1 access packet without hardware claim

Given: Exp 1543 reports n=256 schedule-stress simulator parity ready, Exp 1544
reports n=64 diverse-topology simulator parity ready, and Carnot has no
authenticated Extropic Z1/XTR-0 device access.
When: the Exp 1545 readiness workflow runs from the Carnot project root for run
date 20260508.
Then: it writes the markdown readiness packet, writes a transcript JSON schema,
writes a terminal artifact with all required fields, lists the prior simulator
artifacts and access blockers, confirms no hardware execution claim is made,
and provides rollback criteria for any future authenticated run.

**Implementation Status:** Planned (Exp 1545)

## REQ-SAMPLE-056: Exp 1548 THRML/Carnot independent-RNG parity audit

Carnot SHALL provide an independent-RNG audit for THRML/Carnot simulator
parity that treats bit-identical stochastic summaries as a failure signal, not
as evidence of stronger parity. The audit SHALL use disjoint root seeds and
separate PRNG lineages for Carnot and THRML, SHALL inspect whether the Carnot
lane imports or calls the THRML sampler, SHALL compare bounded distributional
agreement at n=32, n=64, and n=128 across complete, sparse-random, lattice,
and scale-free graph families, and SHALL preserve simulator-only/no-hardware
claim boundaries.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1548_thrml_carnot_parity_independent_rng_audit.json`
  with `status="in_progress"` before parity execution completes.
- The audit SHALL write a seed manifest with disjoint Carnot and THRML root
  seeds and per-case lineage records, and SHALL reject manifests that reuse a
  root seed or derive both sampler seeds by splitting one shared key object.
- The audit SHALL record `code_path_independent=false` if the Carnot lane
  imports or calls the THRML sampler path, and SHALL not claim an independent
  code-path result in that case.
- The sampled comparison SHALL record sample-path hashes, histogram counts,
  mean energy deltas, KL divergence, and a KS-style distribution check for
  every n/topology pair.
- `independent_rng_audit_ready=true` is valid only when there are no
  byte-identical stochastic sample-hash pairs, at least one nonzero
  stochastic delta is observed, KL is positive but bounded, KS checks pass,
  simulator-only/no-TSU-hardware boundaries remain true, and focused tests
  pass.
- The terminal artifact SHALL include `status`, `milestone`,
  `independent_rng_audit_ready`, `rng_path_independent`,
  `code_path_independent`, `rng_seed_manifest_path`, `n_values_tested`,
  `topologies_tested`, `sample_path_hashes`, `byte_identical_pairs`,
  `nonzero_stochastic_delta_observed`, `per_case_results`,
  `max_mean_energy_delta_abs`, `max_kl_divergence`, `min_ks_p_value`,
  `bounded_kl_passed`, `ks_test_passed`, `rng_path_not_independent`,
  `simulator_only`, `no_tsu_hardware_claim`, `focused_tests_passed`, and
  `honest_verdict`.
- `simulator_only=true` and `no_tsu_hardware_claim=true` SHALL remain set for
  every terminal outcome, and the artifact SHALL NOT claim Z1, XTR-0, TSU,
  FPGA, board, synthesis, bitstream, latency, or other hardware execution.

**Implementation Status:** Planned (Exp 1548)

### SCENARIO-SAMPLE-084: Exp 1548 audits THRML/Carnot parity with independent RNG

Given: prior THRML/Carnot parity artifacts reported byte-identical stochastic
histograms at larger n, and no Extropic TSU/Z1/XTR-0 hardware claim is
available.
When: the Exp 1548 independent-RNG audit runs from the Carnot project root for
run date 20260508.
Then: it writes a seed manifest, executes or prototypes independent Carnot and
THRML simulator lanes at n=32, n=64, and n=128 across four topology families,
reports non-identical stochastic sample hashes with bounded distribution
agreement, fails any byte-identical stochastic pair as not independent, and
writes the required terminal artifact fields without hardware claims.

**Implementation Status:** Planned (Exp 1548)

## REQ-SAMPLE-057: Exp 1561 kinetic-defense zero-coupling null-space test

Carnot SHALL provide a bounded simulator-only kinetic-defense audit for the
DT-MCMC-NULL zero-coupling falsifiable predicate. The audit SHALL compare
single-site Metropolis-Hastings, single-site Glauber Gibbs, and THRML 0.1.3
graph-color block-Gibbs semantics on the same planted 64-bit verifier-null-space
landscape without claiming Extropic TSU, Z1/XTR-0, board, synthesis, bitstream,
or hardware execution.

Acceptance criteria:
- The experiment SHALL create
  `results/experiment_1561_kinetic_defense_zero_coupling_test.json` with
  `status="complete"` after writing a bootstrap-safe terminal artifact.
- The landscape SHALL use `n=64` binary variables, 15 independent 4-bit
  verifier blocks, a 4-bit free null suffix, target block `1111`, and
  `E(y) = -10 * sum_i 1{block_i = target}` at temperature `t=1.0`.
- The default run SHALL initialize 10,000 chains at all-zero verifier blocks and
  measure checkpoints `K in {10, 50, 100, 500, 1000}` sweeps.
- The audit SHALL record empirical single-block hitting-time estimates in
  bit-update steps for Algorithm 1 single-site Metropolis-Hastings,
  single-site Glauber Gibbs, and THRML graph-color block-Gibbs semantics.
- The audit SHALL record per-checkpoint current null-space mass, cumulative
  null-space hit mass, mean global hitting sweeps, and per-sample energy
  distributions for each sampler.
- The terminal artifact SHALL include `kinetic_defense_in_depth_validated`,
  `mh_hitting_time_steps_per_block`,
  `single_site_gibbs_hitting_time_steps_per_block`,
  `thrml_block_gibbs_hitting_time_steps_per_block`,
  `thrml_security_parity_with_single_site_gibbs`, `p_n_at_k100_mh`,
  `p_n_at_k100_single_site_gibbs`, `p_n_at_k100_thrml_block_gibbs`,
  `blockers`, and `honest_verdict`.
- `thrml_security_parity_with_single_site_gibbs=true` is valid only when the
  THRML block-Gibbs hitting time is greater than or equal to the single-site
  Gibbs hitting time. If THRML reaches the null space at MH-class rates, the
  artifact SHALL set `kinetic_defense_in_depth_validated=false`, record a
  falsification note, and include a mitigation proposal.
- `honest_verdict` SHALL use a terminal prefix such as `complete_` for every
  complete outcome.

**Implementation Status:** Implemented (Exp 1561)

### SCENARIO-SAMPLE-085: Exp 1561 writes kinetic-defense falsification evidence

Given: DT-MCMC-NULL predicts MH plateau traversal should beat single-site
Gibbs, while THRML block-Gibbs should preserve Gibbs-class plateau friction.
When: the Exp 1561 zero-coupling audit runs from the Carnot project root for
run date 20260508.
Then: it writes the required terminal artifact, reports all three empirical
hitting-time estimates and checkpoint energy distributions, validates the
kinetic-defense claim only when THRML is no faster than single-site Gibbs, and
records a mitigation note if the THRML graph-color update schedule reaches the
planted null space at MH-class rates.

**Implementation Status:** Implemented (Exp 1561)

## REQ-SAMPLE-058: Exp 1564 vendored THRML block-Gibbs inference sampler

Carnot SHALL replace the inference-time hand-rolled Gibbs sampler adapter with
a thin adapter over vendored Extropic THRML 0.1.3 block-Gibbs sampling. The
adapter SHALL initialize each inference chain from the user-provided
`candidate` in the verifier API payload `{prompt, candidate}`, never from a
random, cached, or previous-prompt state.

Acceptance criteria:
- The repository SHALL include a vendored THRML 0.1.3 source subtree under
  `python/carnot/sampling/_vendored_thrml/` with Apache-2.0 license provenance.
- `python/carnot/sampling/gibbs.py` SHALL import the vendored THRML package and
  dispatch Ising sampling through THRML `IsingEBM`, `IsingSamplingProgram`, and
  `sample_states` block-Gibbs APIs.
- The adapter SHALL expose a verifier-payload entry point that accepts
  `{prompt, candidate}` and SHALL use `candidate` as the initial free-block
  state passed to THRML.
- A zero-coupling K=1 sanity check SHALL show THRML block-Gibbs randomizes from
  the supplied initial state with binomial-center Hamming-distance behavior.
- An Exp 1548-style constructive parity audit SHALL report
  `kl_to_thrml_after_vendoring = 0.0` because the Carnot adapter and THRML
  reference path execute the same vendored transition operator.
- `results/experiment_1564_thrml_vendored_block_gibbs_replacement.json` SHALL
  include `status`, `thrml_vendoring_complete`,
  `kl_to_thrml_after_vendoring`, `candidate_warm_start_implemented`,
  `regression_tests_passed`, `mirror_repo_url`, and `honest_verdict`.

**Implementation Status:** Planned (Exp 1564)

### SCENARIO-SAMPLE-086: Exp 1564 uses candidate-warm-started vendored THRML

Given: a verifier API payload containing `prompt` and a binary `candidate`
state, and an Ising model with deterministic biases and couplings.
When: the Carnot Gibbs adapter runs the inference sampler.
Then: the first THRML chain state is initialized from the supplied candidate,
the returned samples come from vendored THRML block-Gibbs rather than Carnot's
legacy random initialization, and the Exp 1564 deliverable records constructive
KL parity with THRML as `0.0`.

**Implementation Status:** Planned (Exp 1564)

## REQ-SAMPLE-059: Exp 1565 Soft-Gibbs residual BRS for contradictory verifier composition

Carnot SHALL provide both a hard rejection sampler and a Soft-Gibbs residual
sampler for verifier-composed prior sampling.  The hard sampler SHALL draw
from a caller-supplied prior sampler and accept only states in the caller's
hard accept set.  The soft sampler SHALL draw from the same prior interface,
compute the residual violation count
`V(y) = sum_i 1{verifier_i(y) is false}`, and accept each proposal with
probability `A(y) = exp(-beta * V(y))`.

Acceptance criteria:
- `python/carnot/sampling/brs_residual.py` SHALL expose
  `hard_brs(prior_sampler, accept_set_S, n_steps)` and
  `soft_brs(prior_sampler, verifiers, beta, n_steps)`.
- The Exp 1565 harness SHALL build an `n=8` latent EBM prior by sampling a
  continuous latent vector and pushing it forward through `sgn` to produce
  uniform `{-1, +1}` spin states.
- The harness SHALL include three deliberately contradictory verifiers:
  `y_1 = +1`, `y_1 = -1`, and `y_2 = +1 OR y_3 = +1`, whose hard
  intersection is logically empty and whose minimum violation count is `1`.
- The harness SHALL run hard and soft BRS for `N in {10, 100, 1000, 10000}`
  and soft beta values `{0.5, 1.0, 2.0, 5.0}`, tracking empirical SubOpt,
  acceptance rate, and the distribution over minimum-violation states.
- Hard-BRS SHALL report empirical acceptance rate `0.0` for the contradictory
  verifier set at every requested `N`.
- Soft-BRS SHALL report exponential SubOpt decay consistent with
  `(1 - Z_beta)^N` for at least one tested beta and SHALL find a
  minimum-violation state with `V(y) = 1`.
- `results/experiment_1565_soft_gibbs_residual_implementation.json` SHALL
  include `status`, `soft_gibbs_residual_implemented`,
  `hard_brs_acceptance_rate`, `soft_brs_decay_confirmed`,
  `min_violation_state_found`, `z_beta_curve`, and `honest_verdict`.

**Implementation Status:** Implemented (Exp 1565)

### SCENARIO-SAMPLE-087: Exp 1565 soft residual survives empty hard intersection

Given: the Exp 1565 `n=8` sign-pushforward prior and the contradictory
three-verifier composition.
When: Hard-BRS and Soft-BRS are each run for the requested step counts.
Then: Hard-BRS accepts no samples, Soft-BRS accepts with beta-dependent rates
near `Z_beta`, the empirical no-acceptance SubOpt curve decays like
`(1 - Z_beta)^N` for at least one beta, and accepted Soft-BRS samples include
states with the exact minimum residual violation count `V(y) = 1`.

**Implementation Status:** Implemented (Exp 1565)

## REQ-SAMPLE-060: Exp 1566 candidate warm-start vs cold-start benchmark

Carnot SHALL provide a bounded, held-out benchmark for the DT-MCMC-STATELESS
prediction that verifier inference chains must initialize from the current
`{prompt, candidate}` payload rather than a random or cached previous-prompt
state.  The benchmark SHALL compare candidate-warm-start, cold-start, and
cached-state-warm-start initialization policies over a corpus of at least 200
`(prompt, candidate, oracle_verdict)` rows spanning correct, incorrect, and
edge-case structurally valid candidates.

Acceptance criteria:
- The benchmark SHALL evaluate THRML block-Gibbs-compatible sweeps at
  `K in {10, 50, 100, 500, 1000}` for each initialization policy.
- Candidate-warm-start SHALL set `y_init = bits(candidate)`, cold-start SHALL
  set `y_init` from uniform random bits, and cached-state-warm-start SHALL set
  `y_init` from a recent chain state belonging to a different prompt.
- For every policy and K, the artifact SHALL report end-to-end verification
  accuracy against the oracle, mean terminal energy, and 95th-percentile
  latency per request rounded to 10 ms granularity.
- `candidate_warm_start_validated=true` only when candidate-warm-start at
  `K=100` reaches at least 99% of its `K=1000` accuracy and also outperforms
  cold-start at `K=100`.
- `cold_start_accuracy_drop_percent_at_k100` SHALL equal the relative accuracy
  drop from cold-start `K=1000` to cold-start `K=100`; the DT gate passes only
  when the drop is at least 50%.
- `cached_state_worse_than_cold_start=true` only when cached-state-warm-start
  at `K=100` is less accurate than cold-start at `K=100`.
- `results/experiment_1566_candidate_warm_start_vs_cold_start_benchmark.json`
  SHALL include `status="complete"`, `candidate_warm_start_validated`,
  `cold_start_accuracy_drop_percent_at_k100`,
  `cached_state_worse_than_cold_start`,
  `recommended_deployment_policy`, and an `honest_verdict` prefixed with
  `complete:`.

**Implementation Status:** Planned (Exp 1566)

### SCENARIO-SAMPLE-088: Exp 1566 validates stateless candidate initialization

Given: a held-out corpus with at least 200 structurally valid verifier payloads
and a recent-state cache whose entries come from different prompts.
When: the Exp 1566 benchmark runs candidate-warm-start, cold-start, and
cached-state-warm-start policies over the requested K sweep values.
Then: the terminal artifact records per-policy/per-K accuracy, energy, and
latency measurements, validates candidate-warm-start only under the DT gates,
reports the cold-start K=100 relative accuracy drop, and recommends deploying
candidate-warm-start while rejecting cached previous-prompt state reuse.

**Implementation Status:** Planned (Exp 1566)

## REQ-SAMPLE-061: Exp 1567 rho(C) measurement for k=6 verifier ensemble

Carnot SHALL provide a reproducible Exp 1567 harness that measures the
compute-dependent Q11 TSS false-positive inflation curve `rho(C)` for the
k=6 AND-composed verifier ensemble over a holdout of at least 200
oracle-incorrect base-generator rows.  The harness SHALL use the existing
calibration corpus machinery where available and SHALL record whether the
measurement is a deterministic replay/proxy or a fresh GPU run.

Acceptance criteria:
- The harness SHALL build a holdout corpus with `N >= 200` rows whose oracle
  labels are incorrect (`y not in S*`).
- The harness SHALL sweep the Q11 TSS compute budgets
  `C in {2^0, 2^2, 2^4, 2^6, 2^8}` GPU-hours and pass the optimized
  adversarial responses through the k=6 AND ensemble: Z3 SMT, AST structural,
  semantic consistency, ThinkPRM v2, SOSKAN-Energy v3, and SemEnergy probe.
- For every compute budget, the artifact SHALL report
  `FPR_AND(C) = n_passed / N` and `rho(C) = FPR_AND(C) - FPR_iid`.
- The harness SHALL fit a monotone saturating `rho(C)` curve, report
  `rho_C_r_squared`, and set `rho_C_curve_fitted=true` only when
  `R^2 >= 0.9`.
- The harness SHALL compute empirical `C*` and `C_inv` from the fitted curve
  using the paper-v6 formulas and SHALL include 95% confidence intervals.
- The harness SHALL evaluate the SRS inversion predicate at a budget strictly
  above `C_inv` and SHALL set `inversion_empirically_confirmed=true` only when
  the accepted accuracy is below `s_r*`.
- `results/experiment_1567_rho_of_C_measurement_k6_ensemble.json` SHALL include
  `status="complete"`, `rho_C_curve_fitted`, `rho_C_r_squared`,
  `C_star_estimate`, `C_star_ci_lower`, `C_star_ci_upper`,
  `C_inv_estimate`, `C_inv_ci_lower`, `C_inv_ci_upper`,
  `inversion_empirically_confirmed`,
  `srs_accepted_accuracy_at_C_above_C_inv`, and an `honest_verdict` prefixed
  with `complete:`.

**Implementation Status:** Planned (Exp 1567)

### SCENARIO-SAMPLE-089: Exp 1567 fits rho(C) and confirms inversion

Given: at least 200 checked oracle-incorrect base-generator rows and the
paper-v6 calibration constants `FPR_iid`, `TPR`, `FNR`, and `s_r*`.
When: the Exp 1567 harness sweeps the five Q11 TSS compute budgets and fits the
monotone `rho(C)` curve.
Then: the terminal artifact reports an `R^2 >= 0.9` fit, finite ordered 95%
confidence intervals for `C*` and `C_inv`, and confirms inversion only when
the measured SRS accepted accuracy at `C > C_inv` is below `s_r*`.

**Implementation Status:** Planned (Exp 1567)

## REQ-SAMPLE-062: Exp 1570 Soft-Gibbs Jensen coverage-bound calibration

Carnot SHALL provide a reproducible Exp 1570 harness that empirically verifies
the DT-OT-RESIDUAL Jensen lower bound for the Soft-Gibbs residual over the
paper-v6 k=6 verifier ensemble.  The harness SHALL measure
`alpha_i = P_mu(y notin S_i)` for each verifier on a calibration corpus with
`N >= 500` rows, then compare the Jensen prediction
`Z_beta >= product_i exp(-beta * alpha_i)` against measured Soft-BRS
acceptance rates.

Acceptance criteria:
- The harness SHALL use the checked-in FoVer calibration corpus when it
  provides at least 500 rows, or deterministically build an equivalent
  calibration corpus of at least 500 rows when the checked-in corpus is
  unavailable.
- The verifier ensemble SHALL contain exactly the six paper-v6 k=6 verifier
  channels: Z3 SMT, AST structural, semantic consistency, ThinkPRM v2,
  SOSKAN-Energy v3, and SemEnergy probe.
- The harness SHALL compute and report `alpha_i_per_verifier` as a list of six
  floats in k=6 verifier order.
- For beta values `{0.1, 0.5, 1.0, 2.0, 5.0, 10.0}`, the harness SHALL report
  `z_beta_jensen_bound` rows containing `beta` and `predicted_lower`, and
  SHALL run corpus-level Soft-BRS to report `z_beta_empirical` rows containing
  `beta` and `empirical_acceptance_rate`.
- The acceptance gate SHALL set `jensen_bound_holds_for_all_beta=true` only
  when every empirical acceptance rate is greater than or equal to the matching
  Jensen-predicted lower bound.
- The harness SHALL determine `optimal_beta_for_deployment` by maximizing
  coverage tightness times empirical acceptance rate, where tightness is
  `predicted_lower / empirical_acceptance_rate`.
- `results/experiment_1570_soft_gibbs_coverage_bound_empirical_verification.json`
  SHALL include `status="complete"`, `alpha_i_per_verifier`,
  `z_beta_jensen_bound`, `z_beta_empirical`,
  `jensen_bound_holds_for_all_beta`, `optimal_beta_for_deployment`, and an
  `honest_verdict` prefixed with `complete:`.

**Implementation Status:** Planned (Exp 1570)

### SCENARIO-SAMPLE-090: Exp 1570 verifies the Jensen lower bound

Given: a calibration corpus with at least 500 rows and the six k=6 verifier
channels in paper-v6 order.
When: the Exp 1570 harness measures per-verifier alpha values and sweeps the
requested beta values through corpus-level Soft-BRS.
Then: every empirical `Z_beta` acceptance rate is at least the Jensen-predicted
lower bound, the artifact records the beta-wise predicted and empirical
curves, and the selected deployment beta is one of the swept beta values.

**Implementation Status:** Planned (Exp 1570)

## REQ-SAMPLE-067: Exp 2883 THRML Sampler Portability Smoke V2

Carnot SHALL provide a tiny THRML sampler portability smoke that checks the
active Python/JAX/THRML/fallback-sampler preconditions, runs a THRML sampler
lane only when `thrml` is already importable, runs the local Carnot fallback
sampler when possible, compares sample shapes and energy-histogram sanity, and
writes a terminal JSON artifact without installing THRML or claiming Extropic
TSU/hardware access.

Acceptance criteria:
- The experiment SHALL write
  `results/experiment_2883_thrml_sampler_portability_smoke_v2.json`.
- The terminal artifact SHALL include `honest_verdict`,
  `thrml_portability_ready`, `blocked_reason`, `preconditions_checked`,
  `thrml_import_available`, `jax_devices`, `local_fallback_ran`,
  `problem_spec`, `sample_count`, `parity_metrics`,
  `hardware_claim_made`, `tests_run`, `field_principles`, `run_date`, and
  `duration_s`.
- The precondition probe SHALL record the Python version, JAX availability and
  device inventory, THRML import availability, and local fallback sampler
  availability without installing packages.
- If THRML is unavailable, the artifact SHALL set
  `blocked_reason="blocked_thrml_unavailable"`, SHALL NOT attempt a THRML
  sampler run, and SHALL still run the local Carnot fallback sampler for the
  same tiny Ising/PGM problem when possible.
- If THRML is available, the smoke SHALL run the THRML and local fallback
  lanes with fixed seeds on the same tiny Ising problem, compare sample shape,
  mean-energy delta, energy histogram non-emptiness, available update-count
  metadata, and runtime.
- `hardware_claim_made=false` SHALL remain set for every terminal outcome, and
  the artifact SHALL NOT claim TSU, Z1, XTR-0, FPGA, board, synthesis,
  bitstream, latency, or hardware acceleration.

**Implementation Status:** Implemented (Exp 2883)

### SCENARIO-SAMPLE-095: Exp 2883 Writes Tiny THRML Portability Smoke

Given: the active project environment may or may not have importable THRML.
When: the Exp 2883 sampler portability smoke runs from the Carnot project root
for run date 20260522.
Then: it writes the required terminal artifact fields, records
`blocked_thrml_unavailable` if THRML is absent, runs the local fallback sampler
when available, runs and compares the THRML lane only when already importable,
and preserves a software-only no-hardware-claim boundary.

**Implementation Status:** Implemented (Exp 2883)

## REQ-SAMPLE-096: Exp 2901 THRML Local Import Repair + N=16 Reattempt

Carnot SHALL provide a local THRML import repair runner that reproduces the Exp
2883 import failure with a full traceback, confirms the active JAX version,
repairs a missing or stale THRML package inside the project virtualenv, and
reattempts the n=16 THRML/Carnot software parity smoke against the real
installed THRML package.

Acceptance criteria:
- The experiment SHALL write
  `results/experiment_2901_thrml_local_import_repair_v1.json`.
- The terminal artifact SHALL include `honest_verdict`,
  `inference_substrate`, `thrml_import_succeeded`,
  `thrml_version_installed`, `jax_version`, `parity_energy_delta`,
  `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The runner SHALL execute `.venv/bin/python -c "import jax; print(jax.__version__)"`
  or its command-equivalent and record the observed JAX version.
- The runner SHALL reproduce the THRML import state before repair and preserve
  the complete traceback text when import fails.
- The repair command SHALL be scoped to the project virtualenv and SHALL use
  `pip install -U thrml` unless a caller supplies an explicit THRML package
  specifier.
- The parity reattempt SHALL use the deterministic Exp 1527 n=16 signed
  ring-chord case and SHALL report the absolute Carnot/THRML energy delta.
- The artifact SHALL remain software-only and SHALL NOT claim TSU, Z1, XTR-0,
  FPGA, board, synthesis, bitstream, latency, or hardware acceleration.

**Implementation Status:** Planned (Exp 2901)

### SCENARIO-SAMPLE-096: Exp 2901 Repairs Local THRML Import And Writes Artifact

Given: Exp 2883 previously reported a local THRML import failure.
When: the Exp 2901 import repair runner executes from the Carnot project root.
Then: it records the pre-repair traceback, repairs THRML in the project
virtualenv when safe, imports the installed THRML package, runs the deterministic
n=16 Carnot/THRML parity comparison, and writes the required terminal JSON
artifact with a stable reproducibility checksum.

**Implementation Status:** Planned (Exp 2901)

## REQ-SAMPLE-097: Exp 6793 matched-update temporal exchange Ising comparison

Carnot SHALL provide a CPU-only single-site Ising sampler that compares ordinary
Gibbs updates with time-dimensional exchange coupling from arXiv:2608.21753.
The same interface SHALL run ordinary Gibbs, nonzero temporal exchange, and
temporal exchange with zero coupling. The comparison SHALL count one attempted
single-spin conditional draw as one spin update in every arm.

Acceptance criteria:
- For spatial energy
  `E(s) = -sum_i b_i s_i - sum_(i<j) J_ij s_i s_j`, the sampler SHALL use
  `P(s_i=+1) = sigmoid(2 * (b_i + sum_j J_ij s_j + kappa p_i) / T)`.
  Here `p_i` is the matching spin in the previous completed configuration.
- The sampler SHALL store current and previous configurations separately. The
  previous configuration SHALL stay fixed during one complete random-order
  sweep. It SHALL become a copy of the completed current configuration only at
  the sweep boundary.
- Negative `kappa` SHALL implement low-temperature antiferromagnetic temporal
  coupling. Positive `kappa` SHALL implement high-temperature ferromagnetic
  temporal coupling. The frozen grid SHALL be `{-0.08, 0, +0.08}`. The selected
  schedule SHALL use `-0.08` at `T=0.75` and `+0.08` at `T=2.0`.
- The headline panel SHALL use three fixed graph families with 6, 7, and 8
  spins. Every graph-temperature target law SHALL be exhaustively enumerated.
  Coefficients SHALL satisfy `abs(b_i) <= 0.15` and `abs(J_ij) <= 0.60`.
- The headline panel SHALL use seeds 679300 through 679319. Every matched cell
  SHALL use the same current state, previous state, random seed, burn-in,
  collection interval, sample count, and attempted spin-update count.
- Each headline row SHALL collect 1,024 samples after 128 complete burn-in
  sweeps. It SHALL preserve the exact target marginal, empirical marginal,
  energy trace, best state, autocorrelation, effective samples, optimum hitting
  updates, diversity, row wall time, and update count.
- The zero-coupling temporal arm SHALL be bit-identical to ordinary Gibbs for
  every matched row. This includes collected states, energy traces, optimum
  hitting updates, and update counts.
- The artifact SHALL derive target-law error, energy and magnetization error,
  integrated autocorrelation, effective samples per update, optimum hitting,
  and diversity per graph-temperature stratum before any cross-stratum summary.
- Positive credit SHALL require a positive paired 95% lower confidence bound
  for energy effective samples per update in every headline stratum. It SHALL
  also require the temporal-minus-Gibbs total-variation upper bound to be at
  most `0.03` in every headline graph family.
- A 32-spin stress panel MAY run only after all headline rows complete. Stress
  rows SHALL stay separate and SHALL not contribute to target-law fidelity or
  the positive gate.
- The artifact SHALL set `physical_hardware_invoked=false`. It SHALL make no
  FPGA, TSU, latency, power, energy-use, or hardware-availability claim.
- A failed sampler, enumeration, coefficient, clock, memory, or wall-budget
  precondition SHALL produce `complete_blocked_temporal_exchange_ab` with the
  failed check, expected value, and observed value in `gate_check_summary`.

**Implementation Status:** Planned (Exp 6793)

### SCENARIO-SAMPLE-097: Temporal conditional and explicit previous state

Given: a finite symmetric Ising graph, a positive temperature, explicit current
and previous bipolar states, and a coupling from the frozen grid.
When: the temporal sampler attempts a single-site update.
Then: it uses the specified conditional probability, keeps the previous state
fixed for the sweep, applies the declared coupling sign, and increments the
attempted spin-update count by one.

**Implementation Status:** Planned (Exp 6793)

### SCENARIO-SAMPLE-098: Matched collection and disabled-coupling equivalence

Given: the frozen graph, temperature, seed, initial-state pair, burn-in, sample
count, and update budget.
When: ordinary Gibbs and both temporal arms run through the common interface.
Then: all three arms collect at the same update calls, and ordinary Gibbs is
bit-identical to temporal exchange with zero coupling.

**Implementation Status:** Planned (Exp 6793)

### SCENARIO-SAMPLE-099: Exact target-law artifact and terminal null allowance

Given: all preconditions pass and every headline row is attributable.
When: Exp 6793 enumerates each headline target and reduces retained rows.
Then: `temporal_exchange_comparison_completed=true` regardless of measured
effect, while `verdict_class=positive` only when both the efficiency and
target-law gates pass. A complete measured null remains terminal.

**Implementation Status:** Planned (Exp 6793)

## REQ-MODEL-031: SCEnergyModel — Set-Level Energy Function for Statement Consistency (Exp 944)

SCEnergyModel SHALL implement a permutation-invariant set-level energy function that assigns
a scalar energy to a list of natural-language statements. Architecture: TF-IDF embedding
per statement, mean pooling, 2-layer MLP to scalar energy. Training: contrastive loss
pushing E(coherent_set) << E(contradictory_set). Trained model SHALL achieve AUROC > 0.60
on held-out coherent vs contradictory set pairs derived from GSM8K solutions.

- REQ-MODEL-031-1: energy(statements) SHALL accept a list of strings and return a scalar float.
- REQ-MODEL-031-2: The energy function SHALL be permutation-invariant (mean pooling).
- REQ-MODEL-031-3: TFIDFEmbedder SHALL be fittable on a corpus and embeddable per statement.
- REQ-MODEL-031-4: train() SHALL accept coherent_sets and contradictory_sets of equal length
  and return per-epoch loss history.
- REQ-MODEL-031-5: SCEnergyModel SHALL be importable from carnot.models.sc_energy.

**Implementation Status:** Implemented (Exp 944)

### SCENARIO-MODEL-016: Coherent vs Contradictory Set Discrimination

Given: A fitted SCEnergyModel trained on GSM8K-derived coherent/contradictory pairs.
When: energy() is called on coherent_set (same-problem steps) vs contradictory_set
      (cross-problem mixed steps).
Then: The model assigns lower mean energy to coherent sets than contradictory sets,
      achieving AUROC > 0.60 on held-out pairs.

**Implementation Status:** Implemented (Exp 944)


### REQ-TRAIN-007: Energy-Based Fine-Tuning (EBFT)
The system shall implement sequence-level feature matching via EBFT, where:
- The objective minimizes expert sequence energy while maximizing rollout sequence energy.
- The objective calculation supports gradient flow for differentiable energy functions.
- It operates without explicit external verifiers.

### SCENARIO-TRAIN-007: EBFT Gradient Flow
Given expert and rollout sequences
When the EBFT loss is calculated
Then the gradient properly flows through the energy function to the parameters.

### REQ-SAMPLE-038: Denoising Thermodynamic Model (DTM) Simulation

**Description:** The system SHALL provide a simulation of Denoising Thermodynamic Models (DTM) using `thrml`.

#### SCENARIO-SAMPLE-038-1: DTM Diffusion-like Sampling (Exp 1806)

**Given** an environment with `thrml`
**When** running the DTM simulation script
**Then** it SHALL output distribution convergence and `thrml_import_ready`.

### REQ-INFER-1956: NCO-style negative constraint pattern matching

The system shall provide a token-level negative constraint layer for decoding
that:
- registers forbidden literal and regex patterns without compiling them into a
  single cross-product automaton;
- rejects a candidate token before commitment when the current suffix plus that
  candidate would match a forbidden pattern;
- works with a positive-mask trie when one is present, applying positive
  admissibility before negative rejection;
- records rejected token IDs and the matching constraint names for audit; and
- writes `results/experiment_1956_nco_negative_constraints.json` with
  `status`, `nco_negative_constraint_layer_ready`,
  `negative_constraints_upheld`, `overhead_vs_positive_trie`,
  `rejected_token_examples`, `tests_run`, and `honest_verdict`.

**Implementation Status:** Implemented (Exp 1956)

### SCENARIO-INFER-1956: Negative constraints reject split-token continuations

Given: a registry containing literal and regex negative constraints and a
candidate token stream where a forbidden value is split across tokens.
When: the decoder evaluates each candidate token online.
Then: the token completing the forbidden pattern is rejected before it is
committed, the fallback candidate is selected, and the artifact reports the
measured overhead relative to positive-mask trie decoding.

**Implementation Status:** Implemented (Exp 1956)
