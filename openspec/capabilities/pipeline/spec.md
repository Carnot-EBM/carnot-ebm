# Pipeline Capability Specification

**Capability:** pipeline
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-08

## Overview

Defines the multi-tier cascade pipeline that routes LLM-generated reasoning steps through
progressively more expensive verifiers (Tier 1: fast energy gate, Tier 2: JEPA ranking,
Tier 3: full formal verification).  The JEPA predictor tier must be loaded from a
version-tagged checkpoint to prevent silent rollbacks to sub-threshold models.

## Requirements

### REQ-INFRA-043: JEPA Cascade Version-Pinned Checkpoint Loading

The Tier 2 JEPA predictor in the cascade MUST load its model via a version-tagged
checkpoint path.  The active version is recorded in the conductor exclusion manifest
under the `jepa_v18_active` flag.  Loading an older version (v15/v16/v17) is
prohibited while the manifest marks them excluded; attempting to load an excluded
version MUST raise `ValueError`.

**Rationale:** JEPA v15/v16/v17 all produced OOD AUC below random chance (0.47–0.48),
meaning the cascade would actively harm verification quality.  The manifest is the
authoritative gate that prevents accidental rollback to a below-threshold model.

**Acceptance criteria:**
- `tier2_jepa.load_v18_from_manifest()` returns a `JEPALambdaRankV18` instance.
- Passing `version="v17"` raises `ValueError` with a message naming the blocked version.
- The loaded model's `predict_score()` returns a float for any string input.

### REQ-INFRA-044: Cascade AUC Gating

The cascade integration smoke test (Exp 718) MUST achieve cascade_auc >= 0.70 on a
50-question held-out GSM8K validation set before the JEPA v18 deployment is considered
successful.  If cascade_auc < 0.70, the honest_verdict MUST be "cascade_deploy_auc_fail"
and the gate file for Exp 719 MUST record `gate: "fail"`.

### REQ-INFRA-045: Cascade Latency Budget

The per-question latency overhead added by the JEPA v18 Tier 2 scorer MUST be less than
5 ms (latency_delta_ms < 5).  Exceeding this budget causes honest_verdict
"cascade_deploy_latency_fail" regardless of AUC.

## Scenarios

### SCENARIO-INFRA-052: Version-Blocked Model Raises Error

**Given** the exclusion manifest marks `jepa_v17_blocked: true`
**When** code calls `tier2_jepa.load_v18_from_manifest(version="v17")`
**Then** a `ValueError` is raised with message containing "blocked"

**Spec traces:** REQ-INFRA-043

### SCENARIO-INFRA-053: v18 Loads and Scores Successfully

**Given** a trained `JEPALambdaRankV18` instance
**When** `predict_score("Step 1: 3 + 5 = 8.")` is called
**Then** a scalar float is returned without raising any exception

**Spec traces:** REQ-INFRA-043

### SCENARIO-INFRA-054: Cascade AUC on Held-Out Groups

**Given** the cascade is loaded with JEPA v18 as Tier 2
**When** `evaluate_auc(eval_groups)` is called on 50 held-out GSM8K groups
**Then** the returned float is in [0, 1]

**Spec traces:** REQ-INFRA-044

### REQ-INFRA-046: EORM Confidence Gate for Tier 3 Ising Skip

The cascade router MUST support an EORM confidence gate that skips Tier 3 Ising
sampling when the EORM confidence score exceeds a configurable threshold (default 0.92).
When EORM confidence > eorm_ising_skip_threshold, the result is marked "verified_fast"
and Tier 3 Ising is not run.  The threshold is configurable at CascadeRouter
construction time.  Each query MUST log ising_skip (bool) and eorm_confidence (float).

**Acceptance criteria:**
- `CascadeRouter(eorm_ising_skip_threshold=0.92)` skips Ising when EORM confidence > 0.92.
- `CascadeRouter(eorm_ising_skip_threshold=0.92)` runs Ising when EORM confidence <= 0.92.
- Per-query logs contain ising_skip and eorm_confidence fields.

### REQ-INFRA-047: EORM Gate False-Negative Delta

The EORM confidence gate MUST NOT increase the false-negative rate by more than 5
percentage points versus the full cascade (no gate) on a representative test set.
Formally: fn_delta = false_negative_rate_gated - false_negative_rate_baseline < 0.05.

**Acceptance criteria:**
- Exp 727 measures fn_delta < 0.05 on 200-question test set at threshold=0.92.

## Scenarios

### SCENARIO-INFRA-055: EORM Gate Skips Ising Above Threshold

**Given** a CascadeRouter with eorm_ising_skip_threshold=0.92
**When** EORM returns confidence=0.95 for a query
**Then** Tier 3 Ising is NOT invoked and result is marked "verified_fast"

**Spec traces:** REQ-INFRA-046

### SCENARIO-INFRA-056: EORM Gate Does Not Skip Ising Below Threshold

**Given** a CascadeRouter with eorm_ising_skip_threshold=0.92
**When** EORM returns confidence=0.80 for a query
**Then** Tier 3 Ising IS invoked as normal

**Spec traces:** REQ-INFRA-046

### REQ-INFRA-046b: Conductor Dispatch Manifest Enforcement

The conductor MUST call `validate_manifest_at_dequeue(task_id)` before dispatching
any experiment to execution.  If the function returns False (task_id is in the exclusion
manifest), the task MUST be silently skipped — no agent spawned, no GPU allocated.

**Acceptance criteria:**
- `validate_manifest_at_dequeue("exp308-legacy")` returns False when exp 308 is in manifest.
- `validate_manifest_at_dequeue("exp999-new")` returns True when exp 999 is not in manifest.
- Retired tasks are never dispatched to an agent subprocess.

**Spec traces:** REQ-INFRA-046b (replaces text-level-only exclusion, closes .55 787-min gap)

### REQ-INFRA-047b: GPU VRAM Clean at Milestone Start

All GPU devices MUST have < 100 MB VRAM allocated at the start of each conductor
milestone.  If any device exceeds 100 MB, the conductor MUST kill the holding process
before dispatching the first experiment.

**Acceptance criteria:**
- `gpu1_vram_mb < 100` measured after zombie kill at milestone start.
- Conductor pre-flight logs the before/after VRAM delta.

### SCENARIO-INFRA-055b: Manifest Validator Blocks Excluded Task

**Given** `conductor_exclusion_manifest.json` lists experiment_id=308
**When** `validate_manifest_at_dequeue("exp308-legacy")` is called
**Then** the function returns False and logs "task_id=exp308-legacy allowed=False"

**Spec traces:** REQ-INFRA-046b

### SCENARIO-INFRA-056b: Manifest Validator Passes Unknown Task

**Given** `conductor_exclusion_manifest.json` does not list experiment_id=999
**When** `validate_manifest_at_dequeue("exp999-new")` is called
**Then** the function returns True and logs "task_id=exp999-new allowed=True"

**Spec traces:** REQ-INFRA-046b

### REQ-INFRA-048: Exp 527 Class Mandatory Retirement

Exp 527 (live 100-question precision inference) MUST be present in the conductor
exclusion manifest before milestone 2026.04.57 dequeue.  This retirement is mandated by
governance rule "3-consecutive-mandatory": an experiment that appears in the slowest-5
for three consecutive milestones is automatically retired regardless of research value.

**Acceptance criteria:**
- `ExclusionManifest.is_excluded(527)` returns True after Exp 740 runs.
- The manifest entry includes `governance_rule: "3-consecutive-mandatory"`.
- The entry includes `retired_in_milestone: "2026.04.57"`.

**Spec traces:** REQ-INFRA-048 (governance: Exp 308/309 precedent, RETRO-033)

### REQ-INFRA-049: EORM+JEPA Retrain MUST Use DualGPU ThreadPoolExecutor

EORM+JEPA retrain MUST use a `ThreadPoolExecutor(max_workers=2)` with EORM on
`cuda:0` and JEPA on `cuda:1` when both GPUs are available.  Sequential GPU
training for this class is retired as of milestone 2026.04.57.  The validated
speedup from Exp 685 (2.0175x) is the baseline; any new parallel implementation
MUST achieve >= 1.5x speedup vs sequential.

**Acceptance criteria:**
- `DualGPURetrain.retrain_parallel()` submits both tasks concurrently to a `ThreadPoolExecutor`.
- When only 1 GPU is available, the implementation falls back to sequential execution without error.
- Speedup measurement >= 1.5x on a 2-GPU host.

**Spec traces:** REQ-INFRA-049 (Exp 685 validated 2.0175x, 11 milestones idle GPU 1)

### SCENARIO-INFRA-057: Exp 527 Appears in Exclusion Manifest After Exp 740

**Given** Exp 527 has appeared in the slowest-5 for three consecutive milestones
**When** Exp 740 runs and adds Exp 527 to the exclusion manifest
**Then** `ExclusionManifest.is_excluded(527)` returns True and the entry contains
  `governance_rule: "3-consecutive-mandatory"` and `retired_in_milestone: "2026.04.57"`

**Spec traces:** REQ-INFRA-048

### SCENARIO-INFRA-058: DualGPURetrain Falls Back to Sequential on Single GPU

**Given** only 1 CUDA GPU is available
**When** `DualGPURetrain.retrain_parallel(eorm_model, jepa_model, data)` is called
**Then** both models train sequentially on `cuda:0` without raising any exception,
  and the result dict contains `fallback_reason: "single_gpu"`.

**Spec traces:** REQ-INFRA-049

### REQ-INFRA-050: EORM+JEPA Joint Retrain MUST Use DualGPU When 2 GPUs Available

EORM+JEPA joint retrain calls MUST use `DualGPURetrain.retrain_parallel()` via
`ThreadPoolExecutor` when 2 or more CUDA GPUs are detected.  Sequential single-GPU
retrain of the combined EORM+JEPA pair is deprecated for all Exp 383-class runs.

**Rationale:** Exp 383 appeared in the slowest-5 for 11 consecutive milestones.  Exp 685
validated 2.0175x speedup with GPU 1 idle the entire time.  Exp 746 cements this as a
permanent infrastructure default: `retrain_parallel()` replaces any call site that ran
EORM and JEPA sequentially on a host with >= 2 GPUs.

**Spec traces:** REQ-INFRA-050 (Exp 685 validated 2.0175x; Exp 746 production rollout)

### SCENARIO-INFRA-059: DualGPU EORM+JEPA Retrain Achieves >= 1.8x Speedup

**Given** two CUDA GPUs are available (cuda:0, cuda:1)
**When** `DualGPURetrain.retrain_parallel(eorm_fn, jepa_fn)` runs on FoVer v2 data
**Then** the measured `speedup = wall_time_sequential / wall_time_parallel >= 1.8`
  and both `eorm_loss_after` and `jepa_loss_after` are finite positive floats.

**Spec traces:** REQ-INFRA-050

### REQ-HW-010: Ising Sampler v4 HLS C++ Kernel

The Ising sampler v4 MUST be expressed in Vitis HLS C++ with loop-pipelining
pragmas embedded as comments (so the same file compiles as plain C++ for CPU
validation).  The top-level function `update_spin_kernel` MUST include:

- Sequential Gibbs updates with xorshift32 RNG (HLS-compatible, no stdlib rand)
- EMA inertia field `h_ema[i]` per spin, blending instantaneous and historical fields
- All HLS PIPELINE / UNROLL / ARRAY_PARTITION pragmas as `// #pragma HLS ...` comments
- A CPU-compilable `main()` guarded by `#ifndef __SYNTHESIS__` for validation

The same `ising_sampler_hls.cpp` MUST:
1. Compile under `g++ -O2 -std=c++17` without errors.
2. Produce a final energy within 20% of the ground-state energy for a 4-spin test case.
3. Be synthesisable by Vitis HLS 2024.2 when `synth_ising_hls.tcl` is executed.

**Rationale:** KV260 bitfile synthesis is blocked locally due to missing Vivado
installation.  HLS C++ can be synthesised on any cloud instance with AMD Vitis 2024.2
without requiring a full Vivado install.  The dual-compile approach (same C++ for CPU
and FPGA) allows local validation before remote synthesis.

**Acceptance criteria:**
- `hardware/kv260/ising_sampler_hls.cpp` exists and compiles with g++.
- Compiled binary returns exit code 0 (energy within tolerance).
- `hardware/kv260/synth_ising_hls.tcl` references the correct KV260 part number.
- `results/experiment_750_vitis_hls_ising_v4.json` records `cpp_compiles: true`.

### SCENARIO-HW-010: HLS Kernel CPU Validation

**Given** `hardware/kv260/ising_sampler_hls.cpp` is compiled with
  `g++ -O2 -std=c++17 hardware/kv260/ising_sampler_hls.cpp -o /tmp/ising_hls_test`
**When** the resulting binary is executed
**Then** it prints "PASS" and exits with code 0, meaning the final energy of a
  4-spin antiferromagnetic chain is within 20% of the -3.0 ground-state energy.

**Spec traces:** REQ-HW-010

### REQ-SAMPLE-017: DWaveNealBackend Protocol Implementation

``DWaveNealBackend`` MUST implement the ``SamplerBackend`` protocol by providing
a ``sample()`` method.  It MUST convert an ``IsingEBM`` (``IsingModel``) coupling
matrix and bias vector to a ``dimod.BinaryQuadraticModel`` via a ``to_bqm()``
method before submitting to ``neal.SimulatedAnnealingSampler``.

**Acceptance criteria:**
- ``DWaveNealBackend().available`` is True when dwave-ocean-sdk is installed.
- ``to_bqm(ising_ebm)`` returns a BQM with ``num_variables == ising_ebm.config.input_dim``.
- Quadratic interactions in the BQM match non-zero entries of ``ising_ebm.coupling``.
- Linear biases in the BQM match ``ising_ebm.bias``.

**Spec:** REQ-SAMPLE-017

---

### REQ-SAMPLE-018: DWaveNealBackend Reports Energy and Wall Time

``DWaveNealBackend.sample()`` MUST return a ``SampleResult`` with:
- ``spins``: boolean array of shape ``(n_spins,)`` (the lowest-energy configuration
  found across all ``num_reads`` SA runs).
- ``energy``: float energy of ``spins`` under the IsingEBM Hamiltonian, computed in
  the ``{0,1}`` convention (compatible with ``IsingModel.energy``).
- ``wall_time_s``: float wall-clock seconds for the full call.

**Acceptance criteria:**
- ``result.energy`` is a float.
- ``result.wall_time_s > 0``.
- ``result.spins.shape == (n_spins,)`` and ``result.spins.dtype == bool``.

**Spec:** REQ-SAMPLE-018

---

### SCENARIO-SAMPLE-030: Neal vs Gibbs Energy Comparison on Random Problems

**Given** 20 synthetic IsingModel instances with n=50 spins and coupling sparsity=0.3
**When** both DWaveNealBackend and CpuBackend (Gibbs) are run on each instance
**Then** ``energy_improvement_pct`` is computed as
  ``(mean_energy_gibbs - mean_energy_neal) / |mean_energy_gibbs| * 100``
  and ``honest_verdict`` is one of
  ``{"neal_better_energy", "neal_comparable_energy", "neal_worse_energy"}``.

**Spec traces:** REQ-SAMPLE-017, REQ-SAMPLE-018

### SCENARIO-SAMPLE-031: DWaveNealBackend Blocked on Dependency

**Given** dwave-ocean-sdk is not installed (``neal`` import fails)
**When** ``DWaveNealBackend().available`` is False
**Then** ``sample()`` returns a ``SampleResult`` with ``energy == float('inf')``
  and the experiment artifact records ``honest_verdict == "blocked_on_dependency"``.

**Spec traces:** REQ-SAMPLE-017

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-SAMPLE-017 | Implemented | Exp 751 — dwave_neal_backend.py + to_bqm() |
| REQ-SAMPLE-018 | Implemented | Exp 751 — SampleResult with energy + wall_time_s |
| REQ-HW-010 | Implemented | Exp 750 — ising_sampler_hls.cpp + synth_ising_hls.tcl |
| REQ-INFRA-043 | Implemented | Exp 718 — tier2_jepa.py |
| REQ-INFRA-044 | Implemented | Exp 718 smoke test |
| REQ-INFRA-045 | Implemented | Exp 718 latency measurement |
| REQ-INFRA-046 | Implemented | Exp 727 — cascade_router.py |
| REQ-INFRA-047 | Implemented | Exp 727 fn_delta measurement |
| REQ-INFRA-046b | Implemented | Exp 731 — conductor_manifest_validator.py; patch in results/manifest_fix_patch.txt |
| REQ-INFRA-047b | Implemented | Exp 731 — GPU 1 zombie cleared, vram_after=4 MiB |
| REQ-INFRA-048 | Implemented | Exp 740 — Exp 527 added to exclusion manifest |
| REQ-INFRA-049 | Implemented | Exp 740 — DualGPURetrain in python/carnot/pipeline/dualgpu_retrain.py |
| REQ-INFRA-050 | Implemented | Exp 746 — DualGPU retrain made default; sequential deprecated |
