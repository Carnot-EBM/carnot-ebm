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

### REQ-INFRA-051: Manifest Patch MUST Be Applied at Dispatch Site

The guard clause calling `validate_manifest_at_dequeue(task_id)` MUST be present in
`scripts/research_conductor.py` inside `research_step()`, immediately after the three
`logger.info("RESEARCH STEP: ...")` lines and before the `if dry_run:` check.
Enforcement via code change at the dispatch site, not retro text or manifest-only update.

**Rationale:** Four consecutive milestones (.54-.57) closed with the patch unnapplied,
wasting 1,264 minutes (21.1 hours) cumulative.  String IDs like "jepa_v15_cascade" bypass
`_task_is_excluded`'s integer regex; only a dispatch-site guard closes this gap.

**Acceptance criteria:**
- `grep validate_manifest_at_dequeue scripts/research_conductor.py` returns at least one match.
- The match appears inside the `research_step()` function body.
- The patch from `results/manifest_fix_patch.txt` is fully applied (no diff).

**Spec traces:** REQ-INFRA-051 (closes 4-milestone enforcement gap, Exp 754)

### REQ-INFRA-052: Pre-flight v10 MUST Confirm Patch Application

The pre-flight v10 artifact (`results/experiment_754_preflight_v10.json`) MUST include a
`patch_applied` boolean field set by searching `scripts/research_conductor.py` for the
guard clause pattern `validate_manifest_at_dequeue`.  Only a code-level search counts;
inspecting the patch file alone is insufficient.

**Acceptance criteria:**
- `artifact["patch_applied"] == True` when guard clause is present in the file.
- `artifact["patch_applied"] == False` when guard clause is absent.
- `honest_verdict` is one of: "preflight_v10_patch_applied_gpu_clean",
  "preflight_v10_patch_applied_gpu_dirty", "preflight_v10_patch_failed",
  "preflight_v10_exp527_leak".

**Spec traces:** REQ-INFRA-052 (Exp 754 pre-flight v10)

### SCENARIO-INFRA-060: Dispatch Guard Blocks Excluded Task at Dequeue

**Given** `conductor_exclusion_manifest.json` lists experiment_id=527
**When** `research_step()` is called with task `{"id": "exp527-legacy", ...}`
**Then** `validate_manifest_at_dequeue("exp527-legacy")` returns False, the task is
  skipped without spawning an agent, and `research_step()` returns True.

**Spec traces:** REQ-INFRA-051

### SCENARIO-INFRA-061: Pre-flight v10 Records patch_applied=True After Patch Application

**Given** `scripts/research_conductor.py` contains the guard clause calling
  `validate_manifest_at_dequeue`
**When** the pre-flight v10 check reads the file and searches for the guard clause
**Then** `artifact["patch_applied"]` is True and `honest_verdict` is
  "preflight_v10_patch_applied_gpu_clean" (assuming GPUs are clean and 527 is excluded).

**Spec traces:** REQ-INFRA-052

### REQ-INFRA-053: Exclusion Manifest Check MUST Be Applied at ALL Dequeue Sites

Every site in `scripts/research_conductor.py` where a task/experiment is fetched from any
source (YAML, history, queue) MUST call `_task_is_excluded(task)` before dispatching an agent.
No dequeue may bypass the manifest check.  A single unguarded dequeue is sufficient to re-admit
a retired experiment.

**Rationale:** Exp 425 appeared for the 22nd consecutive milestone (.37 through .58, 1,672 min
cumulative = 27.9 hours of zero-value compute) because the Exp 754 manifest patch covered only
the conductor's managed cycle.  Other dequeue sites existed without the guard.  Full coverage
means EVERY site is guarded.

**Acceptance criteria:**
- `coverage_pct = guarded_sites / total_dequeue_sites * 100 == 100.0`
- `full_coverage == True` in the pre-flight v11 artifact.
- `honest_verdict == "full_manifest_coverage_achieved"` when all sites are guarded and
  `n_excluded_total >= 27`.

**Spec traces:** REQ-INFRA-053 (Exp 767 pre-flight v11)

### REQ-INFRA-054: Exps 425, 491, 603, 627 MUST Be in conductor_exclusion_manifest.json

`scripts/conductor_exclusion_manifest.json` MUST contain entries for experiment IDs 425, 491,
603, and 627 with `completed_milestone` set to at least "2026.04.58" (the milestone where they
last appeared in the slowest-5 full-milestone timing).

**Rationale:**
- Exp 425: 22nd consecutive slowest-5 appearance, 1,672 min cumulative overhead.
- Exp 491: JEPA curriculum diagnostic, 12th appearance, unbounded training loop.
- Exp 603: CoACEExtractorV4 repeated carry-over from unguarded historical queue source.
- Exp 627: interwhen mid-generation monitor, repeated carry-over from unguarded source.

**Acceptance criteria:**
- `manifest["n_excluded_total"] >= 27` (23 before this patch + 4 new entries).
- `new_exclusions_added` list in pre-flight v11 artifact contains all four IDs.

**Spec traces:** REQ-INFRA-054 (Exp 767 pre-flight v11)

### SCENARIO-INFRA-062: Full Dequeue Coverage Confirmed at 100% After v11 Patch

**Given** all dequeue sites in `scripts/research_conductor.py` have been audited
**When** the pre-flight v11 script counts guarded vs unguarded dequeue sites
**Then** `coverage_pct == 100.0`, `full_coverage == True`, and `guarded_sites_after_patch` equals
  `total_dequeue_sites`.

**Spec traces:** REQ-INFRA-053

### SCENARIO-INFRA-063: New Exclusions 425/491/603/627 Present in Manifest After v11

**Given** `conductor_exclusion_manifest.json` previously had 23 entries
**When** the pre-flight v11 script adds Exps 425, 491, 603, 627 for milestone "2026.04.58"
**Then** the manifest has at least 27 entries and all four IDs are excluded when queried
  via `ExclusionManifest.is_excluded()`.

**Spec traces:** REQ-INFRA-054

### REQ-INFRA-055: kill_gpu_zombies() MUST Be Called Before Model Load in setup_gpu()

`kill_gpu_zombies()` from `carnot.pipeline.gpu_zombie_killer` MUST be called inside
`ExperimentTemplate.setup_gpu()` before any model load attempt when `CARNOT_FORCE_LIVE=1`.
The function MUST use `subprocess` to run `nvidia-smi --query-compute-apps=pid
--format=csv,noheader,nounits` to enumerate PIDs holding GPU memory, then send `SIGKILL`
to each PID that is NOT the current process and NOT in the caller-supplied exclude list.
The result MUST be recorded in setup_gpu()'s return dict under `zombie_kill_result`.

**Rationale:** RETRO-028 (Gemma4 14.89 GiB allocation fails with 15 GiB already in use)
and RETRO-SOTA-GGUF-TIMEOUT (Exp 769 timeout) share a common root cause: zombie processes
holding GPU VRAM before model load.  Fixing only at setup() (session start) is insufficient
because mid-session failures can accumulate zombies between experiments.

**Acceptance criteria:**
- `setup_gpu()` return dict contains `zombie_kill_result` key.
- When CARNOT_FORCE_LIVE=1 and zombie PIDs exist, they are sent SIGKILL.
- The calling process PID is never in the kill list.

**Spec traces:** REQ-INFRA-055

### REQ-INFRA-056: kill_gpu_zombies() MUST Be a No-Op When No Zombies Exist

When `nvidia-smi` reports no compute processes on the target GPU, `kill_gpu_zombies()`
MUST return a `GPUZombieResult` with `pids_killed=[]`, `vram_freed_mb=0.0`, and
`honest_verdict="no_zombies_found"`.  The function MUST NOT kill the calling process
itself under any circumstances.  When `nvidia-smi` is unavailable, `honest_verdict`
MUST be `"nvidia_smi_unavailable"`.

**Acceptance criteria:**
- Empty nvidia-smi output → `honest_verdict="no_zombies_found"`, `pids_killed=[]`.
- Calling PID is always in `exclude_pids`; never sent SIGKILL.
- Missing nvidia-smi → `honest_verdict="nvidia_smi_unavailable"`.

**Spec traces:** REQ-INFRA-056

### SCENARIO-INFRA-064: kill_gpu_zombies() No-Op on Clean GPU

**Given** `nvidia-smi --query-compute-apps=pid` returns no output (empty GPU)
**When** `kill_gpu_zombies(gpu_index=0)` is called
**Then** `honest_verdict="no_zombies_found"`, `pids_killed=[]`, `vram_freed_mb=0.0`

**Spec traces:** REQ-INFRA-056

### SCENARIO-INFRA-065: kill_gpu_zombies() Excludes Calling Process

**Given** the calling process PID appears in `nvidia-smi --query-compute-apps=pid` output
**When** `kill_gpu_zombies(gpu_index=0)` is called without explicit exclude_pids
**Then** `os.getpid()` is never sent SIGKILL

**Spec traces:** REQ-INFRA-055, REQ-INFRA-056

### REQ-INFRA-057: NPU Unblock Automated Install Strategy Limit

**Statement:** NPU unblock experiments MUST attempt Option A (GitHub Releases wheel) first,
MUST attempt Option B (Ryzen AI SDK installer) if and only if Option A fails, and MUST NOT
attempt more than 2 automated install strategies per experiment run.

**Why this matters:**
    Eight consecutive milestones (Exps 292, 303, 314, 335, 435, 714) were blocked by the
    same root cause — mlir-aie not on PyPI and VitisAI requiring a compiled-in onnxruntime.
    Without a hard cap on strategy attempts, experiments can spiral into open-ended install
    loops that consume 45+ minutes without producing a binary verdict.  The two-strategy
    cap forces a clean "exhausted" verdict so the conductor can escalate rather than retry
    forever.  Option A (GitHub Releases wheel) is tried first because it requires no auth
    and targets the exact missing package; Option B (Ryzen AI SDK installer) requires AMD
    account credentials and is therefore a fallback.

**Spec traces:** Exp 790, RETRO-NPU-v9

### SCENARIO-INFRA-066: NPU Unblock Option B Tried Only After Option A Failure

**Given** Option A (GitHub Releases wheel) install fails (option_a_success=False)
**When** the NPU unblock script proceeds to Option B
**Then** option_b_attempted=True in the result artifact

**Spec traces:** REQ-INFRA-057

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

---

## REQ-PUBLISH-001: HuggingFace Model Card Requirements

Every model published to the Carnot-EBM HuggingFace organisation MUST include a model card with:
- Architecture description (what the model does and WHY the design choices were made)
- Training data citation (dataset name, size, and collection methodology)
- Evaluation metrics (AUC, AUROC, FP rate, latency as appropriate)
- Usage example showing `pip install carnot` and inference code
- Apache 2.0 license declaration
- Explicit labeling of any simulated or synthetic evaluation results

This requirement exists because novel model artifacts without model cards are invisible to the
community. A discoverable, well-documented model card is the primary mechanism for directing
users to `pip install carnot` and establishing the Carnot-EBM HuggingFace presence.

Where REQ-SAFE-011 (teacher-duration invariant) applies, the model card MUST cite it.

**Spec traces:** SCENARIO-PUBLISH-001

---

### SCENARIO-PUBLISH-001: HuggingFace Artifact Preparation

**Given** two production-quality models exist (StepLevelJEPAProbe from Exp 738,
  KAN Tier 0b from Exp 735) with validated weights and evaluation metrics
**When** the operator runs `models/hf_upload_commands.sh` after `huggingface-cli login`
**Then** both models are published to HuggingFace with complete model cards,
  safetensors weights, and config JSON — all satisfying REQ-PUBLISH-001.

**Acceptance criteria:**
- Model cards have no emojis (professional presentation standard).
- Config JSON contains all required fields (model_type, metrics, architecture, training_data).
- Upload script references valid local file paths.
- `honest_verdict` is one of `{"hf_artifacts_ready", "hf_artifacts_partial", "hf_jepa_weights_missing"}`.

**Spec traces:** REQ-PUBLISH-001

### REQ-LOADER-010: Gemma4 Models MUST Use GemmaTransformersLoader

All model loading for `google/gemma-4-*` HuggingFace model IDs MUST use
`GemmaTransformersLoader`.  The llama.cpp backend MUST NOT be used for any
`google/gemma-4-*` model until the tokenizer bug (llama.cpp issue #21516) is
confirmed fixed upstream.  This requirement covers non-GGUF (FP16) model loading;
GGUF-quantized variants loaded via `Gemma4QuantizedLoader` are excluded because
the Q4_K_M GGUF format bypasses the problematic tokenizer path.

**Rationale:** RETRO-028: llama.cpp's Gemma4 tokenizer emits infinite `<unused8>`
tokens (token_id=14), causing 0% accuracy on all benchmarks.  This blocked Gemma4
experiments in milestones .55, .56, .57, and .58.

**Acceptance criteria:**
- All call sites loading `google/gemma-4-E4B-it` (non-GGUF) use `GemmaTransformersLoader`.
- Exp 768 loader_test_passed=True: `GemmaTransformersLoader.generate("Hello", max_new_tokens=5)` returns text with no `<unused>` tokens.
- `GemmaTransformersLoader.is_valid_output(result)` returns True for the 5-token smoke test.

**Implementation Status:** Planned (Exp 768)

### SCENARIO-LOADER-010: GemmaTransformersLoader Smoke Test Passes

**Given** `GemmaTransformersLoader("google/gemma-4-E4B-it", device="cuda:0")`
**When** `.load()` then `.generate("Hello", max_new_tokens=5)` is called
**Then** the returned string contains no `<unused8>` / `<unusedN>` tokens and `is_valid_output()` returns True

**Spec traces:** REQ-LOADER-010
**Implementation Status:** Planned (Exp 768)

### REQ-LOADER-011: kill_gpu_zombies() MUST Be Called Before Any Gemma4 Load Attempt

A Gemma4 model load attempt MUST call `kill_gpu_zombies(gpu_index=0)` before any
`GemmaTransformersLoader.load()` call.  The result's `vram_after_mb` MUST be recorded
as `free_vram_mb_after_kill` in the artifact.  If `free_vram_mb_after_kill < 12000`,
the experiment MUST NOT attempt the load and MUST write `honest_verdict="blocked_insufficient_vram"`.

**Rationale:** RETRO-028 Exp 768 failed with CUDA OOM: 14.89 GiB allocation on a 24 GiB
card with ~15 GiB occupied by zombie processes.  With the GPU cleared, 24 GiB - 14.89 GiB
= 9.11 GiB free overhead — sufficient for the loader.

**Acceptance criteria:**
- `kill_gpu_zombies(gpu_index=0)` is called before any `GemmaTransformersLoader.load()`.
- `free_vram_mb_after_kill` is recorded in every artifact, regardless of outcome.
- If `free_vram_mb_after_kill < 12000`, artifact contains `honest_verdict="blocked_insufficient_vram"`.
- If load succeeds, `loader_test_passed=True` is recorded.

**Implementation Status:** Planned (Exp 786)

### SCENARIO-LOADER-011: Insufficient VRAM After Zombie Kill Blocks Load

**Given** `kill_gpu_zombies(gpu_index=0)` returns a result where `vram_after_mb` corresponds
to less than 12000 MB free (i.e., total VRAM minus vram_after_mb < 12000 MB)
**When** the experiment checks the VRAM threshold
**Then** the artifact records `honest_verdict="blocked_insufficient_vram"` and `GemmaTransformersLoader.load()` is NOT called

**Spec traces:** REQ-LOADER-011
**Implementation Status:** Planned (Exp 786)

---

### REQ-PROBE-020: SemanticEnergyProbe Logit-Space Energy Computation

`SemanticEnergyProbe` MUST compute energy as `E = -sum_i log p(t_i)` where `p(t_i)` is
the token probability from logits (TF-IDF as proxy when logits unavailable).  It MUST
group responses into semantic clusters via TF-IDF cosine similarity with threshold=0.9.
`SemanticCluster.compute_cluster_energy(responses)` MUST return the mean of per-response
energies computed as `-sum(log(tfidf_score(token) + eps))`.

**Rationale:** arXiv 2508.14496 ("Semantic Energy", August 2025) shows logit-space energy
outperforms entropy-based detection by retaining intensity information lost during softmax.
The TF-IDF proxy is used for offline/text-only evaluation; real logits provide the full signal.

**Acceptance criteria:**
- `SemanticEnergyProbe().score(text)` returns a non-negative float for any string.
- `SemanticCluster().compute_cluster_energy(responses)` returns the mean of negative log-score sums.
- `SemanticCluster().group_by_semantics(responses)` partitions all responses into clusters.
- Every test in `tests/python/test_experiment_772_semantic_energy_probe.py` passes.

### REQ-PROBE-021: Tier 0g Advisory Flag

`SemanticEnergyProbe` MUST report `semantic_energy_score` and `is_high_energy=True` when
the score exceeds `energy_threshold`.  The probe is ADVISORY — it does NOT short-circuit
the pipeline; Tiers 1-3 still run.  Wiring as Tier 0g requires `auc >= NUP v4 AUC - 0.05`.

**Acceptance criteria:**
- `is_high_energy(text)` returns True when `score(text) > energy_threshold`.
- `is_high_energy(text)` returns False when `score(text) <= energy_threshold`.
- Exp 772 records `tier0g_deployed` (bool) and `honest_verdict` in its artifact.

### SCENARIO-PROBE-030: SemanticCluster Groups High-Similarity Responses

**Given** two responses with cosine similarity >= 0.9
**When** `SemanticCluster(threshold=0.9).group_by_semantics([r1, r2])` is called
**Then** both responses appear in the same cluster (one cluster total)

**Spec traces:** REQ-PROBE-020

### SCENARIO-PROBE-031: is_high_energy Flags Above-Threshold Response

**Given** a `SemanticEnergyProbe(energy_threshold=0.0)`
**When** `is_high_energy("some non-empty response")` is called
**Then** the method returns True (any non-empty text has energy > 0.0)

**Spec traces:** REQ-PROBE-021

### REQ-PUBLISH-010: HuggingFace Upload Requires Authentication

The HuggingFace upload MUST be executed via `huggingface-cli upload`.  Upload MUST NOT
be attempted if `huggingface-cli whoami` returns a non-zero exit code (HF_TOKEN absent or
login session expired).  The honest_verdict MUST be `blocked_hf_not_authenticated` when
the authentication check fails.

### REQ-PUBLISH-011: All Existing Carnot-EBM Model READMEs Must Include pip install carnot

All 16 existing Carnot-EBM model READMEs MUST include a "## Production Use" section
pointing users at `pip install carnot` and clarifying that the per-token activation EBMs
are Phase 1 research artifacts (confidence detection, not correctness).  The update MUST
be idempotent: re-running when the section already exists MUST succeed without re-uploading.

### SCENARIO-PUBLISH-010: Blocked When HF_TOKEN Not Set

**Given** `huggingface-cli whoami` returns exit code 1
**When** `run_experiment(tmpl)` is called
**Then** `honest_verdict == "blocked_hf_not_authenticated"` and no upload is attempted

**Spec traces:** REQ-PUBLISH-010

### SCENARIO-PUBLISH-011: README Updated With pip install carnot

**Given** an existing Carnot-EBM model README without a "## Production Use" section
**When** `update_readme_with_production_section(repo_id)` is called
**Then** the README gains a section containing "pip install carnot" and the GitHub URL

**Spec traces:** REQ-PUBLISH-011

### REQ-PUBLISH-005: HuggingFace Authentication Token MUST Be Stored via SOPS Encryption

**Statement:** The HuggingFace authentication token (HF_TOKEN) MUST be stored at rest using
SOPS encryption (age or PGP key).  Plaintext HF_TOKEN values MUST NOT appear in any
committed file.  The token MUST be decrypted at runtime via `sops -d secrets/hf_token.yaml`
and injected into the conductor environment with `eval $(sops -d ... | grep HF_TOKEN)`.

**Why this matters:**
    Exp 777 (.59) revealed that HF_TOKEN was absent from the conductor environment,
    blocking all model publishing.  The root cause was no standardised secret-injection
    workflow.  SOPS with age keys provides at-rest encryption (keys never committed),
    per-repo access control via .sops.yaml, and a single decryption command that works
    in both interactive and automated (conductor) sessions without requiring a secrets
    manager service.

**Spec traces:** CLAUDE.md security requirements, RETRO-HF-AUTH, Exp 803

### REQ-PUBLISH-006: models/hf_upload_commands.sh MUST Provide Authenticated Push Commands

**Statement:** `models/hf_upload_commands.sh` MUST contain `huggingface-cli upload` commands
for all three model tiers: Ising (carnot-ising-sampler-v1), KAN (carnot-kan-energy-tier),
and EORM (carnot-eorm-55m).  The script MUST source HF_TOKEN from SOPS before calling
huggingface-cli login.  The script MUST be executable and idempotent.

**Why this matters:**
    Without a single authoritative upload script, each publish attempt re-discovers the
    correct repo IDs and file lists from scratch.  A versioned script with SOPS wiring
    ensures the conductor can re-run publishes deterministically without manual token
    injection each time.

**Spec traces:** REQ-PUBLISH-005, Exp 803

### SCENARIO-PUBLISH-009: HF_TOKEN Present; huggingface-cli Login Succeeds; README Updated

**Given** HF_TOKEN is present in the environment (from SOPS decryption or env var)
**And** `huggingface-cli whoami` returns exit code 0
**When** `run_experiment(tmpl)` is called
**Then** at least one model README is updated via `huggingface-cli upload`
**And** `honest_verdict == "hf_models_published"`

**Spec traces:** REQ-PUBLISH-005, REQ-PUBLISH-006

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
| REQ-PUBLISH-001 | Implemented | Exp 752 — model cards, safetensors exports, hf_upload_commands.sh |
| REQ-INFRA-050 | Implemented | Exp 746 — DualGPU retrain made default; sequential deprecated |
| REQ-INFRA-051 | Implemented | Exp 754 — manifest patch applied to research_conductor.py dispatch site |
| REQ-INFRA-052 | Implemented | Exp 754 — pre-flight v10 confirms patch application via guard clause search |
| REQ-INFRA-053 | Implemented | Exp 767 — pre-flight v11 confirms 100% dequeue-site manifest coverage |
| REQ-INFRA-054 | Implemented | Exp 767 — Exps 425, 491, 603, 627 added to exclusion manifest (.58) |
| REQ-INFRA-055 | Implemented | Exp 780 — kill_gpu_zombies() in gpu_zombie_killer.py; wired into ExperimentTemplate.setup_gpu() |
| REQ-INFRA-056 | Implemented | Exp 780 — kill_gpu_zombies() is a no-op when no GPU zombies exist |
| REQ-INFRA-057 | Implemented | Exp 790 — NPU unblock: Option A (GitHub wheel) first, Option B fallback, max 2 strategies |
| REQ-INFRA-058 | Scaffolding | Exp 793 — manifest full-scope audit; patch spec written to results/experiment_793_manifest_full_scope_audit.json |
| REQ-INFRA-059 | Scaffolding | Exp 793 — WARNING-level logging requirement documented; patch required in pick_next_task |
| REQ-LOADER-010 | Planned | Exp 768 — Gemma4 call site audit + GemmaTransformersLoader enforcement |
| REQ-LOADER-011 | Planned | Exp 786 — kill_gpu_zombies() mandatory before Gemma4 load; VRAM threshold guard |
| REQ-PROBE-020 | Implemented | Exp 772 — SemanticEnergyProbe + SemanticCluster in python/carnot/pipeline/semantic_energy_probe.py |
| REQ-PROBE-021 | Implemented | Exp 772 — is_high_energy advisory flag; tier0g_deployed=False (AUC=0.46, below NUP v4 baseline) |
| REQ-PUBLISH-010 | Implemented | Exp 777 — huggingface-cli upload executed; blocked cleanly when HF_TOKEN absent |
| REQ-PUBLISH-011 | Implemented | Exp 777 — all existing Carnot-EBM model READMEs updated with pip install carnot pointer |
| REQ-PUBLISH-005 | Implemented | Exp 803 — SOPS HF_TOKEN spec in docs/sops-hf-token-setup.md |
| REQ-PUBLISH-006 | Implemented | Exp 803 — models/hf_upload_commands.sh with SOPS wiring for all 3 tiers |

## EBM Calibration Alignment (Exp 789)

### REQ-CALIB-001

**Statement:** EBMCalibrator MUST compute Expected Calibration Error (ECE) from energy-binned
accuracy using 10 equal-frequency bins, and MUST apply isotonic regression to learn an
energy -> P(correct) mapping.

**Why this matters:**
    arXiv 2603.06604 "Know When You're Wrong" shows SFT models have well-calibrated
    confidence but RL-trained models are overconfident by 15-25pp.  Carnot energy is
    currently a discriminative signal (violated/not-violated).  This requirement makes
    energy a calibrated probabilistic signal: low energy = high P(correct).

**Rationale:** ECE (Expected Calibration Error) measures the gap between predicted
confidence and observed accuracy.  Equal-frequency binning ensures each bin has
enough samples to estimate accuracy reliably.  Isotonic regression is the standard
non-parametric post-hoc calibration method (Zadrozny & Elkan 2002).

**Spec traces:** Exp 789, arXiv 2603.06604, arXiv 2602.11364

### REQ-CALIB-002

**Statement:** The calibration curve MUST be saved to results/ebm_calibration_curve.json.
ECE_before and ECE_after MUST be reported in the experiment artifact.

**Why this matters:**
    Without persisting the calibration curve, downstream experiments cannot use the
    fitted isotonic regression to convert raw energy scores to calibrated probabilities.

**Spec traces:** Exp 789

### SCENARIO-CALIB-001: Perfectly Calibrated Energies Yield ECE=0.0

**Given** a set of energies where sigmoid(-energy) exactly equals label accuracy in each bin
**When** compute_ece(energies, labels) is called
**Then** ECE == 0.0

**Spec traces:** REQ-CALIB-001

### SCENARIO-CALIB-002: Isotonic Regression Reduces ECE

**Given** a set of uncalibrated energies with ECE_before > 0
**When** fit_isotonic(energies, labels) is applied and ECE_after is computed
**Then** ECE_after <= ECE_before (isotonic regression never worsens calibration on training data)

**Spec traces:** REQ-CALIB-001, REQ-CALIB-002

### REQ-INFRA-058: ExclusionManifest.check() MUST Be Called at ALL Dequeue Sites

**Statement:** ExclusionManifest.check() (via _task_is_excluded()) MUST be called at EVERY
location in the research conductor where a task_id is selected for execution from any queue
or list data structure. A "dequeue site" is any line where a task moves from a data structure
into the dispatch pipeline — including for-loops over RESEARCH_TASKS, .pop() calls,
.popleft() calls, next(iter(...)), random.choice(), and queue.get() patterns. Placing the
manifest check only in the primary dispatch path (pick_next_task) is insufficient if
secondary code paths bypass pick_next_task and touch RESEARCH_TASKS directly.

**Why this matters:**
    Exp 527 appeared in the slowest-5 for 7+ consecutive milestones after being added to
    the exclusion manifest, because the manifest check in pick_next_task was not adjacent
    to the for-loop that iterates RESEARCH_TASKS. The five-line window heuristic used by
    the audit scanner (Exp 793) confirmed the check is present but logically distant —
    making it easy for future refactors to accidentally bypass. Placing the check immediately
    at the point of dequeue (within 5 lines of the loop/pop/choice statement) is the
    enforcement pattern that prevents recurrence. This requirement documents the FULL-SCOPE
    enforcement goal — every dequeue site must independently guard against excluded tasks.

**Spec traces:** Exp 793, RETRO-MANIFEST-FULL-SCOPE

### REQ-INFRA-059: Excluded Tasks MUST Be Logged at WARNING Level Before Skip

**Statement:** When _task_is_excluded(task) returns True (excluded), the conductor MUST
emit a log.warning() that includes the task title, experiment ID, and the exclusion reason
string before skipping the task. The warning MUST use logger.warning() not logger.info()
so that exclusion events appear in stderr-level log aggregators even when INFO logging is
suppressed. Silently skipping an excluded task without a WARNING-level log makes it
impossible to audit whether the manifest check actually fired for a given run.

**Why this matters:**
    The RETRO-MANIFEST-FULL-SCOPE investigation required manually correlating seven
    milestones of conductor logs to confirm Exp 527 ran despite being manifested.
    If every exclusion emitted a WARNING with the experiment ID, any single conductor log
    would have shown the absence of that WARNING — proving immediately that the guard
    did not fire. This requirement transforms exclusion enforcement from implicit
    (absence of evidence) to explicit (presence of WARNING).

**Spec traces:** Exp 793, RETRO-MANIFEST-FULL-SCOPE, REQ-INFRA-058

### SCENARIO-INFRA-067: Conductor Dequeues Exp 527 From Unmanaged Path; Manifest Guard Fires

**Given** Exp 527 is listed in conductor_exclusion_manifest.json
**And** a dequeue site calls _task_is_excluded() on any task with exp_id=527
**When** _task_is_excluded() is evaluated
**Then** is_excluded=True is returned
**And** the conductor emits logger.warning with "EXCLUDED" in the message
**And** the task is skipped without calling run_agent()

**Spec traces:** REQ-INFRA-058, REQ-INFRA-059

### SCENARIO-INFRA-068: Conductor Dequeues Exp 793 (Not in Manifest); Task Runs Normally

**Given** Exp 793 is NOT listed in conductor_exclusion_manifest.json
**When** the conductor evaluates _task_is_excluded() for a task with exp_id=793
**Then** is_excluded=False is returned
**And** the conductor proceeds to call run_agent() with the task prompt

**Spec traces:** REQ-INFRA-058

### REQ-INFRA-060: MILESTONE_PREREQS.md MUST Exist and Gate Experiment Execution

**Statement:** A MILESTONE_PREREQS.md file MUST exist at the project root listing all
IMMEDIATE-class actions from the prior milestone retro. Each action MUST be marked as
either verified_complete or escalated_retro before any milestone experiment runs.
The file MUST contain a checklist that the conductor or operator verifies manually.
Without this gate, the retro process generates documentation overhead with zero
operational improvement, as observed across three consecutive milestones (.59, .60, .61).

**Why this matters:**
    The .61 retro identified that IMMEDIATE-class improvements were documented but never
    applied, because there was no structural enforcement mechanism. The prereqs gate
    converts the retro from a record-keeping exercise into an actionable pre-flight check.

**Spec traces:** Exp 806, RETRO-.61-PREREQS-GATE

### REQ-INFRA-061: JEPA Retrain Scripts MUST Assert augmentation_ratio > 1.0 at Startup

**Statement:** All JEPA retrain experiment scripts MUST assert augmentation_ratio > 1.0
before any model training begins. Failure raises AssertionError with message:
"CPMI corpus not wired in — check training data loader merges all sources."
This invariant catches the Exp 798→799 disconnect where JEPA trained without CPMI triples,
producing the all-time low ood_auc=0.2444 due to missing data augmentation.

**Why this matters:**
    Exp 799 trained for 5+ minutes before the missing wiring was detected manually.
    An assertion at startup would have caught this in under 1 second and preserved
    the experiment slot for a corrected run. The ood_auc=0.2444 result was an
    implementation error, not an algorithmic failure — this requirement prevents recurrence.

**Spec traces:** Exp 806, RETRO-.61-JEPA-ASSERT

### SCENARIO-INFRA-069: Prereqs Gate Reads MILESTONE_PREREQS.md; All IMMEDIATE Items Verified; Gate Passes

**Given** MILESTONE_PREREQS.md exists at project root
**And** all IMMEDIATE-class items are marked verified_complete or escalated_retro
**When** the prereqs gate check runs
**Then** prereqs_gate_ready is returned
**And** experiment execution proceeds normally

**Spec traces:** REQ-INFRA-060

### SCENARIO-INFRA-070: JEPA Retrain Script Startup; augmentation_ratio=1.0 Detected; AssertionError Raised

**Given** a JEPA retrain script is invoked
**And** augmentation_ratio is computed as 1.0 (no CPMI triples augmenting input pairs)
**When** check_cpmi_wiring() is called at startup
**Then** AssertionError is raised with message "CPMI corpus not wired in — check training data loader merges all sources."
**And** training does NOT begin
**And** the experiment writes a blocked artifact

**Spec traces:** REQ-INFRA-061

### REQ-REPAIR-056: GGUF Loader Import Self-Diagnostic

The GGUF model loader MUST succeed `from llama_cpp import Llama` before any inference
experiment proceeds.  If `ImportError` is raised at load time, the experiment MUST
diagnose the error, log the full error message, and attempt auto-repair via
`pip install --upgrade llama-cpp-python`.  If the import still fails after auto-repair,
the experiment writes a blocked artifact with `honest_verdict="still_blocked_import"`.

**Rationale:** Exp 811 produced `honest_verdict="blocked_model_load_failed"` due to a
Python `ImportError` on `carnot.pipeline.gguf_cache`, blocking every live code repair
experiment since milestone .58.  RETRO-028 resolution shifted the gate from OOM to import
error.  The auto-repair loop prevents the same single-package absence from blocking
multiple consecutive milestones.

**Acceptance criteria:**
- When `from llama_cpp import Llama` raises `ImportError`, the error message is logged
  and `import_repair_attempted` is set to `True` in the artifact.
- When auto-repair via `pip install --upgrade llama-cpp-python` succeeds and the subsequent
  import succeeds, `import_repair_succeeded` is set to `True`.
- When auto-repair fails and the import still raises, the artifact sets
  `honest_verdict="still_blocked_import"` and `import_repair_succeeded=False`.

**Spec traces:** REQ-REPAIR-056 (RETRO-GGUF-CACHE-IMPORT, closes milestone .58 blocker)

### SCENARIO-REPAIR-089: GGUF Loader Import Failure Triggers Auto-Repair

**Given** `from llama_cpp import Llama` raises `ImportError` at experiment startup
**When** the experiment's import diagnostic runs
**Then**
  1. The error message is logged with the full exception text.
  2. `subprocess.run(["pip", "install", "--upgrade", "llama-cpp-python"])` is called.
  3. The import is retried.
  4. If the retry succeeds: the experiment proceeds to GPU setup and inference.
  5. If the retry fails: the experiment writes a blocked artifact with
     `honest_verdict="still_blocked_import"`, `import_repair_attempted=True`,
     `import_repair_succeeded=False`.

**Spec traces:** REQ-REPAIR-056

### REQ-VERIFY-145: Cross-Domain PRM Degradation Reporting

**Statement:** For cross-domain PRM evaluation, Carnot MUST compute and report
`cross_domain_degradation = auc_in_dist - auc_ood` for each OOD domain (HumanEval,
ARC-Challenge).  If the maximum degradation across domains exceeds 0.08 (the 8% baseline
published in arXiv 2506.00027), the experiment MUST identify which domain shows the
largest gap.  The artifact MUST include `beats_baseline` (bool), `published_baseline=0.08`,
and `honest_verdict` drawn from {"above_baseline", "at_baseline", "below_baseline",
"data_unavailable"}.

**Rationale:** arXiv 2506.00027 reports that PRMs trained on math reasoning degrade ~8%
AUC when applied to code verification.  Without a concrete cross-domain metric, Carnot
cannot claim its JEPA-based verifier generalises better than the published baseline.
This requirement creates a traceable, reproducible benchmark comparison.

**Acceptance criteria:**
- `cross_domain_degradation_humaneval` and `cross_domain_degradation_arc` are computed
  as `in_dist_auc - auc_domain` and recorded in the artifact.
- `beats_baseline` is True iff `cross_domain_degradation_max <= 0.08`.
- `honest_verdict` is "above_baseline" when beats_baseline is True, "at_baseline" when
  abs(degradation_max - 0.08) <= 0.01, "below_baseline" when degradation_max > 0.09.
- `corroboration_rate` = fraction of 20 VerificationCertificates where z3_verdict
  direction agrees with jepa_energy_delta direction (unsat ↔ high energy).

**Spec traces:** Exp 826, arXiv 2506.00027, arXiv 2601.17223

### SCENARIO-VERIFY-174: Load Exp 825 AUC; Compute Degradation; Emit Certificates for Failed OOD Steps

**Given** Exp 825 results file exists with `auc_gsm8k`, `auc_humaneval`, `auc_arc`,
  `overall_ood_auc`, and 20 `verification_certificates`
**And** Exp 824 results file exists with `in_dist_auc`
**When** Exp 826 runs cross-domain PRM benchmark
**Then**
  1. `cross_domain_degradation_humaneval = in_dist_auc - auc_humaneval` is computed.
  2. `cross_domain_degradation_arc = in_dist_auc - auc_arc` is computed.
  3. `cross_domain_degradation_max = max(degradation_humaneval, degradation_arc)`.
  4. `beats_baseline = (cross_domain_degradation_max <= 0.08)`.
  5. `corroboration_rate` is computed from Exp 825 certificates (unsat ↔ energy_delta > 0).
  6. If degradation_max > 0.08: `worst_domain` is identified as the higher-degradation domain.
  7. Artifact is written with all required fields per REQ-VERIFY-145.
  8. `honest_verdict` reflects the degradation comparison against 0.08 baseline.

**Spec traces:** REQ-VERIFY-145

### REQ-VERIFY-146: ActivationJailbreakProbe Layer Activation Extraction

**Statement:** ActivationJailbreakProbe MUST extract intermediate layer activations
from a small transformer model (Qwen3.5-0.8B or fallback hash projection) at layers
[4, 8, 12, 16] and train a LogisticRegression probe on labeled jailbreak/benign examples.
CPU inference latency for the probe forward pass (activation extraction + LR predict)
MUST be < 1 ms per query.

**Rationale:** arXiv 2602.11495 shows that jailbreak prompts produce a linear signal
in intermediate transformer layers detectable by logistic regression trained on 100
examples with AUC >= 0.90 at < 1 ms CPU latency.  This is orthogonal to the TF-IDF
KAN signal in Tier 0h: the KAN detects surface n-gram patterns, the activation probe
detects where the prompt sits in the model's internal representation space.

**Acceptance criteria:**
- `extract_activations(prompt)` returns np.ndarray of shape (n_layers * hidden_dim,).
- `train(prompts_labeled)` returns a fitted sklearn.linear_model.LogisticRegression.
- `evaluate(probe, test_labeled)` returns (auc: float, latency_ms: float).
- latency_ms < 1.0 for the LR forward pass alone (activation extraction excluded from
  latency budget since it is amortised across all probes in the pipeline).

**Spec traces:** Exp 828, arXiv 2602.11495

### REQ-VERIFY-147: ActivationJailbreakProbe Viability Threshold

**Statement:** ActivationJailbreakProbe probe_auc MUST be >= 0.85 on a 50/50 balanced
holdout (25 jailbreak + 25 benign, after 60/40 train/test split from 100 total) to be
considered viable for production wiring alongside Tier 0h KAN.  If probe_auc >= 0.85
AND latency_ms < 1.0 then probe_viable MUST be True; otherwise probe_viable MUST be False.

**Rationale:** The 0.85 AUC threshold is the minimum for a useful complementary signal.
Below this level, the probe adds false positives without sufficient jailbreak detection
gain to justify the additional inference cost.  The 0.85 threshold is 5 percentage
points below the published 0.90 baseline to account for the smaller training set (60
examples vs. 100 in the paper) and the synthetic vs. real JailbreakBench distribution gap.

**Acceptance criteria:**
- On 40-example holdout (20 jailbreak + 20 benign): probe_auc is computed and recorded.
- probe_viable = (probe_auc >= 0.85 and latency_ms < 1.0).
- honest_verdict in {"probe_viable", "probe_partial", "probe_not_viable"}.

**Spec traces:** Exp 828, arXiv 2602.11495

### SCENARIO-VERIFY-175: Activation Probe Train/Eval on Synthetic JailbreakBench

**Given** 50 synthetic jailbreak prompts (seed=42) + 50 synthetic benign prompts (seed=42)
**And** 60/40 train/test split: 30 jailbreak + 30 benign train, 20 jailbreak + 20 benign test
**When** ActivationJailbreakProbe.train() is called on 60 labeled prompts
**And** ActivationJailbreakProbe.evaluate() is called on 40 labeled holdout prompts
**Then**
  1. extract_activations returns shape (n_layers * hidden_dim,) for every prompt.
  2. LogisticRegression fits without error on 60 examples.
  3. probe_auc is computed from ROC AUC on the 40-example holdout.
  4. latency_ms is measured as mean of 20 predict_proba calls on one prompt.
  5. probe_viable = (probe_auc >= 0.85 AND latency_ms < 1.0).
  6. If probe_viable=True: honest_verdict = "probe_viable".
  7. If probe_auc >= 0.85 but latency_ms >= 1.0: honest_verdict = "probe_partial".
  8. If probe_auc < 0.85: honest_verdict = "probe_not_viable".
  9. Artifact written to results/experiment_828_activation_jailbreak_probe.json.

**Spec traces:** REQ-VERIFY-146, REQ-VERIFY-147

### REQ-INFRA-062: HuggingFace Model Cards MUST Include Phase 1 Disclaimer

**Statement:** All HuggingFace model cards published under the Carnot-EBM organisation
MUST include a disclaimer section with the exact text: "Phase 1 research artifact.
Trained on simulated data unless explicitly stated as live-GPU-validated. Do not use
in production without independent validation."

**Rationale:** Carnot-EBM's first 16 published models were trained on simulated data
and have not been validated on live GPU runs.  Without an explicit disclaimer, downstream
users may mistake these research artifacts for production-ready models, eroding trust
in the project and violating the project's own honesty principle ("all headline results
must have live GPU provenance").  This requirement ensures every model card is honest
about its provenance.

**Acceptance criteria:**
- huggingface_hub.list_models(author="Carnot-EBM") returns >= 1 model after this update.
- Each returned model's README contains the substring "Phase 1 research artifact".
- The disclaimer appears before any usage section.

**Spec traces:** Exp 829, CLAUDE.md honesty principle

### SCENARIO-INFRA-070: Carnot-EBM Model Count >= 17 After Exp 829 Publish

**Given** 16 existing Carnot-EBM models on HuggingFace (trained on simulated data)
**And** at least one new model artifact (JEPA v23 or IsingConstraintInjector) is eligible for publish
**When** experiment_829_huggingface_v3_publish.py runs with a valid HF_TOKEN
**Then**
  1. huggingface_hub.list_models(author="Carnot-EBM") returns >= 17 models.
  2. Every model card in the list contains "Phase 1 research artifact".
  3. n_cards_updated >= 1 (at least one existing README was updated).
  4. honest_verdict in {"hf_publish_success", "hf_publish_partial", "hf_auth_blocked"}.
  5. Artifact written to results/experiment_829_huggingface_v3_publish.json.

**Spec traces:** REQ-INFRA-062, Exp 829
