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
| REQ-LOADER-010 | Planned | Exp 768 — Gemma4 call site audit + GemmaTransformersLoader enforcement |
| REQ-PROBE-020 | Implemented | Exp 772 — SemanticEnergyProbe + SemanticCluster in python/carnot/pipeline/semantic_energy_probe.py |
| REQ-PROBE-021 | Implemented | Exp 772 — is_high_energy advisory flag; tier0g_deployed=False (AUC=0.46, below NUP v4 baseline) |
| REQ-PUBLISH-010 | Implemented | Exp 777 — huggingface-cli upload executed; blocked cleanly when HF_TOKEN absent |
| REQ-PUBLISH-011 | Implemented | Exp 777 — all existing Carnot-EBM model READMEs updated with pip install carnot pointer |
