# Research Roadmap v39 — Milestone 2026.04.39

**Status:** Proposed
**Milestone:** 2026.04.39
**Title:** Close the Credibility Gap — JIT VRAM, Seventh Attempt, DualGPU Verified
**Planned Experiments:** 513–524 (12 experiments)
**Planned Date:** 2026-04-19 onwards

---

## What Milestone 2026.04.38 Proved

Milestone .38 achieved three meaningful closures:

1. **RETRO-048 RESOLVED** — `Gemma4QuantizedLoader` confirmed `is_within_budget=True`. The five-milestone VRAM budget problem is solved. Quantized Gemma4-INT4 fits alongside the conductor process within 24 GiB GPU 0.

2. **RETRO-031 CLOSED** — KAEM sampling advantage found on `gaussian_mixture` distribution family (Exp 508). Three-milestone carry resolved.

3. **RETRO-050 CLOSED** — Energy-magnitude replay outperforms SuRe surprise replay (`isolation_improvement=1.1172` vs SuRe's `-0.1172`). The energy function IS the right priority signal for replay selection — this validates the core architectural principle.

But the credibility benchmarks (Exps 502-504) deferred for the **sixth consecutive milestone**. The root cause shifted: the VRAM budget forecast passed, but runtime CUDA OOM still blocked model loading. The budget problem is solved; the execution-time load sequencing is not.

**New RETRO items from .38:**

| RETRO | Priority | Description |
|-------|----------|-------------|
| RETRO-051 | Critical | Runtime CUDA OOM despite budget forecast passing. Fix: just-in-time VRAM check immediately before each `model.load()` call, not just at planning time. Retry after 30s cool-down. |
| RETRO-052 | Medium | DualGPU sweep (Exp 505) found 0 scripts to patch — GPU 1 still 0% compute. Need controlled test routing one model to `cuda:1`. |

**Carried open RETRO items:**

| RETRO | Priority | Description |
|-------|----------|-------------|
| RETRO-033 | Critical | Live 100q positive — SIXTH consecutive milestone miss. |
| RETRO-038 | Critical | Live 200q VeriCoT+VPRM statistically significant result. |
| RETRO-039 | High | GSM-Symbolic adversarial thesis unconfirmed. |
| RETRO-049 | Medium | NUP Probe v3 AUC = 0.400, below 0.700 Tier 0c threshold. v3 feature enrichment failed; need architectural redesign (contrastive training). |

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: Runtime VRAM OOM after budget fix (FR-12 — Verifiable Reasoning)

RETRO-048 is resolved at the budget level: the planning-time VRAM forecast correctly predicts that quantized Gemma4 (~9 GiB) + conductor process (~9 GiB) fits within 24 GiB. But three experiments (502/503/504) still failed at runtime with CUDA OOM errors.

**Root cause (RETRO-051):** The VRAM forecast is computed once at planning time against a stale VRAM snapshot. By the time `model.load()` executes — possibly minutes later, after other setup code runs — the VRAM state has changed. Another process may have allocated memory; JAX JIT compilation may have grown the conductor footprint; or the quantized model loads in stages that temporarily exceed the budget.

**The fix is just-in-time VRAM gating.** A `JITVRAMCheck.gate_model_load(model_id, min_free_gb, retry_wait_s=30)` method that:
1. Calls `pynvml.nvmlDeviceGetMemoryInfo()` immediately before the load
2. If available < required: wait 30s and re-check once
3. If still insufficient: emit a `deferred_retro_051` artifact with the actual VRAM measurement

This is a 30-line change that converts silent CUDA OOM into a fast-fail with a diagnostic. RETRO-051 is the last link in a six-milestone chain. Fix it, and the credibility benchmarks run.

**Evidence of urgency:** Six consecutive milestones (.33/.34/.35/.36/.37/.38) ended with RETRO-033 open. The progression: GPU offline → env not propagating → zombie VRAM → conductor VRAM → VRAM budget → runtime VRAM state. Milestone .39 closes the final link.

### Gap 2: GPU 1 at 0% forward-pass compute (Hardware Efficiency)

The retroactive DualGPU sweep (Exp 505) found `n_scripts_found=0, n_scripts_patched=0`. Either all scripts were already patched by prior harness work, or the sweep's pattern matching was too narrow. GPU 1 (`cuda:1`, 24 GiB RTX 3090) contributed 0% forward-pass compute across all milestones.

A controlled dual-GPU test (Exp 517) that explicitly loads one model on `cuda:0` and another on `cuda:1`, then measures `nvmlDeviceGetUtilizationRates()` on each GPU during inference, will either:
- Confirm GPU 1 utilization > 0% (DualGPU harness is working, sweep missed it), OR
- Show GPU 1 still at 0% (device_map routing still broken, needs deeper fix)

Either result is actionable. Currently the project is running at 50% VRAM capacity and unknown compute capacity. The credibility benchmarks (Exps 514-516) ideally load Gemma4-INT4 on `cuda:0` and Qwen3.5-0.8B on `cuda:1` simultaneously, cutting benchmark wall time approximately in half.

### Gap 3: No real-data self-learning loop (FR-11)

JEPA retrain v5 (Exp 510) produced `fr11_synthetic_only` because the credibility benchmarks (502-504) were deferred. FR-11 requires the self-learning loop to close on real data — the energy function evaluates real model outputs, generates real violation labels, and trains a predictor on real (partial_response, violation) pairs.

Once Exps 514-516 run:
- They generate ~200-300 real CoT pairs with FOVER labels
- JEPA retrain v6 (Exp 522) trains on these pairs using the LeWorldModel two-term objective (arXiv 2603.19312) for stable training
- Target: AUC >= 0.800 on held-out real pairs (vs the synthetic-only 0.967 ceiling from Exp 492)

The credibility benchmarks and FR-11 are thus tightly coupled: live data from the benchmark is the training corpus for the predictor.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MILESTONE 2026.04.39 PIPELINE                      │
│                                                                        │
│  PHASE 1: RETRO-051 Fix (JIT VRAM — the final load-sequence blocker) │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 513: JITVRAMCheck + Sequential Model Loading                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 2: Credibility (RETRO-033/038/039 — 7th/5th/5th attempts,    │
│                         unblocked by JIT VRAM fix)                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 514: Live 100q Precision v7 (RETRO-033 — 7th attempt)       │ │
│  │ Exp 515: Live 200q VeriCoT+VPRM v5 (RETRO-038 — 5th attempt)   │ │
│  │ Exp 516: GSM-Symbolic Adversarial v5 (RETRO-039 — 5th attempt)  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 3: DualGPU Proof (RETRO-052 — controlled execution test)      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 517: Controlled DualGPU Parallel Execution Test              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 4: Legacy Debt (BatchedInferenceRunner migration sprint)       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 518: Top-20 Legacy Script BatchedInferenceRunner Migration   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 5: New Research (arxiv 2025-2026 findings)                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 519: CIKANEnergy — Constraint-Informed KAN (arXiv 2412.03710)│ │
│  │ Exp 520: LeWorldModel-JEPA Stable Training (arXiv 2603.19312)   │ │
│  │ Exp 521: Hallucination Basin Detector (arXiv 2604.04743)         │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 6: Self-Learning (FR-11 mandatory)                             │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 522: JEPA Live Retrain v6 (on Exps 514-515 CoT pairs)       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 7: NUP Probe v4 Contrastive Redesign (RETRO-049)              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 523: NUP Probe v4 — Contrastive Training Objective           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 8: Retrospective                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 524: Milestone 2026.04.39 Retrospective                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

```
Exp 513 (JIT VRAM)  ──────► Exp 514 (100q v7) ──► Exp 515 (200q v5)
                                                  ──► Exp 516 (adversarial v5)
                                                  ──► Exp 522 (JEPA retrain v6)

Exps 514/515 (live CoT pairs) ──► Exp 522 (JEPA retrain v6)
Exps 514/515 (live CoT pairs) ──► Exp 523 (NUP Probe v4, GPU features)

Exp 517 (DualGPU) ─────── independent (GPU)
Exp 518 (batching) ─────── independent (CPU)
Exp 519 (CIKAN) ─────────── independent (CPU)
Exp 520 (LeWorldModel) ──── independent (CPU)
Exp 521 (Basins) ─────────── independent (CPU)

Exp 524 (Retro) ──► requires all prior experiments complete
```

---

## Phase Descriptions

### Phase 1: RETRO-051 Fix (Experiment 513)

**Exp 513 — JITVRAMCheck + Sequential Model Loading:** The gap between planning-time VRAM forecast and runtime load is the last link in a six-milestone blocking chain. Implement `JITVRAMCheck` in `python/carnot/pipeline/jit_vram_check.py`:

- `JITVRAMCheck(device_id: int = 0)` — wraps pynvml to query real-time VRAM state
- `gate_model_load(model_id: str, required_gb: float, retry_wait_s: int = 30) -> JITVRAMResult` — checks VRAM immediately before `model.load()`, waits and retries once if insufficient, returns `is_cleared` bool + `available_gb` measurement
- `sequential_load_gate(model_specs: List[Dict]) -> List[JITVRAMResult]` — gates each model in a list sequentially, ensuring each model clears VRAM before the next loads

Wire `JITVRAMCheck.gate_model_load()` into `Gemma4QuantizedLoader.load()` and `GemmaTransformersLoader` as a pre-load check. CI path: when pynvml is not installed, return `is_cleared=True` with `available_gb=24.0` (mocked). This ensures the test suite passes without GPU.

**Key requirement:** The JIT check must happen immediately before the actual `model.from_pretrained()` or `llama_cpp.Llama()` call — not at script startup, not at gate initialization.

### Phase 2: Credibility Benchmarks (Experiments 514–516)

These three experiments are the credibility goal of the entire research program. They have been deferred for SIX consecutive milestones. Phase 1 removes the final blocker.

**Exp 514 — Live 100q Precision v7 (RETRO-033):** 100 GSM8K questions, 2 models:
- Gemma4-INT4 (Gemma4QuantizedLoader) on `cuda:0`, guarded by JITVRAMCheck
- Qwen3.5-0.8B on `cuda:1`, guarded by JITVRAMCheck
- VeriCoT+VPRM+CRANE extraction (full stack)
- Wilson 95% CI reported for baseline and pipeline accuracy
- `is_positive=True` closes RETRO-033 after six milestones
- Write 100 CoT pairs to `results/exp514_cot_pairs.json` for Exp 522

**Exp 515 — Live 200q VeriCoT+VPRM v5 (RETRO-038):** 200 GSM8K questions with full extraction stack, using LongRunBenchmarkExecutor for checkpoint/resume. `is_statistically_positive=True` (Wilson 95% CI lower bound > 0) is the **first publishable credibility claim** for Carnot. Write 200 CoT pairs to `results/exp515_cot_pairs.json` for Exp 522.

**Exp 516 — GSM-Symbolic Adversarial v5 (RETRO-039):** Apple arXiv 2410.05229 adversarial benchmark. The thesis: Carnot's improvement should be LARGER on adversarial variants because Ising constraint verification is independent of surface form — a symbolic substitution that makes the problem look different but leaves the arithmetic constraints unchanged does not fool the Ising sampler. Expected: Carnot degrades less than baseline on adversarial variants.

### Phase 3: DualGPU Proof (Experiment 517)

**Exp 517 — Controlled DualGPU Parallel Execution Test (RETRO-052):** The DualGPU sweep (Exp 505) found zero scripts to patch. This experiment answers definitively: is GPU 1 actually running forward-pass compute?

Design: (1) Load Gemma4-INT4 with explicit `device_map={'': 'cuda:0'}`, (2) load Qwen3.5-0.8B with explicit `device_map={'': 'cuda:1'}`, (3) run 10 inference passes on each model simultaneously in threads, (4) measure `nvmlDeviceGetUtilizationRates()` on both GPUs during inference. Report `gpu1_compute_pct` — if > 10%, DualGPU is verified working; if 0%, root cause is still unresolved.

The key deliverable is a truthful `gpu1_compute_pct` measurement, not a binary pass/fail.

### Phase 4: Legacy Debt (Experiment 518)

**Exp 518 — Top-20 Legacy BatchedInferenceRunner Migration:** The .38 retro identified that 77 scripts (Exp 481 audit) still use sequential inference loops. The top-5 slowest experiments (Exps 221 at 78min, 226 at 73min, 260 at 74min, 308 at 105min, 425 at 76min) predate the `BatchedInferenceRunner` pattern.

Task: read the Exp 481 audit output, rank scripts by measured wall time (from `ops/metrics.md` or the retro JSONs), migrate the top-20 highest-impact scripts to `BatchedInferenceRunner`. Write a migration report (`results/experiment_518_batching_migration.json`) with: `n_migrated`, `estimated_wall_time_savings_min`, `scripts_migrated`, `honest_verdict`.

This is a CPU-only experiment (code migration, no GPU inference). It addresses the .38 retro finding that "legacy batching migration (~3%)" is an achievable wall-time savings.

### Phase 5: New Research (Experiments 519–521)

**Exp 519 — CIKANEnergy — Constraint-Informed KAN (arXiv 2412.03710):** CIKAN initializes KAN splines with higher resolution near hard constraint boundaries — the energy function naturally concentrates complexity where constraints are most active. Implement `CIKANEnergy` in `python/carnot/models/cikan_energy.py`:

- Extends `KAEMEnergy` with `constraint_boundaries: List[float]` initialization parameter
- Near each boundary, the spline's knot density is increased by a factor of `k` (default k=4)
- `fit_with_constraints(data, boundaries)` uses the boundary-aware initialization
- Benchmark vs `KAEMEnergy` on constraint verification problems where ground-truth boundaries are known

Expected: sharper energy gradients near constraint boundaries, better AUROC on near-boundary examples.

**Exp 520 — LeWorldModel-JEPA Stable Training (arXiv 2603.19312):** Current JEPA training is unstable across sessions (AUC regression from 0.667 to 0.400 in Exp 472, requiring curriculum recovery in Exp 492). The two-term LeWorldModel objective provides principled regularization:

```
L_total = L_prediction + λ * L_regularization
L_prediction = MSE(predicted_embedding, actual_embedding)
L_regularization = KL(q(z) || N(0, I))  # KL to standard Gaussian
```

Implement `LeWorldModelJEPATrainer` in `python/carnot/pipeline/lw_jepa_trainer.py`. Apply to CoT step embeddings using the same training pairs as prior JEPA experiments. Compare training stability (AUC variance across 3 runs) and final AUC vs current curriculum approach. CPU-only (small model, ~15M params).

**Exp 521 — Hallucination Basin Detector (arXiv 2604.04743):** Hallucinations arise from shallow latent-space basins with high escape probability. Implement `HallucinationBasinDetector` that:

1. Extracts hidden-state trajectory from LLM generation (sequence of layer activations)
2. Estimates local basin depth using finite-difference energy approximation: `depth(t) = energy(x_t) - min(energy(x_t + δ_i))` over k random perturbations
3. Returns a scalar `basin_risk_score` — high score means the trajectory is in a shallow basin (hallucination-prone)

Benchmark vs SpilledEnergy (Tier 0b) as an alternative Tier 0c signal. CPU-only with synthetic activation tensors; GPU optional for real model activations.

### Phase 6: Self-Learning (Experiment 522)

**Exp 522 — JEPA Live Retrain v6 (FR-11 Tier 3):** Retrain JEPA predictor on:
- Primary: CoT pairs from Exps 514-515 (real live data if available)
- Fallback: synthetic pairs from existing FOVER-labeled corpus

Use `LeWorldModelJEPATrainer` from Exp 520 (two-term objective) for stable training without curriculum scheduling. Target: AUC >= 0.800 on held-out pairs. Write checkpoint to `results/jepa_predictor_522_live.safetensors`.

This is the mandatory self-learning experiment per research-program.md. FR-11 relay status: Exp 510 produced `fr11_synthetic_only` because Exps 502-504 were deferred. Exp 522 closes the loop on real data if Exps 514-515 succeed.

### Phase 7: NUP Probe v4 Contrastive (Experiment 523)

**Exp 523 — NUP Probe v4 Contrastive Redesign (RETRO-049):** v3 achieved AUC = 0.400 (below 0.700 Tier 0c threshold) despite feature enrichment with CLAP cross-layer attention. The feature extraction is richer but the training objective is unchanged — binary cross-entropy on individual steps.

The redesign: **contrastive training** rather than per-step binary classification.

- Treat (correct CoT step, incorrect CoT step) pairs as positives/negatives
- Train the probe to maximize the energy gap between correct and incorrect steps: `L_contrastive = max(0, margin - (E(x_incorrect) - E(x_correct)))`
- Use CoT pairs from Exps 514-515 (real data) or the FOVER-labeled corpus (synthetic fallback)
- The contrastive objective directly trains the probe to distinguish high-energy from low-energy steps, which is exactly the verification signal Carnot needs

Target: AUC > 0.700 for Tier 0c promotion. GPU recommended for feature extraction; CPU-only for training.

### Phase 8: Retrospective (Experiment 524)

**Exp 524 — Milestone 2026.04.39 Retrospective:** Analyze all 12 experiments. Mandatory headline check: did RETRO-033 finally close? If `is_positive=True` from Exp 514, this is the first positive live verify-repair number after 7 consecutive attempts. Report: execution efficiency, GPU utilization measurements, RETRO closure rates, wall-time improvements, top-3 bottlenecks for .40.

---

## Arxiv Findings Incorporated

| Finding | Paper | Experiment |
|---------|-------|-----------|
| Hallucination basin dynamics (latent-space attractor geometry) | arXiv 2604.04743 | Exp 521 |
| LeWorldModel JEPA stable two-term training objective | arXiv 2603.19312 | Exp 520, Exp 522 |
| CIKAN constraint-informed KAN spline initialization | arXiv 2412.03710 | Exp 519 |
| Constrained decoding with near-zero overhead (DOMINO/XGrammar) | arXiv 2604.14862 | Filed for .40+ |
| Low-rank logit energy landscape (2-11 singular components) | arXiv 2604.04384 | Filed for .40+ |

---

## Hardware Requirements

| Experiment | Hardware | Minimum VRAM | Duration Estimate |
|-----------|----------|-------------|-------------------|
| Exp 513 | CPU only | — | 20 min |
| Exp 514 | 2x RTX 3090 | 18 GiB (JIT gated) | 120 min |
| Exp 515 | 2x RTX 3090 | 18 GiB (JIT gated) | 150 min |
| Exp 516 | 2x RTX 3090 | 18 GiB (JIT gated) | 120 min |
| Exp 517 | 2x RTX 3090 | 18 GiB | 30 min |
| Exp 518 | CPU only | — | 30 min |
| Exp 519 | CPU only | — | 25 min |
| Exp 520 | CPU only | — | 25 min |
| Exp 521 | CPU only (GPU optional) | — | 25 min |
| Exp 522 | GPU recommended | 8 GiB | 30 min |
| Exp 523 | GPU recommended | 8 GiB | 30 min |
| Exp 524 | CPU only | — | 20 min |

**JIT VRAM constraint:** After Exp 513, all GPU experiments gate model loading with `JITVRAMCheck.gate_model_load()` which checks real-time VRAM immediately before the load call. This converts runtime OOM into a fast-fail with diagnostic output. The expected VRAM budget (from Exp 500):

- Conductor process: ~0 GiB (CPU routing via `JAX_PLATFORMS=cpu`)
- Gemma4-INT4 (Q4_K_M): ~8-10 GiB (cuda:0)
- Qwen3.5-0.8B: ~1.5 GiB (cuda:1)
- Total: ~9.5-11.5 GiB (fits with ~12.5-14.5 GiB headroom across both GPUs)

---

## Success Criteria

| Criterion | Experiment | Target |
|-----------|-----------|--------|
| retro_051_resolved | Exp 513 | JITVRAMCheck gates model load, no silent OOM |
| retro_033_closed | Exp 514 | `is_positive=True` on 100q live benchmark |
| retro_038_closed | Exp 515 | `is_statistically_positive=True` (Wilson 95% CI lower > 0) |
| retro_039_confirmed | Exp 516 | Carnot degrades less than baseline on adversarial variants |
| gpu1_compute_verified | Exp 517 | `gpu1_compute_pct > 10%` during dual-model inference |
| batching_migration_done | Exp 518 | Top-20 legacy scripts migrated to BatchedInferenceRunner |
| cikan_advantage | Exp 519 | AUROC improvement on near-boundary examples vs KAEMEnergy |
| leworldmodel_stable | Exp 520 | AUC variance across 3 runs < 0.05 (vs current ~0.4 variance) |
| basin_detector_viable | Exp 521 | AUC > SpilledEnergy baseline on synthetic CoT |
| fr11_live_relay | Exp 522 | AUC >= 0.800 on held-out live pairs (or `fr11_synthetic_fallback` if Exps 514/515 defer) |
| nup_probe_promoted | Exp 523 | AUC > 0.700 (Tier 0c threshold via contrastive training) |

---

## Meta-Reflection from .38

The .38 retrospective identified four top priorities for .39:

1. **(a) Run RETRO-033 benchmark FIRST before any infrastructure work.** — Adopted: Phase 1 (Exp 513) fixes the final infrastructure blocker, Phase 2 (Exps 514-516) runs the benchmarks immediately. No new infrastructure work before the benchmarks.

2. **(b) Dedicated BatchedInferenceRunner migration sprint for top-20 legacy scripts.** — Adopted: Exp 518 (Phase 4) is a dedicated migration sprint.

3. **(c) Enforce DualGPU parallel execution as default for dual-model benchmarks.** — Adopted: Exp 517 (Phase 3) verifies GPU 1 actually runs compute; Exps 514-516 use explicit `cuda:0`/`cuda:1` device assignment with JIT VRAM gates.

4. **(d) Two-milestone RETRO carry limit.** — Policy adopted: any RETRO item not closed in 2 consecutive milestones gets escalated to a BLOCKING prerequisite (not just a priority). RETRO-033 has carried 6 milestones — Exp 514 is a blocking prerequisite for .40's planning phase.

The estimated 18% additional wall-time savings from the .38 retro (DualGPU parallelism 8%, legacy batching 3%, VRAM scheduling 3%, session consolidation 1%, RETRO carry limit 3%) becomes achievable once Exps 513 and 517 close their respective RETROs and Exp 518 addresses the legacy debt.
