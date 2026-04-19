# Research Roadmap v37 — Milestone 2026.04.37

**Status:** Proposed
**Milestone:** 2026.04.37
**Title:** Break the VRAM Deadlock — Credibility at Last, JEPA Recovery, and Surprise-Driven Replay
**Planned Experiments:** 487–499 (13 experiments)
**Planned Date:** 2026-04-19 onwards

---

## What Milestone 2026.04.36 Proved

Milestone .36 improved **GPU character** (zombie-held → work-held at GPU 0) and **process adoption** (60% retro improvement adoption, up from 50%) while leaving the headline credibility question unanswered for the **fourth consecutive milestone**.

Root cause analysis from ops/retro .36:

- **RETRO-044 is the master blocker:** GPUVRAMGate checks VRAM first, kills zombies after failing. This is backwards. Zombie processes often re-acquire VRAM between the kill signal and the re-check, making the gate ineffective. RETRO-033 (100q live positive), RETRO-038 (200q statistically significant), and RETRO-039 (GSM-Symbolic adversarial thesis) all deferred_to_gpu in .36 despite GPUVRAMGate existing — because the kill order was wrong.
- **JEPA AUC continued regressing:** 0.667 → 0.400 → 0.281. The quality-gated retrain (Exp 477) made it WORSE, not better. Root cause: the quality filter removed too many pairs (low label_confidence < 0.7), leaving a heavily imbalanced corpus that caused the predictor to overfit to the dominant class. Curriculum training — not quality filtering — is the correct fix.
- **GPU 1 structurally underloaded (11%):** device_map='auto' routes layer storage to GPU 1 but all forward-pass compute runs on GPU 0. 53/64 dual-model harnesses still don't use explicit cuda:1. DualGPUHarness (Exp 480) was audited but not enforced — harnesses need to be patched, not just documented.
- **Enforcement gap persists:** Batching hook (RETRO-045) and thermal gate (RETRO-046) were not implemented. Both require CI/pre-commit tooling, not conductor documentation.

**Open RETRO items carried into .37:**

| RETRO | Priority | Description |
|-------|----------|-------------|
| RETRO-031 | Low | KAEM no 5x speedup crossover at n_vars=1000. Extend to n=5000. |
| RETRO-033 | Critical | Live 100q positive — FOURTH consecutive milestone miss. Blocked by RETRO-044. |
| RETRO-038 | Critical | 200q VeriCoT+VPRM CUDA OOM — THIRD attempt needed. Blocked by RETRO-044. |
| RETRO-039 | High | GSM-Symbolic thesis not confirmed — THIRD attempt. Blocked by RETRO-044. |
| RETRO-040 | Critical | JEPA AUC regressed 0.667→0.400→0.281. Quality-gate approach failed. Curriculum training required. |
| RETRO-044 | Critical | GPUVRAMGate kill-order wrong: checks VRAM then kills. Must kill FIRST, then check. |
| RETRO-045 | High | Batching enforcement: 77 violations documented, no pre-commit hook. |
| RETRO-046 | High | GPU thermal gate: third consecutive milestone miss. |
| RETRO-047 | Medium | NUP Probe AUC=0.600, below 0.700 Tier 0c threshold. Needs logprob features. |

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: RETRO-044 blocks all credibility work (FR-12 — Verifiable Reasoning)

GPUVRAMGate's kill order is inverted. The current logic: check VRAM → if < 8GB free, kill zombies → wait → recheck. The bug: zombie processes hold VRAM in GPU driver memory for 5-15 seconds after receiving SIGKILL because the driver must drain their queued operations. If the recheck runs before the drain completes, free VRAM appears below threshold and the experiment defers. The fix: kill zombies FIRST → sleep 15s (driver drain time) → check VRAM → proceed or defer. This single-line reorder would have prevented all four deferred_to_gpu experiments in .36.

**Evidence:** Exp 476 (100q), Exp 478 (200q), Exp 479 (adversarial) all hit CUDA OOM or gpu_vram_insufficient despite GPUVRAMGate firing — because the gate killed zombies, the check ran before VRAM was released, and the experiment deferred unnecessarily.

### Gap 2: JEPA regression to AUC 0.281 (FR-11 — Autonomous Self-Learning Loop)

Three consecutive milestones of JEPA regression: 0.667 → 0.400 → 0.281. The quality-gate approach (Exp 477) removed 73% of training pairs as "low confidence" — leaving a corpus of only 15 high-confidence pairs, which is too small for any model to train on without overfitting. The predictor collapsed to the majority class (correct steps) because correct steps have higher label_confidence.

**The correct fix is curriculum training, not filtering:** Start with all pairs, ordered by label_confidence (high → low). Train for 100 epochs on confident pairs first, then fine-tune on all pairs. This prevents the information loss of aggressive filtering while still giving the model a clear learning signal early.

### Gap 3: GPU 1 compute utilization at 11% (hardware waste)

device_map='auto' is a PyTorch feature that distributes model layers across available GPUs to fit within VRAM constraints. It routes WEIGHT STORAGE to GPU 1 but runs FORWARD-PASS COMPUTE on GPU 0 (the primary device). This is not GPU parallelism — it's tensor offloading. The result: GPU 1 runs at 11% utilization (weight fetches) while GPU 0 runs at 100% (all compute). To get true parallel compute: each model must be loaded with explicit device_map={'': 'cuda:0'} or {'': 'cuda:1'}. 53/64 dual-model harnesses still use 'auto'. DualGPUHarness.apply() from Exp 480 is the fix — but it needs to be applied, not just available.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MILESTONE 2026.04.37 PIPELINE                      │
│                                                                        │
│  PHASE 1: Fix the Kill Order (RETRO-044 — the master blocker)        │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 487: GPUVRAMGateV2 — kill FIRST, check after 15s drain      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 2: Credibility Benchmarks (RETRO-033/038/039 — 4th/3rd        │
│                                    attempt, unblocked by Phase 1)     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 488: Live 100q Precision v5 (RETRO-033 — 5th attempt)       │ │
│  │ Exp 489: Live 200q VeriCoT+VPRM v3 (RETRO-038 — 3rd attempt)   │ │
│  │ Exp 490: GSM-Symbolic Adversarial v3 (RETRO-039 — 3rd attempt)  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 3: JEPA Recovery — Curriculum Training (RETRO-040)            │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 491: JEPA Curriculum Diagnostic (why did quality-gate fail?) │ │
│  │ Exp 492: JEPA Curriculum Retrain v3 (ordered high→low confidence)│ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 4: Enforcement Tooling (RETRO-045/046 — install, don't doc)   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 493: Batching Pre-Commit Hook (RETRO-045 enforcement)        │ │
│  │ Exp 494: GPU Thermal Gate (RETRO-046 — 3rd attempt, just build)  │ │
│  │ Exp 495: DualGPU Harness Enforcement v2 (patch 53 harnesses)     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 5: New Research (arxiv 2026 + roadmap items)                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 496: NUP Probe v2 (Bayesian entropy + logprobs, RETRO-047)  │ │
│  │ Exp 497: SuRe Surprise-Driven EBM Replay (arXiv 2511.22367)     │ │
│  │ Exp 498: KAEM Extended Profile n=5000 (RETRO-031 closure)        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                         │
│  PHASE 6: Retrospective                                               │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Exp 499: Milestone 2026.04.37 Retrospective                      │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

```
487 (GPUVRAMGateV2)
 ├── 488 (100q v5)      ─┐
 ├── 489 (200q v3)      ─┤── 496 (NUP Probe v2, uses CoT pairs from 488/489)
 └── 490 (adversarial)  ─┘
491 (JEPA diagnostic)
 └── 492 (JEPA retrain v3)
493 (batching hook)     [independent]
494 (thermal gate)      [independent]
495 (DualGPU harness)   [independent]
497 (SuRe replay)       [independent]
498 (KAEM n=5000)       [independent]
499 (retro) ─── ALL
```

---

## Phase Descriptions

### Phase 1: Fix the Kill Order

**Exp 487 — GPUVRAMGateV2** is the foundation of this entire milestone. It makes a single logical change: kill_first=True by default. The new sequence: (1) kill all GPU zombie processes → (2) sleep 15 seconds (GPU driver drains their queued operations) → (3) check free VRAM → (4) if < min_free_gb, wait another 30 seconds → (5) if still insufficient, defer with gpu_vram_insufficient. This 15-second drain period is what was missing in v1. Zombies receive SIGKILL but their GPU contexts are not immediately freed by the driver — the driver holds the context until all pending GPU operations flush. 15 seconds is conservative but sufficient for RTX 3090.

GPUVRAMGateV2 replaces GPUVRAMGate in ExperimentTemplate.setup_gpu(). All experiments that import GPUVRAMGate should import GPUVRAMGateV2 going forward.

### Phase 2: Credibility Benchmarks

Three experiments that have been deferred 3-4 times due to zombie VRAM, now unblocked by GPUVRAMGateV2:

**Exp 488 — Live 100q Precision v5:** 100 GSM8K questions, Gemma4-E4B-it on cuda:0 and Qwen3.5-0.8B on cuda:1 (explicitly assigned, not 'auto'). IntegratedExtractor (VeriCoTStepValidator + VPRMArithmeticVerifier). Wilson 95% CI. Writes CoT pairs to results/exp488_cot_pairs.json for NUP Probe v2 (Exp 496).

**Exp 489 — Live 200q VeriCoT+VPRM v3:** 200 GSM8K questions with 95% CI ±3.5pp. This is the headline statistical test — a positive result here is the first credible claim Carnot can make. If is_statistically_positive is True (signed improvement with CI lower bound > 0), RETRO-038 is CLOSED.

**Exp 490 — GSM-Symbolic Adversarial v3:** Three-condition test (standard baseline, adversarial baseline, adversarial + Carnot). If thesis_confirmed=True (adversarial improvement > standard improvement), Carnot has its most compelling result: we fix the failure mode that breaks ALL other models including o1.

### Phase 3: JEPA Recovery

**Exp 491 — JEPA Curriculum Diagnostic:** Analyze the .36 quality-gate failure. Load the quality-gated corpus from Exp 477 (expected: ~15 pairs after filtering, heavily skewed toward correct steps). Compare learning curves on: (a) quality-gated corpus, (b) all pairs unfiltered, (c) curriculum-ordered (high confidence first). The diagnostic will identify whether the problem is data imbalance, label noise, domain shift, or model capacity. Produces a diagnostic report with recommendation for Exp 492.

**Exp 492 — JEPA Curriculum Retrain v3:** Implement JEPACurriculumTrainer with three stages: Stage 1 trains on label_confidence >= 0.85 pairs for 100 epochs, Stage 2 fine-tunes on all pairs for 100 epochs, Stage 3 augments with EBM-guided synthetic pairs if n_total < 200. This "easy-first" curriculum prevents the majority-class collapse seen in Exp 477 while still using all available data. Target: AUC > 0.600 (recovery from 0.281 regression). This is the Tier 3 self-learning experiment for this milestone (FR-11).

### Phase 4: Enforcement Tooling

All three items in this phase are things that have been documented but not installed. The pattern of "write standards" → "no enforcement" → "violation accumulates" ends here.

**Exp 493 — Batching Pre-Commit Hook:** Actually modify .pre-commit-config.yaml to add a hook running BatchingEnforcementAudit and failing the commit if new high-severity violations are introduced. This installs enforcement, not documentation. New violations cannot be committed without bypassing the hook.

**Exp 494 — GPU Thermal Gate:** Install a thermal check that runs before model loads. Uses pynvml.nvmlDeviceGetTemperature(). If temperature > 85°C, wait with exponential backoff (15s, 30s, 60s) until temperature drops below 80°C. If temperature doesn't drop within 5 minutes, defer with honest_verdict='gpu_thermal_throttle'. This has been requested for 3 milestones; it's a 50-line module.

**Exp 495 — DualGPU Harness Enforcement v2:** HarnessAudit (Exp 480) identified 53 scripts with dual-model loads not using cuda:1. This experiment patches them: auto-apply DualGPUHarness.apply() to all model_specs in identified scripts. After patching, re-run the audit to confirm n_missing_cuda1 = 0.

### Phase 5: New Research

**Exp 496 — NUP Probe v2:** NUP Probe v1 (Exp 484) used character-entropy as a logprob substitute and got AUC=0.600. The fix: use the real CoT pairs from Exps 488/489, which include inference metadata. Implement BayesianEntropyEstimator from arXiv 2603.22812: adaptively allocate sampling budget based on entropy uncertainty, compute credible intervals around entropy estimates, use upper credible bound as the "confident high entropy" signal. This should push AUC above the 0.700 Tier 0c threshold, adding NUP Probe as a zero-latency cascade tier (no LLM call, no Ising sampling).

**Exp 497 — SuRe Surprise-Driven EBM Replay:** Implement SuRePriorityReplay for PPSConstraintLearner (arXiv 2511.22367). When new constraint violations arrive, rank existing violations in the replay buffer by energy surprise: violations where EBM energy significantly exceeds the running expected energy for that domain partition get highest priority for replay. This should improve partition_isolation_score by ensuring the most "surprising" cross-domain violations are always included in replay, preventing catastrophic forgetting at domain boundaries. This is the Tier 2 self-learning experiment.

**Exp 498 — KAEM Extended Profile n=5000:** RETRO-031 found no 5x speedup crossover at n_vars=1000. The theoretical prediction (O(n²) MCMC vs O(n log n) KAEM) should produce crossover between n=1000 and n=5000. Profile at n_vars=(1000, 2000, 3000, 5000), stopping at the first n where speedup >= 5x. If no crossover at n=5000, close RETRO-031 as "KAEM does not achieve 5x on this architecture" and note for FPGA implementation where bisection is native hardware.

### Phase 6: Retrospective

**Exp 499** evaluates five headline questions:
1. Was RETRO-044 finally fixed? (n_deferred_to_gpu = 0 is the only acceptable answer)
2. Was RETRO-033 closed? (any is_positive at 100q+)
3. Was RETRO-040 closed? (JEPA AUC > 0.600 after curriculum training)
4. Did adoption reach >= 70%? (3 enforcement items installed: hook, thermal gate, harness patch)
5. Was the GSM-Symbolic thesis confirmed? (thesis_confirmed = True at any scale)

---

## Hardware Requirements

| Experiment | Hardware | Requirement |
|-----------|---------|------------|
| 487-495, 497-498 | CPU | No GPU needed |
| 488 | GPU 0+1 | cuda:0 + cuda:1 explicit, CARNOT_FORCE_LIVE=1 |
| 489 | GPU 0+1 | cuda:0 + cuda:1 explicit, CARNOT_FORCE_LIVE=1 |
| 490 | GPU 0+1 | cuda:0 + cuda:1 explicit, CARNOT_FORCE_LIVE=1 |
| 496 | CPU | Uses saved CoT pairs from 488/489 |
| 499 | CPU | Reads result files |

GPU experiments (488, 489, 490) depend on Exp 487 GPUVRAMGateV2 deployed first.

---

## New arxiv Findings to Incorporate

From the 2026-04-19 planning scan (see research-references.md):

| Paper | Relevance | Incorporated As |
|-------|-----------|----------------|
| arXiv 2509.14252 (LLM-JEPA) | JEPA embedding stability for EBM scoring | Informing Exp 492 curriculum design |
| arXiv 2603.22812 (Bayesian Semantic Entropy) | Richer entropy features for NUP Probe | Exp 496 (NUP Probe v2) |
| arXiv 2511.22367 (SuRe surprise replay) | Priority replay for PPSEBM | Exp 497 (SuRe EBM Replay) |

---

## Self-Learning Coverage

| Tier | Experiment | Description |
|------|-----------|-------------|
| Tier 1 | (in pipeline) | Online constraint weighting runs per-query |
| Tier 2 | Exp 497 | SuRe surprise-driven replay for PPSConstraintLearner |
| Tier 3 | Exp 492 | JEPA Curriculum Retrain v3 — predict violations from partial CoT |
| Tier 4 | (future) | Adaptive energy landscape structure — FPGA/TSU dependent |

---

## Success Criteria

| Criterion | Target | Experiment |
|-----------|--------|-----------|
| n_deferred_to_gpu | 0 | All GPU experiments |
| live_100q_positive | True | Exp 488 |
| live_200q_statistically_positive | True | Exp 489 |
| jepa_auc_recovered | > 0.600 | Exp 492 |
| nup_probe_tier_0c_viable | AUC > 0.700 | Exp 496 |
| retro_improvement_adoption | >= 70% | Exp 499 |
| batching_hook_installed | True | Exp 493 |
| thermal_gate_installed | True | Exp 494 |
