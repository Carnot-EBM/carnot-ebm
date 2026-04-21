# Research Roadmap v49 — Milestone 2026.04.49

**Status:** Proposed
**Milestone:** 2026.04.49
**Title:** HERMES v2 Live Generation Loop + Platt JEPA + Parallel Ising Inertia
**Planned:** 2026-04-21
**Experiments:** 640–651 (12 experiments)

---

## What Milestone 2026.04.48 Proved

Milestone 2026.04.48 ran 13 experiments (627–639) and produced these honest findings:

**Partially resolved:**
- SymCodeVerifier confirmed as the strongest verification primitive (AUC=0.804, distribution-invariant via eval()).
- HERMES step-boundary architecture: recall improved from 4% (post-hoc) to 12% (HERMES v1, Exp 633). First architecture to show meaningful improvement over raw extraction.
- interwhen mid-generation recall also 12% (Exp 627). Early detection rate = 1.0 (violations detectable early in the sentence, not just at end).
- JEPA v14 OOD AUC=0.912 (architecture excellent). ECE=0.132 — calibration target 0.10 NOT MET.
- Multilevel KAN training implemented (Exp 634): KnotRefinementInterpolator + MultilevelKAEMTrainer.
- AdapTrack backtracking implemented (Exp 635): proportional backtrack on SymCodeVerifier violation.
- FPGA TCL v2 written (Exp 636): synth_ising_v2.tcl targets synchronous RTL. Vivado still not installed.

**Still blocked — escalated:**
- RETRO-070 (CRITICAL, carry 3): interwhen AND HERMES both achieve 12% recall. Both run post-hoc on COMPLETED responses. Root cause confirmed: IT models write violations in prose; no post-hoc extractor crosses 20% recall. The fix is a LIVE generation loop — generate one step, verify mid-generation, inject feedback, generate next step. This was NOT implemented in .48 (Exp 633 was CPU prototype on completed responses, not a live loop).
- RETRO-033 (carry 16): VR attempt #16 blocked (gate_open=False, recall < 20%). Sixteen consecutive 0% attempts. Gate upgraded to 30% minimum recall before attempt #17.
- RETRO-071: DualGPU 13B proof — model_load_failed (HF weights not cached). Need pre-downloaded Qwen2.5-7B-Instruct.
- RETRO-057: SparseKAEM sparse_vs_dense_error=0.429 — far outside 5% threshold. Multilevel + sparse COMBINED approach required.
- JEPA v14: ECE=0.132 (target 0.10) — Platt temperature scaling is the next calibration intervention.

**Research advances:**
- ORACLE corpus builder implemented (Exp 628). Step-level SymCodeVerifier labels on live responses.
- Sparse KAEM architecture designed (Exp 637). Combined multilevel+sparse is next step.
- HERMES v1 adapter operational. v2 live loop is the .49 critical path.
- arXiv 2604.17109 (parallel dense Ising with inertia): new FPGA architecture, 35x speedup, v3 RTL path identified.

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Post-hoc Approaches Architecturally Capped at 12% Recall (RETRO-070, CRITICAL)

**Evidence:** After 16 VR attempts, 6 extractor architectures, and both interwhen and HERMES v1, live recall is stuck at 12%. The ceiling is not an implementation bug — it is a distribution mismatch:
1. IT models write "The total of 47 apples and 28 oranges is 76 fruits." Completed responses show only "76" at the end of a natural language sentence.
2. SymCodeVerifier AUC=0.804 on completed responses means it catches ~80% of violations WHEN it can find an arithmetic expression. The problem is that only ~12% of violations produce a parseable expression in the completed response.
3. Post-hoc running on the completed response loses context: "47 apples and 28 oranges" precede the violation; they appear in adjacent sentences, not as a standalone equation.

**Fix:** HERMES v2 live generation loop. Generate step-by-step: prompt → step_1 → SymCodeVerifier(step_1) → [optional: inject hint] → prompt+step_1+hint → step_2 → SymCodeVerifier(step_2) → ... When SymCodeVerifier detects a violation mid-generation, the violation signal arrives while "47" and "28" are still visible in the generation context, not buried in a completed response. The correction hint informs the next step BEFORE the model commits to the wrong path.

**Gate for VR #17 (UPGRADED):** Combined ensemble recall >= 0.30 on 25 known-incorrect responses (Exp 643). Gate increased from 0.20 to 0.30 because 12 consecutive 0.12 results confirm 0.20 was not meaningful signal. 0.30 represents genuine extraction capability above current ceiling.

### Gap 2: JEPA v14 Calibration Incomplete — ECE=0.132 vs Target 0.10

**Evidence:** JEPA v14 achieves OOD AUC=0.912 (sound architecture), but ECE=0.132 means the model's confidence scores are overconfident — a score of 0.80 predicts only ~68% accuracy. This matters for Tier 3 (predictive verification): if the predictor says "80% likely to violate," the pipeline should apply proportional verification effort. Overconfidence leads to either wasted verification or missed violations depending on threshold setting.

**Fix:** Platt scaling (temperature parameter T): calibrated_prob = sigmoid(logit/T). Optimize T to minimize ECE on a validation set. Expected to reduce ECE from 0.132 to ~0.05 without changing AUC. This is a 5-minute postprocessing step once v14 weights are available.

### Gap 3: DualGPU Unproven for Real Models (RETRO-071) and Structural Inefficiency Persisting

**Evidence:**
- RETRO-071: Exp 632 DualGPU 13B proof — model_load_failed because HF weights for Qwen2.5-14B-Instruct were not cached. Two RTX 3090s (48GB total) available but never demonstrated simultaneously carrying a real model.
- Operational: Exclusion manifest wire-in NOT DONE for the 13th consecutive milestone. Pre-flight test suite consuming ~491 min per milestone (11% wall time). Exp 383 (combined EORM+JEPA retrain) runs sequentially on one GPU — DualGPU parallelization would cut 62 min to ~35 min.

**Fix:**
- Exp 640: Wire exclusion manifest into conductor (MANDATORY FIRST ACTION, 13th attempt). Also implement DualGPU EORM+JEPA retrain (parallel forward passes on cuda:0 and cuda:1).
- Exp 649: Pre-download Qwen2.5-7B-Instruct weights, re-run DualGPU proof with pre-cached model.

---

## Architecture Diagram — Verification Cascade (Post-.49)

```
LLM Generation (in progress)
     │
     ├─── HERMES v2 Live Loop (NEW .49):
     │    Every sentence: generate_step → SymCodeVerifier → inject hint → next step
     │    [step-level feedback loop — live generation, not post-hoc]
     │
     ▼
LLM Response (completed)
     │
     ▼
[Tier 0a] CarnotThinkProbe — generative CoT verdict (optional)
     │
     ▼
[Tier 0b] SpilledEnergyDetector — token logit discrepancy
     │
     ▼
[Tier 0c] NUP Probe v6 — AUC=0.964 (deployed .47)
     │
     ▼
[Tier 0d] HallucinationBasinDetector — latent basin depth
     │
     ▼
[Tier 0e] HalluField — thermodynamic instability (advisory)
     │
     ▼
[Tier 1]  SinkProbe — attention sink concentration
     │
     ▼
[Tier 2]  EORM — CoT energy reward model (55M params)
     │    OR: OTV One-Token Verifier (candidate .49, arXiv 2603.01025)
     │
     ▼
[Tier 2.5] SymCodeVerifier — executable Python arithmetic verification
     │
     ▼
[Tier 2.6] HermesVerifierAdapter v2 — live generation loop (NEW .49)
     │
     ▼
[Tier 3]  Ising — full constraint verification + repair
```

---

## Phase Descriptions

### Phase 0: Operational Pre-Flight (MANDATORY FIRST — Exp 640)

**Objective:** Wire exclusion manifest into conductor before the 5 chronic slow experiments (308, 425, 309, 410, 383) re-queue for the 14th consecutive milestone. Also implement DualGPU EORM+JEPA parallel retrain to eliminate Exp 383 as a structural bottleneck.

**Key deliverable:** `scripts/conductor_exclusion_manifest.json` wired into conductor with `conductor_consulted=True` verification. DualGPURetrain class running EORM on cuda:0 and JEPA on cuda:1 simultaneously.

### Phase 1: RETRO-070 Resolution — HERMES v2 Live Generation Loop (Exps 641-643)

**Objective:** Implement the first live-generation verification loop where SymCodeVerifier intercepts at step boundaries DURING generation, not after. This is architecturally different from all prior approaches which processed completed responses.

**Exp 641 (HermesV2LiveLoop):** Qwen3.5-0.8B generates one sentence at a time. After each sentence, SymCodeVerifier runs. If violation detected: inject correction hint into the generation context. Measure recall on 25 known-incorrect questions (live generation, not from live_pairs corpus).

**Exp 642 (CausalReasoningVerifier):** Extend SymCodeVerifier with step-entailment checking (arXiv 2601.21210). Given step_k text and step_{k+1} text: does step_k causally justify step_{k+1}? This catches logical incoherence violations that arithmetic checking misses.

**Exp 643 (EnsembleRecallGateV2):** OR ensemble: any_violation = hermes_v2 OR interwhen OR causal_check. Compute combined recall on 25 incorrect + 10 correct from live corpus. gate_open = combined_recall >= 0.30.

### Phase 2: VR Attempt #17 + FR-11 (Exps 644-645)

**Exp 644 (LiveVRAttempt17):** GATED on Exp 643 gate_open=True. Use ensemble extractor. 25 live questions. Compare baseline vs repaired correctness.

**Exp 645 (FR11Tier1Relay):** FR-11 mandatory. If signed_improvement > 0 from Exp 644: use real violations. If not: semi-real from Exp 643.

### Phase 3: JEPA Calibration + OTV Verifier (Exps 646-647)

**Exp 646 (JEPAv14PlattScaling):** Apply temperature scaling T to v14 logits. Grid search T in [0.5, 2.0]. Minimize ECE on held-out validation set. Target ECE < 0.10 (current: 0.132).

**Exp 647 (OTVVerifier):** One-Token Verifier (arXiv 2603.01025): attach LoRA verification head to Qwen3.5-0.8B. Train on 100 live FOVER pairs. Compare AUC vs EORM (55M params). If AUC >= EORM_AUC - 0.05: recommend as Tier 2 default (10ms → sub-1ms).

### Phase 4: New Research + Hardware (Exps 648-650)

**Exp 648 (ParallelDenseIsingInertia):** Implement inertia Ising dynamics (arXiv 2604.17109). ParallelDenseIsingSampler with alpha parameter. Benchmark convergence vs checkerboard Gibbs. Generate v3 RTL specification for KV260 synthesis.

**Exp 649 (DualGPU13BProofV2):** RETRO-071 closure attempt. Pre-download Qwen2.5-7B-Instruct. Load with explicit layer-to-GPU mapping. Measure GPU-1 sustained utilization during 10 forward passes. Target: peak_gpu1_util > 50%.

**Exp 650 (KAEMMultilevelSparse):** RETRO-057 closure attempt. Combine MultilevelKAEMTrainer (Exp 634) + SparseKAEMEnergy (Exp 637). Train SparseKAEMEnergy at each multilevel schedule step (16→32→64 knots with sparsification). Target: energy accuracy within 5% vs dense baseline.

### Phase 5: Retrospective (Exp 651)

Analyze: RETRO-070 resolved (recall >= 0.30)? RETRO-033 first positive? RETRO-071 DualGPU proven? RETRO-057 KAEM accuracy? JEPA v14 calibrated (ECE < 0.10)? Top 3 priorities for .50.

---

## Dependency Graph

```
Exp 640 (pre-flight infra)
    │
    ├─── Exp 641 (HERMES v2 live loop) ─── GPU REQUIRED
    │         │
    │    Exp 642 (causal verifier) ─────── CPU
    │         │
    │    Exp 643 (ensemble gate v2) ──────  gate_open check
    │              │
    │    Exp 644 (VR #17) ──────────────── GATED on gate_open=True, GPU
    │              │
    │    Exp 645 (FR-11 relay) ─────────── reads Exp 644
    │
    ├─── Exp 646 (JEPA Platt) ──────────── reads Exp 631 weights
    │
    ├─── Exp 647 (OTV verifier) ────────── GPU
    │
    ├─── Exp 648 (Ising inertia) ───────── CPU
    │
    ├─── Exp 649 (DualGPU v2) ─────────── GPU, reads Exp 632 result
    │
    ├─── Exp 650 (KAEM multilevel+sparse)  CPU, reads Exps 634+637
    │
    └─── Exp 651 (retro) ───────────────── reads all above
```

---

## Hardware Requirements

| Experiment | GPU Required | Notes |
|-----------|--------------|-------|
| Exp 640 | No | CPU-only infrastructure |
| Exp 641 | Yes | CARNOT_FORCE_LIVE=1, Qwen3.5-0.8B live generation |
| Exp 642 | No | CPU, post-hoc causal verification |
| Exp 643 | No | CPU, ensemble on live_pairs corpus |
| Exp 644 | Yes | CARNOT_FORCE_LIVE=1, GATED on gate_open=True |
| Exp 645 | No | CPU, constraint addition |
| Exp 646 | No | CPU, Platt scaling on saved weights |
| Exp 647 | No | CPU, OTV LoRA training on saved FOVER pairs |
| Exp 648 | No | CPU, Ising simulation |
| Exp 649 | Yes | Both RTX 3090s, CARNOT_FORCE_LIVE=1 |
| Exp 650 | No | CPU, KAEM training |
| Exp 651 | No | CPU, retro analysis |

---

## Open RETROs Addressed

| RETRO | Status | Action |
|-------|--------|--------|
| RETRO-033 | Carry 16 — gated on RETRO-070 resolution | Exp 644 (VR #17, gated on 0.30 recall) |
| RETRO-057 | Carry 5 — SparseKAEM 43% error | Exp 650 (multilevel + sparse combined) |
| RETRO-060 | Carry 5 — JEPA ECE 0.132 | Exp 646 (Platt scaling) |
| RETRO-070 | Carry 3 — recall 12%, post-hoc cap | Exp 641 (HERMES v2 live loop) |
| RETRO-071 | Carry 1 — DualGPU model_load_failed | Exp 649 (pre-downloaded weights) |
| RETRO-CRITICAL | Carry 13 — exclusion manifest not wired | Exp 640 (MANDATORY FIRST) |

---

## Success Criteria

| Criterion | Pass | Experiment |
|-----------|------|-----------|
| Exclusion manifest wired | conductor_consulted=True | Exp 640 |
| HERMES v2 live recall | >= 0.20 (target 0.30) | Exp 641 |
| Ensemble recall gate | >= 0.30 | Exp 643 |
| VR attempt #17 | signed_improvement > 0 | Exp 644 (if gate opens) |
| JEPA v14 ECE | < 0.10 | Exp 646 |
| OTV vs EORM | AUC within 0.05 | Exp 647 |
| Ising inertia speedup | convergence steps reduced >= 20% | Exp 648 |
| DualGPU proven | peak_gpu1_util > 50% | Exp 649 |
| KAEM accuracy | error < 5% | Exp 650 |
| FR-11 relay | fr11_real_violations_confirmed=True | Exp 645 (if VR positive) |

---

## Key New Papers (from 2026-04-21 Scan)

| Paper | Filed For | Relevance |
|-------|-----------|-----------|
| arXiv 2604.17109 (Parallel Dense Ising + Inertia) | Exp 648 | 35x speedup, inertia dynamics |
| arXiv 2601.04358 (Energy-Time-Accuracy Thermodynamics) | .50+ | EDD bounds for hardware calibration |
| arXiv 2512.21911 (Sparse Speculative Verification) | .50+ | Selective SymCodeVerifier calls |
| arXiv 2601.21210 (Causal Symbolic Verification) | Exp 642 | Step-entailment checker |
