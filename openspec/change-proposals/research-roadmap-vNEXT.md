# Research Roadmap — Milestone 2026.04.54

**Title:** JEPA v17 RankNet Loss + Gemma4 VR Diagnostic + PSV PaCoRe K=2

**CalVer:** 2026.04.54 (sequence increment from 2026.04.53)

**Authored:** 2026-04-22

**Previous Milestone:** 2026.04.53 — "JEPA v16 Recovery + VR Credibility Hardening + Prompt-Injection True Distillation"

---

## What Milestone 2026.04.53 Proved

Milestone .53 delivered a ninth consecutive wall-time improvement (-74 min, -1.9%), though the
improvement rate decelerated sharply from .52's -248 min (-5.9%). Per-experiment average held at
7.1 min (effectively flat vs .52 best of 7.0 min — no structural throughput gain). Key outcomes:

- **Exp 692 (Pre-flight v5):** `preflight_v5_complete` — Slowest-5 manifest updated; Exps
  425/410/383 retirement candidates flagged; conductor pre-flight confirmed operational.
- **Exp 693 (JEPA v15 Root Cause):** `root_cause_identified_v16_specced`, root_cause =
  `pure_loss_anti_correlation` — JEPA v15 OOD AUC=0.4751 caused by scalar loss allowing
  the model to hedge to P=0.5 globally. InfoNCE identified as v16 fix candidate.
- **Exp 694 (VR Cross-Model):** `vr_cross_model_no_improvement` — Qwen3.5-0.8B
  `signed_improvement=1.0` (confirmed at 200q, RETRO-033 CLOSED). Gemma4-E4B-it
  `signed_improvement=-0.8`. `cross_model_delta=-1.8`. VR HURTS Gemma. Root cause unknown.
- **Exp 695 (Formal Step Verifier Tier 2.8):** `tier_28_no_candidate` — FoVer v1 corpus
  degenerate for benchmarking (only `step_correct=True` labels, no parseable Z3 verdicts).
- **Exp 696 (I-CALM Abstention):** success — confidence-gated abstention reduces FP rate.
- **Exp 697 (PSV Real Self-Play):** `psv_real_fp_degrading`, `fp_rate_trend_slope=0.004242`
  — PSV REVERSED from improving (.52 Exp 688) to degrading. Single-chain K=1 appears to
  saturate after 10 iterations, accumulating FP noise in the constraint pool.
- **Exp 698 (JEPA v16 InfoNCE):** `jepa_v16_still_below_random`, `v16_ood_auc=0.4759` —
  InfoNCE did NOT fix the anti-correlation. Only +0.0008 delta over v15. Still below random.
- **Exp 699 (HalluSAE Integration):** `hallusae_integration_no_improvement`, `hallusae_v16_ood_auc=0.2616`
  — SAE sparse features DEGRADED JEPA v16 by -0.2142 AUC. Integration incompatible.
- **Exp 700 (Publication Readiness):** `publication_ready=True` — VR headline result on
  Qwen3.5-0.8B 200q is publication-grade. Model card written. However, distillation
  AUROC=0.7995 < 0.90 gate for the injection KAN component.
- **Exp 701 (KV260 Synthesis):** `synthesis_blocked_no_tool` — neither Vivado nor yosys
  installed. RETRO-072 unresolved.

**Still open after .53:**
- RETRO-072: KV260 Ising v3 synthesis blocked (requires Vivado or yosys installation — human action)
- RETRO-CRITICAL: JEPA cascade blocked — v16 OOD AUC=0.4759 (two consecutive failed retrains)
- Slowest-5 UNCHANGED for FIFTH consecutive milestone (longest frozen streak in project history)
  - Exp 425: 17th consecutive milestone (1,292 min cumulative overhead)
  - Exp 410: 14th consecutive milestone (716 min cumulative overhead)
  - Exp 383: 8th consecutive milestone (DualGPU fix validated in .52 but STILL NOT DEPLOYED)
  - Exp 380-382: 5th consecutive milestone (formal retirement threshold crossed)
  - Exp 346: 5th consecutive milestone (formal retirement threshold crossed)
- PSV self-play degrading (slope=0.004242, reversed from improving in .52)
- Prompt-injection KAN distillation AUROC=0.7995 (below 0.90 gate for full publication)

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: JEPA Cascade Blocked — Two Consecutive Failed Retrains (RETRO-CRITICAL)

**State:** JEPA v16 OOD AUC=0.4759 (below random 0.5). Two consecutive failed retrain
strategies: PUREMinFormLoss (.51) → InfoNCE (.53) both failed. HalluSAE integration
worsened it. Root cause is confirmed as `pure_loss_anti_correlation`: any scalar loss that
produces a single probability per step allows the model to hedge to P=0.5 globally.

**Root cause analysis (from Exp 693):** The JEPA predictor's training objective must enforce
a STRICT ORDERING between correct and incorrect steps, not just maximize per-step accuracy.
Both BCE and InfoNCE treat each (step, label) pair independently — the model can satisfy
the loss by hedging all outputs to 0.5 without learning any discrimination.

**The fix for v17 — RankNet pairwise ranking loss:**
- For each training batch, form (correct_step, incorrect_step) pairs from the same question
- Apply RankNet loss: L = -log(sigmoid(score(incorrect) - score(correct)))
  This requires score(incorrect) > score(correct) for every pair simultaneously.
  The model CANNOT hedge — it must strictly rank incorrect above correct (high energy = wrong).
- Hard negative mining: for each correct step, find the most similar incorrect step
  (hardest negative) to prevent the model from learning trivially separable pairs
- This is the approach that provably eliminates the anti-correlation root cause:
  hedging to P=0.5 gives loss = log(2) per pair, but correctly ranking gives loss → 0

**Target:** OOD AUC >= 0.75 on GSM8K 500-699 (never seen in training).

**Data:** FoVer formal v1 (200 Z3-labeled pairs, Exp 686). Plus FoVer v2 (Exp 712, PDDL-scaled)
if available before JEPA v17 retrains.

### Gap 2: VR Hurts Gemma4-E4B-it — Cross-Model Failure Undiagnosed

**State:** VR pipeline works for Qwen3.5-0.8B (`signed_improvement=1.0`) but HURTS Gemma4-E4B-it
(`signed_improvement=-0.8`, `cross_model_delta=-1.8`). The root cause is unknown: could be
format mismatch (Gemma writes arithmetic differently), repair quality (repairs introduce
new errors for Gemma), or constraint threshold miscalibration (FP rate too high for Gemma's
accurate outputs).

**Diagnostic strategy (Exp 706):** Trace the VR failure for Gemma to the exact pipeline step:
1. Extraction: does SymCodeVerifier fire on Gemma's correct outputs? (FP check)
2. Verification: are the violations real? Or regex failures on Gemma's format?
3. Repair: does the repair degrade the correct answer?
This requires instrument-mode logging with `pipeline_step` granularity.

**Fix (Exp 707):** Model-adaptive constraint thresholds. Use the Tier 1 self-learning tracker
to suppress constraint types with FP rate > TP rate per model. If SymCodeVerifier fires
falsely on Gemma's correct COMPUTE: lines, disable SymCodeVerifier for Gemma responses above
a confidence threshold. Per the research-program.md guidance on self-learning Tier 1.

### Gap 3: Structural Execution Bottleneck — 5th Consecutive Milestone Unchanged

**State:** Per the .53 retro, the slowest-5 composition is UNCHANGED for the FIFTH consecutive
milestone — the longest frozen streak in project history. Exp 425 has appeared in 17 consecutive
milestones (1,292 min cumulative overhead). Exp 383's DualGPU fix was VALIDATED in .52 but
is STILL NOT DEPLOYED.

**Fix (Phase 0, Exp 703):** This is a governance failure, not a technical one. The retirement
threshold (3 consecutive milestones) was crossed by Exps 380-382 and 346 in .53. They MUST
be formally retired in Phase 0 before any research begins. Additionally, Exp 383 (sequential
EORM+JEPA retrain, 62 min) must be replaced by the DualGPU pattern (35 min) from Exp 685
as the default going forward. The pre-flight v6 must verify these retirements before the
conductor proceeds.

---

## Architecture: Verification Pipeline (Updated for .54)

```
Input: LLM Response

Tier 0a: CarnotThinkProbe      (~50-200ms GPU)    [DEPLOYED]
Tier 0b: SpilledEnergyDetector (~0ms)             [DEPLOYED]
Tier 0c: NUP Probe v4          (~0ms)             [DEPLOYED]
Tier 0d: HallucinationBasin    (~0ms)             [DEPLOYED]
Tier 0e: HalluField            (~1ms CPU)         [DEPLOYED, advisory]
Tier 1:  SinkProbe             (~0ms)             [DEPLOYED]
Tier 2:  EORM                  (~10ms)            [DEPLOYED, v15 architecture]
         JEPA cascade          (BLOCKED until v17 OOD AUC >= 0.75)
Tier 2.5: SymCodeVerifier      (~1-500ms)         [DEPLOYED, Qwen-calibrated]
Tier 2.6: HermesVerifierAdapter (~1-500ms)        [PROTOTYPE, CPU]
Tier 2.7: CausalReasoningVerifier (~1ms/step)     [DEPLOYED]
Tier 2.8: (no winner yet — FoVer corpus degenerate per Exp 695)
Tier 2.9: SC-Energy (CANDIDATE — Exp 711)
Tier 3:  Ising VerifyRepairPipeline (~0.006ms)    [DEPLOYED]
         I-CALM Abstention     (DEPLOYED, Exp 696)

Self-Learning (FR-11):
Tier 1: Online weight updates  [DEPLOYED, needs JEPA v17 to have real positives]
Tier 2: ConstraintTemplateLibrary [DEPLOYED, last wired in Exp 683]
Tier 3: JEPA predictive verify [BLOCKED — v17 must unblock cascade first]
Tier 4: Adaptive structure     [FUTURE — depends on Tier 3]

Hardware:
KV260 FPGA: RTL written (v3 EMA inertia), SYNTHESIS BLOCKED (no Vivado/yosys)
DualGPU:    CONFIRMED working (Exp 685, 2.0175x speedup)
D-Wave:     CONFIRMED working (Exp 598, 26.24x speedup)
AMD NPU:    BLOCKED (ninja/openblas missing, IRON path untested)
```

---

## Phase Descriptions

### Phase 0: Operational Pre-Flight v6 (Mandatory First)

**Scope:** Governance, infrastructure, and slowest-5 forced retirement.

The retro's verdict is clear: "fifth consecutive milestone, longest frozen streak." This is
a governance failure. Phase 0 takes 1 experiment to fix it permanently:

- Formally retire Exps 380-382 and 346 (3-consecutive-milestone threshold crossed in .53)
- Formally retire Exps 425 and 410 (threshold crossed in .50/.51 respectively — they have
  appeared 17 and 14 consecutive milestones; no further delay is acceptable)
- Deploy the Exp 685 DualGPU pattern as the default for Exp 383 jobs going forward
- Wire all retirements into conductor_exclusion_manifest.json
- Verify conductor_pre_flight.py confirms manifest_consulted=True

**Success criterion:** Zero slowest-5 entries for Exps 380-382, 346, 425, 410 in .54.

### Phase 1: JEPA v17 — RankNet Pairwise Ranking Loss

**Scope:** Fix the root cause of JEPA anti-correlation with a strictly ordering loss.

Two experiments:
1. Exp 704: Implement and train JEPARankNetV17 using pairwise RankNet loss + hard negative
   mining. Train on FoVer formal v1 (200 pairs). Evaluate OOD AUC on GSM8K 500-699.
   Gate: v17_ranknet_ood_auc >= 0.75 → cascade unblocked.
2. Exp 705: If gate opens, deploy JEPA v17 to Tier 2 cascade and validate end-to-end.
   If gate fails, emit a root cause audit artifact explaining why RankNet also failed
   and recommend v18 architecture (e.g., listwise LambdaRank).

**Why RankNet over InfoNCE:** InfoNCE maximizes mutual information between (anchor, positive)
pairs, which can still allow anti-correlation if the anchor distribution is narrow (all
arithmetic carry errors). RankNet directly enforces a partial order constraint:
score(incorrect) > score(correct) for every sampled pair. This is a harder constraint that
prevents hedging. With hard negative mining, the model must learn to rank the most similar
incorrect step above the correct step — forcing genuine discrimination.

### Phase 2: Gemma4 VR Diagnostic + Adaptive Thresholds

**Scope:** Diagnose and fix VR's harm to Gemma4-E4B-it.

Three experiments:
1. Exp 706: Instrument-mode VR run on 25 Gemma4 responses (known-correct + known-incorrect).
   Log per-step: extractor fired, constraint violated, repair applied, final answer changed.
   Produce diagnostic_artifact with failure_mode (extraction_fp, repair_regression, threshold).
2. Exp 707: Implement ModelAdaptiveThresholdGate. Per-model constraint precision tracker:
   for each (model_id, constraint_type), track TP/FP counts. Gate constraint verification
   for Gemma4 if FP_rate > TP_rate for that constraint type on that model. Integrates into
   Tier 1 self-learning infrastructure (ConstraintStateMachine from Exp 125).
3. Exp 708: Live VR Attempt #19 — Gemma4 with adaptive thresholds enabled. GATED on
   Exp 707 (adaptive thresholds implemented). Target: Gemma4 `signed_improvement >= 0`.
   (No harm is the minimum bar; net improvement is stretch goal.)

### Phase 3: PSV PaCoRe K=2 — DualGPU Diversity Recovery

**Scope:** Fix PSV self-play reversal by adding diversity via parallel chains.

One experiment:
1. Exp 709: Implement PSV-PaCoRe: K=2 parallel PSV chains using DualGPU (EORM on cuda:0,
   JEPA on cuda:1). For each PSV iteration: run chain A and chain B independently on 10
   questions each. Merge via SymCodeVerifier energy vote: for each question, select the
   chain response with lower violation energy. Accumulate 10 iterations (100 questions total).
   Hypothesis: diverse chains prevent the saturation pattern that caused .53's reversal.
   Target: `fp_rate_trend_slope < 0` (restoring improvement direction).

### Phase 4: New Research — Distillation + SC-Energy + FoVer Scaling

**Scope:** Three independent research experiments.

1. Exp 710: Prompt-injection KAN Distillation v2. Current AUROC=0.7995 < 0.90 gate.
   Fix: increase training corpus from 1000 to 2000 teacher-labeled examples (doubling the
   teacher inference budget), and increase KAN capacity (more knots per spline segment).
   Also apply L2 regularization to prevent overfitting. Target: distillation_auroc >= 0.90.

2. Exp 711: SC-Energy Set Consistency Verifier (arXiv 2503.10695). Implement
   SetConsistencyVerifier that computes a global energy over the full CoT step set, not
   just pairwise transitions. This catches "correct arithmetic at each step but globally
   contradictory conclusion" errors. Evaluate on FoVer formal v1 corpus as Tier 2.9 candidate.
   Target: AUC >= 0.75 on multi-step contradiction detection.

3. Exp 712: FoVer v2 Dataset Synthesis via PDDL Planning (arXiv 2604.17957). Scale the
   FoVer formal v1 corpus (200 Z3 pairs) by adding PDDL-based step labels for procedural
   arithmetic (noun-quantity state transitions). Target: 1000+ combined pairs in
   fover_v2_combined.json, ready as JEPA v17 retraining data if v17 fails in Phase 1.

### Phase 5: FR-11 Self-Learning Relay (Mandatory)

**Scope:** Wire JEPA v17 violations into ConstraintTemplateLibrary if cascade unblocked.

1. Exp 713: FR-11 Tier 2 Relay — GATED on Exp 705 (JEPA v17 cascade unblocked).
   If unblocked: wire verified violations from JEPA v17's cascade run into the
   ConstraintTemplateLibrary. Advance FR-11 self-learning from Tier 1 (weight updates) to
   Tier 2 (cross-session constraint memory). Report fr11_tier_advancement.

### Phase 6: Hardware + Retrospective

1. Exp 714: AMD XDNA NPU Unblock v8 — IRON Toolchain Fresh Approach. Previous 7 attempts
   blocked by missing ninja/openblas for VitisAI path. New approach: IRON toolchain
   (mlir-aie) which does NOT require ninja/openblas. Try `pip install mlir-aie` → if
   importable, run bare-metal NPU GEMM benchmark (arXiv 2504.03083 pattern). Also try
   AMD's pre-built custom onnxruntime wheel (Python 3.12 available via .venv-npu/).

2. Exp 715: Milestone 2026.04.54 Operational Retrospective.

---

## Dependency Graph

```
Exp 703 (Pre-flight) ──────────────────────┐
                                            ▼
Exp 704 (JEPA v17 RankNet) ──────────────► Exp 705 (JEPA v17 Deploy, GATED on v17 AUC >= 0.75)
                                            │
                                            └──────────────────────► Exp 713 (FR-11 Relay, GATED)
Exp 706 (Gemma4 Diagnostic) ─────────────► Exp 707 (Adaptive Thresholds) ──► Exp 708 (VR #19)

Exp 709 (PSV PaCoRe K=2) — independent

Exp 710 (KAN Distill v2) — independent
Exp 711 (SC-Energy) — independent
Exp 712 (FoVer v2 PDDL) ────────────────► optional input to JEPA v17 retrain if v17 fails

Exp 714 (NPU IRON) — independent

All complete ──────────────────────────────► Exp 715 (Retro)
```

---

## Success Criteria

| Experiment | Gate Metric | Success Value | Failure Path |
|------------|-------------|---------------|--------------|
| Exp 703 | preflight_v6_complete | True | Block until fixed |
| Exp 704 | v17_ranknet_ood_auc | >= 0.75 | Emit v18 spec, JEPA remains blocked |
| Exp 705 | cascade_unblocked | True (gated on 704) | Skip if 704 gate fails |
| Exp 706 | failure_mode identified | extraction_fp OR repair_regression | Research finding |
| Exp 707 | adaptive_thresholds_implemented | True | Block Exp 708 |
| Exp 708 | gemma_signed_improvement | >= 0.0 | Research finding: model-adaptive not sufficient |
| Exp 709 | fp_rate_trend_slope | < 0 | Research finding: diversity insufficient |
| Exp 710 | distillation_auroc | >= 0.90 | Research finding: KAN capacity insufficient |
| Exp 711 | sc_energy_auc | >= 0.75 | Research finding: SC-Energy not viable for Tier 2.9 |
| Exp 712 | fover_v2_n_pairs | >= 1000 | Partial result if PDDL synthesis incomplete |
| Exp 713 | fr11_tier_advancement | >= 2 | Skip if Exp 705 gate failed |
| Exp 714 | npu_benchmark_run | True | Research finding: IRON path blocked |

---

## Open RETROs

| RETRO | Status | Expected Resolution |
|-------|--------|---------------------|
| RETRO-072 | OPEN — KV260 synthesis blocked (no tool) | Human must install Vivado/yosys; .54 experiments use yosys fallback |
| RETRO-CRITICAL | OPEN — JEPA cascade blocked v16 AUC=0.4759 | Exp 704 targets v17 fix |
| Slowest-5 frozen (5th milestone) | OPEN — Exp 425 at 17th appearance | Exp 703 forced retirement |
| PSV reversal | OPEN — slope=+0.004242 | Exp 709 PaCoRe K=2 |
| Distillation below 0.90 gate | OPEN — AUROC=0.7995 | Exp 710 v2 |
| AMD NPU blocked (7th milestone) | OPEN — IRON path untested | Exp 714 |

---

## Hardware Requirements

| Experiment | GPU Needed | Notes |
|------------|-----------|-------|
| Exp 704 (JEPA v17) | Optional | CPU trains on FoVer formal v1 (200 pairs) in < 5 min |
| Exp 705 (JEPA cascade deploy) | Optional | Runs cascade on 200 GSM8K questions for validation |
| Exp 706 (Gemma4 diagnostic) | Required | Live GPU inference on Gemma4-E4B-it |
| Exp 708 (VR #19 Gemma4) | Required | Live GPU inference on Gemma4-E4B-it, 25 questions |
| Exp 709 (PSV PaCoRe K=2) | Required | DualGPU: chain A cuda:0, chain B cuda:1 |
| Exp 710 (KAN distill v2) | Optional | CPU distillation from cached teacher labels |
| Exp 714 (NPU) | NPU target | AMD XDNA NPU; falls back to CPU if IRON not available |

All other experiments: CPU only.

---

## New Papers Incorporated

| Paper | Filed Experiment | Key Idea |
|-------|-----------------|----------|
| arXiv 2503.10695 (SC-Energy) | Exp 711 | Set-level consistency energy; Tier 2.9 candidate |
| arXiv 2604.17957 (PRM PDDL) | Exp 712 | PDDL step labels; scale FoVer corpus 5x |
| arXiv 2602.12566 (Multi-domain RL) | Exp 706-708 | Explains Gemma4 cross-model failure; informs adaptive thresholds |
| arXiv 2601.05593 (PaCoRe) | Exp 709 | Parallel coordinated reasoning; K=2 chains per PSV iteration |
