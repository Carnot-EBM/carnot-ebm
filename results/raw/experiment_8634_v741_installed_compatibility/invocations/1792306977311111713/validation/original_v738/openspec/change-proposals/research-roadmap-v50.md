# Research Roadmap — Milestone 2026.04.50

**Title:** Prompt-Injection Safety KAN + Structured Equation Forcing + SpecGuard Mid-Generation Verification

**CalVer:** 2026.04.50 (sequence increment from 2026.04.49)

**Authored:** 2026-04-21

**Previous Milestone:** 2026.04.49 — "HERMES v2 Live Generation Loop + Platt JEPA + Parallel Ising Inertia"

---

## What Milestone 2026.04.49 Proved

Milestone .49 delivered the first genuine structural improvement in milestone efficiency:
- Wall time -87 min (-1.9%) while experiment count INCREASED by 18 (491→509)
- Per-experiment average: 8.6 min (project best)
- Exps 308 and 309 exited the slowest-5 for the first time in 12 milestones

Key research results (pending full .49 retro at Exp 651):
- HermesV2LiveLoop (Exp 641): step-by-step generation with mid-generation verification
- CausalReasoningVerifier (Exp 642): step entailment checking as Tier 2.7
- Ensemble recall gate v2 (Exp 643): computed combined recall of all three extractors
- Live VR attempt #17 (Exp 644): gated on Exp 643 gate_open
- JEPA v14 Platt scaling (Exp 646): temperature calibration for ECE < 0.10
- OTV one-token verifier (Exp 647): EORM replacement candidate
- Parallel Ising inertia (Exp 648): Python simulation, v3 RTL spec written
- Prompt-injection EBM (Exp 652): distilled from gpt-oss-safeguard-20b, queued for .50

Still open after .49:
- RETRO-033: 17 consecutive VR attempts at 0% improvement (root cause: extraction ceiling at 12% recall)
- RETRO-057: LowRankKAEM multilevel+sparse did not improve over sparse-only (Exp 650: retro_057_resolved=false)
- RETRO-070: Extraction recall ceiling at 12% — post-hoc extraction architecturally capped
- RETRO-071: DualGPU 13B proof failed (model not cached)
- RETRO-072: KV260 N=128 sizing overflow; N=64 rebuild in-flight as of 2026-04-21
- Exclusion manifest wire-in: 13th consecutive milestone unconfirmed (RETRO-CRITICAL)

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: RETRO-033 / RETRO-070 — Extraction Ceiling Blocks All VR Progress

**Root cause (now confirmed after 17 attempts):** Post-hoc extraction is architecturally capped at ~12% recall because instruction-tuned models write arithmetic in natural language prose, not in the format `X op Y = Z` that SymCodeVerifier and HERMES v1 were designed to parse.

**The fix (not tried yet):** Structured equation forcing. Instead of extracting equations from the model's free-form response, PROMPT the model to write its arithmetic as explicit equations at each step. Example system prompt addendum: "At each reasoning step, write arithmetic as 'COMPUTE: X op Y = result' before continuing." Then SymCodeVerifier can detect violations in 100% of the forced-equation steps, not 12% of free-form steps.

This is fundamentally different from every previous attempt (regex, LLM-as-extractor, Z3, HERMES). It changes the generation protocol, not the extraction protocol.

**Experiments:** Exps 653-656 (StructuredEquationForcer → HERMES v2 Structured → Ensemble gate v3 → VR #18)

### Gap 2: Safety Product Line — Prompt-Injection EBM Not Yet Delivered

**Root cause:** Four prior attempts (Exps 387, 393, 407, 416) all landed in "partial" state with no honest verdict. The design doc (openspec/change-proposals/prompt-injection-ebm.md) is complete, teacher model cached, spec requirements written (REQ-SAFE-007/008/009).

**The fix:** Exp 652 (already in research-roadmap.yaml): one consolidated run with 90-min hard watchdog, enforced honest_verdict enum, and teacher model pre-verified.

**Experiments:** Exp 652 (already queued, starts the milestone)

### Gap 3: JEPA + Cascade Deployment — Models Not Wired Into Production Path

**Root cause:** JEPA v14 is calibrated (Exp 646 Platt scaling) and OTV is evaluated (Exp 647). Neither has been deployed in ThreeTierPipeline as the default Tier 2. The cascade still uses the old EORM+JEPA without temperature calibration. Similarly, SpecGuard (arXiv 2604.15244) offers a new Tier 0f that's faster than any existing cascade member, but hasn't been implemented.

**The fix:** Exps 657-658 deploy the calibrated JEPA v14 and implement SpecGuard as a new cascade tier.

**Experiments:** Exps 657-658 (JEPA cascade deployment + SpecGuard step verification)

---

## Architecture: Verification Cascade After Milestone 2026.04.50

```
Input: LLM response (text)
         |
[Tier 0a] CarnotThinkProbe  — generative 3-step CoT verdict
         | short-circuit if verdict='incorrect'
[Tier 0b] SpilledEnergyDetector — per-token logit-discrepancy signal
         | skip if model confident
[Tier 0c] NUPProbeV4  — contrastive energy probe (AUC=1.0 in-distribution)
         | skip if score <= nup_threshold
[Tier 0d] HallucinationBasinDetector — latent-space basin depth
         | skip if basin_risk_score <= basin_threshold
[Tier 0e] HalluField  — thermodynamic partition-function variance
         | advisory only (no early-exit)
[Tier 0f] SpecGuardVerifier [NEW .50] — log-prob + attention step verification
         | skip if step_rejection_score <= specguard_threshold
[Tier 1 ] SinkProbe  — attention sink concentration (skip >30% at FNR 0%)
         | skip if mean_sink_score >= threshold
[Tier 2 ] EORM → JEPA v14 + Platt [UPGRADED .50] — CoT energy reward model
         | skip if energy < eorm_threshold
[Tier 2.5] SymCodeVerifier  — executable Python step verification (AUC=0.804 live)
         | skip if no arithmetic violations
[Tier 2.6] HermesV2LiveLoop + StructuredEquationForcer [NEW .50]
         | step-by-step generation with forced equations + HERMES v2 verification
[Tier 2.7] CausalReasoningVerifier  — step entailment checking (causal_recall > 0.12)
         | skip if no causal violations
[Tier 3 ] Ising  — full constraint verification (0.006ms/constraint)
```

New in .50:
- Tier 0f: SpecGuardVerifier (Exp 658) — log-prob + attention, sub-millisecond
- Tier 2 upgrade: JEPA v14 + Platt calibration (Exp 657)
- Tier 2.6 upgrade: StructuredEquationForcer (Exp 653) wired into HermesV2LiveLoop

---

## Dependency Graph

```
Exp 652 (prompt-injection EBM)           — independent
Exp 653 (StructuredEquationForcer)       — independent (CPU)
Exp 654 (HERMES v2 Structured, GPU)      — depends on Exp 653
Exp 655 (Ensemble gate v3)               — depends on Exps 641,642,654
Exp 656 (VR #18, GPU, GATED)             — depends on Exp 655 gate_open
Exp 657 (JEPA v14 cascade deploy)        — depends on Exp 646 results
Exp 658 (SpecGuard step verifier)        — independent (CPU, uses cached logits)
Exp 659 (FR-11 Tier 2 relay, mandatory) — depends on Exp 656 results
Exp 660 (LSEBMCL constraint memory)     — independent (CPU)
Exp 661 (KV260 N=64 benchmark, FPGA)    — GATED on CARNOT_KV260_BITFILE
Exp 662 (Ising v3 RTL implementation)   — independent (writes Verilog, no HW needed)
Exp 663 (HALP pre-generative probe)     — independent (CPU, needs cached activations)
Exp 664 (DualGPU parallel retrain, GPU) — depends on Exp 640 DualGPURetrain module
Exp 665 (retro)                          — depends on all
```

Recommended execution order: 652 || 653 || 657 || 658 || 660 || 662 || 663
                              → 654 (GPU, after 653)
                              → 655 (after 641/642/654 results)
                              → 656 (GPU, gated on 655)
                              → 659 (FR-11, after 656)
                              → 661 (FPGA, if bitfile ready)
                              → 664 (GPU, after 640 module verified)
                              → 665 (retro)

---

## Phase Descriptions

### Phase 0: Prompt-Injection Safety (Exp 652)
Close the 4-attempt-old safety/jailbreak classifier thread in one consolidated run.
Teacher model (gpt-oss-safeguard-20b Q4_K_M) pre-cached in models/.
Success criterion: auroc >= 0.90 on held-out 400-example set.
Honest verdict enum enforced via 90-min ExperimentTimeoutWatchdog.

### Phase 1: Structured Equation Forcing (Exps 653-655)
Root cause fix for RETRO-070/033. Instead of extracting equations from prose,
FORCE the model to write equations. StructuredEquationForcer (Exp 653) prepends
a system instruction: "At each arithmetic step, write: COMPUTE: X op Y = Z."
HERMES v2 Structured (Exp 654) runs the live loop on forced-equation responses.
Ensemble gate v3 (Exp 655) combines all detection signals; threshold 0.30.

### Phase 2: VR Attempt #18 (Exp 656, GATED)
Live verify-repair using structured equation forcing + ensemble extractor.
Gated on Exp 655 gate_open=True. If gate closed: write blocked artifact, do NOT run.
RETRO-033 attempt #18: this is the first attempt with forced-equation generation.

### Phase 3: Cascade Upgrade (Exps 657-658)
Exp 657: Deploy Platt-calibrated JEPA v14 as default Tier 2; measure ECE and throughput.
Exp 658: SpecGuard step verification — implement log-prob + attention step-boundary signal
         from arXiv 2604.15244. CPU-safe prototype using cached generation logits.
         If AUC >= 0.70 on live_pairs_578: deploy as Tier 0f.

### Phase 4: Self-Learning (Exps 659-660, FR-11 Mandatory)
Exp 659: Tier 2 FR-11 relay — wire Exp 656 violations into ConstraintTemplateLibrary;
         measure cross-session FP rate delta.
Exp 660: LSEBMCL continual learning (arXiv 2501.05495) — EBM replay to prevent forgetting
         when updating constraint templates across sessions.

### Phase 5: Hardware (Exps 661-662)
Exp 661: KV260 N=64 benchmark — GATED on CARNOT_KV260_BITFILE.
         N=64 rebuild was in-flight as of 2026-04-21; target hardware_latency < 100us.
         RETRO-072 closure: if N=64 fits XCK26 (48.5% LUT post-synth), deploy bitfile.
Exp 662: Ising sampler v3 RTL — implement hardware/kv260/ising_sampler_v3.v from
         the spec written by Exp 648. Add h_ema register + EMA update stage.
         Even if Vivado not available: write and verify RTL+testbench for future synthesis.

### Phase 6: Research Frontiers (Exps 663-664)
Exp 663: HALP pre-generative probe (arXiv 2603.05465) — question-end hidden-state MLP.
         Trains on FOVER corpus, requires NO generation. If AUC >= 0.75: Tier 0g candidate.
Exp 664: DualGPU parallel EORM+JEPA retrain — FINALLY prove dual-GPU parallel training.
         Use DualGPURetrain module (Exp 640), ThreadPoolExecutor, EORM on cuda:0 + JEPA on cuda:1.
         Measure GPU-1 utilization > 50% during training (RETRO-071 alternative resolution).

### Phase 7: Retrospective (Exp 665)
Analyze success criteria, open/close RETROs, wall-time comparison vs .49.

---

## Success Criteria

| Criterion | Experiment | Threshold |
|-----------|------------|-----------|
| prompt_injection_auroc_met | Exp 652 | auroc >= 0.90 |
| equation_forcer_parses_100pct | Exp 653 | detection_rate_on_forced == 1.0 |
| hermes_v2_structured_recall | Exp 654 | recall >= 0.30 |
| ensemble_gate_v3_open | Exp 655 | ensemble_recall >= 0.30 |
| retro_033_resolved | Exp 656 | signed_improvement > 0 |
| jepa_v14_deployed | Exp 657 | cascade_ece < 0.10 AND auc_delta <= 0.02 |
| specguard_viable | Exp 658 | specguard_auc >= 0.70 |
| fr11_real_violations | Exp 659 | fr11_real_violations_confirmed == True |
| lsebmcl_no_forgetting | Exp 660 | forgetting_rate < 0.05 across 3 sessions |
| kv260_n64_hardware | Exp 661 | hardware_latency_us < 100 |
| ising_v3_rtl_written | Exp 662 | rtl_written == True |
| halp_viable | Exp 663 | halp_auc >= 0.75 |
| dualgpu_parallel_proven | Exp 664 | peak_gpu1_util > 50 during training |

---

## Open RETROs Addressed

| RETRO | Description | .50 Action |
|-------|-------------|------------|
| RETRO-033 | VR 17 consecutive failures | Exp 654-656: structured forcing + VR #18 |
| RETRO-057 | LowRankKAEM accuracy gap | Filed for .51 (multilevel approach needed) |
| RETRO-070 | Extraction ceiling 12% recall | Exp 653-655: structured equation forcing |
| RETRO-071 | DualGPU unproven | Exp 664: parallel EORM+JEPA retrain |
| RETRO-072 | KV260 N=64 sizing | Exp 661: N=64 hardware benchmark |
| RETRO-CRITICAL | Exclusion manifest unwired | Exp 640 done; human must wire into conductor |

---

## New arxiv Findings Incorporated

| Paper | Where Used |
|-------|------------|
| arXiv 2604.15244 (SpecGuard) | Exp 658 — Tier 0f step verification |
| arXiv 2603.05465 (HALP) | Exp 663 — pre-generative hallucination probe |
| arXiv 2501.05495 (LSEBMCL) | Exp 660 — continual EBM replay for constraint memory |

---

## Hardware Requirements

| Experiment | Hardware | Requirement |
|------------|----------|-------------|
| Exp 652 | CPU (teacher model inference) | gpt-oss-safeguard-20b Q4_K_M in models/ |
| Exp 654 | GPU REQUIRED | CARNOT_FORCE_LIVE=1, Qwen3.5-0.8B on cuda:0 |
| Exp 656 | GPU REQUIRED | CARNOT_FORCE_LIVE=1, 90-min budget |
| Exp 661 | FPGA REQUIRED | CARNOT_KV260_BITFILE set (N=64 bitfile) |
| Exp 664 | 2x GPU REQUIRED | 2 RTX 3090s, CARNOT_FORCE_LIVE=1 |

All other experiments run on CPU only.

---

## Experiment Summary

| ID | Title | Hardware | Deliverable |
|----|-------|----------|-------------|
| 652 | Prompt-Injection EBM KAN | CPU (teacher model) | results/experiment_652_prompt_injection_kan.json |
| 653 | StructuredEquationForcer | CPU | results/experiment_653_equation_forcer.json |
| 654 | HERMES v2 Structured Live | GPU | results/experiment_654_hermes_v2_structured.json |
| 655 | Ensemble Recall Gate v3 | CPU | results/experiment_655_ensemble_gate_v3.json |
| 656 | Live VR Attempt #18 (GATED) | GPU | results/experiment_656_live_vr_attempt_18.json |
| 657 | JEPA v14 Cascade Deployment | CPU | results/experiment_657_jepa_cascade_deploy.json |
| 658 | SpecGuard Step Verifier | CPU | results/experiment_658_specguard_verifier.json |
| 659 | FR-11 Tier 2 Cross-Session Relay | CPU | results/experiment_659_tier2_fr11_relay.json |
| 660 | LSEBMCL Constraint Memory | CPU | results/experiment_660_lsebmcl_memory.json |
| 661 | KV260 N=64 Benchmark (GATED) | FPGA | results/experiment_661_kv260_n64_benchmark.json |
| 662 | Ising Sampler v3 RTL | CPU (writes RTL) | hardware/kv260/ising_sampler_v3.v |
| 663 | HALP Pre-Generative Probe | CPU | results/experiment_663_halp_probe.json |
| 664 | DualGPU Parallel Retrain | 2x GPU | results/experiment_664_dualgpu_retrain.json |
| 665 | Milestone 2026.04.50 Retrospective | CPU | results/experiment_665_retro_2026_04_50.json |
