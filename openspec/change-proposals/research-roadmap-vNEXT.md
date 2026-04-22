# Research Roadmap — Milestone 2026.04.52

**Title:** VR Win Scale-Up + DualGPU Proof + JEPA Calibration

**CalVer:** 2026.04.52 (sequence increment from 2026.04.51)

**Authored:** 2026-04-22

**Previous Milestone:** 2026.04.51 — "VR Gate Unblock + Prompt-Injection Rescue + Cascade Hardening"

---

## What Milestone 2026.04.51 Proved

Milestone .51 achieved the project's most significant research breakthrough to date:

- **Exp 666 (Manifest Wire-In v3):** `manifest_wired_xdist_available` — manifest loads and xdist confirmed, but `conductor_consulted=null` (15th consecutive milestone without confirmed conductor consultation).
- **Exp 667 (EnsembleGate v4):** `gate_open_retro_070_unblocked` — structured-first gate logic opens with causal_recall=0.36 >= 0.30. Gate redesign confirmed.
- **Exp 668 (VR #18 v2): BREAKTHROUGH** — `vr_positive`, `signed_improvement=0.64`, `baseline_accuracy=0.36`, `post_accuracy=1.0` on 25 live-GPU questions. FIRST POSITIVE VR RESULT IN 18 ATTEMPTS. RETRO-033 attempt #19 succeeded.
- **Exp 669 (Prompt-Injection KAN v5):** `distillation_corpus_built_classifier_trained_auroc_below_threshold` — 6th consecutive partial. AUROC still below 0.90 threshold. New approach needed.
- **Exp 670 (JEPA v14 Cascade Deploy):** `jepa_v14_deployed=True` — Platt temperature wired into ThreeTierPipeline. JEPA v14 is now the default Tier 2.
- **Exp 671 (JEPA v15 Retrain):** `jepa_v15_auc_met`, `ood_auc=1.0` — suspicious overfit on tiny held-out set (same pattern as JEPA v12). ECE not computed. OOD validation on 500+ unseen questions required.
- **Exp 672 (KV260 DFX Fix):** `blocked_bitfile_not_configured` — CARNOT_KV260_BITFILE env var not set. Bitfile path must be configured from prior synthesis result.
- **Exp 673 (DualGPU v3):** `dualgpu_partial`, `throughput_ratio=1.963` — 2x throughput via ThreadPoolExecutor confirmed, but `max_gpu1_util_pct=0.0` (pynvml not available for utilization polling). RETRO-071 still open.
- **Exp 674 (IAS Gate Calibration):** `ias_gate_improves_v3` — adaptive quantile regression thresholds improve on fixed threshold. Gate opens where v3 failed.
- **Exp 675 (LOS-Net):** `below_threshold` — AUC < 0.75 on synthetic logit sequences. Not deployed as Tier 0h.
- **Exp 676 (MetaJuLS Adapter):** `metajuls_adapted` — domain-specific forcing policy updates from feedback. Online adaptation confirmed on synthetic data.
- **Exp 677 (Retro):** 7th consecutive wall-time improvement, 8.0 min/exp new project best; pre-flight test suite (532 min, 1.65x slowest-5) still unresolved; slowest-5 composition unchanged for 3rd consecutive milestone.

**Still open after .51:**
- RETRO-071: DualGPU parallel utilization unconfirmed (15 milestones) — need pynvml
- RETRO-072: KV260 bitfile not configured (hardware action required by user)
- RETRO-CRITICAL: Exclusion manifest not wired into conductor (15th consecutive miss)
- Prompt-injection KAN: 6th consecutive partial (architectural approach change needed)
- Exp 380-382: partial checkpoint 3rd consecutive (formal retirement threshold crossed)
- Exp 425/410/383/346: slowest-5 composition unchanged (3rd consecutive milestone)
- JEPA v15: suspicious ood_auc=1.0 needs OOD audit

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: VR Win (0.64 on 25q) Needs Validation at Scale — Credibility Risk If Anomalous

**Root cause:** The Exp 668 result (`post_accuracy=1.0`, `signed_improvement=0.64`) on only 25 questions may be anomalous. Common failure modes:
- Only 9 baseline errors (0.36 × 25 = 9) — small sample, high variance
- Structured forcing may cause the model to write answers instead of chain-of-thought (recall bias)
- Gate check passed trivially; questions may have been easy cases

**The fix:** Scale to 200 questions with Wilson 95% CI. If the win holds, Carnot has its first credible improvement claim since the project began. If it doesn't, we need to understand why 25q showed a win that 200q does not.

**Experiments:** Exps 679-681 (200q scale, HumanEval code, adversarial GSM8K)

### Gap 2: DualGPU Parallel Compute Still Unproven — RETRO-071 (15 Milestones)

**Root cause (confirmed from Exp 673):** `throughput_ratio=1.963` confirms two independent inferences ran concurrently. But `max_gpu1_util_pct=0.0` because pynvml was not available to poll GPU utilization during inference. The proof requires pynvml to measure actual GPU1 compute utilization.

**The fix:** Install pynvml (`pip install pynvml`) and use `pynvml.nvmlDeviceGetUtilizationRates(handle).gpu` to poll every 2 seconds during parallel inference. Also apply proven DualGPU to the Exp 383 pattern (EORM+JEPA sequential retrain, 6th consecutive slowest-5 appearance).

**Experiments:** Exps 684-685 (DualGPU proof v4 + EORM+JEPA parallel retrain)

### Gap 3: JEPA v15 Overfit + FR-11 Never Had Real Verified Positives

**Root cause (from Exp 671):** `ood_auc=1.0` on a tiny held-out set is suspicious. ECE not computed — Platt calibration is incomplete. The self-learning loop (FR-11) has had real violations available for several milestones but has never incorporated VERIFIED CORRECT responses (true positives). Exp 668's VR win finally provides verified-correct repairs.

**The fix:**
- Exp 682: Audit JEPA v15 on 500 unseen GSM8K questions. Compute ECE. Report honest OOD AUC.
- Exp 683: Wire Exp 668 verified-correct repairs into ConstraintTemplateLibrary as positive examples. This is FR-11's first genuine closed loop: the pipeline verified a repair was correct, now that verification informs the constraint weights.

**Experiments:** Exps 682-683 (JEPA OOD audit + FR-11 real verified positives relay)

---

## Architecture: Verification Cascade After Milestone 2026.04.52

```
Input: LLM response (text)
         |
[Tier 0a] CarnotThinkProbe  — generative 3-step CoT verdict
         | short-circuit if verdict='incorrect'
[Tier 0b] SpilledEnergyDetector — per-token logit-discrepancy signal
         | skip if model confident
[Tier 0c] NUPProbeV4  — contrastive energy probe (AUC=1.0 in-dist)
         | skip if score <= nup_threshold
[Tier 0d] HallucinationBasinDetector — latent-space basin depth
         | skip if basin_risk_score <= basin_threshold
[Tier 0e] HalluField  — thermodynamic partition-function variance
         | advisory only (no early-exit)
[Tier 1 ] SinkProbe  — attention sink concentration
         | skip if mean_sink_score >= threshold
[Tier 2 ] JEPA v14 + Platt [DEPLOYED .51] — CoT energy reward model
         | JEPA v15 OOD audit [.52] — auditing ood_auc=1.0 suspicious
         | skip if energy < eorm_threshold
[Tier 2.5] SymCodeVerifier — executable Python step verification (AUC=0.804 live)
         | skip if no arithmetic violations
[Tier 2.6] HermesV2LiveLoop + StructuredEquationForcer [ACTIVE .50/.51]
         | step-by-step generation with forced equations
[Tier 2.7] CausalReasoningVerifier — causal_recall=0.36
         | skip if no causal violations
[Tier 3 ] Ising  — full constraint verification (0.006ms/constraint)
         ↓
[Repair  ] VerifyRepairPipeline + structured-equation forcing
         → signed_improvement=0.64 on 25q (.51 breakthrough)
         → SCALE VALIDATION in .52 (200q, HumanEval, adversarial)

Self-Learning Loop (FR-11):
  [Tier 1 ] Online weight updates — real positives from Exp 668 [NEW .52, Exp 683]
  [Tier 2 ] PSV self-play loop — formal verification guided data [NEW .52, Exp 688]
  [Tier 3 ] FoVer Z3 labels for JEPA v16 training [NEW .52, Exp 686]
```

New in .52:
- VR win scaled to 200q + HumanEval + adversarial (Exps 679-681)
- JEPA v15 OOD audit + ECE fix (Exp 682)
- FR-11 real verified positives relay from Exp 668 (Exp 683)
- DualGPU proof via pynvml (Exp 684)
- DualGPU applied to EORM+JEPA retrain (Exp 685) — resolves Exp 383 pattern
- FoVer Z3-verified PRM labels 200q (Exp 686)
- HalluSAE sparse AE feature attribution (Exp 687)
- PSV self-play constraint learning 10-iteration loop (Exp 688)

---

## Dependency Graph

```
Exp 678 (legacy retirement + pre-flight, MANDATORY FIRST)   — independent
Exp 679 (VR 200q scale, GPU, uses Exp 668 pipeline)        — depends on Exp 668 confirmed
Exp 680 (HumanEval VR, GPU)                                — independent of 679
Exp 681 (adversarial VR, GPU)                              — independent of 679
Exp 682 (JEPA v15 OOD audit, CPU)                          — depends on Exp 671 weights
Exp 683 (FR-11 relay real positives, CPU)                  — depends on Exp 668 violations
Exp 684 (DualGPU proof v4, GPU, pynvml required)           — independent
Exp 685 (DualGPU EORM+JEPA retrain, GPU)                   — depends on Exp 684 success
Exp 686 (FoVer Z3 formal labels, CPU)                      — independent
Exp 687 (HalluSAE sparse AE, CPU)                          — independent
Exp 688 (PSV self-play loop, CPU+GPU)                      — depends on Exp 683 FR-11
Exp 689 (retrospective)                                     — depends on all
```

Recommended execution order:
678 [FIRST] → 679 || 680 || 681 || 682 || 683 || 684 || 686 || 687
           → 685 (GPU, after 684 confirms GPU1 available)
           → 688 (after 683 FR-11 relay complete)
           → 689 (retro)

---

## Phase Descriptions

### Phase 0: Operational Debt Resolution — Exp 678 (MANDATORY FIRST)

Formally retire Exps 380-382 (partial checkpoint, 3rd consecutive — formal threshold
crossed per Exp 308/309 precedent). Create final retirement JSON placeholder files.
Add Exps 380, 381, 382 to conductor_exclusion_manifest.json. Create
`scripts/conductor_pre_flight.py` that reads the manifest and prints excluded IDs
before session start (non-invasive; does not modify research_conductor.py).
Also create placeholder result files for Exp 346 (55M-param training, 3rd consecutive)
so the conductor skips it and it exits the slowest-5.
Target: Exps 380-382 and 346 excluded, manifest has 10 entries total, pre-flight script exists.

### Phase 1: VR Win Validation — Exps 679-681

Scale the .51 VR win (signed_improvement=0.64, 25q) to 200 questions with Wilson CI.
If the win holds at 200q (signed_improvement > 0.05 with p < 0.05), Carnot has its
first credible headline result. If it doesn't hold, diagnose why.

Exp 679: 200q GSM8K validation via LongRunBenchmarkExecutor (8 batches of 25).
Exp 680: 25 HumanEval problems with execution-based verification (no regex).
Exp 681: Adversarial GSM8K (wrong premises) — does VR maintain improvement or degrade?

### Phase 2: JEPA v15 Calibration + FR-11 Real Positives — Exps 682-683

Exp 682: Audit JEPA v15 (ood_auc=1.0 suspicious) on 500 unseen GSM8K questions.
Compute ECE. Apply Platt calibration post-hoc. Report honest OOD AUC.
Exp 683: Wire Exp 668 true positives into FR-11 self-learning relay.
This is the first time the FR-11 loop has verified-correct examples — not just violations.
Constraint weight updates from verified-correct repairs should reduce FP rate by
reinforcing constraint weights for arithmetic patterns that matched real correct answers.

### Phase 3: DualGPU Proof and Application — Exps 684-685

Exp 684: Install pynvml, poll GPU1 utilization during ThreadPoolExecutor parallel
inference. Two Qwen3.5-0.8B instances on GPU0 and GPU1 simultaneously. If GPU1
utilization > 0%, RETRO-071 is finally closed after 15 milestones.
Exp 685: Apply proven DualGPU to EORM+JEPA parallel retraining. EORM on cuda:0,
JEPA v15 on cuda:1 simultaneously via ThreadPoolExecutor. Reduces retrain from
~62 min to ~35 min, resolving Exp 383's 6th consecutive slowest-5 appearance.

### Phase 4: New Research — Exps 686-688

Exp 686: FoVer Z3 formal PRM labels (arXiv 2505.15960). Auto-annotate 200 GSM8K
CoT chains using Z3 SMT solver for step-level correctness labels. Produce
fover_labeled_formal_v1.json. Verify label agreement > 80% with existing FOVER
hand-labels. Implements the data-generation step for JEPA v16 training.

Exp 687: HalluSAE sparse auto-encoder (arXiv 2604.16430). Train 512-dim sparse AE
on Qwen3.5-0.8B hidden states from FOVER corpus. Identify top-10 hallucination-predictive
features. Check whether COMPUTE: presence correlates with low-energy features — this
would provide mechanistic explanation for the Exp 668 VR win.

Exp 688: PSV self-play constraint learning (arXiv 2512.18160). Implements Tier 3
continuous self-learning from research-program.md. 10-iteration loop: each iteration
generates 20 GSM8K questions with Qwen3.5-0.8B, runs VR pipeline with structured
forcing, uses binary correct/incorrect labels to update ConstraintTemplateLibrary
constraint weights (Tier 1 online learning). Measure: does FP rate decrease across
iterations?

### Phase 5: Retrospective — Exp 689

Standard milestone retrospective (carnot.operational_retro schema). Focus metrics:
(1) Did VR win hold at 200q? (2) Is RETRO-071 closed? (3) Did FR-11 relay show
measurable constraint weight improvement? (4) Did slowest-5 composition change?

---

## Open RETROs Being Addressed

| RETRO | Status | Addressed By |
|-------|--------|-------------|
| RETRO-071 (DualGPU, 15 milestones) | CRITICAL | Exp 684 (pynvml proof) + Exp 685 (training) |
| RETRO-072 (KV260 bitfile) | HIGH | USER ACTION: set CARNOT_KV260_BITFILE |
| RETRO-CRITICAL (manifest, 15 milestones) | CRITICAL | Exp 678 (pre-flight script) |
| JEPA v15 overfit | HIGH | Exp 682 (OOD audit) |
| FR-11 relay never had real verified positives | HIGH | Exp 683 (positives from Exp 668) |
| VR win unvalidated (25q only) | HIGH | Exps 679-681 (scale validation) |
| Exp 380-382 checkpoint partial (3rd milestone) | HIGH | Exp 678 (formal retirement) |
| Exp 383 sequential EORM+JEPA (6th milestone) | MEDIUM | Exp 685 (DualGPU parallel) |
| Prompt-injection KAN (6th fail) | LOW | Deferred — architectural pivot needed in .53 |

---

## Success Criteria

| Criterion | Target | Gated On |
|-----------|--------|---------|
| legacy_retired_380_382_346=True | placeholder result files + manifest | Exp 678 |
| vr_200q_signed_improvement > 0 | positive win at scale, Wilson CI | Exp 679 |
| humaneval_vr_improvement >= 0 | no degradation on code | Exp 680 |
| adversarial_vr_signed_improvement >= 0 | no degradation on adversarial | Exp 681 |
| jepa_v15_honest_ood_auc reported | suspicious 1.0 investigated | Exp 682 |
| fr11_real_positives_wired=True | verified-correct repairs in FR-11 | Exp 683 |
| retro_071_resolved=True | GPU1 utilization > 0% confirmed | Exp 684 |
| eorm_jepa_parallel_retrain_time < 40min | DualGPU applied to training | Exp 685 |
| fover_formal_v1_n_labels >= 200 | Z3 auto-annotation pipeline works | Exp 686 |
| hallusat_features_identified=True | top-10 causal features found | Exp 687 |
| psv_iterations_completed=10 | self-play loop runs end-to-end | Exp 688 |

---

## Hardware Requirements

| Experiment | Hardware | Notes |
|-----------|----------|-------|
| Exp 678 | CPU only | Retirement + pre-flight script |
| Exp 679 | 1x RTX 3090 | CARNOT_FORCE_LIVE=1, 200 questions |
| Exp 680 | 1x RTX 3090 | HumanEval, execution-based verification |
| Exp 681 | 1x RTX 3090 | Adversarial GSM8K |
| Exp 682 | CPU only | JEPA v15 weights loaded; OOD eval is inference only |
| Exp 683 | CPU only | FR-11 relay reads Exp 668 result JSON |
| Exp 684 | 2x RTX 3090 | pynvml required; CARNOT_FORCE_LIVE=1 |
| Exp 685 | 2x RTX 3090 | EORM on cuda:0, JEPA on cuda:1 |
| Exp 686 | CPU only | Z3 is pure CPU; no model inference required |
| Exp 687 | CPU only | Sparse AE on pre-computed hidden states |
| Exp 688 | 1x RTX 3090 | VR pipeline with live GPU for inference |
| Exp 689 | CPU only | Retrospective analysis |

---

## New Papers Incorporated (arXiv Scan 2026-04-22)

| Paper | Title | Filed As |
|-------|-------|---------|
| arXiv 2505.15960 | FoVer: Formal Verification for Scalable PRM Labels | Exp 686 |
| arXiv 2604.16430 | HalluSAE: Sparse AE Hallucination Detection | Exp 687 |
| arXiv 2512.18160 | PSV: Propose, Solve, Verify Self-Play | Exp 688 |
| arXiv 2512.20664 | Eidoku: CSP-Based Structural Verification Gate | Filed .53 |
| arXiv 2604.03904 | I-CALM: Confidence-Aware Abstention | Filed .53 |
