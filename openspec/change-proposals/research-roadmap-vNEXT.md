# Research Roadmap — Milestone 2026.04.53

**Title:** JEPA v16 Recovery + VR Credibility Hardening + Prompt-Injection True Distillation

**CalVer:** 2026.04.53 (sequence increment from 2026.04.52)

**Authored:** 2026-04-22

**Previous Milestone:** 2026.04.52 — "VR Win Scale-Up + DualGPU Proof + JEPA Calibration"

---

## What Milestone 2026.04.52 Proved

Milestone .52 delivered the largest single-milestone wall-time gain in project history
(-248 min, -5.9%), with per-experiment average reaching a new project best of 7.0 min/exp.
Key research outcomes:

- **Exp 678 (Legacy Retirement v2):** `retirements_complete_preflight_confirmed` —
  Exps 380-382 and 346 formally retired. Exclusion manifest updated. Conductor pre-flight
  script confirmed operational. `manifest_consulted=True` for the 11-experiment .52 cycle.
- **Exp 679 (VR 200q Scale):** `vr_200q_positive`, `signed_improvement=1.0` —
  VR win scales from 25q to 200q. Wilson 95% CI holds. RETRO-033 attempt #19 confirmed
  at full scale. First credible 200q headline result.
- **Exp 680 (HumanEval VR):** `code_vr_positive` — execution-based code verification
  shows improvement. Assertion-comment forcing extracts verifiable intermediate claims.
- **Exp 681 (Adversarial VR):** `adversarial_robust` — structured-equation forcing
  does not introduce adversarial brittleness. Signed improvement >= 0 under adversarial
  perturbations.
- **Exp 682 (JEPA v15 OOD Audit):** `jepa_v15_ood_below_random`, `true_ood_auc=0.4751`
  — JEPA v15 is anti-correlated on truly unseen GSM8K questions (500-699). This is an
  architecture regression, not a data issue. CASCADE DEPLOYMENT BLOCKED.
- **Exp 683 (FR-11 Real Positives):** `positives_wired_fp_reduced` — Exp 668
  verified-correct repairs wired into ConstraintTemplateLibrary. fp_rate_delta < 0.
  First FR-11 cycle closing the complete loop: pipeline verified a repair, now that
  verification informs the constraint weights.
- **Exp 684 (DualGPU pynvml):** `dualgpu_confirmed`, `max_gpu1_util_pct > 0` —
  RETRO-071 CLOSED after 15 consecutive milestones. pynvml installed, GPU1 utilization
  confirmed during parallel inference.
- **Exp 685 (DualGPU EORM+JEPA):** `dualgpu_retrain_success`, `speedup=2.0175x` —
  EORM+JEPA parallel retrain on dual GPUs. Exp 383 pattern resolved. Slowest-5
  composition improved for first time in 7 milestones.
- **Exp 686 (FoVer Z3 Formal Labels):** `fover_z3_success` — 200 Z3-labeled step pairs
  generated. Agreement with hand-labels > 80%. fover_labeled_formal_v1.json ready as
  JEPA v16 training data.
- **Exp 687 (HalluSAE Sparse AE):** `hallusat_compute_line_causal` — Top-10
  hallucination features identified. COMPUTE: line count is in the top-10, validating
  the structured-equation forcing VR win mechanism.
- **Exp 688 (PSV Self-Play):** `psv_synthetic_mode` — 10-iteration PSV loop ran in
  synthetic mode (FR-11 gate condition not met for live mode). FP rate trend slope < 0
  (improving direction) in synthetic evaluation.

**Still open after .52:**
- RETRO-072: KV260 bitfile not configured (Vivado required, user hardware action)
- JEPA v15: OOD AUC=0.4751 (below random) — CASCADE BLOCKED, .53 Phase 0 mandatory audit
- Exp 425/410: slowest-5 recurring (exit threshold crossed — Exp 425 is 16th consecutive)
- PSV self-play: ran in synthetic mode; needs real live data in .53
- User-pinned: Exp 690 (Prompt-Injection True Distillation) and Exp 691 (Cross-Dataset)

---

## The 3 Biggest Gaps Between Current State and PRD Vision

### Gap 1: JEPA v15 Architecture Regression — Cascade Blocked (Priority 0)

**Root cause:** JEPA v15 OOD AUC=0.4751 means the predictor is worse than random on
unseen GSM8K questions. This is not a training data problem — FoVer formal v1 provides
200+ Z3-labeled pairs. The regression is architectural. Possible root causes:
- CPMI hard-negative pairs from a narrow distribution (all GSM8K carry errors) create
  anti-correlated scores on diverse OOD arithmetic patterns
- PUREMinFormLoss min-form objective pushes scores DOWN on incorrect steps but also
  DOWN on many correct steps the model sees as borderline
- JEPA latent features lack compositional structure for OOD generalization

**The fix (two-stage):**
1. Exp 693: Root cause audit — identify which of the three hypotheses is responsible.
   Use the discrete symbol probing approach from arXiv 2603.20327 to check whether JEPA
   latents lack compositional structure. If PUREMinFormLoss is the issue, redesign to use
   contrastive ranking loss instead.
2. Exp 698: JEPA v16 retrain on FoVer formal v1 data with the fixed architecture.
   Target: OOD AUC >= 0.70 on GSM8K 500-699 (never seen in training).

### Gap 2: VR Win Credibility — First Publishable Result Needs Cross-Model Validation

**Root cause:** Exp 679 confirmed signed_improvement=1.0 on 200q with Qwen3.5-0.8B.
This is statistically significant but single-model. The PRD requires results that
generalize across architectures. If improvement is model-specific, it might be an
artifact of how Qwen3.5-0.8B responds to structured forcing rather than a general
constraint-verification improvement.

**The fix:** Run the same VR pipeline on Gemma4-E4B-it. If both Qwen3.5-0.8B and
Gemma4-E4B-it show positive signed_improvement with Wilson CI lower bound > 0, the
result is credible for publication. Also test on 50 "hard" questions (model baseline < 40%)
to confirm the improvement is not driven by easy recall-bias questions.

**Experiments:** Exp 694 (VR cross-model validation + hard questions)

### Gap 3: Self-Learning FR-11 Loop Still Synthetic — PSV Needs Real Live Data

**Root cause:** Exp 688 (PSV Self-Play) ran in synthetic mode because the FR-11 gate
condition was not met in .52 (PSV requires fr11_real_positives_confirmed=True, which
came after PSV was executed). Now that Exp 683 confirmed positives wired and
signed_improvement=1.0, the PSV loop can run with real live VR data as training signal.

The complete self-learning loop (research-program.md Tier 3) requires:
1. VR pipeline produces verified-correct repairs (DONE: Exp 683)
2. PSV loop uses VR results as binary verification labels (NEEDED: Exp 697)
3. JEPA predictor trained on PSV-generated labels (NEEDED: after Exp 698 v16 base)

**Experiments:** Exp 697 (PSV real-data 10-iteration loop), Exp 698 (JEPA v16 with PSV data)

---

## Architecture: Verification Cascade After Milestone 2026.04.53

```
Input: LLM response (text)
         |
[Tier 0a] CarnotThinkProbe  — generative 3-step CoT verdict
         | short-circuit if verdict='incorrect'
[Tier 0b] SpilledEnergyDetector — per-token logit-discrepancy
         | skip if model confident
[Tier 0c] NUPProbeV4  — contrastive energy probe (AUC=1.0 in-dist)
         | skip if score <= nup_threshold
[Tier 0d] HallucinationBasinDetector — latent-space basin depth
         | skip if basin_risk_score <= basin_threshold
[Tier 0e] HalluField  — thermodynamic partition-function variance
         | advisory only
[Tier 1 ] SinkProbe  — attention sink concentration
         | skip if mean_sink_score >= threshold
[Tier 2 ] JEPA v16 [TARGET .53, REPLACING v15 BLOCKED] — CoT energy reward
         | skip if energy < eorm_threshold
[Tier 2.5] SymCodeVerifier — executable Python step verification (AUC=0.804 live)
         | skip if no arithmetic violations
[Tier 2.6] HermesV2LiveLoop + StructuredEquationForcer
         | step-by-step generation with forced equations
[Tier 2.7] CausalReasoningVerifier — causal_recall=0.36
         | skip if no causal violations
[Tier 2.8] FormalStepVerifier — FOL intermediary + Z3 entailment check [NEW .53, Exp 695]
         | skip if all steps formally entailed
[Tier 3 ] Ising  — full constraint verification (0.006ms/constraint)
         ↓
[Repair  ] VerifyRepairPipeline + structured-equation forcing
         → signed_improvement=1.0 on 200q (.52 confirmed)
         → CROSS-MODEL VALIDATION in .53 (Gemma4, hard questions, Exp 694)

Self-Learning Loop (FR-11):
  [Tier 1 ] Online weight updates — real verified positives (.52, Exp 683)
  [Tier 2 ] Constraint memory + PSV self-play [REAL DATA .53, Exp 697]
  [Tier 3 ] JEPA v16 trained on FoVer formal v1 [.53, Exp 698]
  [Tier 4 ] HalluSAE causal feature integration [.53, Exp 699]

New in .53:
- JEPA v15 root cause + v16 architecture design (Exp 693)
- Prompt-injection true distillation (Exp 690, user-pinned)
- Cross-dataset generalization gate (Exp 691, user-pinned)
- VR cross-model + hard questions (Exp 694)
- FormalStepVerifier Tier 2.8 (Exp 695, arXiv 2603.29500)
- I-CALM repair abstention (Exp 696, arXiv 2604.03904)
- PSV real-data 10-iteration self-play (Exp 697)
- JEPA v16 retrain on FoVer formal v1 (Exp 698)
- HalluSAE feature integration into JEPA v16 (Exp 699)
- VR publication readiness (Exp 700)
- KV260 Ising v3 synthesis attempt (Exp 701)
- Operational retrospective (Exp 702)
```

---

## Phases and Experiments

### Phase 0: Operational Pre-Flight + JEPA Audit (MANDATORY FIRST)

**Exp 692: Operational Pre-Flight v5 + Slowest-5 Formal Retirement**
- Retire Exp 425 (ExperimentTimeoutWatchdog demo, 16th consecutive slowest-5 appearance)
  and Exp 410 (BatchedInferenceRunner target, 13th consecutive) via retirement files
- Create their missing final result files so conductor skips them
- Update exclusion manifest with 425 and 410
- Verify conductor pre-flight script from Exp 678 still executes correctly
- Confirm manifest_consulted field in pre-flight output
- Deliverable: results/experiment_692_preflight_v5.json

**Exp 693: JEPA v15 Root Cause Audit + v16 Architecture Design**
- Load JEPA v15 weights and FoVer formal v1 labels (from Exp 686)
- Test three hypotheses: (1) CPMI pair distribution mismatch, (2) PUREMinFormLoss
  anti-correlation on OOD, (3) latent features lack compositional structure
- Apply linear probing to JEPA v15 hidden layers (discrete symbol extraction per
  arXiv 2603.20327) — if symbols are non-compositional, confirm hypothesis (3)
- Design JEPA v16 architecture spec: contrastive ranking loss (not min-form),
  discrete symbol features (16-dim probe on latent), train on FoVer formal v1
- Deliverable: results/experiment_693_jepa_v16_design.json

### Phase 1: User-Pinned — Prompt-Injection True Distillation

**Exp 690: Prompt-Injection KAN v1 — True Teacher Distillation (USER-PINNED)**
- Run actual gpt-oss-safeguard-20b inference (not source-label shortcut)
- MANDATORY invariant: teacher_inference_duration_s >= len(corpus) * 0.5
- KAN retrained on teacher labels (not source labels)
- Deliverable: results/experiment_690_prompt_injection_kan_true_distillation.json

**Exp 691: Cross-Dataset Generalization Gate (USER-PINNED, gates on Exp 690)**
- Evaluate KAN v1 on HackAPrompt, BIPIA, and synthetic OWASP mutations
- Mean AUROC >= 0.80: publishable; 0.65-0.80: shareable with caveats
- Deliverable: results/experiment_691_prompt_injection_kan_cross_dataset.json

### Phase 2: VR Credibility Hardening

**Exp 694: VR Cross-Model Validation + Hard Questions (GPU REQUIRED)**
- Run VR pipeline on Gemma4-E4B-it (second model) — same 200q set as Exp 679
- Also run on 50 "hard" GSM8K questions (select where Qwen3.5-0.8B baseline < 40%)
- Apply grammar-constrained decoding for COMPUTE: lines (arXiv 2602.01090 approach)
- Honest verdict: "vr_cross_model_confirmed" if BOTH models positive with CI lower > 0
- Deliverable: results/experiment_694_vr_cross_model.json

### Phase 3: Cascade Tier Improvements

**Exp 695: FormalStepVerifier Tier 2.8 (arXiv 2603.29500 + arXiv 2512.20664, CPU)**
- Implement Tier 2.8: for each CoT step, prompt model to output one FOL proposition,
  verify Z3 entailment from prior steps (arXiv 2603.29500 formal intermediaries)
- Also implement Eidoku CSP structural check (arXiv 2512.20664) as alternative gate
- Compare FormalStep vs Eidoku vs SymCodeVerifier on FOVER corpus
- Winner deployed as Tier 2.8 default
- Deliverable: results/experiment_695_formal_step_verifier.json

**Exp 696: I-CALM Repair Abstention Layer (arXiv 2604.03904, CPU)**
- When energy > violation threshold but < confidence threshold: output "abstain" instead
  of attempting repair (reduces false positive repairs)
- Calibrate confidence threshold on FOVER corpus: maximize F1(abstain/repair/pass)
- Measure FP reduction on 100 synthetic test questions
- Deliverable: results/experiment_696_icalm_abstention.json

### Phase 4: FR-11 Self-Learning Closure

**Exp 697: PSV Real-Data Self-Play 10 Iterations (GPU REQUIRED, Tier 3 Self-Learning)**
- GATE: fr11_real_positives_confirmed=True from Exp 683
- Run 10 PSV iterations with live GPU inference (CARNOT_FORCE_LIVE=1)
- Use PaCoRe-style K=2 parallel chains per GPU (arXiv 2601.05593) to increase diversity
- Each iteration: 20 questions, verify with SymCodeVerifier, update constraint weights
- Measure FP rate trend slope — must be < 0 for self-play working verdict
- Deliverable: results/experiment_697_psv_realdata.json

**Exp 698: JEPA v16 Retrain on FoVer Formal v1 (CPU)**
- Use JEPA v16 architecture from Exp 693 design (fixed loss, discrete symbol features)
- Train on fover_labeled_formal_v1.json (200+ Z3-labeled pairs from Exp 686)
- Target: OOD AUC >= 0.70 on GSM8K 500-699 (never in training)
- Deploy as Tier 2 default if threshold met; cascade-blocked if not
- Deliverable: results/experiment_698_jepa_v16.json + results/jepa_predictor_v16.safetensors

**Exp 699: HalluSAE Feature Integration into JEPA v16 (CPU)**
- GATE: Exp 698 jepa_v16_ood_target_met required
- Replace raw text features with top-10 HalluSAE causal features (from Exp 687)
- Retrain JEPA v16.1 with HalluSAE feature input
- Measure: does causal feature input improve OOD AUC vs. raw features?
- Deliverable: results/experiment_699_jepa_v16_hallusat.json

### Phase 5: Publication

**Exp 700: VR Publication Readiness + HuggingFace Update (CPU)**
- GATE: Exp 694 vr_cross_model_confirmed required for full publication
- If gate open: update README.md with credibility-hardened numbers, update all 16 HF
  model card READMEs, generate carnot-verify-repair-v1 model card on HuggingFace
- If gate closed: update README with honest negative (single-model result, not cross-model)
- Also: update KAN formal verification model card (arXiv 2602.06737 credit)
- Deliverable: results/experiment_700_publication_readiness.json

### Phase 6: Hardware

**Exp 701: KV260 Ising v3 RTL Synthesis Attempt (FPGA, CPU fallback)**
- Synthesize ising_sampler_v3.v (with EMA inertia dynamics from Exp 662)
- If Vivado available: run synthesis, capture post-synth utilization (N=64, MAX_DEGREE=16)
- If not: generate updated synth_ising_v3.tcl targeting v3 RTL; document what changed
  from v2 synth script; write blocked artifact with synthesis instructions
- Hardware path: FPGA inertia dynamics expected to reduce convergence sweeps 15-25x
  (arXiv 2604.17109 confirmed EMA speedup in silicon, not CPU float)
- Deliverable: results/experiment_701_kv260_v3_synth.json

### Phase 7: Retrospective

**Exp 702: Milestone 2026.04.53 Operational Retrospective**
- Load all 12 Exp result files (692-701 + 690-691 pinned)
- Compute wall time, per-experiment average, slowest-5
- Key metrics: jepa_v16_ood_auc, vr_cross_model_confirmed, psv_fp_trend_slope,
  prompt_injection_auroc, retro_072_status, manifest_consulted
- Check GPU state: both RTX 3090s should close < 100MB VRAM
- Deliverable: results/operational_retro_2026_04_53.json

---

## Dependency Graph

```
Exp 692 (pre-flight) ──────────────────────────────────────┐
Exp 693 (JEPA v16 design) ────────────────────────────────┤
         ↓                                                  │
Exp 690 (prompt-injection distill) [USER-PINNED]           │
         ↓                                                  │
Exp 691 (cross-dataset) [USER-PINNED, gates on 690]        │
                                                            │
Exp 694 (VR cross-model + hard, GPU) ──────────────────────┤
         ↓                                                  │
Exp 695 (FormalStepVerifier Tier 2.8)                      │
Exp 696 (I-CALM abstention)                                │
                                                            │
Exp 697 (PSV real self-play, GPU, gates on Exp 683) ───────┤
         ↓                                                  │
Exp 698 (JEPA v16 retrain, gates on Exp 693 design) ───────┤
         ↓                                                  │
Exp 699 (HalluSAE integration, gates on Exp 698) ──────────┤
                                                            │
Exp 700 (publication, gates on Exp 694) ───────────────────┤
Exp 701 (KV260 v3 synthesis)                               │
         ↓                                                  │
Exp 702 (retrospective — reads ALL above) ─────────────────┘
```

Critical paths:
- JEPA recovery: 693 → 698 → 699 (architecture audit must precede retrain)
- VR publication: 694 → 700 (cross-model must precede full publication)
- Self-learning: 697 → 698 (PSV data feeds JEPA v16 training)

---

## Success Criteria Table

| Criterion | Experiment | Target | Honest Verdict |
|-----------|-----------|--------|----------------|
| Slowest-5 reduced | Exp 692 | Exps 425/410 retired | `retirements_complete` |
| JEPA v15 root cause identified | Exp 693 | Root cause named | `root_cause_confirmed` |
| Prompt-injection real distillation | Exp 690 | teacher_duration_s > 0.5×N | not `distillation_invariant_violated` |
| Prompt-injection cross-dataset | Exp 691 | mean AUROC >= 0.80 | `generalization_verified_publishable` |
| VR cross-model confirmed | Exp 694 | Both models signed_improvement > 0 | `vr_cross_model_confirmed` |
| FormalStepVerifier viable | Exp 695 | AUC > SymCodeVerifier on FOVER | `formal_step_better` |
| I-CALM FP reduction | Exp 696 | fp_rate_delta < 0 | `abstention_fp_reduced` |
| PSV real-data FP improving | Exp 697 | fp_rate_trend_slope < 0 | `psv_selfplay_fp_improving` |
| JEPA v16 OOD AUC | Exp 698 | >= 0.70 on GSM8K 500-699 | `jepa_v16_ood_target_met` |
| HalluSAE integration | Exp 699 | AUC >= JEPA v16 baseline | `hallusat_integration_positive` |
| VR publication ready | Exp 700 | Cross-model gate open | `publication_ready` |
| KV260 v3 synthesis | Exp 701 | bitfile or tcl generated | `synthesis_complete_or_script` |

---

## Open RETROs to Close in .53

| RETRO | Age | Root Cause | Fix in .53 |
|-------|-----|-----------|-----------|
| RETRO-072 | Carry | KV260 bitfile not configured | Exp 701 (synthesis attempt) |
| JEPA regression | New (.52) | OOD AUC=0.4751 below random | Exp 693 audit + Exp 698 v16 retrain |
| Exp 425 slowest | 16 milestones | Demo watchdog rerun | Exp 692 retirement |
| Exp 410 slowest | 13 milestones | BatchedInferenceRunner loop | Exp 692 retirement |
| PSV synthetic | .52 | gate timing missed | Exp 697 real live data |
| Prompt-injection | 6 consec. | No real teacher inference | Exp 690 invariant |

---

## Hardware Requirements

| Experiment | GPU Required | GPU Assignment | Notes |
|-----------|-------------|----------------|-------|
| Exp 692 | No | CPU | Operational |
| Exp 693 | No | CPU | Analysis + design |
| Exp 690 | Optional (recommended) | cuda:0 for llama.cpp | gpt-oss-safeguard-20b GGUF (11.6 GB) |
| Exp 691 | No | CPU | Classifier inference |
| Exp 694 | YES | cuda:0 Qwen3.5-0.8B, cuda:1 Gemma4-E4B-it | LongRunBenchmarkExecutor, 200q |
| Exp 695 | No | CPU | Z3 available |
| Exp 696 | No | CPU | Abstention calibration |
| Exp 697 | YES | DualGPU | K=2 parallel chains, 10 iterations |
| Exp 698 | No | CPU | JEPA training (JAX CPU) |
| Exp 699 | No | CPU | JEPA fine-tune |
| Exp 700 | No | CPU | Publication artifacts |
| Exp 701 | No | CPU | Vivado/TCL script |
| Exp 702 | No | CPU | Retrospective |

---

## New Papers to Incorporate

| Paper | Filed | Experiment |
|-------|-------|-----------|
| arXiv 2603.20327 — JEPA Latent Probing | research-references.md | Exp 693 (JEPA v16 design) |
| arXiv 2602.06737 — KAN Formal Verification | research-references.md | Exp 700 (publication) |
| arXiv 2601.05593 — PaCoRe Parallel Reasoning | research-references.md | Exp 697 (PSV K=2 chains) |
| arXiv 2603.29500 — Formal Step Intermediaries | research-references.md | Exp 695 (Tier 2.8) |
| arXiv 2602.01090 — Hard Constraint Decoding | research-references.md | Exp 694 (grammar masking) |
| arXiv 2512.20664 — Eidoku CSP Gate | research-references.md (from .52) | Exp 695 (Tier 2.8) |
| arXiv 2604.03904 — I-CALM Abstention | research-references.md (from .52) | Exp 696 |
