# Research Roadmap — Milestone 2026.04.88

**Title:** arXiv Submission + AND-Compose k=5 Fix + GRPO Full Training + Cascade Calibration + Verifier Robustness

**CalVer:** 2026.04.88 (sequence increment from 2026.04.87)
**Planned Experiments:** Exps 1127–1138 (12 experiments)
**Date Designed:** 2026-05-02
**Prerequisite:** Milestone 2026.04.87 retro complete (Exp 1126)

---

## What Milestone 2026.04.87 Proved

Milestone .87 met all 11 success criteria (first perfect run in project history). Key results:

**Wins:**
- **Energy inversion FIXED:** Mean correct energy 0.689→1.648, mean incorrect 0.621→2.096. AUROC=0.9774 post-retrain on 7329-pair corpus. Root cause confirmed: FoVer was trained on base-model outputs; SOTA RL-optimized outputs caused OOD distribution shift. Fix: extend corpus with SOTA outputs + EBRM noise filtering.
- **GRPO + ThinkPRM v2 first POSITIVE result:** +4pp (24%→28%) on 25-question holdout, breaking 3-consecutive RLVR+SSD negative streak. DualGPU used. Training wall budget hit at 240s (only 42/50 questions completed).
- **All 4 infrastructure bottlenecks deployed (exp1117):** dispatch manifest YAML structural bug fixed, CARNOT_BATCH_DOC_RECONCILE defaulted to 1, grace_period_s schema added, CARNOT_FAST_EVAL flag added. Estimated 111 min/milestone savings going forward.
- **WOPR Hashi cartridge:** E=0 at convergence. Gallery updated.

**Critical findings for .88:**
- **k=5 AND-compose AUROC=0.5547 < individual best SemEnergyProbe AUROC=0.8964.** SOSKANEnergyV3 individual AUROC=0.333 (below chance). The ensemble is WORSE than the best individual verifier due to SOSKANEnergyV3 adding anti-correlated noise. Root cause and fix are mandatory before k=5 is used as the production default.
- **GRPO training wall budget hit:** 240s budget was too tight; only 42/50 training questions processed. Need 600s budget and 100 training questions for a full proof-of-concept.
- **Lagrangian cascade router accuracy degraded 22.9pp** vs fixed cascade (TP 0.743 vs 0.971). The MLP router lacks verifier-score features; accuracy/cost balance needs retuning.
- **arXiv manual upload still required:** pdflatex/tectonic absent from conductor environment; 2026-05-15 deadline is 13 days away. **HIGHEST URGENCY.**

---

## Architecture Diagram

```
                    Carnot Verification Cascade (.88 target state)
                    ═══════════════════════════════════════════════

User Query
    │
    ▼
Tier 0a: ThinkPRM v2 ──────────────────── AUROC=0.9946 (step-level)
    │ verdict=uncertain
    ▼
Tier 0b: SpilledEnergyDetector ─────────── logit-discrepancy (fast)
    │ high_spill=False
    ▼
Tier 0c: SemEnergyProbe ────────────────── AUROC=0.8964 (0.017ms)  ← PRIMARY
    │ score≤threshold
    ▼
Tier 3: k=5 AND-compose [NEEDS FIX] ─────  AUROC=0.5547 → target >0.80
    │   SOSKANEnergyV3*  0.333 → FIX
    │   SemEnergyProbe   0.896
    │   ASTStructure     0.556
    │   SemConsistency   0.528
    │   Z3MathVerifier   0.691
    │ (*SOSKANEnergyV3 is the bottleneck; fix or replace)
    ▼
Repair: GRPO-energy-PRM guided (target: full training >32% accuracy)
    │
    ▼
Certificate + Lagrangian Router v2 (target accuracy_delta > -5pp)
```

---

## Phase Descriptions

### Phase 0 — arXiv Submission (CRITICAL, unconditional, run first)

**exp1127: arXiv PDF Compilation + Final Submission**

The arXiv bundle (121KB tar.gz) was built in exp1116 with author identity filled. The only
remaining blockers: pdflatex/tectonic not installed and no browser session. This experiment
tries pip install tectonic as a self-contained LaTeX engine (no system packages needed), then
compiles the PDF, and produces an exact checklist for the manual arXiv submission. Deadline:
2026-05-15. Every day of delay reduces review buffer.

Acceptance: arxiv_submitted=True OR (pdf_compiled=True AND manual_steps_complete_enough_to_submit_today).

### Phase 1 — AND-Compose k=5 Fix (MANDATORY)

**exp1128: SOSKANEnergyV3 Root Cause + k=5 AND-Compose Repair**

SOSKANEnergyV3 individual AUROC=0.333 means it is producing INVERTED scores — the model is
below chance, actively degrading the k=5 ensemble. Root causes to diagnose:
1. Energy score polarity inverted (higher energy = correct when it should = incorrect)
2. Model not converged / trained on inverted labels
3. SOS-KAN gradient normalization causing activation saturation

Fix plan: (a) inspect SOSKANEnergyV3 score output on known-correct vs known-incorrect examples;
(b) if polarity inverted, flip sign in output layer; (c) re-train with fresh label check;
(d) if unfixable, replace with DualSOSKANEnergyProbe using calibrated isotonic regression
wrapper. After fix, re-run k=5 benchmark on 500-example holdout. Target: k5 AUROC > 0.80.

Acceptance: sos_kan_root_cause_identified=True AND k5_ensemble_auroc_above_08=True.

### Phase 2 — GRPO Full Training (GPU, DualGPU MANDATORY)

**exp1129: GRPO Energy PRM Full Training v2**

exp1118 proved GRPO+ThinkPRM v2 is positive (+4pp) but hit the training wall at 240s. This
experiment runs the full training with: n_training=100 questions, budget_s=600, N=8 completions
per question, ThinkPRM v2 as continuous reward. Applies DRA-GRPO diversity penalty (arXiv
2505.09655) to prevent mode collapse on the 8-completion groups. Uses CPPO proxy reuse strategy
(arXiv 2503.22342) to reduce inference cost per group. Evaluates on 50 holdout questions.

Target: improvement_over_baseline > 0.05 (5pp improvement on 50-question holdout). DualGPU
MANDATORY — this will not run on single GPU in 600s budget.

Prior failures: exp1118 (training_wall_budget_hit=True, 240s too tight for 50 questions).

Acceptance: grpo_v2_honest_result=True AND improvement_over_baseline recorded.

### Phase 3 — Continuous Self-Learning (GPU)

**exp1130: Zenil α_t Measurement with Post-Retrain Verifier**

The verifier was retrained on 7329-pair SOTA corpus with AUROC=0.9774 (exp1120). The prior
α_t measurement (exp1112) used the pre-retrain verifier. Re-measure α_t = μ_P(E_verifier) with
the new verifier on 50 live SOTA model outputs. If α_t > 0.38 (prior) the retrained verifier
is grounding self-distillation better. Log as a FR-11 continuous self-learning data point.

Acceptance: zenil_alpha_t_post_retrain_measured=True.

### Phase 4 — Cascade and Router Calibration

**exp1131: Lagrangian Cascade v2 — Accuracy-Preserving Router**

exp1123 achieved 99.98% cascade cost savings but at -22.9pp accuracy (TP 0.743 vs 0.971 fixed).
The MLP was trained without verifier-score features — the router had no signal about correctness.
Fix: (a) extract SemEnergyProbe score + ThinkPRM confidence as input features; (b) increase
MLP hidden size 32→128; (c) add minimum-TP constraint to the Lagrangian dual (min TP rate ≥ 0.90
at any cascade depth). Target: accuracy_delta > -5pp vs fixed cascade while preserving >40% cost
savings.

Prior failures: exp1123 (accuracy_delta=-0.2286, no verifier-score features, MLP=32 hidden).

Acceptance: cascade_v2_accuracy_delta_above_neg05=True AND cost_savings_pct_positive=True.

### Phase 5 — Verifier Robustness Validation

**exp1132: Goodfire LLM Failure Exemplar Cascade TP Rate**

From known-issues.md: exp1112 built the LLM failure exemplar corpus (data/llm_failure_exemplars.jsonl,
≥30 named failure modes). Now feed those exemplars through the full Carnot verifier cascade and
measure per-tier TP rate. Key test: does Z3MathVerifier catch "9.11 > 9.9" (arithmetic error from
version-number interference)? Does SemEnergyProbe catch trolley-problem moral framing errors?
Results validate Carnot's engineering claim vs mechanistic interpretability tools.

Acceptance: goodfire_exemplar_tp_rate_measured=True AND per_tier_results_logged=True.

**exp1133: PRM-BiasBench Adversarial Test on k=5 Ensemble**

arXiv 2603.06621 released PRM-BiasBench: adversarial exemplars targeting stylistic shortcuts
in PRMs (43% of PRM reward attributable to formatting/padding, not reasoning quality). Test the
k=5 AND-compose ensemble against PRM-BiasBench-style stylistic attacks to measure how many are
caught vs missed. Tests whether AND-composition provides better null-space coverage than individual
verifiers against style-based gaming.

Acceptance: prm_biasbench_attack_tp_measured=True.

### Phase 6 — Hardware Path

**exp1134: KV260 v4 Beta/Alpha Self-Adaptive Parameter Tuning**

exp1122 Python simulation showed KL=0.134 at best (alpha=0.1, beta=2.0), above 0.05 threshold.
Retro suggested trying beta=3.0/4.0. This experiment: (a) implements self-adaptive λ update
from arXiv 2501.04971 (Lagrange Ising) to auto-tune penalty coefficients; (b) sweeps beta=2-5
at alpha=0.1; (c) alpha sweep 0.02-0.2 at best beta; (d) if KL still above 0.05, documents
the parameter space boundary and updates the RTL spec with empirically-derived feasibility limits
(e.g., "KL < 0.05 requires beta > 6.0, which exceeds 16-bit fixed-point range on XCZU5EV").

Prior failures: exp1122 (KL=0.134, above 0.05, best_alpha_ema=0.1 insufficient).

Acceptance: kv260_v4_kl_below_05=True OR kv260_v4_feasibility_limits_documented=True.

### Phase 7 — Position Paper + WOPR Gallery

**exp1135: Position Paper v3 — Integrate .87/.88 Experimental Findings**

Position paper main.tex needs the .87 experimental results integrated: energy inversion fix
(AUROC 0.977), GRPO first positive (+4pp), k=5 AND-compose deployment, cascade cost savings.
Also: add Related Work comparison to HIVE (arXiv 2604.26139) as a complementary hallucination
detection system. Target: main.tex ready to re-compile with new experimental section.

Acceptance: position_paper_findings_updated=True.

**exp1136: WOPR Slitherlink Puzzle Cartridge**

Next WOPR game: Slitherlink (Nikoli/Nurikabe loop puzzle). Each cell has a clue digit 0-3
encoding how many of its 4 edges are part of the loop. Ising spins encode edge membership.
Energy = (violated clue penalties) + (loop connectivity penalty). E=0 iff exactly one closed
loop visits all constrained edges. agent_type: codex (formulaic graph constraint encoding).

Acceptance: slitherlink_cartridge_shipped=True AND canonical_e_at_convergence==0.0.

**exp1137: HF Spaces Gallery Update**

Deploy Slitherlink cartridge to HF Spaces gallery. Gated on exp1136.slitherlink_cartridge_shipped.

Acceptance: gallery_updated=True.

### Phase 8 — Retrospective

**exp1138: Milestone 2026.04.88 Retrospective**

Standard operational retrospective. Measure: criteria met/total, wall time vs .87 (891 min),
slowest-5 composition (does exp906 FINALLY absent? exp1117 fixed the YAML structural bug).
Document whether the 111 min/milestone infrastructure savings from exp1117 are observable in .88.

---

## Dependency Graph

```
exp1127 (arXiv CRITICAL) ─ unconditional
exp1128 (AND-compose fix) ─ unconditional
exp1129 (GRPO v2, GPU) ─── unconditional, DualGPU MANDATORY
exp1130 (Zenil α_t) ─────── unconditional, GPU
exp1131 (cascade v2) ─────── unconditional (uses FoVer corpus)
exp1132 (Goodfire TP) ─────── unconditional (uses exp1112 exemplar corpus)
exp1133 (PRM-BiasBench) ─── gated on exp1128.k5_ensemble_auroc_above_08 (needs fixed ensemble)
exp1134 (KV260 v4 tuning) ── unconditional
exp1135 (position paper) ─── gated on exp1129.grpo_v2_honest_result AND exp1130.zenil_alpha_t_post_retrain_measured
exp1136 (Slitherlink) ─────── unconditional (codex)
exp1137 (gallery) ─────────── gated on exp1136.slitherlink_cartridge_shipped
exp1138 (retro) ──────────── last
```

---

## Hardware Requirements

- **DualGPU (2x RTX 3090 CUDA):** exp1129 (GRPO v2) — MANDATORY. exp1130 (Zenil α_t) — preferred.
- **CPU only:** exp1127, 1128, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138.
- **KV260 FPGA (192.168.51.98):** exp1134 — board is reachable; hardware sampling attempted only if Python simulation resolves KL.
- **Vivado:** Still not installed. exp1134 stays Python simulation only.

---

## 11 Success Criteria

1. **arxiv_submitted_or_pdf_compiled** (exp1127) — CRITICAL. 2026-05-15 deadline.
2. **sos_kan_root_cause_identified** (exp1128) — inverted AUROC diagnosed and fixed.
3. **k5_ensemble_auroc_above_08** (exp1128) — ensemble better than individual best.
4. **grpo_v2_honest_result** (exp1129) — full training run, improvement recorded.
5. **zenil_alpha_t_post_retrain_measured** (exp1130) — FR-11 self-learning data point.
6. **cascade_v2_accuracy_delta_above_neg05** (exp1131) — cost savings without accuracy collapse.
7. **goodfire_exemplar_tp_rate_measured** (exp1132) — per-tier TP on named failure modes.
8. **prm_biasbench_adversarial_tp_measured** (exp1133) — stylistic attack resistance measured.
9. **kv260_v4_kl_below_05_or_feasibility_documented** (exp1134) — KV260 v4 path resolved.
10. **position_paper_v3_findings_integrated** (exp1135) — arXiv main.tex ready to recompile.
11. **retro_complete** (exp1138).

---

## Key Architectural Decisions for .88

- **No gemini agent_type** (429-rate-limited since .84).
- **Codex for WOPR Slitherlink** (formulaic graph constraint encoding — codex excels at these).
- **DualGPU MANDATORY for exp1129** — hard constraint in prompt (failure mode from exp1118).
- **SOSKANEnergyV3 fix BEFORE PRM-BiasBench test** — exp1133 gated on exp1128 fixing the ensemble; testing a broken k=5 against adversarial exemplars would produce invalid baseline.
- **exp1127 runs unconditionally FIRST** — arXiv deadline risk increases every day.
- **grace_period_s: 2400 for exp1129** — GRPO training at 600s + inference at 160s = 760s minimum; add margin.
- **No manifest for exp906 expected** — exp1117 fixed the YAML structural bug; if exp906 appears in .88 slowest-5, that is a new regression requiring investigation.

---

## New arxiv Findings Incorporated

| Paper | arXiv ID | Incorporated in |
|-------|----------|-----------------|
| DRA-GRPO: Diverse Reasoning Paths | 2505.09655 | exp1129 diversity penalty |
| CPPO: 3.48x GRPO Acceleration | 2503.22342 | exp1129 proxy reuse |
| Why Self-Distillation Degrades | 2603.24472 | position paper §4 |
| Continuous Ising via DC Programming | 2509.01928 | future exp (milestone .89+) |
| Self-Adaptive Ising Machines | 2501.04971 | exp1134 self-adaptive λ |
| GRPO + Reflection Reward | 2603.14041 | future exp (milestone .89) |
| HIVE Hallucination Verification | 2604.26139 | position paper Related Work |

---

## Estimated Wall Time

| Experiment | Model | GPU | Est. Min |
|-----------|-------|-----|---------|
| exp1127 arXiv | opus | no | 20 |
| exp1128 AND-compose fix | sonnet | no | 30 |
| exp1129 GRPO v2 | opus | DualGPU | 50 |
| exp1130 Zenil α_t | sonnet | GPU | 30 |
| exp1131 Cascade v2 | sonnet | no | 25 |
| exp1132 Goodfire TP | sonnet | no | 25 |
| exp1133 PRM-BiasBench | sonnet | no | 25 |
| exp1134 KV260 v4 | opus | no | 35 |
| exp1135 Position paper | sonnet | no | 25 |
| exp1136 Slitherlink | codex | no | 20 |
| exp1137 Gallery | sonnet | no | 15 |
| exp1138 Retro | sonnet | no | 15 |
| **Total** | | | **~315 min** |

Total estimated wall time: ~315 min (well below .87's 891 min — these experiments are focused
and directly follow from clear .87 findings, avoiding long exploratory GPU runs except where
mandatory).
