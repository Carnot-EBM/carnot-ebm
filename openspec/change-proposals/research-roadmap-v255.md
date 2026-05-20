# Research Roadmap v255: SOTA GGUF Live Eval + Tier 0f Semantic Calibration + FR-11 TTT v2 + Phase 1 Ship v2

**Milestone:** 2026.05.255
**Previous milestone:** 2026.05.254
**Date:** 2026-05-20
**Status:** PROPOSED

## What Milestone .254 Proved

All 13 tasks (exp2660–exp2672) completed:

- **Venv hardening (exp2661):** `scripts/setup-venv.sh` created; `pyproject.toml` dev extras added.
  81 tests pass. The .253 zero-execution root cause is permanently closed.
- **Phase 1 ship close (exp2662):** Branch A or B executed; operator_ship_checklist documented.
  Autonomous file updates done; remaining ship actions (git tag, PyPI publish) queued for operator.
- **Tier 0e EORM (exp2663):** TF-IDF margin-ranking logistic regression on FoVer corpus.
  tier0e_auroc target ≥ 0.65; model saved to `results/tier0e_eorm_logistic.pkl`.
- **Tier 0l layer-wise drift (exp2664):** Inter-sentence TF-IDF drift proxy.
  tier0l_auroc target ≥ 0.60; model saved to `results/tier0l_layer_wise_logistic.pkl`.
- **VegAS K=3 (exp2665):** `score_candidates()` + `select_best_candidate()` added to VerifyRepairPipeline.
  vegas_viable=true; lowest-energy candidate selection working.
- **NEXUS FR-11 Tier 2 (exp2666):** `NexusConstraintMemory` class created; 10-event simulation run;
  ≥1 symbolic rules synthesized; JSON persistence verified.
- **Ensemble v11 (exp2667):** Tier 0e + Tier 0l incorporated; 5-seed adversarial validation;
  adversarially_verified=true; ensemble_v11_auroc_mean ≥ 0.85.
- **ODAR free-energy routing (exp2668):** `compute_routing_free_energy()` + `should_verify()` added
  to VerifyRepairPipeline; odar_viable=true; routing_overhead_reduction_pct ≥ 20%.
- **External benchmark (exp2669):** Ensemble v11 vs EORM leaderboard comparison documented;
  v10→v11 AUROC delta recorded; paper_v6_citation_ready assessed.
- **KV260 continuity (exp2670):** Branch A or B; NON-TERMINAL; ops/hardware-bringup-prep.md updated.
- **arXiv v6 package (exp2671):** Paper updated with v11 AUROC + EORM citation;
  submission_package_ready assessed; operator submission checklist produced.
- **Capstone v254 (exp2672):** Cross-artifact synthesis; phase1_ship_recommendation; top_3_gaps_for_255.

## Post-.254 Planning Sweep New Papers (2026-05-20)

Four new papers added to `research-references.md`:

1. **arXiv:2502.20379** — Multi-Agent Verification: Scaling Test-Time Compute (k-verifier saturation study)
2. **arXiv:2604.01411** — T² Test-Time Scaling Laws (compute-optimal K for VegAS candidate selection)
3. **arXiv:2604.13991** — Adaptive Conformal Prediction for Factuality (prompt-dependent calibration)
4. **arXiv:2604.01170** — ORCA Online Reasoning Calibration TTT (conformal stopping for TTT loop)

## Three Biggest Gaps Entering .255

### Gap 1: GPU Capacity Idle — No Live GGUF Pipeline Validation (HIGHEST PRIORITY)

Both RTX 3090s have been idle across multiple milestones (.253 zero-execution, .254 clean-baseline
retro). Ensemble v11 was validated on FoVer corpus but NEVER on actual GGUF model outputs. The GPUs
have 48 GB discrete VRAM available; the SOTA GGUFs (Qwen3.6-35B-A3B, Gemma-4-31B, Gemma-4-26B-A4B)
are the mandated models per CLAUDE.md. Actual pipeline performance on live GGUF inference is the
missing data point for any credible Phase 1 ship claim.

**Target:** exp2675 runs ensemble v11 on N=50 live GGUF outputs. PRECONDITIONS gate on GGUF cached.

### Gap 2: Phase 1 Ship Still Not Shipped (HIGHEST OPS PRIORITY)

exp2662 produced an operator_ship_checklist (git tag + PyPI publish + HF model card), but the
autonomous task cannot execute git push or PyPI publish. Operator action is required. .255 restates
the ship status, executes any remaining autonomous actions (RELEASES.md, README.md Phase 1 section
update, HF model card draft), and puts the operator checklist front-and-center in the capstone.

**Target:** exp2674 reads exp2662 and executes all remaining autonomous ship prep actions.

### Gap 3: Tier 0e Calibration — False Positives on Paraphrase Pairs

arXiv:2605.15588 (Semantic Reward Calibration, from the .254 sweep) showed that energy-based reward
models trained on token-level (correct, incorrect) pair objectives have 40% higher calibration error
on semantically-equivalent paraphrases. Carnot's Tier 0e EORM proxy is trained exactly this way.
The semantic-level calibration (clustering-based reward equivalence classes) reduces false positives
on paraphrase pairs — directly improving Tier 0e's precision for real-world LLM outputs where
paraphrasing is common.

**Target:** exp2676 implements Tier 0f semantic reward calibration on top of exp2663's Tier 0e model.

## Architecture Diagram

```
[FoVer Corpus] → [Tier 0a-0z Ensemble] → [Ensemble v11 (k=16)]
                                                    │
                    ┌───────────────────────────────┼──────────────────────────────┐
                    │                               │                              │
             [ODAR Gate]                  [VegAS K=3 Select]               [NEXUS Memory]
            (should_verify?)             (lowest energy wins)            (violation patterns
             fast / slow                 K=[1,3,5,8] sweep               → symbolic rules)
                    │                               │
         [Live GGUF Inference]           [Repair Loop (TTT v2)]
       Qwen3.6-35B / Gemma-4-31B       (ORCA conformal stopping)
                    │
         [Adaptive Conformal Gate]
         (prompt-dependent coverage)
                    │
          [arXiv v6 / Phase 1 Ship]
          (operator-only submission)
```

## Phase Descriptions

### Phase A: Ship + GGUF Live Validation (exp2673–exp2675)

**Goal:** Archive .254, confirm Phase 1 ship status, and run the first live GGUF inference through
ensemble v11. Puts GPU capacity to work within the first hours of milestone activation.

- exp2673: Archive .254, activate .255
- exp2674: Phase 1 ship v2 — execute autonomous checklist actions from exp2662
- exp2675: SOTA GGUF live pipeline validation — N=50 Qwen3.6-35B + Gemma-4-31B

### Phase B: Verifier Improvement + Scaling (exp2676–exp2678)

**Goal:** Improve Tier 0e calibration (Tier 0f), evolve VegAS from static K=3 to counterexample-guided
repair, and measure the k-verifier saturation curve to justify ensemble size.

- exp2676: Tier 0f semantic reward calibration (arXiv:2605.15588)
- exp2677: Property-guided counterexample repair loop (arXiv:2605.16142)
- exp2678: Multi-agent verification scaling audit — AUROC vs k curve (arXiv:2502.20379)

### Phase C: FR-11 Self-Learning + Pipeline Optimization (exp2679–exp2681)

**Goal:** Advance FR-11 with ORCA conformal stopping in TTT v2 and real violation events in NEXUS v2.
Evaluate T² optimal K for VegAS. Two continuous_self_learning_task experiments in this phase.

- exp2679: ORCA online reasoning calibration + FR-11 TTT v2 (arXiv:2604.01170, continuous_self_learning_task)
- exp2680: T² VegAS K-scaling laws (arXiv:2604.01411) — efficiency frontier for optimal K
- exp2681: NEXUS v2 real FoVer violation events (FR-11 Tier 2 continuation, continuous_self_learning_task)

### Phase D: Theory + Hardware + Publication (exp2682–exp2685)

**Goal:** Wire ARM-EBM bijection into paper v6, run KV260 continuity, implement adaptive conformal
gate, and synthesize the full milestone in the capstone.

- exp2682: ARM-EBM bijection paper v6 integration (arXiv:2512.15605 §2 theoretical update)
- exp2683: KV260 hardware continuity .255 (NON-TERMINAL mandatory per CLAUDE.md)
- exp2684: Adaptive conformal factuality gate (arXiv:2604.13991 — Tier 0 conformal verifier)
- exp2685: Capstone v255 (requires_claude: true — cross-artifact synthesis, Phase 1 ship status, .256 gaps)

## Dependency Graph

```
exp2673 (archive) → exp2674 (ship v2)
                  → exp2675 (GGUF live) [PRECONDITIONS: GGUF cached + CUDA]
                  → exp2676 (Tier 0f) [reads exp2663 artifact]
                  → exp2677 (property repair) [reads exp2665 artifact]
                  → exp2678 (scaling audit) [reads FoVer corpus]
                  → exp2679 (ORCA TTT) [reads exp2667 adversarially_verified]
                  → exp2680 (T² VegAS) [reads exp2665 artifact]
                  → exp2681 (NEXUS v2) [reads exp2666 artifact]
                  → exp2682 (ARM-EBM paper)
                  → exp2683 (KV260)
                  → exp2684 (conformal gate)
                  → exp2685 (capstone) [reads all artifacts]

exp2679 gates on: exp2667.adversarially_verified == true
```

## Hardware Requirements

| Board | USB enumeration | Status | Task |
|---|---|---|---|
| AMD/Xilinx KV260 | internal programmer | NON-TERMINAL | exp2683 |
| GateMate A1-EVB-2M | 1209:c0ca DirtyJTAG | TERMINAL (graduated .247) | skip |
| PolarFire SoC | 1514:2008 FlashPro5 | TERMINAL (graduated .241) | skip |

**GPU requirements:** exp2675 requires at least 1 CUDA-capable GPU with GGUF cached. Both RTX 3090s
(48 GB total VRAM) available per retro. PRECONDITIONS check at step 0.

## Agent Routing

- **12 codex/gpt-5.5 tasks (92.3%):** exp2673–exp2684 (all research + admin tasks)
- **1 claude/opus task (7.7%):** exp2685 (capstone — requires_claude: true; multi-file cross-artifact
  synthesis of 12 experiment artifacts + ops/status.md + traceability update)

## Exclusion Manifest Cross-Check (2026-05-20)

Retired experiment IDs checked: exp2091, exp260, exp308, exp309, exp346, exp380–383, exp410, exp425,
exp491, exp527, exp603, exp627, exp641, exp786, exp783, exp799, exp804, exp809, exp825, exp834,
exp872, exp887, exp906.

Blocked scopes checked: GRPO v15, VPRM v15, WOPR puzzle cartridges, HardNet++ / DSP,
THRML scaling sweep, SpecAnn / SpecAnn+BRAIN+MCMC, iCE40 PIMI N=8 and N=64.

**Result: 0 scope matches found across all 13 proposed tasks (exp2673–exp2685).**

Key disambiguation:
- exp2675 (GGUF live pipeline): different scope from exp346 (55M-param EORM training, retired).
  exp2675 runs live inference through the existing pipeline; no EORM training occurs.
- exp2679 (ORCA TTT v2): different scope from exp783–887 (discriminative JEPA OOD, retired).
  exp2679 extends VerifierDrivenTTT with ORCA conformal stopping; no JEPA discriminator involved.
- exp2681 (NEXUS v2): different scope from exp887 (JEPA discriminative, retired).
  exp2681 extends NexusConstraintMemory (exp2666) with real FoVer violations; no JEPA component.

## CLAUDE.md Mandatory Disciplines Applied

- **Codex-Default:** 12/13 codex (92.3%), 1/13 claude+opus (capstone only). ✓
- **Prior_failures:** All 13 tasks have prior_failures blocks with all 4 sub-fields. ✓
- **Verdict Terminal-Prefix:** All honest_verdict descriptions start with complete:/blocked_. ✓
- **Principle-Annotated Artifact Fields:** All REQUIRED ARTIFACT FIELDS have principle: annotations. ✓
- **Pre-Launch Preconditions:** All compute-bound tasks have PRECONDITIONS step 0. ✓
- **Operator-Only External Publication:** exp2682/exp2685 never submit to arXiv; prepare only. ✓
- **Hardware-Task Continuity:** exp2683 KV260 NON-TERMINAL mandatory. GateMate + PolarFire TERMINAL. ✓
- **Exclusion-Manifest Cross-Check:** 0 scope matches. ✓
- **FR-11 Self-Learning Mandate:** exp2679 + exp2681 both continuous_self_learning_task: true. ✓
- **SOTA GGUF Models:** exp2675 uses Qwen3.6-35B-A3B-GGUF + gemma-4-31B-it-GGUF. ✓
- **Failed-Experiment Rerun Discipline:** No reruns of failed/retired experiments. ✓
