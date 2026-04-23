# Research Roadmap — Milestone 2026.04.60

**Title:** JEPA v20 Data Surge + SOTA GGUF Confirmed + Constraint Memory to Constraint Generation

**CalVer:** 2026.04.60 (sequence increment from 2026.04.59)
**Planned Experiments:** Exps 780-792 (13 experiments)
**Date Designed:** 2026-04-23
**Prerequisite:** Milestone 2026.04.59 retro complete (Exp 779)

---

## What Milestone 2026.04.59 Proved

Milestone .59 (Exps 767-779) achieved significant governance and efficiency wins:
- Manifest enforcement extended to ALL dequeue sites (Exp 767) — Exp 425 absent for first time since .37
- EBRM comparison (Exp 771) confirmed EORM (AUC=0.993) outperforms EBRM (AUC=0.943): step-level granularity is correct
- Carnot uses 6x fewer oracle calls than SETS (Exp 773): oracle_call_ratio=6.0
- Adaptive PSV sampling achieves 75% sample reduction with +0.013 AUC improvement (Exp 774)
- JailbreakKAN Tier 0h deployed with AUC=1.0, precision=1.0 (Exp 775)
- Tier 3.5 gate governance confirmed working (Exp 778): blocked correctly when OOD AUC below gate

**Four open retros carry forward:**
- RETRO-028: Gemma4 CUDA OOM — 14.89 GiB allocation fails with ~15 GiB already occupied; GPU not cleared before load
- RETRO-JEPA-OOD-V19: ood_auc=0.5667 < 0.75 gate — 57 pairs insufficient for OOD generalization
- RETRO-SOTA-GGUF-TIMEOUT: Exp 769 timed out at 120 min — model load + 50 problems too large for 120 min
- RETRO-HF-AUTH: HuggingFace authentication unavailable in conductor environment

---

## The 3 Biggest Gaps vs PRD Vision

### Gap 1: Tier 3 Self-Learning Not Deployed (FR-11 partial)

JEPA v19 OOD AUC=0.5667 blocks Tier 3.5 cascade deployment (FR-11). Root cause: 57 training
pairs from Exp 442 are predominantly Qwen3.5-0.8B arithmetic errors on GSM8K questions 1-300 —
insufficient diversity to generalize to unseen question distributions. Fix requires two things:
(1) More real data: 100+ new CoT steps from a second live benchmark covering different question types
(2) Smarter selection: EDU-PRM (arxiv 2503.22233) identifies highest-uncertainty steps for training,
    focusing the model on discriminative examples rather than easy clear-cut correct/incorrect pairs

### Gap 2: No Headline Code Repair Number (SOTA GGUF blocked)

The research-program.md's strongest credible result is code repair via execution verification.
Exp 769 timed out because Qwen3.6-35B-A3B Q4_K_M requires ~20 GiB VRAM and the system had
~15 GiB already occupied (GPU zombie). With GPU cleared (Exp 780 fix), the 35B model loads.
Additionally, scoping to 25 problems with a 90-min budget (not 50 problems at 120 min) gives
each problem a 3.6-min budget including model load — achievable.

### Gap 3: Self-Learning Architecture Ceiling (Tier 1-2)

research-program.md highest priority: constraint ADDITION from memory patterns. The session
memory (Tier 2) accumulates error patterns across queries. But the pipeline only uses this data
for weight reweighting (proven ineffective in Exp 134). The correct mechanism is to read the
memory and GENERATE new IsingEBM coupling rows: "arithmetic carry errors are common in step 2"
→ add coupling J[carry_bit, result_bit] to the active constraint set. This upgrades the system
from a fixed-topology EBM to a memory-guided growing EBM — the key step toward FR-11 Tier 2.

---

## Architecture Diagram

```
Query
  |
  v
[Tier 0a] CarnotThinkProbe (generative CoT verdict)
  |  fast-path on "incorrect" verdict
  v
[Tier 0b] SpilledEnergyDetector (logit-discrepancy, arXiv 2602.18671)
  |
  v
[Tier 0c] NUP Probe v4 (contrastive energy, AUC=1.0, Exp 523)
  |
  v
[Tier 0d] HallucinationBasinDetector (latent basin depth, arXiv 2604.04743)
  |
  v
[Tier 0e] HalluField (thermodynamic instability, advisory, arXiv 2509.10753)
  |
  v
[Tier 0h] JailbreakDetectionKAN (safety gate, AUC=1.0) [.59 DEPLOYED]
  |  returns SAFETY_GATE if jailbreak detected
  v
[Tier 1]  SinkProbe (attention sink concentration, arXiv 2604.10697)
  |
  v
[Tier 2]  EORM (CoT energy reward model, 55M params)
  |
  v
[Tier 2.5] SymCodeVerifier (executable arithmetic, AUC=0.804 live)
  |
  v
[Tier 2.6] HermesVerifierAdapter (step-boundary feedback loop)
  |
  v
[Tier 2.7] CausalReasoningVerifier (causal entailment across steps)
  |
  v
[Tier 3]  IsingEBM (full constraint verification, 0.006 ms/check)
  |
  v
[Tier 3.5] JEPA v20 Predictor [TARGET for .60] ← blocks Tier 3 for low-risk queries
  |
  v
[VerifyRepairPipeline]
  |
  v
[ConstraintGenerator] [NEW in .60] ← memory patterns → new IsingEBM coupling rows
  |
  v
Result + Certificate
```

---

## Phase Descriptions

### Phase 0: Infrastructure + GPU Hygiene (Exps 780-781, run first)

**Exp 780: Pre-flight v12 — GPU Zombie Killer + Kill-Before-Load (MANDATORY FIRST)**

The root cause of RETRO-028 and RETRO-SOTA-GGUF-TIMEOUT is the same: GPU memory occupied
by prior processes prevents model load. The fix: every experiment that loads a GPU model MUST
call `kill_gpu_zombies()` before any model load attempt. This parallels apply_env_autofix()
(RETRO-022) — a mandatory systemic fix applied at ExperimentTemplate level.

This experiment implements `kill_gpu_zombies()` in ExperimentTemplate, updates the pre-flight
manifest, and validates that Gemma4 loads cleanly after killing zombies.

**Exp 781: JEPA v20 Live Data Collection — 100q Qwen3.5-0.8B GSM8K (GPU REQUIRED)**

Run 100 GSM8K questions through the live verify-repair pipeline using Qwen3.5-0.8B (known to
work). Extract all CoT steps, apply FOVER annotation (Z3 step labeling), write to
results/fover_labeled_steps_live_v2.json. Target: n_labeled >= 80 new real pairs, tripling
the training corpus from 57 to 137+ pairs. This directly unblocks JEPA v20.

### Phase 1: Tier 3 Self-Learning — JEPA v20 (Exps 782-784)

**Exp 782: EDU-PRM Uncertainty Step Selection (CPU)**

Implement EDUPRMStepSelector (arxiv 2503.22233): for each step in the pooled FoVer corpus,
compute model prediction variance across bootstrap samples. Select the top 30% highest-
variance (most uncertain) steps for JEPA v20 training. The intuition: clear-cut correct
and incorrect steps add little information; the hard borderline cases teach the most.

**Exp 783: JEPA v20 Retrain — Pooled Real Data + EDU-PRM Selection (CPU)**

Train MultiStepJEPAv20 on pooled data: Exp 442 (57 pairs) + Exp 770 real pairs + Exp 781
(80+ new pairs). Apply EDU-PRM selection from Exp 782 to weight training toward uncertain steps.
Target: OOD AUC > 0.75 (unblocks Tier 3.5 deployment). OOD test: GSM8K 800-999, same as Exp 770.

**Exp 784: JEPA v20 Cascade Deploy (GATED on Exp 783 OOD AUC > 0.75)**

Wire JEPA v20 as Tier 3.5 in ThreeTierPipeline. Same deployment logic as Exp 778, but with
higher-quality predictor. If successful: FR-11 Tier 3 self-learning loop closes.

### Phase 2: SOTA GGUF + Code Ranking (Exps 785-787)

**Exp 785: SOTA GGUF Code Repair v2 — 25 Problems, 90-min Budget (GPU REQUIRED)**

Retry Exp 769 with proper scoping. GPU cleared by Exp 780 fix. Load Qwen3.6-35B-A3B Q4_K_M
(~20 GiB VRAM). If not enough VRAM: fallback to Qwen3.5-7B-Instruct Q4_K_M (~4 GiB VRAM).
Run 25 HumanEval problems, 2-round repair. Per-problem timeout: 3 min. Checkpoint every 5.
Target: signed_improvement > 0 on live GPU, resolving RETRO-SOTA-GGUF-TIMEOUT.

**Exp 786: Gemma4 OOM Fix v3 + VR Threshold Grid (GPU REQUIRED, RETRO-028)**

With kill_gpu_zombies() from Exp 780 in place, retry loading Gemma4-E4B-it. Run the
5-threshold VR grid (thresholds [0.10, 0.20, 0.30, 0.40, 0.50]) on 50 GSM8K questions.
Target: loader_test_passed=True (RETRO-028 CLOSED) AND positive_threshold_found=True.

**Exp 787: S* Energy-Ranked Code Selection (CPU, arxiv 2502.14382)**

Implement S*-style energy-ranked candidate selection. For each HumanEval problem, generate
N=4 code candidates, compute Carnot energy for each, select the lowest-energy candidate
before running execution tests. Compare: n_tests_run vs pure execution selection, pass@1
(does energy pre-ranking improve? reduce? tie?). Report energy_prefilter_saves_tests fraction.

### Phase 3: Constraint Memory to Constraint Generation (Exps 788-789)

**Exp 788: Constraint Addition from Memory — Tier 2 Upgrade (CPU)**

Implement ConstraintGenerator that reads session memory error patterns and synthesizes new
IsingEBM coupling rows. Algorithm:
1. After each verify-repair cycle, session memory records (violation_type, step_pattern) pairs.
2. ConstraintGenerator.synthesize(memory) → new IsingEBM couplings for detected pattern types.
3. Run 50 GSM8K questions with dynamic constraint addition (adaptive) vs fixed constraints (baseline).
4. Compare: net_improvement (adaptive) vs net_improvement (baseline).
This implements the "constraint addition" fix from research-program.md #1 priority, replacing
the proven-ineffective precision-based reweighting (Exp 134).

**Exp 789: EBM Calibration Alignment (CPU, arxiv 2603.06604 + 2602.11364)**

Measure whether Carnot energy is calibrated to correctness probability. On 200 GSM8K questions
with Qwen3.5-0.8B live data: bucket by Carnot energy decile, plot P(correct) per bucket.
Compute Expected Calibration Error (ECE). Apply isotonic regression to learn energy → P(correct)
mapping. This tells us whether energy is a reliable probability signal or just discriminative.
Also implement diffusion-style reconstruction energy (arxiv 2602.11364) as a second signal,
compare which is better calibrated and whether their fusion improves ECE.

### Phase 4: Hardware + Publishing + Retro (Exps 790-792)

**Exp 790: NPU Unblock v9 — GitHub Releases mlir-aie Wheel (CPU/NPU)**

Previous 8 attempts blocked by same issue. Option A (RETRO-NPU-v8): download mlir-aie wheel
from AMD GitHub Releases directly (not PyPI). Option B: download Ryzen AI Software installer.
Binary verdict: NPU GEMM runs or doesn't. If neither option works: escalate with detailed
human-action instructions and close the loop on automated attempts for this milestone.

**Exp 791: KV260 N=32 Reduced Design Synthesis (CPU, open-source flow)**

N=128 caused 2.48x LUT overflow in full Vivado run (.59 session). N=64 had 48.5% LUT / 87%
DSPs per ops/status.md. N=32 spins with MAX_DEGREE=8 uses ~600-800 LUTs — fits iCE40 HX8K
(7680 LUTs) with 90%+ headroom. Run: Yosys synth_ice40 + nextpnr-ice40 (available from .59
Exp 776). Target: pnr_success_ice40=True AND bitstream_generated=True. First actual bitstream.

**Exp 792: Milestone 2026.04.60 Operational Retrospective**

Standard retro reading all Exps 780-791 results, computing milestone metrics, closing/opening
RETROs, updating ops/status.md and ops/changelog.md.

---

## Dependency Graph

```
Exp 780 (GPU zombie fix) → Exp 781 (live 100q, GPU)
                                 |
                         Exp 782 (EDU-PRM selection, CPU)
                                 |         |
                                 └────┬────┘
                                      v
                               Exp 783 (JEPA v20 retrain)
                                      |
                               Exp 784 (Cascade deploy, GATED)

Exp 780 → Exp 785 (SOTA GGUF code repair v2, GPU)
Exp 780 → Exp 786 (Gemma4 OOM fix v3, GPU)

Exp 787 (S* energy ranking, CPU) — standalone
Exp 788 (Constraint generation, CPU) — standalone
Exp 789 (EBM calibration, CPU) — standalone
Exp 790 (NPU unblock v9, CPU/NPU) — standalone
Exp 791 (KV260 N=32 synthesis, CPU) — standalone

All → Exp 792 (Milestone retro)
```

---

## Hardware Requirements

| Experiment | GPU | Notes |
|-----------|-----|-------|
| Exp 780 | Optional (RTX 3090) | GPU zombie kill implementation + validation |
| Exp 781 | RTX 3090 #0 (24GB) | 100q live Qwen3.5-0.8B; kill_gpu_zombies first |
| Exp 782-784 | None | CPU-only JEPA |
| Exp 785 | RTX 3090 #0 (24GB) | SOTA GGUF; kill_gpu_zombies first |
| Exp 786 | RTX 3090 #0 (24GB) | Gemma4; kill_gpu_zombies first |
| Exp 787-789 | None | CPU-only |
| Exp 790 | AMD XDNA NPU | Attempt Option A: GitHub releases wheel |
| Exp 791 | None | iCE40 FPGA synthesis via nextpnr |
| Exp 792 | None | Retro |

---

## Success Criteria

| Criterion | Target | Exp |
|-----------|--------|-----|
| gpu_zombie_fix_deployed | kill_gpu_zombies() in ExperimentTemplate, validated | 780 |
| jepa_v20_data_collected | n_labeled >= 80 real pairs from live 100q | 781 |
| edu_prm_diversity | uncertainty_selected_pct >= 0.30 | 782 |
| jepa_v20_ood_viable | ood_auc > 0.75 | 783 |
| tier35_deployed | FR-11 Tier 3 CLOSED or gated correctly | 784 |
| sota_code_repair_positive | signed_improvement > 0 AND live_gpu | 785 |
| gemma4_retro028_closed | loader_test_passed=True | 786 |
| sstar_prefilter_tested | energy_prefilter_saves_tests measured | 787 |
| constraint_addition_tested | constraint_gen_net_improvement measured | 788 |
| ebm_calibration_measured | ECE and isotonic calibration curve saved | 789 |
| npu_iron_attempted | new option tried, verdict honest | 790 |
| kv260_n32_bitstream | pnr_success_ice40=True | 791 |

---

## New arxiv Papers Incorporated (added to research-references.md 2026-04-23)

| Paper | arXiv ID | Applied In |
|-------|----------|-----------|
| EDU-PRM: Entropy-Driven Uncertainty for PRMs | 2503.22233 | Exp 782 |
| S*: Test Time Scaling for Code Generation | 2502.14382 | Exp 787 |
| Know When You're Wrong: Calibration Alignment | 2603.06604 | Exp 789 |
| Energy of Falsehood: Diffusion-Based Energy | 2602.11364 | Exp 789 |
| Beyond Outcome Verification: Verifiable PRMs | 2601.17223 | Exp 788 |
| Adaptive Test-Time Compute via Constrained Policy | 2604.14853 | Exp 788 |
