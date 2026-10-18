# Research Roadmap v253: Phase 1 Ship Launch + EORM Tier 0e + ODAR Free-Energy Routing + Ensemble v11

**Milestone:** 2026.05.253
**Previous milestone:** 2026.05.252
**Date:** 2026-05-20
**Status:** PROPOSED

## Post-.252 Planning Sweep New Papers (2026-05-20)

Six new papers added to research-references.md from the .253 planning sweep:
- **arXiv:2505.14999** (EORM: Energy Outcome Reward Model) — 55M-param EBM verifier, 90.7% GSM8k; direct Tier 0e candidate; validates Carnot's energy-based ranking approach.
- **arXiv:2605.12620** (VegAS: Verifier-Guided Action Selection) — test-time K-candidate scoring via verifier; 8.7% math improvement; extends VerifyRepairPipeline score_candidates.
- **arXiv:2604.16217** (Layer-Wise Representation Conformal) — internal layer-wise info scores for OOD-robust conformal prediction; Tier 0l candidate.
- **arXiv:2604.01413** (MiCP: Adaptive Stopping Conformal) — multi-turn early stopping with coverage guarantee; 34% cost reduction; extends PASC (exp2641).
- **arXiv:2605.09387** (NEXUS: Continual Symbolic Constraint Learning) — FR-11 Tier 2 reference; violation-pattern → symbolic rule synthesis.
- **arXiv:2602.23681** (ODAR: Free-Energy Routing) — Phase 4 active inference + Fast-Slow Variant merger; KL-gated fast/slow path; 23 benchmarks.

---

## What Milestone .252 Proved (8 success criteria)

### Expected Wins (from .252 task designs — all confirmed complete per user)
1. **GGUF full-scale benchmark 100 examples** (exp2635): pipeline_e2e_latency_mean_s ± std on N=100 examples with SOTA GGUF model; first statistically valid latency claim.
2. **Tier 0w AvgWD/EigenWD implemented** (exp2636): embedding-geometry hallucination verifier; tier0w_auroc on FoVer real corpus; tier0w_viable bool.
3. **Behavioral entanglement audited** (exp2637): pairwise correlation matrix; de-entangled weights saved to ensemble_deentangled_weights.json.
4. **Ensemble v10 adversarially validated** (exp2638): incorporates Tier 0w + de-entangled reweighting; 5-seed; adversarially_verified bool; headline AUROC for paper-v6.
5. **TTT statistical significance established** (exp2639): N=100, 5 seeds, bootstrap CI, paired t-test; fr11_tier3_headline_closed bool.
6. **Symbolic-KAN assessed** (exp2640): symbolic_kan_viable bool; human-readable activation explanations.
7. **PASC joint coverage achieved** (exp2641): multi-stage pipeline coverage_joint_90pct vs. Bonferroni.
8. **Phase 1 ship readiness audited** (exp2642): phase1_ship_ready bool; n_gates_passed; operator_action_checklist.
9. **arXiv v5 package built** (exp2643): submission_package_ready bool (OPERATOR-ONLY submission).
10. **KV260 continuity maintained** (exp2644): Branch A/B.
11. **Capstone v252** (exp2645): phase1_ship_recommendation 'SHIP' or 'HOLD'; top_3_gaps_for_253.

### Gaps (entering .253)
1. **Phase 1 ship NOT yet executed** — .252 exp2642 audited the gates and exp2645 produced a SHIP/HOLD recommendation. But the actual ship actions (PyPI version tag release, GitHub release, announcement docs) require execution. If phase1_ship_ready=True, .253 exp2648 closes the gap.

2. **No EORM-class trained energy verifier** — .252 added Tier 0w (training-free, AvgWD/EigenWD embedding geometry). The ensemble still lacks a TRAINED energy-based verifier analogous to arXiv:2505.14999's EORM. Tier 0e EORM fills this: a logistic-regression energy ranker trained on (correct, incorrect) pairs from FoVer.

3. **ODAR free-energy routing not yet prototyped** — Carnot verifies unconditionally (every claim → full ensemble). ODAR (arXiv:2602.23681) demonstrates that free-energy-gated routing (verify only when KL divergence is high) reduces computational overhead across 23 benchmarks. This directly addresses Phase 4 (active inference) and the stalled alpha_t measurement track.

---

## Three Biggest Gaps Between Current State and PRD Vision

### Gap 1: Phase 1 Ship Execution (HIGHEST PRIORITY)
**State:** .252 exp2642 determined phase1_ship_ready (True or False). If True: the SHIP action was recommended but not executed by the autonomous system (per Operator-Only External Publication rule — the announcement itself is operator-driven, but the CI release tag and HF model card updates CAN be done autonomously as internal operations).

**Why it matters for PRD:** "Phase 1 ship gate is purely software-operational: all FR-* implemented, PyPI package + Apache-2.0 shipped, HuggingFace mirror per Rule 3, MCP server + CLI documentation, at least one independent reproducer." If all gates pass, Phase 1 is shippable. The gap is execution, not evidence.

**Plan:** exp2648 reads exp2642 artifact. If phase1_ship_ready=True: executes ship actions (git tag, CI trigger, HF model card announcement section). If False: executes exactly the listed operator_action_checklist steps that can be done autonomously.

### Gap 2: EORM-Class Trained Energy Verifier as Tier 0e
**State:** Ensemble v10 has tier0s/tier0u (TF-IDF logistic, trained), tier0z (Boltzmann, training-free), tier0w (AvgWD/EigenWD, training-free). Missing: a TRAINED verifier that explicitly models the energy difference between (correct, incorrect) output pairs.

**Why it matters for PRD:** arXiv:2505.14999 (EORM) demonstrates 90.7% GSM8k coverage with a 55M-param energy ranker trained on CoT pairs. Carnot's ensemble currently trains each verifier independently on binary labels; EORM's insight is to train directly on PAIRWISE energy differences (margin ranking loss). This is theoretically stronger and aligns with Carnot's Gibbs sampling energy foundation.

**Plan:** exp2649 implements Tier0eEROM using sklearn LogisticRegression on pairwise FoVer feature differences. Fitting on (correct_claim_features, incorrect_claim_features) pairs with margin ranking loss. Target: tier0e_auroc >= 0.70.

### Gap 3: ODAR Free-Energy Routing for Phase 4 / Fast-Slow Variant
**State:** Carnot's VerifyRepairPipeline runs full verification on every claim. Phase 4 (active inference hypothesis) aimed to measure alpha_t variational free energy but stalled in exp1715/1721/1741/1745.

**Why it matters for PRD:** ODAR (arXiv:2602.23681) solves the Phase 4 measurement problem by pivoting from metric measurement to operational deployment: use the free-energy principle as a ROUTING GATE (fast path if KL < threshold, slow path if KL ≥ threshold). This is exactly what Phase 4 was trying to prove — that the verifier IS the free-energy term in the active inference loop. ODAR provides the working implementation.

**Plan:** exp2654 implements ODAR-style gating in VerifyRepairPipeline. Compute KL divergence between the ensemble's prior estimate and its posterior (after repair). If KL < threshold (fast path): skip iterative repair. If KL ≥ threshold: run full repair loop. Measure: computational overhead reduction AND accuracy retention.

---

## Architecture Snapshot (entering .253)

```
Verifier Ensemble v10 (entering .253 — adversarially validated via .252 exp2638):
  Group A (logprob): tier0a, tier0b, tier0c
  Group B (semantic): tier0d, tier0e, tier0f
  Group C (type/logic): tier0g, tier0h, tier0i
  Group D (Curry-Howard): tier0r (AUROC=0.9123)
  Group E (hallucination-specific): tier0t (dynamical), tier0v (HalluField proxy)
  Group E-retrained: tier0s (TF-IDF logistic, .250), tier0u (TF-IDF cosine, .250)
  Group E-training-free: tier0z (Boltzmann semantic energy, .250)
  Group E-embedding: tier0w (AvgWD/EigenWD, .252)
  Group F (safety): tier0x v2 (FJD logit-temp)
  De-entangled weights: ensemble_deentangled_weights.json (.252 exp2637)
  Pending .253:
    Tier 0e (EORM margin ranking — arXiv:2505.14999) — prototype in .253
    Tier 0l (Layer-wise info drift — arXiv:2604.16217) — prototype in .253

Ensemble v10 AUROC: from exp2638 adversarially_verified result
v9 carry-forward: 0.9857 (adversarially verified)

FR-11 Self-Learning Stack:
  Tier 1: Online weight updates — WIRED
  Tier 2: Constraint memory — WIRED; NEXUS accumulation pattern queued for .253
  Tier 3: JEPA online_update() — WIRED + evaluated on real data (.250 exp2617)
  Tier 3 TTT: VerifierDrivenTTT — SCALED (.252 exp2639, N=100, significance testing)
  Tier 3 TTT HEADLINE: fr11_tier3_headline_closed from exp2639
  Tier 4: Adaptive energy (KAN structural) — PROTOTYPED
  Phase 4: ODAR free-energy routing — QUEUED for .253 exp2654

Pipeline:
  VerifyRepairPipeline (unconditional verify) — ACTIVE
  VegAS candidate selection (K=3 repair candidates) — QUEUED for .253 exp2651
  ODAR fast-path gating — QUEUED for .253 exp2654
  MiCP early stopping — QUEUED for reference

Hardware:
  GateMate A1-EVB-2M: TERMINAL (.247 capstone exp2580) — graduated
  KV260: NON-TERMINAL (SD card absent; synthesis succeeded; MANDATORY per .253)
  PolarFire SoC: TERMINAL (.241 exp2501) — graduated
  RTX 3090 x2: available (CUDA + JAX GPU)
  AMD Strix Point gfx1150: ROCm 7.2.3 verified

Publication:
  arXiv v5 package: prepared by exp2643 (.252) — OPERATOR-ONLY submission
  paper_updated_with_v10: pending exp2657 result from .253
  Phase 1 ship: phase1_ship_ready from exp2642 (.252) — execution in exp2648 (.253)
```

---

## Dependency Graph

```
exp2647 (archive .252 + activate .253)
    │
    ├── exp2648 (Phase 1 Ship Close — reads exp2642 outcome) ───────────────────────────────┐
    │                                                                                         │
    ├── exp2649 (Tier 0e EORM margin ranking verifier) ────────────────────────────────────┐ │
    │                                                                                       │ │
    ├── exp2650 (Tier 0l layer-wise info drift verifier) ──────────────────────────────────┤ │
    │                                                                                       │ │
    ├── exp2651 (VegAS K=3 candidate selection in VerifyRepairPipeline) ─────(pipeline)    │ │
    │                                                                                       │ │
    ├── exp2652 (FR-11 Tier 2: NEXUS symbolic constraint memory) ─(FR-11 mandate)          │ │
    │   [continuous_self_learning_task: true]                                               │ │
    │                                                                                       │ │
    ├── exp2653 (ensemble v11: Tier 0e + Tier 0l + 5-seed adversarial) ◄───────────────────┘ │
    │   [gated_on: exp2649.tier0e_viable == true]                                             │
    │                                                                                         │
    ├── exp2654 (ODAR free-energy routing prototype) ─────────────────────────────────────   │
    │                                                                                         │
    ├── exp2655 (external benchmark: EORM leaderboard comparison) ─────────────────────────  │
    │                                                                                         │
    ├── exp2656 (KV260 hardware continuity) ─(MANDATORY per CLAUDE.md)                       │
    │                                                                                         │
    ├── exp2657 (arXiv v6 package, OPERATOR-ONLY) ◄──────────────────────────────────────────┘
    │   [gated_on: exp2653.adversarially_verified == true]
    │
    └── exp2658 (capstone v253, claude+opus) ◄ reads all above
            │
        exp2659 (retro v253)
```

---

## Phase Descriptions

### Phase 0: Archive and Activation (exp2647)
Archive milestone .252 into `research-complete.yaml`. Copy `research-roadmap-next.yaml` →
`research-roadmap.yaml`. Records .252 outcomes: GGUF full-scale benchmark, Tier 0w viable,
entanglement audit, ensemble v10 AUROC, TTT statistical significance, Phase 1 ship_ready,
arXiv v5 package status.

### Phase 1: Phase 1 Ship Close (exp2648)
**Execute ship actions based on exp2642 phase1_ship_ready outcome.**

If phase1_ship_ready == true:
- Identify current PyPI version: `python -c "import importlib.metadata; print(importlib.metadata.version('carnot-ebm'))"`
- Determine if a new release tag is needed. If version < target (0.2.0+): prepare release notes.
- Update README.md with Phase 1 announcement section (no external announcement — operator does that).
- Update HF model card to mark Phase 1 milestone.
- Write operator checklist: exact git tag command, exact `gh release create` command.

If phase1_ship_ready == false:
- Execute the specific blocking actions from exp2642.operator_action_checklist that CAN be done autonomously.
- Document which steps require operator action.

### Phase 2a: Tier 0e EORM Verifier (exp2649)
**Energy Outcome Reward Model verifier** per arXiv:2505.14999. Implements pairwise margin ranking
energy: for each (correct_claim, incorrect_claim) pair from FoVer, the verifier learns that the
correct claim has lower energy. Implemented as a LogisticRegression on (correct - incorrect) TF-IDF
feature differences. This is the trained-energy analog to Tier 0z (training-free Boltzmann energy).

Target: tier0e_auroc >= 0.70 on FoVer test split (N >= 40).
Paper-v6 §5 cite: "independent parallel work (arXiv:2505.14999) validates our pairwise energy ranking approach."

### Phase 2b: Tier 0l Layer-Wise Info Drift Verifier (exp2650)
**Layer-wise information drift verifier** per arXiv:2604.16217. Proxy implementation: compute per-
sentence TF-IDF vectors, measure cosine drift from sentence 1 to last sentence. Hallucinations
grounded in early-context claims should have low drift; hallucinations introducing new information
should have high drift. Orthogonal to Tier 0w (AvgWD/EigenWD captures within-claim variance; Tier
0l captures across-sentence information drift).

Target: tier0l_auroc >= 0.65 on FoVer test split.

### Phase 2c: Ensemble v11 Build + Adversarial Validation (exp2653)
**Ensemble v11 incorporating Tier 0e + Tier 0l**, with 5-seed adversarial validation. Gated on
exp2649.tier0e_viable == true. Runs the same 5-seed validation protocol as exp2622 and exp2638.
Provides adversarially-verified headline AUROC for paper-v6 update (arXiv v6).

Target: ensemble_v11_auroc_mean >= v10 adversarially-verified AUROC.

### Phase 3: VegAS Pipeline Enhancement (exp2651)
**Verifier-Guided Action Selection in VerifyRepairPipeline** per arXiv:2605.12620. Adds a
`score_candidates(context, claims_list)` method that scores K candidates with the ensemble and
returns the one with lowest energy. Evaluate on FoVer: does candidate-ranking improve precision
vs. the pipeline's current single-repair approach?

This implements the "score_candidates MCP tool" from research-program.md Tier A (Candidate Ranker product).

### Phase 4: FR-11 Tier 2 NEXUS Symbolic Constraints (exp2652)
**FR-11 Tier 2 scale-up using NEXUS pattern** (arXiv:2605.09387). Implements:
1. Cross-session violation event accumulation in ConstraintStateMachine
2. Symbolic rule synthesis from repeated violation patterns
3. Pattern consolidation (merge redundant templates)
4. Auto-add constraints for repeat-offender patterns

Evaluates: after 3 simulated sessions, session 3 benefits from constraint memory vs. session 1 cold start.
continuous_self_learning_task: true — this is the FR-11 mandate for .253.

### Phase 5: ODAR Free-Energy Routing (exp2654)
**ODAR-style fast/slow path routing** per arXiv:2602.23681. Implements:
- Fast path: if ensemble confidence is high (low KL divergence from prior), skip iterative repair.
- Slow path: if KL divergence is high, run full verify-repair loop.
- KL threshold tuned on validation set to preserve accuracy while reducing verification calls.

This is the Phase 4 (active inference) implementation: the verifier IS the free-energy gate.

### Phase 6: External Benchmark Comparison (exp2655)
**EORM leaderboard comparison** — compare Carnot ensemble v11 AUROC against EORM benchmark
results from arXiv:2505.14999 (GSM8k, HumanEval) and HalluScan/PARALLAX results from .251
exp2623. Determines Carnot's position relative to EORM-class systems.

### Phase 7: Hardware (exp2656)
**KV260 hardware continuity** per CLAUDE.md Hardware-Task Continuity Discipline.
Branch A: SD card detected → PYNQ flash + latency transcript.
Branch B: SD absent → update prep script.

### Phase 8: Publication (exp2657) — gated on adversarial validation
**arXiv Final Package v6** — updated with ensemble v11 adversarially-validated AUROC (exp2653).
Gated on exp2653.adversarially_verified == true. OPERATOR-ONLY submission.

### Phase 9: Synthesis (exp2658, exp2659)
**exp2658 (capstone, claude+opus):** Synthesizes all .253 experiments. Evaluates 8 success criteria.
Determines Phase 1 ship status after execution attempt. Documents top 3 gaps for .254.

**exp2659 (retro, codex):** Operational retrospective with timing analysis.

---

## Hardware Requirements

| Board | State | .253 Task | Next Gate |
|---|---|---|---|
| KV260 | NON-TERMINAL | exp2656 (Branch A: SD flash; Branch B: prep update) | SD card insertion + PYNQ flash |
| GateMate A1-EVB-2M | TERMINAL | None | Graduated |
| PolarFire SoC | TERMINAL | None | Graduated |
| RTX 3090 x2 | Available | exp2649 (if model needed); exp2651 (K=3 candidates) | None |

---

## Decentralization Compliance Check (CLAUDE.md Rules 1–7)

1. **Local-first open models** — all verifier evals use local FoVer corpus. EORM uses TF-IDF proxy (no external model). ODAR uses local ensemble scores. ✓
2. **Closed-weight integration optional** — no closed-weight API calls in any .253 task. ✓
3. **Distribution mirroring** — exp2657 updates HF primary + IPFS secondary per Rule 3; exp2648 Phase 1 ship updates HF model card. ✓
4. **Multiple integration surfaces** — VegAS (exp2651) extends the Python API + MCP score_candidates tool; no surface drift. ✓
5. **Hardware portability** — all verifier tasks run on CPU (TF-IDF, logistic regression). KV260 is sovereignty hardware. ✓
6. **Data minimization** — no closed-weight LLM calls → no data flows to external vendors. ✓
7. **No vendor abstractions in core** — all .253 code targets python/carnot/verify/ via abstract protocols. ✓

---

## Exclusion Manifest Cross-Check

Checked `ops/exclusion_manifest.yaml` against .253 task scopes. All retired IDs checked: 2091, 260, 308, 309, 346, 380-383, 410, 425, 491, 527, 603, 627, HalluSAEGeometricProbe, 887/783/799/804/809/825/834/872, iCE40-PIMI, GRPO/VPRM v15, WOPR puzzle cartridges, HardNet++/DSP, THRML scaling sweep, SpecAnn.

| .253 Task | Pattern checked | Result |
|---|---|---|
| exp2647 (archive .252) | No retired archive pattern | CLEAR |
| exp2648 (Phase 1 ship close) | No retired ship-execution pattern | CLEAR |
| exp2649 (Tier 0e EORM) | HalluSAEGeometricProbe (retired SAE geometry) — checked: EORM uses margin ranking on TF-IDF pairs, NOT SAE probes; arXiv:2505.14999 | CLEAR |
| exp2650 (Tier 0l layer-wise) | No retired layer-wise info scope | CLEAR |
| exp2651 (VegAS candidate selection) | No retired candidate-ranking scope | CLEAR |
| exp2652 (FR-11 Tier 2 NEXUS) | discriminative JEPA OOD (retired) — different: NEXUS is symbolic constraint accumulation, not JEPA discriminative OOD | CLEAR |
| exp2653 (ensemble v11) | No retired ensemble v11 pattern | CLEAR |
| exp2654 (ODAR routing) | No retired ODAR routing scope | CLEAR |
| exp2655 (external benchmark) | No retired external benchmark pattern | CLEAR |
| exp2656 (KV260) | iCE40-PIMI (retired) — different: PIMI retired; KV260 SD flash is non-PIMI track | CLEAR |
| exp2657 (arXiv v6) | No retired arXiv submission scope | CLEAR |
| exp2658 (capstone) | No retired capstone pattern | CLEAR |
| exp2659 (retro) | No retired retro pattern | CLEAR |

Zero manifest matches across all 13 tasks.

---

## Failed-Experiment Rerun Compliance

| Task | Prior related experiment | Addressed by |
|---|---|---|
| exp2647 (archive .252) | exp2634 (archive .251) — complete | Different lifecycle: archiving .252 |
| exp2648 (Phase 1 ship) | exp2642 (Phase 1 audit) — complete | .252 AUDITED gates; .253 EXECUTES ship actions. Different scope. |
| exp2649 (Tier 0e EORM) | exp2636 (Tier 0w AvgWD) — complete | Different: Tier 0w is training-free embedding geometry; Tier 0e is TRAINED margin-ranking energy; different paper; different training objective |
| exp2650 (Tier 0l layer-wise) | exp2641 (PASC pipeline conformal) — complete | Different: PASC is multi-stage coverage; Tier 0l is a new verifier using layer-wise info drift signal |
| exp2651 (VegAS selection) | exp2624 (TTT loop) — complete | Different: TTT selects FEW-SHOT context examples; VegAS selects among K REPAIR CANDIDATES via energy ranking. Different mechanism, different output. |
| exp2652 (NEXUS Tier 2) | exp2617 (JEPA real-data eval) — complete | Different tier: exp2617 was Tier 3 JEPA online_update; NEXUS is Tier 2 symbolic constraint accumulation across sessions |
| exp2653 (ensemble v11) | exp2638 (ensemble v10 adversarial val) — complete | Different ensemble: v11 adds Tier 0e + Tier 0l; different composition vs v10 |
| exp2654 (ODAR routing) | exp2639 (TTT scale-up) — complete | Different mechanism: TTT adapts FEW-SHOT CONTEXT; ODAR gates WHETHER to verify at all via free-energy KL threshold |
| exp2655 (external benchmark) | exp2623 (HalluScan+PARALLAX .251) — complete | Different target: exp2623 evaluated ensemble v9; .253 exp2655 compares ensemble v11 against EORM leaderboard position |
| exp2656 (KV260 .253) | exp2644 (KV260 .252) — complete | .253 continuation; Hardware-Task Continuity Discipline mandate |
| exp2657 (arXiv v6) | exp2643 (arXiv v5 .252) — complete | Different package: v6 incorporates ensemble v11 AUROC (v10 in v5) + external benchmark comparison |
| exp2658 (capstone) | exp2645 (capstone .252) — complete | Different evidence base: 12 .253 artifacts vs 12 .252 artifacts |
| exp2659 (retro) | exp2646 (retro .252) — complete | Standard retro continuation |

---

## Agent Routing

| Task | Agent | Why |
|---|---|---|
| exp2647–exp2656, exp2659 | codex + gpt-5.5 | Formulaic: archive, ship actions, verifier prototypes, pipeline extension, FR-11 symbolic memory, entanglement math, ensemble build, ODAR gating, benchmark comparison, hardware branch, retro. Each is single-scope with deterministic gates or documented blocked-if verdicts. |
| exp2657 (arXiv v6) | codex + gpt-5.5 | Formulaic LaTeX update (insert v11 AUROC numbers, add EORM comparison section) — mechanical splice. |
| exp2658 (capstone) | claude + opus, requires_claude: true | Cross-artifact synthesis of 12 deliverables: Phase 1 ship execution result, EORM viability, VegAS pipeline impact, FR-11 Tier 2 NEXUS result, ODAR overhead reduction, ensemble v11 AUROC, external benchmark position, arXiv v6 readiness. Requires cross-context judgment and open-ended Phase 1 announcement recommendation. Meets all 3 positive-criterion conditions: (1) synthesis under ambiguity; (2) 12 files; (3) no deterministic threshold covers all decisions. |

**Routing distribution:** 12 codex/gpt-5.5 (92.3%), 1 claude+opus (7.7%) — codex-default discipline maintained.

---

## Critical Path

```
exp2649 (Tier 0e EORM) → exp2653 (ensemble v11) → exp2657 (arXiv v6) [all gated]
```

If Tier 0e fails (tier0e_viable == false), exp2653 falls back to "ensemble v10 + Tier 0l only" mode.
The arXiv v6 gate is hard (adversarially_verified == true from exp2653).

---

## What Success Looks Like for .253

- `phase1_ship_executed: true` (exp2648) OR `operator_action_checklist` updated with autonomous steps taken
- `tier0e_viable: true` AND `tier0e_auroc >= 0.70` (exp2649) — EORM verifier viable
- `tier0l_auroc >= 0.65` (exp2650) — layer-wise info verifier viable
- `vegas_candidate_selection_improves_precision: bool` (exp2651) — VegAS impact measured
- `nexus_session3_improvement: bool` (exp2652) — FR-11 Tier 2 cross-session benefit demonstrated
- `ensemble_v11_auroc_mean >= ensemble_v10_auroc_mean` with `adversarially_verified: true` (exp2653) — v11 improves on v10
- `odar_overhead_reduction_pct: float > 0` (exp2654) — ODAR reduces verification overhead
- `n_experiments_completed >= 8` (exp2658 capstone) — sustained execution
- `submission_package_ready: true` (exp2657) — arXiv v6 ready for OPERATOR submit
