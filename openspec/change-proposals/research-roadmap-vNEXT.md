# Research Roadmap — Milestone 2026.06.431

**Status:** PROPOSED (outer-loop Claude Opus 4.8 planner, 2026-06-24)
**Theme:** THE WALL IS CANDIDATE GENERATION ("make-a-winner-appear"), NOT SELECTION — six milestones
(`.425–`.430) of verifier/selection levers all nulled on live solve-rate; the convergent residual points
at GENERATION. `.431 pivots to the SOTA-flagged structural generation methods: **hierarchical subgoal
search** (A1, HEADLINE) and **PoE-World factored-executable subgoal planner** (A2), reusing the `.430
levers as COMPONENTS, gated on a decisive diagnostic that resolves whether the live wall is L1-first-contact
or L2-deepening.
**Sprint:** ARC-AGI-3 Submission Sprint (CLAUDE.md, through **2026-06-30** — 6 days to the ARC Prize
Milestone #1 deadline, $25K).
**Predecessor:** 2026.06.430 (`research-roadmap.yaml`, capstone exp4674).

---

## 1. What the previous milestone (.430) proved

`.430 attacked the 0.0 live multi-level solve-rate wall from the operator-pre-staged diagnosis: the
DEGENERATE L2 GOAL PREDICATE. The capstone (exp4674) verdict: **`capability_grew_58_to_59`** — A3 banked a
new reproducible level, but **`bridge_crossed_for_solve = FALSE` for the 6th consecutive milestone**, and
BOTH headline mechanisms nulled on live SOLVE-RATE.

| Task | Lever | Verdict | Outcome |
|---|---|---|---|
| A1 (exp4664) | L2-goal-predicate induction (capture the L1 win-grid exemplar + a goal-satisfiability check + fix the metric harness) | `complete: l2_goal_induction_no_deepening_residual_single_exemplar_goal_insufficient` | **NULL + RETIRED.** `win_state_exemplar_injected=false`, `goal_predicate_satisfiable={lp85:false, sc25:false}`, `l2_plan_reaches_goal=false`, `bare_control_passed=false` (sc25 reached only L0 generically). `retire_if_same_verdict:true` fired. |
| A2 (exp4665) | DAgger-lite distribution-shift value-routing | `complete: dagger_distribution_corrected_no_live_lift_residual_logged` | **NULL.** The correction TOOK mechanically (shift score 0.699→0.0) but `first_win_rate_delta=0.0`, `solve_rate_delta=0.0`. Residual: `missing_verifier_gap_live_frontier_not_separated`. |
| A3 (exp4666) | Level-up self-play (rotation target dc22) | `success: dc22_L2_offline_reproduced` | **BANKED** dc22 L2 → `reproducible_total_levels` 58→59. Learned verifier checkpointed. |
| A4 (exp4667) | Submission-package refresh | `success` | `live_submittable_level_count=59` (>33), `ready_for_operator_submit=True`. |
| B1 (exp4670) | Multi-level-harness CI-gate + proposer-port-hygiene guard | `success` | The degenerate metric (target_levels<2 / break-at-first-win) can no longer silently recur; "Qwen" generation measurements must `/props`-verify the served model (the port-8919 gemma-squat). |
| Capstone (exp4674) | Scorecard + G1–G4 | `complete: capability_grew_58_to_59`, `paper_ready=True` | FoVer 0.9131 frozen; `bridge_crossed_for_solve=False`. |

### The decisive new input: the wall is GENERATION, not selection

Three independent threads converged in `.430, relocating the wall from SELECTION (the verifier/value head)
to CANDIDATE GENERATION (what gets proposed for the verifier to rank):

1. **A2's honest measurement.** Under the FIXED harness (target_levels≥2, no early break, `/props`-verified
   Qwen on a free port), the generic `E3AgentPolicy` reaches first-win on **only 1 of 25 public games
   (lp85)** — `first_win_rate = 0.04`, NOT the ~0.59 the `.430 roadmap assumed. The distribution-corrected
   value head and the `.429 winning-path baseline both score 0.04: value-routing changes nothing because
   the agent fails to reach L1 on 24/25 games *regardless of how the candidates are ranked*. The bottleneck
   is upstream of ranking.

2. **The recurring residual across the persisted primitives.** `ops/arc_solve_registry.yaml`
   `transfer_dead_ends` says the same thing on ≥5 distinct primitives, verbatim: *"If the winning action is
   absent from the candidate group, [the operator] can only reorder generated candidates; candidate
   generation remains the residual bottleneck."* The verifier/value head can only RANK what is generated.

3. **The representation is already done.** The 2026-06-23 re-diagnosis
   (`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`, which SUPERSEDES the
   "perception is the binding constraint" memory) measured v3 features at LOO-AUROC **0.725** (up from the
   0.515 v2 baseline). "Add more features is not the lever." Compute-cost was fixed `.429; distribution-shift
   was fixed `.430 (0.699→0.0) — and *still* no live lift.

**Synthesis (the `.431 thesis).** Every lever `.425–`.430 improved SELECTION — goal-energy, expansion-prior,
value-routing cost, energy-fitness QD, L2-goal-induction, DAgger distribution-shift. All six nulled on live
solve-rate. The operator named this in the 2026-06-22 energy-config-space steer: **"the wall is
make-a-winner-appear, not select."** `.431 stops adding selection levers and attacks GENERATION.

---

## 2. The three biggest gaps (current state vs. north star)

The north star (`ops/north-star.md` §0) is the LIVE agent self-discovering hidden-game solves accurately
and efficiently. The gaps, ranked:

1. **GENERATION — the winning action/plan is not in the candidate pool (HEADLINE GAP).** Flat best-first
   search + a single global goal predicate cannot make a multi-step winner appear: at L1 first-contact on
   24/25 games, and at every L1→L2 boundary. The fix is structural: decompose the goal into reachable
   subgoals and run bounded search per subgoal so a long winning trajectory is assembled from short
   reachable legs. This is the SOTA-flagged `.431 direction.

2. **A measurement we cannot yet trust end-to-end.** The 0.04-vs-0.59 first-win discrepancy means the loop
   has, for several milestones, optimized against a metric whose denominator it did not understand. Before
   `.431 commits a generation build to "deepen to L2", a decisive diagnostic must establish whether the live
   wall is L1-first-contact (the agent can't generically solve 24/25 games) or L2-deepening
   (offline-null-may-be-a-harness-artifact discipline).

3. **The reusable generation scaffolding does not exist.** `arc_solver_kit` has rich SELECTION/routing
   primitives (value heads, trust-energy gates, action-effect memory) but no SUBGOAL-DECOMPOSITION or
   FACTORED-EXECUTABLE-MODEL generation primitive. Whichever `.431 generation lever clears its gate must be
   persisted so the LIVE solver reuses it.

---

## 3. Architecture — where `.431 acts

```
                          THE LIVE ARC AGENT (E3AgentPolicy — the SCORED deliverable)
                          ════════════════════════════════════════════════════════
  frame ──▶ PERCEPTION ──▶ CANDIDATE GENERATION ──▶ VERIFIER-ROUTING ──▶ plan_in_model ──▶ REPLAY GATE
            (v3 feats,      (StepwiseExplorer,        (value head,         (induced          (arc_solver_kit
             LOO 0.725       LLM world-model           trust-energy        dynamics+goal,     .reproduce)
             — DONE)         induction)                gate — DONE-ish)    flat search)
                                   ▲                          ▲                  ▲
                                   │  ◀── .431 A1 ──┐         │ ◀── reused as ──┘
                                   │  HIERARCHICAL  │         │     COMPONENTS:
                                   │  SUBGOAL SEARCH │        │     .430 A1 goal-induction → SUBGOAL PROPOSER
                                   │  (.425–.430     │        │     .430 A2 value head     → within-subgoal
                                   │   levers as     │        │                              TIE-BREAKER
                                   │   components)   │
                                   │  ◀── .431 A2 ──┘
                                   │  POE-WORLD FACTORED-EXECUTABLE SUBGOAL PLANNER
                                   │  (programmatic object-experts; product model; subgoal-conditioned plan)
                                   ▼
                          THE .425–.430 WALL WAS HERE  ── selection of a pool that never contained the winner
                          THE .431 PIVOT              ── make the winner APPEAR in the pool (generation)
```

`.431 changes ONLY the live modules in the `E3AgentPolicy` import closure (`arc_competition_agent`,
`arc_llm_reinduction`, `arc_executable_world_model`, `arc_value_learner`, `arc_solver_kit`) so
`scripts/arc_orphan_solver_lint.py` stays green and `tests/python/test_arc_submitted_agent_parity.py`
stays green — the measured agent IS the deployed agent (ARC Live-Path Reachability Discipline).

---

## 4. Phases

### Phase 0 — Transition (exp4675)
Archive `.430 → activate `.431; assert the YAML parses + the smart-subset pre-test gate is green; record the
TRUE `.430 close-state (A3 dc22 L2 → 59; A1 L2-goal-induction NULL+RETIRED; A2 DAgger shift-corrected but no
live lift; generic first-win honest = 0.04; bridge_crossed=FALSE 6th milestone; package 59>33;
paper_ready=True). Record the `.431 pivot to GENERATION. Mechanical, codex.

### Phase A — ARC north star (majority of the milestone)

- **A1 — HEADLINE: Hierarchical subgoal search over the live E3 frontier (exp4676).** STEP 1 (decisive
  diagnostic): measure generic first-win + multi-level across configs (explore vs value_routed, action
  budgets, the variant set) to resolve 0.04-vs-0.59 and pinpoint the wall (L1-first-contact vs L2-deepening).
  STEP 2: build a bounded subgoal-search prototype targeting the CONFIRMED wall — mine candidate subgoals
  from failed search trees and from `.430 A1's goal-induction (now a subgoal PROPOSER, not a single global
  terminal predicate), run bounded low-level search per subgoal with `.430 A2's distribution-corrected value
  head as the within-subgoal TIE-BREAKER, chain legs, and let live E3 replay-verify. **Gate:** the subgoal
  search makes a winner appear where flat search fails — the GENERIC live agent reaches a NEW level (a
  previously-unreached L1 on a hard game OR an L2), offline-reproduced (`live_agent_self_discovery`), with a
  matched no-subgoal ablation AND a random-subgoal ablation that do NOT. `live_llm_inference`
  (Qwen3.5-9B-MTP, iGPU, free port + `/props`). `verifier_is_oracle:false`. `retire_if_same_verdict`.
  Sources: arXiv:2604.03208, 2506.07255, 2504.04366, 2605.12913, 1011.0686.

- **A2 — Second generation mechanism: PoE-World factored-executable subgoal planner (exp4677).** Induce
  small programmatic object-level experts (precondition/effect), weight by held-out transition trust, compose
  only replay-stable factors, and plan subgoal-conditioned sequences through the product model; `.430 A1's
  goal-induction proposes the subgoal predicates each factor must make reachable; `.430 A2's value head
  scores which product-model states deserve live expansion; live E3 executes and audits every emitted plan.
  This is the strongest answer to A2's no-winner residual because a factored feasibility model is what the
  `.429 energy-fitness QD lacked before it mutated. **Gate:** candidate-generation COVERAGE up (the winning
  action/plan now appears in the pool where the matched flat-search baseline did not generate it) AND a live
  first-win/solve lift on ≥1 game (CI excludes the baseline), offline-reproduced. Second INDEPENDENT
  mechanism — if A1's subgoal search nulls, A2 may cross. `verifier_is_oracle:false`. `retire_if_same_verdict`.
  Sources: arXiv:2505.10819, 2605.05138.

- **A3 — Level-up guarantee + self-play (exp4678).** Bank +1 reproducible level (59→60+) on a CLEAN game NOT
  deepened in `.426–`.430 (PREFER a first-contact L1→L2 of a hard clean public game from the L1-only set:
  bp35/re86/sb26/s5i5/g50t/r11l/lf52; alternatives m0r0/cn04/ar25 L2→L3) + train+checkpoint the learned
  verifier. INDEPENDENT of A1/A2 so the ARC Level-Up Attempt Guarantee holds even if they null.
  `solve_provenance: development_proxy`. `verifier_is_oracle:false`.

- **A4 — Score / package refresh (exp4679).** Fold A3's bank + any A1/A2 new variant into the refreshed
  operator-resubmit package; keep `live_submittable_level_count` STRICTLY > 33 (now ≥59); submission
  operator-only.

- **A5 — Self-learning persist+transfer (exp4680).** Persist this milestone's winning GENERATION primitive
  (the hierarchical-subgoal-search operator OR the PoE-World factored-planner operator, whichever cleared
  its gate; else the strongest characterized component) into `arc_solver_kit` + a registry general_gotcha,
  and measure cross-game transfer (characterize the null honestly).

- **A6 — Integration (exp4681).** Fold the winning A1/A2 config into `SUBMITTED_AGENT_CONFIG` (single source
  of truth); re-measure integrated live first-win + multi-level solve-rate; keep parity green.

### Phase B — Infrastructure (2 reserved slots)

- **B1 — Candidate-generation-coverage CI-metric + honest generic-first-win floor (exp4682).** The
  generation analog of `.430's multi-level-harness gate: (1) a metric/gate measuring "is the winning
  action/plan present in the GENERATED candidate pool" (coverage) so generation improvement is mechanically
  trackable; (2) lock in the HONEST generic-first-win measurement (the 0.04 reality on the standard config)
  so a future change can neither silently regress it NOR silently inflate it via a permissive harness; (3) a
  generation-coverage floor. Unit tests.

- **B2 — adversarial_verify hardening for the `.431 generation lever class (exp4683).** (1)
  SUBGOAL-SEARCH-WITHOUT-DECOMPOSITION-EVIDENCE — a "reached a new level via subgoal search" claim that does
  NOT report the subgoal decomposition + per-subgoal reachability + the no-subgoal AND random-subgoal
  ablations + offline-reproduced is flagged (it may be a flat-search win mislabeled, or a "subgoal" that is
  just the global goal); (2) GENERATION-COVERAGE-WITHOUT-BASELINE — a coverage claim without the matched
  flat-search baseline coverage is flagged. Honest artifacts NOT flagged. Unit tests.

### Phase C — Hardware continuity (1 per-board slot)
- **C (exp4684).** Per-board reachability audit: KV260 via `ssh kria` (SSH-only, NEVER host SD-card),
  PolarFire via `ssh polarfire`, GateMate via `openFPGALoader -c dirtyJtag --detect`. No bitstream build,
  no fabric-acceleration claim.

### Phase D — SOTA-ingestion → `.432 (1 reserved slot)
- **D (exp4685).** Focused literature on the NEXT fallback if GENERATION (A1 subgoal + A2 PoE-World) nulls:
  learned / intrinsic-motivation DIRECTED EXPLORATION for L1-first-contact generation (the agent needs
  better exploration to MAKE the winning trajectory appear on the 24/25 games it can't generically solve) +
  program-synthesis action-model induction. Map 3–5 SOTA methods with real arXiv IDs + implement-cost +
  fails_when; flag the strongest for `.432. Reliable sweep + WebSearch/WebFetch only; `/deep-research` BANNED.

### Phase E — Capstone (exp4686)
Aggregate the scorecard + the HEADLINE DECISION: did GENERATION (hierarchical subgoal A1 / PoE-World A2)
cross the offline→live bridge for SOLVE-RATE/DEPTH where six milestones of SELECTION could not — the GENERIC
agent reaches a NEW level via a candidate-generation improvement, offline-reproduced, with passing ablations?
Did A3 bank +1 (59→60+)? Skip flagged / control-failed / decomposition-missing artifacts. Confirm
`verifier_is_oracle:false` on every value claim. Re-affirm G1–G4 `paper_ready` (FoVer 0.9131 frozen).
Submission operator-only.

---

## 5. Dependency graph

```
exp4675 (Phase 0 transition)
   │
   ├─▶ exp4676 (A1 HEADLINE: hierarchical subgoal search; step 1 = decisive diagnostic)
   ├─▶ exp4677 (A2: PoE-World factored-executable subgoal planner)   [independent of A1]
   ├─▶ exp4678 (A3: level-up bank + self-play)                       [independent — guarantee holds if A1/A2 null]
   ├─▶ exp4682 (B1: generation-coverage CI-metric + first-win floor) [independent]
   ├─▶ exp4683 (B2: adversarial_verify hardening)                    [independent]
   ├─▶ exp4684 (C: hardware continuity)                              [independent]
   └─▶ exp4685 (D: SOTA-ingestion → .432)                            [independent]
        │
        ▼
   exp4679 (A4: package refresh)        ◀── folds A3 + A1/A2 new variants
   exp4680 (A5: persist+transfer)       ◀── persists the winning A1/A2 generation primitive
   exp4681 (A6: integration)            ◀── folds winning A1/A2 config into SUBMITTED_AGENT_CONFIG
        │
        ▼
   exp4686 (E: capstone .431 — the GENERATION bridge decision)
```

---

## 6. Hardware requirements

- **A1 / A2 live induction arms:** Qwen3.5-9B-MTP GGUF on the **iGPU (Radeon 890M)** — the frozen ARC live
  generator (operator 2026-06-19; `project_arc_live_generator`). NEVER the RTX 3090s. Constructed on a FREE
  port (e.g. 8920) + `/props`-verified Qwen (the port-8919 gemma-squat confound). On Kaggle this is moot.
- **A2 / A3 / A5 / B1 offline arms:** CPU (offline arcade + cached candidates + value-head training). No GPU.
- **C:** SSH/USB reachability to KV260 / PolarFire / GateMate.

---

## 7. Disciplines honored

- **ARC-AGI-3 Submission Sprint Forcing Function (through 2026-06-30):** majority ARC (A1–A6); ≥1 level-up
  that BANKS a new reproducible level (A3); self-play EVERY milestone (A3 trains+checkpoints the verifier);
  2 reserved infra (B1/B2); 1 per-board hardware (C); 1 SOTA-ingestion (D). All experiments codex/gpt-5.5;
  planner/retro stay Claude Opus.
- **ARC Level-Up Attempt Guarantee:** A3 banks a new reproducible level (verified by
  `scripts/arc_levelup_guarantee_lint.py`).
- **ARC Live-Path Reachability Discipline:** every change is in the `E3AgentPolicy` import closure
  (orphan-solver-lint green, parity-test green); `solve_provenance` on every ARC solve task, preferring
  `live_agent_self_discovery` (A1) over `development_proxy` (A3); no outer-loop-RE, no offline-ground-truth
  calibration solves.
- **Circularity / Oracle-Distinctness:** `verifier_is_oracle:false` on every value claim (the subgoal
  proposer / value head / programmatic experts are oracle-distinct from the executable reproduction
  win-check).
- **Adversarial Artifact Verification + Sample-Size Rigor:** matched controls + ablations on every live
  claim; CI on every lift; FALSE_NEGATIVE_RISK controls; the offline-null-may-be-a-harness-artifact caution
  drives A1's mandatory step-1 diagnostic.
- **Failed-Experiment Rerun Discipline:** A1/A2/A3 carry `prior_failures` blocks (the `.430 nulls), with
  `addressed_by` naming the technique change (structural generation, not flat selection) and
  `retire_if_same_verdict:true`. Routine continuations (transition/package/persist/integration/infra/
  hardware/SOTA/capstone) carry `operator_override`.
- **Operator-Only External Publication:** all submission tasks PREPARE + offline-validate only;
  `leaderboard_submission=false`.

---

## 8. What success looks like

`.431 succeeds if it produces the FIRST positive `bridge_crossed_for_solve` in seven milestones: a GENERIC
live-agent solve of a NEW level driven by a candidate-generation improvement (subgoal search or factored
planner), offline-reproduced, with passing no-subgoal/random-subgoal ablations and `verifier_is_oracle:false`.
Failing that, the honest secondary success is a *characterized* null that converts the "make-a-winner-appear"
thesis into a measured, mechanism-localized finding (e.g. "subgoal decomposition makes the winner appear in
the pool but bounded low-level search still can't reach the subgoal") — which retires the precise lever and
feeds the `.432 SOTA-ingestion fallback. A3 banks +1 regardless (59→60+), so solve CAPABILITY grows every
milestone even when the live bridge stays open.
