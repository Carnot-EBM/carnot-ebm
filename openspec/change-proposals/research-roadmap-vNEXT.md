# Research Roadmap — Milestone 2026.06.430

**Status:** PROPOSED (outer-loop Claude Opus 4.8 planner, 2026-06-24)
**Theme:** THE MULTI-LEVEL WALL IS THE DEGENERATE L2 GOAL PREDICATE — fix L2-goal induction (A1, operator-pre-staged HEADLINE) AND the distribution-shift in value-routing (A2, B1-localized), the two independent measured mechanisms for chaining a 2nd live level-up.
**Sprint:** ARC-AGI-3 Submission Sprint (CLAUDE.md, through **2026-06-30** — 6 days to the ARC Prize Milestone #1 deadline, $25K).
**Predecessor:** 2026.06.429 (`research-roadmap.yaml`, capstone exp4662).

---

## 1. What the previous milestone (.429) proved

`.429 attacked the 0.04 live multi-level solve-rate wall with two generation-GUIDANCE levers on the
SCORED `E3AgentPolicy`. The capstone (exp4662) verdict: **`capability_grew_57_to_58`** — A3 banked a
new reproducible level, but BOTH headline levers NULLED on live solve-rate. The pattern (a generation
lever crosses for EFFICIENCY/first-win but NOTHING crosses for SOLVE-RATE/DEPTH) held for the 5th
consecutive milestone.

| Task | Lever | Verdict | Outcome |
|---|---|---|---|
| A1 (exp4652) | Productionize the compute-cost fix (`scipy.ndimage.label`, 13ms→<1ms/node, output identical) + raise `value_weight` off 0.0 | `complete: value_routing_cost_fixed_no_live_lift_residual_dist_shift_or_calibration` | Cost fix WORKED (per-node <1ms, no timeout); but the affordable value head gave `first_win_rate_delta=0.0`, `solve_rate_delta=0.0` — **NULL** |
| A2 (exp4653) | Energy-as-fitness QD/MAP-Elites over action sequences | `complete: energy_fitness_qd_no_winner_generated_honest_null_gap_sharpened` | **NULL** — generated no winner the best-first search missed |
| A3 (exp4654) | Level-up self-play (rotation target vc33) | `success: vc33_L2_offline_reproduced` | **BANKED** vc33 L2 → `reproducible_total_levels` 57→58 |
| B1 (exp4658) | Value-routing CI-gate + distribution-shift-vs-calibration diagnostic | `success: value_routing_cigate_plus_diagnostic_shipped_tests_green` | **Localized the residual:** `distribution_shift_score=0.699`, `calibration_changes_routing=False`, `dominant_residual_cause=distribution_shift` |
| Capstone (exp4662) | Scorecard + G1–G4 | `complete: capability_grew_57_to_58`, `paper_ready=True` | FoVer 0.9131 frozen; live multi-level solve-rate STILL 0.0 |

**The decisive new input (the diagnosis the operator pre-staged for .430).** A clean-Qwen, fully
instrumented outer-loop diagnosis (`docs/research-notes/multi-level-deepening-diagnostic-2026-06-23.md`,
committed `8bb8a4cfd`; root-cause workflow `wf_fcab5470-68f`) **pinpointed exactly where L1→L2 fails**,
ending the lever-guessing:

- The live first-win rate is ~0.59 but the live multi-level (≥2 levels on a fresh game) rate is ~0 — the
  generic `E3AgentPolicy` reaches L1 by exploration but never deepens to L2.
- The `level_up_reinduction` path (`arc_competition_agent.py:_induce_and_plan` →
  `arc_llm_reinduction.execute_bounded_llm_reinduction`) is a COMPOUND failure, but the BINDING
  constraint is the **degenerate L2 GOAL PREDICATE**: when the proposer succeeds and the dynamics model
  passes the (vacuous) held-out gate, `plan_in_model` returns `no_reachable_plan`
  (`arc_llm_reinduction.py:114`). Root cause: at the level-up the active-transition window has ZERO
  L2-win positives, so the induce prompt's WIN-STATE block is absent
  (`arc_executable_world_model.py:308-311` only emits it when a transition has `level_after>level_before`).
  The LLM writes `is_level_complete` for L2 from NO exemplar, and it is NEVER verified (the held-out gate
  checks DYNAMICS only) → unsatisfiable → the planner has no reachable goal.
- **Two cheap levers are DEAD (do not re-propose):** relaxing the held-out gate is a NO-OP (it passes
  vacuously on ~0 held-out data); delaying the one-shot BACKFIRES (the explorer hits `explored_out` at
  ~5 post-L1 transitions → a stall-induction preempts the reinduction).
- **A measurement-harness artifact must be fixed too:** `live_multi_level_solve_rate` is a constant 0.0
  **by construction** — the rollout (`experiment_4628…run_variant_attempt`) breaks at the first level-up
  (`target_levels=1` + `break`), so `depth≥2` is structurally impossible in the metric's input. No lever
  can move this metric until the harness uses `target_levels≥2` + no early break + a real proposer on a
  non-colliding port (the port-8919 confound: a persistent gemma server squats the default proposer port).

**Conclusion carried into .430:** the wall is GENERATION/INDUCTION at the L1→L2 boundary (the degenerate
goal predicate), NOT exploration, NOT reranking, NOT representation (`cross_game_features_v3` is
LOO-AUROC 0.725). Attack it from the two independent measured mechanisms below.

---

## 2. The three biggest gaps (current state → PRD/north-star vision)

1. **The live agent cannot self-discover a 2nd level on a fresh game (the north-star deliverable).**
   All 9 multi-level games reach L2+ ONLY via hand-built `GameAdapter`s (`development_proxy`). A GENERIC
   agent reaching L2 from its OWN attempts is unproven AND, per the harness artifact, was never measured.
   This is the literal ARC-AGI-3 deliverable (a live hidden-game discovery agent). **Gap closed by A1+A2.**
2. **The verifier's learned value head does not transfer to the live frontier (the verifier-moat risk).**
   B1 localized the value-routing null to distribution-shift: the win-reachability head is trained on
   winning-path states but applied to an off-path live frontier. The verifier earns its place only if its
   learned signal works WHERE THE AGENT ACTUALLY IS. **Gap closed by A2 (DAgger-lite).**
3. **Measurement integrity: a degenerate metric hid the wall for a full milestone.** `.429's two levers
   were evaluated against a metric that cannot move. **Gap closed by A1's harness fix + B1's CI-gate.**

---

## 3. Architecture — the two independent attacks on one wall

```
                          THE L1 -> L2 MULTI-LEVEL WALL (live multi-level solve-rate ~0)
                                              |
            +---------------------------------+---------------------------------+
            |                                                                   |
   A1: LLM-GOAL-INDUCTION PATH (HEADLINE, operator pre-staged)        A2: LEARNED-VALUE-ROUTING PATH (B1-localized)
   arc_llm_reinduction.execute_bounded_llm_reinduction               arc_value_learner win-reachability head
            |                                                                   |
   die at no_reachable_plan: L2 is_level_complete induced              null at live frontier: trained on
   from ZERO L2 positives -> unsatisfiable goal                        winning-path, applied off-path (shift 0.699)
            |                                                                   |
   FIX: (1) capture the L1 win-grid at _begin_level_goal_episode       FIX: DAgger-lite -- re-collect value
        -> inject as the WIN STATE exemplar in _transitions_block           training data on SEARCH-DISTRIBUTION
        ("a state that COMPLETED the prev level; next level likely          (off-path frontier) states the agent
        structurally similar")                                              actually visits, re-train, re-route
        (2) GOAL-satisfiability check before planning (reject a
        constant-False is_level_complete with counterexample
        degenerate_goal_predicate)                                     SOTA: arXiv:2605.12913 (DAgger/LLM-agents),
   SOTA: arXiv:2511.19355, 2506.06303, 2603.09036                            1011.0686 (DAgger), 2506.23793
            |                                                                   |
            +---------------------------------+---------------------------------+
                                              |
                    GATE: generic live agent reaches L2 on lp85 AND/OR sc25,
                    offline-reproduced via arc_solver_kit.reproduce
                    (solve_provenance: live_agent_self_discovery -- the real deliverable)
                                              |
   PREREQUISITE (A1 step 1 + B1 CI-gate): FIX the live_multi_level_solve_rate HARNESS
   (target_levels>=2, no early break, non-colliding proposer port + /props verify Qwen) -- else no lever is measurable
```

Both levers are **on the SCORED `E3AgentPolicy`**, **live-path-reachable** (`arc_orphan_solver_lint`
green), and **`verifier_is_oracle: false`** (the goal predicate and the value head are oracle-distinct
from the executable win-check). Diversified attack: if A1 (induction) nulls, A2 (value-routing) may
cross; if both null, B1 re-localizes and D flags the structural subgoal-search lever for .431.

---

## 4. Phases & tasks (12 tasks)

| Phase | id | Track | Agent | Substrate | Deliverable |
|---|---|---|---|---|---|
| 0 transition | exp4663-phase0 | transition | codex | aggregation | archive .429 / activate .430 |
| **A1 HEADLINE** | exp4664-a1 | arc-north-star | codex | **live_llm_inference** | L2-goal-predicate induction fix + harness fix |
| **A2** | exp4665-a2 | arc-north-star | codex | verifier_scoring | DAgger-lite distribution-shift value-routing |
| A3 level-up | exp4666-a3 | arc-north-star | codex | verifier_scoring | bank +1 level (58→59+) + train verifier |
| A4 score | exp4667-a4 | arc-north-star | codex | verifier_scoring | refresh operator-resubmit package (>33) |
| A5 self-learning | exp4668-a5 | arc-north-star | codex | verifier_scoring | persist+transfer the winning primitive |
| A6 integration | exp4669-a6 | arc-north-star | codex | verifier_scoring | fold winning config into SUBMITTED_AGENT_CONFIG |
| B1 infra | exp4670-b1 | infra | codex | verifier_scoring | multi-level harness CI-gate + port-hygiene guard |
| B2 infra | exp4671-b2 | infra | codex | aggregation | adversarial_verify: L2-goal + degenerate-metric guards |
| C hardware | exp4672-c | hardware | codex | hardware_smoke | per-board reachability audit |
| D SOTA→.431 | exp4673-d | sota-ingestion | codex | aggregation | structural-deepening ingestion (subgoal-search/PoE-World) |
| E capstone | exp4674-e | capstone | codex | aggregation | scorecard + headline decision + G1–G4 |

**Sprint-compliance:** 6 ARC tasks (A1–A6) of the 8 non-reserved/non-transition slots = MAJORITY ARC ✓.
Reserved: 2 infra (B1/B2) + 1 hardware (C) + 1 SOTA-ingestion (D) ✓. Level-Up Attempt Guarantee = A3
(banks a new reproducible level) ✓. Self-play-every-milestone = A3 (trains+checkpoints the verifier) ✓.
All experiments `codex`/`gpt-5.5`; planner/retro stay Claude Opus 4.8 ✓ (operator 2026-06-19).

### Phase A1 (HEADLINE — operator pre-staged 2026-06-24): L2-goal-predicate induction
The exact fix the operator pre-staged (known-issues 2026-06-24). **Step 1 (prerequisite):** fix the
`live_multi_level_solve_rate` harness (`target_levels≥2`, drop the break-at-first-win, construct the
`LocalGGUFProposer` on a non-colliding port + verify via `/props` it serves Qwen3.5-9B-MTP, not the gemma
squatting 8919). **Step 2:** capture the level-up grid at `_begin_level_goal_episode` and inject it into
the L2 reinduction's WIN-STATE block (`arc_executable_world_model.py:_transitions_block`), labeled "a
state that COMPLETED the previous level; the next level's completion likely looks structurally similar".
**Step 3:** add a GOAL-satisfiability check (evaluate the induced `is_level_complete` over the grids
`plan_in_model` visits; reject constant-False with counterexample `degenerate_goal_predicate` and refine).
**Gate:** on lp85 AND sc25, the L1→L2 reinduction produces a non-degenerate `is_level_complete` (True on
≥1 reachable grid) AND `plan_in_model` returns a non-empty plan with `reaches_goal=True` AND the GENERIC
live agent reaches L2, **offline-reproduced** (`solve_provenance: live_agent_self_discovery`).
`live_llm_inference` (Qwen GGUF). `retire_if_same_verdict: true`. DEAD-ENDS (do not re-attempt): gate-relax
(NO-OP), evidence-delay (BACKFIRE).

### Phase A2 (B1-localized): DAgger-lite distribution-shift value-routing
B1 localized the `.429 A1 value-routing null to distribution-shift (0.699, NOT calibration). Re-collect
value training data on the SEARCH-DISTRIBUTION (off-path frontier) states the live agent actually visits
(DAgger-lite, arXiv:1011.0686 / 2605.12913), re-train the win-reachability head
(`arc_value_learner.py:558` / `collect_trajectory_data:477`), keep the affordable cheap-feature routing
from `.429 A1, and re-measure live first-win + multi-level solve-rate vs the `.429 winning-path-trained
baseline. **Gate:** live first-win OR solve-rate up (bootstrap CI excludes the baseline) AND the
distribution-shift score drops. `verifier_is_oracle: false`. `retire_if_same_verdict: true`.

### Phase A3: level-up self-play (Level-Up Attempt Guarantee + self-play-every-milestone)
Bank +1 NEW reproducible level on a clean game NOT deepened in `.425-.429 (PREFER bp35/re86/sb26
first-contact L1→L2 — the genuine hard-to-bank targets; alternatives r11l/g50t/lf52/s5i5 L1→L2 or
m0r0/cn04/cd82/sp80/su15 deeper if a grounded delta exists). SKIP vc33 (.429), ft09 (.428), ls20 (.427),
sk48 (.426), dc22 (.425), ka59/wa30 (hidden-state-bound). Train+checkpoint the learned verifier on the
run's pos/neg traces. **Gate:** `offline_reproduced` new level → `reproducible_total_levels` 58 → 59+.
`solve_provenance: development_proxy`.

### Phases A4–A6, B1–B2, C, D, E
A4 refreshes the operator-resubmit package (fold A3 bank, keep >33, submission operator-only). A5 persists
the winning A1/A2 primitive + measures cross-game transfer (characterize null honestly). A6 folds the
winning config into `SUBMITTED_AGENT_CONFIG` (single source of truth, parity test green). B1 ships the
multi-level harness CI-gate (the fixed `target_levels≥2`/no-break rollout cannot silently revert to the
degenerate 0.0 metric; first-win/solve-rate floor; port-hygiene assertion). B2 extends `adversarial_verify`
with two guards for the `.430 lever class: L2-goal-induction-without-satisfiability-check (a "reached L2 via
induction" claim that omits `goal_predicate_satisfiable`+`l2_plan_reaches_goal` is flagged) and
multi-level-without-nondegenerate-metric (a multi-level claim that omits `target_levels≥2`+no-early-break
is flagged). C audits per-board reachability (KV260 SSH-only, PolarFire SSH, GateMate USB; no bitstream
build). D ingests the structural-deepening SOTA for `.431 (hierarchical-subgoal-search 2604.03208/2506.07255/2504.04366
+ PoE-World 2505.10819/2605.05138 — the fallback if A1/A2 null; reliable sweep + WebSearch only,
/deep-research BANNED). E aggregates the scorecard + the HEADLINE DECISION (did A1 or A2 cross the bridge
for live multi-level solve?), re-affirms G1–G4 `paper_ready` (FoVer 0.9131 frozen), submission operator-only.

---

## 5. Dependency graph

```
exp4663-phase0 (archive/activate)
   |
   +--> exp4664-a1 (L2-goal induction + harness fix)  ---------\
   +--> exp4665-a2 (DAgger-lite value-routing)        ----------\
   +--> exp4666-a3 (level-up bank + verifier train)   -----------+--> exp4667-a4 (refresh package)
   |                                                              +--> exp4668-a5 (persist+transfer)
   |                                                              +--> exp4669-a6 (integration)
   +--> exp4670-b1 (harness CI-gate)   [serves A1 measurement]
   +--> exp4671-b2 (adversarial_verify guards)
   +--> exp4672-c  (hardware audit)
   +--> exp4673-d  (SOTA ingestion -> .431)
        |
        v
   exp4674-e (capstone: A1/A2 bridge decision + A3 bank + G1-G4)
```

A1–A6 are independent of each other for execution (A4/A5/A6 read A1/A2/A3 artifacts but degrade
gracefully to "unchanged" if a lever nulls — no hard cross-task gate that could cascade-block). The
capstone (E) reads all upstream artifacts last.

---

## 6. Hardware requirements

| Task | Hardware | Notes |
|---|---|---|
| A1 | iGPU (Radeon 890M, ROCm) for Qwen3.5-9B-MTP | **NEVER the RTX 3090s** (frozen ARC live-generator stack). `live_llm_inference` for the induction arm only. PRECONDITION: Qwen GGUF cached + a non-colliding proposer port (port-8919 gemma confound). |
| A2–A6, B1–B2 | CPU (offline arcade, value head, aggregation) | `verifier_ensemble_against_cached_candidates` / `aggregation_from_upstream_artifacts`; no LLM load. |
| C | SSH to `kria`/`polarfire`, USB for GateMate | Reachability audit only; KV260 SSH-only (NEVER host SD-card). No bitstream build. |
| D | network (arXiv/Semantic-Scholar/WebSearch) | Reliable sweep; /deep-research BANNED in the autonomous loop. |

---

## 7. Disciplines honored

- **ARC-AGI-3 Submission Sprint Forcing Function** (through 2026-06-30): majority ARC, monotonic
  `reproducible_total_levels`, all experiments codex, planner/retro Claude Opus.
- **ARC Level-Up Attempt Guarantee:** A3 banks a new reproducible level (rotation target).
- **ARC Live-Path Reachability + Self-Solve Provenance:** every ARC task declares `solve_provenance`;
  A1 targets `live_agent_self_discovery` (the real deliverable); all levers live-path-reachable
  (`arc_orphan_solver_lint`). Registry-precheck: A3 rotation skips already-solved levels.
- **Circularity / Oracle-Distinctness:** `verifier_is_oracle: false` on every value claim.
- **Pre-Launch Preconditions:** A1 PRECONDITIONS = Qwen GGUF cached + offline arcade + importable live
  modules + non-colliding proposer port verified via `/props`.
- **Failed-Experiment Rerun + Exclusion-Manifest:** A1/A2/A3 carry `prior_failures:` with all four
  sub-fields + `retire_if_same_verdict: true`; routine tasks carry `operator_override:`.
- **Principle-Annotated Artifact Fields:** every REQUIRED ARTIFACT FIELD has a `principle:`.
- **Adversarial Artifact Verification + False-Negative-Risk:** null claims require a positive control
  (reachable headroom) + matched baseline; B2 adds the lever-class guards.
- **SOTA-Ingestion Cycle:** D reserves the ingestion slot, flags the strongest for .431.
- **Operator-Only External Publication:** A4 prepares the package; submission is operator-only.

---

## 8. Success criteria for .430

1. **HEADLINE:** A1 or A2 crosses the offline→live bridge for SOLVE-RATE/DEPTH — the GENERIC live agent
   reaches L2 on lp85 and/or sc25, offline-reproduced (`live_agent_self_discovery`). This would be the
   FIRST generic-agent multi-level solve and the decisive break in the 5-milestone null streak.
2. **CAPABILITY:** A3 banks +1 reproducible level (58 → 59+) — monotonic growth regardless of A1/A2.
3. **MEASUREMENT:** the `live_multi_level_solve_rate` harness is fixed + CI-gated (it can never again be
   a degenerate 0.0-by-construction metric).
4. **PUBLICATION INVARIANT:** G1–G4 `paper_ready` re-affirmed, FoVer 0.9131 frozen (never substituted).
5. **PACKAGE:** operator-resubmit-ready, live-submittable count > 33, submission operator-only.

If A1 AND A2 both null with passing controls: the residual is localized (B1), the structural
subgoal-search lever is flagged for .431 (D), and per the operator's strategic note first-win BREADTH
(0.59) becomes the cheaper deadline ROI — to be weighed at .431 planning. The honest-null is itself a
publishable narrowing of the multi-level-deepening problem.
