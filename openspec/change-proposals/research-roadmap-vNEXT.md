# Research Roadmap — Milestone 2026.06.428

**Headline:** ENERGY DRIVES GENERATION — wire `exp4020`'s graded `is_goal`
as a LIVE goal-ENERGY heuristic (operator menu #1; closes
GAP-ARCH-GOAL-NOT-VERIFIED), on top of the .427-PROVEN action-effect
predictor, and deepen that proven predictor from a candidate RANKER to a
search EXPANSION PRIOR. .427 proved generation levers cross the
offline→live bridge; .428 takes the operator's explicit cheapest-first
energy lever that has been unwired for three milestones.

**Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-23.
**Milestone doc for:** `research-roadmap-next.yaml` (activate as the .428 roadmap).
**Sprint:** ARC-AGI-3 submission sprint, 7 days to the ARC Prize Milestone #1
deadline (2026-06-30).

---

## 1. What the previous milestone (.427) PROVED

.427 was the milestone where the offline→live bridge was finally crossed —
on EFFICIENCY, by a GENERATION lever, exactly as the operator's repeated
diagnosis predicted.

| Phase | Lever | Result | Reading |
|---|---|---|---|
| A1 (headline) | dense curiosity / learning-progress exploration bonus (Curiosity-Critic arXiv:2604.18701) | **NULL on solve-rate** (loop 0.04 == bare 0.04), but **+2 state-coverage** | Reranking-for-solve-rate nulled a THIRD time (.425 linear, .426 SpatialValueNet, .427 curiosity). Coverage rose but did not convert — raw surprise explores, it does not solve. |
| A2 | graduate a self-supervised CNN action-effect / frame-change predictor into the live explorer's candidate ranking (the StochasticGoose leaderboard-leader steal) | **WIN** — median actions-to-first-levelup **2 → 1**, first-win-rate **0.407 → 0.591 (+0.18)**, efficiency-term 1.0, solve-rate preserved, live-path-reachable, parity green | **The FIRST live lift across the bridge.** A GENERATION/action-pruner lever — it makes the agent ACT on frame-changing (effective) actions, so it reaches first-win more often. |
| A3 | self-play bank +1 (ls20 L1→L2) | **WIN** — `reproducible_total_levels` 55 → **56** | The standing self-play loop banks a level every milestone. |
| A4 | refresh submission package | live_submittable **56**, ready_for_operator_submit, beats the standing 33 | Package stays operator-resubmit-ready. |
| A5 | persist + transfer the winning primitive | **WIN** — the action-effect ranker (PersistentAEM) transfers: cd82 first-win **+0.5**, sp80 value-added | The proven lever GENERALIZES across games it was not tuned on. |
| A6 | integrate into SUBMITTED_AGENT_CONFIG | action-efficiency shipped; the integration artifact was **flagged** (a `live_solve_rate_bare == integrated == 0.04` TAUTOLOGY false-positive, correctly quarantined) | The efficiency win is in the config; the solve-rate equality was a true null, not a fabrication. |
| Capstone | the .427 scorecard | **`success: bridge_crossed_live_efficiency_up_1`** | The headline finding. |

**The decisive, now-triply-confirmed lesson:** *GENERATION levers cross the
live bridge; RERANKING levers do not.* The wall is **make-a-winner-appear**,
not **select-the-winner**. The action-effect predictor wins because it shapes
what the agent generates (it acts on effective actions); value-head rerankers
do not generate the winning candidate, so they null.

**The remaining wall (the .428 target):** the agent now reaches the FIRST
level-up at **0.59** live, but LIVE SOLVE-RATE (≥2 levels) is still **0.04**,
and the live submission is **33** vs **56** offline. Efficiency/first-win
crossed the bridge; multi-level SOLVE has not. The gap from "reach first
level-up" to "solve multiple levels live" is the new frontier.

---

## 2. The .428 strategic pivot — ENERGY DRIVES GENERATION

Two SOTA-grounded, operator-mandated generation levers, both built ON the
.427 proven action-effect predictor (which stays ON in the config):

1. **Operator menu #1 — wire `exp4020`'s graded `is_goal` as a LIVE goal-ENERGY
   heuristic (HEADLINE, A1; closes GAP-ARCH-GOAL-NOT-VERIFIED).** The operator's
   2026-06-22 directive ("work energy judgement into the live agent so it can
   refine and embrace an energy config space ... to provide guidance to the
   agent loops as it tries to tackle each game level iteratively") names this as
   the **cheapest-first #1** lever — and it has been UNWIRED for three
   milestones while the loop chased value-head rerankers. The concrete wire
   (energy-config-space note §#1): make the `graph_explore_solve_v2` search
   heuristic a convex combination of `arc_goal_distance` (navigation energy) AND
   a **graded** goal-satisfaction energy compiled from `exp4020`'s induced
   predicate (fraction of target-groups satisfied, not the binary
   `unsatisfied_targets == 0`); emit a plan to the pool only when the predicate
   fires. This is **generation** — the `is_goal` energy is the per-game well the
   rollout descends, it DIRECTS what the search expands toward, and it is
   **oracle-distinct** (it predicts the win from VISIBLE state at held-out
   precision 1.0, it never reads the env's win counter → `verifier_is_oracle:
   false`). It instantiates the project's core thesis — *the energy function is
   ground truth* — as the LIVE generation driver, on the SCORED agent.

2. **Deepen the .427 winner — action-effect predictor as a search EXPANSION PRIOR
   (A2; SOTA-D #3 `clickability_action_effect_expansion_prior`, arXiv:2601.10904
   + arXiv:2603.24621).** .427 used the predictor to rank the immediate candidate
   actions. A2 graduates it from a candidate RANKER to a search EXPANSION PRIOR —
   predicted frame-change prioritizes which BRANCHES the best-first search
   expands, directing generation toward effective action SEQUENCES (not just the
   next action). The test: does this convert the +0.18 first-win lift into a
   deeper LIVE solve (a second live level-up)? This is the lowest-risk path to a
   second live win because it extends the only lever that has crossed the bridge.

The two levers compose: the **goal-energy** says WHERE to go (descend toward the
goal well); the **action-effect expansion prior** says WHICH effective branches
to expand to get there. Both are oracle-distinct generation signals on the
SCORED `E3AgentPolicy`, both measured against matched controls with bootstrap
CIs and the energy-config-mandated **uniform-energy ablation control** (an
energy win must beat a uniform/random-energy baseline, else it is the search,
not the energy, doing the work).

**SOTA-ingestion (.427 D) flagged these for .428** — the plan honors them:
`clickability_action_effect_expansion_prior` (→ A2),
`curiosity_critic_learning_progress_dense_reward` /
`noisy_tv_aware_action_effect_uncertainty_gate` (the A1 curiosity refinement;
carried as the next-wave lever in D), and
`graph_executable_world_model_action_effect_planner` (energy-config menu #2/#4,
the .429 generator candidate).

---

## 3. Architecture (where the levers attach)

```
                 ARC-AGI-3 hidden game (live, scored)
                              |
        +---------------------+----------------------+
        |           E3AgentPolicy  (the SCORED agent) |
        |   arc_competition_agent.py                  |
        |   +--------------------------------------+  |
        |   | StepwiseExplorer / graph_explore     |  |
        |   |  rich_action_candidates -------------+--+-- A2: action-effect
        |   |   |                                  |  |    EXPANSION PRIOR
        |   |   v best-first search heuristic -----+--+-- A1: graded is_goal
        |   |      = a*arc_goal_distance           |  |    goal-ENERGY well
        |   |      + b*graded is_goal energy  <-----+--+--  (exp4020, prec 1.0,
        |   |   |     (emit plan iff predicate fires)  |    oracle-distinct)
        |   |   v                                  |  |
        |   |  online world-model (arc_live_ttt)   |  |
        |   |  WorldModelVerifier gate             |  |
        |   +--------------------------------------+  |
        +---------------------------------------------+
                              |
            reproduction gate (arc_solver_kit.reproduce)
                              |
        ops/arc_solve_registry.yaml   (reproducible_total_levels 56 -> 57+)
                              |
   SUBMITTED_AGENT_CONFIG (single source of truth; parity-tested; orphan-lint)
```

Both new modules MUST be in the live import closure
(`arc_graph_explore` / `E3AgentPolicy` / `arc_loop_solve`) — enforced by
`scripts/arc_orphan_solver_lint.py` (ARC Live-Path Reachability Discipline). No
orphaned `scripts/experiments` solvers. No 3090s; any LLM arm runs on the iGPU
Qwen3.5-9B-MTP and declares `live_llm_inference` for that arm only.

---

## 4. Phases (12 tasks)

| ID | Phase | Track | What | agent |
|---|---|---|---|---|
| exp4639 | PHASE 0 | transition | archive .427 -> activate .428; record the true .427 close-state (`bridge_crossed_live_efficiency_up_1`) | codex |
| exp4640 | A1 (HEADLINE) | arc-north-star | wire graded `is_goal` goal-ENERGY into the live `graph_explore_solve_v2` heuristic (operator menu #1, GAP-ARCH-GOAL-NOT-VERIFIED); uniform-energy ablation control; measure live solve-rate + actions-to-win vs nav-only/action-effect baseline | codex |
| exp4641 | A2 | arc-north-star | graduate the .427 action-effect predictor from candidate RANKER -> search EXPANSION PRIOR; measure deeper live solve / 2nd level-up vs the .427 ranker-only baseline | codex |
| exp4642 | A3 | arc-north-star | self-play bank +1 NEW reproducible level (rotate to a clean game NOT deepened in .422-.427; 56->57+); train+checkpoint the learned verifier | codex |
| exp4643 | A4 | arc-north-star | refresh the operator-resubmit package (live_submittable stays > 33); operator-only | codex |
| exp4644 | A5 | arc-north-star | persist the winning primitive (A1 goal-energy operator / A2 expansion-prior) + cross-game transfer | codex |
| exp4645 | A6 | arc-north-star | integrate winners into SUBMITTED_AGENT_CONFIG; re-measure; parity + orphan-lint green; avoid the .427 solve-rate TAUTOLOGY | codex |
| exp4646 | B1 | infra | canonical `live_multi_level_solve_rate` co-headline metric (the new wall: >=2-level live solves vs first-win 0.59) + asserting tests | codex |
| exp4647 | B2 | infra | adversarial_verify hardening: GOAL-ENERGY-WITHOUT-ABLATION-CONTROL guard (an energy-driven-generation win must beat a uniform-energy control) + asserting tests | codex |
| exp4648 | C | hardware | per-board reachability audit (KV260 SSH-only, GateMate, PolarFire) | codex |
| exp4649 | D | sota-ingestion | ingest energy-as-fitness QD evolution / macro-action vocabulary / hierarchical-search SOTA (menu #2/#6, the .429 generator) | codex |
| exp4650 | E (CAPSTONE) | capstone | the .428 scorecard: did goal-energy (A1) / the expansion prior (A2) raise live solve-rate? A3 bank? package > 33? all co-headline metrics | codex |

**Reserved-slot compliance:** majority ARC (A1-A6); >=1 level-up BANK attempt
(A3, ARC Level-Up Attempt Guarantee); 2 reserved infra (B1/B2); 1 per-board
hardware (C); 1 SOTA-ingestion (D). All experiments `codex`/`gpt-5.5`;
planner/retro stay Claude Opus (sprint routing). Submission stays operator-only.

---

## 5. Dependency graph

```
exp4639 (phase0)
   |
   +-- exp4640 (A1 goal-energy)  -+
   +-- exp4641 (A2 expansion)     +- independent generation levers (parallel)
   +-- exp4642 (A3 bank)         -+   (A3 may USE A1/A2 routing but is independent)
   |
   +-- exp4643 (A4 package)   <- folds A3 bank + any A1/A2 new variant
   +-- exp4644 (A5 transfer)  <- persists the A1/A2 winner
   +-- exp4645 (A6 integrate) <- ships A1/A2 winners into SUBMITTED_AGENT_CONFIG
   |
   +-- exp4646 (B1 metric)    <- reads A1/A2 for live_multi_level_solve_rate
   +-- exp4647 (B2 guard)     <- edits adversarial_verify.py only
   +-- exp4648 (C hardware)   <- independent
   +-- exp4649 (D sota)       <- independent; feeds the .429 roadmap
   |
   +-- exp4650 (E capstone)   <- aggregates A1-A6 + B1/B2 (skips flagged)
```

A3/A4 satisfy the level-up + score guarantees independently of A1/A2, so the
milestone advances `reproducible_total_levels` and keeps the package
resubmit-ready even if both generation levers null.

---

## 6. Hardware requirements

- **iGPU (Radeon 890M)** for any LLM arm (Qwen3.5-9B-MTP); **NEVER the 3090s**
  (frozen live-generator stack, sprint rule).
- ARC experiments are offline-arcade / verifier-scoring (CPU + small conv net);
  `inference_substrate: verifier_ensemble_against_cached_candidates`.
- Phase C audits the three attached FPGA boards (KV260 SSH-reachability only,
  GateMate USB detect, PolarFire SSH) per Hardware-Task Continuity Discipline;
  KV260 is near-terminal (north-star §3).

---

## 7. Success metrics (co-headline, reported every milestone)

- **live solve-rate** on the SCORED agent vs bare (did energy-driven generation
  cross the bridge for SOLVE, not just efficiency) — the HEADLINE.
- **live_multi_level_solve_rate** (B1, new) — fraction of live attempts solving
  >=2 levels; the new wall (first-win is 0.59, multi-level is the gap).
- **live_action_efficiency** = `min(human/agent,1)^2` (the leaderboard score
  term; A2 expansion prior should hold/raise it).
- **first-win-rate** on the SCORED agent (0.591 baseline; must not regress).
- **offline_to_live_transfer_ratio** (the bridge co-metric).
- **reproducible_total_levels** (registry; A3 bank, 56 -> 57+).
- **live_submittable_level_count** (A4; must stay > 33).

Every value claim carries `verifier_is_oracle: false`. A goal-energy win must
clear the **uniform-energy ablation control** (B2 guard). Submission to the
official leaderboard remains **operator-only**.

---

## 8. Cross-references

- `ops/north-star.md` §0 (ARC-AGI-3 north star), §5 (verifier-as-action-pruner)
- `ops/arc_solve_registry.yaml` (`reproducible_total_levels: 56`)
- memory `project_arc_energy_config_space` (operator 2026-06-22: energy DRIVES generation; cheapest-first menu #1 = wire `exp4020` goal-energy)
- memory `project_arc_live_agent_learning_gaps` (the bridge re-diagnosis; the defensible ordering)
- `docs/research-notes/arc-generation-wall-energy-config-space-2026-06-22.md` (menu #1/#2/#6, real arXiv IDs)
- `docs/research-notes/intrinsic-motivation-action-effect-literature-2026-06-23.md` (.427 D ingestion; the four flagged methods)
- `ops/verifier_gaps.md` GAP-ARCH-GOAL-NOT-VERIFIED (A1 closes it), GAP-ARCH-FRAME-CHANGE-PREDICTOR (A2 deepens)
- `results/experiment_4638_capstone_v427.json` (`bridge_crossed_live_efficiency_up_1`)
- CLAUDE.md: ARC-AGI-3 Submission Sprint Forcing Function, ARC Level-Up Attempt Guarantee, ARC Live-Path Reachability Discipline, Circularity/Oracle-Distinctness Discipline
