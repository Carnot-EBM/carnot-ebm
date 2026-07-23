# ARC-AGI-3 lever triangulation — where the binding constraint actually is (2026-07-23)

**Status:** RESEARCH SYNTHESIS. No live-path file modified by this note itself. Consolidates a 5-lens
diagnostic triangulation (workflow `wqn31sxaz`, 5 parallel Opus agents, read-only over existing artifacts)
run after a session that tested/fixed five ARC mechanisms and found them all null on the scored path.
Every load-bearing null below was spot-checked against its result artifact's `honest_verdict` (not taken
from the agent summaries alone).

**Reads as required input (per CLAUDE.md):** "ARC-AGI-3 IS a Live Hidden-Game Discovery Agent",
"ARC Live-Path Reachability Discipline", "Failed-Experiment Rerun Discipline", "Adversarial Artifact
Verification".

## The one-paragraph conclusion

Two whole classes of lever are now **confirmed-dead on real games**, and the binding constraint is pinned:

- **Candidate RANKING / selection is dead** — 7-9 distinct re-ranking A/Bs (tier-schedule, small-object-first,
  learned candidate-router, spatial value-head, frame-change scorer, imitation/object-history action-prior,
  structural-energy rerank) each moved **zero live levels**. `ops/verifier_gaps.md` already states "STATIC
  PERCEPTUAL click ordering is not the binding constraint." Re-proposing a ranking build violates the
  Failed-Experiment Rerun Discipline.
- **Search DEPTH / lookahead is dead** — the 2026-07-20 winner audit found all three Milestone-1 winners are
  *greedy single-commit* generators with **no** tree/beam/MCTS and no subgoal decomposition; Carnot already
  has strictly *more* search than any of them. And candidate-coverage attribution finds bucket-b (in-set
  no-op that only pays off downstream = the lookahead signal) **== 0 exactly across both independent runs**
  (0 of 247 winning-path actions, 12 games).
- **The binding constraint is world-model-induction-grade SEQUENCE routing** — composing a correct
  13-33-action ordering out of individually-available, individually-frame-changing candidates. This needs
  either a runtime-induced world model accurate enough for the (already-superior) search, or a goal-progress
  gradient for it to climb. A 9B local model produces neither (induce-completion ~0/12 on the stalled
  roster). **And this is NOT merely a "16GB forces a 9B" problem:** a 31B generator swapped in *offline*
  (constraint relaxed) moved **0 live levels** (`experiment_5722`, `delta_0.0 floor_persists`) with held-out
  induction still only 0.378 (`experiment_5764`) — far below the near-1.0 dynamics accuracy a 14-step plan
  needs (0.378^14 ~ 0). This is at or near a genuine capability frontier, not an obvious engineering fix.

## The evidence chain (verified against artifacts)

| Claim | Artifact / source | Verified value |
|---|---|---|
| Re-ranking moves 0 levels (small-object-first, built to fix the exact click-rank gap) | `results/experiment_5758_click_ranking_fix_ab.json` | `any_config_beats_baseline_levels=False`; r11l/su15 `fixed:False` |
| Tier-schedule rerank null | `results/proto_tier_ab.json` | `TIER_NULL_no_win` |
| Learned candidate-router null (never changed order live) | `results/experiment_4556_verifier_router_generic_transfer.json` | `verifier_router_no_value_added_honest_null` |
| Spatial value-head null | `results/experiment_4617_graduate_spatial_value_head_live.json` | `spatial_value_head_graduated_no_live_value` |
| Object-history action-prior null ("ranking is not the binding constraint here") | `results/experiment_5740_object_history_salience_11game_ab.json` | `any_config_beats_baseline_levels=false` |
| Bigger (31B) generator swapped offline moves 0 live levels | `results/experiment_5722_generator_swap_gemma31_ab.json` | `floor_persists_stronger_generator_no_movement_delta_0.0` |
| 31B induction still far below planning-grade | `results/experiment_5764_gemma31b_singleshot_induction_ab.json` | pooled held-out 0.378 (vs 27B 0.188) |
| Lookahead signal empty (both runs) | `experiment_5757_...` + `outer_loop_arc_candidate_coverage_attribution_20260723...` | bucket-b `0/92` and `0/155` = `0/247` |
| Real scored path near-total failure | `results/arc_live_oracle_gap.json` | live 4 / oracle 183 levels |

## Honest caveats (do not overclaim)

1. **bucket-b == 0 is measured only on KNOWN winning paths** (efficient adapter solutions, where dead-in-
   isolation setup moves are nearly excluded by construction). It does NOT, by itself, prove a lookahead
   planner would not help the LIVE agent *discover* an unknown sequence. The earlier "b==0 refutes search"
   phrasing over-reaches. The verdict survives only because both readings converge: any planner is starved
   by the 0/12 induction failure, so induction+goal-signal is the binding upstream root either way (the
   project's own 2026-07-22 known-issues note already corrected to this).
2. **The whole candidate-coverage attribution is a public-game, adapter-trajectory PROXY.** It measures "is
   the *known adapter's* winning action in the candidate set" — necessary-not-sufficient for *discovering* an
   unknown path with no adapter. Public-game perception/affordance conventions are, by the benchmark's
   design, non-transferable to hidden games. So coverage/states_expanded are proxies, not the scored metric
   (hidden-game levels).
3. **The 23.23% "selection-miss" headline from this session's run is ~2x inflated** by a stingy top-3 rank
   threshold vs 5757's top-12. Threshold-robust selection is ~11.6%, and it is *monochromatically a
   click-ranking problem* (winning clicks at ranks 26-43 of ~52) — the identical, already-nulled signal, not
   a fresh lever. (Several bp35 "selection" misses are actually `exact_match_rank=None` — a generation gap,
   not selection.)

## What genuinely survives as deliverable-relevant

1. **The class-level perception bug** (`GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS`): the LIVE candidate
   generator (`object_centric_digest` -> `rich_action_candidates`, reachable from the scored
   `E3AgentPolicy._candidates`) **excludes the single most-common color wholesale as background**, so it can
   never propose a click on the dominant color. This is a real live-path defect that would bite ANY hidden
   game whose winning interaction is on the dominant-background color — NOT bp35-specific, if reframed at the
   class level. (The bp35-specific 8px-tile fix shipped this session is opt-in/default-off and, on its own,
   is per-game proxy-polishing until validated by a LIVE held-out *discovery* test, not coverage.)
2. **The two dead-lever negatives themselves** (ranking dead; search-depth dead) are portable, method-level
   results that correctly steer future effort away from those wells.

## Candidate next levers (ranked, with honest risk)

1. **[cheapest, tests the real metric] Validate the perception survivor LIVE, not by coverage.** Wire the
   grid-fallback flag into the scored `E3AgentPolicy._candidates` path (opt-in), then run the
   Generalization-Testing-Floor protocol: E3AgentPolicy on a dominant-background game with its GameAdapter
   DISABLED, flag on vs off, measured by **levels actually DISCOVERED** (>0), not candidate coverage. If it
   helps live discovery, that justifies generalizing + wiring it on. If it nulls, we've cheaply learned the
   perception fix is also proxy-polishing.
2. **[the diagnosis's "next lever", big, may null] Offline-distilled cross-game goal-progress signal.** Train
   a goal-progress value/router BIG offline on the 3090s (goal-progress = predicted level-progress), distill
   to run at 9B live-inference cost (decoupling the needed capacity from the 16GB live limit), wire as the
   ordering gradient for the EXISTING StepwiseExplorer/`plan_in_model` search (closes
   `GAP-ARCH-GOAL-NOT-VERIFIED`). **Mandatory:** a pre-registered hidden/leave-one-game-out transfer gate up
   front. Honest risk: it is partly circular with induction (a goal-progress signal keyed on "progress under
   a world model" needs the world model that is 0/12), and prior learned signals (candidate_router,
   value_head) already nulled — this may too. It is the best-supported *untested, constraint-compatible*
   lever, not a guaranteed fix.
3. **[strategic] SOTA scan on HIDDEN-game discovery.** Even the winners are <0.4% on the hidden set. Before a
   big goal-progress build, a focused scan of what (if anything) actually transfers to hidden-game discovery
   would tell us whether the goal-progress lever is worth the spend or a different paradigm is needed.

## Cross-references

- Workflow `wqn31sxaz` (this triangulation) — 5 agents: threshold-reconcile, selection-prior-art,
  binding-constraint, bp35-solve-impact, adversarial-framing.
- `docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md` — the winner audit.
- `ops/verifier_gaps.md`: `GAP-ARC-CLICK-SELECTION-5758` (ranking dead), `GAP-ARCH-NO-HIERARCHICAL-SEARCH`
  (search dead), `GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS` (the perception survivor),
  `GAP-ARCH-GOAL-NOT-VERIFIED` (the goal-progress lever).
- `ops/known-issues.md` 2026-07-22/23 entries (the five null mechanisms + the bp35 fix).
- `results/experiment_5757_candidate_coverage_attribution.json` +
  `results/outer_loop_arc_candidate_coverage_attribution_20260723.json` (the two attribution runs).
