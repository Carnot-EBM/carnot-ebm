# ARC mechanism-parity audit + hazard-work salvage

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

## The operator's question

> "Is the mechanism we are using here in the outer loop the same mechanism that the live agent uses to find
>  solves? Otherwise, it seems to me that we are intentionally wasting the additional effort."

Asked after a multi-milestone outer-loop effort built a hazard-aware nav world model (`arc_nav_world_model`:
`InducedNavWorldModel` → `HazardAwareNavWorldModel` + `escalating_deepen` + an `omni` interception-zone
calibration + a constructed validation game) to "solve" tu93's charging-enemy levels (L2/L3).

## The answer: NO — different mechanism, and it duplicated an already-solved level

A parity audit (workflow `arc-mechanism-parity-audit`: 6 parallel code-mappers → synthesis → adversarial
refutation; verdict `different_mechanism`, refutation `synthesis_holds`) plus direct verification found:

1. **Orphaned from the live path.** `arc_nav_world_model` / `HazardAwareNavWorldModel` / `escalating_deepen`
   were imported ONLY by `scripts/experiments/*` and the unit test — by **zero** live-agent files. The import
   closures of both live entrypoints exclude them.
2. **The live mechanism already deep-solves tu93 to L3.** Running the live loop right now
   (`scripts/arc_loop_solve.py --game tu93 --target-level 3`) returns `reached_level: 3`,
   `offline_reproduced: True`, `states_expanded: 2947`, `verifier_src: hand_verifier_cold_start`. tu93 is a
   registered `GameAdapter` (`_tu93`) solved by plain verifier-routed best-first search over the 4 nav
   actions + a player→goal Manhattan-distance verifier. It models no charger: walking into one is just a
   dead-end the search prunes. The registry records tu93 at `levels_reproduced: 5`.
3. **The calibration didn't transfer.** The `omni` lethal rung's correctness was established by an
   EXHAUSTIVE position-keyed real-env BFS over tu93 L3 ground-truth labels — which a hidden-game agent
   cannot run under an action budget.

So the hazard-world-model line produced no live capability (the level was already solved) via a mechanism
the live agent never calls. The operator's concern was justified.

## The two live entrypoints (what "the live agent" means)

| Entrypoint | Role | Mechanism |
|---|---|---|
| `arc_competition_agent.py` (`make_carnot_agent` → `E3AgentPolicy`) | the **SCORED** hidden-game agent | per-action verifier-routed cascade over its OWN transitions: StepwiseExplorer → online world-model induction (`arc_live_ttt` / `LocalGGUFProposer`, gated by `WorldModelVerifier`) → `e3.plan_in_model` |
| `scripts/arc_loop_solve.py` (`OfflineSolver` + `GameAdapter`) | offline development twin (registry/dev) | raw-action verifier-routed best-first search replayed in the offline sim, reproduction-gated |

## The salvage: hazard work as a live-path efficiency feature (not a parallel solver)

The hazard model's one transferable value is EFFICIENCY: the blind search wastes expansions walking into
chargers. `arc_hazard_pruner.HazardMovePruner` turns it into a **move-pruner the live `OfflineSolver`
consumes**:

- Fits a hazard model from the search's **OWN observed death transitions** — no offline ground-truth BFS.
- Selects the `toward`/`omni` rung by **in-sample observed-transition trust**: which rung's `is_lethal`
  best classifies the deaths/safes the agent actually observed (scored on those same transitions, so
  optimistic — used as a conservative gate, not a guarantee). This removes the non-transferable
  BFS-calibration dependency; the reproduction gate remains the correctness backstop.
- Only prunes when the rung clears a conservative trust + specificity bar (a false-positive prune would
  forbid a safe move and could break the solve). **No-ops when no hazard is detected**, so it is safe for
  any game.

**Measured A/B on tu93 L3** (`results/arc_hazard_prune_ab_tu93.json`):

| run | states_expanded | reached | reproduced | pruner |
|---|---|---|---|---|
| baseline (`--no-hazard-prune`) | 2947 | L3 | True | — |
| hazard-pruned | **2859** | L3 | True | online `omni`, trust 1.0, specificity 1.0, **88 pruned / 6 deaths** |

The reduction (88 states) **equals the pruned-move count** — a clean causal demonstration — and the L3 solve
is **preserved** (reproduced, specificity 1.0 = no false-positive prunes). Honest magnitude: **~3%**. The
pruner only saves death-move expansions; most of tu93's best-first search is non-lethal exploration it cannot
reduce. The gain would be larger on a denser-hazard game. The natural next step is to feed the same
pruner / `HazardAwareNavWorldModel.engine` into the SCORED `E3AgentPolicy.plan_in_model` path.

This makes `arc_nav_world_model` **reachable from the live path** (`arc_loop_solve` → `arc_hazard_pruner` →
`arc_nav_world_model`), so the work now flows to the agent that plays hidden games.

## The discipline + lint (so this can't recur)

- **CLAUDE.md "ARC Live-Path Reachability Discipline" (MANDATORY).** (a) Registry-precheck before any per-game
  RE / world-model solver work — if the live mechanism already reaches the target level, improve the live
  path, don't build a parallel solver. (b) Every ARC solver/world-model module must be reachable from a live
  entrypoint, or explicitly allow-listed.
- **`scripts/arc_orphan_solver_lint.py`** (pre-commit `arc-orphan-solver-lint`): computes the live import
  closure and refuses a commit that leaves a solver-like `arc_*` module orphaned. It surfaced two
  pre-existing offline prototypes (`arc_execution_guided_world_model`, `arc_world_model_synth`) now
  allow-listed with reasons and flagged for future review.

## Honest scope

The hazard model is good science; the failure was process — it was a parallel mechanism the live agent never
ran, against a level the live agent already solved, because the registry/adaptered-level precheck was skipped.
The salvage is efficiency-only and modest (3% on tu93), but it is now on the live path and the lint guards
against the next orphan.

## Artifacts

- `python/carnot/agentic/arc_hazard_pruner.py` — the live-path hazard move-pruner (+ `tests/python/test_arc_hazard_pruner.py`)
- `python/carnot/agentic/arc_solver_kit.py` — `OfflineSolver` `move_pruner` hook (all branch modes)
- `scripts/arc_loop_solve.py` — `--hazard-prune` (default on) + recorded `hazard_pruner_stats`
- `scripts/arc_orphan_solver_lint.py` + `.pre-commit-config.yaml` — the reachability lint
- `results/arc_hazard_prune_ab_tu93.json` — the states-expanded A/B
- CLAUDE.md "ARC Live-Path Reachability Discipline"
