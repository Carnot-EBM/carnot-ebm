# Re-induction loop with hazard escalation — one self-deepening mechanism

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

Integrates the two prior pieces into a single loop:
- the **mechanic-conditioned re-induction trigger** (`mechanic-conditioned-reinduction-trigger-2026-06-22.md`):
  re-fit the world model at each level; a deterministic, budget-unexhausted game-over after a level-up is the
  trigger that a mechanic shifted.
- the **hazard-aware model class** (`hazard-aware-world-model-2026-06-22.md`): learns a line-charger hazard
  from death transitions and routes around it.

The wiring (`scripts/experiments/experiment_reinduction_hazard_loop.py:escalating_deepen`): at each level the
loop re-induces a **nav** model from transitions collected there and plans+executes. On a level-up it banks
and continues. On the **trigger specialised to a hazard** — a game-over where the avatar was **REMOVED** (the
nav engine only translates/blocks, so it plans straight into a hazard) — it **escalates**: it fits a
**hazard-aware** model from the level's transitions plus the trigger's own death (the nav suicidal plan's
death IS the hazard signal), and **re-plans the same level**. On any other stall (move-budget / wall) it
stops (out of the current model classes' scope). The escalation ladder is **nav → hazard-aware**, fired
purely by avatar-removal detection — no per-level or per-game hand-holding.

## Result — the loop auto-deepens tu93 L1→L2 with no hand-holding

`results/experiment_reinduction_hazard_loop.json` (5 seeds):

```
every seed:  deepest L2, reproduced L2   |   chain: L1(nav) -> L2(hazard_aware)
VERDICT: reinduction loop AUTO-ESCALATES nav -> hazard-aware at the avatar-removal trigger and deepens
         tu93 to L2, reproduced on every seed
```

L1 is solved by the **nav** model; the nav re-fit then dies at the L2 charging enemy (avatar removed); the
loop detects that, escalates to the **hazard-aware** model, and deepens to L2 — banked and **reproduced on a
fresh env**. The whole thing runs as one loop with no manual step between levels.

## Adversarial review — SURVIVES

A hostile reviewer re-ran it and traced the control flow:
- **Escalation is generic** — fired by `avatar_removed` detection (`escalating_deepen` line ~118), with no
  `level==2` / game-name branch and no hardcoded hazard colours (hazard colours/axis/range are all
  data-fitted; the `8/15` literals appear only in docstrings). The `per_level` chain confirms L1 via the nav
  model and L2 via the **escalated** hazard-aware model — not both by a pre-supplied hazard model.
- **Reproduction is genuine** — the gate makes a brand-new env and reads the env's own `levels_completed`
  (set by the real obfuscated tu93 logic on a sprite-collision win); the world model cannot set that counter,
  so `verifier_is_oracle: false` is honest, not leaking.
- **Escalation does real work** — within the loop, the nav model returns `game_over_avatar_removed` and the
  escalated hazard-aware model returns `level_up`; the escalation is structurally necessary.

## Scope (honest)

Single game (tu93), single hazard mechanic (line-charger), two levels. The loop **stalls at L3**
(`no_plan_in_model`) — L3 needs the next primitive (a further hazard type or a win-vocabulary extension), not
captured by nav + line-charger. The hazard signal is self-supplied via the nav suicidal plan (the trigger's
own death), which is fair but does not test whether unguided exploration alone would surface enough deaths.

## Forward

- The escalation ladder is the extension point: add the next model class (pursuer / proximity / multi-hazard
  hazards; non-reach-goal win vocab) as a new rung, fired by its own trigger signature, to push past L3.
- This loop is ready to drop into the standing solver as the level-boundary behaviour: re-induce, and on a
  hazard-shaped trigger, escalate before re-planning.

## Artifacts

- `scripts/experiments/experiment_reinduction_hazard_loop.py` — the integrated escalating loop
- `results/experiment_reinduction_hazard_loop.json` (adversarial-verify clean; 5/5 seeds)
- builds on `arc_nav_world_model.py` (InducedNavWorldModel + HazardAwareNavWorldModel),
  `experiment_reinduction.py` (reach-L1 + collect-at-level), `experiment_hazard_aware.py` (nav-death signal)
