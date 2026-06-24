# Lever #4 (mechanic-template) hits — and reveals — the PERCEPTION wall unifying all 5 multi-level levers (outer-loop, 2026-06-24)

Operator chose lever #4 (mechanic-template goal synthesis) after #1/#2 and the conductor's A1/A2 nulled on the
L1->L2 wall. Built it; a design panel correctly picked the largest cleanly-template-able class
(`align_to_goal_offset`: lp85, r11l, g50t, ls20) with the key idea of a SMOOTH `goal_energy` (residual-sum
descent gradient) to escape `plan_in_model`'s `no_reachable_plan`. The build is sound and verified on synthetic
data — but on REAL grids it exposes a deeper, unifying obstacle.

## The build (committed `dd2140c1c`, branch `outer-loop/multiwin-goal`)

`python/carnot/agentic/arc_align_offset_template.py`: grid-grounded `detect_align_offset_params(start, win)`
(the classifier+instantiator) + `make_align_offset_predicate(params)` → `(is_level_complete, goal_energy)`.
Verified on a synthetic align case: detects offset (1,1), `is_complete(win)=True / (start)=False`, smooth
`energy 9→0`, perturbation-robust. `verifier_is_oracle: TRUE` (execution-grounded structural predicate — NOT an
oracle-distinct moat; per the circularity discipline). `scripts/experiments/proto_align_template_gateA.py` is
the GATE-A generalization test.

## The decisive finding: the logical grid does NOT expose the structural objects the win conditions need

GATE A on real games + a grid-cleanness probe (CPU, no LLM):

| game | class | grid behavior start→win | detector |
|---|---|---|---|
| **lp85** | align | **whole board RECOLORS** — start colors {1,2,8,9,10,11,15} vanish, win gains {4,5}; no stable object identity | rejects (no stationary goal) |
| **r11l** | align (drag) | 30 moved components, **zero stationary goal colors** | rejects |
| **tu93** | reach-target | win adds NEW colored cells (a trail); **no clean player→goal moved/stationary** | rejects |
| **bp35, g50t** | reach-target | **explorer cannot reach L1 at all** | n/a |

So a grid-grounded generic detector recovers the structural condition on **none** of them. The reason is
structural, not (only) detector-naivety: **ARC-AGI-3 win conditions are defined over INTERNAL game objects**
(stable sprite tags like `bghvgbtwcb`/`goal`), and the per-game `GameAdapter`s succeed precisely by reading
that internal object state — which **does not generalize to hidden games**. The RENDERED logical grid (what
`plan_in_model`'s engine produces, and the only thing a generic agent sees on a hidden game) **loses object
identity**: lp85's rotation recolors the board, tu93 leaves a trail, r11l has no stationary goal anchor. A
generic goal predicate — template OR LLM-induced — cannot reliably reconstruct the object-relational win
condition from those pixels.

## Why this UNIFIES the five lever nulls

Every multi-level lever this week died on the same root cause, now named:
- **#1 value-biased deepening** — exploration/value can't *reach* a structured win it can't *perceive*.
- **#2 multi-positive goal induction** — canonical wins (one solved config) + the win is object-relational, not
  grid-literal.
- **conductor A1 hierarchical-subgoal** — "value head still not separating": the value head sees grids, not objects.
- **conductor A2 PoE-World factored** — "no coverage gain": factored over grid features, not objects.
- **#4 mechanic-template** — the template needs object structure the grid doesn't expose.

The L1->L2 wall (and generic ARC generalization broadly) is fundamentally a **PERCEPTION / object-identity
problem**: recovering stable structural objects (pieces, goals, player, walls) from the rendered grid, across
frames, despite recoloring/trails/rendering. This matches the project's prior independent finding that
**perception is the binding constraint** (`project_arc_live_agent_learning_gaps`: frame-only order-1 features
LOO ≈ chance). Goal-induction, search, and planning all sit *downstream* of an object-perception layer the
generic agent does not have; the per-game adapters substitute hand-RE'd internal-state access for it.

## Recommendation

Stop attacking the wall at the goal/search/planning layer — five levers confirm it is not there. The
highest-leverage redirect is the **object-perception layer**: a generic, grid-grounded module that segments the
frame into persistent OBJECTS with stable identity across transitions (track by shape/connectivity/motion, not
color — robust to recoloring), exposing object-relational features (this object at that object + offset; player
overlaps goal). With that layer, BOTH the mechanic-template (this build's `goal_energy` is ready to consume it)
AND LLM goal-induction become grid-grounded-yet-object-aware. Without it, every generic goal is fighting the
pixels. This is a bigger build than a single lever and is the honest next direction — or, for the 6/30 deadline,
first-win *breadth* remains the cheaper play while perception is a longer-horizon investment.

Cross-refs: `arc-mechanic-template-design` workflow (the class enumeration + design), `multiwin-goal-induction-2026-06-24.md`
(#2, canonical wins), `multi-level-deepening-diagnostic-2026-06-23.md` (the original diagnosis), branch
`outer-loop/multiwin-goal`, `project_arc_live_agent_learning_gaps` (memory: perception is the binding constraint).
