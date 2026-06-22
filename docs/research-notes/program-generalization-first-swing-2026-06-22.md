# Program-generalization (Executable World Models) — first swing

Date: 2026-06-22 · Outer-loop (Claude as proposer) · OFFLINE, zero quota · `verifier_is_oracle: false`

## Why

The ARC-AGI-3 leaderboard leader (Executable World Models, arXiv:2605.05138, RHAE ~58%) DEEPENS not by
stumbling into deeper levels via real-env search, but by inducing an executable `transition + goal` MODEL
once and PLANNING IN IMAGINATION. Our prior deepening attempts were bounded a different way: the learned
value (`experiment_value_q_head` v5/v6) was GRADIENT/seed-bound — it could route L1 7.6x faster but never
reached an L2 state to learn from. The operator asked for a first swing at program-generalization. The
question: does the leader's lever work on OUR stack, and where does it wall?

We did NOT reinvent — we reused the existing framework
(`python/carnot/agentic/arc_executable_world_model.py`: `collect_transitions` → `WorldModelVerifier`
(exact + changed-cell-recall) → `plan_in_model` BFS-in-imagination). The Carnot verifier is the moat; the
planner is generic BFS. Harness: `scripts/experiments/experiment_program_gen.py`.

## What we measured

Three games, two arms:

| Game | Model | re-verify exact / cell-recall | imagination reached | reproduced | finding |
|---|---|---|---|---|---|
| **ka59** | existing E3 (genuine logic engine) | 0.19 / 0.43 | L0 (0 actions) | L0 | engine too noisy for BFS to plan even L1 |
| **sc25** | existing E3 (`PATCH_BY_KEY` table) | 0.41 / **0.06** | L0 (1 action) | L0 | a memorized replay table, not a generalizing model |
| **tu93** | **hand-induced** (this swing) | 0.00 / 0.32* | **L1 (18-action imagination plan)** | **L1 (fresh-env)** | lever works at L1; L2 mechanic differs (model doesn't generalize) |

\* tu93 cell-recall is dragged down by the unmodeled move-counter strip (a 1-cell tick per blocked move);
on the avatar-MOVE transitions the hand-induced engine is **100% accurate** (99/99 moves, 101/101 blocks,
0 false-blocks — measured directly).

## The hand-induced tu93 model (the positive control)

tu93 is clean 4-direction maze nav (`results/arc_e3/tu93/world_model_nav.py`, reverse-engineered from
offline transitions): avatar = 3x3 colour-9 block + colour-4 centre; each ACTION translates it exactly 6px
(1=up,2=down,3=left,4=right); colour 5 = impassable wall, colour 2 = passable doorway, colour 0 = open
room, colour 14 = goal; a move is allowed iff the swept 3x3 mid-gap is the colour-2 door. Win = avatar
covers the colour-14 goal. This transition+goal is LEVEL-INVARIANT by construction — exactly the property
the leader exploits.

With that faithful model, `plan_in_model` planned an 18-action path to the L1 goal ENTIRELY IN
IMAGINATION (zero real actions spent searching), executed it in the real env, and **leveled up — verified
by the fresh-env reproduction gate.** The lever is real: a verified world model lets us plan a level we
never searched for in the real env.

## Where it walls: the L2 MECHANIC DIFFERS (model-generalization failure), not the planner or local fidelity

> **Correction (2026-06-22, post adversarial review).** An earlier version of this note claimed the L2 wall
> was tu93's "non-idempotent-reset hidden parity." A hostile review read the actual `tu93.py` env source and
> refuted it; an independent re-measurement confirmed the refutation. The corrected finding is below.

Deepening to L2 stalls. The model planned an 8-action L2 path; execution matched reality move-for-move for
the first steps and then the env hit **game-over at a DETERMINISTIC step** (step 3 in **4/4** fresh-env
trials), with the **move budget unexhausted** (50 steps, only ~3 used — and the env source shows the
counter near-full at the stall). Determinism *rules out* the non-idempotent-reset parity gotcha (#7), which
produces *run-dependent* outcomes; budget-fullness rules out move-exhaustion. The real cause, confirmed
against the env source: **L2 introduces a different move mechanic** — new colours plus a sprite
pixel-buffer move-validation and a multi-phase rotation state machine that calls `lose()` on an invalid
arrangement (`tu93.py`). The L1-induced blocking rule (colour-5 in the swept gap) is not L2's rule, so BFS
plans a move that is fatal under L2.

So the wall here is **NOT** the value approach's gradient wall, and NOT hidden state — it is a clean
**model-generalization failure: tu93's levels are not mechanically identical.** This is the precise cost the
Executable-World-Models lever pays: a single induction deepens *only across levels that share the mechanic*;
when the mechanic shifts, the model must be re-induced. The bottleneck for program-generalization here is
**per-level mechanic re-induction (detecting the shift and re-fitting)**, NOT the planner (BFS found the
plan) and NOT local L1 fidelity (100% accurate on L1 moves).

## Forward levers (for the energy-config-space direction)

1. **Detect the level-boundary mechanic shift and RE-INDUCE.** The model is correct within a mechanic and
   walls at the boundary. The deepening loop should treat a deterministic, budget-unexhausted env game-over
   after a level-up as a *re-induction trigger* (collect fresh L2 transitions, re-fit), exactly as the
   leader pays per mechanic. A divergence detector (predicted ≠ observed on a non-fatal step) is the cheap
   trip-wire.
2. **The energy/config space should be MECHANIC-CONDITIONED, not global.** The operator's energy-config-space
   directive — refine an energy over each game's config space — must allow the energy/transition to switch
   when the level's mechanic switches (tu93 L1 nav ≠ L2 rotation-validated nav), rather than assuming one
   model spans all levels.
3. **Induction fidelity gates everything.** The two pre-existing E3 models could not plan even L1 (ka59 too
   noisy, sc25 memorized). A faithful model is the precondition; our local/codex inductions don't yet reach
   it on these games, while a careful hand-induction does. This is the concrete target for the local-GGUF
   proposer.
4. **(Separately) the non-idempotent-reset parity is real but is a DIFFERENT problem** — it bit the *real*
   solver's reuse-one-env branch search (registry gotcha #7, fixed via fresh-env-per-candidate). It is not
   what walls this imagination-planning L2 attempt.

## Artifacts

- `results/experiment_program_gen_tu93.json` — positive control (L1 reproduced, hidden-state-bound deepening)
- `results/experiment_program_gen_ka59.json`, `..._sc25.json` — existing-model contrast (no generalization)
- `results/arc_e3/tu93/world_model_nav.py` — the hand-induced faithful nav world model
- `scripts/experiments/experiment_program_gen.py` — the harness (all 3 adversarial-verify clean)
