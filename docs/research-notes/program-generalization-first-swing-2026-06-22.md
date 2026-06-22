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
| **tu93** | **hand-induced** (this swing) | 0.00 / 0.32* | **L1 (18-action imagination plan)** | **L1 (fresh-env)** | lever works at L1; deepening is HIDDEN-STATE bound |

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

## Where it walls: HIDDEN ENV STATE, not the planner or local fidelity

Deepening to L2 stalls. The model planned an 8-action L2 path; execution matched reality move-for-move for
3–4 steps and then the env hit **game-over** — at a step that is RUN-DEPENDENT (step 3 "after match" one
run, step 4 "after divergence" the next). That run-to-run variation IS the signature of tu93's documented
**non-idempotent-reset hidden parity** (registry `tu93` gotcha #7: "env.reset() leaves a parity-toggling
hidden state"). A pure grid→grid world model cannot represent that parity, so an L2 plan computed from the
visible grid walks into a parity-contingent game-over.

This is the same CLASS of wall the value approach hit, named precisely: **deepening on this game family is
HIDDEN-STATE bound.** tu93 is in the wa30/ls20 family (registry), and `experiment_8` independently found
wa30 ~53% observed-state nondeterminism. The bottleneck for program-generalization here is modeling the
hidden state, NOT the planner (BFS found the plan) and NOT local transition fidelity (100% accurate).

## Forward levers (for the energy-config-space direction)

1. **The energy/config state must include LATENT env state, not just the visible grid.** The operator's
   energy-config-space directive — refine an energy over each game's config space — must carry the hidden
   parity/budget dimension for the wa30/ls20/tu93 family, or imagination-planning diverges at depth.
2. **Fresh-env-per-candidate branch search in imagination** (the registry's own fix that got tu93 L3 to
   reproduce) is the planner-side workaround: don't reuse one env across the deepening chain.
3. **Induction fidelity gates everything.** The two pre-existing E3 models could not plan even L1 (ka59 too
   noisy, sc25 memorized). A faithful model is the precondition; our local/codex inductions don't yet reach
   it on these games, while a careful hand-induction does. This is the concrete target for the local-GGUF
   proposer.

## Artifacts

- `results/experiment_program_gen_tu93.json` — positive control (L1 reproduced, hidden-state-bound deepening)
- `results/experiment_program_gen_ka59.json`, `..._sc25.json` — existing-model contrast (no generalization)
- `results/arc_e3/tu93/world_model_nav.py` — the hand-induced faithful nav world model
- `scripts/experiments/experiment_program_gen.py` — the harness (all 3 adversarial-verify clean)
