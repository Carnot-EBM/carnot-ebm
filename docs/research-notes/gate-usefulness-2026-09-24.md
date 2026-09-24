# Does a looser world-model gate accept useful models? (Experiment 10012, 2026-09-24)

## Question

The live agent trusts an induced world model only if it reproduces 8 held-out
transitions exactly. The think-ON pilot showed good models exist and all fail that
gate. This experiment asked whether a looser gate would accept models that help.

A gate is only right if it predicts usefulness. So each model got a ground-truth
label from the real offline simulator: plan inside the model with the live
planner, play the plan in the real game the way the scored agent does, and count
USEFUL when the real game advances a level.

Artifact: `results/experiment_10012_gate_usefulness.json` (REQ-ARC-WMTE-10012).
Evidence: `results/raw/experiment_10012_gate_usefulness/`.

## A correction first (append-only record)

The first run (v1) did not execute the way the scored agent does. The outer loop's
brief described the scored agent wrongly, and an independent review caught it:

| Point | v1 (wrong for the scored agent) | Scored agent (`E3AgentPolicy`) |
|---|---|---|
| Divergence | stop at the first step where reality differs from the prediction | play the plan to the end; nothing compares prediction and reality |
| Start state for a stall induction | the state after the recent moves | the level's reset board, after sending RESET |

The halt rule belongs to the offline development twin (`plan_and_execute`), not the
scored agent. v1 therefore reported "keep exact 1.0; relaxing only adds harm",
and the outer loop passed that on to the operator. It does not hold. Amendment 1
in the spec made the scored-agent execution the primary arm (LIVE_SCORED). It kept
v1 as a labelled second arm; the v1 artifact is
`results/raw/experiment_10012_gate_usefulness/v1_offline_twin_halt_artifact.json`.

## Result (LIVE_SCORED arm)

- 3 of 10 windows are label-informative, meaning the correct expert model levels up
  from the real start state: su15, ft09, m0r0.
- **One non-expert model is useful: su15 think-ON.** It levels up in 7 real
  actions. Its predicted grids go wrong at step 2, but its plan is the same 7
  clicks as the expert's winning plan.
- **Every candidate gate rejects it.** Its held-out exact accuracy is 0.125.
- Relaxed gates add only models that do not help. Among the real candidates
  (think-ON and codeonly, controls removed):

| Gate | Accepted and useful | Accepted, not useful | Rejected but useful | Rejected, not useful |
|---|---|---|---|---|
| live exact 1.0 | 0 | 0 | 1 | 11 |
| masked exact 1.0, change fidelity >= 0.9, cell recall >= 0.8 | 0 | 1 | 1 | 10 |
| masked exact 0.875, change fidelity >= 0.7 | 0 | 2 | 1 | 9 |
| masked exact 0.75 | 0 | 4 | 1 | 7 |

- With the expert controls included, the exact 1.0 gate also rejects a useful
  model: the m0r0 expert scores 0.5 unmasked because of the step-counter row, and a
  masked gate accepts it.

**In this sample, held-out transition accuracy did not predict usefulness.** The
one useful induced model was scored low; the models the looser gates added did not
help.

## Other findings

- **Wrong win conditions.** Two non-expert models predicted the real moves
  correctly for many steps (su15 codeonly 19/19 from the induction state; ft09
  think-ON 8/8 from the reset board). Their own `is_level_complete` then returned
  True on a real state that had not won. The dynamics were right and the goal was
  wrong. Two cases only.
- **The live goal check cannot catch this here.** The gate's goal-predicate
  consistency veto fires only when the window holds a real level-up, and none of
  the 10 windows does.
- **The scored agent never checks its plan against reality.** A model that is
  accepted but wrong spends real actions until its plan runs out. The artifact
  records these wasted actions per pair.
- **Planner budget matters.** With 150,000 nodes (a NOT-LIVE arm), ka59 and ar25
  become informative: their experts win, but need more search than the live
  20,000 nodes.

## Why 7 windows could not answer

| Window | Reason |
|---|---|
| sp80, dc22, wa30 | the expert's win condition is always False (the positive control built experts for transitions, not goals) |
| sb26 | the expert has no win condition |
| ka59, ar25 | planner budget (informative at 150k nodes, not live) |
| g50t | unresolved: the search empties without budget or depth limits |

## What this means

1. Do not loosen the gate on transition accuracy. On this sample that adds bad
   accepts and still misses the useful model.
2. The signal that mattered is whether the model's plan leads to a real win, and
   whether its win condition is right. That points toward checking the plan and
   the goal, not the per-cell dynamics.
3. The sample is very small: one useful induced model. Any gate choice needs more
   informative windows first. The cheapest way to get them: give the experts real
   win conditions (sp80, dc22, wa30, sb26), and add windows.

## Limits

- 10 windows, one first-call prompt each; 3 informative in the live arm.
- The usefulness label covers one plan from one start state. It does not include
  the explorer fallback a rejected model leads to.
- Masks and exclusions came from the positive control, chosen after the held-out
  rows were seen.
