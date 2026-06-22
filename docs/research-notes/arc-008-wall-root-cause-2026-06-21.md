# The 0.08 wall has ONE unifying root cause: the exact-match world-model trust gate (2026-06-21 capstone)

A full day of offline, zero-quota investigation (this session) converges on a single, unifying explanation
for why the ARC-AGI-3 Kaggle submission is stuck at **0.08** on the hidden games — and why every
"smart" augmentation we have built fails to move it.

## The finding

**The agent always reduces to the bare exploration floor, because every world-model path is gated out by
the same unmeetable exact-match trust gate.**

`WorldModelVerifier.score` counts an induced world-model "correct" on a transition ONLY when
`np.array_equal(pred, next_grid)` — a FULL-GRID exact match (arc_executable_world_model.py:166) — and the
agent skips any model whose held-out accuracy is `< 0.5` (arc_competition_agent.py:1455-1457; threshold
**1.0** in the e3 induction positive control, exp4557). A 64×64 dynamics model that is genuinely ~55%
changed-cell-accurate scores ~0 on full-grid exact match. So:

| Path | Result | Why |
|---|---|---|
| **TTT learned dynamics** (prior-warmstarted CNN) | gate fires **0/5** on unseen games | exact-match held-out acc ≈ 0 (cell-recall 0.55) — `results/arc_ttt_loo_gate_probe.json` |
| **e3 LLM world-model induction** | **0/6** gap-1 games; no-op | induced `world_model.py` fails the same gate (cn04 verify acc 0.1; exp4557 held-out 0.0 vs threshold 1.0) |

Both fail the SAME gate → the induce→verify→plan superstructure produces a **no-op plan** → the agent
**falls back to bare exploration**. That bare floor is **1/11 first-win** (8/32 in the larger sample) =
**the 0.08**.

## The wall is OVERDETERMINED — three compounding failures, no single lever

1. **The exact-match trust gate** is unmeetable for any imperfect world-model (the chokepoint above). Both
   the learned-dynamics and the LLM-induced model die here.
2. **EXPLORE-SAW-NO-WIN.** On hard games the explorer reaches level 1 on only ~2/25 — so even when
   induction fires it has no win example to condition the goal on (`arc3_e3_pipeline_diagnostic.json`:
   `failing_stages=[EXPLORE_SAW_NO_WIN, INDUCE_FAILED]`). Directed exploration-to-first-win is unsolved.
3. **Exact-execution-divergence halt.** `plan_and_execute` halts the instant `pred != obs` exactly — so
   even a gate-passing imperfect model can't drive a generalization solve (TTT solve test: 0/4).

And model SIZE is provably NOT the lever: e3 induction is **byte-identical** between gemma-4-12B and
Qwen-35B (identical fallback-explore action counts) — the bottleneck is the pipeline, not the model
(`results/arc_induction_arm_stronger_proposer.json`).

## What this session refuted (all offline, zero shots spent)

- **prior+TTT learned dynamics** → gated out (exact-match). The prior's real transfer (cell-recall
  0.314→0.5485) is invisible to the gate.
- **goal-bias explorer (fixed AND direction-agnostic, both modes, weight sweep)** → frontier reordering
  doesn't move the floor; best_first unlocks +1 game but at ~0 efficiency, depth_first is inert.
- **e3 LLM world-model induction (incl. a 35B proposer)** → no-op, gated out, model-size-independent.

Every road leads to the same place: **the explorer floor is the ceiling, and the world-model
superstructure is inert because of the exact-match gate.**

## The implied lever (coordinated, not a single knob — and uncertain)

Moving 0.08 needs ALL THREE fixed together, because each alone is necessary-not-sufficient:

1. **Cell-recall / change-region trust gate** (already built for TTT: `trust_cell_recall`) so imperfect
   world-models can be USED — apply it to the e3 induction verify gate too.
2. **Divergence-TOLERANT execution** (replan-on-divergence, not exact-match halt) so a ~0.6-accurate model
   can drive planning despite per-cell error.
3. **Directed exploration-to-first-win** so hard games trigger a level-up that induction can condition on.

This is a coordinated redesign of the induce→verify→plan→execute loop, not a quick win — and even fully
built, success is uncertain (the per-cell error may still break multi-step plans). It is, however, the
ONLY path the evidence supports; bigger models, goal-bias frontier heads, and the current cascade are all
refuted.

## The meta-result: the discipline worked

The offline-first gate did exactly its job: across this session we measured the wall, mapped its single
root cause, and refuted four candidate levers (prior+TTT, fixed goal-bias, confirming goal-bias, e3
induction incl. a 35B model) — all offline, **spending zero of the ~2 scarce L4x4 shots**. We now KNOW
why 0.08 is stuck, definitively, instead of having burned submissions discovering it live. Gate verdict
unchanged: NOT gate-ready.

## Artifacts

- `results/arc_compete_sim.json` — explorer floor 1/11; goal-bias variants
- `results/arc_ttt_loo_gate_probe.json` / `arc_ttt_solve_test.json` — TTT gated out (exact-match) + 0/4 solve
- `results/arc_induction_arm_stronger_proposer.json` — e3 0/6, model-size-independent
- `results/arc3_e3_pipeline_diagnostic.json` — EXPLORE_SAW_NO_WIN + INDUCE_FAILED stages
- `results/experiment_4557_*.json` — e3 induction held-out acc 0.0 vs threshold 1.0
- `docs/research-notes/arc-gate-readiness-prior-ttt-2026-06-21.md` — the full investigation chain
