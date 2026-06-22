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

## Coordinated-redesign attempt (pieces 1 + 2 built) — converges on piece 3 as the binding constraint

The operator authorized building the redesign on the TTT (learned-dynamics) path. Outcome:

- **Piece 1 — cell-recall verify gate (built, both branches, parity-safe).** But scoring the SAVED e3
  induced models (`results/arc_e3_induced_model_quality.json`) overturned its premise for the LLM path:
  the LLM-induced world-models predict NEAR-IDENTITY (cell-recall ~0-0.05, *lower* than exact-match which
  was inflated by no-op transitions). The gate correctly REJECTS them — the e3 induction bottleneck is
  induction QUALITY, not the metric. Cell-recall remains the right gate for the *learned-dynamics* (TTT)
  path (the CNN is genuinely 0.55-useful) and is the more honest metric (it exposes identity-predictors).
- **Piece 2 — divergence-tolerant TTT solve loop (built, runs clean).** Replan-on-divergence + learn-from-
  surprise, so a 0.55-cell-recall model can drive live execution instead of halting on the first exact
  mismatch. But it **cannot be exercised**: `plan_in_model` needs an OBSERVED win-state, and the loop's
  exploration never reaches a first level-up on the hard games (cd82: 0 wins in 400 actions). No win → no
  goal → no plan → piece 2 never fires.

**Both pieces are blocked by the SAME thing the whole investigation keeps returning to: EXPLORATION-TO-
FIRST-WIN (piece 3).** Until the agent can trigger a first level-up on an unseen game, the cell-recall gate
has no useful model to pass and the divergence-tolerant executor has no plan to execute. The naive
salient-cycle explorer in the loop is even weaker than the real `depth_first_ride` explorer (1/11 floor);
testing piece 2's *deepening* value would require wiring the real explorer into the loop's explore phase
to reach level 1 first, then handing off to the TTT planner.

**The unifying conclusion of the entire session:** the 0.08 wall is fundamentally a **sparse-reward
exploration problem** — reaching the first level-up on a never-seen game within ~5n actions. Every piece
of world-model machinery (LLM induction, learned dynamics, trust gates, execution) is *downstream* of it
and moot until it is solved. Directed exploration / sub-goal discovery is the real, hard, open lever — the
same bottleneck the SOTA scan flagged (Family-B executable world-models induce; the search/exploration is
the moat). It is not a quick win, and bigger models / better gates / better executors do not address it.

## PIECE 3 BREAKTHROUGH — exploration diversity is the lever; the hybrid explorer beats the floor

The binding constraint (exploration-to-first-win) finally yielded a POSITIVE, measured, shippable result.

- **No denser reward signal exists** — the frame exposes only `levels_completed` + `win_levels=6` (no
  score/sub-progress). So it is pure sparse-reward search; no reward shaping is possible.
- **Diagnostic breakthrough:** a random walk over salient candidates reaches first-win on **3/11**
  (r11l, sp80, lp85) vs the structured `depth_first_ride` explorer's **1/11**. r11l is trivially
  reachable randomly (372/600 steps post-win) but the structured explorer scored 0 in ~2000 actions:
  **it over-commits to ONE depth-first branch and misses easy "structure-missed" wins.** The failures
  split: 3 structure-missed (diversity-recoverable) + 8 genuinely-hard (random gets 0 too).
- **The hybrid deliverable** (`scripts/arc_hybrid_explore_measure.py`): structured-first +
  random-restart-on-stall. **3/11 first-win, and it DOMINATES both baselines** — lp85 kept efficient
  (structured @20, eff 2.0069) while r11l (@901) + sp80 (@777) are recovered via random diversity. So
  its averaged `min(h/a,1)²` score strictly beats pure-structured (1/11) AND pure-random (3/11 but lp85
  inefficient).

**This is the first lever that moves the wall.** It is general (diversity, not game-specific), so it
should transfer to the hidden eval, and it is SHIPPABLE: wire random-restart-on-stall into the submitted
explorer (behind a flag, parity-safe) → a real candidate to beat 0.08. The 8-game hard tail remains the
frontier-research part (multi-step specific sequences in a sparse-reward env), not diversity-recoverable.

| explorer | first-win | lp85 efficiency | averaged score |
|---|---|---|---|
| structured (submitted) | 1/11 | 20 actions (eff 2.0069) | baseline |
| pure random | 3/11 | 142 actions (~0.04) | WORSE (lost efficiency) |
| **hybrid (structured+random-on-stall)** | **3/11** | **20 actions (eff 2.0069)** | **best — dominates both** |

## Artifacts

- `results/arc_compete_sim.json` — explorer floor 1/11; goal-bias variants
- `results/arc_ttt_loo_gate_probe.json` / `arc_ttt_solve_test.json` — TTT gated out (exact-match) + 0/4 solve
- `results/arc_induction_arm_stronger_proposer.json` — e3 0/6, model-size-independent
- `results/arc3_e3_pipeline_diagnostic.json` — EXPLORE_SAW_NO_WIN + INDUCE_FAILED stages
- `results/experiment_4557_*.json` — e3 induction held-out acc 0.0 vs threshold 1.0
- `docs/research-notes/arc-gate-readiness-prior-ttt-2026-06-21.md` — the full investigation chain
