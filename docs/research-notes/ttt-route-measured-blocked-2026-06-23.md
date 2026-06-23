# TTT-route follow-on — wired + measured: blocked by the exploration-to-first-win wall (outer-loop, 2026-06-23)

Operator asked to wire + measure the TTT-route surfaced by the trust-gate measurement: route live trust
to the TTT learned-dynamics on the games where they pass cell_recall (sc25 0.80, tn36 0.87, lp85 0.59,
ka59 0.91), then measure whether a trusted TTT model drives `plan_in_model` to a live win.

**Finding: it is ALREADY wired (`arc_ttt_solve_loop.ttt_solve`, pieces 1+2+3), and measurement shows it
does NOT yield a live win — it is blocked by the exploration-to-first-win wall, the same
generation-guidance constraint everything this session hit. The TTT-dynamics-planning value itself
remains UNRESOLVED (the clean isolation is confounded — see below), but it is gated behind directed
exploration regardless.**

## What was measured

**1. The wired loop (`ttt_solve`, CPU/GPU, sc25).** `levels_reached=0, first_levelup_actions=None,
plan_attempts=0`. The loop's exploration is a self-described "NAIVE salient-cycle placeholder (piece 3 =
directed exploration is the real fix here)"; on sc25 it never reaches a first level-up → `_win_states`
stays empty → no goal → `plan_in_model` never even fires. The TTT dynamics never get exercised for
planning. (Matches the prior `arc_ttt_solve_test.json`: 0/4, `n_win_states=0`.) Not compute-bound —
more CNN epochs cannot make a naive cycle find a win.

**2. The strong-cell_recall games are not even offline-solved.** sc25 and tn36 (cell_recall 0.80 / 0.87,
the best passers) have `arc_loop_solve_*.json status: needs_per_game_RE` — no banked solve exists. Their
cell_recall came from a LOO *corpus* probe, not a live solve. So there is no goal to inject for the two
games the lever most wanted.

**3. Injected-win test on lp85 (the one game with a real L4 solve), GPU.** Replayed the 54-action L4
trajectory into a prior-warmstarted CNN TTT model: `n_win_states=4` (goal present), gate passes. Then
`plan_in_model` from root toward any win-state, two engines:
- `plan_full` (L0 exact memorization + CNN): **no plan found.**
- `plan_cnn` (CNN-only generalization): **no plan found.**

**Even the FULL-memorization engine** (all 54 transitions stored exactly) could not plan to a win — which
means `plan_in_model`'s candidate enumeration cannot reproduce lp85's own solve actions (lp85 wins via
specific clicks the centroid/keyboard candidate set does not enumerate). So this test is **confounded**:
it cannot isolate whether the TTT *dynamics generalize*, because the planner cannot reach the win even
with perfect dynamics. (The held-out cell_recall=1.0 is likewise L0-contaminated — not a clean
generalization number.)

## Honest conclusion
- The TTT-route's **binding constraint is the exploration-to-first-win wall** (the loop's own flagged
  "piece 3, the real fix"). Routing trust to TTT dynamics does nothing while the loop can't reach a first
  win to plan from. This is the **generation-guidance wall** — the same one macro-depth, click-coverage,
  and the gate-flip all hit, and the one `.428` goal-energy / expansion-prior attack.
- The narrower question "do trusted TTT dynamics generalize a plan to a win, given a goal?" is **not
  cleanly resolved** — the lp85 isolation is confounded by `plan_in_model` candidate enumeration, and the
  strong games (sc25/tn36) lack a banked goal. A clean isolation would need `plan_in_model` seeded with
  the recorded solve actions as candidates on an offline-solved game — a further harness, not pursued
  (diminishing returns; the exploration wall gates the lever regardless).

## Disposition
Measurement scripts ran on the prototype branch `outer-loop/ttt-route` (`proto_ttt_route.py`,
`proto_ttt_fast`, `proto_ttt_injected_win`), now **purged** — prototype code not retained (a definitive
null); the findings + numbers above are preserved in this note. The TTT-route is **not a quick live win**; it is gated on
directed-exploration-to-first-win (piece 3). The session's four follow-on levers (macro, click-heatmap,
gate-flip, TTT-route) all converge on the same wall: **reach/generate the first win** — exactly `.428`'s
focus. The honest next bet is the directed-exploration / goal-energy generation work already in flight,
not another offline lever.
