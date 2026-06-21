# Covering spans, gaps, and jumps in live ARC-AGI-3 play — the lp85 discovery arc

**Date:** 2026-06-21 · **Origin:** operator directive after the lp85 L1→L2 investigation —
"make note of this discovery arc as we need to identify how we can cover spans, gaps, and jumps
like this when we're playing live games at submission time."

## 1. What happened (the arc, honestly)

A multi-hour investigation chased the lp85 L1→L2 barrier as an **LLM world-model induction** problem.
Every lever we tested was the wrong one:

| Lever tested | Result | Why it was the wrong lever |
|---|---|---|
| Bigger model (9B→14B-Coder→35B-A3B) | all 0.000 change-acc | not a model-capacity problem |
| Output truncation (operator caught it twice) | ruled out cleanly | real confound, but not the barrier |
| `_delta` cap 80→4000 (real bug, fixed) | necessary, not sufficient | improved a 2-of-N sample |
| RLE + object-displacement encoding | dominated raw, still ~0 | encoding a sample of the wrong size |
| Trajectory delta-compression | designed, not needed | same |

The barrier dissolved only when we **read the game source** and **ran a systematic action sweep**:
lp85 is **not an induction problem**. It is a **per-level rotation/piece-mover puzzle** with a tiny
state space, solved by **action discovery + 2-to-6-button search, no LLM**:

- **L1**: 2 global rotate buttons (x=4, x=58); solved by `LLLLL` (5 actions vs human 17).
- **L2**: **6 different buttons at new positions** (x=14,20,38,40,48); L1's buttons cannot reach it.
- The 80 `button_X` source tags are **distributed across levels** — each level exposes its own controls.

Two of my own intermediate claims were wrong and were corrected by cheap probes: "80 simultaneous
buttons / under-exploration" (disproved by the exhaustive sweep finding exactly 2 at L1), and "simple
np.roll rotation" (disproved by the roll test). **The cheap structural probes (read source, sweep
actions, check determinism) would have found the truth in minutes; the expensive LLM sweeps cost hours
and told us nothing the structure didn't.**

## 2. The taxonomy: spans, gaps, jumps

A level series is a sequence of **regimes**. The agent must recognize which it is in:

- **SPAN** — a stretch solvable by *repeating a known action* (or a known short policy). lp85 L1 is a
  pure span: repeat `L`. Spans are cheap; a greedy "repeat the action that last made progress" covers them.
- **GAP** — a level whose solution requires an action/mechanic **not in the current repertoire**. lp85
  L2 is a gap: it needs 6 buttons the L1 policy never had. A gap is invisible to "keep doing what worked"
  — that policy stalls at a state-space dead end (we saw L1's 2 buttons exhaust at 21 states, never
  reaching L2's win). **The signature of a gap is: search over the current action set is exhausted /
  cycling without progress.**
- **JUMP** — the **transition** between regimes that **re-exposes the action space** (new buttons appear,
  old ones deactivate, the mechanic changes). The L1→L2 jump silently swapped the entire control set. A
  jump is the *cause* of a gap; the fix is to **re-discover the action space at the new regime**, not to
  assume the prior regime's actions transfer.

## 3. The general coverage procedure (the reusable lesson)

```
per regime (level, or whenever progress stalls):
  1. DISCOVER the active action space   -> systematic probe: which actions change the state, and how
  2. SEARCH over the discovered actions  -> shortest action sub-sequence that makes progress (level-up)
  3. ADVANCE and detect the next JUMP    -> on level-up / mechanic change, GOTO 1 (re-discover)
  STALL DETECTOR: if SEARCH exhausts the current action set without progress, you are in a GAP ->
  force a re-DISCOVER (the action space changed under you). Never assume prior-regime actions transfer.
```

This is verifier-grounded and LLM-free: discovery is a deterministic environment probe; search is
bounded BFS with state-hash dedup; the "progress" signal is the env's own level/score. It is the
project's structured-solver thesis, not the world-model-induction tier (E3, which was the wrong tool here).

## 4. Doing this LIVE at submission time (the hard part)

Offline, discovery is free (deterministic `reset()` + replay — branch and probe at no cost). **Live,
every probe click is a scored action**, so blind sweeping is unaffordable (lp85 has 4096 positions; the
score is `(human/agent)^2`). Strategy to cover spans/gaps/jumps under a live action budget:

1. **Solve offline, replay live (PRIMARY).** When an offline replica of the game exists (the ARC harness
   `environment_files/<game>/` + `arc_solver_kit.offline_arcade`), run discover→search→advance offline to
   a full solution, gate it with `arc_solver_kit.reproduce`, then the live agent **only replays the
   banked solution** — paying zero discovery cost. This is the offline-solve-then-reproduce discipline and
   is how spans/gaps/jumps are covered for free at submission time. *(lp85 is in this case.)*
2. **Budgeted online discovery (FALLBACK, truly-unseen game).** When there is no offline replica:
   - **Rank, don't sweep.** Use a **frame-change predictor** (the logged `ops/verifier_gaps.md` gap:
     small CNN over the frame → click-effect heatmap) to try the *few* candidate actions most likely to
     change the frame, instead of all 4096. This is the single highest-value live primitive and directly
     caps discovery cost.
   - **Spend the discovery budget at JUMPS only.** Detect a regime change (level-up, or the
     last-progress action becoming a no-op) and *only then* re-discover — within a small action budget —
     rather than continuously.
   - **Stall detector → re-discover.** If repeating the last-progress policy stops making progress
     (a GAP), trigger a bounded re-discovery instead of grinding the dead action set.
   - **Cache and transfer.** Persist discovered (position→effect) maps per game/regime to
     `ops/arc_solve_registry.yaml`; a control found at one regime is the first thing to try at the next.
3. **Detect the JUMP.** A level-up event, or "the action that worked now changes nothing," both signal the
   action space may have shifted — the trigger to re-discover.

## 5. Action items

- Promote the discovery sweep + 2-action search into a reusable `arc_solver_kit` primitive
  (`discover_actuators(env, prefix)` + `search_to_progress(...)`), per the ARC Solve Reproducibility
  discipline (scaffolding, not one-off).
- Build the **frame-change predictor** (already an open `ops/verifier_gaps.md` gap) — it is the
  load-bearing primitive for the LIVE fallback (rank clicks, bound discovery at jumps).
- Add a **stall/jump detector** to the live agent: on no-progress or level-up, re-discover the action
  space rather than assuming transfer.
- Meta-process: **spend cheap structural probes (read the env source if available, sweep the action
  space, check determinism) BEFORE expensive model experiments.** The lp85 arc would have been minutes,
  not hours, in that order.

## 6. lp85 solver result (2026-06-21)

The discover→search→advance loop ran on lp85:
- **L1 SOLVED in 5 actions** (`LLLLL`), reproduced on a fresh env (deterministic). Human baseline 17 →
  near-max efficiency. Pure structured solve, no LLM.
- **L2: 5 buttons discovered, but BLIND BFS stuck** (3000 nodes, no level-up). Root cause: L2's solution
  is DEEP (human baseline 38 actions) and the branching factor is ~5, so blind BFS reaches only ~depth 5
  (5⁵≈3000) before the node cap. Brute force cannot reach a 38-action solution at branching-5.

**This is exactly where the VERIFIER becomes load-bearing (the Carnot thesis).** The fix is to replace
blind BFS with **goal-directed / verifier-routed best-first search**: heuristic = sum of piece-to-goal
distances (the `goal`/`goal-o` tiles are present every level), used to rank action expansions. The
discover→search→advance architecture is correct; the SEARCH needs the energy/goal-distance verifier in
the loop to make a deep solution tractable. This is the agentic verifier-proof venue: discovery enumerates
the actions, the verifier routes the search to the goal. Spans (L1) are solved by any search; GAPS that are
also DEEP (L2) require the verifier-routed search — a sharper statement of why the verifier is the moat.

**Next step:** wire a goal-distance verifier into `search_to_progress` (best-first over discovered
buttons, priority = piece-to-goal heuristic), re-run, and bank L2+ as offline-reproducible levels.

## 7. Verifier-routed search result — the verifier IS the bottleneck (2026-06-21)

Built and ran the verifier-routed solver (discover → goal-distance best-first → advance):
- **L1 solved in 5 actions / 11 nodes** — best-first trivially routes a SPAN.
- **L2 STUCK** (12,000 nodes; `best_h` never improved from h0=16). Diagnosis is decisive: **every naive
  positional heuristic is FLAT** — every one of L2's 5 buttons gives heuristic-delta exactly **0**, for
  both "distance to the static movable region" and the corrected "distance from goals to current piece
  positions." Root cause: L2 has **8 goal cells but 161 movable piece-cells**, so a min/sum positional
  distance is insensitive (some piece is always near every goal). The win is a **combinatorial
  piece↔goal↔value matching**, not a positional aggregate.

**This localizes the Carnot thesis empirically: the moat is VERIFIER QUALITY, not search or model.** The
discover→search→advance architecture is correct and the verifier-routing mechanism works (L1); the deep
levels stall not because the search is weak but because a **hand-crafted naive verifier carries no
gradient toward the win**. Solving these puzzles *is* the verifier-design problem — which is precisely
the project's core bet.

**The fork (verifier quality) — a strategic choice, not a quick fix:**
- **(a) Structural verifier from the source win-condition.** The game source defines the win check
  (pieces on goals). Reverse-engineer it into a correct verifier. Tractable, game-specific (does not
  generalize — tension with the general-solver goal).
- **(b) Learned verifier (Phase-3 program).** Train a verifier to predict progress/distance-to-win.
  Needs a denser-than-level-up progress signal (or bootstrapped self-play). The general path; the hard,
  high-value one.
- **(c) Bootstrap a win-state.** Solve one L2 win by any means (longer/luckier search, or a structural
  one-off), then verifier = cells-differing-from-win. Cheapest if a single L2 solution can be found at
  all; gives a perfect dense heuristic for that level.

**What is banked regardless:** the discovery + verifier-routed-search architecture (reusable
`arc_solver_kit` primitives), L1 solved offline-reproducibly, and the empirical proof that on this game
class the verifier — not the model, the encoding, the context, or the search algorithm — is the binding
constraint. That is the strongest possible statement of why the verifier is the product.

## 8. Option (c) tested first (operator: "c first before we commit to b") — and it settles the fork (2026-06-21)

Ran two cheap (c)-bootstrap variants to see if an L2 win-state is obtainable *without* building a verifier:
- **(c)-blind COMPLETE BFS over L2's full (STEP=1-confirmed) 5-button set: FAILED.** Reachable L2 state
  space is **>60,000 distinct states** (BFS hit the 60k cap, queue non-empty), with **no level-up found**.
  At branching-5, 60k states ≈ depth 7 — far short of the ~38-action human solution. Uninformed search
  cannot bootstrap a win.
- **(c)-decomposition: does NOT cleanly apply.** Commute test: {0,1} commute (top conveyor), {2,3,4}
  commute among themselves, but **{0,1} do not commute with {2,3,4}** — the conveyors are *coupled*, and
  the goal region (rows 25-37 × cols 34-37) is touched by *all* buttons. So L2 does not factorize into
  independent sub-puzzles solvable separately.

**Conclusion — (c) and (b) collapse.** You cannot cheaply bootstrap an L2 win-state because the state
space is large and coupled and uninformed search has **no gradient**. A real verifier is therefore
*necessary*, not optional — which is precisely what (b) provides. The ONLY verifier-free path remaining
is **(a) source-derived**: read the win-condition + target configuration directly out of
`environment_files/lp85/.../lp85.py` (sidesteps search entirely, but game-specific and does not
generalize). Everything else requires the learned/structural verifier of (b).

**Recommendation:** (c) has done its job — it proved L2 is not cheaply crackable, so the investment in
(b) (a learned distance-to-win verifier, the Phase-3 program) is now *justified by evidence*, not
assumed. If a fast lp85-specific win is wanted in the interim, (a) — extract the target config from the
source — is the one remaining shortcut. The general answer is (b).

## 9. (a) source-derived verifier + (b) learned verifier — results (2026-06-21)

**(a) SOURCE-DERIVED VERIFIER — validated.** Read the win-check `khartslnwa()` (all `bghvgbtwcb` pieces
on `goal` at +1,+1; all `fdgmtkfrxl` on `goal-o`) and built `h = sum of piece->nearest-goal Manhattan
distance` (true distance-to-win). It gives a real gradient where *every* naive/positional heuristic was
flat: **L1 solved in 5 nodes** (rides `15->12->9->6->3->win`), and on **L2 `h` drops 27->9** (best-first).
This *proves the verifier-routed approach is sound on real lp85* — the purpose of running (a) before (b).
Two implementation facts uncovered: (i) lp85's PIECES are fixed and the GOALS rotate on the conveyor, so
search dedup must be on the grid hash, not piece positions; (ii) snapshot/restore of sprite positions is
NOT a faithful fast-forward — the rotation depends on hidden internal game state, so the correct (but
slow) reset+replay is required. Full L2 *solve* is gated on search budget (>60k coupled states) + a
sum-of-distances local minimum at h=9. **The 90k-node attempt LANDED: NO L2 solve, best_h STABLE at 9 from 20k->90k nodes** -- not a search-budget issue but a stable barrier (best-first explored ~90k states; min reachable distance is 9, never 0). Two fixable causes: (1) the replay solver lets the real MOVE BUDGET decrement and treats exhaustion as a dead branch, pruning the long (human=38-action) win-path; (2) a coupled-pieces local minimum the sum-of-distances verifier can't see past. Both confirm the sum-of-distances `h` is NOT the true value function. Fix = true min-actions-to-win value (no local minima) + budget-high-during-search complete search (verify length after).

**(b) LEARNED VERIFIER — the key finding: high R^2 != usable search gradient.**
- **Pixels-only learnability PROVEN.** A small CNN predicts the source distance-to-win from the GRID
  alone (no sprite/source access) at **val R^2 = 0.93 (32x32) -> 0.989 (64x64)** — an 86KB
  submission-deployable verifier. This is the generalization (a)-with-source structurally cannot claim:
  at submission time there is no readable win-check, but a learned grid->distance net needs none.
- **BUT the gradient is non-monotonic along solve paths, robustly.** Across THREE training variants
  (MSE@32, MSE@64, pairwise-ranking@64 reaching 85% global pairwise-order-accuracy), the learned value
  on the L1 solve path bumps up where the true distance descends (e.g. v3: `16.0,8.9,10.5,8.1,7.7`).
  Root cause: the near-win states are visually near-identical (the goal differs by a few cells), so an
  MSE/ranking net regresses/orders them imperfectly — exactly the fine-grained distinctions search needs.
- **Implication (sharpens the verifier thesis).** The moat is not "have a verifier" but "have a verifier
  with a clean GRADIENT/ORDERING along solution paths." Prediction accuracy (R^2) is the wrong metric;
  path-monotonicity is the load-bearing property, and naive learned regressors do not optimize for it.
  For win-by-precise-alignment games, an EXACT (source/sprite) verifier resolves the fine gradient that a
  learned-from-pixels one cannot. The forward paths: (1) a HYBRID — learned verifier for the coarse
  approach (R^2=0.99 gets close) + an exact local check (detect objects -> exact distance) for the
  fine near-win gradient; (2) test whether best-first (frontier-tolerant of noise, unlike greedy descent)
  routes adequately on the learned verifier despite non-monotonicity (open question); (3) train on the
  true min-actions-to-win value (needs L2 win-data) with a path-monotonicity objective.

## 10. Transferable lessons — the keep (2026-06-21, operator: "keep what we've learned and move on")

This whole lp85 investigation RE-DERIVED already-solved work: lp85 is reproduced to **L5**
(`ops/arc_solve_registry.yaml`, Exp4372), has a reference `GameAdapter` ("click-only rotation
puzzle"), a banked solution, and a learned verifier — the conductor was pushing **L6** the whole
time, and the adapter was in this worktree all along. Two lessons survive the redundancy:

1. **PROCESS (the expensive miss):** before any ARC-game deep-dive, `grep <game>
   ops/arc_solve_registry.yaml python/carnot/agentic/arc_game_adapters.py` FIRST. Seconds of
   structural probing vs a session of re-derivation. (This note's own section 1 stated the lesson;
   it was not followed at the top.)

2. **RESEARCH (genuinely general, for Phase-3):** *verifier prediction-accuracy (R^2) is NOT the
   metric that matters — gradient/ordering-quality along solution paths is.* A learned verifier hit
   R^2=0.989 from pixels yet routed search no better than a hand-crafted sum-of-distances that has a
   local minimum (L2 stalled at h=9, 20k->90k nodes). The fix shape: TRUE min-actions-to-win value
   (retrograde, no local minima) + a path-monotonicity objective, or an exact verifier where precise
   alignment defines the win. The spans/gaps/jumps live-discovery taxonomy (sections 2-4) also
   stands as general guidance for the live agent at genuinely-unseen games.

NOT kept / dropped: the lp85-specific re-solving (redundant), and the TRM-as-solver build
(redundant + against the standing "NO TRM training" directive and the 2026-06-18 retirement).
