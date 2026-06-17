# ARC-AGI-3: the LLM is a GAP-FILLER, not a solver (2026-06-17)

Operator strategic correction (2026-06-17): "knowing that the local SOTA model is almost
always going to get 0% on ARC games itself, we need to be strategic on where such an LLM
can help dynamically identify and potentially help fill implementation gaps with the other
components that can actually make progress."

## The evidence: LLM-as-direct-solver ≈ 0%

- gemma-4-12B (local) induced world model for cd82: **0%** verifier accuracy (returned a
  scalar, omitted is_level_complete). codex (dev ceiling): **27%** — also too low to plan a
  solve. Frontier LLMs score **<1%** on ARC-AGI-3 directly. The verifier-grounding correctly
  REFUSED to trust these (no plan, no fabricated solve) — but that means E3-as-full-world-
  model-induction is the WRONG frame: the LLM cannot be the solver.

## The reframe: LLM fills the GAPS of the components that DO make progress

What actually makes progress on ARC-AGI-3 (measured this session):
- **Training-free explorer** (`arc_graph_explore` + frontier-distance): **8/11** public games
  solved from scratch at competition budget.
- **Verifier-routed search + learned verifier** (`arc_solver_kit`): efficiency (10.75x fewer
  states on lp85).
- **GameAdapter** (`arc_game_adapters.py`): the per-game DELTA — `state_key` (the load-bearing
  hidden state, e.g. sc25 tank-facing), `hand_verifier` (win/goal-distance predicate),
  salient `action_labels`. This is SMALL, focused, and is what unlocks a stalled game.
- **Verifier ensemble** + `ops/verifier_gaps.md` (the discrimination the ensemble lacks).

**The LLM's RIGHT job** = the focused REASONING + CODE tasks the LLM is good at, applied to
those components — NOT solving the spatial/combinatorial game itself:

1. **Write/fix the GameAdapter DELTA** for a stalled game. From observed transitions, the
   LLM infers the small per-game pieces: the win-condition predicate, the load-bearing
   hidden state for the dedup key, the salient action set / click-data schema (e.g. tn36's
   non-{x,y} ACTION6). This is the "reverse-engineer only the DELTA" of the ARC Solve
   Reproducibility discipline — the harness (search, verifier, gate, routing) is REUSED.
2. **Diagnose WHY a component stalled** and propose the targeted fix. "Explorer reached L0
   because the win needs click #27 that salience deprioritizes" / "dedup merges two states
   that differ only in facing" → a one-line state_key or candidate-ordering fix.
3. **Propose a new verifier INVARIANT** for a missing-verifier gap (`ops/verifier_gaps.md`):
   the discriminator the ensemble can't compute — feeding the GAP-3/GAP-4 verifier program.

## Why this is higher-leverage

- The LLM does what it's GOOD at (reason about why a component fails + write focused code),
  not what it's BAD at (directly solve the game = 0%).
- The per-game cost collapses to the DELTA (the adapter / the fix), which an LLM can write —
  not a whole solver. The heavy lifting (complete search, energy/verifier scoring,
  reproduction gate) stays in the deterministic components that already work.
- It is gain-not-churn: each LLM-filled gap is a durable component (a registered adapter, a
  new verifier) that compounds across games (the self-learning thesis).

## Architecture: the verifier-routed cascade, REFRAMED

The tier-3 LLM escalation does NOT solve the game; it **writes the missing COMPONENT** so the
working search can:
```
explorer / verifier-routed search        (tiers 1-2, deterministic, make progress)
   → STALL on game X
   → CHARACTERIZE the gap  (which component, what signal is missing — done by the verifier/
                            instrumentation: best level reached, where it stuck, what class
                            of transition/state it can't discriminate)
   → LLM (iGPU, local) FILLS the specific gap:  a GameAdapter delta, a state_key/candidate
                            fix, or a verifier invariant  (focused code, NOT a world model)
   → plug the filled component into the harness → re-run → measure (the focused loop)
```
The Carnot verifier remains the ground (it validates the LLM's adapter/invariant against
real transitions, exactly as it grounded — and rejected — the 0% world model).

## Consequence for the E3 work

E3's `plan_in_model`/full-world-model path is DE-PRIORITIZED (LLM-as-solver = 0%). The E3
machinery that KEEPS value: the transition collection, the `WorldModelVerifier` (now used to
GROUND the LLM's adapter/invariant, not a full model), and the local-iGPU LLM substrate —
repurposed for gap-filling. The next build is the **gap-characterizer → adapter-writer**
loop, targeting the games the explorer stalls on (wa30, cn04, sk48) and the tn36 click-schema
delta, with the LLM writing the small per-game piece, not the solver.

## FINDINGS from the first gap-fill runs (2026-06-17) — what actually blocks it

1. **Autolearning capture-and-reuse: BUILT + correct.** `gap_fills/` (save/load),
   reuse-first (skip LLM if captured), save-on-success (reproduction-gated), and a runtime
   `validate` callback that retries on buggy code. It correctly REFUSED to capture every
   broken heuristic — which is the point.
2. **Local models write buggy heuristics (prompt-tunable, not size-bound).** gemma-4-12B AND
   Qwen3.6-35B-A3B both wrote `goal_distance` functions that fall through to `None`;
   token-capping to avoid that truncated the code instead. A hard "must end in an
   unconditional float return" contract + a right-sized token budget is the fix; this is
   prompt-adherence, not model capability (the 35B failed the same way as the 12B).
   gemma-4-31B DENSE is too slow on the iGPU (timeout) — use a MoE (Qwen-35B-A3B / gemma-26B-A4B).
3. **THE REAL BLOCKER (architectural): a heuristic needs the RIGHT search engine.** The
   gap-fill plugged the heuristic into `graph_explore_solve_v3` (value-guided). But v3 does
   NOT solve cn04 even with a PERFECT deterministic diff-from-win heuristic (nor does
   novelty-only v3) — whereas `graph_explore_solve_v2` (systematic BFS-frontier) DOES solve
   cn04. So on cn04 the heuristic was doomed regardless of the LLM: v3 is the wrong engine.
   **A goal-distance heuristic's value is EFFICIENCY (fewer actions) in a search that already
   reaches the win** — the proven lp85 pattern (`OfflineSolver` + `hand_verifier` → 10.75x
   fewer states), NOT making v3 solve-from-scratch a game it structurally can't.

**Correction to the next build:** the gap-fill must target where a heuristic provably helps:
(a) plug it into the verifier-routed `OfflineSolver` (needs the per-game adapter the LLM also
writes), or (b) make the STRONG explorer `graph_explore_solve_v2` heuristic-GUIDED (prioritize
its frontier by the heuristic) so it keeps its solving power AND gains efficiency. Validate on
a game where a heuristic demonstrably helps (lp85), then generalize. cn04-in-v3 was the wrong
test combo; the loop machinery is sound.

## SECOND build + findings (2026-06-17, outer-loop): v2 heuristic-guided A* shipped

Implemented correction (b). `graph_explore_solve_v2` now takes an optional
`heuristic=goal_distance` (+ `heuristic_weight`, `stats`): when given, the frontier is
ordered **A\*-style by `depth + weight*heuristic(frame)`** (a heap), not FIFO. The `g`
(depth) term is the fix for why greedy `v3` (pure-`h`) fails — it keeps v2's COMPLETENESS
(no local-minimum trap) while letting the heuristic prioritize. When `heuristic is None` the
path is **byte-for-byte the original pure-BFS** (zero regression; the proven 8/11 solves and
the 20 graph-explore tests still pass). `stats` records `expansions`/`states` so an A/B can
measure the EFFICIENCY win (fewer states to the same win), which the action-count metric
misses on shortest-path games. `scripts/arc_gap_fill.py` now A/Bs the heuristic through this
v2 path (not v3) and credits "helped" on fewer-actions OR (equal-actions AND fewer-expansions).

**Empirical A/B of a naive diff-from-win (Hamming) heuristic vs pure BFS** (`(grid!=WIN).sum()`,
budget 8000, from L0; all solve+reproduce under BOTH arms — the engine is correct):

| game | A* actions / expansions | BFS actions / expansions | verdict |
|------|-------------------------|--------------------------|---------|
| cd82 | 5 / 955  | 5 / 525  | A* WORSE (+430 exp) |
| sp80 | 6 / 1085 | 4 / 301  | A* WORSE (+784 exp AND a LONGER 6-vs-4 path) |
| su15 | 7 / 1406 | 7 / 1746 | A* HELPED (−340 exp) |

**The honest lesson — heuristic QUALITY, not the engine, is now the gap.** The A* engine is
correct and the right infrastructure, but a Hamming-distance-to-win heuristic is too crude: it
helped only 1/3 games and on sp80 it produced a SUBOPTIMAL (longer) trajectory (A* with an
inadmissible `h` is not optimal). This SHARPENS the LLM-gap-filler value proposition: the LLM's
job is to write a heuristic that captures GAME STRUCTURE better than Hamming (e.g. misplaced-
object count weighted by manhattan-to-target, or a progress monotone), now directly MEASURABLE
via this A/B (expansions + reproduction gate). The reproduction-gated capture correctly REFUSES
to bank a heuristic that doesn't help — so autolearning only compounds verified efficiency wins.
Also confirmed: cn04 does NOT solve from plain v2-L0 at budget 8000 (with or without
`mask_hud`) — its registry solve is a captured TRAJECTORY, so cn04 is the wrong A/B target for
plain v2; use games v2 solves from L0 (cd82/sp80/su15/…) when measuring a heuristic.

## FIRST capture + 7-game characterization (2026-06-17, outer-loop)

**FIRST autolearning capture landed: `gap_fills/su15_goal_distance.py`.** The iGPU LLM
(Qwen3.6-35B-A3B) wrote `goal_distance`; the v2-A* A/B confirmed it solves su15 in the same 7
actions as BFS but with expansions **1746→1406 (−19.5%)**, reproduced=True both arms → CAPTURED,
reproduction-gated. The loop is proven end-to-end (generate → A/B → gate → capture → reuse with
no LLM call, bundle-able for the offline competition). The LLM chose Hamming, which for su15 IS
the clean efficiency win.

**A 7-game headroom characterization of Hamming-A* vs pure BFS (budget 8000, from L0):**

| game | A* acts / exp | BFS acts / exp | read |
|------|---------------|----------------|------|
| su15 | 7 / 1406  | 7 / 1746  | **clean win** (same acts, −340 exp) → captured |
| m0r0 | 21 / 2398 (repro) | — / 8000 (NO solve) | A* **solves where BFS exhausts budget**, but 21 acts vs registry's 15 (budget-capability, worse path) |
| r11l | 5 / 335   | 3 / 2236  | A* explores −85% states but LONGER path (5 vs 3) |
| ls20 | 13 / 2201 | 13 / 1777 | A* worse (+424 exp) |
| tu93 | 18 / 2114 | 18 / 2114 | identical (heuristic inert) |
| sk48 | — / 8000 (NO solve) | 14 / 4365 | A* **fails where BFS solves** (Hamming misleads) |
| wa30 | — / 8000  | — / 8000  | neither solves @8000 (needs depth/budget) |

**The data-backed pattern: Hamming-guided A* trades exploration-efficiency for path-optimality.**
It explores far fewer states when its surface-diff gradient points at the win (su15 −19.5%, r11l
−85%, m0r0 solves-where-BFS-can't), but because Hamming is INADMISSIBLE it often finds a LONGER
path (r11l 5-vs-3, m0r0 21-vs-15, sp80 6-vs-4) or is MISLED into a dead basin (sk48 fails where
BFS solves). It cleanly helps only where surface-diff tracks true distance-to-goal — su15 of the
7. The capture gate correctly banked ONLY su15 and refused the mixed/negative cases (m0r0's worse
path is NOT banked — it would not improve on the known 15-action registry trajectory). **So the
open lever is confirmed with data: a STRUCTURE-aware, more-admissible heuristic (object-
displacement / manhattan-to-target / a progress monotone) is what unlocks the games Hamming
can't — exactly the LLM-gap-filler's job, now measurable on this A/B. Next: de-steer the
gap_fill prompt from Hamming toward structural heuristics, and target r11l (recover the optimal
3-action path at low expansions) + sk48 (solve where Hamming dead-ends).**

## DECISIVE root cause: CELLS ≠ MOVES (2026-06-17, outer-loop)

De-steered the gap_fill prompt toward a structural signal and re-ran r11l. The LLM (Qwen3.6-35B)
wrote a reasonable MASKED mismatch (`((grid!=WIN) & ((grid!=0)|(WIN!=0))).sum()`, start=1375 vs
full-grid Hamming) — but A* STILL found the 5-action path (335 exp), not the optimal 3. A free
(no-LLM) **heuristic_weight sweep on r11l with the masked count** is decisive:

| weight | r11l actions | expansions |
|--------|--------------|------------|
| 1.0   | 5 (suboptimal) | 335  |
| 0.1   | 4              | 1229 |
| 0.02  | **3 (optimal)**| 2791 |
| 0.005 | **3 (optimal)**| 2678 |
| BFS   | 3 (optimal)    | **2236** |

**There is NO weight at which a cell-count heuristic gives r11l BOTH the optimal 3-action path
AND fewer expansions than BFS** — its optimality↔efficiency Pareto frontier is dominated by BFS
at the optimal-path end. Root cause, now proven: in r11l a single click changes ~hundreds of
cells, so "1375 cells wrong" is a terrible proxy for "3 MOVES to win." **CELLS ≠ MOVES.** A
cell-diff heuristic is only admissible (and only helps) in LOW-cell-impact games where one
action changes few cells so cell-count ≈ move-count — su15 of the games tested. In HIGH-cell-
impact games (r11l / cn04 / m0r0 / sp80) cell-count massively over-estimates move-distance →
A* goes greedy → suboptimal path; and no engine-tuning (weight) rescues it.

**Conclusion for the gap-fill program.** (1) The autolearning loop + v2-A* engine are correct
and shipped; su15 is a genuine captured efficiency win. (2) The REAL unlock for high-cell-impact
games is a **MOVE-distance-aware heuristic** (estimate game-actions-to-win, not cells-to-win) —
which requires modelling the game's action granularity (how many cells an action flips), i.e.
exactly the per-game DELTA the LLM-gap-filler / GameAdapter is meant to reverse-engineer. That
is genuine research (an admissible move-metric per game), not an overnight prompt tweak. The
honest stopping point: ONE clean capture banked (su15), the cell-count ceiling characterized and
explained, and the next lever (move-distance heuristics, fed by the per-action cell-impact the
prompt already shows) identified for a directed session.

Cross-refs: CLAUDE.md "ARC Solve Reproducibility + Solver-Reuse" (RE only the delta) +
"Missing-Verifier Gap Logging"; `python/carnot/agentic/arc_game_adapters.py`,
`arc_solve_learning.py` (recommend_approach), `ops/verifier_gaps.md`;
`docs/research-notes/arc-agi3-focused-loop-and-engine-2026-06-17.md`.
