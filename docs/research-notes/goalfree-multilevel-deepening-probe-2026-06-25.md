# Goal-free L1→L2 deepening probe (proto_goalfree_deepen) — valid null + the alignment reframe

**Date:** 2026-06-25 · **Author:** outer-loop (operator-directed: "what is our next multi-level live
agent improvement iterative step?" → "prototype it now")
**Result:** VALID NULL — goal-free systematic exploration does not deepen lp85 to L2 — **plus two
findings that change how we read the recent multi-level milestones.**
**Artifacts:** `results/proto_goalfree_deepen.json`. **Code:** `scripts/experiments/proto_goalfree_deepen.py`,
`python/carnot/agentic/arc_go_explore.py` (bug fix).

## The question and the probe

The `.430` goal-predicate fix (win-state exemplar + satisfiability check) shipped, ran (exp4664), and
nulled-and-retired (`single_exemplar_goal_insufficient`) — so "fix the goal predicate" is a doomed
rerun. The 2026-06-24 leader-gap analysis reframed the wall: the leaderboard #1 deepens multi-level
*for free* with no goal predicate (reward-driven exploration + per-level reset). Carnot already has that
machinery — `GoExploreReplayArchive` (return-then-explore) wired into `StepwiseExplorer` but disabled in
the shipped path. This probe is the cheap, decisive disambiguator: does goal-free systematic exploration,
riding past L1, deepen lp85 to L2 — *before* committing to the expensive CNN-as-driver build?

Design (parity-safe, no shipped edit): monkeypatch `_policy_for_mode` to build the integrated policy with
`go_explore_archive` on/off + `target_levels=2`; run `run_variant_attempt` with `DEEPEN=1` (ride past L1)
+ `DISABLE_INDUCTION=1` (pure goal-free, no LLM). NoOpProposer → CPU-only. Two arms × 3 lp85 variants ×
1200-action budget.

## Result

| arm | L1 reached | L2 reached | Go-Explore injections |
|---|---|---|---|
| no_archive (control) | 3/3 (reproduced) | 0/3 | 0 |
| go_explore | 3/3 (reproduced) | 0/3 | **232 prefixes / 597 actions** |

`crossed_bar=False`; positive control passes (lp85 reaches L1 reproducibly); the lever is **genuinely
exercised** (232 return-prefix injections). adversarial_verify: 0 flags. **Verdict: a VALID null —
goal-free systematic exploration is insufficient to deepen lp85 to L2.**

## Finding 1 — a second silent representation bug (the Go-Explore archive was dead)

Building the probe surfaced a bug in the *shipped* Go-Explore code, the same class as the exp4710 CNN
dict-candidate bug: `GoExploreReplayArchive._frame_grid` grabbed `FrameDataRaw.frame` — a **(1,64,64)
3-D array** — via `hasattr(frame, "frame")`, so `observe()`'s `grid.ndim != 2` guard early-returned on
**every live frame** and the archive stored nothing (0 observations despite 399+ calls). `grid_of()`
returns the correct 2-D grid. Fixed (prefer `grid_of`, squeeze a leading singleton). **Implication: the
conductor's `.433` A2 — which enabled Go-Explore live — was testing a DEAD archive, so its null is
suspect.** Combined with the exp4710 CNN bug, *two* of the recent "goal-free / generation levers all
null" results were partly measuring dead code. Worth re-examining the `.428-.433` generation-lever nulls.

## Finding 2 — lp85's L2 is goal-DIRECTED (alignment), which is *why* goal-free fails

The registry win condition for lp85 is `marker_pair_shape_alignment`: **"align each moveable piece with
its goal sprite"** (click-only). This is a **sparse-reward, goal-directed** task — you only level up when
*all* pieces are exactly aligned to their target sprites. Goal-free exploration cannot stumble onto that
in 1200 actions without knowing the target. So the null is not "goal-free deepening is broken" — it's
"**goal-free deepening cannot solve a goal-directed alignment L2**", which is the expected result.

This means lp85 was a poor stress-test of the *broad* goal-free thesis (the leader's mechanism is built
for **reward-dense** deepening, where any progress flows toward the next level). Our corpus lacks a clean
**reward-dense** L1→L2 game to test goal-free on cleanly (the other L1-reachers are hidden-state /
no-grounded-delta / recently-deepened). So:

- **For lp85 specifically:** goal-free deepening is the wrong tool — its L2 needs a goal. The CNN-driver
  build is **NOT justified** for lp85.
- **For the broad goal-free thesis:** still open, but untestable on our current corpus without a
  reward-dense multi-level game.

## The redirected next step (this falsifies the cheap paths and points the way)

lp85's L2 goal is **observable in the frame** — the goal sprites are visible. So the right multi-level
step is a **perception-grounded STRUCTURAL goal**, not an exemplar-replay goal:

- Detect the moveable pieces and their goal sprites (object-centric perception), and induce
  `is_level_complete = "every piece is aligned to its goal sprite"` — a structural predicate from a
  **single live frame**, no L2-win exemplar needed.
- This is *exactly why the single-exemplar goal-fix nulled*: a flat exemplar grid does not decompose into
  "pieces aligned to sprites"; you need object-centric perception to *express* the alignment goal.
- **This connects directly to the conductor's `.433` A1 (object-centric/relational perception).** The
  multi-level deepening for an alignment game rides on the same perception primitive: object-centric
  perception → an expressible alignment goal → a satisfiable L2 predicate → a reachable plan.

So the next iterative multi-level step is **a perception-grounded structural-alignment goal for lp85's
L2** (built on the `.433` A1 object-centric representation), NOT the CNN driver and NOT another
exemplar-replay variant. Falsifiable gate: lp85 reaches a genuine `live_agent_self_discovery` L2,
offline-reproduced, with `goal_predicate_satisfiable=True` and the goal expressed as a piece→sprite
alignment over detected objects.

## Method note

- The Go-Explore archive only *injects* return prefixes when a node is exhausted (no untested actions)
  or at max_depth AND a finer-than-default binning yields eligible cells — the default 6-bin archive
  stored 2 cells and injected 0 (the lever was untested). Use `bins≥16` + a long budget; assert
  `prefixes_injected>0` before trusting a no-L2 null (the false-negative guard).
