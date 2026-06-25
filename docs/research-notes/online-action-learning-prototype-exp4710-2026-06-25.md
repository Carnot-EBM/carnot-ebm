# exp4710: goal-free online action-learning prototype — measured on the exp4605 first-win harness

**Date:** 2026-06-25 · **Author:** outer-loop (operator-directed: "prototype the goal-free online loop on
the exp4605 harness" → "launch the full 4-arm sweep" → "get to the bottom of these errors")
**Result:** VALID NULL (positive control passing, mechanism exercised) — with an important caveat on scope.
**Artifacts:** `results/experiment_4710_arms_summary.json` + the four per-arm
`results/experiment_4710_online_action_learning_arms_*.json`. **Code:**
`python/carnot/agentic/arc_online_action_effect_scorer.py`, `python/carnot/experiment_4710_online_action_learning_arms.py`.

## What was built

A prototype of the StochasticGoose-leader mechanism (per
`docs/research-notes/leader-gap-online-action-learning-vs-multilevel-wall-2026-06-24.md`): an
`OnlineActionEffectScorer` that trains a binary frame-change CNN **online per game** on free
self-supervised labels (did the frame change?), Adam lr=1e-4 BCE every 5 actions, hash-deduped buffer,
per-level reset, plus a **CNN coord head that proposes click coordinates** added to the candidate set. It is
a drop-in for `frame_change_scorer`, wired into the live `E3AgentPolicy`/`StepwiseExplorer` via two
parity-safe guarded hooks (ground-truth observe at `_ingest`; coord-propose in `_candidates`) — no-ops for the
shipped frozen scorer, so the submitted agent is byte-identical (parity green).

Measured on the **exp4605 held-out first-win harness**: 25 color-permuted public-game variants, exploration-only
(LLM proposer silenced → CPU, no Qwen), budget 200, `first_win` = a live level-up that passes the offline
reproduction gate. Baseline (shipped) first-win = **0.04** (only lp85 solves).

## The result (after the false-negative fix)

| arm | first_win | errors | fits | isolates |
|---|---|---|---|---|
| shipped-frozen | 0.04 | 0 | 0 | baseline (CNN discarded by the bug below) |
| frozen-fixed | 0.04 | 0 | 0 | **fixing the CNN-discard bug: +0.00** |
| online-warm | 0.04 | 0 | 935 | **online training: +0.00** |
| online-warm-propose | 0.04 | 0 | 935 | **+ coord generation: +0.00** |

All four arms solve only lp85. Positive control passes (frozen reproduces the shipped 0.04; the CNN provably
contributes — 4737 observations, 935 gradient steps, **0 errors**). **Verdict: a VALID null** — online
action-effect learning, as prototyped, does not lift held-out first-win.

## The false negative we caught (operator-directed)

The first sweep showed all arms = 0.04 with **79k–108k scorer errors**. The operator asked whether this was a
false negative. It was. Per-site exception capture found a single bug, 79,442×:

```
CNN | AttributeError: 'dict' object has no attribute 'action_id'
```

`FrameChangeScorer.candidate_score` reads `getattr(candidate, "action_id")`, but the explorer's **frontier**
scores plain dict rows `{"action","data"}` (not `ArcAction` objects) on ~20/25 games — so the CNN term raised
and its score was **silently discarded**. The memory term survived (PersistentAEM tolerates dicts), so only the
CNN failed. **The shipped `LiveActionEffectScorer` has the same bug** (a bare `except: pass` swallows it), so
the online-trained CNN contributed nothing in *every* arm → the original "online == frozen" was never a real
test. Fix: `_as_action_like()` normalizes dict candidates before the CNN call (verified: ft09 errors
17,014 → 0). The corrected re-run above is the valid measurement.

> **Bonus finding for the operator (parity-affecting, left for review):** the same dict-candidate bug silently
> disables the action-effect CNN term in the **shipped** agent on most games. Fixing the shipped
> `FrameChangeScorer` would make its 0.05-weight CNN term actually contribute — a free correctness fix,
> though (per this null) not one that lifts first-win on its own.

## Why the null (honest diagnosis)

1. **The CNN is a 0.05-weight re-rank term vs memory's 1.0.** Even a trained, working CNN barely perturbs the
   memory-dominated ranking, so it cannot change *which* game solves. `frozen-fixed == online-warm` proves
   online training adds nothing *at this weight*.
2. **Coord generation (weight-independent) also ties.** The online-CNN's top-k heatmap coords, added to the
   candidate set, did not surface a winning trajectory on the 24 non-solving games within budget 200.

## Scope caveat — what this does and does NOT say

This tests the leader's mechanism as a **graft onto our architecture**, NOT a faithful reproduction of the
leader's full agent. The leader's CNN is the **sole driver** (no memory term, no object-centroid candidates,
budget≈∞/8 h, per-level reset, hierarchical CNN-coord sampling). Our prototype attenuates it to a 0.05 add-on
on top of a memory-dominated scorer + object-centroid candidates + budget 200. So:

- **Honest verdict:** *online action-effect learning, bolted onto our memory-dominated re-ranker as a
  0.05-weight term (or an additive coord-proposer), does not lift held-out first-win.*
- It does **not** refute the leader's mechanism — it shows the **cheap graft** of it doesn't work. A faithful
  test requires making the online CNN the **driver** (sole/dominant candidate source + scorer, higher budget,
  per-level reset), which is the larger architectural pivot.

## Forward path (weighed against the multi-level wall)

1. **Retire the cheap grafts** — online-re-rank-at-0.05 and additive-coord-proposal are empirically nulled.
2. **The faithful leader-mechanism test is the DRIVER config** (CNN-coord sampling replaces object-centroid
   candidate generation; memory down-weighted; budget up; per-level reset). High-effort, and it collides with
   the **same candidate-generation wall** the conductor's `.433` A1/A2 (object-centric perception; amortized
   cross-game exploration + Go-Explore archive) are independently nulling on — multiple generation
   interventions all failing to lift live first-win is itself the signal: first-win on hidden games is the hard
   open problem, and incremental grafts don't move it.
3. **Multi-level wall:** untested here (first-win only). The goal-free-**driver** loop + per-level reset is what
   the leader-gap note argued could cross the L1→L2 wall by *demoting* goal-induction; this prototype never
   reached the driver config, so the multi-level claim remains open and would ride on the same DRIVER build.

## Method note (reproducibility / what tripped us up)

- Run the arms **sequentially** (solo CPU). Four parallel online-training processes torch-segfaulted under CPU
  contention at ~32 min. `setsid` detaches cleanly; the orphan-cleanup janitor only kills python with elapsed
  ≥ 1 h, so short solo runs are safe.
- The frozen arm MUST reproduce 0.04 (positive control); the prototype monkeypatches `_policy_for_mode` and
  reuses the REAL `measure_policy_pair` rather than cloning the loop (an early clone lost lp85's solve).
- Always read `scorer_diagnostics.errors` before trusting a null — a non-zero error count means the mechanism
  may not have been exercised (the false-negative trap this whole exercise turned on).
