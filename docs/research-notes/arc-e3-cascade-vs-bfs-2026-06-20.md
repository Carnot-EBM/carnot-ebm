# The E3+v3 "stronger" cascade is worse than bare BFS in bounded compute (2026-06-20)

Second local sim (post value_weight 5->0 revert) + a clean bare-BFS comparison. Decisive finding:
**the submitted E3 cascade + v3 head loses to bare BFS on solve-rate, completion, speed, AND action
efficiency (the scoring metric).**

## Apples-to-apples (25 games, frame-only, budget 8000, 120s cap, no LLM)

| Metric | E3 cascade + v3 head (current submitted, value_weight=0) | bare explorer (BFS) |
|---|---|---|
| solved (L1+) | 3/25 (lp85, sp80, vc33) | **4/25** (+ m0r0) |
| completed (no timeout) | 6/25 | **24/25** |
| timed out @120s | 19/25 | **1/25** (bp35) |
| wall/game | 83-101s | **1-56s** (mostly 13-40s) |
| lp85 cost | 7792 actions / 34s | **21 actions / 1s** |
| vc33 cost | 7731 actions / 96s | **1759 / 5s** |

## Why (root cause)

- value_weight=0 already removed the A* per-node weighting, but the v3 HEAD IS STILL LOADED and its
  (expensive: connected-components + relational + delta) features are computed per node to break ties --
  pure overhead at weight 0 (no routing benefit).
- The E3 cascade also runs the world-model INDUCTION tier on stall (CPU-expensive even without the LLM).
- Net: the cascade burns the whole action budget (7700+) and most of the wall budget without finding
  short solutions, where bare BFS finds them fast + cheap.

## Action-efficiency = the scoring metric, and this is catastrophic

Score = (human_actions/agent_actions)^2. lp85: BFS 21 actions ~= near-human (high score); E3 cascade 7792
actions = score ~0. So even on games the E3 cascade SOLVES, it scores ~0 on efficiency. BFS is the
opposite. This is the single biggest reason the cascade is the wrong submitted default.

## Recommendation

1. **Gate the v3 head LOAD on value_weight>0** (load_cross_game_value_head returns None when weight==0):
   at weight 0 the head only adds per-node featurization cost for zero routing benefit. This restores the
   8/32-baseline StepwiseExplorer speed.
2. **Strongly consider submitting the bare-BFS explorer path** (the 8/32 bridge baseline) rather than the
   E3 cascade for the bounded eval. The cascade's only justification is the LLM induction tier on hard
   games -- which measured 0/6 value-added (exp .414 bridge) -- and it destroys action efficiency.
3. The LLM tier (disabled in this sim) is the cascade's reason to exist; before keeping the cascade,
   measure WITH the LLM tier whether it cracks games BFS can't AND at acceptable action cost. Current
   evidence says no.

This sharpens .416 B2 (lazy/cheap value-eval) and reframes A1: the issue isn't just value_weight, it's
the whole E3+v3 stack's per-node cost vs bare BFS.

Cross-refs: results/arc_offline_to_live_bridge_v2.json (8/32 BFS baseline, 0/6 LLM tier),
arc-a1-value-weight-regression-2026-06-20.md (the value_weight half), the two local sims (wf wugw36c2c
value_weight=5: 1/25; wf wh18hnu61 value_weight=0: 3/25; bare BFS: 4/25 + 24/25 completed).
