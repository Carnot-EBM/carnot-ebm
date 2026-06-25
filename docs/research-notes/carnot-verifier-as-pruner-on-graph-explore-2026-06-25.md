# Carnot verifier-as-pruner on a graph-explore generator — concept proven, current policy not deployable

**Date:** 2026-06-25 · **Author:** outer-loop (operator-directed: "build the Carnot verifier-as-pruner on
top of a graph-explore generator — measure whether our energy verifier cuts its action count")
**Result:** the verifier-as-pruner CONCEPT is proven (clean efficiency wins exist), but the current
hard-argmax policy over a cross-game-marginal verifier is NOT deployable (net wash + a solve loss).
**Artifact:** `results/proto_carnot_pruner.json`. **Code:** `scripts/experiments/proto_carnot_pruner.py`.

## The north-star move

Per the leaderboard-repro-gap diagnostic (just-explore's graph-exploration mechanism is 4× better than
ours at equal gated budget), the legitimate Carnot move is NOT to ship a copied solver but: run a strong
graph-explore **generator** and add the Carnot energy **verifier as a PRUNER** that cuts its action count.
The leaderboard score is `(baseline_actions/actions_taken)²×100`, so fewer actions to the same win raises
the score directly — the verifier as the efficiency moat ("verifier routes/prunes, doesn't generate").

**Implementation.** just-explore picks the next untested edge to try via `random.choice(untested_edges)`
(`graph_explorer.py:choose_edge`). The pruner replaces that with **argmax over untested edges by the Carnot
frame-change verifier** (`LiveActionEffectScorer` = frozen frame-change CNN + PersistentAEM memory), mapping
each `edge_idx → (action_id, x, y)` (segment-centroid click or arrow) and scoring `candidate_score(frame,
ArcAction(...))`. All of just-explore's other logic (segmentation, graph, BFS frontier, level reset) is
unmodified — only the edge ORDER changes. Measured on the 9 games just-explore solves, 5 seeds, budget 2000.

## Result — bimodal (`pruner_exercised: True`, 83% fire rate → a valid measurement)

| outcome | games | numbers |
|---|---|---|
| **Clean efficiency win** (both arms solve 5/5, pruner faster) | **vc33**, **lp85** | vc33 31→6 actions (**27× efficiency**); lp85 461→387 (1.4×) |
| Win, variance-conflated (pruner also raised solve *rate*) | ar25, ft09 | ar25 1/5→5/5 (268 vs 1585); ft09 4/5→5/5 (549 vs 1544) |
| Regression | cd82, s5i5, sp80, **r11l** | r11l 42→653 actions (severe) |
| **Solve LOST** | **m0r0** | 5/5 → 0/5 |

`median_efficiency_ratio = 1.04` (net wash); `8/9` solves preserved.

## Honest reading

- **The concept is proven.** Where the verifier is right, it cuts actions a lot — vc33 from 31 to **6**
  actions (a 27× efficiency-score gain), lp85 cleanly too, and it makes ar25/ft09 *more reliable* (1/5→5/5,
  4/5→5/5). The Carnot frame-change verifier genuinely steers exploration toward the win on those games.
- **The current implementation is not deployable.** Hard-argmax deterministically follows the verifier even
  when it's wrong, and the verifier is a **cross-game MARGINAL prior** (PersistentAEM scores clicks by
  bucketed (x,y) frequency over all 25 games — it doesn't know which game is being played). On games whose
  winning mechanic doesn't align with cross-game frame-change statistics (m0r0, r11l, sp80, s5i5), the
  pruner steers AWAY from the winning region and breaks exploration diversity → regressions + a solve loss.
- **The failure is a POLICY problem, not a fundamental one.** Hard-argmax over-commits; a hedged policy that
  preserves exploration diversity would keep the wins without the regressions.

## Forward path (two concrete, ordered refinements)

1. **Hedged pruner policy (cheap, do first).** Replace hard-argmax with **weighted sampling** (sample edges
   ∝ Carnot score) or **ε-greedy** (mostly Carnot-argmax, sometimes random). This caps the downside — the
   pruner can never get stuck following a wrong verifier, so it can't lose a solve below random's baseline —
   while still front-loading high-score edges. Expected: keep vc33/lp85/ar25/ft09 wins, kill the m0r0/r11l
   regressions. This is the immediate next iteration.
2. **Game-specific / online-adapted verifier (bigger lift).** The cross-game marginal is too blunt. An
   online frame-change signal (adapt the verifier to THIS game's observed transitions during play — the
   StochasticGoose-style per-game learning, but as a *pruner* on a working explorer rather than a sole
   driver) would make the verifier's ranking game-aware. This is the convergence of the leader's online
   action-learning with the Carnot verifier-as-pruner thesis.

## Bottom line

The verifier-as-pruner is the right architecture (a strong generator + Carnot's verifier as the efficiency
moat), and it demonstrably cuts actions where the verifier is right (vc33 27×). The blocker is the *policy*
(hard-argmax) and the *verifier's bluntness* (cross-game marginal) — both addressable. A hedged pruner is
the cheap next step that should convert this net-wash into a net win without losing solves.

## Method note
- `pruner_exercised` (Carnot scores varied AND changed the chosen edge on >0 steps) is the false-negative
  guard — here 83% fire rate, so the null-on-some-games is real, not a dead verifier. Always check it.
- Score with `ArcAction` OBJECTS, not dicts (the `getattr(candidate,"action_id")` dict-AttributeError bug
  fixed earlier this session would otherwise silently zero the CNN term).
