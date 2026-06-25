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

## UPDATE (2026-06-25) — the hedged pruner: NO deployable policy; the blocker is the VERIFIER, not the policy

Tested the forward-path hypothesis (hedge the policy to stop hard-argmax over-committing): 3 hedge arms
(ε-greedy 0.3, ε-greedy 0.5, weighted-sample T=0.5) over the same verifier + 9 games + 5 seeds + budget 2000.
Artifact `results/proto_carnot_pruner_hedged.json`. All arms `pruner_exercised=True` (fire 0.76–0.83).

| arm | median_eff (squared) | solves_preserved | regressed |
|---|---|---|---|
| hard_argmax | 0.68 | 4/9 | 5 |
| eps_greedy_0.3 | **1.15** | **6/9** | 3 |
| eps_greedy_0.5 | 1.42 | 5/9 | 4 |
| weighted_sample | 1.32 | 6/9 | 3 |

**Result: NO arm preserves every solve AND keeps median_eff>1.** The hedges *fixed the efficiency problem*
(median_eff 1.15–1.42 > 1.0 — they genuinely cut net actions), **but every arm still regresses 3–4 games'
solve rates.** The binding constraint is **solve loss, not efficiency**. m0r0 canary: more randomness recovers
more m0r0 solves (argmax 0/5 → ε=0.5 3/5) but **no arm preserves vanilla's 4/5** — a hedge that injects enough
randomness to undo the misdirection has, by definition, stopped being verifier-guided.

**The sharpened conclusion: the blocker is the VERIFIER, not the policy.** The Carnot frame-change verifier is
a **blunt cross-game MARGINAL** (PersistentAEM over all 25 games; it doesn't know which game it's playing) that
is *genuinely wrong* about which edges lead to a level-up on m0r0/s5i5/sp80 — it actively steers exploration
into dead regions. No edge-selection policy can be both verifier-guided and solve-safe when the verifier itself
is wrong. (Adversarial sub-agent review: no fabrication; caveats — efficiency_ratio is the SQUARED action ratio
= the ARC reward shape; per-game survivorship; N=5 < the N≥30 bar — recorded in the artifact. Ran twice, seeds
4731+4732, identical conclusion.)

**Revised forward path:**
1. **A game-specific / online-adapted verifier as the pruner** — adapt the frame-change signal to *this* game
   from its observed transitions during play (the StochasticGoose online-learning, but as a *pruner* on a
   working explorer, not a sole driver). This is the one untested verifier variant that could be right
   game-by-game where the cross-game marginal is wrong.
2. **OR an abstaining verifier** — fall back to vanilla random on edges/states where the verifier's confidence
   is low, so it can only *help* (never misdirect). A calibrated/abstaining pruner is strictly safe by
   construction.
3. **Deadline-relevant (independent of the pruner):** the clean graph-explore *mechanism* itself (vanilla
   just-explore at budget 2000 = 0.36 vs our 0.08 on public games) is the deployable win NOW — adopt the
   explorer mechanism; add the verifier-as-pruner only once it is online/game-specific or abstaining.

## UPDATE 2 (2026-06-25) — abstaining pruner + the deadline generator: the verifier is too SATURATED to prune

Two more results closed the loop (`results/proto_carnot_pruner_abstain.json`, `results/proto_just_explore_budget_scan.json`).

**(a) Abstaining no-op-deferral pruner — still NO deployable policy, and the root cause is now nailed.** The
design never promotes an edge (try predicted-live edges in *vanilla* random order; defer confident-no-ops), so
it structurally cannot misdirect like hard-argmax. Yet it fails — for a deeper reason: **the Carnot
frame-change CNN's P(change) is SATURATED.** Over ~7,700 real untested-edge scorings, P(change) ∈ [0.26, 0.81]
(p50 0.38), *never below 0.2* — the verifier confidently predicts ~every action changes the frame, so it
**cannot discriminate no-op from live edges at all**. Consequence: absolute thresholds {0.05,0.1,0.2} defer
*zero* edges (no-op); percentile-deferral {0.2,0.3,0.5} fires (defers 58k–146k edges) but median efficiency < 1
(0.75–0.84) AND every arm loses m0r0's solve. So **across three pruner prototypes — hard-argmax, hedge,
abstain — the Carnot verifier-as-pruner adds no deployable value**, because the current cross-game-marginal
frame-change verifier carries no usable discriminative signal on these games.

**(b) The deadline generator — clean graph-explore at raised budget.** Vanilla just-explore on the held-out
proxy: 0.16 @budget200 → **0.28 @budget2000** → 0.272 @budget4000 (plateau). **Wall-clock is NOT the
constraint:** a full budget-4000 pass over all 25 games costs ~360s vs the 43,200s (12h) envelope — 120×
headroom; the max affordable per-game budget is ≈361,000 actions (~900× our current `MAX_ACTIONS=400` cap). So
raising our cap 400→2000 is trivially affordable — **capability, not budget, is the limit.** Best deployable:
**budget 2000, gated first-win 0.28** (ties b4000 at half the wall-clock; just-explore plateaus — the games it
can solve, it solves early; the 13 it never solves aren't budget-limited). That is **~7× our E3AgentPolicy
baseline (0.04)** on the proxy.

**THE DEADLINE PLAY (synthesis).** The deployable lever for 2026-06-30 is the **clean graph-explore generator at
budget 2000**, NOT the verifier-as-pruner (which is now a research direction needing a *discriminating* /
game-specific / online verifier — the saturated frame-change marginal can't prune). Honest tensions for the
operator: (1) submission is **operator-only** (external publication); (2) the deadline-winning move uses **none
of Carnot's verifier** — it's a graph-explorer that beats our agent but isn't "Carnot," so shipping it is a
strategy/thesis call, not an autonomous one; (3) **0.28 is an upper bound** (public games just-explore was tuned
on; hidden-game transfer is lower) and N=5 is below the N≥30 bar. The Carnot-native path remains: a working
graph-explore generator + an *online/game-specific* verifier as the pruner — but that verifier does not exist
yet, and the saturated frame-change CNN is not it.

## Method note
- `pruner_exercised` (Carnot scores varied AND changed the chosen edge on >0 steps) is the false-negative
  guard — here 76–83% fire rate, so the null is real, not a dead verifier. Always check it.
- Score with `ArcAction` OBJECTS, not dicts (the `getattr(candidate,"action_id")` dict-AttributeError bug
  fixed earlier this session would otherwise silently zero the CNN term).
