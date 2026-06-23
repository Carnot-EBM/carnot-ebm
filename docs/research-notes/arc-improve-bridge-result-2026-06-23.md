# How to improve the .427 bridge result (action-effect predictor)

**2026-06-23, outer-loop (operator-requested "how do we improve that result?").** Method: deep-read
our predictor vs the StochasticGoose full recipe vs the active `.428` plan → propose improvements
across 4 axes → adversarially verify each lifts a LIVE metric and isn't already queued → synthesize
(17-agent workflow `arc-improve-bridge-result`). **Both load-bearing code claims independently
verified (see §4).**

## 1. Where we are

`.427` (exp4629) crossed the offline→live bridge with a GENERATION lever, not a reranker: the
action-effect predictor (PersistentAEM action-type effects @ weight 1.0 + a 5%-weight
`SmallFrameChangeCNN` re-scorer) graduated into the scored E3 path → live first-win **0.407→0.591
(+0.184)**, median actions-to-first-levelup **2→1**, `verifier_is_oracle:false`, parity green,
solve-rate preserved; it **transferred** (exp4632: cd82 +0.5). **Honest boundary:** efficiency /
first-win-on-engaged-games win, measured on an offline-arcade cached-transition replay (11.8s, not a
leaderboard run); live multi-level **solve-rate stays flat at 0.04**; per-game signal is small-sample
(~17/25 positive, several n≤1). **`.428` already attacks this** from two angles over the SAME
object-centroid candidate pool: A1 (exp4640) graded `is_goal` as a live goal-energy heuristic; A2
(exp4641) predictor ranker→search-expansion-prior. So this note focuses on what `.428` does NOT
queue: candidate **generation outside the centroid pool**, and **horizon collapse**.

## 2. Ranked improvements (lead with solve-rate movers — the prize)

### PURSUE_HIGH — Macro-action vocabulary induction (horizon collapse)
- **Lever:** induce a per-game macro vocabulary by clustering observed action *sequences* by
  frame-delta effect (push-until-blocked, cycle-color, toggle-then-step); expose each as a composite
  `ArcAction` so best-first search plans over **macros, not primitives**. A 13-primitive plan
  collapses to ~3 macros, pulling the exponential horizon into the ~5n budget. Shared library seeded
  from solved games (cross-game prior); refined online per-game (self-discovery).
- **Why it grows the result:** the 0.04 solve-rate is a *multi-level depth* wall — A1/A2 reach the
  first level-up faster but don't chain to a 2nd/3rd. Horizon collapse is the most direct attack on
  **depth**, the prize.
- **Energy role:** empowerment (the macro-keep criterion) = channel capacity from a macro to its
  reachable frame-delta set, an information-theoretic energy. `verifier_is_oracle:false` (macro value
  is its *observed* frame-delta, never a read of the env win-counter).
- **LIVE gate:** on ≥1 hard-tail game (pre-confirmed horizon-bound via the cheap `cell_recall`
  probe), macro-augmented `graph_explore_solve_v2` banks a NEW reproducible level
  (`arc_solver_kit.reproduce`) that primitive-only does NOT reach **at equal total budget (induction
  cost charged to the macro arm)**, winning plan strictly shorter in macros than primitives
  (horizon_reduction > 1×, the anti-noise check), no first-win regression, bootstrap-CI excludes 0.
  `retire_if_same_verdict`. **Moves `live_solve_rate` / `reproducible_total_levels`.**

### PURSUE_MED — Click-heatmap-as-GENERATOR (off-centroid candidate source)
- **Lever:** our per-pixel `click_head` (verified §4) is computed but only *read at object centroids*.
  Add `propose_click_cells(frame, k=8)` = NMS top-k of the 64×64 sigmoid heatmap, injected into
  `rich_action_candidates` as click candidates NOT already in the centroid set. Turns the predictor
  from a *ranker of ≤48 centroids* into a *generator of click coordinates* — the StochasticGoose
  coordinate-head capability we structurally lack.
- **Why it grows the result:** for click games whose winning cell is NOT an object centroid (empty
  cell, slot, seam), the winner is *absent from the pool* and no reranker — including A2's
  expansion-prior — can recover it. The only lever that adds the missing candidate.
- **Energy role:** the sigmoid heatmap IS a learned per-pixel action-effect energy field over click
  coordinates (low energy = high frame-change prob); top-k = sampling its low-energy modes.
  `verifier_is_oracle:false` (BCE on *observed* frame-change).
- **PRE-FLIGHT FALSIFIER FIRST (~30 min, no training):** over the cached corpus, compute
  `winning_click_centroid_coverage` — for every recorded frame-changing winning action-6 transition,
  is its (x,y) on a `_components_detailed` centroid? **If coverage is high, STOP — the premise is
  dead** (this *tests* the prior "subsumed by centroids" kill rather than asserting it wrong).
  Caveat: the cached corpus may have ~0 coord-labeled level-up clicks, so the falsifier may need a
  small live-collection arm. Only if a non-trivial off-centroid fraction exists: A=centroid-only,
  B=centroids+heatmap-top-k. **Primary gate = `winner_generated`** (structural, n-independent): the
  winning candidate appears in B's pool where it was absent in A, ≥2 newly-generated winners.
  Sequence AFTER a softened (cell-recall) trust gate so a generated winner isn't rejected by exact-
  match. Retrain `click_head` at hidden=24 w/ more examples first (exp4629's hidden=8/497-ex/loss-0.466
  net is near-prior → diffuse heatmap → top-k = no-ops). **Moves `live_solve_rate` via `winner_generated`.**

### PURSUE_MED — On-level-up reset/decay of the action-effect memory
- **Lever:** decay (never hard-wipe) `PersistentAEM` + refit the live CNN's click bias on observed
  in-episode level-up, from the trajectory's OWN transitions — StochasticGoose's
  buffer-clear-on-score-increase, scoped to our single live trajectory; keep static AEM as a fallback
  floor; gate on a min-observed-transition count.
- **Why:** levels relayout, so a cross-game static count memory mis-prices effects at L2+; per-level
  refit specializes the predictor to the current hidden game from its own attempts.
- **Energy role:** n/a (online state-management on a count model). `verifier_is_oracle:false`.
- **LIVE gate:** measured as a delta ON TOP of A2's expansion-prior baseline (dedup A2's marginal
  value); B reduces median actions-to-2nd-levelup OR raises solve depth, paired bootstrap-CI excludes
  the A2 baseline; L1 first-win not regressed. **Moves `live_action_efficiency`** (scope its win
  there — do NOT claim solve-rate).

## 3. The judgment call: is +0.184 robust enough to build on?

**Build on it now; do NOT spend an A-slot hardening it.** The aggregate is real and directionally
consistent (550 paired groups, 17/25 positive, 0 negative; flats are ceiling-saturated or n≤1 — not a
"lucky 11"). The alarming-looking bits are artifacts, not weaknesses: the `[1.0,1.0]` CI is the wrong
metric (integer median collapsing under bootstrap), and "cached/offline" is the substrate the whole
`.427` family used. The one genuine hole — the headline first-win delta ships with no CI — is worth at
most **one cheap B-slot** of bootstrap hygiene, never an A-slot (it's offline-only and solve-rate-
irrelevant). The robust move is to put new generation levers across the bridge and let the *live* gate
adjudicate.

## 4. The coordinate-head question, resolved (code-verified)

**YES, conditionally, and genuinely un-subsumed.** Verified in-repo:
`arc_frame_change_predictor.py:SmallFrameChangeCNN` has a real per-pixel `click_head =
nn.Conv2d(hidden,1,1)` → `sigmoid` 64×64 heatmap; but `arc_graph_explore.py:rich_action_candidates`
enumerates clicks ONLY as `_components_detailed` object centroids (`max_click=48`,
`ArcAction(6,…,'object_click')`). So the heatmap's off-centroid modes are never proposed as actions —
the coordinate-head capability exists in the net but is amputated at the candidate-pool. It helps iff
winning clicks actually fall off centroids — exactly the §2 pre-flight falsifier. Gated behind that
30-min test so we don't build a generator the centroid enumerator already covers.

## 5. Honest dead-ends (do not re-propose)

- **Already in `.428`:** goal-energy heuristic (A1/exp4640), ranker→expansion-prior (A2/exp4641),
  `live_multi_level_solve_rate` (B1), uniform-energy ablation guard (B2).
- **Reranking-class (thrice-nulled):** UCB epistemic-novelty tie-break, confidence-gated budget
  (also already shipped as `apply_adaptive_budget`/exp4513), dense curiosity (exp4628 nulled
  solve-rate a 3rd time). All re-score the same centroid pool — generation crosses the bridge,
  reranking does not.
- **Offline-only / characterization:** bootstrap-CI on the first-win headline, headroom-normalized
  stratification — ≤1 B-slot of rigor, not an A-slot.
- **Fabrication-risk:** any off-centroid proposal justified by a "16.4% off-centroid" statistic —
  that number exists nowhere in the repo; the lever survives only re-grounded on the live falsifier.

## 6. Flagged candidate for `.429+`

`exp4660-macro-vocab-horizon-collapse` (codex; `solve_provenance: live_agent_self_discovery`;
`verifier_is_oracle: false`; gate = bank a NEW reproducible level at equal budget that primitive-only
doesn't reach + horizon_reduction > 1× + bootstrap-CI excludes 0; `retire_if_same_verdict`). Does NOT
displace the level-up-guarantee slot.

## Cross-references
- exp4629 (the bridge-crossing), exp4632 (transfer), exp4638 (capstone)
- `research-roadmap.yaml` `.428` A1=exp4640 (goal-energy), A2=exp4641 (expansion-prior)
- `python/carnot/agentic/arc_frame_change_predictor.py` (`SmallFrameChangeCNN.click_head` — verified)
- `python/carnot/agentic/arc_graph_explore.py` (`rich_action_candidates` centroid-only — verified)
- `/home/ianblenke/arc-sota-refs/ARC3-solution/custom_agents/action.py` (StochasticGoose coord head)
- `docs/research-notes/arc-sota-energy-avenues-2026-06-23.md` (prior note; this sharpens its coordinate-head kill)
