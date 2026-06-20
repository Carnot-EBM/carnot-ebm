# .417 shaping — ACTION EFFICIENCY is the bottleneck (2026-06-20)

Operator: "start shaping .417." This is the design draft the .417 roadmap is built from once .416 closes
(the .416 B2 lazy-eval + capstone results refine the task list). Direction is unambiguous.

## The thesis (confirmed by 3 independent measurements this session)

The live agent's wall is **action efficiency**, not solve-rate and not config tuning:
- The live StepwiseExplorer explores ~7760 actions to find solutions ~21 actions long (lp85). The score
  is `(human_actions/agent_actions)^2`, so even a SOLVE scores ~0 on efficiency.
- The leaderboard leaders win precisely here (StochasticGoose: learn what's clickable -> stop wasting
  actions). The 30-day report names action efficiency as what separates winners from brute force.
- Our guidance signals all came back NULL/insufficient in .415/.416: the value head helps offline (LOO
  0.674) but is too slow per-node live (value_weight reverted to 0); the frame-change predictor was a
  null (A2 "corpus shortfall"); the energy-augmented RANKING was a null (A3). None reduced actions.
- Config tuning is EXHAUSTED: value_weight (re-measured, kept 0), the 3 cascade fixes (value-guard,
  induction-skip, nav-edges) restored SPEED + solve-rate (1->4 solved, 6->0 timeouts) but left action
  efficiency UNCHANGED (median ~7760). The gap is the explorer's EXPLORATION STRATEGY, architectural.


> **UNBLOCKED 2026-06-20:** the frame-change predictor's "corpus shortfall" is RESOLVED — the FULL human-replay corpus is now staged locally (14,797 examples vs the truncated 10,000; 97% frame-changed; 14,020 normalize to a valid action_id; behavior/imitation prior builds). Source: ARC Public Demo via Kaggle mirror jihangli1121/arc-agi-3-replays-v1 (CC BY 4.0, attribution; LOCAL training only, gitignored, NOT bundled). `.417` A2 (re-train the predictor) + A3 (imitation prior) can now run on real/full data.

## The metric (already instrumented)

The local submission gate (`scripts/kaggle/arc_local_submission_gate.py`) tracks **median
actions-to-solve** on 8 games; current live baseline = 7760. **.417 success = drive that down materially
toward the solution length** (offline solutions are ~21-7219; human action counts per level are the real
target). Every .417 task is measured by actions-to-first-levelup reduction WITHOUT dropping solve-rate.

## Candidate tasks (the action-efficiency program)

1. **PRUNE, don't just rank.** The explorer expands ALL salient candidates per node (breadth = the 7760).
   Use the validated structural energy (LOO 0.674) + a frame-change predictor to PRUNE candidates the
   model predicts are no-ops / low-progress, so the explorer never tries them. Pruning (not re-ordering)
   is what cuts the action count. (A3 ranked but did not prune -> null.)
2. **Make the frame-change predictor actually work.** A2 was null on a "corpus shortfall" (.416 B1 staged
   only the attributed-mirror format). Properly stage the FULL human-replay corpus (or build a larger
   self-supervised (frame,action->frame_delta) corpus from the offline arcade) and train a predictor that
   reliably flags no-op actions. This is the StochasticGoose lever, with the metric = actions saved.
3. **Imitation prior from the 342 human replays.** Use human action sequences as a behavior-cloning prior
   so the explorer tries human-like (efficient) actions first -- humans changed the frame on 97% of
   actions. Even the marginal action distribution prunes the candidate set.
4. **Best-first with the LAZY value head** (depends on .416 B2). Once value-head eval is cheap (top-K /
   frame-hash cache), best_first over the LOO-0.674 head may find solutions in fewer actions -- re-measure
   value_weight>0 with the lazy eval (the .416 A1 null was at full per-node cost).
5. **Forward-edge navigation hardening** (my .416 nav-edge fix recorded edges but did not move actions):
   investigate WHY -- is _shortest_path actually used, or does replay still dominate? Close that loop.
6. **Verifier-grounded ADAPTIVE per-step budget** (new 2026-06-20, from the LoopWM/arXiv:2606.18208
   ingestion -- really ACT/PonderNet, Graves 2016; LoopWM is just the citable instance). The explorer
   currently spends the SAME search width/depth on every frame. Add a cheap per-step gate from
   ALREADY-computed signals (energy/value-head margin + predicted-no-op-under-the-induced-model + frame
   novelty): easy/unambiguous frame -> commit 1 candidate immediately; ambiguous frame -> expand the
   budget. This is the "spend compute only on hard frames" idea, and unlike candidate 1 (prune) it cuts
   actions by NOT expanding when the frame is easy. Zero new model, zero training, 16GB/offline-safe.
   Metric = actions-to-solve at equal solve-rate, offline-reproduced via `arc_solver_kit.reproduce`.
   Frame it as "ACT-style adaptive budget for our explorer," NOT "implementing LoopWM." Cross-ref:
   docs/research-notes/loopwm-2606.18208-ingestion-2026-06-20.md.

## What .417 is NOT

Not more config sweeps (value_weight, target_levels -- done). Not banking more public-game levels
(reproducible_total_levels is not the headline). Not the cascade vs BFS debate (the cascade is fixed +
gated). The single question: **make the live explorer find solutions with FEWER actions.**

## Sequencing

After .416 closes. The .416 B2 (lazy value-eval) feeds candidate 4; the .416 capstone's
frame-change/energy verdicts refine candidates 1-3. Pull the strongest from the .416 SOTA-ingestion (D:
action-efficient exploration / affordance learning) into the lead task.

Cross-refs: arc-cascade-regression-fixes-2026-06-20.md (config tuning exhausted),
arc-frame-change-predictor-spec.md, arc-human-replay-application-spec.md, arc-energy-augmented-strategy.md,
arc-leaderboard-competitive-intel-2026-06-20.md (StochasticGoose), the gate (the efficiency metric).
