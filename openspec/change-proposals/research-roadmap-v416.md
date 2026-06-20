# Research Roadmap v416 — re-measure value_weight, build the frame-change predictor, use the energy-augmentation

**Milestone:** 2026.06.416 · pre-staged by the outer-loop (operator-directed 2026-06-20 "queue the
re-measurement"). All experiments codex/gpt-5.5; planner/retro stay Claude Opus 4.8.

## Why this milestone
`.415` validated the energy-augmentation differentiator (v3 features, cross-game LOO 0.503→0.674) and
wired the v3 head into the live agent — but the `value_weight=5` it shipped was a measured regression
(the richer v3 eval is too expensive per node), reverted 5→0. `.416` closes the loop:
- **A1 (operator-requested):** RE-MEASURE — does the wired v3 head at `value_weight ∈ {0,0.5,1,2,5}` beat
  bare-BFS on held-out solve-rate AND finish in the eval time budget? Raise `SUBMITTED_VALUE_WEIGHT` only
  if both hold; else keep 0 and report the null.
- **A2:** re-run the human-replay frame-change predictor (`.415` B1 staged the corpus that blocked it).
- **A3:** USE the validated energy-augmentation — rank the frontier by `P(frame_change)·(−ΔE)` over the v3
  structural features.
- **A4/A5:** deepen ar25/ka59 (HUD registers) + an adapter game to L2 (level-up guarantee).
- Reserved: submitted-agent scoreboard + a lazy/cheap value-eval prototype (so a future weight>0 doesn't
  pay full v3 cost per node), hardware continuity, affordance-SOTA ingestion, capstone.

`reproducible_total_levels` is not the headline — held-out solve-rate + action efficiency are.
