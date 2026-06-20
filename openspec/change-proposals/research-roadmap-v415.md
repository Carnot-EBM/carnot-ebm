# Research Roadmap v415 — turn the 2026-06-20 strategy into execution

**Milestone:** 2026.06.415 · pre-staged by the outer-loop (operator-concurred 2026-06-20) per the
Pre-Staged Roadmap Convention. All experiments codex/gpt-5.5; planner/retro stay Claude Opus 4.8.
Live generator Qwen3.5-9B-MTP, never the 3090s. No leaderboard submission in-loop (operator-only).

## Why this milestone

`.414` closed the public-25 (re86 + bp35 + lf52 → 25/25 first-contacted), wired the stronger generic
stack into the **submitted** agent (A1 integration, parity-gated so the 0.08 "ships bare BFS" divergence
can't recur), and built features v3 + the discriminative head. `.415` executes the score-drivers this
session validated:

**Score-drivers (lead):**
- **A1 — Human-replay frame-change predictor + action prior.** The #1 score lever (action efficiency =
  the scoring metric, which we have *none* of at 0.08). The ARC Public Demo human replays are a CONFIRMED
  ready supervised corpus — 14,672 `(state, action, click) → frame_delta, level_progress` examples across
  all 25 games; humans change the frame on 97% of actions. The leaderboard leader's (StochasticGoose) core
  technique, now de-risked with real data. (`arc-human-replay-application-spec.md`.)
- **A2 — World-model trust energy** for hidden-state games — the oracle-distinct moat (the EBM thesis's
  one load-bearing ARC slot). (`arc-world-model-trust-energy-spec.md`.)
- **A3 — Energy-augmentation LOO gate.** Does energy over the v3 *structural* features now transfer
  cross-game (LOO-AUROC > 0.6 vs the 0.503 baseline)? The whole differentiator lives or dies here.
  (`arc-energy-augmented-strategy.md`.)
- **A4/A5 — deepen to L2** (the level-up guarantee): goal-predicate verify + HUD-register state
  (re-scoped `.414` A4 → ar25/ka59), and a derive-from-env GameAdapter deepen (cd82/lf52/r11l/ls20).

**Reserved:** B1 human-replay corpus staging (license-clean), B2 submitted-agent held-out-solve-rate
scoreboard (track the real signal, not levels), C hardware continuity, D SOTA imitation-learning
ingestion, E capstone.

`reproducible_total_levels` is **not** the headline — generic held-out solve-rate + action efficiency +
variant transfer (currently 7/25) are. The wall is generalization (human 100% is first-contact; ours is
not), and every task is measured against it.

## Provenance

Outer-loop Claude, grounded in the 2026-06-20 session: leaderboard competitive intel, the energy-augmented
strategy directive, the human-replay corpus verification, and the deepen+variant sweep's per-game RE map.
