# A1 "stronger agent" is NOT stronger — value_weight=5 is a slow regression (2026-06-20)

Operator asked to sim the A1-integrated submitted agent before submitting. The sim + a controlled
check converge on a clear, honest finding: **A1 wired the stronger stack but did NOT improve
capability, and the value_weight=5.0 it set is a SPEED REGRESSION.**

## Evidence (convergent)

1. **A1's OWN benchmark** (`results/experiment_4475_wire_stronger_generic_stack.json`):
   `before_generic_solve_rate=0.143 (1/7)`, `after=0.143 (1/7)`, **`generic_solve_rate_delta=0.0`**.
   A1 honestly recorded ZERO solve-rate improvement, but shipped the config anyway (verdict
   "complete: ...wired" — the integration succeeded, the capability did not). NOTE: duration_s=0.0 is
   a methodology gap, but delta=0.0 is the conservative/honest conclusion.
2. **Local sim** (25 games, frame-only, budget 2500, 100s cap, A1 config value_weight=5/target_levels=5):
   **1/25 solved, 20/25 TIMED OUT** at 100s. The value-weighted A* search is so slow it can't finish a
   2500-expansion search in 100s on most games.
3. **Controlled check** (same 5 easy games): fast BFS (value_weight=0) FINISHES each in 8-23s (solves
   sp80 L1); value_weight=5 TIMED OUT on all 5 at 100s. -> the slowdown is the value-weighting, not the
   games.
4. **Prior corroboration** (`arc_offline_to_live_bridge_v2.json`): value-weighted A* (w5) REGRESSED to
   6/32 vs the bare-BFS 8/32 ("lost cd82,m0r0; cn04 unlock did NOT transfer").

## Root cause

The cross-game value head is **LOO-AUROC 0.503 = chance** (it does not generalize). Setting
`value_weight=5.0` makes every search node pay an expensive value evaluation that provides NO useful
guidance — pure slow noise. So the "stronger" config is slower AND no smarter. Worse, per-game slowness
is a real threat to the 12h / ~110-game eval budget (~6.5 min/game): a config that can't finish a bounded
search per game will score LOWER, not higher.

## Recommendation (pre-submission)

**Do NOT submit the value_weight=5 config as-is.** Either:
- (A) **Revert `SUBMITTED_AGENT_CONFIG.value_weight` to 0** (the faster, equal-or-better bare-BFS — 8/32,
  finishes games in seconds) + update `test_arc_submitted_agent_parity.py` (which currently asserts
  value_weight>0) consistently. This is the safe submission config NOW. Keep router/world-model-DSL wired
  (those are free); just don't weight the broken value head. OR
- (B) **Hold the submission** until the value head is actually predictive — the `.415` frame-change
  predictor (real human-replay data) + features-v3 LOO gate (A3). value_weight should only be raised once
  the value head clears chance (LOO>0.6).

**Principle:** value_weight is a multiplier on the value head's quality. On a chance-level head, any
value_weight>0 is a regression. Gate value_weight on the LOO-AUROC, not on "we wired it."

Cross-refs: `results/experiment_4475_wire_stronger_generic_stack.json` (delta 0.0),
`arc_offline_to_live_bridge_v2.json` (w5 regress 8->6), `results/arc_discriminative_verifier.json`
(LOO=0.503), `arc-energy-augmented-strategy.md` (the value head only helps over structural features).
