# Research Roadmap v414 — Close the public-25 + throughput fixes + variant/verifier boost

**Milestone:** 2026.06.414 · **Pre-staged** by the outer-loop (operator-directed 2026-06-19) per the
Pre-Staged Roadmap Convention. All experiments `agent_type: codex / gpt-5.5` (ARC sprint quota-conserve;
planner/retro stay Claude). Live generator Qwen3.5-9B-MTP, never the 3090s. No leaderboard submission
in-loop (operator-only).

## Why this milestone (REVISED 2026-06-19 — lead with the score-drivers)

The 2026-06-19 step-back gap audit (`ops/verifier_gaps.md` GAP-LIVE-INTEGRATION + GAP-ARCH-*) found the
0.08 first live score's ceiling is the **submitted agent**, not the solver research — and that
`reproducible_total_levels` (what the sprint optimizes) is largely a **mirage** for the leaderboard. So
.414 leads with the things that actually move the score, then closes the public-25 for the level-up
guarantee:

**Score-drivers (lead):**
- **A1 — GAP-LIVE-INTEGRATION (the #1 lever):** the submitted `make_carnot_agent → E3AgentPolicy` ships
  bare BFS (8/32 in-distribution, ~0 OOD) + a 0/6-value LLM tier, `target_levels=1`, `value_weight=0.0`.
  The stronger `arc_strategy_router` + `arc_world_model_dsl` exist but aren't imported. Wire them in; raise
  `target_levels`; forward-edge nav; measure held-out generic solve-rate before/after (frame-only). Integration, not modeling.
- **A2 — GAP-ARCH-FEATURES:** the verifier features are frame-only order-1 → the discriminative head shipped
  this session is in-sample AUROC 0.726 but **LOO 0.503 == chance**. Build `cross_game_features_v3`
  (relational + Δframe + action-conditioned + predicate-distance); re-run the `--discriminative` LOO-AUROC gate.
- **A3 — per-game discriminative lever (shipped):** wire the `DiscriminativeVerifier` + off-path negatives to
  train online per game during exploration and prune traps the steps-to-go value head misses.
- **A4 — GAP-ARCH-GOAL-NOT-VERIFIED + GRID-ONLY-STATE:** verify the E3 goal predicate (not just dynamics);
  add HUD registers to the state — the root cause of the ar25/ka59/ft09 L2 deepening stall.

**Public-25 closeout (level-up guarantee):** A5 re86 (sprite-resize verifier, GAP-4471), A6 bp35+lf52
first-contact → 25/25; A7 variant transfer benchmark.

**Throughput / reserved:** B1 `--no-cov` default lint (the dc22-class block), B2 gated_on decouple +
registry reconcile, C hardware-continuity, D SOTA-ingestion, E capstone.

`reproducible_total_levels` is reported but is explicitly **not** the headline; generic OOD solve-rate +
action efficiency are.

## Provenance

Authored by the outer-loop Claude (commit `[outer-loop]`), grounded in a parallel codebase audit of the
solve pipeline, verifier training, and conductor throughput. See `ops/arc-submission-checklist.md` and the
2026-06-19 session for the supporting analysis.
