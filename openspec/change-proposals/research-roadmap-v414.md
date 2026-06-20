# Research Roadmap v414 — Close the public-25 + throughput fixes + variant/verifier boost

**Milestone:** 2026.06.414 · **Pre-staged** by the outer-loop (operator-directed 2026-06-19) per the
Pre-Staged Roadmap Convention. All experiments `agent_type: codex / gpt-5.5` (ARC sprint quota-conserve;
planner/retro stay Claude). Live generator Qwen3.5-9B-MTP, never the 3090s. No leaderboard submission
in-loop (operator-only).

## Why this milestone

.413 banked dc22 + sc25 (L2–L5) + sb26 (39 → 45 reproducible levels, **22 of 25** public games
first-contacted). Three asks from the 2026-06-19 step-back analysis drive .414:

1. **Close the breadth gap (the 3 remaining unsolved public games).**
   - `re86` (A1): blocked on a **missing** pattern-match-sprite-resize verifier (GAP-4471) — build it, then solve.
   - `bp35` (A2) + `lf52` (A3): never first-contacted (no registry entry) — adapter-free `graph_explore_solve_v2`.
   - Plus `A4`: deepen one shallow L1 game by +1. ≥4 level-up attempts (ARC Level-Up Attempt Guarantee).

2. **Throughput fixes** the .412/.413 diagnosis identified (whole-day stalls):
   - `B1`: make `--no-cov` the **default** for ARC solve preconditions as a durable lint — the dc22-class
     `baseline_pytest_coverage` block (a whole .412 milestone lost) becomes structurally impossible.
   - `B2`: decouple independent solve tasks from `gated_on` cascades; reconcile `reproducible_total_levels`
     (stale 39 vs authoritative 45).

3. **Variant/verifier training boost** (consume the bridge built 2026-06-19):
   - `A5`: wire + **measure** the reflection-augmented v2 cross-game verifier vs un-augmented (honest null
     allowed) — validated this session: reflection regularizes an overfit position-specific weight 214 → 42
     toward the reflection-invariance the held-out 110 eval games reward. Color-permutation is a no-op for
     the color-agnostic features (validated); reflection + v2 is the augmentation that adds signal.
   - `A6`: build the variant transfer benchmark (the `exp4472` test exists, the module does not) — the
     cheapest dev-side proxy for the unseen eval games, since the 25 public games are all there are.
   - `A7`: generic-operator LOO consolidation (amortize the per-game RE delta, the core bottleneck).

Reserved slots: 2 infra (B1/B2), 1 hardware-continuity (C: KV260/GateMate/PolarFire), 1 SOTA-ingestion
(D: neural-guided / induced-world-model search literature), capstone (E).

## Provenance

Authored by the outer-loop Claude (commit `[outer-loop]`), grounded in a parallel codebase audit of the
solve pipeline, verifier training, and conductor throughput. See `ops/arc-submission-checklist.md` and the
2026-06-19 session for the supporting analysis.
