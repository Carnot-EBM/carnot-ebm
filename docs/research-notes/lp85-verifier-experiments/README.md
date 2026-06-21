# lp85 verifier experiments — reusable methods (2026-06-21)

Reusable verifier/search code from the lp85 deep-dive. The lp85-SPECIFIC solving is redundant
(lp85 is already reproduced to L5 via `arc_game_adapters.py:_lp85`); these are kept for the
**general methods + the R²≠gradient finding**, as input to the Phase-3 verifier program. Full
write-up + findings: `../arc-discovery-arc-spans-gaps-jumps-2026-06-21.md`.

| File | Method | Finding it produced |
|---|---|---|
| `button_discovery_sweep.py` | Systematic action-discovery: from a deterministic env, click every grid position, cluster effective clicks into distinct "buttons" by effect-signature. | Disproved "under-exploration" — found exactly the active controls; reusable as `discover_actuators`. |
| `structured_discover_search_advance.py` | The discover→BFS-search→advance loop (per-level action re-discovery). | L1 solved in 5 actions, no LLM; L2 deep (branching-5) → blind BFS insufficient → motivates verifier-routing. |
| `goal_distance_best_first.py` | Verifier-routed best-first over discovered actions (naive positional heuristic). | Naive positional heuristics are FLAT (every action delta-0) → naive verifiers don't route. |
| `source_verifier_routed_search.py` | **Source-derived verifier**: read the game's win-check, build `h = sum(piece→nearest-goal dist)`, route best-first. Includes the budget-high variant. | Real gradient where naive was flat (L1 in 5 nodes; L2 h 27→9). But sum-of-distances has a **local minimum** (L2 stalled at h=9, 20k→90k nodes) → it is NOT the true value function. |
| `learned_verifier_from_pixels.py` | **Learned verifier**: CNN maps grid→distance-to-win (source distance as label); trained on play data. | **THE key finding: high R² ≠ usable search gradient.** R²=0.93→0.989 from pixels alone (deployable, no source access) BUT non-monotonic gradient on solve paths (also reproduced with a 64×64 net and a pairwise-ranking loss). The moat is gradient/ordering-quality along solution paths, not prediction accuracy. |
| `rle_object_displacement_encoder.py` | RLE-lossless + object-displacement delta encoding for induce prompts. | Dominated raw per-cell deltas; not the binding lever (lp85 wasn't an induction problem). |

**Phase-3 takeaway:** train verifiers on the TRUE min-actions-to-win value (retrograde from
win-states → no local minima) with a **path-monotonicity** objective, OR use an exact verifier
where precise alignment defines the win. Prediction-accuracy is the wrong training target.

All scripts run against `carnot.agentic.arc_solver_kit.offline_arcade` (CPU-only). They hardcode
lp85's discovered buttons for the demo; the methods generalize.
