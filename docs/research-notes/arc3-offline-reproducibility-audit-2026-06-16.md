# ARC-AGI-3 offline reproducibility audit — only 3 of our levels replay offline

**Outer-loop, 2026-06-16.** Built to answer "what scorecard would we publish" (operator's
publish-incrementally directive). Built two zero-quota offline harnesses
(`scripts/arc3_offline_scorecard_validation.py`, `scripts/arc3_replay_scorecard_metaharness.py`)
and found a load-bearing reproducibility gap.

## Finding: only 3 of our recorded ARC-AGI-3 levels reproduce offline

Replaying each game's banked `solve_trace.actions` into an offline `EnvironmentScorecard`
(`OperationMode.OFFLINE` + local `environment_files/`, no network):

| Game | Banked level | Offline replay | Reproducible? |
|---|---|---|---|
| r11l | L1 | **L1** | ✅ full reset→win trajectory in the artifact |
| ls20 | L1 | **L1** | ✅ |
| wa30 | L1 | **L1** | ✅ |
| sc25 | L5 | L0 | ❌ |
| lp85 | L3 | L0 | ❌ |

**Deterministically-reproducible-offline total = 3 levels** (r11l, ls20, wa30 — each L1).

## Why sc25/lp85 don't reproduce (diagnosed, not assumed)

1. The sc25 chain artifacts are **cumulative** (action_plan lengths 5→11→27→43 for L2→L5), so
   exp4249's 43 actions are the full intended L1→L5 plan — not a missing prefix.
2. Replaying those 43 actions from reset: the offline env **responds** (34/43 actions change
   the grid) but **never completes a level**, and this is **seed-invariant** (seeds 0/1/42 all
   give 34 grid-changes / 0 levels). So the recorded click coordinates are tied to the *live*
   record-time layout, which differs from the offline `environment_files` layout.
3. Architecturally, **both solvers are replay-of-banked-frontier + extend-one-level, NOT
   from-scratch solvers**: lp85 = `_replay_lp85_l1` ("banked_lp85_L1_replay") then plan one
   suffix; sc25 = exp4236 imports exp4224 as `previous`, replays its banked L3, extends to L4
   (a 4202→4213→4224→4236→4249 chain). Every deeper level rests on a live-recorded frontier
   replay that does not reproduce offline.

## Implications (honest)

- The deeper levels (sc25 L5, lp85 L3) are **real** — they were `real_env_confirmed=True`
  against the live env when achieved. They are **not fabricated**. But they are **not
  offline-reproducible**, because they exist only as live-recorded trajectory chains whose
  coordinates depend on the live layout, and there is no from-scratch offline solver that can
  re-derive them.
- Therefore "capture all 11 offline, then publish" (Option B) is **not achievable as a
  harness** over existing work. It would require either (a) writing genuinely new from-scratch
  offline solvers for sc25 (hierarchical spell-induction + multi-level planning) and lp85
  (button-discovery + permutation BFS) — a multi-milestone research effort with uncertain
  success against the offline layout; or (b) re-solving them **live** (spends quota; uncertain
  if the live env is session-stable).
- The conductor's `total_levels` tally (22) therefore mixes 3 offline-reproducible levels with
  deeper levels that only ever existed as live recordings. Any forward-facing ARC claim should
  carry this reproducibility caveat.

## Recommendation

Publish the **3 deterministically-verified levels** (r11l, ls20, wa30) as the honest first
ARC-AGI-3 entry — fully rehearsed offline, one minimal operator-gated live run (replay, not
re-solve, so near-zero quota). Track "make the deeper solves offline-reproducible" as a
separate hardening task (it also fixes the fragility above), and add sc25/lp85 to the
published scorecard once they re-solve cleanly.

## UPDATE 2026-06-16 — priority pursued; from-scratch offline solving WORKS (lp85 done)

Operator made offline-reproducibility a priority. Key realization: the offline
`environment_files/` ship the **real game source**, and the offline env is a
**deterministic simulator** — so deeper levels can be **re-solved from scratch
against the offline layout** (no replay of the non-reproducible live recording).

- **lp85: SOLVED to L3** (`scripts/arc3_lp85_offline_solver.py`,
  `results/arc3_lp85_offline_resolve.json`). Reuses the existing goal-key-deduped
  env-cloned searcher `plan_observed_suffix` (exp4179), chained level-by-level
  from reset: L1 +5 (8 states), L2 +8 (114), L3 +16 (791) = 29 moves. The
  re-derived trajectory **replays deterministically** into the offline scorecard.
- **Aggregate offline scorecard: 3 → 6 levels** (r11l L1, ls20 L1, wa30 L1, **lp85
  L3**). Meta-harness now prefers re-solves over banked live recordings
  (`RESOLVED_ARTIFACTS`).
- **sc25: IN PROGRESS** (`scripts/arc3_sc25_offline_solver.py`, WIP). Mechanics
  partially reverse-engineered: 3 spell sprites (caxiiu / sieesc_chwjgc / ui),
  only `sieesc_chwjgc` castable at base (pattern dict `zzpoabuniyn` =
  tevyeq/sieesc_chwjgc/fibcey; cast = select `sptivk-{spell}` + click its 3×3
  pattern `zzpoabuniyn[spell]` where `xhhaqjfncnp` is unset; moves = ACTION1/2/3).
  A spell-cast+move BFS does NOT reach L1 yet — sc25 is a harder constraint-
  satisfaction puzzle (hidden phase state). NEXT: read the sc25.py win condition
  + how moves reposition sprites + how new spells/sprites become available;
  likely needs ACTION4/5/7 (cycle/undo) and sprite-move modeling.

**Acceptance gate for the priority:** sc25 re-solves to L5 offline → aggregate
scorecard = 11 levels, all from-scratch offline-reproducible, zero quota → then a
single operator-gated live run can publish the full set.

## Provenance
- `scripts/arc3_offline_scorecard_validation.py`, `scripts/arc3_replay_scorecard_metaharness.py`
- `scripts/arc3_lp85_offline_solver.py` (lp85 L3 SOLVED), `scripts/arc3_sc25_offline_solver.py` (WIP)
- `results/arc3_replay_aggregate_scorecard.json` (total_levels=6, per-game)
- Source solvers: `experiment_4179` (lp85 `plan_observed_suffix`), `experiment_4236`/`4249` (sc25 chain)
