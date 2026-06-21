# ARC-AGI-3 submission-path: state + prioritized next steps (2026-06-21 outer-loop)

Grounded by a 5-way parallel investigation of the banked-replay submission path. This note is the
worklist for raising the live-submittable level count toward the package/registry ceiling.

## Two distinct paths (do not conflate)

1. **Generic-explorer cascade** (`E3AgentPolicy`, integration gate `experiment_4536`): for UNSEEN games.
   Reaches ~L1 burning ~7,800 actions; `core_efficiency` pinned at 2.0074; the `.420` capstone is an
   honest NULL (no lever moved it). This is the conductor's long-horizon research lane.
2. **Banked-replay submission** (`scripts/arc3_live_submit.py` -> `arc3_replay_scorecard_metaharness.py`):
   replays our OFFLINE-reproduced action sequences against the live scored env. This is what the
   2026-06-17 FIRST submission used (13 levels / 11 games). **This is the lever for a better leaderboard
   score**, and it was leaving levels on the table.

## What landed this session (commits fb7bff878, 0181af9e4)

- The submit driver was **stale**: hardcoded to the 11-game/13-level set. It now sources its replay set
  **dynamically from the newest validated package** (`results/experiment_*submission_package*.json`),
  intersected with metaharness-replayable games, and **caps each claim to the offline-aggregate
  flat-replay depth** (`results/arc3_replay_aggregate_scorecard.json`) so it never over-claims.
- Recovered **dc22 + vc33** into the metaharness (had banked trajectories on disk, only missing a
  `GAME_ARTIFACTS` pointer). Offline aggregate: **32 -> 34 levels / 16 -> 18 games**, zero quota.
- **Honest submittable floor = 34 levels** (vs the 13-level 2026-06-17 baseline = **+21**). The earlier
  39 was an over-claim (package nominal 41; flat-replay reaches 34 after the +2).

## Why the offline floor is 34, not the package's 45 / registry's 52

| gap source | levels | reason | lane |
|---|---|---|---|
| sc25 L1->L5, tu93 L3->L5, lp85 L4->L5 | **+7** | deeper levels REPRODUCE (exp4468/4436/4372) but are not **flat-banked**: their plans are cell/`click` TOKENS, not the `{action,x,y}` pixel dicts the live replay needs. Blocked on the `clickR,C`->pixel translator (lives in the e3 reproduction machinery; `cellR,C` is a clean affine x=24+5C y=49+5R but `click*` is not derivable from the L1 source). | see Rank A |
| s5i5, sb26, g50t | **+3** | env_matched in the package but `arc_loop_solve_<game>.json` is routing-only (no solution); the real solve artifacts must be LOCATED before registering. The investigation's cited `experiment_44XX_solve_*.json` paths do NOT exist. | Rank B |
| ft09 | **+1** | has a dict trajectory (`arc_explore_trajectory_ft09.json`) but is a **conductor `.421` A4 deepen target** editing `arc_game_adapters.py` — DEFER (file-race). Pure metaharness dict-add once that lands. | conductor-coordinate |
| cd82/sp80/su15/m0r0 L1->L2 (+4) and lf52/re86/bp35 first-contact (+3) | **+7** | reproduced on disk but the exp4473 package froze them at L1 / post-date it. A **package refresh** (reads the registry, never writes it) would expose them; then the driver picks them up automatically. | Rank C / conductor |

Ceiling: ~34 now -> ~38 (Rank B + ft09) -> ~41 (Rank A) -> 45 (package) -> 52 (registry), each contingent on
**live env-match** (operator-gated, unverifiable offline).

## Prioritized next steps

- **Rank A (+7, biggest single gain): flat-bank sc25/lp85/tu93 deeper levels.** Clean unblock is upstream
  and conductor-lane: instrument the deeper-reproduction experiment to EMIT executed *pixel* steps (the
  `executed_steps` pattern exp4341 already uses for L1), then flat-banking is a trivial capture + a
  `RESOLVED_ARTIFACTS` repoint + offline-gate. Do NOT reverse-engineer the `click*` translator by hand.
- **Rank B (+3): locate the real s5i5/sb26/g50t solve artifacts** (the routing-only files are not solves),
  add a per-game label translator into the metaharness load path, offline-gate. Outer-loop-mechanical
  ONCE the real artifacts are found.
- **Rank C (+7): refresh the submission package** to cite the on-disk L2 deepenings (cd82/sp80/su15/m0r0)
  and the new first-contacts (lf52/re86/bp35). Reads `ops/arc_solve_registry.yaml`, never writes it
  (conductor-written this milestone). The driver auto-tracks the newest package, so no driver edit needed.

## Operator runbook (operator-only; submission is External-Publication-gated)

1. `./.venv/bin/python scripts/arc3_replay_scorecard_metaharness.py` — offline sanity, zero quota.
2. `./.venv/bin/python scripts/arc3_live_submit.py` (NO `--submit`) — **VALIDATE** live; opens a scorecard,
   plays every game, prints `claimed Lx -> LIVE Ly MATCH/MISMATCH`, leaves the scorecard OPEN (no record).
3. Go/no-go: keyboard-only paths (tr87 L6, tu93) are layout-robust; absolute-click paths (lp85, tn36 L7)
   risk **live version drift** — the one irreducible, offline-unverifiable risk. Confirm `live_total_levels`
   >= 13. Adding games can only ADD score (monotone-safe; per-game error is caught and the loop continues).
4. `./.venv/bin/python scripts/arc3_live_submit.py --submit` — **SUBMIT** (closes scorecard, irreversible).
