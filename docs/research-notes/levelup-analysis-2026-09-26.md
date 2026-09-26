# What caused the first live level-ups in Experiment 10009?

Research only. No production path changed.

## Method and denominator

I used `results/raw/experiment_10009_b2_induction_gate_measurement_v3/live_session.json:episodes[].action_rows` and `induction_gate_telemetry.jsonl:record_type,step_index,phase,planned,skipped`. I used the earlier all-episode offline replay in `results/raw/goal_induction_design_2026_09_25/live_replay.json:episodes[].first_levelup_action_index,levels_completed`. I also replayed all 39 started traces with each recorded seed. The replay read `frame.levels_completed`. The harness's `peak_level` reader is broken (`docs/research-notes/b2-induction-failure-triage-2026-09-23.md:134-138`).

There were **39 started sessions** and 105 unstarted sessions. Seven started sessions reached L1. None reached L2. All seven were vc33 or sp80. Every started seed of a given game had the **same action and data sequence**, so there were only **12 distinct traces**. The 7/39 count is not seven independent discoveries. The sample is timed and incomplete. It is not a hidden-game win rate. Per-session records and source hashes are in `results/raw/levelup_analysis_2026_09_26/summary.json`. All `sessions/` paths below are relative to that evidence directory.

All 39 started episode rows say `adapter_disabled=true`, `banked_trajectories_disabled=true`, `stored_engines_disabled=true`, and `game_source_read=false`. The recorded policy was `E3AgentPolicy` (`live_session.json:episodes[].policy_entry`). The 42 telemetry induction attempts all have `planned=false`. The 33 `world_model_hypothesis_gate` decisions have `accepted_by_threshold=false` and `plan_found=false`. Thus the wins were caused by live exploration actions, not an accepted induced engine, executed model plan, adapter, stored engine, or banked route. Induction calls consumed time but produced no plan. The 180 recorded actions per started session came from the explorer; no action came from `execute.plan_step`.

## The seven wins

`action_index` below is one-based and includes the first `RESET`. `before win` counts induction attempts whose telemetry `step_index` precedes that action. The last 20 actions for **each** session, with phase and provenance assessment, are in its `sessions/<game>__seed-<seed>.json:last_20_actions_to_first_levelup`.

| Game | Seed | L1 action | Winning action | Induction before win | Explorer attribution |
|---|---:|---:|---|---|---|
| vc33 | 7491001 | 11 | `ACTION6` click `(61,33)` | 0 | Depth-first ride, current-node untested action draw; reconstructed |
| vc33 | 7491002 | 11 | `ACTION6` click `(61,33)` | 0 | Same trace and reconstruction |
| vc33 | 7491003 | 11 | `ACTION6` click `(61,33)` | 0 | Same trace and reconstruction |
| vc33 | 7531001 | 11 | `ACTION6` click `(61,33)` | 0 | Same trace and reconstruction |
| sp80 | 7491001 | 44 | `ACTION5` spill commit | 1 `proposer_failed` at step 28 | Depth-first ride, current-node untested action draw; reconstructed |
| sp80 | 7491002 | 44 | `ACTION5` spill commit | 1 `exception` at step 28 | Same trace and reconstruction |
| sp80 | 7491003 | 44 | `ACTION5` spill commit | 1 `exception` at step 28 | Same trace and reconstruction |

In vc33 the last 11 actions are `RESET`, then ten clicks. The lower control at `(61,33)` is clicked at actions 6, 7, 9, and 11. Action 11 advances L1. The registered L1 solution uses three lower-control clicks (`ops/arc_solve_registry.yaml:game=vc33,action_model`; `results/arc_loop_solve_vc33.json:solution_labels`). Extra clicks and other controls make the live path 10 moves after reset. A CPU-only replay of the **current** policy matched all 11 recorded prefix actions in each seed. It labeled actions 2–11 `explore.explorer` / `depth_ride.pop_untested` (`results/raw/levelup_analysis_2026_09_26/provenance_reconstruction.json:sessions[].last_20_provenance`; `python/carnot/agentic/arc_competition_agent.py:4792-4855,7787`).

In sp80 the explorer tried click, move, and commit actions, then `RESET` at action 32. Its last 20 actions are 25–44. The final segment includes keyboard moves, splitter-selection clicks, a commit with no level-up at 39, and the winning commit at 44 (`sessions/sp80__seed-7491001.json:last_20_actions_to_first_levelup`). The registered L1 route is three `ACTION4` moves and `ACTION5` commit (`results/arc_loop_solve_sp80.json:solution_labels[0:4]`; `ops/arc_solve_registry.yaml:game=sp80,action_model`). The earlier induction attempt failed to produce a plan (`induction_gate_telemetry.jsonl:episode_id=sp80:seed-7491001,record_type=induction_attempt,step_index=28,planned=false`). The same is true in the other two seeds. A CPU-only replay using the harness's disabled cross-game loaders matched all 44 actions in each seed. It labels action 32 `frontier.navigate`/reset, action 33 `pending_drain`/probe, and the winning action 44 `depth_ride.pop_untested` (`provenance_reconstruction.json:sessions[].last_20_provenance`; `python/carnot/agentic/arc_competition_agent.py:4804-4923`). The telemetry's `candidate_action_selection.chosen_option_id` is a candidate-ranking observation, not the final action: at sp80 step 43 it names `ACTION4`, while action 44 is `ACTION5`.

The original action rows have no `_prov_top` or equivalent. The provenance replay matched every winning-prefix action after seeding the policy and disabling the same cross-game loaders as the live harness (`python/carnot/experiment_7471_v654_arc_seam_observation.py:_disable_cross_game_loaders`). It stubbed the induction call to avoid an LLM. Its labels are a reconstruction, not original captured provenance. The stub can change induction-side state even when the subsequent actions match (`provenance_reconstruction.json:method`).

## The 32 sessions without L1

Each of these sessions spent **180/180 recorded actions in explorer-controlled play**, including `RESET`s. Each made one induction attempt, and none installed a plan. That is 5,760 actions and 32 failed or aborted induction attempts across the 32 sessions (`summary.json:sessions`; `sessions/*.json:phase_breakdown,induction_attempts`). The telemetry records some decision rows in `phase=induce`, but the policy returned to explorer actions because no attempt planned. These are action-source counts, not wall-clock phase shares.

For a cheap closeness check, I replayed the recorded actions and each registered L1 route in the offline environment. The table shows the smallest pixel Hamming distance to the rendered route state **just before** its winning action. It compares 64×64 grids within a game. It is not a distance to the hidden win predicate. `Exact stage` is the furthest identical rendered pre-win route frame reached, with 0 meaning only the route opening. Route length comes from `results/arc_loop_solve_<game>.json:solution_labels` via `python/carnot/experiment_10012_gate_usefulness.py:_first_level_prefix`; wa30 uses the first 33 labels of `results/outer_loop_fable5_wa30_probe_l9.json:action_sequence`. All repeated seeds of a game have the same values. Full measurements and closest action indices are in `sessions/*.json:registered_route_state_comparison`.

| Game | Sessions | Live action vocabulary | L1 route moves | Closest pixels | Exact stage |
|---|---:|---|---:|---:|---:|
| sb26 | 4 | click, commit, `ACTION7` | 9 | 97 | 1 |
| su15 | 4 | click, `ACTION7` | 7 | 18 | 1 |
| g50t | 3 | keyboard | 17 | 81 | 0 |
| m0r0 | 3 | keyboard, click | 15 | 100 | 0 |
| dc22 | 3 | keyboard, click | 23 | 9 | 0 |
| wa30 | 3 | keyboard | 33 | 116 | 2 |
| ka59 | 3 | keyboard, click | 11 | 36 | 1 |
| bp35 | 3 | keyboard, click, `ACTION7` | 17 | 1,226 | 1 |
| ft09 | 3 | click | 4 | 72 | 0 |
| ar25 | 3 | keyboard, click, `ACTION7` | 15 | 148 | 1 |

The closest comparisons can still miss the win. dc22 came within 9 pixels of its registered pre-win image at action 117, but did not reach L1. su15 came within 18 pixels. Both remained at `levels_completed=0` (`sessions/dc22__seed-7491001.json:registered_route_state_comparison`; `sessions/su15__seed-7491001.json:registered_route_state_comparison`). A raw image distance can be small while a required control state is wrong. The action-only route comparison is weaker: g50t and m0r0 contain their entire L1 route labels as **ordered subsequences with extra actions**, yet did not win. Their exact prefixes from a reset are both zero (`sessions/g50t__seed-7491001.json:route_action_match`; `sessions/m0r0__seed-7491001.json:route_action_match`). Extra actions change the state. The recorded `state_sha256` must not be used as a rendered-state count: g50t has 8 distinct recorded hashes but 139 distinct rendered grids in the seeded replay (`python/carnot/experiment_7491_e6_timed_live_profile.py:1266`; `sessions/g50t__seed-7491001.json:distinct_recorded_state_sha256,registered_route_state_comparison.distinct_rendered_grids`).

## What distinguishes the winners?

| Comparison | vc33 | sp80 | What the comparison supports |
|---|---|---|---|
| Registry mechanic | support-clearance configuration | spill/splitter placement | Two different puzzle types |
| Registered L1 route | 3 clicks | 3 moves, then commit | Both have short first routes |
| Live first win | action 11, click | action 44, commit | Explorer found each through a longer path |
| Live vocabulary | clicks only | clicks, keyboard, commit | No common winning action class |
| Closest pixels to route pre-win state | 7 | 17 | Both approached a known winning configuration; neither exactly replayed that route state |
| Induced plan before win | none | none; failed attempt at step 28 | Induction did not choose either winning action |

The registered route lengths are `ops/arc_solve_registry.yaml:games[].action_model` and the two `results/arc_loop_solve_<game>.json:solution_labels` files. The live actions and rendered-state distances are in `sessions/vc33__seed-7491001.json` and `sessions/sp80__seed-7491001.json`. Short routes may help a bounded explorer, but they do not explain everything: ft09 has a four-move registered L1 route and failed in all three started seeds. There is no click-only advantage: vc33 wins with clicks; sp80 needs moves and a commit; su15 and ft09 are click games with no win. The two winners share a generic `StepwiseExplorer` path and a real level signal, not a game mechanic or induced goal rule.

The identical within-game action traces rule out a seed-specific lucky outcome in these replays. They do **not** rule out favorable candidate ordering or accidental discovery by the explorer. There is no explorer ablation here. Both winning actions reconstruct as depth-first expansion, but the original exit branch and per-action candidate-rank cause were not recorded. The source branch rules are in `python/carnot/agentic/arc_competition_agent.py:4792-4923`; the recorded rows in `live_action_rows.jsonl` contain actions and state hashes, not `_prov_top`.

## Implication for a hidden game

The only demonstrated first-win mechanism in these 39 started sessions is **environment-grounded exploration**: propose visible actions, keep playing, and recognize the actual `levels_completed` increment. No accepted induced model or plan won a level in this run. An induced engine may still help in another run; this experiment gives no positive example of that path.

The most direct next test is to improve explorer action choice and return to promising states while keeping the real level increment as the success test. Target visible controls and commits, and measure state-changing actions from the rendered grid. vc33's repeated support clicks and sp80's reset/retry show why these choices matter (`sessions/*.json:last_20_actions_to_first_levelup`; `python/carnot/agentic/arc_competition_agent.py:4800-4923`). This is a proposal, not measured lift. Once a first win occurs, bank its real action trace as within-game evidence and test transfer separately; no such bank was used for these L1 wins.

A follow-up should count **distinct game traces**, capture `top_branch`, `explorer_branch`, and `explorer_serve_kind` on every chosen action, and replay every action with `frame.levels_completed`. It should compare explorer variants on new games, not count identical seeds as independent wins. The broken harness level reader and absent original per-action provenance currently block a sharper causal attribution (`python/carnot/agentic/arc_competition_agent.py:7427-7542`; `docs/research-notes/b2-induction-failure-triage-2026-09-23.md:134-138`).

## Follow-up: explorer variants pilot (2026-09-26, append-only)

REQ-ARC-WMTE-10016/10017, branch `explorer-pilot` (commit f494c6cfc3), not merged. All 25 public
games, 3 seeds, the scored 2,000-action limit, with induction, adapters, banked routes and stored
engines off.

| Variant | Games with a first level-up | Median actions to it | Total levels | Lost vs V0 |
|---|---|---|---|---|
| V0 today's explorer | 13 | 579 | 40 | - |
| V1 control-targeting | 13 | 453 | 43 | none |
| V2 return to promising state | 12 | 616 | 35 | tu93 |

Neither variant meets the promotion rule (+2 games, no loss, no >10% action increase on shared wins);
V1 broke the action guard on bp35, cd82, m0r0. Every winning action was an explorer action, never a
plan step. Per-action provenance recording is proven passive (identical traces in 10 paired replays)
and is the part worth merging if wanted.

