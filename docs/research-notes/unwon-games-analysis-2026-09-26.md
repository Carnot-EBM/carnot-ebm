# Why the bare explorer did not level up on 12 public games

Experiment 10017 tested the live explorer on public offline Arcade games. It used no induction, adapter, banked route, or stored engine. All claims below are about this bounded public proxy, not unseen games. The 12 failures have **several causes**. They do not share one inert control or one search depth limit.

## Evidence and counting rules

The aggregate is `results/experiment_10017_explorer_variants.json`. Its `episodes[]` fields `charged_actions`, `levels_completed`, `actions_to_first_levelup`, and `error` give the outcome. The action logs are `results/raw/experiment_10017_explorer_variants/<game>__<variant>__seed-<seed>.jsonl`. Each row's `action`, `data`, `top_branch`, `explorer_branch`, `explorer_serve_kind`, `grid_hash`, and `levels_completed` is recorded **after** the action. The writer is `python/carnot/experiment_10017_explorer_variants.py:run_episode`. `grid_hash` hashes the rendered grid via `arc_agi3_world_model.py:frame_hash`. An unchanged hash proves no rendered-grid change. A changed hash does not prove useful game progress or a changed hidden state.

The files `results/raw/unwon_games_analysis_2026_09_26/game_summary.json` and `episode_summary.json` give exact per-seed values and source paths. `won_comparison.json` gives the V0 win checkpoints. The seed order in each table is 7491001, 7491002, 7491003. `2k³` means 2,000 actions in each of the three seeds. `H` is the count of distinct post-action `grid_hash` values per V0 seed. The last-state column gives the first eight hex digits of each V0 terminal `grid_hash`; the JSON has all 16 digits and the V1/V2 terminal states. This is a compact last *rendered* state, since the log has no frame or hidden state.

Every logged action in these 12 runs has a non-null `explorer_branch`: **100% explorer-selected**. One action per run has `top_branch=induce.no_plan.explorer`; it still uses the explorer. All 108 episodes have `levels_completed=0` and `error=null`. All end at the budget except `g50t` V0/V1, which stop at 1,982. With no level-up or error, `StepwiseExplorer.is_done` implies the early stop is `explored_out`; this reason is inferred from code, not logged. The logs do not record game-over status.

## Per-game outcome and last rendered state

| Game | Registry `mechanic_class`; registered control model | Logged action IDs | Actions V0 / V1 / V2 | V0 `H` by seed | V0 last `grid_hash` prefix by seed |
| --- | --- | --- | --- | --- | --- |
| ar25 | reflection_mirror_object_alignment; keyboard 1–5 | 1–7 | 2k³ / 2k³ / 2k³ | 1201 / 1279 / 1249 | ddbee803 / 261847ea / a28fda9a |
| cn04 | marker_pair_shape_alignment; move, rotate, select click | 1–6 | 2k³ / 2k³ / 2k³ | 1499 / 1540 / 1536 | f2d16c58 / 2aee13b9 / 275c38c4 |
| g50t | config_toggle_target_offset; keyboard 1–5 | 1–5 | 1982³ / 1982³ / 2k³ | 1234³ | 4169b863³ |
| ka59 | null; mixed push and select click | 1–4, 6 | 2k³ / 2k³ / 2k³ | 1632 / 1623 / 1638 | da31d868 / dcc23ca8 / 0497ff2b |
| ls20 | clean_navigation_shape_color_rotation_step_counter; keyboard 1–4 | 1–4 | 2k³ / 2k³ / 2k³ | 1110³ | cb3d621a³ |
| re86 | pattern_match_sprite_resize; keyboard move and select | 1–5 | 2k³ / 2k³ / 2k³ | 1918³ | a242ff52³ |
| sb26 | color_match_slot_sequence; item click, slot click, validate | 5–7 | 2k³ / 2k³ / 2k³ | 850 / 732 / 818 | e27d4186 / 8631508c / 0ff94225 |
| sc25 | two_phase_cast_grid_then_tank_exit; cast click then tank keys | 1–4, 6 | 2k³ / 2k³ / 2k³ | 397 / 359 / 359 | a18becb4 / 70badc60 / b9c8a66c |
| sk48 | chain_color_reorder; keyboard 1–4 | 1–4, 6–7 | 2k³ / 2k³ / 2k³ | 505 / 436 / 468 | 58601f38 / 645b4802 / fc1224b8 |
| tn36 | program_editor; click only | 6 | 2k³ / 2k³ / 2k³ | 37³ | b97556e0³ |
| tr87 | config_substitution; keyboard 1–4 | 1–4 | 2k³ / 2k³ / 2k³ | 1936³ | a98f13b0³ |
| wa30 | null; keyboard 1–5 | 1–5 | 2k³ / 2k³ / 2k³ | 1699³ | 6d67f9d3³ |

The registry is `ops/arc_solve_registry.yaml:games[]`. Its route uses per-game knowledge. Its `mechanic_class` and `action_model` describe a solved route, not what the live explorer inferred. The observed vocabulary can include 6 and 7 even when the registered route does not. The aggregate and raw logs agree on the action counts and zero level-ups. For `g50t`, V0/V1 have one distinct trace across the three seeds; the same is true for `ls20`, `re86`, `tr87`, and `wa30`. Those three seeds are repeated deterministic traces, not three independent search paths.

## What blocked each game

The numbers below use V0 logs unless a variant is named. A `grid_hash` comparison counts visible frame changes, not successful puzzle steps. `explorer_branch` is mainly `depth_ride.pop_untested` in most games. `pending_drain` and `explorer_serve_kind=navigation` identify paid replay or travel actions.

| Game | Recorded action and state evidence | Best supported blocker |
| --- | --- | --- |
| ar25 | ACTION6 is unchanged on **835/835** clicks over three seeds. ACTION1–5 usually change the grid. The runs still reach 1201–1279 hashes, with new hashes in the last 500 actions. | Bad click candidates waste 12–15% of actions. The remaining search still fails to align selected objects and mirrors. The registered 1–5 control sequence matters; mere motion is not enough. |
| cn04 | Each V0 run makes 993–1006 clicks at 174–190 distinct coordinates. About 84–85% of all actions change the hash. V1 also makes about 1,014–1,040 clicks per seed and gets no level-up. | The click action is often effective, but selection, movement, and rotation must align multiple marker pairs. Click salience alone does not rank closeness to that joint condition. |
| g50t | V0 makes 314 ACTION5 presses; 257 change the hash. It spends 1,272 actions in `explorer_serve_kind=navigation`, then stops at 1,982 with zero levels. V2 adds two replay triggers and reaches 2,000 with no level. | The explorer exhausts its visible frontier while missing a clone/plate history that needs ordered commits and traversal. More unstructured replay is unlikely to suffice. |
| ka59 | V0 clicks 762–794 times; about 596–604 clicks change the hash. The registry gives an 11-action L1 control signature `4,4,4,3,2,3,3,3,6,1,4`. No V0/V1/V2 trace contains that full signature; the longest contiguous matching prefix is 3–4 actions. | A short route still requires a specific push sequence, then selection of the second block. Generic depth-first changes do not keep the required setup intact. Other winning routes remain possible. |
| ls20 | V0 has 1,708 vertical key actions versus 248 horizontal ones in each seed. 98.15% of adjacent hashes differ, and 1,110 hashes appear, yet zero level-ups. V1 is exactly the same trace; V2 replays three times per seed without a win. | The target requires a specific shape/color/rotation tuple and counter-reset path. Visible motion does not expose the goal gradient to this explorer. |
| re86 | Each V0 trace has 853 ACTION1 and 896 ACTION2 moves, but only 29 ACTION5 selection changes. It reaches 1,918 hashes and keeps finding new ones through action 1,999. V1 is identical to V0. | The search mostly moves one selected overlay instead of coordinating selected sources and target coverage. This is a control-order and goal-scoring problem, not a no-op problem. |
| sb26 | V0 makes 1,586–1,612 clicks but uses only ten click coordinates per seed. It makes 322–345 immediate item-row to other-row click pairs and 181–194 ACTION5 validations per seed. Clicks change the hash 948–999 times. | Single pairs and validation are already tried. The missing part is a retained, ordered multi-item slot assignment. Random later clicks can undo useful placements. |
| sc25 | V0 makes 1,138–1,161 clicks, **none** of which changes the hash. Across V0/V1/V2, all 10,416 clicks miss the registry cast-cell region `x=24..59, y>=49`; observed lower clicks are at `x=12..18` or `x=62`. V1 still has zero effective clicks. | The candidate generator does not offer the cast-grid targets. V1 only reorders offered clicks. Tank movement alone cannot complete the two-phase task. The exact cast coordinates come from the registered solve and serve only as a coverage check. |
| sk48 | V0 makes 1,109–1,114 ACTION6 clicks per seed; all leave `grid_hash` unchanged. The registered L1 path is 14 keyboard actions. The trace still spends only about 870 non-click, non-reset actions per seed. V1 leaves 1,105–1,118 no-effect clicks. | The explorer repeatedly pays for an inert control that the registered L1 route does not use. V1 reorders clicks within a candidate list but does not remove them across states. |
| tn36 | Every V0 click changes the hash, but 2,000 actions visit only **37** hashes. No new hash appears after action 361. Each seed makes 324 RESETs; `pending_drain` accounts for 1,622 actions. V1 is the same, and V2 still visits 37 hashes. | This is a click-editor cycle, not a failed click detector. The agent revisits a small program-state set without learning edit-then-execute semantics. V0 ends on the initial reset hash. |
| tr87 | Every V0 action changes the hash; 1,936 hashes appear per seed. ACTION1/2 value cycles occur 1,689 times, selector moves 295 times. The registry's L1 route uses 14 actions, but none of the three identical V0 traces levels up. | The rule target must be inferred. Unrestricted value cycling creates many distinct frames without selecting the correct rewritten sequence. V1 cannot affect this keyboard-only trace. |
| wa30 | Each V0 trace visits 1,699 hashes and changes the hash on 94.95% of adjacent actions. ACTION5 occurs 71 times, with 29 visible changes. All three V0 seeds are the same trace. | The multi-block delivery task requires a persistent placement plan and interaction with a helper robot. Movement creates states but supplies no score for correct partial deliveries. |

These are distinct failure modes. Inert click spending is strong in `ar25`, `sc25`, and `sk48`, but absent in `tr87`, `re86`, `ls20`, and `tn36`. `tn36` cycles a tiny rendered state set; `tr87` makes 1,936 rendered states. `g50t` pays heavily for navigation and explores out; most others run to the budget while still seeing new hashes. Thus neither sparse *visible change*, large action space, nor cycling explains all 12. The common high-level limit is that the explorer has no reliable measure of progress toward a compound goal before the first level-up.

## Contrast with the 13 V0 wins

V0 wins in 33 of the 39 episodes for its 13 winning games. The first level-up occurs at action 7–1,710; the median is 579. Counting from the last RESET through the winning action, the successful contiguous segments have 5–141 non-reset actions, median 26; 31/33 have at most 71. This is **not** a shortest-route claim. `vc33` wins in 7–28 actions; `lp85` in 27–371; `sp80` in 115–304. Yet `dc22` wins only near action 1,650, and `bp35` wins in only one seed. The successful group is not uniformly easy.

A testable pattern is that the current depth-first ride finds goals when a viable local sequence can survive its intervening choices and reset/replay cycle. Short registered routes alone do not predict that: `ka59` has an 11-action registered L1 sequence and `tr87` a 14-action one, yet neither wins. Rendered change rate does not predict it either: won `r11l` changes on every action, while won `lp85` changes on only about 16–24% of actions. A useful follow-up test would compare each episode's longest intact task-relevant setup, not raw hash novelty, against first-level-up probability. The logs lack the task-relevant state labels needed to perform that test here.

## Ranked next attempts with falsification checks

This is one ranking of **cheap, general, non-per-game** mechanisms. It is a priority for experiments, not a predicted solve rate. Each arm should use the same 2,000-action budget and report first level-up, action provenance, visible transitions, and distinct traces against V0. A registered route may explain the task but must not be loaded by the live agent.

| Rank | Game | General mechanism and why this rank | What would falsify the proposed blocker or sufficient fix |
| ---: | --- | --- | --- |
| 1 | sk48 | Learn a cross-state action-effect prior. Suppress an action class after repeated unchanged outcomes, then spend the saved probes on keyboard actions. Over half of V0 actions are inert clicks; the registered L1 route has 14 keyboard actions. V1's click ordering left the click count almost unchanged. | If click suppression increases keyboard probes but matched runs still fail and do not reach new task-relevant chain states, wasted clicks were not the sufficient cause. |
| 2 | sb26 | Search cumulative item-to-slot assignments over the small observed click set. Keep a placement if a visible item/slot relation improves; validate only after several pairs. V0 already tried hundreds of isolated pairs, so a single-then-paired sweep alone is insufficient. V1 still used only ten coordinates and won zero. | If a bounded systematic sweep covers distinct full assignments and validations but no level-up occurs, this click set or assignment model is incomplete. |
| 3 | sc25 | Expand click proposals to repeated grid-like controls and probe coordinates within newly detected regions. Then learn a click phase followed by a movement phase. V1 ranked its existing candidates but never clicked a registered cast cell. V2's return-to-best replay also cannot add missing candidates. | If new in-region probes remain unchanged, the inferred target region is wrong. If they change the grid but no phase transition or level-up follows, targeting alone is insufficient. |
| 4 | re86 | Sweep ACTION5 selection before balanced short directional moves. Score overlap between moved shapes and target-like cells from the live frame. The V0 trace used selection only 29 times and mostly moved vertically. V1 is identical because this game has no clicks. | If the sweep increases distinct selected-piece placements but not target coverage or level-ups, selection imbalance was not the main barrier. |
| 5 | ar25 | Suppress cross-state inert clicks, then favor selection followed by short runs of each direction and score reflected target coverage from the frame. V1 changed click order but still spent 272–301 clicks per seed with zero effect. | If click waste disappears and intact selection/move sequences expand but no target coverage or level-up rises, the missing part is mirror/object reasoning rather than the spent clicks. |
| 6 | ka59 | Use a bounded ordered-control search that keeps a changed push state, tests further movement, and introduces selection after the setup. The known 11-action route shows a compound precondition, while the logs never hold even its action-ID prefix beyond four steps. | If the generic search reaches new two-block arrangements but no level-up, that route-order model or its goal scoring is inadequate. |
| 7 | g50t | Track ACTION5 commits as events in the state key and systematically sweep paths before and after each commit. Prefer a retained clone/plate setup over repeated long navigation. V0 explored out at 1,982; V2 replay reached 2,000 without a win. | If histories with distinct clone/plate outcomes are reached and the task still fails, visible-frontier aliasing or replay cost was not the sufficient cause. |
| 8 | cn04 | Detect movable objects and same-color markers in the current frame. Score the count of correctly paired markers after select/move/rotate tests, then return to the best arrangement. About a thousand V0 clicks per seed already changed the frame; V1 salience did not help. | If marker pairing improves to the registered all-paired condition without a level-up, the proposed progress metric is wrong or incomplete. |
| 9 | ls20 | Learn the effect of trigger visits on player shape, color, rotation, and step counter. Search paths that preserve resets and reach the target with the required tuple. More random movement only repeats the identical V0 trace. | If the agent reaches the visible target with the target tuple and counter budget intact but no level-up occurs, the inferred task state omits a condition. |
| 10 | tr87 | Infer a small symbolic rewrite rule from visible glyph rows, then set editable values deliberately. Every action already changes the hash; no-op pruning and V1 click ranking have no route to help. | If an inferred rewritten sequence visibly matches the editable row and still does not level up, the rule model or comparison is false. |
| 11 | wa30 | Track block, bay, player, and helper positions. Keep placements that improve a delivery count and plan around the helper's motion. V0 sees many changing frames but gives no credit to partial delivery. | If the live frame shows the required delivery count and no level-up occurs, the progress measure is incomplete; if it never increases, the planner did not test the premise. |
| 12 | tn36 | First identify click roles through short causal sequences: edit a program slot, execute it, and observe object displacement. Detect the 37-hash loop and stop replaying it. V1 left the loop intact; V2 still saw only 37 hashes. | If diverse edited/executed programs still leave the object relation unchanged and the run stays near 37 hashes, the assumed edit/execute grammar is wrong. |

A larger budget is a secondary test for `cn04`, `ka59`, `re86`, `tr87`, and `wa30`, because their V0 traces still find new hashes near action 2,000. It is a poor first response to `g50t` (explored out), `tn36` (37-state cycle), or `sc25` (no cast-cell proposals). V2's novelty measure is pixel difference from the root plus replay of a best prefix; the logs show no new first level-up on these 12. Pixel novelty is not a goal score. The later V1b amendment is absent from this worktree, so this note does not assign it action counts or terminal states.

## Limits

The JSONL has no frame pixels, game-over flag, hidden state, object identities, or distance to each game's win condition. `grid_hash` can include harmless counters or animation. A no-change click can still alter hidden state, though the registered action model and repeated outcomes make inert-click hypotheses strong for `ar25` and `sk48`. The registry's solved routes are explanatory context, not evidence that the explorer saw their preconditions. We cannot prove the unique reason for any failed level-up from these logs alone. The ranked mechanisms are falsifiable follow-up experiments.


## Follow-up: V3 inert-click deferral pilot (2026-09-26, append-only)

Branch `inert-click-suppress` (commit 87ad5cb49f, off `explorer-pilot`), not merged — its code
depends on `arc_explorer_variants.py`, which lives only on `explorer-pilot` and was never merged to
main (only the passive per-action provenance, REQ-ARC-WMTE-10016, was split out and merged).

**Design.** Deprioritize a click once CONFIRMED inert from the exact same rendered state, keyed on
`(action, pre-action grid_hash)`. Never fully excludes a candidate; falls back to V0's order if every
candidate at a state is confirmed inert. Default off, `CARNOT_ARC_EXPLORER_VARIANT=V3`. Promotion
rule fixed before running: at least one new first level-up on the 12 unwon games, no lost V0 win, no
more than 10% median-action regression on shared wins.

**Verdict: no promotion.** 75 episodes, 143,986 charged actions, 25 games x 3 seeds. Zero of the 12
unwon games gained a first level-up. All 33 V0 winning seeds held (cd82 seed 7491003 gained an extra
level without losing its win). The shared-win median guard held.

**Why it did not fire.** The mechanism this rule targets — an exact `(action, grid_hash)` repeat —
almost never recurs in these runs. ar25 had zero repeated pairs across all three seeds; sk48 had only
2/1/7. The 835 and 1,100+ "wasted" clicks this note counted for those two games are mostly distinct
exact actions or distinct rendered states, not exact repeats of the same action from the same state.
Net effect on executed inert repeats was +32 (44 more elsewhere against 12 saved), a downstream
consequence of the four times the rule did fire changing later play.

**What this means.** Exact-state click memory is not the missing piece for sk48 or ar25. Their
"wasted clicks" need a broader notion of futility than byte-identical repetition — for example
grouping visually or spatially similar clicks, not just identical ones. 35/35 tests pass; adversarial
verification flags zero artifacts. Public games are a development proxy, not a hidden-game estimate.


## Follow-up: V4 object-signature deferral pilot (2026-09-26, append-only)

Branch `object-sig-defer` (commit 68db99a823, off `explorer-pilot`), not merged — same reason as V3:
its code depends on `arc_explorer_variants.py`, which lives only on `explorer-pilot`.

**Design.** V3's exact `(action, grid_hash)` key almost never repeats. V4 groups clicks more broadly:
by the `objects()`-segmented target's majority color, a floor-log2 area bucket, and a bounding-box
size capped at 4 pixels per side. Once 3+ clicks on a signature are ALL inert, defer every candidate
sharing that signature (never fully exclude; fall back to V0 order if all candidates are deferred).
Same fixed promotion rule as V1-V3: pre-registered before running.

**Verdict: no promotion, and a regression V3 did not have.** 75 episodes, 142,927 charged actions, 25
games x 3 seeds. Zero of the 12 unwon games gained a first level-up, and dc22 seed 7491002 LOST its
V0 win. ft09 gained two extra winning seeds without losing its own win.

**The grouping does fire, unlike V3 — but firing did not help.** ar25 crossed the deferral threshold
on both its tracked signatures in every seed, reaching up to 476 rendered states V0 never saw in that
seed. sk48 deferred its one signature and still spent 23-26 of its actions on the "everything is
deferred" fallback path. sb26 bypassed 25-42 deferred candidates per seed and reached 415-553 new
states. In every case: more exploration, zero level-ups.

**What this means.** Object-signature is a real, more general grouping than exact state repetition —
it correctly identifies "this kind of thing doesn't work" far more often. But avoiding a bad click
just spends the freed budget on a DIFFERENT bad click; visiting more distinct rendered states is not
the same as making progress toward whatever these games actually require. Two candidate-reordering
mechanisms (V1, V4) and two click-avoidance mechanisms (V3, V4) have all failed to unblock any of the
12. The bottleneck looks upstream of "which action to try" — closer to "the explorer has no notion of
what would count as progress." Public games are a development proxy, not a hidden-game estimate.
