# Go-Explore activation regression diagnosis (2026-09-26)

## Scope and method

This is a trace diagnosis of REQ-ARC-WMTE-10020, not a new pilot. The paired result is in `results/experiment_10020_go_explore_activation.json`, especially `episodes[*].actions_to_first_levelup`, `go_explore_replays_fired`, and `archive_diagnostics`. The V5 action rows are in `results/raw/experiment_10020_go_explore_activation/<game>__V5__seed-*.jsonl`. The matching V0 rows are in `results/raw/experiment_10017_explorer_variants/<game>__V0__seed-*.jsonl`, as the result's `raw_evidence` fields specify. V0 files are not in the 10020 raw directory. Per-seed audit records, including every replay block, are in `results/raw/go_explore_regression_diagnosis_2026_09_26/`.

The source note on `main`, `docs/research-notes/unwon-games-analysis-2026-09-26.md` ("V5 go-explore activation pilot"), reports 9 lost V0 wins, the new sk48 win, and dc22's shared win moving from action 1,649 to 1,969. One premise in the question needs correction: m0r0 loses **one** V0 win, seed 7491001. Its seeds 7491002/3 did not win under V0. The 9 losses are cd82 3, lf52 3, dc22 2, and m0r0 1. All three dc22 seeds are diagnosed because seed 7491003 regressed on the action guard.

`GoExploreReplayArchive.observe()` in `python/carnot/agentic/arc_go_explore.py` stores a shortest known action prefix for each coarse grid cell. `select_prefix()` picks the eligible cell with fewest `visits`, then greatest prefix `depth`; it excludes the current path. `arc_competition_agent.py` calls it when the current node has no open tier or reaches the depth cap. `_begin_go_explore_replay()` queues one `RESET` and then each prefix action. A selected cell is **replayed**, never teleported to. The JSONL `explorer_branch="go_explore_replay"` labels only the triggering `RESET`; its prefix rows have `explorer_branch="pending_drain"` and `explorer_serve_kind="navigation"`. I counted each trigger through its contiguous prefix rows as archive cost. Other `pending_drain` rows belong to normal frontier navigation. Every row's `action_index`, `grid_hash`, and `levels_completed` describe the state **after** that action.

Here, *fresh* means an action with `explorer_branch` `depth_ride.pop_untested` or `frontier.pop_untested`, or `explorer_serve_kind="probe"`. Other cost means non-archive reset or navigation. These disjoint categories sum to each log's `charged_actions`. A V0 winning path means the contiguous action+data and grid-hash sequence from its last `RESET` through its first `levels_completed > 0` row. Prefix matches below compare that sequence with every V5 reset-to-reset segment. This tests actual executed paths; the JSONL does not expose every unexecuted graph edge or the archive's coarse cell key. Target `grid_hash` below is the **observed replay endpoint**, a trace identifier, not the unavailable coarse cell signature.

## First changed decision

Seed suffixes abbreviate 7491001/2/3. `R` is the first archive-triggering reset. `D` is the first differing `action`+`data` row in matched V0/V5 logs. A shared reset can make `D=R+1`. `Depth` counts replayed prefix actions; `cost=depth+1` includes the reset. All first blocks complete in the log.

| Game | Seed | R | D | V0 at D → V5 at D | First target depth; endpoint `grid_hash`; cost |
| --- | ---: | ---: | ---: | --- | --- |
| cd82 | 1 | 45 | 46 | `3` → `5` | 5; `981f0b2f759dd4d8`; 6 |
| cd82 | 2 | 43 | 44 | `1` → `5` | 9; `b342aa60c885cfa7`; 10 |
| cd82 | 3 | 27 | 28 | `1` → `5` | 5; `6373908f69a3cd3a`; 6 |
| dc22 | 1 | 215 | 216 | `4` → `1` | 2; `5de4f03be2a8cdf7`; 3 |
| dc22 | 2 | 309 | 310 | `6 (48,19)` → `6 (48,36)` | 1; `4d064159cce19a9d`; 2 |
| dc22 | 3 | 308 | 309 | `6 (10,40)` → `6 (48,36)` | 1; `4d064159cce19a9d`; 2 |
| lf52 | 1 | 98 | 98 | `6 (24,19)` → `RESET` | 2; `eeeacc82537c4742`; 3 |
| lf52 | 2 | 98 | 98 | `6 (24,19)` → `RESET` | 2; `eeeacc82537c4742`; 3 |
| lf52 | 3 | 166 | 166 | `6 (30,19)` → `RESET` | 2; `eeeacc82537c4742`; 3 |
| m0r0 | 1 | 113 | 114 | `2` → `1` | 9; `02a954f71fd397b3`; 10 |
| sk48 | 1 | 146 | 147 | `2` → `1` | 10; `da11745d01f49b35`; 11 |

The `R` rows and prefix rows can be checked directly in each per-seed JSON's `v5_first_replay` and `first_action_data_divergence` fields, then in the source JSONL at those `action_index` values. For cd82, dc22, m0r0, and sk48, V0 also reset at `R`, so the first different **action** follows the reset. For lf52, the archive reset immediately replaces V0 frontier navigation.

## Charged action budget

Every number below counts JSONL rows, including resets. `V0 R/N/F` means other resets, navigation, and fresh probes over the complete V0 log. `V5 archive` separates archive resets and prefix steps; `V5 F/O` means fresh probes and all other actions. For every row, the listed V5 categories total 2,000. `V0 F→win` is fresh probes through V0's first level-up, so it is a useful search-depth comparison, not a counterfactual prediction. The raw counts and every archive block appear in each game's `results/raw/go_explore_regression_diagnosis_2026_09_26/<game>__seed-749100*.json`, fields `v0_budget`, `v0_fresh_probes_to_win`, `v5_budget`, and `v5_replay_blocks`.

| Game | Seed | V0 win action | V0 total; R/N/F | V0 F→win | V5 archive resets+prefix | V5 F/O | Fires; V5 win |
| --- | ---: | ---: | --- | ---: | --- | --- | --- |
| cd82 | 1 | 974 | 2000; 52/204/1744 | 788 | 260+1563=1823 | 176/1 | 260; none |
| cd82 | 2 | 169 | 1470; 32/58/1380 | 158 | 258+1554=1812 | 187/1 | 258; none |
| cd82 | 3 | 108 | 1409; 36/51/1322 | 104 | 258+1609=1867 | 132/1 | 258; none |
| dc22 | 1 | 1710 | 2000; 36/527/1437 | 1243 | 94+188=282 | 1127/591 | 94; none |
| dc22 | 2 | 1645 | 2000; 33/368/1599 | 1281 | 104+104=208 | 1196/596 | 104; none |
| dc22 | 3 | 1649 | 2000; 39/565/1396 | 1220 | 98+98=196 | 1233/571 | 98; 1969 |
| lf52 | 1 | 799 | 2000; 46/675/1279 | 544 | 606+1211=1817 | 171/12 | 606; none |
| lf52 | 2 | 797 | 2000; 48/664/1288 | 543 | 603+1205=1808 | 180/12 | 603; none |
| lf52 | 3 | 748 | 2000; 48/677/1275 | 502 | 430+1289=1719 | 233/48 | 430; none |
| m0r0 | 1 | 710 | 2000; 153/979/868 | 596 | 99+1361=1460 | 539/1 | 99; none |

V0 stops at action 1,470 and 1,409 for cd82 seeds 2/3 after completing its run; those are not missing trace rows. V5 consumes all 2,000. The `episodes[*].charged_actions` and `archive_diagnostics.selected_prefixes` in `results/experiment_10020_go_explore_activation.json` agree with the row counts and the `Fires` column. Some final replay prefixes are cut by the budget; only charged rows are counted. The source increments `selected_prefixes` when a prefix is selected, even if the episode ends before all queued steps run.

## Did V5 execute the V0 winning route?

`Prefix` is the longest exact action+data **and** after-action `grid_hash` match between V0's last-reset-to-win route and any V5 reset segment. A value below the full V0 route length means the full winning route was absent. `V5 break` is the action just after the best matched prefix, if one exists. `Pre-win visits` counts V5 rows with V0's grid hash immediately before its winning action, at the same `levels_completed`; it is zero in all ten cases. These measures come from `v0_winning_reset_action`, `v0_winning_path_length`, `v5_longest_winning_path_prefix_length`, `v5_winning_path_break_action`, and `v5_pre_win_grid_hash_visits` in the per-seed JSON.

| Game | Seed | V0 winning reset→win | V5 best prefix; break | Route relation |
| --- | ---: | --- | --- | --- |
| cd82 | 1 | 969→974 (5 steps) | 0/5; no match | winning route absent |
| cd82 | 2 | 157→169 (12 steps) | 0/12; no match | winning route absent |
| cd82 | 3 | 98→108 (10 steps) | 0/10; no match | winning route absent |
| dc22 | 1 | 1639→1710 (71 steps) | 10/71; V5 action 1753 | branch off route |
| dc22 | 2 | 1579→1645 (66 steps) | 11/66; V5 action 1989 | branch off route |
| dc22 | 3 | 1582→1649 (67 steps) | 7/67; V5 action 1286 | branch off V0 route; V5 later wins differently |
| lf52 | 1 | 783→799 (16 steps) | 2/16; V5 action 1999 | branch off route; archive resets again |
| lf52 | 2 | 782→797 (15 steps) | 2/15; V5 action 1996 | branch off route; archive resets again |
| lf52 | 3 | 732→748 (16 steps) | 2/16; V5 action 1997 | branch off route |
| m0r0 | 1 | 569→710 (141 steps) | 0/141; no match | winning route absent |

The match is an executed-path test. A common `grid_hash` elsewhere does not prove the same full simulator state or that the explorer would choose the winning next action. Conversely, a missing executed path does not prove the graph lacked a candidate edge. The logs do prove that V5 never reached the V0 pre-win `grid_hash` in these seeds.

## Per-game diagnosis

The labels describe what these traces support. A first replay that changes the action sequence proves a scheduling change. It alone does not prove that the abandoned action at that moment led directly to the later win. Fresh-probe counts compare search opportunity, but equal counts would not guarantee equal choices.

| Game | Classification | Evidence and limit |
| --- | --- | --- |
| cd82 | **BOTH** | The archive takes 1,823/1,812/1,867 actions in 260/258/258 replays, leaving only 176/187/132 fresh probes. Seed 1 needed 788 V0 fresh probes before its action-974 win, so crowding is substantial. Yet seeds 2/3 get **more** V5 fresh probes than V0 needed by their wins (187>158, 132>104), with 0/12 and 0/10 of the V0 winning reset paths executed. Purely counting available fresh probes cannot explain those two losses; action selection was redirected. Only 5 coarse cells survived in each V5 seed (`archive_diagnostics.stored_cells`). |
| dc22 | **INDETERMINATE** | V5 spends 282/208/196 actions on 94/104/98 replays. It has 1127/1196 fresh probes in the lost seeds, short of V0's 1243/1281 probes by its wins. That makes crowding plausible. V5 also branches after only 10/71 and 11/66 steps of V0's winning reset paths, and reaches neither V0 pre-win state. The path breaks at V5 actions 1753/1989 are `pending_drain` navigation, not archive triggers, so the trace cannot assign those choices specifically to a wrong archive cell. Seed 3 still wins at 1969 rather than 1649 (+320 actions), with 196 charged replay actions, showing both replay cost and changed traversal but not their separate causal shares. |
| lf52 | **PASSIVE_CROWDING_OUT** | The first replay replaces V0 frontier navigation at action 98/98/166. Then 606/603/430 replays consume 1817/1808/1719 actions. V5 gets 171/180/233 fresh probes versus V0's 544/543/502 by its action-799/797/748 wins. Its first two seeds finally reproduce the first 2 steps of V0's 16/15-step winning reset path near action 1996/1993, then reset again at 1999/1996. That late interruption is real, but the large fresh-search deficit makes crowding the supported explanation relative to V0's route. Only 3 coarse cells remain per seed, with archived max depth 2/2/4. |
| m0r0 | **PASSIVE_CROWDING_OUT** | Seed 1 spends 1460 actions in 99 replays, leaving 539 fresh probes; V0 needed 596 before its action-710 win. The 141-step V0 winning reset path starting at 569 never begins as an exact V5 reset path. V0 and V5 first differ at action 114, long before that winning reset, so the log does not show the archive actively abandoning the particular winning path. V5 explores elsewhere, and the 73% replay share can account for the missing V0 search opportunity. Seeds 2/3 were not V0 wins. |

For dc22, a controlled replay-cost-only counterfactual is needed to distinguish the candidates: charge the same reset and prefix actions at the same decision points while resuming the original V0 frontier choice, then compare the first level-up and route hashes. A second arm that keeps the archive's cell choice but removes or caps the charge would isolate its direction choice. Neither counterfactual is in the recorded result; it would require a future pilot. This note does not infer it from timestamps alone.

## Positive and clean-winner controls

**sk48, seed 7491001.** V0 takes 2,000 actions, including 1,939 fresh probes, and never levels up (`results/experiment_10020_go_explore_activation.json`, `episodes`, V0 row). V5 first replays at action 146: one reset and ten prefix actions end at action 156, `grid_hash=da11745d01f49b35`. That hash is absent from the complete V0 trace; action 156 is V5's first hash not seen in V0. The next fresh action is 157. Before V5's action-814 win, only four archive blocks have fired, costing **26** actions, less than even dc22's 196 replay actions and far less than the other losses' full-episode costs. The fourth block is reset 629 plus one prefix action at 630; from that return, V5 executes 184 further actions and reaches level 1 at action 814. Its action-813 pre-win hash `bcbfaa7a89536c71` and action-814 win hash `d03d5e8869171cff` are absent from V0. V5's 185-step final reset-to-win path matches at most two initial steps of any V0 reset path. The archive changed the route while replay cost was small **before** the win; the trace does not prove which replay was necessary. The full V5 log later accumulates 72 replays and 1,210 charged replay actions; 68 replays and 1,184 of those actions occur **after** the first win. The total-episode overhead would give the wrong impression of the cost of this win. See `results/raw/go_explore_regression_diagnosis_2026_09_26/sk48__seed-7491001.json`, `v5_replay_blocks`, and the sk48 V0/V5 source JSONL.

**r11l clean winner.** Seeds 7491001/2 have byte-identical V0/V5 `action`+`data` sequences for all 2,000 actions, zero archive fires, and unchanged first wins at actions 871/816. Seed 7491003 wins at action 835 in both arms; its only replay is a three-action block at 1986–1988, after that win. It first differs from V0 at action 1987. Thus archive activation did no harm to the measured first wins because it did not fire until after them (or at all). The result's `archive_diagnostics.selected_prefixes` is 0/0/1 for these seeds. See the paired r11l JSONL and `results/raw/go_explore_regression_diagnosis_2026_09_26/r11l__seed-749100*.json`.

## Design implications for a future pilot

1. **Replay flooding:** Limit consecutive archive replays and their cumulative charged actions. Require a minimum amount of fresh frontier work before selecting another prefix. lf52's 606 fires and 1,817 replay actions, and cd82's roughly 258 fires, show the scale a cap must address. A selector should account for `1 + prefix depth` and remaining budget, not only `visits` and depth.
2. **Route displacement:** Preserve a path with unresolved actions when a replay is proposed. At least allow the current frontier choice to proceed before resetting, and avoid repeatedly selecting a tiny set of cells when the replay reveals no new state. cd82 seeds 2/3 used enough fresh probes in total to cover V0's probe count to win yet never started its winning route.
3. **Late wins:** Judge cost at `actions_to_first_levelup`, not only at action 2,000. On sk48 the winning search paid 26 replay actions; on dc22 seed 3 it paid 196 before its late action-1969 win. Compare a budget-capped replay arm with a direction-preserving cost control before attributing dc22's delay to either cause.

The archive cell signature, internal graph frontier, and unexecuted candidate actions were not serialized. The selected prefix's **executed** depth and endpoint can be recovered; the exact selected coarse-cell key and a no-replay counterfactual cannot. This limits causal attribution, particularly for dc22. No implementation or environment files were changed.
