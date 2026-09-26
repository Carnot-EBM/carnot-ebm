# Rescoring the 2026-09-26 explorer pilots with the ARC scorecard

The old promotion gate counted first level-ups and rejected any lost V0 win or shared-win median action regression above 10%. That gate is in `python/carnot/experiment_10017_explorer_variants.py:184-229` and the later pilot summaries. The correction in `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md:42-86` says the graded quantity is different. This note reads existing logs only. It does not run an agent.

## Sources and method

The installed `arc-agi` package is version 0.9.8. `arc_agi/scorecard.py:146-183` stores `min((baseline_actions / actions_taken)^2 * 100, 115)` for a completed level and zero for an incomplete level. `EnvironmentScoreCalculator.to_score()` at `scorecard.py:185-206` uses level indices as weights, includes incomplete levels in the denominator, then clamps to the human-speed ceiling for the completed levels. The gateway differences cumulative level-up checkpoints at `scorecard.py:474-491`. I called this installed calculator with those same per-level inputs, as `scripts/arc_leaderboard_eval.py:1029-1073` does. The full source path is `/home/ianblenke/github.com/ianblenke/carnot/.venv/lib/python3.12/site-packages/arc_agi/scorecard.py`.

The parent checkout's downloaded SDK environment metadata supplies `baseline_actions` for **all 25 games**. The SDK scans those `metadata.json` files into `EnvironmentInfo` (`arc_agi/base.py:208-225`), which the scorecard reads at `scorecard.py:474-490`. Examples: `environment_files/cd82/fb555c5d/metadata.json:8-15` gives `[55,8,41,21,23,23]`; `environment_files/r11l/495a7899/metadata.json:8-15` gives `[22,33,51,26,52,49]`. The JSON evidence records the exact metadata path and complete vector for each game. `ops/arc_solve_registry.yaml:221-242,249-277` calls several cached transfer experiments' *agent* first-level action counts `baseline_actions_to_first_levelup: 2.0`; those are V0 comparison counts, not human baselines. No human baseline had to be guessed.

I used aggregate `episodes[].game`, `variant`, `seed`, `actions_to_first_levelup`, `levels_completed`, `charged_actions`, and `raw_evidence`. The raw JSONL fields are `action_index`, `action`, and `levels_completed`. The runner writes one row for each move, including `"RESET"`, and defines `charged_actions` as rows written (`python/carnot/experiment_10017_explorer_variants.py:123-172`). I checked every aggregate against its raw row count, first level-up index, and peak level. I took each first crossing of a new level as its checkpoint. The remaining trace goes to the first incomplete level and scores zero. Three seeds, 7491001–7491003, are scored separately; tables show their mean game score. This is a **trace-index rescore using the real formula**, conditional on the trace row count being the gateway charge count. The RESET limit is below.

V0 uses `results/experiment_10017_explorer_variants.json` and its local raw files. V1/V2 aggregates and raw files were read with `git show explorer-pilot:<path>`; V3 with `git show inert-click-suppress:<path>`; V4 with `git show object-sig-defer:<path>`. V5/V6/V6a/V6b/V6c/V7/V8 use `results/experiment_10020_*` through `experiment_10023_*`. No branch checkout or experiment run was made. The evidence file is `results/raw/rescore_arc_formula_2026_09_26/per_game_per_arm_scores.json`; it has 729 individual episodes with each level's baseline, trace actions, stored score, and game score. V6a–V6c were only run on the six games below.

**Sanity check.** A one-level 20-action baseline solved in 15 actions stores 115.0 and returns game score 100.0. For an eight-level game at baseline speed, solving levels 1, 2, 4, or 8 yields 2.777778, 8.333333, 27.777778, or 100.0. These reproduce `docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md:67-79` against the installed calculator.

## Disputed games

Each cell is the **three-seed mean game score**, on the scorer's 0–100 scale. `Δ` is versus V0 for that game. The raw JSON contains the three scores and every per-level score. Baseline vectors have 6/6/10/6/8/6 levels for cd82/dc22/lf52/m0r0/sk48/r11l. All recorded disputed episodes reach at most level 1 except cd82: V1 has two level-2 seeds; V3, V6, V6a, and V6b each have one. Those second levels are included.

| Arm | cd82 | dc22 | lf52 | m0r0 | sk48 | r11l |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 0.584837 | 0.005963 | 0.003058 | 0.002834 | 0.000000 | 0.003268 |
| V1 | 0.041389 (-0.543448) | 0.004234 (-0.001730) | 0.003449 (+0.000391) | 0.001968 (-0.000866) | 0.000000 (+0.000000) | 0.003279 (+0.000011) |
| V2 | 0.415033 (-0.169803) | 0.003644 (-0.002319) | 0.003225 (+0.000167) | 0.000435 (-0.002399) | 0.000000 (+0.000000) | 0.003316 (+0.000048) |
| V3 | 0.586305 (+0.001468) | 0.005963 (+0.000000) | 0.003058 (+0.000000) | 0.002834 (+0.000000) | 0.000000 (+0.000000) | 0.003268 (+0.000000) |
| V4 | 0.584837 (+0.000000) | 0.003922 (-0.002042) | 0.003058 (+0.000000) | 0.002834 (+0.000000) | 0.000000 (+0.000000) | 0.003268 (+0.000000) |
| V5 | 0.000000 (-0.584837) | 0.001425 (-0.004538) | 0.000000 (-0.003058) | 0.000000 (-0.002834) | 0.005200 (+0.005200) | 0.003268 (+0.000000) |
| V6 | 0.056202 (-0.528635) | 0.005393 (-0.000571) | 0.002675 (-0.000383) | 0.002560 (-0.000274) | 0.005200 (+0.005200) | 0.003268 (+0.000000) |
| V6a | 0.056202 (-0.528635) | 0.005393 (-0.000571) | 0.002675 (-0.000383) | 0.000822 (-0.002012) | 0.005200 (+0.005200) | 0.003268 (+0.000000) |
| V6b | 0.023301 (-0.561536) | 0.001425 (-0.004538) | 0.001456 (-0.001602) | 0.000000 (-0.002834) | 0.005200 (+0.005200) | 0.003268 (+0.000000) |
| V6c | 0.000000 (-0.584837) | 0.001425 (-0.004538) | 0.000000 (-0.003058) | 0.000000 (-0.002834) | 0.005200 (+0.005200) | 0.003268 (+0.000000) |
| V7 | 0.000000 (-0.584837) | 0.001425 (-0.004538) | 0.000000 (-0.003058) | 0.000000 (-0.002834) | 0.000000 (+0.000000) | 0.003268 (+0.000000) |
| V8 | 0.058674 (-0.526163) | 0.005393 (-0.000571) | 0.003008 (-0.000050) | 0.001451 (-0.001383) | 0.000000 (+0.000000) | 0.003268 (+0.000000) |

### V0 across all 25 public games

The table uses the same three seeds in order. `L` is completed levels by seed. V0's mean across the 25 per-game means is **0.171888**. Thirteen games have at least one level-up; 12 have none. The score is a public-game proxy, not a hidden-leaderboard score.

| Game | L by seed | Score 1001 | Score 1002 | Score 1003 | Mean |
|---|---|---:|---:|---:|---:|
| ar25 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| bp35 | 0 / 0 / 1 | 0.000000 | 0.000000 | 0.001896 | 0.000632 |
| cd82 | 1 / 1 / 1 | 0.015184 | 0.504351 | 1.234976 | 0.584837 |
| cn04 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| dc22 | 1 / 1 / 1 | 0.005669 | 0.006126 | 0.006096 | 0.005963 |
| ft09 | 1 / 0 / 0 | 0.006658 | 0.000000 | 0.000000 | 0.002219 |
| g50t | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ka59 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| lf52 | 1 / 1 / 1 | 0.002916 | 0.002931 | 0.003328 | 0.003058 |
| lp85 | 1 / 2 / 1 | 0.005832 | 0.039534 | 1.101204 | 0.382190 |
| ls20 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| m0r0 | 1 / 0 / 0 | 0.008502 | 0.000000 | 0.000000 | 0.002834 |
| r11l | 1 / 1 / 1 | 0.003038 | 0.003461 | 0.003306 | 0.003268 |
| re86 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| s5i5 | 1 / 1 / 1 | 0.002747 | 0.001193 | 0.001247 | 0.001729 |
| sb26 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| sc25 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| sk48 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| sp80 | 1 / 1 / 1 | 0.078372 | 0.330664 | 0.547664 | 0.318900 |
| su15 | 1 / 1 / 1 | 0.003208 | 0.004337 | 0.004688 | 0.004078 |
| tn36 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| tr87 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| tu93 | 2 / 2 / 2 | 0.344516 | 0.344516 | 0.344516 | 0.344516 |
| vc33 | 2 / 2 / 2 | 7.274286 | 0.411734 | 0.242885 | 2.642968 |
| wa30 | 0 / 0 / 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## What moves on the graded scale

- **cd82:** The median first level-up moves from V0's 169 actions to V6's 589. With the real level-1 baseline 55 and six level weights totaling 21, those particular seeds score **0.504351 vs 0.041522**, a **0.462829-point** drop, or about 12.1-fold. The three-seed mean drops **0.584837 to 0.056202** (−0.528635). This is measurable, though both numbers are small on a 0–100 game scale. It is not correct to call this pair identical at zero. The two compared level-1 traces have 5 and 22 `"RESET"` rows respectively. If every such row were free, with other rows still charged, the scores would be 0.535573 and 0.044806; the difference remains. This is a RESET-only sensitivity, not an asserted gateway correction. The V6 seed that reaches level 2 has checkpoints 427 and 557; level costs 427 and 130. Its stored level scores are 1.659089 and 0.378698, yielding game score 0.115071. The second level matters, but does not erase the regression. V1's two level-2 seeds also score only 0.070654 and 0.032803 because their second-level costs are 1153 and 924 against an 8-action baseline.
- **dc22, lf52, m0r0, sk48, r11l:** First-level baselines are 59, 32, 30, 61, and 22. Every disputed pair's three-seed mean delta in these five games is at most **0.005200 point** in absolute value. Thus the binary lost wins on dc22/lf52/m0r0 and the new sk48 win are nearly invisible to the score. The sk48 win shared by V5/V6/V6a/V6b/V6c is one 814-action seed: its stored first-level score is 0.561579, but eight-level weighting makes its game score **0.015599** for that seed and **0.005200** across three seeds. V0 and V5–V8 all score about **0.003268** on r11l across seeds; V1/V2 differ by at most 0.000048 in mean.
- **Post-solve tail:** It goes into the first incomplete level's zero-score bucket (`scorecard.py:478-490`). Cutting actions after the last level-up does not improve the score. Extra depth can: cd82's level-2 examples show this directly.

Fable's central objection is **mostly validated for binary wins and 10% action changes**: the threshold has no special meaning in the scorecard, and five of the six disputed games have deltas below 0.0052. Its stronger suggestion that the whole cd82 169-versus-589 debate is numerically indistinguishable from zero is **refuted** by the 0.462829-point seed difference and 0.528635-point three-seed mean loss. V0 itself is weak: 0.171888 mean over 25 public games, with most of its score concentrated in vc33, cd82, lp85, tu93, and sp80. This does not estimate performance on private games (`docs/research-notes/arc-agi3-kaggle-submission-requirements-2026-06-17.md:27-40`).

### Effect on the promotion rule

For the nine arms run on all 25 games, the equal-game mean of three-seed mean scores is:

| Arm | 25-game mean | Δ vs V0 | Games with any level-up |
|---|---:|---:|---:|
| V0 | 0.171888 | +0.000000 | 13 |
| V1 | 0.196656 | +0.024768 | 13 |
| V2 | 0.154164 | -0.017724 | 12 |
| V3 | 0.171946 | +0.000059 | 13 |
| V4 | 0.172139 | +0.000251 | 13 |
| V5 | 0.181882 | +0.009995 | 11 |
| V6 | 0.141343 | -0.030545 | 14 |
| V7 | 0.180304 | +0.008416 | 10 |
| V8 | 0.139467 | -0.032421 | 13 |

The old promotion rule should have used paired **game-score changes and depth**, with the same full level list and human baselines, before interpreting new or lost first level-ups. On this public proxy V6's extra sk48 win is worth +0.005200 in that game, while its cd82 and tu93 losses are −0.528635 and −0.267679. Its 25-game mean falls from 0.171888 to **0.141343**. The no-promotion verdict for V6 remains sound, but its reason is score loss, not a generic 10% median action guard. V8 also loses score (0.139467). V6a–V6c have only six-game data, so no 25-game promotion comparison is supported.

The score formula changes the reading of other pilots. V1 has **0.196656** against V0's 0.171888 despite cd82 falling 0.543448; gains on sp80 (+0.817400), lp85 (+0.213184), and ft09 (+0.123667) more than compensate in this public proxy. V5 and V7 also have slightly higher 25-game means (0.181882 and 0.180304), mainly from vc33 gains of +1.162876 and +1.143396. Those gains are far larger than the sk48 first-level gain. No pilot establishes a private-game improvement from three public seeds. The published blanket `complete_no_promotion_public_proxy` verdicts are therefore reasonable deployment decisions, but their win-count and action guard do not rank the arms by the graded quantity. V1, V5, and V7 would merit a score-based follow-up if new runs were allowed; this research task makes no such run.

## RESET charging and limits

The pilot's `"RESET"` JSONL rows consume an `action_index`, and its `charged_actions` is simply the row count (`python/carnot/experiment_10017_explorer_variants.py:125-172`). This differs from the older offline evaluator's `actions` counter, which omits RESET (`scripts/arc_leaderboard_eval.py:889-901,962-980`). The correction note's statement that the offline evaluator is optimistic by reset count (`...submission-requirements-2026-06-17.md:84-86`) cannot be applied as a blanket adjustment to these pilots.

The gateway's Card charges ordinary RESET actions (`scorecard.py:701-704,834-845`), but a full reset starts a new play, and scorecard updates require a nonempty frame (`arc_agi/wrapper.py:186-195`). The project's later gateway audit found that post-death empty-frame actions make a simple `actions + resets` reconstruction wrong (`scripts/arc_leaderboard_eval.py:903-925`). The JSONL does not record frame emptiness, `full_reset`, or the Card's `actions_by_level`; thus the exact gateway charge for these old episodes cannot be recovered from these files. The reported numbers are exact outputs of the installed score formula **for the recorded trace indices**, not observed live Card scores. A charge correction would change the scores; these files do not support a universal correction factor.

There are 77 downward `levels_completed` transitions across the 729 rescored traces. None occurs in the six disputed games. Two occur in V0 vc33 traces, so its 25-game contribution is especially sensitive to how the gateway selects a play (`scorecard.py:567-603`). The evidence records `trace_level_drops` and `trace_level_rise_events` for each episode. The pilot aggregate's peak level is not a saved gateway Card. This is a remaining limit on the 25-game means, especially V5/V7's vc33 gains. There were no missing metadata baselines and no aggregate/JSONL count or first-checkpoint mismatches.

To reproduce, read the source aggregate and each `raw_evidence` path (historical files via `git show`), take the first `action_index` where `levels_completed` first exceeds each prior peak, difference successive checkpoints, and call `EnvironmentScoreCalculator.add_level(level_index, completed, actions_taken, baseline_actions)` for every metadata baseline. Call `to_score(include_levels=False).score`. The JSON evidence preserves the complete input paths, checkpoints, per-level scores, reset-row counts, and per-seed output. The analysis script used for this note is `/tmp/rescore_arc_formula_2026_09_26.py`, outside the repository.
