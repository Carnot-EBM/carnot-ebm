# Depth calibration of the cross-game distance estimate (2026-10-18)

## Question

The reading note `latent-and-coarse-to-fine-reasoning-ncm-hybridagent-coconut-2026-10-18.md`
named Idea B. Is a verifier's distance estimate less reliable far from the goal? The hypothesis
came from Coconut Fig 7. It had not been checked on Carnot data.

## Result in one line

The held-out distance estimate has no skill at any depth. The depth question cannot be answered
from this estimator. The test found a label defect in the trainer.

## Method

- Data: the banked solve traces of the public games, replayed on the offline sim.
  Collector: `collect_pooled` in `scripts/arc_cross_game_verifier_train.py`.
  632 rows from 18 games. Each row is one frame plus its steps to the next level-up.
- Fit: ridge regression (lambda 10) on the frame features. Two feature sets: v1 (5 scalars) and
  v3 (relational, delta, action and predicate context).
- Split: leave one game out. Each game is scored by a model that never saw it.
- Target: log(1 + steps to next level-up). Rows are bucketed by raw steps: 0-1, 2-5, 6-15, 16+.
- Baseline: the mean of the other games' targets.
- Script: `scripts/experiments/outer_loop_depth_calibration_20261018.py`.
  Numbers: `depth-calibration-cross-game-verifier-2026-10-18.data.json`.
- Cost: CPU only. No GPU. No LLM.

## Numbers (held out, log scale, mean absolute error)

| Raw steps | n | v1 model | v1 constant | v3 model | v3 constant |
|---|---|---|---|---|---|
| 0-1 | 74 | 1.89 | 1.83 | 3.57 | 1.83 |
| 2-5 | 139 | 0.82 | 0.71 | 2.53 | 0.71 |
| 6-15 | 251 | 0.47 | 0.28 | 2.03 | 0.28 |
| 16+ | 168 | 1.08 | 1.21 | 2.38 | 1.21 |

- Overall Spearman on the log target: v1 -0.015, v3 -0.174.
- Within-bucket Spearman: between -0.17 and 0.05 for every bucket and both feature sets.
- The model is no better than the constant in three of four buckets for v1. It is worse in
  every bucket for v3.
- The bias has the sign of regression to the mean. Near the goal the model predicts too far
  (+1.9 log units). Far from the goal it predicts too near (-1.1 for v1, -1.2 for v3).

## What this does and does not show

- It shows that this ridge model, on these frame features, does not transfer across games.
  That agrees with the module's own docstring, which calls cross-game transfer a weak signal.
- It does not show that depth matters. A model with no skill has no reliability to lose.
  Idea B is unanswered, not refuted.
- The single bucket where the model beats the constant is 16+ steps (1.08 against 1.21). It has
  168 rows from a few games, within-bucket Spearman is -0.17, and no test was run. Do not read
  it as a finding.
- The rows are replayed solution paths. They are consecutive frames of the same run, so they are
  not independent. A standard error from n=632 would be too small. None is given.
- The estimator is a regressor, not a probability. So this is error by depth, not Expected
  Calibration Error. The Calibrated Decision floor wants a calibration metric. This is a
  precursor, not that metric.
- The shipped `models/arc_verifier_cross_game*.json` files train on all 16 games. Any score of
  them on those traces is in-sample. This test refits leave-one-game-out for that reason.

## Defect found in the trainer

`_steps_to_next_up` in `scripts/arc_cross_game_verifier_train.py` keeps a counter `run` that
rises by one each step back from a level-up, exactly as `d` does. So `run == d + 1`. The label
`d / run` is therefore `d / (d + 1)`. It is not normalized by the segment length, as the
docstring says.

- Effect: the label saturates. 521 of 632 rows have a label of 0.75 or more. No row falls in
  0.25-0.5. The regressor sees little difference between 10 steps and 60 steps from the goal.
- Rank order is kept, so ranking use is less affected than regression use. The live explorer
  orders its frontier by the head's score, so it uses rank.
- Not fixed here. The trainer feeds shipped checkpoints. A change needs its own spec, test and
  retrain. Recorded so the next reader does not re-derive it.
- Check, not prose: the fix would add a regression test that the label for the first row of a
  length-L segment is near 1 and for the middle row is near 0.5. Not built here.

## Open items

1. Decide whether to fix the label and retrain. Re-run this test on the fixed label.
2. A per-game fit (train and test inside one game, split by level) would show whether any depth
   signal exists at all. It was not run.
3. This test does not decide whether Idea B earns a slot. It lowers the expected value of the
   cross-game version.

## Addendum 2026-10-18: label fixed, skill unchanged

The label defect above is fixed in `_steps_to_next_up` (REQ-ARC-WMTE-7830). The divisor is now the
level segment length. Two new tests pin it. The old function fails both.

I reran the same leave-one-game-out test with the fixed label. Numbers:
`depth-calibration-cross-game-verifier-2026-10-18.fixed-label.data.json`.

- The label now spreads evenly. Rows per label quartile: 173, 154, 159, 146. Before: 37, 0, 74, 521.
- Held-out skill did not change. Spearman on the label: v1 -0.032, v3 0.019.
- Mean absolute error on the label: v1 0.252 against 0.250 for a constant guess. v3 0.371 against 0.250.
- Within-quartile Spearman is between -0.09 and 0.15 for every quartile and both feature sets.
- The bias still has the regression-to-the-mean shape: too far near the goal, too near far from it.

So label saturation was not the reason for the missing skill. The frame features do not carry a
cross-game distance signal at the 632-row scale. The saturated label did hide this: it made the
estimator look better-behaved on rank than it is.

Not done, on purpose:

- The shipped `models/arc_verifier_cross_game*.json` checkpoints still use the old label. The live
  explorer loads one of them. A retrain changes the scored agent. Given the result above, a retrain
  would not be expected to help. It needs its own approval and a live A/B.
- No per-game (within-game) fit. That is still the test that could show whether any depth signal
  exists.
- Same limits as before: replayed paths, non-independent rows, a regressor and not a probability.

## Addendum 2026-10-18: within-game fit shows a depth signal for v1

I ran the within-game test. For each game, I train on all but one level of that game. I score the held-out
level. I repeat for every level. Script: `scripts/experiments/outer_loop_depth_within_game_20261018.py`.
Numbers: `depth-calibration-cross-game-verifier-2026-10-18.within-game.data.json`. CPU only.

Only 5 games qualify (dc22, lp85, tn36, tr87, tu93). The rest have fewer than two levels with 20 or more
labeled rows. The test scores 479 rows in 19 to 21 held-out levels.

| Feature set | Pooled Spearman | MAE model | MAE constant | Median Spearman inside one level | Levels with positive Spearman |
|---|---|---|---|---|---|
| v1 | 0.41 | 0.227 | 0.250 | 0.996 | 17 of 19 |
| v3 | 0.47 | 0.220 | 0.250 | 0.942 | 19 of 21 |

- Per game, v1 Spearman is 0.83 (dc22), 0.52 (tu93), 0.20 (tr87), 0.17 (tn36), -0.01 (lp85).
  The model beats the constant on MAE in 3 of 5 games for v1. dc22 gives most of the gain.
- So a depth signal exists inside a game. It does not exist across games (see above). This is the
  split the earlier result left open.

Limits. Read these before you use the numbers:

1. The v3 features include a predicate distance to a goal frame. The harness builds that goal frame from
   the future level-up frame (`_goal_frame_for_index`). A live agent has no goal frame. So v3 leaks the
   label here. Do not trust the v3 row. The v1 features do not read the goal frame, so v1 is the clean row.
   The cross-game run above also passed this goal frame to v3. It found no skill, so the leak did not
   rescue it. I did not fix the leak.
2. The label is position in the level, which grows with time. Any frame feature that moves steadily
   while a level runs (a counter, a bar, a growing filled area) gives a high rank score inside one level.
   The 0.996 median says "the frame carries a clock". It does not say "the frame carries distance to the
   goal". I did not test that split.
3. Rows are consecutive frames of one replayed path. They are not independent. No standard error is given.
4. Five games are a small sample. dc22 dominates. lp85 shows no skill.
5. Inside a held-out level, rank score is what the live explorer needs. MAE and pooled score are
   secondary.

What it means. A within-game depth estimate is plausible. A cross-game one is not, at this scale. The
live agent learns inside the game it plays. So the within-game result is the relevant one for Idea B.
It still needs a test against a clock-only baseline (feature = steps since the level started) before
anyone calls it a distance signal. That test is the next step. I did not run it.

## Addendum 2026-10-18: clock baseline, the within-game signal is a step counter

I ran the baseline the previous addendum named. Same leave-one-level-out split, same 479 rows, same
5 games. Script: `scripts/experiments/outer_loop_depth_clock_baseline_20261018.py`. Numbers:
`depth-calibration-cross-game-verifier-2026-10-18.clock-baseline.data.json`. CPU only.
The clock feature is the number of steps since the level started.

| Features | Pooled Spearman | MAE model | MAE constant |
|---|---|---|---|
| clock only | 0.75 | 0.166 | 0.250 |
| v1 frame features | 0.41 | 0.227 | 0.250 |
| v1 plus clock | 0.74 | 0.181 | 0.250 |

- The clock alone beats v1 on pooled Spearman and on error. Adding v1 to the clock does not help.
  It makes error worse (0.181 against 0.166).
- Per game, the clock alone has Spearman 0.73 to 0.90 in all five games. v1 is near zero in lp85,
  tn36 and tr87.
- dc22 is the one game where v1 beats the clock: Spearman 0.83 against 0.73, error 0.160 against
  0.184. It is one game, and I did not test the gap. Do not read it as a finding.

Correction to the previous addendum. The within-level Spearman (median 0.996) is not informative. Inside
one level, any steadily rising counter ranks the rows perfectly against a label that falls with time.
So that metric cannot separate a distance signal from a clock. I should have seen this before I ran
the test. The pooled score and the error are the metrics that can tell them apart.

Conclusion. The v1 features carry no distance signal beyond a step counter inside a game. The
earlier "depth signal for v1" was a clock. Idea B stays unanswered for frame features, and the
frame features are not the cheap route to it.

Limits. Same as before: five games, non-independent rows, no standard error. The clock needs the level
start step. The live explorer knows it, so a clock feature is available to it. The label depends on the
level length. A clock cannot know the length of a new level, so its error floor is not zero.

What this does not decide. A learned estimator of distance to the goal may exist. This test shows that
these frame features do not beat a counter. It does not test other features.

## Addendum 2026-10-19: review corrections to the two addenda above

An independent reviewer re-derived the label fix, its tests and both within-game scripts from source.
It found no defect in the label code, the segment logic or the leave-one-level-out split. It found
three errors in the prose above. I checked each against the two `.data.json` files. All three are real.

1. The clock addendum says the clock alone has Spearman 0.73 to 0.90 in all five games. That is wrong.
   Per game it is dc22 0.73, lp85 0.54, tn36 0.69, tr87 0.85, tu93 0.90. The range is 0.54 to 0.90.
2. The within-game addendum says v1 beats the constant on error in 3 of 5 games. That is wrong.
   v1 beats the constant in dc22 (0.160) and tu93 (0.215). lp85 is a tie (0.2505 against 0.2508).
   tn36 (0.270) and tr87 (0.258) are worse than the constant. So it is 2 of 5.
3. The clock addendum says adding v1 to the clock does not help. That holds for the pooled numbers
   only. Per game, v1 plus clock raises Spearman in lp85 (0.54 to 0.78) and tu93 (0.90 to 0.94).
   It lowers tu93 error (0.095 to 0.072). It hurts dc22, tn36 and tr87 on error.

Effect on the conclusion. The pooled result stands: the clock alone beats v1, and v1 plus clock is
worse than the clock alone. The data does not rule out a small frame contribution in two of five games.
"The v1 signal is a step counter" is supported. "The frame carries no distance signal at all" is not
shown. Treat the earlier wording as too strong.

One bias is not quantified. Both scripts rank with `argsort(argsort)`, which gives ordinal ranks and
does not average ties. Predictions clipped to 0 or 1, and constant predictions, tie. The tiebreak
follows row order, which falls with the label. This inflates Spearman, mostly the within-level median
(0.996), which this note already calls uninformative. I did not measure its effect on the pooled values.

Not checked by the reviewer, and not checked by me: a re-run of the experiments, the
`_goal_frame_for_index` leak, and that the fixed-label cross-game data file came from the committed code.

## Addendum 2026-10-19 (ties measured)

The two within-game scripts now rank with `scipy.stats.rankdata` (average ranks for ties). Before, they used `argsort(argsort)`, which breaks ties by order. Same data, same split, CPU only.

| Quantity | Ordinal ranks (before) | Tie-averaged (now) |
|---|---|---|
| clock-only pooled Spearman | 0.75 | 0.752 |
| v1 pooled Spearman | 0.41 | 0.405 |
| v1 + clock pooled Spearman | 0.74 | 0.730 |
| v1 within-level median | 0.996 | 0.998 |
| v3 pooled / within-level median | not recorded | 0.467 / 0.942 |

The tie bias is negligible. No conclusion changes: the within-game signal is mostly a step counter. The "before" values are from the earlier text of this note, not from a re-run of the old code.
