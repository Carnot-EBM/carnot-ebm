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
