# RoboJEPA: scaling a latent world model, and what it says about proxy metrics (2026-10-13)

Source: arXiv 2610.10515, "RoboJEPA: Scaling Robotic Latent World Models" (Zholus et al.,
FAIR at Meta, Mila, Polytechnique Montreal), and robojepa.github.io. I read the abstract,
the project page, and the first 100,000 characters of the HTML. The rest, including the
per-task real-robot trial counts (Table 12) and Appendix K, was not read. I did not read code.

## What it is

- A deterministic transformer predictor on a frozen V-JEPA 2.1-G encoder. It predicts the
  next-step features in one forward pass. The loss is per-token L1 over teacher-forced and
  autoregressive-rollout terms.
- Sizes 22M to 8B parameters. Training compute 2e19 to 9.5e22 FLOPs.
- Data: 23 public manipulation datasets, 12 platforms, 2.87M episodes.
- Planning: receding-horizon CEM toward one goal image, using L1 latent distance as the
  cost. Actions are sampled uniformly. The world model only ranks them.

## What was measured

- Imagination error (error of latent rollouts) fits a second-order power law in compute.
  Fit on 22M to 2B models, extrapolated to 4B and 8B. Held-out error (x1e-3), second-order
  versus first-order power law: DROID 0.6 vs 2.0, RoboCasa 1.4 vs 3.7.
- Planning capability appears at compute thresholds. Pose reach near 1e20 FLOPs. Object
  push shows no positive success until about 1e22 FLOPs.
- Real robot (DROID, Franka), progress / success %:

| Task | RoboJEPA-8B | pi0-FAST | pi0.5 |
|---|---|---|---|
| Grasp | 67 / 67 | 35 / 22 | 25 / 5 |
| Object Lift | 65 / 50 | 20 / 12 | 12 / 0 |
| Pick and Place | 42 / 27 | 55 / 40 | 69 / 53 |

The authors call the VLA rows "contextual references rather than directly comparable."

## What the evidence does not support

- Seeds and variance were not in the text I read. Per-task trial counts were not either.
- On Pick and Place, RoboJEPA-8B is worse than both VLAs, and worse than its own 4B model
  on progress. The page's "remarkable ability" wording skips this.
- "Imagination error is a reliable proxy for real-robot evaluation" is a correlation claim.
  Strength and sample size were not in the text I read.
- The law describes the predictor only. The encoder is frozen. Fits are "close to data
  saturation" after multiple epochs.
- "First" and "largest" are priority claims I did not check.

## Mapping onto Carnot

| RoboJEPA idea | Carnot status | Survives on the live agent? |
|---|---|---|
| Cheap offline error predicts expensive downstream success | Our induction gate scores plans offline (`WorldModelVerifier`); success is expensive (45.8 h for 25 cells) | The method transfers. Ours is untested. |
| Plan by ranking sampled actions with a world model, no policy | `plan_in_model` over induced code models | Already similar. |
| Capability thresholds in compute | Our rerun: 10 of 31 attempts planned, one sample per game | Warning only: a threshold needs a sweep, not one point. |
| Scale to 8B on 2.8M episodes | 27B text generator, a few hundred actions per hidden game | No. |

## Candidate experiment (not started)

Test whether an offline score predicts live outcome on our own data. The 31 labeled
induce attempts in `results/arc_induction_plans_20261011_k1/` give each attempt a
verifier score and a plan-installed label. Check score versus label with a stated
sample size and a positive control. If the score does not rank attempts, it is not a
usable proxy and we should know that before building on it. This is the same target as
the planned Clef induction-gate test.

## Honest verdict

The transferable lesson is methodological: validate a cheap proxy against the expensive
outcome before using it to choose. The scaling results need compute we do not have and a
frozen encoder we do not use. 31 attempts from one sample per game is too few to claim a
proxy holds, so any result would be a pilot count, not a rate. Nothing here has been run
on Carnot.
