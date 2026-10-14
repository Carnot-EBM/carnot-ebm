# DeepJEPA: adaptive compute per imagined transition (2026-10-13)

Source: arXiv 2610.00368, "DeepJEPA: Scaling World Models from Within" (Jin, Zhang, Liu,
Liu, Chen, Ye, Liu, Pavone). I read the abstract and the HTML extraction. The extraction
covered the method, Tables 1 to 4 and the stated limits. I did not read the code.

## Idea

A planner spends the same compute on every imagined transition. DeepJEPA lets a weight-tied
recurrent cell run up to 4 updates per transition. A learned "continue" head halts early.
Extra updates land at contact onset and sustained object interaction.

## Measured (success %, 3 seed pairs, 50 episodes each, 150 per cell)

| Method | Reacher | Cube Single | Cube Double | Cube Triple | PushT |
|---|---|---|---|---|---|
| LeWM (flat) | 81.3 | 72.0 | 74.7 | 74.0 | 10.4 |
| Best fixed depth | 84.0 | 79.3 | 72.7 | 74.0 | 10.7 |
| DeepJEPA | 85.3 | 79.3 | 74.0 | 77.3 | 11.6 |

- Mean depth is 1.00 to 1.26 updates per transition.
- Fixed depth is non-monotone: more depth for every step does not help.

## What the numbers do not support

- Gains over the best fixed depth are 0 to 3.3 points. Seed standard deviations run
  2.5 to 8.1 points. Most gaps sit inside one seed spread.
- The exact-batch test on Cube Triple (K=1 vs K=4) gives +1.1 points, CI [-1.3, +3.6],
  p = 0.424. That is no detectable effect.
- Halting thresholds were picked by a sweep on the evaluation data, not a held-out set.
- The authors say mean depth is "architectural depth, not proportional wall-clock speedup."
- PushT best success is 11.6%.

## Mapping onto Carnot

| DeepJEPA idea | Carnot analogue | Survives on the live agent? |
|---|---|---|
| Spend extra compute only where a step is decision-critical | `plan_in_model` search budget; generator calls for induction | Yes, as a rule. A learned halting head is not needed. |
| Learned continue head | Mechanical trigger (contact, frame change, stagnation) | Substitute: a rule on observed events, no training. |
| Fixed depth is non-monotone | Our induction `--k` and attempt count | Same warning: more attempts per cell is not automatically better. |

## Candidate experiment (not started)

Spend search expansions in `plan_in_model` only after a frame-change event, and compare
actions-to-first-level-up (REQ-5720) against uniform expansion at equal total budget.
Needs a flat arm, held-out games, and a stated sample size before any claim.

## Honest verdict

The useful part is the framing: allocate test-time compute by event, and check that
uniform depth does not already win. The measured effect is small and mostly inside seed
noise. Nothing here has been run on Carnot.
