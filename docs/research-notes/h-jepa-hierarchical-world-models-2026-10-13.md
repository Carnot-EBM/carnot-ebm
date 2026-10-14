# H-JEPA: what it offers the Carnot live ARC agent (2026-10-13)

Source: arXiv 2610.06805, "H-JEPA: End-to-End Learning of Hierarchical World Models for
Visual Planning" (Zhang, Terver, Rabbat, LeCun, Balestriero), project page h-jepa.com.
I read the abstract and the first 100,000 characters of the HTML. Appendices E (planner
settings) and the DROID figure values were not in the text I read. I did not read the code.

## What the method is

- Up to four stacked action-conditioned JEPA world models, trained jointly.
- Level 1 is LeWM (ViT-Tiny encoder, causal transformer predictor). Higher levels use a
  two-layer MLP encoder and a causal transformer predictor.
- Each higher level uses stride 2, so it predicts over twice as many environment steps.
- Each level has a prediction loss plus SIGReg (from LeJEPA) to prevent collapse.
- Planning runs top-down by gradient descent over action sequences. The top level
  minimizes distance to the goal. Each lower level matches the subgoal from the level above.

## What was measured

Visual AntMaze, success %, 3 seeds, mean and standard error. LeWM baseline: 18.0 +/- 3.5.

| Cost used for planning | 2 levels | 3 levels | 4 levels |
|---|---|---|---|
| Top-level projection only | 31.3 | 20.7 | 26.7 |
| Hierarchical (top-down subgoals) | 39.3 | 73.3 | 63.3 |

- The gain comes from the hierarchical cost, not from the extra levels alone.
  Projection-only costs stay near the 18% baseline.
- Depth is not monotone. Three levels beat four.
- Push-T: two levels match the flat model. Three levels do poorly. The paper blames data:
  three levels see 58% of the transitions per epoch, and four see 14%.
- DROID: without the inverse-dynamics loss the model collapses to zero. The reported
  gain is an offline, open-loop path-fidelity score. It is not task success.

## Correction to the project page

The page says Push-T deeper models do worse "likely" because episodes are short. The paper
gives a more specific cause: fewer usable training transitions per level.

## Mapping onto Carnot

| H-JEPA mechanism | Carnot status | Survives on the live agent? |
|---|---|---|
| Top-down subgoals handed to a lower planner | `E3AgentPolicy.plan_in_model` plans flat | Idea transfers. The substrate does not. |
| Learned latent spaces, SIGReg | No learned pixel encoder on the scored path | No. Needs training the live agent cannot do on a hidden game. |
| Probes showing nuisance factors fade at higher levels | Not measured | Yes, as a cheap diagnostic on induced world models. |
| Stride-2 temporal abstraction | ARC levels are natural subgoals | Possible substitute: use the game's level boundary as the coarse step. |

## Candidate experiments (none started)

1. **Subgoal handoff in the induced model.** Use `plan_in_model` with a two-step plan:
   first reach a predicted "level-progress" state, then plan the primitive path. Gate:
   actions-to-first-level-up (REQ-5720) on held-out games, not a proxy. This is
   Generalization-Testing Floor activity 2 (a method fix, not a per-game fix).
2. **Selective-abstraction probe.** Check whether win-relevant state stays recoverable
   from the induced model while distractor state does not. Cost: low. No new training.
3. **Depth sweep with a positive control.** H-JEPA shows depth can hurt. Any hierarchy
   test must include a flat arm and a count of how many episodes each level sees.

## Where it fails for us

- The training recipe needs millions of transitions (10.05M on AntMaze). A hidden ARC game
  gives a few hundred actions. Weights trained on the 25 public games do not carry over.
- The headline number is one environment with 3 seeds.
- The planner is gradient descent over continuous actions. ARC actions are discrete.
- "Less planner compute" is measured along curves the authors chose, not one fixed budget.

## Honest verdict

Useful as a design pattern for subgoal handoff. Not a method we can adopt as written.
Nothing here has been run on Carnot. No claim about ARC follows from the paper.
