# H-JEPA: hierarchical world models for visual planning: reading note (2026-11-05)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source read in full from the PDF text: Zhang, Terver, Rabbat, LeCun, Balestriero, "H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning", arXiv 2610.06805 v2 (2026-11-01). Code release and replications were not checked.

## What the method is

- A stack of action-conditioned JEPAs (models that predict the next latent state, not pixels). Each level predicts farther ahead in its own learned latent space. Example strides: 5, 10 and 20 environment steps.
- Each level trains with a prediction loss plus SIGReg (a normality regularizer that prevents collapse). All levels train end to end.
- Planning runs top-down. The top level minimizes latent distance to the goal by gradient descent on macro-actions. Its predicted states become subgoals for the level below. Level 1 outputs raw actions.
- On real DROID video, an inverse-dynamics loss is added. Without it the model collapses onto slow background features.

## Reported results

| Item | Result |
|---|---|
| Visual AntMaze, flat baseline | 18.0 +/- 3.5 % success |
| Visual AntMaze, 3 levels | 73.3 +/- 3.5 % |
| Visual AntMaze, 4 levels | 63.3 +/- 7.7 % |
| Planner compute | lower for the same or higher success on FourRoom, AntMaze, Cube |
| Cost-only ablation (AntMaze, 2-level model) | native level-1 cost 23.3 %, level-2 projected cost 31.3 % |
| Subgoal-tracking cost monotonicity (Spearman) | 0.89 to 1.00 |
| Full-horizon goal cost monotonicity | 0.48 to 0.75 |

All figures are mean +/- standard error over three seeds.

Two separate effects are reported:

1. **Cost geometry.** With the level-1 dynamics held fixed, scoring the goal distance in an upper-level latent space improves flat planning. Level-1 distance is nearly constant away from the goal. Upper-level distance rises smoothly along corridors.
2. **Temporal decomposition.** Subgoal segments give a cost that falls steadily. A flat full-horizon cost can stall or rise even along expert trajectories.

## Limits

- Simulation only: four environments, three seeds.
- The DROID result is offline open-loop planning fidelity. The authors name closed-loop control on a physical robot as the next step.
- Push-T fails at three levels. The authors' likely cause is short episodes: the 3-level model sees 58% of the flat model's training transitions. This is a hypothesis, not a tested result.
- On Push-T and Cube, where they observe no selective abstraction, projected costs give no clear benefit. The cost-shaping gain depends on abstraction forming.
- The margin over HWM (a hierarchy that uses one shared latent space) is shown in a figure, not a table.

## What it means for Carnot

### Usable ideas

1. **Landscape shaping.** Planning here is energy minimization with a learned cost. A flat cost gives gradient search no signal. A learned abstract space gives a smoother one. This applies directly to Phase 3 EBM and JEPA work.
2. **Two cheap diagnostics.**
   - A cost-to-anchor map per level: latent distance against true path length.
   - Spearman monotonicity of cost against time along expert trajectories.
   Both check whether a learned energy can guide search. Neither needs a new model.
3. **Anti-collapse recipe.** SIGReg per level, plus a watch on the prediction-loss and SIGReg curves. A prediction loss near zero with a collapsed SIGReg is the collapse signature. This matches the "degenerate identity" concern in the Phase Prototype + Validation discipline.

### Not actionable now

- The live ARC agent could use subgoal-driven hierarchical planning, but ARC is suspended by the 2026-11-05 operator directive. No ARC task follows.
- The mandated generator (`unsloth/Qwen3.8-27B-GGUF`) is unaffected.

## Rule checks

- No verifier-value claim is made. The Circularity / Oracle-Distinctness rule does not apply.
- No self-consistency baseline applies to a planning-success comparison.
- The paper does not touch the north star or the calibrated-decision floor.

## Verdict

Medium-low value. A well-controlled simulation result from a strong group, with an honest failure case. No experiment follows. If a Phase 3 energy-landscape task is ever scoped, run the two diagnostics above first.

## Related talk (appended 2026-11-05)

Source: Yann LeCun, "World Models: Enabling the next AI revolution" (https://www.youtube.com/watch?v=72Xj8k5WQX4). Conference talk, no date on the page. Read from the auto-generated transcript. No slides were seen.

### Claims in the talk

- Inference should be search by energy minimization, not a fixed forward pass. Perceive, propose an action sequence, predict the outcome with a world model, score it with a cost, then optimize the actions. This is model-predictive control with a learned model.
- Guardrails are extra cost terms applied at every planning step. He says such a system cannot be jailbroken the way a fine-tuned LLM can.
- Predict in representation space (JEPA), not pixels. Most pixel detail cannot be predicted.
- Collapse is the central training problem. He prefers information maximization over variables. SIGReg projects embeddings along many random directions and pushes each projection toward a Gaussian.
- Hierarchical planning is called "completely open". H-JEPA is the paper that addresses it.
- V-JEPA prediction error spikes on physically impossible video (violation of expectation).
- Advice: drop generative models, probabilistic models and contrastive methods. Minimize reinforcement learning.

### The speaker's own limits

- SIGReg is "very recent" and "we haven't scaled it up".
- Planning results shown are simulation only.
- The SIGReg theory result assumes Gaussian underlying causes. He says it is "not a proof that it works in every case".
- Constraints and task objectives need a small trained head on the representation. His answer of "two samples" is not tested in the talk.
- The "abandon" advice is opinion, not measurement.

### What it adds for Carnot

1. The talk and the paper share one planning loop: energy minimization over a learned model. The note's landscape-shaping framing stands.
2. Guardrails as cost terms is the closest match to Carnot's verifier ensemble. Difference: his guardrails live in a learned space and need a trained projection per constraint. Carnot's verifiers are explicit and checkable.
3. Tension to keep visible. He rejects LLM-centered work. Carnot's mandate is an open LLM generator with energy verifiers (memory: hybrid pragmatic architecture). Record both positions. Do not merge them.
4. Violation-of-expectation is a cheap sanity test for any learned world model we train. It extends the cost-geometry diagnostics above.

No experiment follows from the talk.
