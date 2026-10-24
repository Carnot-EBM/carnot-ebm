# World Model Predictive Paths: reading note (2026-10-24)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source read: Lu, Gao, Liu, Li (UIUC), "Self-Improving Agents with World Model Predictive Paths", arXiv 2610.28945v1 (HTML version). The page was read through a summarizing fetch tool, not the full text. The last 367 characters were not read. Every number below is second-hand. Check the tables before you cite any of them.

## What the method is

The method runs at test time. Model weights stay frozen. At each step:

1. The agent proposes up to K = 8 candidate actions.
2. A language world model predicts the next observation for each candidate.
3. A language evaluator gives each predicted outcome a cost from 0 to 100.
4. Gibbs weights over the costs (temperature 10) summarize the candidates. This is the MPPI idea (Model Predictive Path Integral control). The paper gives a KL-regularized optimality argument for the weights.
5. A final controller picks or synthesizes the action to run.

A two-branch text memory updates the planner and the world model from retrieved experience. No parameter changes.

## What the paper reports

Benchmarks: ALFWorld (134 tasks) and ScienceWorld (90 tasks), through AgentBoard, 30 steps per trial. Metrics: success rate at 1 and 5 attempts.

| Model | ALFWorld (1 / 5) | ScienceWorld (1 / 5) |
|---|---|---|
| Gemma-4-26B-A4B | 65.7 / 84.3 | 46.7 / 60.0 |
| Qwen3.5-9B | 28.4 / 54.5 | 32.2 / 50.0 |
| GPT-5.4-mini | 64.9 / 92.5 | 56.7 / 66.7 |

- The authors report first place in nine of twelve comparisons.
- On ALFWorld with Gemma, the strongest baseline (ITP-I) scored 32.8 / 54.5.
- With Qwen3.5-9B on ScienceWorld, ReAct scored higher (34.4 at 1 attempt, 52.2 at 5).
- Ablations on Gemma: K = 8 gave the best success at 1 attempt. Temperature 50 cut ALFWorld to 53.7. Using both memory branches gave the best average (56.2).
- Cost: 6.72 s per step against 0.51 to 1.94 s for the baselines.

## Limits the authors state

- The world model is frozen, so it cannot adapt to changed dynamics.
- Latency and cost grow with the candidate count.
- All tests are text simulations. The authors say real deployment needs human oversight.
- Code is "to be released". No license is stated.

## What this means for Carnot

### Same shape as the live ARC loop

Predict the outcome of each candidate, score it, then choose. The scored ARC path already does this: `WorldModelVerifier` gates the induced model and `plan_in_model` searches in it. The paper adds one outside data point that this shape works with a frozen local model. It adds no new mechanism.

### A knob worth reporting

The score-to-weight temperature mattered: 10 against 50 moved ALFWorld by 12 points on one model. Any Carnot selector with a sharpness setting should report a sweep, not one value.

### Cost belongs next to the gain

The gain came with 3x to 13x latency per step. A Carnot claim from a predict-and-score method must state its cost.

## Where the paper does not apply

- **Circularity.** The evaluator is an LLM judge. The paper does not train it, calibrate it, or check its scores against true outcomes. Under the Circularity / Oracle-Distinctness rule, this is a result about a pipeline. It is not evidence that a verifier adds value. Whether the evaluator and the planner are the same model is not stated in what was read. If they are, the concern is stronger.
- **Energy models.** None is trained. The Gibbs weighting is a softmax over costs. Carnot already has that.
- **Baselines.** There is no self-consistency baseline. The paper never compares against voting.
- **Calibration.** Not measured. The paper only notes that predictions are uncertain.
- **ARC.** Not mentioned. ARC work is shelved as of 2026-10-24 in any case.

## Not checked

- The final 367 characters of the page.
- How ITP-I and the other baselines are defined, and whether they got the same candidate budget.
- Whether the evaluator and the planner share a model.
- Whether the memory branches help beyond retrieval of similar past text.

## Verdict

Low value for the north star. It confirms a pattern the live agent already uses. It does not test either open gap: an oracle-distinct selector, or calibrated accept and escalate decisions. No experiment follows from it.
