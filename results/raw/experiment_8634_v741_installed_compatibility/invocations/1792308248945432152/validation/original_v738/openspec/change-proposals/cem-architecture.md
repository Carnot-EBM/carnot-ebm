# CEM Architecture: Composing DAB Processors

## Objective
Design the Constrained Energy Modeling (CEM) architecture to compose multiple Dual-Process Alignment (DAB) processors (e.g., SyntaxEBM + SemanticsEBM). This enables chaining multiple constraints sequentially or concurrently to reduce hallucinations in long-horizon reasoning.

## Design: Combining Energies
When multiple EBMs are used to constrain an LLM at generation time, their energy outputs must be combined before or during the logit modification step.

Supported combination strategies:
1. **Sum (`sum`)**: Energies are added. $E_{total} = \sum E_i$. This assumes constraints are independent log-probabilities and represents joint probability.
2. **Max (`max`)**: The maximum energy among all constraints is used. $E_{total} = \max_i E_i$. This acts as a strict logical AND (any high energy/violation strongly penalizes the token).
3. **Learned Temperature (`learned`)**: Each EBM has a learned or dynamically adjusted temperature weight. $E_{total} = \sum w_i E_i$.

## Implementation
We introduce `ComposedDABLogitsProcessor` in `carnot.pipeline.dab_adapter` which takes a list of EBMs and a reduction strategy (`sum`, `max`, or `learned`), applies them to the input, aggregates the energies according to the strategy, and subtracts the combined energy from the logits.
