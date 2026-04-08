## Why

36 experiments proved that activation-based hallucination detection doesn't work — EBMs trained on LLM activations detect confidence, not correctness. The model's internal state doesn't distinguish right from wrong. Instead of trying to fix autoregressive models from the outside, we need to build an energy function that IS the reasoning engine: it takes (question, answer) pairs in continuous space and outputs energy where low = correct, high = wrong. Generation becomes optimization — gradient descent to find the minimum-energy answer.

## What Changes

- Add a continuous-space EBT reasoning pipeline: embed QA pairs with a frozen sentence encoder, train a Gibbs EBM to score (question, answer) embedding pairs, then generate answers by optimizing in embedding space via gradient descent
- Add embedding-space repair: given a wrong answer embedding, use gradient descent on the trained energy function to move it toward the correct answer region
- Add nearest-neighbor decoding: map optimized continuous embeddings back to discrete text via cosine similarity against a candidate pool
- Validate on TruthfulQA: does energy-guided optimization in embedding space actually move wrong answers toward correct ones?

## Capabilities

### New Capabilities
- `ebt-reasoning`: Energy-Based Transformer reasoning in continuous latent space — embed QA pairs, train energy function, generate via gradient optimization, decode back to text
- `embedding-repair`: Gradient-based answer repair in embedding space — move wrong-answer embeddings toward low-energy regions

### Modified Capabilities
- `llm-ebm-inference`: Add EBT-based inference path alongside existing logprob/structural verification

## Impact

- **Python package**: New modules in `python/carnot/inference/` for EBT reasoning and embedding repair
- **Dependencies**: Adds `sentence-transformers` for frozen sentence encoding (lightweight, no fine-tuning)
- **Data**: Uses existing TruthfulQA infrastructure, no new datasets needed
- **Rust**: No Rust changes initially — this is a Python/JAX research prototype. Rust implementation follows if the approach works.
- **Tests**: New test files tracing to REQ-EBT-* and SCENARIO-EBT-* requirements
- **Existing code**: No breaking changes — new capability alongside existing pipeline
