---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-qwen3-06b

Per-token hallucination detection EBM trained on Qwen3-0.6B (base model) activations. Highest accuracy (84.5%) due to base model's larger activation gaps between correct and wrong answers.

## Key Stats

| Metric | Value |
|--------|-------|
| Test accuracy | 83.4% |
| Energy gap | 3.2692 |
| Source model | Qwen/Qwen3-0.6B |
| Thinking mode | N/A (base model) |
| Training tokens | 26,800 |
| Architecture | Gibbs [1024 → 256 → 64 → 1], SiLU |

## Usage

```python
from safetensors.numpy import load_file
import jax.numpy as jnp
import jax.random as jrandom
from carnot.models.gibbs import GibbsConfig, GibbsModel

# Load weights
weights = load_file("per-token-ebm-qwen3-06b/model.safetensors")
config = GibbsConfig(input_dim=1024, hidden_dims=[256, 64], activation="silu")
ebm = GibbsModel(config, key=jrandom.PRNGKey(0))

# Set weights
ebm.layers = [
    (jnp.array(weights["layer_0_weight"]), jnp.array(weights["layer_0_bias"])),
    (jnp.array(weights["layer_1_weight"]), jnp.array(weights["layer_1_bias"])),
]
ebm.output_weight = jnp.array(weights["output_weight"])
ebm.output_bias = jnp.array(weights["output_bias"])

# Score an activation vector (from Qwen/Qwen3-0.6B hidden states)
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

## Training

- **Loss:** Noise Contrastive Estimation (NCE)
- **Epochs:** 300, lr=0.005
- **Data:** 26,800 per-token activations from Qwen/Qwen3-0.6B
- **Labels:** correct answer tokens = low energy (data), wrong answer tokens = high energy (noise)

## Limitations

- Only works with activations from **Qwen/Qwen3-0.6B** — different models have different representation spaces
- 83.4% accuracy — use as one signal among many, not as sole verification
- Trained on QA/TruthfulQA — may not generalize to all domains

## 10 Principles from Carnot Research

1. Simpler is better in small-data regimes
2. Token-level features > sequence-level
3. The model's own logprobs are the best energy
4. Overfitting is the main enemy
5. Extract features from generated tokens, not prompts
6. Different energy signals dominate in different domains
7. Statistical difference ≠ causal influence
8. Instruction tuning compresses the hallucination signal
9. Adversarial questions defeat post-hoc detection
10. Chain-of-thought compresses the hallucination signal

See the [Carnot technical report](https://github.com/Carnot-EBM/carnot-ebm/blob/main/docs/technical-report.md) for all 25 experiments.
