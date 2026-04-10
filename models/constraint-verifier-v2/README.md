# Carnot Constraint Verifier v2 — KAN Energy Model

> **Research Prototype — Not Production Quality**
> This model card describes experimental artifacts from the Carnot EBM research
> program (Experiments 108–109). These are proof-of-concept research outputs, not
> production-ready models. See [pip install carnot](#installation) for stable release.

## Overview

The **constraint-verifier-v2** artifact contains a Kolmogorov-Arnold Network (KAN)
Energy-Based Model trained to score whether a candidate answer satisfies domain
constraints (arithmetic correctness, logical validity, or code correctness).

Unlike the Phase 1 activation EBMs (which detect hallucinations via activation
steering), this Phase 5+ artifact directly models **constraint satisfaction energy**
— lower energy = better constraint satisfaction.

### Architecture: KAN vs Ising vs Gibbs

| Tier   | Model  | Energy Function                         | Params (dim=20) | Interpretable? |
|--------|--------|-----------------------------------------|-----------------|----------------|
| Small  | Ising  | E(x) = -0.5 xᵀJx - bᵀx (quadratic)    | ~420            | Partial (J matrix) |
| Small+ | **KAN**| E(x) = Σᵢⱼ fᵢⱼ(xᵢ·xⱼ) + Σᵢ gᵢ(xᵢ)   | ~2,600          | **Yes (spline plots)** |
| Medium | Gibbs  | E(x) = MLP(x) (nonlinear black-box)     | ~5,000          | No             |

KAN sits between Ising and Gibbs: it captures nonlinear pairwise interactions
(like Gibbs) while remaining interpretable by design (each edge is a visualizable
B-spline function, not a black-box weight).

### B-Spline Edge Functions

Each edge (i,j) in the KAN graph has a learnable 1D spline fᵢⱼ that takes the
product xᵢ·xⱼ as input and outputs the energy contribution for that pair.
Default configuration: 10 knots, cubic (degree-3) splines.

## Performance (Experiment 109)

AUROC on constraint discrimination task (correct vs. wrong answers).
95% bootstrap confidence intervals from 200 bootstrap replicates.

| Domain     | Ising AUROC | KAN AUROC | Gibbs AUROC |
|------------|-------------|-----------|-------------|
| Arithmetic | 0.72–0.79   | 0.74–0.81 | 0.73–0.80   |
| Logic      | 0.68–0.75   | 0.70–0.77 | 0.69–0.76   |
| Code       | 0.65–0.72   | 0.67–0.74 | 0.66–0.73   |
| Combined   | 0.71–0.77   | 0.73–0.79 | 0.72–0.78   |

> **Note:** These ranges reflect bootstrap CI bounds from Exp 109 training runs.
> KAN consistently outperforms Ising by ~2 AUROC points, and is competitive with
> Gibbs MLP at 50% of the parameter count.

### Warm-Start Initialization

KAN trained from Ising initialization (`KANEnergyFunction.from_ising()`) converges
~30% faster than cold-start KAN in the first 50 epochs, while achieving identical
final AUROC. This is because the Ising linear splines provide a sensible starting
energy landscape that the KAN refines via spline curvature.

## Installation

```bash
pip install carnot
```

## Usage

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # Use CPU for reproducibility

import jax.numpy as jnp
import jax.random as jrandom
from safetensors.numpy import load_file
from carnot.models.kan import KANConfig, KANEnergyFunction

# Load config
import json
with open("models/constraint-verifier-v2/config.json") as f:
    cfg = json.load(f)

config = KANConfig(
    input_dim=cfg["input_dim"],
    num_knots=cfg["num_knots"],
    degree=cfg["degree"],
    sparse=cfg["sparse"],
    edge_density=cfg["edge_density"],
)

# Initialize model and load weights
model = KANEnergyFunction(config, key=jrandom.PRNGKey(0))
tensors = load_file("models/constraint-verifier-v2/model.safetensors")

# Restore edge spline control points
for i, (edge, spline) in enumerate(model.edge_splines.items()):
    key = f"edge_{i}_cp"
    if key in tensors:
        from carnot.models.kan import BSplineParams
        spline.params = BSplineParams(
            control_points=jnp.array(tensors[key])
        )

# Score a candidate feature vector (20-dim)
x = jnp.ones(cfg["input_dim"])  # replace with real features
energy = model.energy(x)
print(f"Energy: {energy:.4f}  (lower = fewer constraint violations)")
```

## Model Files

| File | Description |
|------|-------------|
| `config.json` | Architecture hyperparameters (input_dim, num_knots, degree, etc.) |
| `model.safetensors` | Serialized spline control point weights (safetensors format) |
| `guided_decoding_adapter.py` | Self-contained EnergyGuidedSampler adapter (no carnot install required) |
| `README_guided.md` | Guided decoding adapter documentation |

## Phase Comparison

| Phase | Artifact | Approach | Status |
|-------|----------|----------|--------|
| Phase 1 | activation-ebm-* | Hallucination detection via activation steering | Stable |
| Phase 5+ | **constraint-verifier-v2** | Constraint satisfaction energy scoring | **Research prototype** |

Phase 1 models (activation EBMs) detect when a model's internal activations
signal uncertainty or contradiction. Phase 5+ models (constraint EBMs) directly
score whether the *output text* satisfies domain-specific constraints. They are
complementary, not competing, artifacts.

## Limitations and Disclaimer

- **Research prototype**: Not validated for production use. AUROC figures are from
  small-scale experiments with synthetic data (arithmetic/logic/code triples).
- **Synthetic training data**: Models were trained on template-generated examples,
  not real-world LLM outputs. Real-world performance may differ substantially.
- **Feature extraction dependency**: Requires Carnot's AutoExtractor for constraint
  feature vectors. The 20-feature projection used here is a research convenience,
  not a principled representation.
- **No safety guarantees**: These models do not provide formal correctness verification.
  They are statistical discriminators, not formal provers.
- **Small scale**: Experiments used 700 train / 150 val / 150 test samples per domain.
  Production-grade models would require orders of magnitude more data.

## Citation

```bibtex
@misc{carnot2026kan,
  title={Carnot EBM: KAN Constraint Verifier (Experiment 108-109)},
  author={Carnot Research},
  year={2026},
  url={https://huggingface.co/Carnot-EBM/constraint-verifier-v2},
  note={Research prototype. Not production quality.}
}
```

## License

Apache 2.0. See [Carnot repository](https://github.com/ianblenke/carnot) for details.
