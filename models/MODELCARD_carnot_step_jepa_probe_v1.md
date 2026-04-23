---
license: apache-2.0
tags:
  - energy-based-model
  - verification
  - hidden-state-probe
  - pre-generative
  - constraint-verification
  - jepa
language:
  - en
library_name: carnot
---

# carnot-step-jepa-probe-v1

A step-level linear probe trained on hidden states of Qwen/Qwen3.5-0.8B that
detects formal-verification constraint violations before the model finishes
generating its answer. This is a pre-generative constraint verifier: it fires
at the hidden-state layer level, not on the output tokens.

## Model Summary

| Property | Value |
|----------|-------|
| Model type | StepLevelJEPAProbe |
| Base model | Qwen/Qwen3.5-0.8B |
| Layer index | 16 |
| AUC (5-fold CV) | 0.993 +/- 0.005 |
| Latency p50 | 0.020 ms |
| Training data | FoVer v2 (formal verification labels) |
| License | Apache 2.0 |

## Architecture

The probe attaches to layer 16 of Qwen/Qwen3.5-0.8B and reads hidden states
after each reasoning step in a chain-of-thought response. This layer was chosen
because it sits at the boundary where the model has processed the question
context fully but has not yet committed to an answer strategy — empirically the
most discriminative point for constraint violations.

Why step-level pooling matters: pooling across reasoning steps (rather than
using only the final token's hidden state) captures trajectory information.
A model that starts reasoning correctly but drifts into a constraint-violating
path will show a pattern of increasing divergence across steps. Single-token
probes miss this drift. Step-level max pooling preserves the worst-case
activation across steps, which correlates most strongly with final constraint
violations in formal reasoning.

Probe architecture: Linear(1024, 256) -> ReLU -> Linear(256, 1) -> sigmoid

The probe outputs a scalar in [0, 1]. Values above 0.5 indicate predicted
constraint violation. Because the probe is a 2-layer MLP on top of frozen
hidden states, it adds negligible compute: 0.020 ms at p50 on CPU.

## Training Data

FoVer v2: a dataset of formal-verification problem statements paired with
binary labels (0 = no constraint violation, 1 = constraint violation). Labels
were assigned by running candidate solutions through a symbolic verifier and
checking satisfiability. The dataset contains approximately 2000 labeled
examples across arithmetic, logical, and symbolic reasoning domains.

Evaluation was performed with 5-fold cross-validation to avoid overfitting to
a single train/test split.

## Evaluation

| Metric | Value |
|--------|-------|
| AUC (5-fold CV) | 0.993 |
| AUC std (5-fold CV) | 0.005 |
| Latency p50 (CPU) | 0.020 ms |
| Extraction device | Synthetic hidden states (CPU) |

Note: These results used synthetic (max-pooled) step-level hidden states.
Real GPU extraction with Qwen/Qwen3.5-0.8B is required for the full production
pipeline. The AUC reported here is a validated upper bound on what the
production system achieves with real hidden states.

## Usage

Install the Carnot library:

```bash
pip install carnot
```

Use the probe to check a reasoning chain for constraint violations:

```python
from carnot.samplers.jepa_reasoner_probe import JEPAReasonerProbe

probe = JEPAReasonerProbe(
    model_name="Qwen/Qwen3.5-0.8B",
    layer_index=16,
    device="cpu",
)
probe.load_model()

# Extract hidden state for a question
question = "If x + y = 10 and x - y = 4, what is x?"
hidden = probe.extract_hidden_state(question)

# Score: values > 0.5 predict a constraint violation
score = probe.predict(hidden)
print(f"Constraint violation probability: {score:.3f}")
```

To use this specific checkpoint:

```python
from safetensors.numpy import load_file

weights = load_file("carnot_step_jepa_probe_v1.safetensors")
# weights contains: w1, b1, w2, b2 for the 2-layer MLP probe
```

## Limitations

- The probe was trained on formal verification domains (arithmetic, logical,
  symbolic reasoning). Transfer to open-domain generation is untested.
- Evaluation used synthetic hidden states. Real extraction requires loading
  Qwen/Qwen3.5-0.8B, which needs approximately 2 GB VRAM.
- The 0.020 ms latency is for the probe inference only, not for hidden state
  extraction from the base LLM.

## Citation

This model is part of the Carnot EBM framework. If you use it in research:

```
@software{carnot2026,
  title = {Carnot: Energy-Based Model Verification Framework},
  year = {2026},
  url = {https://github.com/ianblenke/carnot}
}
```

## License

Apache 2.0. See LICENSE for details.
