---
tags:
  - energy-based-model
  - jepa
  - constraint-verification
  - carnot
  - jax
license: apache-2.0
---

# jepa-predictor-v2

**JEPA Violation Predictor v2** — a multi-domain neural verifier trained to predict
whether a language model answer violates domain constraints (arithmetic, code, logic).

This is a research artifact from the [Carnot EBM framework](https://github.com/ianblenke/carnot).
It is a small MLP (input_dim=200, hidden_dim=128, output_dim=1) trained with class-weighted
binary cross-entropy and early stopping on a balanced multi-domain dataset.

## Performance (Exp 155)

| Domain | AUROC (v2) | vs v1 |
|--------|-----------|-------|
| Arithmetic | 0.721 | +0.018 |
| Code | 0.776 | +0.071 |
| Logic | 0.479 | −0.056 |
| **Macro average** | **0.659** | **+0.011** |

Training details: 963 samples, best epoch 19 / 100, early stopping on
validation macro AUROC (patience=15), class-weighted BCE loss (pos_weight clipped [0.1, 10]).

## Architecture

```
JEPAViolationPredictor(
  input_dim   = 200    # 200-dim binary feature vector (same encoder as Ising models)
  hidden_dim  = 128
  output_dim  = 1      # P(violation | x)
)
```

The predictor is trained on structured (question, answer, violated) triples. Features are
the same 200-dim binary structural encoding used by the Ising constraint models
(Carnot-EBM/constraint-propagation-*). This makes the two model families interoperable.

## Training Data (Exp 154 → Exp 155)

- **Arithmetic**: 800 pairs reused from Exp 143 (carry-chain arithmetic templates)
- **Code**: 200 synthetic pairs (Python type, return, initialisation constraints)
- **Logic**: 200 synthetic pairs (implication, exclusion, disjunction, negation)
- Total: 1,200 pairs; 963 train / 237 validation (stratified by domain × violated)

## Usage

```python
from safetensors.numpy import load_file

weights = load_file("jepa_predictor_v2.safetensors")
# Keys: "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"

# Or load via the carnot package:
from carnot.inference.jepa_predictor import JEPAViolationPredictor
predictor = JEPAViolationPredictor.load("jepa_predictor_v2.safetensors")

# Score a 200-dim binary feature vector
import numpy as np
x = np.zeros(200, dtype=np.float32)
x[:20] = 1.0   # example features
prob_violation = predictor.predict(x)
print(f"P(violation) = {prob_violation:.3f}")
```

## Limitations

1. **Logic AUROC low (0.479)**: byte-level structural features do not capture
   logical implication structure well. Semantic embeddings are needed.
2. **Code domain fast-path issues** (Exp 156): at threshold ≥ 0.3, all code questions
   are fast-pathed (200/200), causing degradation. Use threshold ≥ 0.8 for code.
3. **Target not met**: <2% degradation target at t=0.5 not achieved. Treat as
   directional baseline, not production-ready.
4. **Template training data**: Not validated on real LLM outputs.

## Note: Production Installation

> **Note:** This is a Phase 1 research artifact. For production use of the full
> Carnot EBM framework (constraint verification, guided decoding, energy-based repair), see:
>
> ```bash
> pip install carnot
> ```
>
> Source and documentation: <https://github.com/ianblenke/carnot>

## Spec

- REQ-JEPA-001: Violation predictor trained on multi-domain data.
- REQ-JEPA-002: Fast-path routing using predictor confidence.
- SCENARIO-JEPA-003: Cross-domain AUROC improvement over v1.

## Citation

```bibtex
@misc{carnot2026jepa,
  title  = {Carnot JEPA Violation Predictor v2},
  author = {Carnot-EBM},
  year   = {2026},
  url    = {https://github.com/ianblenke/carnot}
}
```
