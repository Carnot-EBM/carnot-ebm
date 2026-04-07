---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-gemma4-e2b-it-nothink

Per-token hallucination detection EBM for google/gemma-4-E2B-it.

| Metric | Value |
|--------|-------|
| Test accuracy | 75.0% |
| Energy gap | 2.4818 |
| Source model | google/gemma-4-E2B-it |
| Hidden dim | 1536 |
| Architecture | Gibbs [1536 → 512 → 128 → 1], SiLU |
| Training tokens | 4,252 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-gemma4-e2b-it-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
