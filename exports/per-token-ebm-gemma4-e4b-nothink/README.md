---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-gemma4-e4b-nothink

Per-token hallucination detection EBM for google/gemma-4-E4B.

| Metric | Value |
|--------|-------|
| Test accuracy | 79.3% |
| Energy gap | 2.6524 |
| Source model | google/gemma-4-E4B |
| Hidden dim | 2560 |
| Architecture | Gibbs [2560 → 512 → 128 → 1], SiLU |
| Training tokens | 15,465 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-gemma4-e4b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
