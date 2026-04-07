---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-bonsai-17b-nothink

Per-token hallucination detection EBM for prism-ml/Bonsai-1.7B-unpacked.

| Metric | Value |
|--------|-------|
| Test accuracy | 75.0% |
| Energy gap | 1.5849 |
| Source model | prism-ml/Bonsai-1.7B-unpacked |
| Hidden dim | 2048 |
| Architecture | Gibbs [2048 → 512 → 128 → 1], SiLU |
| Training tokens | 6,607 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-bonsai-17b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
