---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-qwen35-4b-nothink

Per-token hallucination detection EBM for Qwen/Qwen3.5-4B.

| Metric | Value |
|--------|-------|
| Test accuracy | 81.8% |
| Energy gap | 4.0656 |
| Source model | Qwen/Qwen3.5-4B |
| Hidden dim | 2560 |
| Architecture | Gibbs [2560 → 512 → 128 → 1], SiLU |
| Training tokens | 6,273 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-qwen35-4b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
