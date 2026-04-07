---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-qwen35-9b-nothink

Per-token hallucination detection EBM for Qwen/Qwen3.5-9B.

| Metric | Value |
|--------|-------|
| Test accuracy | 85.8% |
| Energy gap | 4.1623 |
| Source model | Qwen/Qwen3.5-9B |
| Hidden dim | 4096 |
| Architecture | Gibbs [4096 → 1024 → 256 → 1], SiLU |
| Training tokens | 6,244 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-qwen35-9b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
