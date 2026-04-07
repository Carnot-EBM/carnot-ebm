---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-qwen35-27b-nothink

Per-token hallucination detection EBM for Qwen/Qwen3.5-27B.

| Metric | Value |
|--------|-------|
| Test accuracy | 88.5% |
| Energy gap | 4.8030 |
| Source model | Qwen/Qwen3.5-27B |
| Hidden dim | 5120 |
| Architecture | Gibbs [5120 → 1024 → 256 → 1], SiLU |
| Training tokens | 6,788 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-qwen35-27b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
