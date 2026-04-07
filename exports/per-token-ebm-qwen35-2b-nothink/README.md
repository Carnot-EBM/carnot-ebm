---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-qwen35-2b-nothink

Per-token hallucination detection EBM for Qwen/Qwen3.5-2B.

| Metric | Value |
|--------|-------|
| Test accuracy | 79.3% |
| Energy gap | 3.8469 |
| Source model | Qwen/Qwen3.5-2B |
| Hidden dim | 2048 |
| Architecture | Gibbs [2048 → 512 → 128 → 1], SiLU |
| Training tokens | 6,486 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-qwen35-2b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
