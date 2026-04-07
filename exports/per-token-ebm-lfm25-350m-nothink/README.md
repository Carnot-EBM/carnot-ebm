---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-lfm25-350m-nothink

Per-token hallucination detection EBM for LiquidAI/LFM2.5-350M.

| Metric | Value |
|--------|-------|
| Test accuracy | 80.4% |
| Energy gap | 4.5153 |
| Source model | LiquidAI/LFM2.5-350M |
| Hidden dim | 1024 |
| Architecture | Gibbs [1024 → 256 → 64 → 1], SiLU |
| Training tokens | 2,259 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-lfm25-350m-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
