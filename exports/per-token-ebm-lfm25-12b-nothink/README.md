---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-lfm25-12b-nothink

Per-token hallucination detection EBM for LiquidAI/LFM2.5-1.2B-Instruct.

| Metric | Value |
|--------|-------|
| Test accuracy | 76.7% |
| Energy gap | 2.6127 |
| Source model | LiquidAI/LFM2.5-1.2B-Instruct |
| Hidden dim | 2048 |
| Architecture | Gibbs [2048 → 512 → 128 → 1], SiLU |
| Training tokens | 4,282 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-lfm25-12b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
