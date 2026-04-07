---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---

# per-token-ebm-gptoss-20b-nothink

Per-token hallucination detection EBM for openai/gpt-oss-20b.

| Metric | Value |
|--------|-------|
| Test accuracy | 73.0% |
| Energy gap | 2.3046 |
| Source model | openai/gpt-oss-20b |
| Hidden dim | 2880 |
| Architecture | Gibbs [2880 → 1024 → 256 → 1], SiLU |
| Training tokens | 15,864 |
| Thinking | disabled |

## Usage

```python
from carnot.inference.ebm_loader import load_ebm
ebm = load_ebm("per-token-ebm-gptoss-20b-nothink")
energy = float(ebm.energy(activation_vector))
# Low energy = likely correct, high energy = likely hallucination
```

Trained with [Carnot](https://github.com/Carnot-EBM/carnot-ebm).
