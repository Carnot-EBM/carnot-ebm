---
tags:
  - energy-based-model
  - hallucination-detection
  - jax
  - carnot
license: apache-2.0
---


> **Important: Research Artifact, Not a Production Detector**
>
> This model achieves 88.5% on held-out TruthfulQA test sets, but in practical
> deployment (8 real questions), activation-based EBMs agreed with ground truth only
> 50% of the time. The EBM detects model **confidence**, not **correctness** —
> confident hallucinations get low energy (look fine) while correct-but-hedging
> answers get flagged.
>
> This model is a research artifact documenting activation-space structure.
> It is NOT a reliable hallucination detector for production use.
>
> For practical verification, use structural constraints (test execution, SAT solving)
> rather than activation analysis. See the [Carnot technical report](https://github.com/ianblenke/carnot/blob/main/docs/technical-report.md)
> for 41 experiments and 14 principles learned.

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
