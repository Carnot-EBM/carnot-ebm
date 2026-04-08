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
> This model achieves 80.4% on held-out TruthfulQA test sets, but in practical
> deployment (8 real questions), activation-based EBMs agreed with ground truth only
> 50% of the time. The EBM detects model **confidence**, not **correctness** —
> confident hallucinations get low energy (look fine) while correct-but-hedging
> answers get flagged.
>
> This model is a research artifact documenting activation-space structure.
> It is NOT a reliable hallucination detector for production use.
>
> For practical verification, use structural constraints (test execution, SAT solving)
> rather than activation analysis. See the [Carnot technical report](https://github.com/Carnot-EBM/carnot-ebm/blob/main/docs/technical-report.md)
> for 41 experiments and 14 principles learned.

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
