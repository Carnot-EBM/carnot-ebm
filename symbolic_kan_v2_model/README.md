# Carnot Symbolic-KAN v2

Energy-Based Model for arithmetic reasoning-step verification.
Published from the Carnot project (https://github.com/Carnot-EBM/carnot-ebm).
License: Apache 2.0

## Model description

Symbolic-KAN v2 is a Kolmogorov-Arnold Network whose hidden nodes are constrained
to a discrete vocabulary of arithmetic operations (ADD, MUL, CMP, EQ).  Each node
checks one specific type of arithmetic relationship between two input features.
This design gives interpretable, human-readable explanations for why a reasoning
step is flagged as incorrect.

Architecture:
- Input: 16-dimensional feature vector extracted from a reasoning step string
- Hidden layer: 8 symbolic nodes (each with ADD/MUL/CMP/EQ label + residual spline)
- Output: scalar energy (lower = more correct)

Training objective: contrastive loss that pushes E(correct) below E(incorrect).

## Training provenance

- Experiment: Exp 948 (Symbolic-KAN Real FoVer), milestone 2026.04.73
- Training corpus: results/fover_labeled_steps_live.json from Exp 442
  (57 labeled reasoning-step pairs from real GSM8K responses)
- Violation types covered: arithmetic computation errors (ADD, MUL, CMP, EQ)
- Training AUC (held-out 20% split): 1.0
- Training epochs: 60
- Seed: 948

This deployment (Exp 968, milestone 2026.04.75):
- Integration test AUC: 1.0000 (gate >= 0.9)
- Registered as Tier 3 callable in ThreeTierPipeline via SymbolicKANTier3 wrapper

## Intended use

Primary use: Carnot ThreeTierPipeline Tier 3 verifier for arithmetic reasoning steps.

```python
from carnot.pipeline.symbolic_kan_tier3 import SymbolicKANTier3, load_symbolic_kan
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

model = load_symbolic_kan("symbolic_kan_v2_model/")
pipeline = ThreeTierPipeline(
    sink_probe=...,
    eorm_model=...,
    ising_pipeline=SymbolicKANTier3(model),
)
```

## Limitations

- Validated only on FoVer violation types present in Exp 442 data (ADD, MUL, CMP, EQ).
- Generalisation to other error types (logical fallacies, factual errors) is untested.
- Feature extraction is numeric-token-based; responses without numeric content may
  produce uninformative feature vectors.
- Training set is small (57 pairs total; ~46 training pairs).

## Dual distribution

Model weights are published on both:
- HuggingFace: https://huggingface.co/Carnot-EBM/symbolic-kan-v2
- IPFS (CID recorded in ops/changelog.md for this session)

This satisfies CLAUDE.md rule 3 (distribution mirroring for published artifacts).
