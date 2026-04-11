---
tags:
  - energy-based-model
  - ising-model
  - constraint-propagation
  - verification
  - carnot
license: apache-2.0
---

# constraint-propagation-models

**Three learned Ising constraint models for arithmetic, logic, and code verification.**

Each model is a 200×200 Ising EBM trained via discriminative Contrastive
Divergence (Exp 62/89) to distinguish correct from incorrect LLM responses
in its domain.  No LLM needed at inference time — just encode the response as
a 200-dim binary feature vector and compute the energy.

## Models

| Domain | AUROC | Accuracy | HuggingFace Repo |
|--------|-------|----------|-----------------|
| arithmetic | **0.997** | 99.0% | `Carnot-EBM/constraint-propagation-arithmetic` |
| logic      | **1.000** | 100.0% | `Carnot-EBM/constraint-propagation-logic` |
| code       | **0.867** | 88.0% | `Carnot-EBM/constraint-propagation-code` |

Exp 89 self-bootstrap references: arithmetic 1.0 AUROC, logic 1.0, code 0.91.

## Background

These are the first published learned Ising constraint models on HuggingFace.
No other repository has constraint models trained via Ising EBMs on verified
(correct, wrong) example pairs.

**Why Ising for constraint verification?**
- **Direct hardware mapping**: The coupling matrix J corresponds directly to
  wire strengths in an Ising machine or FPGA implementation. Energy = violation
  count without any nonlinear computation.
- **183× parallel Gibbs speedup**: The quadratic structure enables massively
  parallel Gibbs sampling — each spin update is independent given its neighbors.
- **Fully interpretable**: You can inspect J[i][j] directly to see which feature
  pairs the model learned as important. No black-box activations.
- **Differentiable**: E(x) = -0.5 x^T J x - b^T x is differentiable everywhere,
  enabling gradient-based repair (Exp 87: 44% energy reduction).

## Quick Start

```python
from carnot.inference.constraint_models import ConstraintPropagationModel
from scripts.export_constraint_models import encode_answer

# Load any domain model (local or from HuggingFace Hub)
model = ConstraintPropagationModel.from_pretrained(
    "exports/constraint-propagation-models/arithmetic"
    # or: "Carnot-EBM/constraint-propagation-arithmetic"
)

# Encode a response as a 200-dim binary feature vector
x = encode_answer("What is 47 + 28?", "The answer is 75.")

# Verify — lower energy = more likely correct
energy = model.energy(x)
score = model.score(x)      # sigmoid(-energy), in (0,1)
print(f"Energy: {energy:.2f}, Score: {score:.3f}")

# Batch scoring
import numpy as np
X = np.stack([
    encode_answer("What is 47 + 28?", "The answer is 75."),   # correct
    encode_answer("What is 47 + 28?", "The answer is 74."),   # off-by-one
])
energies = model.energy_batch(X)
print(f"Energies: {energies}")  # correct should be lower
```

## Save Your Own Model

```python
import numpy as np
from carnot.inference.constraint_models import IsingConstraintModel

# After training J and b arrays:
model = IsingConstraintModel(
    coupling=J,    # (200, 200) float32
    bias=b,        # (200,) float32
    config={"domain": "my_domain", "feature_dim": 200, "auroc": 0.95},
)
model.save_pretrained("./my-constraint-model/")
# Creates: my-constraint-model/model.safetensors + config.json

# Load back:
model2 = IsingConstraintModel.from_pretrained("./my-constraint-model/")
```

## Directory Structure

```
constraint-propagation-models/
├── README.md                    ← This file (collection card)
├── arithmetic/
│   ├── model.safetensors        ← J (200×200) + b (200,) as float32
│   ├── config.json              ← metadata + benchmark results
│   └── README.md               ← model card
├── logic/
│   ├── model.safetensors
│   ├── config.json
│   └── README.md
└── code/
    ├── model.safetensors
    ├── config.json
    └── README.md
```

## Technical Details

**Energy function:**
```
E(x) = -(b^T s + s^T J s)    s = 2x - 1 ∈ {-1, +1}^200
```

**Training algorithm:** Discriminative CD — instead of sampling from the model
(as in standard CD), we clamp the negative phase to *wrong* examples. The
gradient then pushes E(correct) down and E(wrong) up with no MCMC needed:

```
ΔJ = -β(⟨s_i s_j⟩_correct - ⟨s_i s_j⟩_wrong) - λ·sign(J) - α·J
Δb = -β(⟨s_i⟩_correct - ⟨s_i⟩_wrong) - α·b
```

Best hyperparameters (from Exp 89 sweep): lr=0.01, L1=0.0, 300 epochs.

**Feature encoding:** 200 binary features per response, grouped as:
- Numeric (20): digit/operator presence
- Structural (40): word/char counts, punctuation, keywords
- Domain-specific (80): arithmetic/logic/code structural patterns
- Consistency (60): question-answer number and keyword agreement

## Limitations

1. **Structural features only**: Not semantic. Correct responses with unusual
   phrasing may score poorly; plausible wrong answers with correct-sounding
   structure may score too high.
2. **Three domains only**: Factual and scheduling domains have near-zero AUROC
   (Exp 89) — the structural features don't discriminate factual correctness.
3. **Binary features required**: You must use the same 200-dim encoder from
   `scripts/experiment_62_domain_constraint_learning.py`. Other encodings
   will produce meaningless energies.
4. **Template-generated training data**: Not validated against real LLM outputs.
   Use as a research artifact, not a production verifier.

## Research Context

These models are part of the Carnot EBM framework:
- **Exp 62**: Domain constraint learning via discriminative CD on 10K triples.
- **Exp 89**: Self-bootstrapping — the pipeline's own verified outputs become
  training data, closing the loop toward autonomous self-improvement (FR-11).
- **Exp 64-66**: Continuous Ising relaxation → differentiable constraints →
  end-to-end 1.0 AUROC on embedding-space features.
- **Exp 87**: Gradient-based repair using EBM energy gradients (44% reduction).

## Citation

```bibtex
@misc{carnot2026constraint,
  title  = {Carnot Constraint Propagation Models: Learned Ising EBMs for Domain Verification},
  author = {Carnot-EBM},
  year   = {2026},
  url    = {https://github.com/ianblenke/carnot}
}
```

## Spec

- REQ-VERIFY-002: Energy-based verification of LLM responses.
- REQ-VERIFY-003: Correctness score derived from energy.
- FR-11: Self-bootstrapped verifier improvement.
