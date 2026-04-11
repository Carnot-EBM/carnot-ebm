---
tags:
  - energy-based-model
  - ising-model
  - constraint-satisfaction
  - logic-verification
  - syllogism
  - carnot
license: apache-2.0
---

> **Research Artifact — Not Production-Ready**
>
> This model verifies logical syllogism responses using structural binary features.
> It achieves AUROC 1.0 on the held-out test set (matching Exp 89 reference).
> It handles modus ponens, modus tollens, disjunctive syllogism, and affirming
> the consequent — not arbitrary logic or modal reasoning.

# constraint-propagation-logic

**Learned Ising constraint model for logical syllogism verification.**

Trained via discriminative Contrastive Divergence (Exp 62/89) on 400 verified
(question, correct, wrong) triples for four syllogism types.  Assigns a scalar
energy to binary-encoded responses — lower energy means the response correctly
identifies whether a syllogism is valid or invalid.

## What It Is

An Ising Energy-Based Model (EBM):

```
E(x) = -(b^T s + s^T J s)    s = 2x - 1 ∈ {-1, +1}^200
```

The coupling matrix J encodes feature pair co-occurrences that distinguish
valid from invalid logic responses (e.g., "follows by modus ponens" + "this
follows" co-occurring is a strong correct-response signal).

## How It Was Trained

**Algorithm:** Discriminative Contrastive Divergence (full-batch)

| Hyperparameter | Value |
|---|---|
| Training pairs | 400 (80% of 500 generated) |
| Feature dimension | 200 binary features |
| Learning rate | 0.01 |
| L1 regularization | 0.0 |
| Weight decay | 0.005 |
| Epochs | 300 |
| Source | Exp 62 (domain CD) + Exp 89 (self-bootstrap) |

**Training data:** Programmatically generated syllogism triples — four types:
1. **Modus ponens** (valid): "If all X are P, and all Y are X, what follows?" → "All Y are P. This follows by modus ponens."
2. **Modus tollens** (valid): "If all X are P, and Y are not P, what follows?" → "Y are not X. This follows by modus tollens."
3. **Disjunctive syllogism** (valid): "Either X are P or X are Q. X are not P. What follows?" → "X are Q. This follows by disjunctive syllogism."
4. **Affirming the consequent** (invalid): "If all X are P, and Y are P, are Y necessarily X?" → "No. This is affirming the consequent."

Wrong answers reverse the conclusion or deny the correct inference pattern.

## Benchmark Results

| Metric | This export | Exp 89 reference |
|--------|-------------|-----------------|
| AUROC (test) | **1.0000** | 1.0000 |
| Accuracy (test) | **100.0%** | 100.0% |
| Test set size | 100 | 31 |
| Baseline AUROC | 0.5 | 0.5 |

Perfect separation: the structural features of valid-logic responses
(keywords "follows by", "modus ponens", "modus tollens", "disjunctive
syllogism", "affirming the consequent") are highly discriminative.

## Usage

```python
import numpy as np
from carnot.inference.constraint_models import ConstraintPropagationModel

# Load model
model = ConstraintPropagationModel.from_pretrained(
    "exports/constraint-propagation-models/logic"
)

# Encode a response
from scripts.export_constraint_models import encode_answer
question = "If all cats are mortal, and all robots are cats, what follows?"
answer = "All robots are mortal. This follows by modus ponens."
x = encode_answer(question, answer)

energy = model.energy(x)  # low = valid syllogism
score = model.score(x)    # high = more likely a correct logic response
print(f"Energy: {energy:.2f}, Score: {score:.3f}")
```

## Limitations

1. **Keyword-dependent**: Perfect AUROC is achieved because valid syllogism
   responses contain distinctive keywords ("modus ponens", "follows by").
   A correct response without these keywords may score poorly.
2. **Four syllogism types only**: Trained on modus ponens, modus tollens,
   disjunctive syllogism, and affirming the consequent. Other logic forms
   (hypothetical syllogism, constructive dilemma) are out of scope.
3. **Template-generated training data**: Real LLM responses vary in phrasing
   and may not contain the expected keyword cues.
4. **Feature encoder required**: Must use the same 200-dim structural encoder.

## Files

| File | Description |
|------|-------------|
| `model.safetensors` | Coupling matrix J (200×200) and bias b (200,) as float32 |
| `config.json` | Training metadata and benchmark results |
| `README.md` | This file |

## Citation

```bibtex
@misc{carnot2026constraint_logic,
  title  = {Carnot Constraint Propagation Model: Logic},
  author = {Carnot-EBM},
  year   = {2026},
  url    = {https://github.com/ianblenke/carnot}
}
```

## Spec

- REQ-VERIFY-002, REQ-VERIFY-003, FR-11
