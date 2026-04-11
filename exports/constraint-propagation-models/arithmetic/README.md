---
tags:
  - energy-based-model
  - ising-model
  - constraint-satisfaction
  - arithmetic-verification
  - carnot
license: apache-2.0
---

> **Research Artifact — Not Production-Ready**
>
> This model verifies arithmetic responses using structural binary features,
> not semantic understanding. It achieves AUROC 0.997 on the held-out test
> set (Exp 89 self-bootstrap reference: 1.0). Factual or freeform answers
> are outside its scope.

# constraint-propagation-arithmetic

**Learned Ising constraint model for arithmetic response verification.**

This model was trained via discriminative Contrastive Divergence (Exp 62/89)
on 400 verified (question, correct answer, wrong answer) triples for arithmetic
problems: addition, multiplication, and modular arithmetic.  It assigns a scalar
energy to any binary feature vector encoding a response.  Lower energy means
the response is more likely correct.

## What It Is

An Ising Energy-Based Model (EBM) with quadratic energy:

```
E(x) = -(b^T s + s^T J s)    s = 2x - 1 ∈ {-1, +1}^200
```

Where:
- `x` is a 200-dimensional **binary feature vector** encoding structural
  properties of an LLM response (presence of numbers, operators, answer
  patterns — see *Feature Encoding* below).
- `J` is a 200×200 **coupling matrix** (symmetric, zero diagonal) — encodes
  which feature pairs co-occur in arithmetic-correct responses.
- `b` is a 200-dim **bias vector** — encodes which features tend to be
  active in correct responses.

Lower energy = response looks structurally like a correct arithmetic answer.

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

**Training data:** Programmatically generated (no LLM) triples of the form
`(question, correct_answer, wrong_answer)` for three sub-types:
- Addition: "What is A + B?" — wrong answers off-by-one, off-by-ten, carry errors.
- Multiplication: "What is A * B?" — wrong answers from adjacent products.
- Modular arithmetic: "What is A mod B?" — wrong answers ±1 or ±2.

## Benchmark Results

| Metric | This export | Exp 89 reference |
|--------|-------------|-----------------|
| AUROC (test) | **0.9970** | 1.0000 |
| Accuracy (test) | **99.0%** | 100.0% |
| Test set size | 100 | 29 |
| Baseline AUROC | 0.5 | 0.5 |

The slight gap from Exp 89 is expected: Exp 89 used pipeline-extracted
constraint features (245 dims) while this export uses the simpler 200-dim
structural encoding from Exp 62.

## Feature Encoding

Each response is encoded as a 200-dimensional binary vector.  The `encode_answer`
function in `scripts/experiment_62_domain_constraint_learning.py` generates
these features:

| Group | Features | Examples |
|-------|----------|---------|
| Numeric (20) | digit/operator presence | has_number, has_plus, has_equals |
| Structural (40) | word/char counts, punctuation | short_answer, has_period |
| Domain-specific (80) | arithmetic/logic/code patterns | has_equation, has_if_then |
| Consistency (60) | question-answer agreement | q_numbers_in_answer |

## Usage

```python
import numpy as np
from carnot.inference.constraint_models import ConstraintPropagationModel

# Load model (local directory or HuggingFace Hub)
model = ConstraintPropagationModel.from_pretrained(
    "exports/constraint-propagation-models/arithmetic"
)

# Encode a response as a 200-dim binary feature vector
from scripts.export_constraint_models import encode_answer
x = encode_answer("What is 47 + 28?", "The answer is 75.")

# Verify
energy = model.energy(x)    # lower = more likely correct
score = model.score(x)      # in (0,1); higher = more likely correct
print(f"Energy: {energy:.2f}, Score: {score:.3f}")
```

## Inspect Coupling Matrix

```python
from safetensors.numpy import load_file
tensors = load_file("model.safetensors")
J = tensors["coupling"]   # shape (200, 200), float32
b = tensors["bias"]       # shape (200,), float32
print(f"Sparsity: {(J == 0).mean():.1%}")
print(f"Max |J_ij|: {abs(J).max():.4f}")
```

## Limitations

1. **Structural features only**: Verifies whether a response *looks like* a
   correct arithmetic answer (contains the right numbers, uses the right
   phrasing). Does not parse or evaluate arithmetic expressions.
2. **Template-generated training data**: Training pairs come from programmatic
   generators, not real LLM outputs. May not generalize to unusual phrasings.
3. **200-dim binary features**: Fine-grained semantic nuances are not captured.
   A plausible wrong answer with the same structural features as a correct one
   will have similar energy.
4. **Feature encoder required**: You must encode responses with the same 200-dim
   encoder used during training. Using a different encoder will produce garbage.
5. **factual/scheduling not supported**: AUROC near 0.5 for those domains in
   Exp 89 — the structural features don't discriminate factual correctness.

## Files

| File | Description |
|------|-------------|
| `model.safetensors` | Coupling matrix J (200×200) and bias b (200,) as float32 |
| `config.json` | Training metadata, benchmark results, limitations |
| `README.md` | This file |

## Citation

```bibtex
@misc{carnot2026constraint_arith,
  title  = {Carnot Constraint Propagation Model: Arithmetic},
  author = {Carnot-EBM},
  year   = {2026},
  url    = {https://github.com/ianblenke/carnot}
}
```

## Spec

- REQ-VERIFY-002: Energy-based verification of LLM responses.
- REQ-VERIFY-003: Correctness score derived from model energy.
- FR-11: Self-bootstrapped improvement of the verifier from pipeline outputs.
