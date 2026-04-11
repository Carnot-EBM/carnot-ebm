---
tags:
  - energy-based-model
  - ising-model
  - constraint-satisfaction
  - code-verification
  - carnot
license: apache-2.0
---

> **Research Artifact — Not Production-Ready**
>
> This model verifies code implementation responses using structural binary
> features.  It achieves AUROC 0.867 on the held-out test set (Exp 89 reference:
> 0.9096).  It detects common off-by-one, wrong-initialization, and wrong-logic
> bugs — not arbitrary code errors.

# constraint-propagation-code

**Learned Ising constraint model for code implementation verification.**

Trained via discriminative Contrastive Divergence (Exp 62/89) on 400 verified
(question, correct, wrong) triples for code implementation tasks.  Assigns a
scalar energy to binary-encoded code responses — lower energy means the code
is more likely to be a correct implementation.

## What It Is

An Ising Energy-Based Model (EBM):

```
E(x) = -(b^T s + s^T J s)    s = 2x - 1 ∈ {-1, +1}^200
```

The coupling matrix J encodes co-occurrence patterns between structural code
features that distinguish correct from buggy implementations (e.g., "has
`range(1, n + 1)`" is strongly coupled to correctness in sum-range tasks).

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

**Training data:** Ten code implementation templates with exactly one bug per
wrong implementation:

| Function | Correct pattern | Bug pattern |
|----------|----------------|-------------|
| sum_range | `range(1, n + 1)` | `range(1, n)` (off-by-one) |
| find_max | `result = lst[0]` | `result = 0` (wrong init) |
| is_even | `n % 2 == 0` | `n % 2 == 1` (inverted) |
| factorial | base case `n == 0` | base case `n == 1` (misses 0!) |
| reverse_string | `s[::-1]` | `s[::1]` (no-op) |
| count_vowels | `s.lower()` | missing `lower()` |
| fibonacci | base case `n <= 0` | base case `n == 1` |
| binary_search | `lo = mid + 1` | `lo = mid` (infinite loop) |
| is_palindrome | `s == s[::-1]` | `s == ''.join(sorted(s))` |
| flatten | `result.extend(flatten(item))` | `result.extend(item)` (shallow) |

## Benchmark Results

| Metric | This export | Exp 89 reference |
|--------|-------------|-----------------|
| AUROC (test) | **0.8669** | 0.9096 |
| Accuracy (test) | **88.0%** | 88.0% |
| Test set size | 100 | 25 |
| Baseline AUROC | 0.5 | 0.5 |

The gap from Exp 89 reflects the richer constraint features (245-dim pipeline
features vs 200-dim structural encoding used here).  Accuracy matches perfectly.

## Usage

```python
import numpy as np
from carnot.inference.constraint_models import ConstraintPropagationModel

# Load model
model = ConstraintPropagationModel.from_pretrained(
    "exports/constraint-propagation-models/code"
)

# Encode a code response
from scripts.export_constraint_models import encode_answer
question = "Write a function that returns the sum of integers from 1 to n."
correct_code = "def sum_range(n):\n    total = 0\n    for i in range(1, n + 1):\n        total += i\n    return total"
buggy_code   = "def sum_range(n):\n    total = 0\n    for i in range(1, n):\n        total += i\n    return total"

x_correct = encode_answer(question, correct_code)
x_buggy   = encode_answer(question, buggy_code)

print(f"Correct energy: {model.energy(x_correct):.2f}")  # should be lower
print(f"Buggy energy:   {model.energy(x_buggy):.2f}")    # should be higher
print(f"Correct score:  {model.score(x_correct):.3f}")   # should be higher
print(f"Buggy score:    {model.score(x_buggy):.3f}")     # should be lower
```

## Limitations

1. **Template bugs only**: Trained on 10 specific bug patterns. Novel bug types
   (e.g., wrong algorithm choice, logic errors in business rules) are outside
   scope.
2. **Structural features only**: Detects structural signals like "has `range(1,
   n + 1)`" or "has `isinstance`" — does not execute or parse the code.
3. **Lowest AUROC of the three domains**: Code structure is less distinctive
   than arithmetic or logic patterns. AUROC 0.867 vs 1.0 for logic/arithmetic.
4. **No semantic understanding**: Two implementations with the same keywords
   but different logic will have similar energies.

## Files

| File | Description |
|------|-------------|
| `model.safetensors` | Coupling matrix J (200×200) and bias b (200,) as float32 |
| `config.json` | Training metadata and benchmark results |
| `README.md` | This file |

## Citation

```bibtex
@misc{carnot2026constraint_code,
  title  = {Carnot Constraint Propagation Model: Code},
  author = {Carnot-EBM},
  year   = {2026},
  url    = {https://github.com/ianblenke/carnot}
}
```

## Spec

- REQ-VERIFY-002, REQ-VERIFY-003, FR-11
