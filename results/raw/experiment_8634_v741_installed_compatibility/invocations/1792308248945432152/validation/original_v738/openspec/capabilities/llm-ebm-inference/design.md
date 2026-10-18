# LLM-EBM Inference — Design Document

**Capability:** llm-ebm-inference
**Version:** 0.1.0

## Architecture Overview

The LLM-EBM inference pipeline connects LLM text output to EBM verification:

```
LLM Output (text)
       │
       ▼
┌──────────────┐
│  PARSE       │  parse_llm_sat_assignment() / parse_llm_coloring()
│  (text→array)│
└──────┬───────┘
       │ jax.Array
       ▼
┌──────────────┐
│  VERIFY      │  ComposedEnergy.verify(x) → VerificationResult
│  (score)     │
└──────┬───────┘
       │ violations detected?
       ▼
┌──────────────┐
│  REPAIR      │  repair(energy, x) — gradient descent on violated constraints
│  (fix)       │
└──────┬───────┘
       │ continuous solution
       ▼
┌──────────────┐
│  ROUND       │  array_to_assignment() — snap to discrete domain
│  (discretize)│
└──────┬───────┘
       │ discrete solution
       ▼
┌──────────────┐
│  CERTIFY     │  ComposedEnergy.verify(rounded) → VerificationResult
│  (prove)     │
└──────────────┘
```

## Constraint Encoding: Continuous Relaxation

Both SAT and graph coloring are discrete problems. EBMs need differentiable energy.
We use continuous relaxation — the same approach as the Sudoku demo:

### SAT: Product Relaxation

For clause (l_1 OR l_2 OR ... OR l_k):

```
E_clause(x) = ∏_i (1 - val(l_i))
```

where val(l_i) = x_j if l_i is positive, (1 - x_j) if negated.

- Satisfied clause (any literal ≈ 1): product ≈ 0
- Violated clause (all literals ≈ 0): product ≈ 1
- Smooth, differentiable, all literals contribute gradient

Binary penalty to encourage discrete solutions:

```
E_binary(x) = Σ_i x_i(1 - x_i)
```

### Graph Coloring: Pairwise Repulsion

For edge (a, b):

```
E_edge(x) = max(0, 1 - |x_a - x_b|)²
```

Identical to Sudoku's UniquenessConstraint pairwise term.

## Verify-and-Repair Pipeline

```python
def verify_and_repair(assignment, energy, step_size=0.1, max_steps=200, round_fn=None):
    # 1. Initial verification
    initial = energy.verify(assignment)
    
    # 2. Gradient repair if violated
    if not initial.verdict.verified:
        repaired = repair(energy, assignment, step_size, max_steps)
    else:
        repaired = assignment
    
    # 3. Round to discrete
    if round_fn:
        rounded = round_fn(repaired)
        rounded_result = energy.verify(rounded)
    
    # 4. Return complete result
    return VerifyRepairResult(initial, repaired_result, rounded_result, trajectory)
```

## Domain Progression

```
SAT/CSP (this spec)  →  JSON schema  →  Type checking  →  Code correctness
    ↑                                                           ↑
  proving ground                                           end goal
```

SAT is chosen as the first domain because:
1. Constraints are trivially encodable as energy terms
2. LLMs are measurably bad at SAT (clear improvement signal)
3. Carnot already has all repair machinery (BaseConstraint, ComposedEnergy, repair, certificates)
4. SAT → code is a natural progression (type checking = constraint satisfaction)
