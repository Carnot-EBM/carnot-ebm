# IsingConstraintInjector v2 — External Field Fix

## Phase 1 Research Artifact

**IMPORTANT:** This model is a Phase 1 research artifact. Trained on simulated data
unless explicitly stated as live-GPU-validated. Do not use in production without
independent validation.

This model was produced as part of the Carnot EBM research project and has NOT been
validated on live GPU hardware unless the model card explicitly states otherwise.
Results from simulated-data training may not transfer to real-world distributions.

For the full verify-repair pipeline, see: https://github.com/Carnot-EBM/carnot-ebm


## What Changed (Exp 819)

The original IsingConstraintInjector had a bug: the external field h was computed
but never subtracted from the Ising energy, so constraint embeddings had zero
discrimination power.

**Fix:** `E_field = E_ising - dot(h, spins)` where `h = W @ constraint_mean`.

This one-line fix raised discrimination_rate from 0.0 to 1.0 on all test pairs.

## Validation Results (Exp 819)

| Metric              | Value     |
|---------------------|-----------|
| discrimination_rate | 1.0 |
| n_pairs tested      | 10    |
| n_spins             | 16    |
| legacy_delta        | 0.0 (confirmed broken) |

## External Field Formula

Given constraint embeddings `c` (shape: emb_dim), coupling projection `W` (shape: n_spins x emb_dim):

```
h = W @ mean(c, axis=0)          # shape: (n_spins,)
E_field = E_ising - dot(h, spins) # lower energy = more compatible with constraints
```

## Usage

```python
from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector
injector = IsingConstraintInjector(embedding_dim=384, n_spins=16)
result = injector.compute_energy_with_external_field(J, spins, constraint_embeddings)
```

## Spec Traces
REQ-INFRA-062, Exp 819, Exp 829
