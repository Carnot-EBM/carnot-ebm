# NLA Feature Description Collision Audit

**Status:** Implementation complete; full GGUF-model audit blocked (gemma-4-31B-it-GGUF not cached).
**Experiment:** 2115
**Date:** 2026-05-15

## What is a Feature Description Collision?

When a Sparse Autoencoder (SAE) is trained on LLM activations, each learned
dictionary atom (decoder column vector) ideally encodes a distinct semantic
concept.  Auto-interpretability systems — whether human annotators or LLM
labellers — then assign natural language descriptions to each feature.

A **collision** occurs when two distinct SAE features point in nearly the same
direction in activation space, causing them to receive near-identical
descriptions (e.g., "activates on the word bank" and "activates on the word
bank (variant)").  Collisions indicate the SAE is wasting capacity on
redundant representations instead of learning a diverse, interpretable basis.

This matters for the NLA-class 16th verifier because: if two features
collapse onto the same direction, the ensemble's effective coverage of semantic
space is lower than `n_features` suggests, and the verifier may systematically
miss semantic errors that only appear in the uncovered subspace.

## Implementation

`NLAClassProbe.feature_description_collision_rate()` in
`python/carnot/verify/nla_probe.py` computes the collision rate as:

```
collision_rate = |{(i,j) : i < j, cos_sim(W[:,i], W[:,j]) >= threshold}|
                 / C(n_features, 2)
```

where `W` is the SAE decoder weight matrix (shape: d_model x d_sae) and each
column is one feature direction.  Cosine similarity is a label-free proxy for
description similarity: two features pointing within `acos(threshold)` degrees
of each other will receive near-identical descriptions from any reasonable
interpretability system.

Default threshold: **0.95** (approximately 18 degrees of separation).

## Synthetic Audit Results (exp2115)

Run on randomly-initialised SAE, d_model=256, n_features=1024, n_samples=128:

| Metric | Value |
|---|---|
| collision_rate | 0.0000 |
| n_collision_pairs | 0 |
| n_total_pairs | 523,776 |
| cosine_threshold | 0.95 |

**Interpretation:** Kaiming-initialised decoder weights in R^256 with 1024
features have near-zero collision rate by the Johnson-Lindenstrauss lemma —
random unit vectors in high-dimensional space are nearly orthogonal.  This is
the expected lower bound.  A trained SAE on real model activations (especially
after many gradient steps with L1 regularisation) may show higher collision
rates because:

1. Model activations are highly anisotropic (most variance concentrated in a
   small number of directions).
2. L1 regularisation can cause features to degenerate toward the principal
   eigenvectors of the activation covariance, producing redundant atoms.
3. Small expansion factors (d_sae / d_model << 10) increase pressure on the
   dictionary to over-use the high-variance directions.

## Mitigation Strategies (if collision_rate > 0.15)

If the production audit with real gemma-4-31B-it-GGUF activations shows
collision_rate > 15%, the following mitigations are recommended in order of
expected impact:

### 1. Increase Expansion Factor

Increase `expansion_factor` from 4 to 8 or 16.  More dictionary atoms gives
the SAE more room to spread features across diverse directions, reducing
pressure on high-variance principal components.

**Trade-off:** Higher memory cost.  For d_model=5376 (gemma-4-31B) and
expansion=16, d_sae=86,016 — feasible on RTX 3090 for inference but adds
training overhead.

### 2. Add Diversity Loss

Add a pairwise orthogonality penalty to the SAE training loss:

```python
# WHY: pushes decoder columns apart during training, penalising
# pairs that are becoming redundant before they fully collapse.
W_norm = F.normalize(sae.decoder.weight, dim=0)  # (d_model, d_sae)
gram = W_norm.T @ W_norm  # (d_sae, d_sae)
off_diag = gram - torch.eye(d_sae, device=gram.device)
diversity_loss = (off_diag ** 2).mean()
total_loss = reconstruction_loss + l1_loss + lambda_div * diversity_loss
```

`lambda_div=1e-4` is a reasonable starting point; tune until collision_rate
drops below 0.10 without degrading reconstruction quality.

### 3. Periodic Resampling of Dead Features

Features that are almost never active ("dead features") are wasted capacity
that other alive features compensate for, sometimes by splitting into redundant
twins.  Resample dead features periodically:

```python
# WHY: prevents the common SAE failure where 20-40% of features become
# permanently inactive, causing alive features to become redundant twins.
if step % resample_every == 0:
    activation_counts = compute_feature_activation_counts(sae, data_loader)
    dead_mask = activation_counts < dead_threshold
    with torch.no_grad():
        # reinitialise dead decoder columns to random directions
        n_dead = dead_mask.sum().item()
        sae.decoder.weight[:, dead_mask] = F.normalize(
            torch.randn(d_model, n_dead, device=sae.decoder.weight.device),
            dim=0
        )
```

### 4. Gram-Schmidt Orthogonalisation at Initialisation

Start with a maximally-spread initialisation to give the SAE the best chance
of learning diverse features:

```python
# WHY: random Kaiming init works well in high d_model, but for d_sae >> d_model
# the decoder columns MUST partially align (pigeonhole). Initialising from a
# random orthonormal frame in the first d_model directions reduces early
# redundancy and speeds convergence.
W_init = torch.randn(d_sae, d_model)
Q, _ = torch.linalg.qr(W_init.T, mode='reduced')  # (d_model, d_model)
# pad remaining columns with random unit vectors
if d_sae > d_model:
    extra = F.normalize(torch.randn(d_sae - d_model, d_model), dim=1)
    W_init = torch.cat([Q.T, extra], dim=0)
else:
    W_init = Q.T[:d_sae]
sae.decoder.weight.data = W_init.T  # shape: (d_model, d_sae)
```

### 5. Lower Cosine Threshold for Evaluation (Diagnostic Only)

If the above mitigations are applied and collision_rate at threshold=0.95 drops
below 5%, re-check at threshold=0.80 to ensure the SAE is not hiding
softer redundancies.  This is a diagnostic step, not a fix — it reveals
whether apparent improvement at 0.95 reflects genuine feature diversification
or just slightly-wider collapse.

## Blocked Audit Note

The production audit with `unsloth/gemma-4-31B-it-GGUF` could not run because
the model is not cached locally.  The `feature_description_collision_rate()`
implementation is complete and all 5 new tests pass.  To run the full audit:

```bash
huggingface-cli download unsloth/gemma-4-31B-it-GGUF
python -c "
from carnot.verify.nla_probe import NLAClassProbe
import torch
# Replace with activations extracted from the target model
activations = torch.randn(512, 5376)  # d_model=5376 for gemma-4-31B
probe = NLAClassProbe(d_model=5376, expansion_factor=4)
result = probe.feature_description_collision_rate(activations)
print(result)
"
```

The gemma-4-26B-A4B-it-GGUF model (same architecture family, d_model likely
~3840) is available locally and can serve as a proxy audit while the 31B
variant is downloaded.
