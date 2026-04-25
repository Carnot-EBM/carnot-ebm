"""JEPA predictor domain-reweighted loss module.

**Why this module exists:**
    JEPA v24 (Exp 834) achieved SVAMP AUC=0.0 and overall OOD AUC=0.49 despite
    having SVAMP examples in training.  Root cause: SVAMP had only 10+10=20 training
    pairs while GSM8K/HumanEval/ARC each had 20+20=40.  With uniform per-sample loss,
    the model saw 2x fewer SVAMP examples and never learned SVAMP-specific patterns.

    This module provides DomainReweightedLoss, which assigns each domain a weight
    inversely proportional to the number of samples in that domain.  Domains with
    fewer samples get higher per-sample weight so that their total contribution to
    the training loss equals that of larger domains.

    Implementation follows DG-PRM (arXiv 2507.17849): the weighting is computed
    once from the corpus before training, then applied as a per-sample scalar
    multiplier on top of the BCE correctness loss.

Spec: REQ-LEARN-050
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax


class DomainReweightedLoss:
    """Per-domain loss reweighting that balances training signal across corpus domains.

    **Detailed explanation for engineers:**
        When a training corpus has imbalanced domain sizes (e.g. 40 GSM8K pairs but
        only 20 SVAMP pairs), a uniform per-sample loss gives the larger domain
        proportionally more gradient influence.  The model learns the larger domain's
        patterns well and ignores the smaller domain — the exact failure observed in
        JEPA v24 where SVAMP AUC=0.0.

        This class computes a weight for each domain:
            weight[domain] = 1.0 / (n_domain + 1e-6)
        then normalises so all weights sum to 1.0.

        The 1e-6 epsilon prevents division by zero for domains with zero samples
        (which would indicate a corpus-building bug, but we guard against it
        defensively rather than crashing).

        A balanced corpus (all domains same size) produces uniform weights.
        An imbalanced corpus (small domain) gives the small domain higher weight.

    Usage:
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)   # once, before training
        loss = loss_fn.weighted_loss(logits, labels, domain_ids, weights)
    """

    def compute_domain_weights(
        self, corpus: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute per-domain training loss weights from corpus domain frequencies.

        **Algorithm:**
            1. Count how many samples each domain contributes to the corpus.
            2. Compute raw weight = 1.0 / (count + 1e-6) for each domain.
            3. Normalise so weights.values() sum to 1.0.

        **Why inverse-frequency weighting:**
            Inverse-frequency weighting is the standard technique for handling
            class/domain imbalance in discriminative training.  It ensures the total
            gradient signal from every domain is equal, regardless of how many
            examples each domain contributes.

        Args:
            corpus: List of dicts.  Each dict must have a "domain" key (str).
                    Typically produced by an experiment's build_*_corpus() function.

        Returns:
            Dict mapping domain name → normalised float weight in (0, 1].
            Weights sum to 1.0 across all domains present in the corpus.
        """
        counts: dict[str, int] = {}
        for pair in corpus:
            domain = pair["domain"]
            counts[domain] = counts.get(domain, 0) + 1

        raw: dict[str, float] = {d: 1.0 / (n + 1e-6) for d, n in counts.items()}
        total = sum(raw.values())
        return {d: w / total for d, w in raw.items()}

    def weighted_loss(
        self,
        logits: jax.Array,
        labels: jax.Array,
        domain_ids: jax.Array,
        domain_weights: jax.Array,
    ) -> jax.Array:
        """Compute domain-reweighted binary cross-entropy loss over a batch.

        **Detailed explanation:**
            For each sample i in the batch:
                per_sample_loss[i] = BCE(sigmoid(logits[i]), labels[i])
                                     * domain_weights[domain_ids[i]]

            The final scalar loss is the mean of per_sample_loss over the batch.

            domain_weights here is a 1D array indexed by domain_idx, NOT the dict
            returned by compute_domain_weights().  The caller converts the dict to
            an array using the corpus's DOMAIN_NAMES ordering.

        Args:
            logits: Pre-sigmoid correctness logits, shape (batch,) or (batch, 1).
            labels: Binary correctness labels (0 or 1), same shape as logits.
            domain_ids: Integer domain indices, shape (batch,).
            domain_weights: Per-domain weight array, shape (n_domains,).
                            Indexed by domain_ids values.

        Returns:
            Scalar mean weighted BCE loss.
        """
        logits_flat = logits.reshape(-1)
        labels_flat = labels.reshape(-1)

        # Per-sample BCE loss using optax's numerically stable implementation.
        per_sample = optax.sigmoid_binary_cross_entropy(logits_flat, labels_flat)

        # Gather the weight for each sample's domain.
        sample_weights = domain_weights[domain_ids]

        return jnp.mean(per_sample * sample_weights)
