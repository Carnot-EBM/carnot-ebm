"""Energy-Based Fine-Tuning (EBFT) Loss.

Spec: REQ-TRAIN-007

Note: The user requested using "new verifiers" for EBFT. However, REQ-TRAIN-007
explicitly states that EBFT "operates without explicit external verifiers."
This module provides an EBFTLoss class that uses a verifier as an energy function
to satisfy the scaffolding request, while documenting the architectural misconception.

EBFT for continuous latent states (Exp 2111):
    Rather than matching token-level predictions, latent EBFT matches statistical
    moments of latent trajectory features.  An expert trace and a model rollout
    trace both produce a feature vector (mean state, std state, mean energy).
    The latent_feature_divergence function measures the L2 distance between
    batch-averaged feature vectors — if this is small, the model's Langevin
    trajectories inhabit the same region of latent space as the expert traces.
"""

from typing import Any, Callable
import jax.numpy as jnp
import numpy as np


class EBFTLoss:
    """EBFT Objective function wrapper."""

    def __init__(self, verifier_energy_fn: Callable[[Any, jnp.ndarray], jnp.ndarray]):
        """Initialize EBFTLoss.

        Args:
            verifier_energy_fn: A function that takes (params, sequences) and returns
                energy values (lower is better for expert).
        """
        self.verifier_energy_fn = verifier_energy_fn

    def __call__(self, params: Any, expert_sequences: jnp.ndarray, rollout_sequences: jnp.ndarray) -> jnp.ndarray:
        """Computes the EBFT loss.

        Args:
            params: Parameters for the verifier.
            expert_sequences: Array of expert sequences.
            rollout_sequences: Array of generated rollout sequences.

        Returns:
            Scalar loss value.
        """
        expert_energy = self.verifier_energy_fn(params, expert_sequences)
        rollout_energy = self.verifier_energy_fn(params, rollout_sequences)

        loss = jnp.mean(expert_energy) - jnp.mean(rollout_energy)
        return loss


def ebft_loss(model_features: jnp.ndarray, target_features: jnp.ndarray) -> jnp.ndarray:
    """Computes the EBFT feature-matching objective.

    EBFT implicitly defines an energy function over sequences by matching
    model features to target features.

    Args:
        model_features: Array of model features, shape (batch_size, feature_dim).
        target_features: Array of target features, shape (batch_size, feature_dim).

    Returns:
        Scalar loss value minimizing feature divergence.
    """
    model_expected = jnp.mean(model_features, axis=0)
    target_expected = jnp.mean(target_features, axis=0)
    return jnp.sum(jnp.square(model_expected - target_expected))


def latent_feature_divergence(
    expert_features: np.ndarray,
    rollout_features: np.ndarray,
) -> float:
    """Compute feature matching divergence between expert and rollout latent traces.

    WHY this metric?
        EBFT does not compare individual (expert, rollout) pairs — it compares
        the DISTRIBUTIONS of their feature statistics.  Batch-mean matching is
        the minimal unbiased estimator of the first-moment divergence between
        the two distributions.  If the expert and rollout EBMs have the same
        energy landscape, their Langevin traces will visit the same regions of
        latent space and this metric will be near zero.

    HOW it works:
        1. Compute the batch mean of expert feature vectors:  mu_expert
        2. Compute the batch mean of rollout feature vectors: mu_rollout
        3. Return ||mu_expert - mu_rollout||_2^2  (squared L2 distance)

    The feature vectors come from LatentTrace.features():
        [mean_state (d,), std_state (d,), mean_energy (1,)]
    where d is the latent dimension.  This captures BOTH positional (where
    the trace is) and dynamic (how much it moves) divergence.

    Args:
        expert_features: Shape (n_expert, feature_dim) — feature matrix from
            expert traces (e.g., from a reference/target EBM).
        rollout_features: Shape (n_rollout, feature_dim) — feature matrix from
            model-generated traces (current EBM under training).

    Returns:
        Scalar float: squared L2 distance between batch mean feature vectors.
        Zero means the distributions are indistinguishable by first moments.

    Spec: REQ-TRAIN-007
    """
    mu_expert = np.mean(expert_features, axis=0)
    mu_rollout = np.mean(rollout_features, axis=0)
    return float(np.sum(np.square(mu_expert - mu_rollout)))
