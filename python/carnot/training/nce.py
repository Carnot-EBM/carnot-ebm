"""Noise Contrastive Estimation (NCE) training for Energy Based Models.

**Researcher summary:**
    NCE trains an EBM by framing density estimation as binary classification
    between real data and noise samples from a known distribution. The loss is:

        L = -mean(log sigmoid(-E(x_data))) - mean(log sigmoid(E(x_noise)))

    This avoids the intractable partition function entirely.

**Detailed explanation for engineers:**
    Energy-Based Models define probability as p(x) = exp(-E(x)) / Z, where
    Z is the partition function (an integral over all possible inputs). For
    high-dimensional continuous inputs, Z is impossible to compute exactly.

    **The NCE trick:**

    Instead of computing Z, NCE frames training as a binary classification task:

    1. Take real data samples x_data (label = "real").
    2. Draw noise samples x_noise from a known distribution like a Gaussian
       (label = "noise"). We know the noise density p_noise(x) exactly.
    3. Train the energy function so a simple classifier can tell them apart:
       - Real data should get LOW energy (high probability).
       - Noise should get HIGH energy (low probability).

    The classifier uses sigmoid(-E(x)) as "probability that x is real":
    - sigmoid(-E(x)) close to 1 when E(x) is very negative (low energy = real).
    - sigmoid(-E(x)) close to 0 when E(x) is very positive (high energy = noise).

    The NCE loss is the negative log-likelihood of this binary classifier:

        L = -mean(log sigmoid(-E(x_data))) - mean(log sigmoid(E(x_noise)))

    **Why this works:**
    Gutmann & Hyvarinen (2010) proved that as the number of noise samples
    increases, the NCE estimator converges to the maximum likelihood estimator.
    The noise distribution provides the normalization reference that the
    partition function Z would otherwise provide.

    **Numerical stability:**
    We use JAX's log_sigmoid function which is numerically stable:
        log_sigmoid(z) = z - softplus(z) = z - log(1 + exp(z))
    This avoids overflow for large |z|.

    **Comparison to other training methods:**
    - CD-k: Requires running MCMC chains during training (expensive, biased).
    - Score matching: Matches gradients; works well but different theory.
    - NCE: Reduces to classification — conceptually simplest.

    **Reference:** Gutmann & Hyvarinen (2010), "Noise-Contrastive Estimation of
    Unnormalized Statistical Models, with Applications to Natural Image
    Statistics."

Spec: REQ-TRAIN-003
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class HasEnergy(Protocol):
    """Minimal protocol for models that provide energy().

    **Researcher summary:**
        Structural subtype requiring only ``energy(x) -> scalar``.
        Satisfied by any EnergyFunction subclass.

    **Detailed explanation for engineers:**
        NCE training only needs the energy function itself — it does not need
        gradients (unlike score matching). This minimal protocol allows the
        training code to work with any object that has ``energy()``, including:
        - Full EnergyFunction models (Ising, Gibbs, Boltzmann)
        - ComposedEnergy from the verify module
        - Simple wrapper objects for testing
    """

    def energy(self, x: jax.Array) -> jax.Array:
        """Scalar energy for a single input x."""
        ...


def nce_loss(
    energy_fn: HasEnergy | callable,
    data_batch: jax.Array,
    noise_batch: jax.Array,
) -> jax.Array:
    """Compute Noise Contrastive Estimation loss for a batch.

    **Researcher summary:**
        L = -mean(log sigmoid(-E(x_data))) - mean(log sigmoid(E(x_noise)))

    **Detailed explanation for engineers:**
        This function computes the NCE loss given pre-generated data and noise
        batches. The noise samples should be provided by the caller (typically
        drawn from a Gaussian). This design makes the function deterministic
        and JIT-compilable.

        **Step by step:**
        1. Compute E(x) for every data sample using vmap.
        2. Compute E(x) for every noise sample using vmap.
        3. Data term: -mean(log sigmoid(-E(x_data)))
           We want E(data) to be low, so sigmoid(-E) should be close to 1.
        4. Noise term: -mean(log sigmoid(E(x_noise)))
           We want E(noise) to be high, so sigmoid(E) should be close to 1.
        5. Return sum of both terms.

        **JAX log_sigmoid:**
        ``jax.nn.log_sigmoid(z)`` = log(sigmoid(z)) computed in a numerically
        stable way. We use this instead of ``jnp.log(jax.nn.sigmoid(z))`` to
        avoid log(0) when sigmoid saturates.

    Args:
        energy_fn: Either an object with an ``energy`` method (satisfying
            the HasEnergy protocol), or a raw callable ``f(x) -> scalar``.
        data_batch: Real data samples, shape (batch_size, dim).
        noise_batch: Noise samples from known distribution, shape (num_noise, dim).

    Returns:
        Scalar NCE loss.

    For example::

        import jax.numpy as jnp
        from carnot.training.nce import nce_loss

        def quadratic_energy(x):
            return 0.5 * jnp.sum(x ** 2)

        data = jnp.zeros((32, 10))      # data near origin
        noise = jnp.ones((32, 10)) * 5  # noise far away
        loss = nce_loss(quadratic_energy, data, noise)

    Spec: REQ-TRAIN-003
    """
    # Resolve the energy function: support both protocol objects and raw callables.
    if callable(energy_fn) and not isinstance(energy_fn, HasEnergy):
        e_fn = energy_fn
    else:
        e_fn = energy_fn.energy

    # Vectorize the energy function over the batch dimension.
    # jax.vmap transforms e_fn (which takes a single 1-D input and returns a scalar)
    # into a function that processes an entire (batch_size, dim) matrix, returning
    # a 1-D array of energies. This is much faster than a Python loop.
    data_energies = jax.vmap(e_fn)(data_batch)
    noise_energies = jax.vmap(e_fn)(noise_batch)

    # Data term: -mean(log sigmoid(-E(x_data)))
    #
    # We want the model to assign low energy to real data. When E is low (negative
    # or small), sigmoid(-E) is close to 1, and log sigmoid(-E) is close to 0.
    # So the loss contribution from well-classified data is small.
    #
    # jax.nn.log_sigmoid(-E) computes log(sigmoid(-E)) in a numerically stable way.
    data_term = -jnp.mean(jax.nn.log_sigmoid(-data_energies))

    # Noise term: -mean(log sigmoid(E(x_noise)))
    #
    # We want the model to assign high energy to noise. When E is high,
    # sigmoid(E) is close to 1, and log sigmoid(E) is close to 0.
    # So the loss contribution from well-classified noise is small.
    noise_term = -jnp.mean(jax.nn.log_sigmoid(noise_energies))

    # Total NCE loss: sum of data and noise classification losses.
    return data_term + noise_term


def nce_loss_stochastic(
    energy_fn: HasEnergy | callable,
    data_batch: jax.Array,
    noise_scale: float,
    key: jax.Array,
) -> jax.Array:
    """Compute NCE loss with freshly sampled Gaussian noise.

    **Researcher summary:**
        Convenience wrapper: samples noise ~ N(0, noise_scale^2 I) internally
        using the provided PRNG key, then delegates to ``nce_loss()``.

    **Detailed explanation for engineers:**
        In actual training loops, you typically want fresh noise each
        iteration. This function samples the noise for you using a JAX PRNG
        key. The noise batch has the same shape as the data batch (1:1 ratio
        of noise to data samples).

        **JAX PRNG keys:**
        JAX uses explicit PRNG keys instead of global random state. Create a
        key with ``jax.random.PRNGKey(seed)`` and split it for each use:
        ``key, subkey = jax.random.split(key)``. This makes random operations
        reproducible and compatible with JIT compilation.

    Args:
        energy_fn: Model or callable providing energy().
        data_batch: Real data samples, shape (batch_size, dim).
        noise_scale: Standard deviation of the Gaussian noise distribution.
        key: JAX PRNG key for sampling noise.

    Returns:
        Scalar NCE loss.

    Spec: REQ-TRAIN-003
    """
    # Sample Gaussian noise with the same shape as the data batch.
    # Multiply by noise_scale to get N(0, noise_scale^2) instead of N(0, 1).
    noise_batch = jax.random.normal(key, shape=data_batch.shape) * noise_scale
    return nce_loss(energy_fn, data_batch, noise_batch)
