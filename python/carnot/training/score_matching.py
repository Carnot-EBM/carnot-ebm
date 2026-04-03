"""Denoising Score Matching (DSM) training for Energy Based Models.

**Researcher summary:**
    Implements DSM loss: L = E_x E_noise [||grad_E(x+noise) + noise/sigma^2||^2].
    Avoids intractable partition function computation required by maximum
    likelihood. More stable than Contrastive Divergence (CD-k) for deep EBMs.

**Detailed explanation for engineers:**
    Training an Energy-Based Model is fundamentally different from training a
    classifier or regressor. The challenge is the **partition function**:

    The probability of a configuration x is:
        p(x) = exp(-E(x)) / Z

    where Z = integral of exp(-E(x)) over all possible x. For continuous,
    high-dimensional x, this integral is intractable (impossible to compute
    exactly). This means we cannot compute the log-likelihood or its gradient
    directly.

    **Why Denoising Score Matching (DSM) instead of CD-k?**

    The traditional approach is Contrastive Divergence (CD-k), which
    approximates the gradient of the log-likelihood using k steps of MCMC
    sampling. Problems with CD-k:
    - Biased gradient estimate (especially for small k)
    - Requires running a sampler during training (expensive)
    - Mode-mixing issues: the MCMC chain may not explore all modes
    - Training can be unstable for deep EBMs

    DSM sidesteps ALL of these issues. The key insight (Hyvarinen 2005,
    Vincent 2011) is:

    Instead of matching p(x) to the data distribution, match the **score
    function** (gradient of log-probability) to the data's score function.
    The score function does not depend on the partition function Z!

    The DSM loss is:
        L = E_x~data E_noise~N(0,sigma^2) [||s_theta(x+noise) - (-noise/sigma^2)||^2]

    Where:
    - s_theta(x) = -grad_E(x) is the model's score function
    - (-noise/sigma^2) is the score of the noisy data distribution
    - The expectation is over data samples and noise samples

    In plain English: we add Gaussian noise to training data, then train the
    model's gradient to point back toward the clean data. This is equivalent
    to matching the score function, but tractable.

    **Advantages of DSM:**
    - No sampling during training (just noise injection)
    - Unbiased gradient estimates
    - Stable training even for deep models
    - The noise level sigma controls the trade-off between bias and variance

Spec: REQ-TRAIN-002

DSM loss:
  L = E_x E_noise [ || grad_energy(x + noise) - (-noise / sigma^2) ||^2 ]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class HasGradEnergy(Protocol):
    """Minimal protocol for models that provide grad_energy.

    **Researcher summary:**
        Structural subtype requiring only ``grad_energy(x) -> array``.
        Satisfied by any EnergyFunction or AutoGradMixin subclass.

    **Detailed explanation for engineers:**
        DSM training only needs the model's gradient function — it does
        not need ``energy()`` or ``energy_batch()``. This minimal protocol
        allows the training code to work with any object that has
        ``grad_energy``, including:
        - Full EnergyFunction models (Ising, Gibbs, Boltzmann)
        - ComposedEnergy from the verify module
        - Simple lambda wrappers for testing
    """

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy with respect to x.

        The model's score function is s(x) = -grad_energy(x).
        """
        ...


def dsm_loss(
    grad_energy_fn: HasGradEnergy | callable,
    batch: jax.Array,
    noise: jax.Array,
    sigma: float,
) -> jax.Array:
    """Compute denoising score matching loss for a batch.

    **Researcher summary:**
        L = (1/N) * sum_i ||grad_E(x_i + noise_i) + noise_i / sigma^2||^2.
        Noise is pre-sampled (deterministic given noise array).

    **Detailed explanation for engineers:**
        This function computes the DSM loss for a batch of data points with
        pre-sampled noise. The loss measures how well the model's score
        function matches the "denoising direction" at each noisy data point.

        **Step by step:**
        1. Add noise to the data: x_noisy = x + noise
        2. Compute the "target" score: -noise / sigma^2
           (this is the direction that points from the noisy data back to
           the clean data, scaled by the noise variance)
        3. Compute the model's score at the noisy point: grad_E(x_noisy)
           Note: we use grad_E (not -grad_E) because the loss already
           accounts for the sign in the target.
        4. Compute the squared difference: ||model_score - target||^2
        5. Average over the batch

        **Why pre-sampled noise?**
        Having noise as a separate argument (rather than sampling it
        internally) makes the function deterministic and testable. For
        stochastic training, use ``dsm_loss_stochastic()`` which samples
        noise internally using a PRNG key.

        **jax.vmap usage:**
        ``jax.vmap(grad_fn)(noisy_batch)`` applies the gradient function
        to each row of the batch independently, producing a matrix of
        per-sample gradients. This is vastly more efficient than a Python
        loop.

    Args:
        grad_energy_fn: Either an object with a ``grad_energy`` method
            (satisfying the HasGradEnergy protocol), or a raw callable
            ``f(x) -> grad_energy``.
        batch: Data batch of shape (batch_size, dim).
        noise: Pre-sampled noise of shape (batch_size, dim). Must have the
            same shape as batch.
        sigma: Noise standard deviation. Controls the scale of perturbation.
            Larger sigma = smoother loss landscape but more bias.

    Returns:
        Scalar DSM loss (averaged over the batch).

    For example::

        import jax.numpy as jnp
        from carnot.models.ising import IsingModel, IsingConfig

        model = IsingModel(IsingConfig(input_dim=10))
        batch = jnp.ones((32, 10))  # 32 samples of dimension 10
        noise = jnp.zeros((32, 10))  # zero noise (for testing)
        loss = dsm_loss(model, batch, noise, sigma=0.1)

    Spec: REQ-TRAIN-002
    """
    # sigma^2 — the variance of the noise distribution
    sigma_sq = sigma * sigma
    # Add noise to data to create "corrupted" samples
    noisy_batch = batch + noise
    # Target score: the direction from noisy data back to clean data,
    # scaled by 1/sigma^2. This is the score of the Gaussian noise kernel.
    target = -noise / sigma_sq

    # Support both protocol objects (with .grad_energy method) and raw callables.
    # This flexibility lets users pass either a model or a simple function.
    if callable(grad_energy_fn) and not isinstance(grad_energy_fn, HasGradEnergy):
        grad_fn = grad_energy_fn
    else:
        grad_fn = grad_energy_fn.grad_energy

    # Vectorize the gradient function over the batch dimension.
    # jax.vmap transforms grad_fn (which takes a single 1-D input) into a
    # function that processes the entire (batch_size, dim) matrix at once.
    # This compiles into efficient vectorized GPU code.
    model_scores = jax.vmap(grad_fn)(noisy_batch)
    # Compute squared difference between model scores and target scores
    diff = model_scores - target
    # Sum squared differences across dimensions for each sample
    per_sample_loss = jnp.sum(diff**2, axis=-1)
    # Average across the batch
    return jnp.mean(per_sample_loss)


def dsm_loss_stochastic(
    grad_energy_fn: HasGradEnergy | callable,
    batch: jax.Array,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    """Compute DSM loss with freshly sampled Gaussian noise.

    **Researcher summary:**
        Convenience wrapper: samples noise ~ N(0, sigma^2 I) internally
        using the provided PRNG key, then delegates to ``dsm_loss()``.

    **Detailed explanation for engineers:**
        In actual training loops, you typically want fresh noise each
        iteration (for stochastic gradient estimation). This function
        samples the noise for you using a JAX PRNG key.

        **JAX PRNG keys:**
        JAX does not use global random state like NumPy. Instead, you
        create a key with ``jax.random.PRNGKey(seed)`` and pass it
        explicitly. To get different random values each iteration, split
        the key: ``key, subkey = jax.random.split(key)``. This design
        makes JAX code reproducible and compatible with JIT compilation.

    Args:
        grad_energy_fn: Model or callable providing grad_energy.
        batch: Data batch of shape (batch_size, dim).
        sigma: Noise standard deviation.
        key: JAX PRNG key for sampling noise. Pass a different key each
            training iteration for stochastic gradient estimation.

    Returns:
        Scalar DSM loss (averaged over the batch).

    For example::

        import jax
        import jax.numpy as jnp

        model = IsingModel(IsingConfig(input_dim=10))
        batch = jnp.ones((32, 10))

        key = jax.random.PRNGKey(42)
        for step in range(100):
            key, subkey = jax.random.split(key)
            loss = dsm_loss_stochastic(model, batch, sigma=0.1, key=subkey)

    Spec: REQ-TRAIN-002
    """
    # Sample Gaussian noise with the given PRNG key.
    # Shape matches the batch so each sample gets independent noise.
    # Multiply by sigma to get N(0, sigma^2) instead of N(0, 1).
    noise = jax.random.normal(key, shape=batch.shape) * sigma
    return dsm_loss(grad_energy_fn, batch, noise, sigma)
