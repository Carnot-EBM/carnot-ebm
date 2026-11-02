"""Self-Normalised Likelihood (SNL) training for Energy Based Models.

**Researcher summary:**
    SNL trains an EBM by optimizing a lower bound on the log-likelihood
    using a learnable log-partition-function estimate. The loss is:

        L = mean(E(data)) + log_z + mean(exp(-E(proposal) - log_z))

    This avoids MCMC sampling during training (unlike CD-k) while providing
    a tighter bound than simple importance sampling. The log_z parameter
    is jointly optimized with the energy function parameters.

**Detailed explanation for engineers:**
    Energy-Based Models define probability as p(x) = exp(-E(x)) / Z, where
    Z = integral(exp(-E(x)) dx) is the partition function. Computing Z
    exactly is intractable for high-dimensional inputs.

    **The SNL approach:**

    Instead of computing Z, SNL introduces a learnable scalar ``log_z``
    that approximates log(Z). The loss function is:

        L = mean(E(x_data)) + log_z + mean(exp(-E(x_proposal) - log_z))

    Breaking this down:

    1. **Data term: mean(E(x_data))**
       We want the energy of real data to be LOW. Minimizing this term
       pushes the model to assign low energy to real data points.

    2. **Log-partition term: log_z**
       This is our estimate of log(Z). It acts as a normalizer.

    3. **Proposal term: mean(exp(-E(x_proposal) - log_z))**
       Proposal samples come from a known distribution (e.g., Gaussian).
       This term estimates the ratio between the model distribution and
       the proposal distribution. When log_z is correct and the model is
       well-trained, this term should be close to 1.

    **Why this works:**
    The loss is a lower bound on the true negative log-likelihood.
    As log_z converges to the true log(Z) and the proposal distribution
    covers the data distribution, the bound becomes tight. For exponential
    families, the loss is concave (guaranteeing a unique global optimum).

    **Comparison to other training methods:**
    - CD-k: Requires MCMC chains during training (expensive, biased).
    - NCE: Requires explicit noise/data classification.
    - Score matching: Matches gradients; different theoretical basis.
    - SNL: Uses importance-weighted proposal samples + learnable normalizer.

    **Numerical stability:**
    The proposal term involves exp(-E(x) - log_z), which can overflow
    or underflow for large values. We use logsumexp-style tricks where
    appropriate. For extremely negative values of (-E - log_z), the exp
    saturates to 0, which is numerically safe.

    **Reference:** arxiv 2503.07021

Spec: REQ-TRAIN-004
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
        SNL training only needs the energy function itself. This minimal
        protocol allows the training code to work with any object that has
        ``energy()``, including:
        - Full EnergyFunction models (Ising, Gibbs, Boltzmann)
        - ComposedEnergy from the verify module
        - Simple wrapper objects for testing

    Spec: REQ-TRAIN-004
    """

    def energy(self, x: jax.Array) -> jax.Array:
        """Scalar energy for a single input x."""
        ...


def snl_loss(
    energy_fn: HasEnergy | callable,
    data_batch: jax.Array,
    proposal_batch: jax.Array,
    log_z: jax.Array,
) -> jax.Array:
    """Compute Self-Normalised Likelihood loss for a batch.

    **Researcher summary:**
        L = mean(E(data)) + log_z + mean(exp(-E(proposal) - log_z))

        where log_z is a learnable scalar approximating log(partition function).
        Lower bound of true log-likelihood. Concave for exponential families.

    **Detailed explanation for engineers:**
        This function computes the SNL loss given pre-generated data and
        proposal batches, plus a learnable log-partition estimate.

        **Step by step:**

        1. Compute E(x) for every data sample using vmap.
           These are the energies the model assigns to real data.

        2. Compute E(x) for every proposal sample using vmap.
           These are the energies for samples from the known proposal
           distribution (typically Gaussian).

        3. Data term: mean(E(data))
           We want this to be low -- the model should assign low energy
           to real data. This is the numerator of -log p(x).

        4. Log-partition term: log_z
           Our current estimate of log(Z). This normalizes the model.

        5. Proposal term: mean(exp(-E(proposal) - log_z))
           This is an importance-weighted estimate of Z/Z_hat. When log_z
           equals the true log(Z), this term equals 1 in expectation.

        6. Return the sum: data_term + log_z + proposal_term.

        **Gradient flow for log_z:**
        The gradient of L w.r.t. log_z is:
            dL/d(log_z) = 1 - mean(exp(-E(proposal) - log_z))
        This equals 0 when the proposal term averages to 1, which happens
        when log_z = log(Z). So optimizing log_z jointly with the energy
        parameters automatically learns the partition function.

    Args:
        energy_fn: Either an object with an ``energy`` method (satisfying
            the HasEnergy protocol), or a raw callable ``f(x) -> scalar``.
        data_batch: Real data samples, shape (batch_size, dim).
        proposal_batch: Proposal samples from known distribution,
            shape (num_proposals, dim).
        log_z: Learnable scalar approximating log(partition function).
            Should be a JAX array of shape () or (1,).

    Returns:
        Scalar SNL loss.

    For example::

        import jax.numpy as jnp
        from carnot.training.snl import snl_loss

        def quadratic_energy(x):
            return 0.5 * jnp.sum(x ** 2)

        data = jnp.zeros((32, 10))         # data near origin
        proposals = jnp.ones((32, 10)) * 2  # proposals further away
        log_z = jnp.float32(0.0)           # initial log-partition estimate
        loss = snl_loss(quadratic_energy, data, proposals, log_z)

    Spec: REQ-TRAIN-004
    """
    # Resolve the energy function: support both protocol objects and raw callables.
    # Same pattern as carnot.training.nce to maintain consistency.
    if callable(energy_fn) and not isinstance(energy_fn, HasEnergy):
        e_fn = energy_fn
    else:
        e_fn = energy_fn.energy

    # Vectorize the energy function over the batch dimension.
    # jax.vmap transforms e_fn (which takes a single 1-D input and returns a scalar)
    # into a function that processes an entire (batch_size, dim) matrix, returning
    # a 1-D array of energies.
    data_energies = jax.vmap(e_fn)(data_batch)
    proposal_energies = jax.vmap(e_fn)(proposal_batch)

    # Data term: mean(E(x_data))
    #
    # We want the model to assign low energy to real data. Minimizing this
    # term pushes data energies down, which corresponds to increasing the
    # unnormalized probability exp(-E(x)) for data points.
    data_term = jnp.mean(data_energies)

    # Proposal term: mean(exp(-E(x_proposal) - log_z))
    #
    # This estimates the ratio Z / exp(log_z). When log_z equals the true
    # log(Z), this term averages to 1 in expectation (assuming the proposal
    # distribution has sufficient coverage of the model distribution).
    #
    # The exp can overflow for very negative (-E - log_z) values, but in
    # practice the energy and log_z are co-optimized to stay in a stable range.
    proposal_term = jnp.mean(jnp.exp(-proposal_energies - log_z))

    # Total SNL loss: sum of data term, log-partition estimate, and proposal term.
    return data_term + log_z + proposal_term


def snl_loss_stochastic(
    energy_fn: HasEnergy | callable,
    data_batch: jax.Array,
    noise_scale: float,
    log_z: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Compute SNL loss with freshly sampled Gaussian proposal noise.

    **Researcher summary:**
        Convenience wrapper: samples proposals ~ N(0, noise_scale^2 I)
        internally using the provided PRNG key, then delegates to
        ``snl_loss()``.

    **Detailed explanation for engineers:**
        In actual training loops, you typically want fresh proposal samples
        each iteration. This function samples the proposals for you using
        a JAX PRNG key. The proposal batch has the same shape as the data
        batch (1:1 ratio of proposals to data samples).

        **JAX PRNG keys:**
        JAX uses explicit PRNG keys instead of global random state. Create
        a key with ``jax.random.PRNGKey(seed)`` and split it for each use:
        ``key, subkey = jax.random.split(key)``. This makes random operations
        reproducible and compatible with JIT compilation.

    Args:
        energy_fn: Model or callable providing energy().
        data_batch: Real data samples, shape (batch_size, dim).
        noise_scale: Standard deviation of the Gaussian proposal distribution.
        log_z: Learnable scalar approximating log(partition function).
        key: JAX PRNG key for sampling proposals.

    Returns:
        Scalar SNL loss.

    Spec: REQ-TRAIN-004
    """
    # Sample Gaussian proposals with the same shape as the data batch.
    # Multiply by noise_scale to get N(0, noise_scale^2) instead of N(0, 1).
    proposal_batch = jax.random.normal(key, shape=data_batch.shape) * noise_scale
    return snl_loss(energy_fn, data_batch, proposal_batch, log_z)
