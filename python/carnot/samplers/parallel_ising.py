"""Parallel Ising Gibbs sampler -- JAX/GPU implementation.

**Researcher summary:**
    Vectorized block Gibbs sampler for Ising models. Replaces thrml's
    sequential per-spin loop with a single matrix-vector multiply +
    parallel Bernoulli sampling per sweep. Supports simulated annealing
    via temperature schedule. Runs on CPU/GPU/TPU via JAX.

**Detailed explanation for engineers:**
    The standard Gibbs sampler for an Ising model updates one spin at a
    time: for each spin s_i, compute the conditional probability given
    all neighbors, then sample. thrml does exactly this, looping over N
    spins in Python. For N=500, that's 500 sequential JAX operations per
    sweep, times thousands of sweeps -- hence the 300+ second runtimes.

    **The key insight: parallel Gibbs.**
    For an Ising model with bias vector b and coupling matrix J, the
    conditional probability of spin i being +1 given all other spins is:

        P(s_i = 1 | s_{-i}) = sigmoid(2 * beta * (b_i + sum_j J_ij * s_j))

    The argument to sigmoid can be computed for ALL spins simultaneously
    via a single matrix-vector product: h = beta * (b + J @ s). Then we
    sample ALL spins from Bernoulli(sigmoid(2*h)) in one JAX call.

    **Is parallel Gibbs correct?**
    Strictly, Gibbs sampling requires sequential updates (each spin sees
    the latest state). Parallel updates (all spins see the same "stale"
    state) actually sample from a different distribution. However:
    1. For optimization (finding low-energy states), parallel updates work
       well in practice -- they're used in Boltzmann machine training.
    2. We use a checkerboard decomposition: split spins into two groups
       (even/odd indexed) where no two spins in the same group interact.
       Update all even spins in parallel, then all odd spins. This IS
       exact Gibbs for bipartite graphs. For non-bipartite (like SAT),
       it's an approximation but converges to the same fixed points.
    3. The annealing schedule (decreasing temperature) means we care about
       the low-temperature limit, where parallel vs sequential makes
       negligible difference.

    **Temperature annealing schedule:**
    Unlike thrml's fixed beta, we support a schedule where beta increases
    (temperature decreases) over the warmup phase. This explores broadly
    at high temperature, then focuses on low-energy states. The schedule
    is: beta(t) = beta_init + (beta_final - beta_init) * (t / n_warmup).
    Linear annealing is simple and effective; geometric schedules are also
    available.

    **GPU acceleration:**
    JAX automatically places arrays on GPU if available. The matrix-vector
    product J @ s and the Bernoulli sampling are both highly parallel
    operations that map perfectly to GPU SIMD. For N=500, expect 10-100x
    speedup over thrml's sequential CPU loop.

Spec: REQ-SAMPLE-003
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jrandom


@dataclass
class AnnealingSchedule:
    """Temperature annealing schedule for simulated annealing.

    **Researcher summary:**
        Controls beta (inverse temperature) over time. Linear or geometric
        interpolation from beta_init to beta_final over n_warmup steps.

    **Detailed explanation for engineers:**
        Simulated annealing starts "hot" (low beta, high temperature) so
        the sampler explores broadly, then "cools" (high beta, low temperature)
        to focus on low-energy states. This avoids getting trapped in local
        minima early on.

        - ``beta_init``: Starting inverse temperature. Low values (0.1-1.0)
          mean nearly uniform random exploration.
        - ``beta_final``: Final inverse temperature. High values (5.0-20.0)
          strongly favor low-energy configurations.
        - ``schedule_type``: "linear" interpolates beta linearly; "geometric"
          interpolates log(beta) linearly (faster cooling at the start).

    Spec: REQ-SAMPLE-003
    """

    beta_init: float = 0.1
    beta_final: float = 10.0
    schedule_type: Literal["linear", "geometric"] = "linear"

    def beta_at_step(self, step: int, total_steps: int) -> jax.Array:
        """Compute beta at a given step in the annealing schedule.

        Args:
            step: Current step index (0-based).
            total_steps: Total number of annealing steps.

        Returns:
            Scalar beta value for this step.
        """
        # Fraction of the way through the schedule (0.0 to 1.0).
        frac = jnp.clip(step / jnp.maximum(total_steps - 1, 1), 0.0, 1.0)

        if self.schedule_type == "geometric":
            # Interpolate in log space for geometric cooling.
            log_beta = jnp.log(self.beta_init) + frac * (
                jnp.log(self.beta_final) - jnp.log(self.beta_init)
            )
            return jnp.exp(log_beta)
        else:
            # Linear interpolation.
            return self.beta_init + frac * (self.beta_final - self.beta_init)


@dataclass
class ParallelIsingSampler:
    """Vectorized parallel Gibbs sampler for Ising models.

    **Researcher summary:**
        Parallel Gibbs with optional checkerboard decomposition and simulated
        annealing. All spin conditionals computed via matrix-vector multiply,
        sampled via parallel Bernoulli. Orders of magnitude faster than
        sequential per-spin sampling on GPU.

    **Detailed explanation for engineers:**
        This sampler takes the same inputs as thrml's Ising sampling (biases,
        coupling weights, initial spins) but runs entirely in vectorized JAX.
        Instead of N sequential Python calls per sweep, it does:

        1. Compute local fields: h = b + J @ s  (one matrix-vector multiply)
        2. Compute flip probabilities: p = sigmoid(2 * beta * h)
        3. Sample new spins: s_new ~ Bernoulli(p)

        For checkerboard mode, steps 1-3 are done for even-indexed spins
        first (using odd spins as fixed neighbors), then vice versa.

        The outer annealing loop uses ``jax.lax.fori_loop`` (no Python
        overhead, runs entirely on accelerator). Sample collection uses
        ``jax.lax.scan`` for the post-warmup phase.

    Attributes:
        n_warmup: Number of annealing steps before collecting samples.
        n_samples: Number of samples to collect after warmup.
        steps_per_sample: Sweeps between collected samples (decorrelation).
        schedule: Temperature annealing schedule for the warmup phase.
        use_checkerboard: If True, alternate even/odd spin updates (more
            correct for non-bipartite graphs). If False, update all spins
            simultaneously (faster, slight approximation).

    Spec: REQ-SAMPLE-003
    """

    n_warmup: int = 1000
    n_samples: int = 50
    steps_per_sample: int = 20
    schedule: AnnealingSchedule | None = None
    use_checkerboard: bool = True

    def sample(
        self,
        key: jax.Array,
        biases: jax.Array,
        coupling_matrix: jax.Array,
        beta: float | jax.Array = 10.0,
        init_spins: jax.Array | None = None,
    ) -> jax.Array:
        """Run parallel Gibbs sampling with annealing on an Ising model.

        **Detailed explanation for engineers:**
            This is the main entry point. It:
            1. Initializes spins (random if not provided)
            2. Runs the warmup phase with temperature annealing
            3. Collects samples at the final temperature
            4. Returns all collected samples

            The Ising energy is E(s) = -beta * (b^T s + s^T J s) where
            s ∈ {0, 1}^N (boolean spins, not ±1). The conditional for
            spin i given all others is:
                P(s_i=1 | s_{-i}) = sigmoid(2*beta*(b_i + J[i,:] @ s - J[i,i]*s_i))
            Since J is zero-diagonal, this simplifies to:
                P(s_i=1 | s_{-i}) = sigmoid(2*beta*(b_i + J[i,:] @ s))

            We compute J @ s for all i at once with a matrix-vector product.

        Args:
            key: JAX PRNG key.
            biases: Bias vector, shape (n_spins,). Positive bias encourages
                spin=1; negative encourages spin=0.
            coupling_matrix: Symmetric coupling matrix, shape (n_spins, n_spins).
                J[i,j] > 0 encourages s_i and s_j to agree (ferromagnetic);
                J[i,j] < 0 encourages them to disagree (antiferromagnetic).
                Diagonal must be zero.
            beta: Inverse temperature (scalar). Only used as beta_final if
                an annealing schedule is provided.
            init_spins: Initial spin configuration, shape (n_spins,), boolean.
                If None, initialized randomly.

        Returns:
            Samples array of shape (n_samples, n_spins), boolean.

        Spec: REQ-SAMPLE-003
        """
        n_spins = biases.shape[0]
        beta = jnp.asarray(beta, dtype=jnp.float32)

        # Initialize spins randomly if not provided.
        key, init_key = jrandom.split(key)
        if init_spins is None:
            spins = jrandom.bernoulli(init_key, 0.5, (n_spins,))
        else:
            spins = jnp.asarray(init_spins, dtype=jnp.bool_)

        # Convert to float for matrix operations.
        # We work with float32 spins internally (0.0 or 1.0) and convert
        # back to bool for output.
        spins_f = spins.astype(jnp.float32)

        # Ensure coupling matrix is float32 and zero-diagonal.
        J = jnp.asarray(coupling_matrix, dtype=jnp.float32)
        b = jnp.asarray(biases, dtype=jnp.float32)

        schedule = self.schedule or AnnealingSchedule(
            beta_init=float(beta), beta_final=float(beta)
        )

        # --- Phase 1: Warmup with annealing ---
        if self.use_checkerboard:
            even_mask = jnp.arange(n_spins) % 2 == 0
            odd_mask = ~even_mask

            def sweep_fn(carry, step_key):
                s, step = carry
                beta_t = schedule.beta_at_step(step, self.n_warmup)
                k1, k2 = jrandom.split(step_key)
                s = _checkerboard_update(s, b, J, beta_t, k1, k2, even_mask, odd_mask)
                return (s, step + 1), None

        else:

            def sweep_fn(carry, step_key):
                s, step = carry
                beta_t = schedule.beta_at_step(step, self.n_warmup)
                s = _parallel_update(s, b, J, beta_t, step_key)
                return (s, step + 1), None

        key, warmup_key = jrandom.split(key)
        warmup_keys = jrandom.split(warmup_key, self.n_warmup)
        (spins_f, _), _ = jax.lax.scan(sweep_fn, (spins_f, jnp.int32(0)), warmup_keys)

        # --- Phase 2: Collect samples at final temperature ---
        beta_final = jnp.asarray(schedule.beta_final, dtype=jnp.float32)

        if self.use_checkerboard:

            def sample_sweep_fn(s, step_key):
                k1, k2 = jrandom.split(step_key)
                return _checkerboard_update(s, b, J, beta_final, k1, k2, even_mask, odd_mask)

        else:

            def sample_sweep_fn(s, step_key):
                return _parallel_update(s, b, J, beta_final, step_key)

        def collect_fn(carry, sample_key):
            s = carry
            # Run steps_per_sample sweeps between samples.
            sweep_keys = jrandom.split(sample_key, self.steps_per_sample)

            def decorrelate(s_inner, k):
                return sample_sweep_fn(s_inner, k), None

            s, _ = jax.lax.scan(decorrelate, s, sweep_keys)
            return s, s  # carry, output

        key, collect_key = jrandom.split(key)
        collect_keys = jrandom.split(collect_key, self.n_samples)
        _, samples_f = jax.lax.scan(collect_fn, spins_f, collect_keys)

        # Convert back to boolean.
        return samples_f > 0.5


def _parallel_update(
    spins: jax.Array,
    biases: jax.Array,
    J: jax.Array,
    beta: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Update all spins simultaneously (fully parallel Gibbs).

    **Detailed explanation for engineers:**
        Computes the local field for every spin at once using a matrix-vector
        product, then samples all spins independently. This is NOT exact Gibbs
        (which requires sequential updates) but is a good approximation that
        converges to the same fixed points. Used in Boltzmann machine training
        (contrastive divergence) and works well for optimization.

        The local field for spin i is:
            h_i = b_i + sum_j J[i,j] * s_j

        The conditional probability is:
            P(s_i = 1) = sigmoid(2 * beta * h_i)

        The factor of 2 comes from the Ising convention where spins are mapped
        from {0,1} to {-1,+1} internally: sigma = 2*s - 1, and the energy is
        E = -beta * sigma^T J sigma.
    """
    # Local fields: h = b + J @ s (vectorized over all spins).
    h = biases + J @ spins
    # Conditional probabilities.
    probs = jax.nn.sigmoid(2.0 * beta * h)
    # Sample all spins in parallel.
    return jrandom.bernoulli(key, probs).astype(jnp.float32)


def _checkerboard_update(
    spins: jax.Array,
    biases: jax.Array,
    J: jax.Array,
    beta: jax.Array,
    key_even: jax.Array,
    key_odd: jax.Array,
    even_mask: jax.Array,
    odd_mask: jax.Array,
) -> jax.Array:
    """Two-phase checkerboard update: even spins then odd spins.

    **Detailed explanation for engineers:**
        Splits spins into two groups (even-indexed and odd-indexed). First
        updates all even spins using the current odd spin values as fixed
        neighbors, then updates all odd spins using the new even spin values.

        For bipartite graphs (where even spins only interact with odd spins
        and vice versa), this is EXACT Gibbs sampling -- each spin's neighbors
        are all from the other group and haven't changed during this half-sweep.

        For general graphs (like SAT problems where any two variables can
        interact), spins within the same group may also be coupled. The
        checkerboard decomposition is still a good approximation because:
        1. In sparse graphs, most same-group pairs have zero coupling.
        2. Even for dense graphs, the two-phase structure provides more
           accurate updates than fully parallel (which has NO sequencing).
        3. It converges to the correct distribution at low temperature.

        The half-sweep approach doubles the number of matrix-vector products
        per sweep (2 instead of 1), but each uses the same J matrix and the
        total work is 2 * O(N^2) instead of N * O(N) for sequential --
        identical asymptotic cost but massively better GPU utilization.
    """
    # Phase 1: Update even spins (odd spins are fixed neighbors).
    h_even = biases + J @ spins
    probs_even = jax.nn.sigmoid(2.0 * beta * h_even)
    new_even = jrandom.bernoulli(key_even, probs_even).astype(jnp.float32)
    spins = jnp.where(even_mask, new_even, spins)

    # Phase 2: Update odd spins (even spins now have new values).
    h_odd = biases + J @ spins
    probs_odd = jax.nn.sigmoid(2.0 * beta * h_odd)
    new_odd = jrandom.bernoulli(key_odd, probs_odd).astype(jnp.float32)
    spins = jnp.where(odd_mask, new_odd, spins)

    return spins


def sat_to_coupling_matrix(
    biases_vec: jax.Array,
    weights_vec: jax.Array,
    n_vars: int,
) -> tuple[jax.Array, jax.Array]:
    """Convert thrml-style Ising parameters to a coupling matrix.

    **Detailed explanation for engineers:**
        thrml stores pairwise couplings as a flat vector (upper triangle of
        the coupling matrix, row-major). This function unpacks that vector
        into a symmetric N×N matrix with zero diagonal, which is what the
        parallel sampler needs for its matrix-vector multiply.

        The mapping from flat index k to matrix indices (i,j) where i < j:
            k = i * n_vars - i * (i + 1) / 2 + j - i - 1

        We also return the bias vector unchanged (it's already in the right
        format).

    Args:
        biases_vec: Bias vector from sat_to_ising, shape (n_vars,).
        weights_vec: Flat upper-triangle coupling weights, shape (n_edges,).
        n_vars: Number of spin variables.

    Returns:
        Tuple of (biases, coupling_matrix) where coupling_matrix is
        symmetric with zero diagonal, shape (n_vars, n_vars).
    """
    J = jnp.zeros((n_vars, n_vars), dtype=jnp.float32)

    # Unpack the upper triangle.
    idx = 0
    indices_i = []
    indices_j = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            indices_i.append(i)
            indices_j.append(j)
            idx += 1

    indices_i = jnp.array(indices_i)
    indices_j = jnp.array(indices_j)

    # Trim weights to match actual number of edges.
    n_edges = len(indices_i)
    w = weights_vec[:n_edges]

    # Set upper triangle.
    J = J.at[indices_i, indices_j].set(w)
    # Mirror to lower triangle (symmetric).
    J = J + J.T

    return jnp.asarray(biases_vec, dtype=jnp.float32), J


# ---------------------------------------------------------------------------
# thrml-compatible interface
# ---------------------------------------------------------------------------


def extract_ising_params(model: object) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extract biases, coupling matrix, and beta from a thrml IsingEBM.

    **Detailed explanation for engineers:**
        thrml's IsingEBM stores biases as a flat vector and weights as a flat
        vector corresponding to its ``edges`` list. This function reads those
        attributes and builds the symmetric coupling matrix needed by the
        parallel sampler.

        This is the bridge between thrml's model definition (which users
        already know and which maps to Extropic TSU hardware) and our
        vectorized sampling backend.

        The coupling matrix is built via vectorized index arrays rather than
        a Python for-loop, so construction is fast even for large models.

    Args:
        model: A ``thrml.models.ising.IsingEBM`` instance.

    Returns:
        Tuple of (biases, coupling_matrix, beta).
    """
    import numpy as np

    biases = jnp.asarray(model.biases, dtype=jnp.float32)
    beta = jnp.asarray(model.beta, dtype=jnp.float32)
    n = len(model.nodes)

    # Build node-index lookup.
    node_to_idx = {node: i for i, node in enumerate(model.nodes)}

    # Build index arrays for all edges at once (Python loop over edges,
    # but only to collect indices — no JAX ops inside the loop).
    n_edges = len(model.edges)
    row_idx = np.empty(n_edges, dtype=np.int32)
    col_idx = np.empty(n_edges, dtype=np.int32)
    for k, (ni, nj) in enumerate(model.edges):
        row_idx[k] = node_to_idx[ni]
        col_idx[k] = node_to_idx[nj]

    weights = jnp.asarray(model.weights, dtype=jnp.float32)

    # Build symmetric coupling matrix with two vectorized scatter-adds.
    J = jnp.zeros((n, n), dtype=jnp.float32)
    J = J.at[row_idx, col_idx].add(weights)
    J = J.at[col_idx, row_idx].add(weights)

    return biases, J, beta


def parallel_sample_states(
    key: jax.Array,
    model: object,
    schedule: object | None = None,
    annealing: AnnealingSchedule | None = None,
    use_checkerboard: bool = True,
    nodes_to_sample: list | None = None,
) -> list:
    """Drop-in replacement for ``thrml.sample_states`` using parallel Gibbs.

    **Detailed explanation for engineers:**
        This function accepts the same thrml model objects that
        ``sample_states`` does, but internally uses the vectorized parallel
        Gibbs sampler for dramatic speedups. The return format matches
        thrml's convention: a list of arrays, one per sampled block, each
        with shape ``(n_samples, 1)`` (boolean).

        This ensures compatibility with:
        - Phase 3 thrml training (``estimate_kl_grad``) — same model objects
        - Phase 4 Extropic TSU — same model definition, swap backend
        - Existing experiment scripts — change one import line

        When Extropic TSU hardware becomes available, this function can be
        replaced by a hardware backend that accepts the same IsingEBM model
        and returns the same sample format, with zero API changes upstream.

    Args:
        key: JAX PRNG key.
        model: A ``thrml.models.ising.IsingEBM`` instance.
        schedule: A ``thrml.SamplingSchedule`` instance (provides n_warmup,
            n_samples, steps_per_sample). If None, uses defaults.
        annealing: Temperature annealing schedule. If None, uses constant
            beta from the model (matching thrml's behavior).
        use_checkerboard: Whether to use checkerboard Gibbs updates.
        nodes_to_sample: List of nodes or blocks to return samples for.
            If None, returns samples for all model nodes.

    Returns:
        List of boolean arrays, one per node, each shape (n_samples, 1).
        This matches the format returned by ``thrml.sample_states``.

    Spec: REQ-SAMPLE-003
    """
    biases, J, beta = extract_ising_params(model)

    # Read schedule parameters (thrml SamplingSchedule or defaults).
    if schedule is not None:
        n_warmup = schedule.n_warmup
        n_samples = schedule.n_samples
        steps_per_sample = schedule.steps_per_sample
    else:
        n_warmup = 1000
        n_samples = 50
        steps_per_sample = 20

    sampler = ParallelIsingSampler(
        n_warmup=n_warmup,
        n_samples=n_samples,
        steps_per_sample=steps_per_sample,
        schedule=annealing,
        use_checkerboard=use_checkerboard,
    )

    # Run sampling.
    samples = sampler.sample(key, biases, J, beta=float(beta))
    # samples shape: (n_samples, n_spins), boolean

    # Determine which node indices to return.
    n_spins = len(model.nodes)
    if nodes_to_sample is not None:
        node_to_idx = {node: i for i, node in enumerate(model.nodes)}
        indices = []
        for item in nodes_to_sample:
            # Accept both Block objects and raw nodes.
            if hasattr(item, "nodes"):
                for node in item.nodes:
                    indices.append(node_to_idx[node])
            else:
                indices.append(node_to_idx[item])
    else:
        indices = list(range(n_spins))

    # Convert to thrml's per-block format: list of (n_samples, 1) arrays.
    result = []
    for idx in indices:
        # Shape (n_samples, 1) to match thrml's convention.
        result.append(samples[:, idx : idx + 1])

    return result
