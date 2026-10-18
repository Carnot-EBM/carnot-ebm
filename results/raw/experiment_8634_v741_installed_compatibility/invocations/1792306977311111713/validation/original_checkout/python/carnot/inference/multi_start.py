"""Multi-Start Self-Verification for Energy Based Models.

**Researcher summary:**
    Multi-start repair runs gradient-based constraint repair from N randomly
    perturbed starting points and returns the result with lowest final energy.
    This is Algorithm 2 from the EBT paper: it mitigates sensitivity to
    initialization by exploring multiple basins of attraction in parallel.

**Detailed explanation for engineers:**
    Single-start gradient descent repair (``carnot.verify.constraint.repair``)
    is sensitive to the starting configuration. If the energy landscape has
    multiple local minima (which is common for combinatorial constraint
    problems like SAT or scheduling), a single run might get stuck in a
    high-energy local minimum instead of finding the global one.

    **Multi-start repair solves this by:**

    1. Taking the original assignment as a starting point.
    2. Generating N perturbed copies: ``x_i = assignment + noise_i * scale``.
    3. Running ``repair()`` independently on each perturbed copy.
    4. Optionally rounding results to discrete values (e.g., for integer
       constraint problems like Sudoku or SAT).
    5. Selecting the result with the lowest final energy.

    The perturbation scale controls how far each start deviates from the
    original. A larger scale explores more of the landscape but may land in
    worse basins. A smaller scale focuses near the original but may not
    escape a bad local minimum.

    **When to use multi-start:**
    - Combinatorial problems (SAT, scheduling, Sudoku)
    - When single repair frequently fails to reach energy = 0
    - When you have compute budget for N parallel repair runs

    **JAX PRNG keys:**
    Multi-start needs N independent random perturbations. We use
    ``jax.random.split(key, n_starts)`` to generate N independent sub-keys
    from a single parent key. This is JAX's way of forking randomness:
    each sub-key produces statistically independent random numbers.

Spec: REQ-INFER-009
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from carnot.verify.constraint import ComposedEnergy, VerificationResult, repair

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MultiStartResult:
    """Result of multi-start repair: the best outcome from N independent runs.

    **Researcher summary:**
        Aggregates N repair runs. Stores the best (lowest energy) repaired
        configuration, all final energies for analysis, and the index of
        the winning start.

    **Detailed explanation for engineers:**
        After running repair from N different perturbed starting points,
        this dataclass captures:

        - ``best_x``: The repaired configuration with the lowest final
          energy. This is your best candidate solution.
        - ``best_energy``: The energy of ``best_x``. Zero means all
          constraints are satisfied.
        - ``best_history``: The full verification history from the winning
          repair run, useful for analyzing convergence.
        - ``all_final_energies``: Final energy from each of the N runs.
          Useful for understanding the landscape: if all energies are
          similar, the landscape has one dominant basin; if they vary
          widely, there are multiple basins.
        - ``best_start_index``: Which of the N starts produced the best
          result (0-indexed).
        - ``n_starts``: Total number of starts attempted.

    Spec: REQ-INFER-009
    """

    best_x: jax.Array
    best_energy: float
    best_history: list[VerificationResult]
    all_final_energies: list[float] = field(default_factory=list)
    best_start_index: int = 0
    n_starts: int = 0


def multi_start_repair(
    assignment: jax.Array,
    energy: ComposedEnergy,
    n_starts: int = 5,
    perturbation_scale: float = 0.1,
    step_size: float = 0.1,
    max_repair_steps: int = 200,
    round_fn: Callable[[jax.Array], jax.Array] | None = None,
    key: jax.Array | None = None,
) -> MultiStartResult:
    """Run repair from N random perturbations, return best result.

    **Researcher summary:**
        Algorithm 2 from the EBT paper:
        1. Generate N perturbations: x_i = assignment + noise_i * perturbation_scale
        2. Run repair() on each x_i independently
        3. Optionally round each to discrete values
        4. Select minimum energy result

    **Detailed explanation for engineers:**
        This function orchestrates multiple independent repair attempts to
        find the lowest-energy solution. Each attempt starts from a slightly
        different point in the energy landscape, created by adding Gaussian
        noise to the original assignment.

        **Step by step:**

        1. **Generate PRNG sub-keys**: Split the parent key into N sub-keys
           using ``jax.random.split()``. Each sub-key produces independent
           random noise.

        2. **Create perturbed starts**: For each sub-key, sample Gaussian
           noise with standard deviation = ``perturbation_scale`` and add it
           to the original assignment. This creates N starting points that
           are close to (but different from) the original.

        3. **Run repair on each**: Call ``repair()`` from
           ``carnot.verify.constraint`` on each perturbed start. This runs
           gradient descent on violated constraints for up to
           ``max_repair_steps`` iterations.

        4. **Optional rounding**: If ``round_fn`` is provided, apply it to
           each repaired result. This is critical for discrete problems
           (e.g., rounding continuous values to integers for Sudoku/SAT).

        5. **Select best**: Evaluate the final energy of each (possibly
           rounded) result and return the one with the lowest energy.

        **Default PRNG key**: If no key is provided, we use
        ``jax.random.PRNGKey(0)`` as a deterministic default. This makes
        the function reproducible by default, but you should pass your own
        key in production to get different results across calls.

    Args:
        assignment: Initial configuration vector, shape (input_dim,).
            This is the starting point before perturbation.
        energy: ComposedEnergy containing all constraints to satisfy.
        n_starts: Number of independent repair attempts. More starts =
            better chance of finding the global minimum, but more compute.
            Default: 5.
        perturbation_scale: Standard deviation of Gaussian noise added to
            create each perturbed start. Larger values explore more of the
            landscape. Default: 0.1.
        step_size: Gradient descent step size for each repair run.
            Passed directly to ``repair()``. Default: 0.1.
        max_repair_steps: Maximum iterations per repair run.
            Passed directly to ``repair()``. Default: 200.
        round_fn: Optional function to round repaired results to discrete
            values. For example, ``jnp.round`` for integer problems. If
            None, results are left as continuous values.
        key: JAX PRNG key for generating perturbations. If None, uses
            ``jax.random.PRNGKey(0)`` for reproducibility.

    Returns:
        MultiStartResult containing the best repaired configuration and
        metadata about all N runs.

    For example::

        from carnot.verify.constraint import ComposedEnergy, repair
        from carnot.inference.multi_start import multi_start_repair

        composed = build_constraints(problem)
        x0 = jnp.zeros(dim)

        result = multi_start_repair(
            x0, composed, n_starts=10, perturbation_scale=0.5
        )
        if result.best_energy < 1e-6:
            print("Found a valid solution!")

    Spec: REQ-INFER-009
    """
    # Use a deterministic default key if none provided, ensuring
    # reproducibility (REQ-VERIFY-007 pattern).
    if key is None:
        key = jax.random.PRNGKey(0)

    # Split the parent key into n_starts independent sub-keys.
    # jax.random.split(key, N) returns an (N, 2) array where each row
    # is an independent PRNG key.
    sub_keys = jax.random.split(key, n_starts)

    # Storage for all repair outcomes
    all_results: list[tuple[jax.Array, float, list[VerificationResult]]] = []

    for i in range(n_starts):
        # Generate Gaussian noise for this start
        noise = jax.random.normal(sub_keys[i], shape=assignment.shape)
        perturbed = assignment + noise * perturbation_scale

        # Run gradient-based repair from this perturbed starting point
        repaired_x, history = repair(energy, perturbed, step_size, max_repair_steps)

        # Optionally round to discrete values (e.g., for SAT or Sudoku)
        if round_fn is not None:
            repaired_x = round_fn(repaired_x)

        # Compute final energy of the (possibly rounded) result
        final_energy = float(energy.energy(repaired_x))
        all_results.append((repaired_x, final_energy, history))

    # Select the result with the lowest final energy
    all_final_energies = [e for _, e, _ in all_results]
    best_idx = int(jnp.argmin(jnp.array(all_final_energies)))
    best_x, best_energy, best_history = all_results[best_idx]

    return MultiStartResult(
        best_x=best_x,
        best_energy=best_energy,
        best_history=best_history,
        all_final_energies=all_final_energies,
        best_start_index=best_idx,
        n_starts=n_starts,
    )
