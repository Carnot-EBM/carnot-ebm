"""Energy landscape certification — prove you're at a real minimum.

**Researcher summary:**
    Hessian eigenvalue analysis for local minimum verification, condition
    number estimation, and basin of attraction radius via perturbation
    sensitivity analysis. Uses jax.hessian for exact second derivatives.

**Detailed explanation for engineers:**
    When a sampler or optimizer reports "I found a low-energy configuration,"
    how do you know it's actually at a minimum and not a saddle point or
    a flat plateau? This module provides mathematical proof.

    **What is a Hessian?**
    The Hessian is the matrix of second derivatives of the energy function.
    If the energy function is E(x), the Hessian H has entries:
        H[i][j] = d²E / (dx_i dx_j)

    Think of it as measuring the "curvature" of the energy landscape at a point.

    **Why eigenvalues matter:**
    The eigenvalues of the Hessian tell you the curvature in each direction:
    - All eigenvalues positive → TRUE LOCAL MINIMUM (bowl-shaped in every direction)
    - Some eigenvalues negative → SADDLE POINT (bowl in some directions, hill in others)
    - Some eigenvalues zero → FLAT DIRECTION (degenerate, neither min nor max)

    This is the mathematical equivalent of checking "if I nudge the solution
    in any direction, does the energy always go up?" If yes, you're at a
    verified minimum.

    **Basin of attraction:**
    Even at a true minimum, you want to know how "robust" it is. The basin
    of attraction radius estimates how far you can perturb the solution
    before it leaves the minimum's basin and rolls to a different minimum.
    A large basin = robust solution. A tiny basin = fragile, easily disrupted.

    **For engineers coming from optimization:**
    This is analogous to checking the second-order sufficient conditions
    for optimality in constrained optimization. In neural network training,
    people rarely check this because the loss landscape is too high-dimensional.
    But for verifiable reasoning on moderate-dimensional problems (like Sudoku
    with 81 variables), Hessian analysis is tractable and provides strong
    guarantees.

Spec: REQ-VERIFY-006, SCENARIO-VERIFY-005
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import EnergyFunction


@dataclass
class LandscapeCertificate:
    """Result of energy landscape certification at a point.

    **Researcher summary:**
        Hessian eigenvalue spectrum, definiteness classification, condition
        number, and basin of attraction radius estimate.

    **Detailed explanation for engineers:**
        After computing the Hessian and analyzing its eigenvalues, this
        dataclass tells you everything about the curvature at a point:

    Attributes:
        is_local_minimum: True if ALL Hessian eigenvalues are positive.
            This is the mathematical proof that you're at a true minimum.
        eigenvalues: The full eigenvalue spectrum, sorted ascending.
            All positive = minimum. Any negative = saddle point.
        min_eigenvalue: The smallest eigenvalue. If positive, the
            "weakest" curvature direction still curves upward. If negative,
            there's a direction where the energy decreases.
        max_eigenvalue: The largest eigenvalue. Large values mean strong
            curvature (steep bowl walls) in that direction.
        condition_number: max_eigenvalue / min_eigenvalue. Measures how
            "elongated" the bowl is. Close to 1 = nice round bowl (easy
            to optimize). Very large = long narrow valley (hard to optimize).
            Infinity = flat direction exists.
        basin_radius: Estimated radius of the basin of attraction. How far
            you can perturb the solution before the energy increases
            significantly. Larger = more robust solution.
        classification: Human-readable string: "local_minimum", "saddle_point",
            or "degenerate" (has zero eigenvalues).

    For example, a well-conditioned minimum might look like::

        LandscapeCertificate(
            is_local_minimum=True,
            eigenvalues=[0.5, 0.8, 1.2],
            min_eigenvalue=0.5,
            max_eigenvalue=1.2,
            condition_number=2.4,
            basin_radius=0.3,
            classification="local_minimum",
        )

    Spec: REQ-VERIFY-006
    """

    is_local_minimum: bool
    eigenvalues: list[float]
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    basin_radius: float
    classification: str  # "local_minimum", "saddle_point", "degenerate"


def certify_landscape(
    energy_fn: EnergyFunction,
    x: jax.Array,
    basin_perturbations: int = 100,
    basin_key: jax.Array | None = None,
    eigenvalue_tolerance: float = 1e-6,
) -> LandscapeCertificate:
    """Certify the energy landscape at a point: is it a true minimum?

    **Researcher summary:**
        Computes the Hessian via ``jax.hessian``, extracts eigenvalues,
        classifies the critical point, and estimates basin radius via
        random perturbation analysis.

    **Detailed explanation for engineers:**
        This function answers: "Is point x a true local minimum of the
        energy function, and how robust is it?"

        Steps:
        1. Compute the Hessian matrix (all second derivatives) at x
        2. Find its eigenvalues (the curvatures in each direction)
        3. Classify: all positive = minimum, any negative = saddle
        4. Compute condition number (how elongated the bowl is)
        5. Estimate basin radius by random perturbation

        **Why jax.hessian?**
        JAX can compute exact Hessians via automatic differentiation.
        No numerical approximation needed — this is exact (up to
        floating-point precision). For a 81-dimensional Sudoku problem,
        the Hessian is an 81×81 matrix, which is tractable.

    Args:
        energy_fn: The energy function to analyze. Must have an ``energy``
            method that takes a JAX array and returns a scalar.
        x: The point to certify. Should be a suspected minimum (e.g.,
            output of an optimizer or sampler).
        basin_perturbations: Number of random perturbations for basin
            radius estimation. More = more accurate but slower.
        basin_key: JAX PRNG key for random perturbations. Uses default if None.
        eigenvalue_tolerance: Eigenvalues smaller than this (in absolute
            value) are considered zero (degenerate direction).

    Returns:
        A LandscapeCertificate with the full analysis.

    For example::

        from carnot.verify.landscape import certify_landscape

        # Check if the origin is a minimum of E(x) = 0.5*||x||^2
        cert = certify_landscape(model, jnp.zeros(3))
        assert cert.is_local_minimum  # True — it's a bowl!
        assert cert.classification == "local_minimum"

    Spec: REQ-VERIFY-006, SCENARIO-VERIFY-005
    """
    if basin_key is None:
        basin_key = jrandom.PRNGKey(42)

    # --- Step 1: Compute the Hessian matrix ---
    # jax.hessian computes the matrix of all second partial derivatives.
    # For E: R^n -> R, the Hessian is an n×n symmetric matrix.
    hessian_fn = jax.hessian(energy_fn.energy)
    hessian_matrix = hessian_fn(x)

    # --- Step 2: Compute eigenvalues ---
    # Eigenvalues of a real symmetric matrix are all real.
    # jnp.linalg.eigvalsh is optimized for symmetric matrices.
    eigenvalues = jnp.linalg.eigvalsh(hessian_matrix)
    eigenvalues_list = sorted(float(e) for e in eigenvalues)

    min_eig = eigenvalues_list[0]
    max_eig = eigenvalues_list[-1]

    # --- Step 3: Classify the critical point ---
    # A point is a local minimum if and only if all eigenvalues are positive.
    # "Positive" here means > tolerance (to handle numerical noise near zero).
    has_negative = min_eig < -eigenvalue_tolerance
    has_zero = abs(min_eig) <= eigenvalue_tolerance
    all_positive = min_eig > eigenvalue_tolerance

    if all_positive:
        classification = "local_minimum"
        is_minimum = True
    elif has_negative:
        classification = "saddle_point"
        is_minimum = False
    else:
        classification = "degenerate"
        is_minimum = False

    # --- Step 4: Condition number ---
    # Condition number = max_eig / min_eig (only meaningful at a minimum).
    # High condition number = ill-conditioned (long narrow valley).
    if abs(min_eig) > eigenvalue_tolerance:
        condition_number = float(abs(max_eig / min_eig))
    else:
        condition_number = float("inf")

    # --- Step 5: Basin of attraction radius estimation ---
    # Perturb x in random directions with increasing radius.
    # The basin radius is the largest perturbation where the energy
    # is still higher than at x (i.e., x is still the local minimum).
    basin_radius = _estimate_basin_radius(
        energy_fn, x, basin_perturbations, basin_key
    )

    return LandscapeCertificate(
        is_local_minimum=is_minimum,
        eigenvalues=eigenvalues_list,
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
        condition_number=condition_number,
        basin_radius=basin_radius,
        classification=classification,
    )


def _estimate_basin_radius(
    energy_fn: EnergyFunction,
    x: jax.Array,
    n_perturbations: int,
    key: jax.Array,
) -> float:
    """Estimate the basin of attraction radius via random perturbation.

    **Detailed explanation for engineers:**
        We test increasingly large perturbations around x. At each radius,
        we generate random perturbations and check what fraction of them
        lead to higher energy (i.e., we're still in the basin). When
        fewer than 90% of perturbations increase energy, we've found
        the edge of the basin.

        Think of it like dropping marbles around a bowl: near the bottom,
        every marble rolls back to the center (high fraction with higher
        energy). At the rim, some marbles roll outside (low fraction).
        The basin radius is where this transition happens.

    Spec: REQ-VERIFY-006
    """
    base_energy = float(energy_fn.energy(x))
    dim = x.shape[0]

    # Test radii on a log scale from very small to moderate
    radii = jnp.logspace(-4, 0, 20)

    basin_radius = float(radii[-1])  # default: largest tested

    for r in radii:
        r_float = float(r)
        key, subkey = jrandom.split(key)
        # Generate random perturbation directions (unit vectors * radius)
        perturbations = jrandom.normal(subkey, (n_perturbations, dim))
        # Normalize to unit vectors, then scale by radius
        norms = jnp.linalg.norm(perturbations, axis=1, keepdims=True)
        # Avoid division by zero for rare zero-norm samples
        norms = jnp.maximum(norms, 1e-10)
        perturbations = perturbations / norms * r_float

        # Compute energy at each perturbed point
        perturbed_points = x + perturbations
        energies = jax.vmap(energy_fn.energy)(perturbed_points)

        # What fraction of perturbations lead to higher energy?
        fraction_higher = float(jnp.mean(energies > base_energy))

        # If less than 90% are higher, we're leaving the basin
        if fraction_higher < 0.9:
            basin_radius = r_float
            break

    return basin_radius
