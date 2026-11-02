"""Constraint-based verifiable reasoning -- JAX implementation.

**Researcher summary:**
    Encodes discrete constraints as differentiable energy terms. Constraint
    satisfaction = zero energy. Composed energy aggregates weighted terms.
    Gradient-based repair descends on violated constraints to fix invalid
    configurations. Enables verifiable reasoning where correctness is
    machine-checkable.

**Detailed explanation for engineers:**
    This module is the heart of Carnot's "verifiable reasoning" capability.
    The key insight is: many logical constraints (like Sudoku rules, type
    checking, or graph coloring) can be expressed as energy functions where
    E(x) = 0 means "constraint satisfied" and E(x) > 0 means "violated."

    **Why encode constraints as energy?**
    Once constraints are differentiable energy terms, we can:
    1. **Compose** them: just add the energies (with weights)
    2. **Sample** valid configurations using Langevin/HMC samplers
    3. **Repair** invalid configurations by gradient descent on violated terms
    4. **Verify** solutions by checking if all energies are below threshold

    This is fundamentally different from SAT solvers or constraint propagation:
    instead of discrete search, we use continuous optimization in a
    differentiable energy landscape. JAX's autodiff computes all gradients
    automatically.

    **Architecture:**
    - ``ConstraintTerm`` (Protocol): Interface for a single constraint.
    - ``BaseConstraint``: Base class with default threshold and auto-grad.
    - ``ComposedEnergy``: Aggregates multiple weighted constraints into one
      energy function. Supports decomposition, verification, and selective
      gradient computation.
    - ``repair()``: Iterative gradient descent on only the violated constraints.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004, REQ-VERIFY-005
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class ConstraintTerm(Protocol):
    """A single verifiable constraint, expressed as a differentiable energy term.

    **Researcher summary:**
        Protocol for constraint energy: satisfied iff energy(x) <= threshold.
        Must provide name, energy, gradient, threshold, and satisfaction check.

    **Detailed explanation for engineers:**
        Each constraint maps a configuration x to a non-negative energy:
        - E(x) = 0: the constraint is perfectly satisfied
        - E(x) > 0: the constraint is violated, with larger values meaning
          worse violations

        The gradient dE/dx tells us how to adjust x to reduce the violation.
        This is what makes constraint-based reasoning work with gradient descent.

    For example, a "uniqueness" constraint for Sudoku might return:
    - 0.0 if all 9 values in a row are distinct
    - A positive value proportional to how many duplicates exist

    Spec: REQ-VERIFY-001
    """

    @property
    def name(self) -> str:
        """Human-readable name for this constraint (e.g., 'row_3', 'clue_r2c5').

        Used in verification reports to identify which constraints pass/fail.
        """
        ...

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute constraint energy. Returns 0.0 if fully satisfied.

        Args:
            x: Configuration vector (e.g., flattened Sudoku grid).

        Returns:
            Non-negative scalar. Zero means satisfied.
        """
        ...

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of constraint energy with respect to x.

        Points in the direction of *increasing* violation. Negate it to
        move toward satisfaction.

        Args:
            x: Configuration vector.

        Returns:
            Array of same shape as x.
        """
        ...

    @property
    def satisfaction_threshold(self) -> float:
        """Threshold below which the constraint is considered satisfied.

        Due to floating-point arithmetic, energy may not be exactly zero
        even for valid configurations. This threshold provides a tolerance.
        Typical values: 1e-6 for exact constraints, 0.01 for soft constraints.
        """
        ...

    def is_satisfied(self, x: jax.Array) -> bool:
        """Is this constraint satisfied for configuration x?

        Returns True if energy(x) <= satisfaction_threshold.
        """
        ...


@dataclass
class ConstraintReport:
    """Report for a single constraint's evaluation.

    **Researcher summary:**
        Per-constraint energy decomposition: raw energy, weighted energy,
        and satisfaction status.

    **Detailed explanation for engineers:**
        When debugging why a configuration fails verification, you need to
        know which specific constraints are violated and by how much. This
        dataclass captures that information for one constraint.

    Attributes:
        name: Human-readable constraint name.
        energy: Raw (unweighted) energy for this constraint.
        weighted_energy: Energy multiplied by its weight in the composed system.
        satisfied: Whether energy <= satisfaction_threshold.

    Spec: REQ-VERIFY-002, REQ-VERIFY-003
    """

    name: str
    energy: float
    weighted_energy: float
    satisfied: bool


@dataclass
class Verdict:
    """Verification verdict -- overall pass/fail plus failing constraint names.

    **Researcher summary:**
        Binary verified flag plus list of failing constraint names.

    **Detailed explanation for engineers:**
        The top-level result of a verification check. ``verified=True`` means
        all constraints are satisfied. If ``verified=False``, the ``failing``
        list tells you exactly which constraints are violated, so you can
        target repair efforts.

    Spec: REQ-VERIFY-003
    """

    verified: bool
    failing: list[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Complete verification result -- total energy, per-constraint reports, verdict.

    **Researcher summary:**
        Full verification output: total composed energy, per-constraint
        decomposition, and binary verdict.

    **Detailed explanation for engineers:**
        This is the comprehensive output of ``ComposedEnergy.verify()``.
        It gives you:
        - ``total_energy``: The sum of all weighted constraint energies. Zero
          means all constraints are satisfied.
        - ``constraints``: A list of ConstraintReport objects, one per constraint,
          showing individual energies and satisfaction status.
        - ``verdict``: The overall pass/fail verdict with a list of failing
          constraint names.

    For example::

        result = composed_energy.verify(x)
        if result.is_verified():
            print("All constraints satisfied!")
        else:
            print(f"Failed: {result.failing_constraints()}")
            for c in result.constraints:
                if not c.satisfied:
                    print(f"  {c.name}: energy={c.energy:.4f}")

    Spec: REQ-VERIFY-003
    """

    total_energy: float
    constraints: list[ConstraintReport]
    verdict: Verdict

    def is_verified(self) -> bool:
        """Returns True if all constraints are satisfied."""
        return self.verdict.verified

    def failing_constraints(self) -> list[str]:
        """Returns names of constraints that are violated."""
        return self.verdict.failing


class BaseConstraint:
    """Base class for constraints with default threshold and auto-gradient.

    **Researcher summary:**
        Provides default satisfaction_threshold (1e-6), is_satisfied check,
        and jax.grad-based gradient. Subclasses only need to implement
        ``energy()`` and ``name``.

    **Detailed explanation for engineers:**
        This is a convenience base class. Instead of implementing all methods
        of the ConstraintTerm protocol from scratch, you can inherit from
        BaseConstraint and only implement:
        - ``energy(self, x)`` — the constraint energy
        - ``name`` property — a human-readable name

        The base class provides:
        - ``satisfaction_threshold``: 1e-6 (override for softer constraints)
        - ``is_satisfied(x)``: checks energy(x) <= threshold
        - ``grad_energy(x)``: uses jax.grad to auto-derive the gradient

    Spec: REQ-VERIFY-001
    """

    @property
    def satisfaction_threshold(self) -> float:
        """Default threshold: 1e-6 (very tight, for exact constraints)."""
        return 1e-6

    def is_satisfied(self, x: jax.Array) -> bool:
        """Check if constraint energy is below the satisfaction threshold.

        Note: converts JAX scalar to Python float for comparison. This
        triggers a device-to-host transfer if running on GPU/TPU.
        """
        return float(self.energy(x)) <= self.satisfaction_threshold

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Auto-derive gradient via jax.grad.

        Uses JAX's automatic differentiation to compute dE/dx without
        any manual gradient code. This works for any energy function
        composed of JAX-compatible operations.

        Spec: REQ-VERIFY-001
        """
        return jax.grad(self.energy)(x)


class ComposedEnergy:
    """An energy function composed of multiple weighted constraint terms.

    **Researcher summary:**
        Weighted sum of constraint energies: E(x) = sum_i w_i * C_i(x).
        Supports decomposition, verification, selective gradient, and batching.

    **Detailed explanation for engineers:**
        This is the main class for building complex verification systems.
        You add individual constraints (e.g., "row 3 has unique values",
        "cell (2,5) equals 7") with weights, and ComposedEnergy aggregates
        them into a single differentiable energy function.

        **Why weights?**
        Some constraints are more important than others. For Sudoku, clue
        constraints (given numbers) might have weight 10.0 while uniqueness
        constraints have weight 1.0. This tells the optimizer/sampler to
        prioritize satisfying clues first.

        **What can you do with a ComposedEnergy?**
        - ``energy(x)``: Total weighted energy (for samplers)
        - ``grad_energy(x)``: Total gradient (for Langevin/HMC)
        - ``energy_batch(xs)``: Batched energy (for score matching training)
        - ``decompose(x)``: See per-constraint energy breakdown
        - ``verify(x)``: Full verification with pass/fail verdict
        - ``grad_violated_only(x)``: Gradient from only failing constraints
          (for targeted repair)

        ComposedEnergy also satisfies the EnergyFunction protocol, so it can
        be used directly with any sampler (Langevin, HMC).

    For example::

        composed = ComposedEnergy(input_dim=81)
        composed.add_constraint(row_constraint, weight=1.0)
        composed.add_constraint(clue_constraint, weight=10.0)

        # Use with a sampler
        sampler = LangevinSampler(step_size=0.01)
        solution = sampler.sample(composed, x0, n_steps=1000)

        # Verify the result
        result = composed.verify(solution)
        print(result.is_verified())

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004
    """

    def __init__(self, input_dim: int) -> None:
        """Create a ComposedEnergy with no constraints.

        Args:
            input_dim: Dimension of the input configuration vector.
                Must match the dimension expected by all added constraints.
        """
        # List of (constraint, weight) tuples
        self._terms: list[tuple[ConstraintTerm, float]] = []
        self._input_dim = input_dim

    def add_constraint(self, term: ConstraintTerm, weight: float) -> None:
        """Add a constraint term with an importance weight.

        Higher weight means this constraint contributes more to the total
        energy and its gradient, causing the optimizer to prioritize it.

        Args:
            term: Any object satisfying the ConstraintTerm protocol.
            weight: Positive scalar weight. Typical range: 1.0 to 100.0.

        Spec: REQ-VERIFY-004
        """
        self._terms.append((term, weight))

    @property
    def num_constraints(self) -> int:
        """Number of constraint terms currently in this composed energy."""
        return len(self._terms)

    @property
    def input_dim(self) -> int:
        """Dimension of the input configuration vector."""
        return self._input_dim

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute total weighted energy: E(x) = sum_i w_i * C_i(x).

        **Detailed explanation for engineers:**
            Iterates over all (constraint, weight) pairs and sums up
            weight * constraint.energy(x). The result is a scalar that
            is zero if and only if ALL constraints are satisfied (assuming
            weights are positive).

        Args:
            x: Configuration vector, shape (input_dim,).

        Returns:
            Scalar total energy.

        Spec: REQ-VERIFY-001
        """
        total = jnp.float32(0.0)
        for term, weight in self._terms:
            total = total + weight * term.energy(x)
        return total

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Total gradient from all constraints via jax.grad.

        Uses automatic differentiation on the ``energy()`` method, which
        correctly propagates through all constraint terms and their weights.
        """
        return jax.grad(self.energy)(x)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Batched energy via jax.vmap.

        **How jax.vmap works here:**
            Transforms ``energy(x)`` (single input) into a function that
            processes an entire batch at once. JAX compiles this into
            efficient vectorized operations.

        Args:
            xs: Batch of configurations, shape (batch_size, input_dim).

        Returns:
            Energies, shape (batch_size,).
        """
        return jax.vmap(self.energy)(xs)

    def decompose(self, x: jax.Array) -> list[ConstraintReport]:
        """Per-constraint energy decomposition.

        **Researcher summary:**
            Returns individual constraint energies and satisfaction status.

        **Detailed explanation for engineers:**
            Evaluates each constraint independently and returns a list of
            ConstraintReport objects. This is the primary debugging tool:
            when a configuration fails verification, decompose tells you
            exactly which constraints are violated and by how much.

        Args:
            x: Configuration vector, shape (input_dim,).

        Returns:
            List of ConstraintReport, one per constraint term.

        Spec: REQ-VERIFY-002
        """
        reports = []
        for term, weight in self._terms:
            raw_energy = float(term.energy(x))
            reports.append(
                ConstraintReport(
                    name=term.name,
                    energy=raw_energy,
                    weighted_energy=weight * raw_energy,
                    satisfied=term.is_satisfied(x),
                )
            )
        return reports

    def verify(self, x: jax.Array) -> VerificationResult:
        """Produce a full verification result.

        **Researcher summary:**
            Decomposes, checks all constraints, returns VerificationResult
            with total energy, per-constraint reports, and binary verdict.

        **Detailed explanation for engineers:**
            This is the primary API for checking if a configuration is valid.
            It:
            1. Calls ``decompose(x)`` to get per-constraint reports
            2. Sums up weighted energies for the total
            3. Collects names of failing constraints
            4. Returns a VerificationResult with a Verdict

            A configuration is "verified" if and only if ALL constraints
            are satisfied (energy below their individual thresholds).

        Args:
            x: Configuration vector, shape (input_dim,).

        Returns:
            VerificationResult with total energy, reports, and verdict.

        Spec: REQ-VERIFY-003
        """
        reports = self.decompose(x)
        total_energy = sum(r.weighted_energy for r in reports)
        failing = [r.name for r in reports if not r.satisfied]
        verdict = Verdict(verified=len(failing) == 0, failing=failing)
        return VerificationResult(
            total_energy=total_energy,
            constraints=reports,
            verdict=verdict,
        )

    def grad_violated_only(self, x: jax.Array) -> jax.Array:
        """Gradient from only the currently violated constraints.

        **Researcher summary:**
            Selective gradient: only violated terms contribute, avoiding
            unnecessary perturbation of already-satisfied constraints.

        **Detailed explanation for engineers:**
            During repair, we only want to fix what's broken. If we used
            the full gradient (from all constraints), we might inadvertently
            break constraints that are already satisfied. This method computes
            the gradient contribution only from constraints where
            ``is_satisfied(x)`` returns False.

            Note: This uses a Python loop with ``is_satisfied()`` checks,
            which means it cannot be JIT-compiled by JAX (the control flow
            depends on runtime values). For JIT-compatible selective gradients,
            you would need ``jnp.where`` masking instead.

        Args:
            x: Configuration vector, shape (input_dim,).

        Returns:
            Gradient array of shape (input_dim,), with contributions only
            from violated constraints.

        Spec: REQ-VERIFY-005
        """
        grad = jnp.zeros(x.shape)
        for term, weight in self._terms:
            # Only include gradient from constraints that are NOT satisfied
            if not term.is_satisfied(x):
                grad = grad + weight * term.grad_energy(x)
        return grad


def repair(
    composed: ComposedEnergy,
    x: jax.Array,
    step_size: float,
    max_steps: int,
    noise_scale: float = 0.0,
    randomize_step_size: bool = False,
    key: jax.Array | None = None,
) -> tuple[jax.Array, list[VerificationResult]]:
    """Gradient-based repair: iteratively descend on violated constraints.

    **Researcher summary:**
        Gradient descent using only violated-constraint gradients. Stops early
        if all constraints are satisfied. Returns repaired configuration and
        verification history. Optionally adds Langevin noise (P6) and/or
        randomized step sizes (P11) to escape local minima.

    **Detailed explanation for engineers:**
        This function attempts to "fix" an invalid configuration by repeatedly:
        1. Checking which constraints are violated (via ``verify()``)
        2. Computing the gradient from only those violated constraints
        3. Taking a gradient descent step to reduce the violations
        4. Optionally adding Langevin noise to explore (prevents local minima)
        5. Optionally randomizing step size (prevents overfitting to path)

        It stops when either:
        - All constraints are satisfied (success!)
        - max_steps iterations are reached (may still have violations)

        **Langevin noise (P6, from EBT paper):**
        When ``noise_scale > 0``, Gaussian noise is added each step:
        ``x = x - step * grad + noise_scale * N(0, I)``
        The EBT paper ablations show 17% improvement from this exploration.

        **Randomized step size (P11, from EBT paper):**
        When ``randomize_step_size=True``, each step uses
        ``step = step_size * (0.5 + uniform(0,1))`` — varies from 0.5x to 1.5x.
        Prevents the optimizer from memorizing a single trajectory.

        **Why only violated constraints?**
        If we descended on ALL constraints (including satisfied ones), we might
        "overshoot" and break constraints that were already fine. By only
        targeting violated constraints, repair is more surgical.

    Args:
        composed: The ComposedEnergy containing all constraints.
        x: Initial (possibly invalid) configuration, shape (input_dim,).
        step_size: Gradient descent step size. Larger = faster but may
            overshoot. Typical range: 0.01 to 1.0.
        max_steps: Maximum number of repair iterations.
        noise_scale: Standard deviation of Langevin exploration noise.
            Default 0.0 (no noise, pure gradient descent). Typical: 0.01-0.1.
        randomize_step_size: If True, multiply step_size by a random factor
            U(0.5, 1.5) each iteration. Default False.
        key: JAX PRNG key for noise/randomization. Required if noise_scale > 0
            or randomize_step_size is True. If None, uses PRNGKey(0).

    Returns:
        Tuple of (repaired_x, history) where history is a list of
        VerificationResult objects from each iteration.

    For example::

        composed = build_sudoku_energy(puzzle)
        x0 = jnp.ones(81) * 5.0  # bad initial guess
        x_repaired, history = repair(composed, x0, step_size=0.1, max_steps=100)

        if history[-1].is_verified():
            print("Repair succeeded!")
        else:
            print(f"Still failing: {history[-1].failing_constraints()}")

    Spec: REQ-VERIFY-005
    """
    import jax.random as jrandom

    history: list[VerificationResult] = []
    needs_key = noise_scale > 0 or randomize_step_size
    if needs_key and key is None:
        key = jrandom.PRNGKey(0)

    for _ in range(max_steps):
        # Check current state of all constraints
        result = composed.verify(x)
        history.append(result)
        # Early exit if everything is satisfied
        if result.is_verified():
            break
        # Compute gradient from only the violated constraints
        grad = composed.grad_violated_only(x)

        # P11: Randomize step size to prevent single-trajectory overfitting
        current_step = step_size
        if randomize_step_size and key is not None:
            key, subkey = jrandom.split(key)
            current_step = step_size * (0.5 + float(jrandom.uniform(subkey)))

        # Gradient descent step (move against the gradient to reduce energy)
        x = x - current_step * grad

        # P6: Langevin noise for exploration (escape local minima)
        if noise_scale > 0 and key is not None:
            key, subkey = jrandom.split(key)
            x = x + noise_scale * jrandom.normal(subkey, x.shape)

    return x, history
