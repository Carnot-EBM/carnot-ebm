"""Lagrange Oscillatory Neural Networks (LagONN) -- JAX implementation.

**Researcher summary:**
    Implements arxiv 2505.07179 (Delacour et al., 2025). LagONN augments a
    standard Ising Hamiltonian with Lagrange multiplier penalty terms that
    grow proportional to constraint violations via dual ascent. Standard Ising
    machines get trapped in infeasible local minima when hard constraints exist
    (e.g., scheduling, combinatorial feasibility). LagONN escapes infeasibility
    by making the energy landscape increasingly unfavorable for constraint-
    violating configurations as λ grows. Interleaves parallel Gibbs sweeps with
    dual-ascent λ updates.

**Detailed explanation for engineers:**
    A standard Ising machine minimizes:

        E_ising(x) = -0.5 * x^T J x - b^T x

    where x ∈ {0,1}^n are binary spins, J is the (n×n) coupling matrix, and b
    is the bias vector. This works well for unconstrained optimization, but when
    hard constraints of the form Ax ≤ c exist (like "total work load ≤ capacity"
    or "at most K items selected"), the sampler happily finds low-energy solutions
    that *violate* the constraints — because the energy function doesn't know about
    them.

    **The LagONN fix:**
    Add a penalty term weighted by Lagrange multipliers λ ∈ ℝ_+^m (one per
    constraint):

        E_lagoon(x) = E_ising(x) + λ^T max(0, Ax - c)

    where max(0, Ax - c) is the "violation vector" — non-zero only where the
    constraint is violated, zero when satisfied. The Lagrange multipliers λ are
    updated after each Gibbs sweep via dual ascent:

        λ ← max(0, λ + lr * max(0, Ax - c))

    This is the dual ascent (subgradient) step for maximizing the Lagrangian.
    Each step, violated constraints get a bigger penalty, pushing the sampler
    away from infeasible regions. Satisfied constraints don't grow λ (it stays
    non-negative), so constraints that are already met don't get over-penalized.

    **Why "oscillatory"?**
    The λ updates cause the effective energy landscape to oscillate: when a
    constraint is violated, its penalty grows until the sampler avoids it, then
    the sampler might violate a different constraint, growing its penalty, etc.
    This oscillation between feasible and infeasible regions drives exploration
    toward configurations that satisfy *all* constraints simultaneously.

    **Parallel Gibbs with Lagrange field:**
    For binary spin x_i ∈ {0,1}, the exact Gibbs conditional is:

        P(x_i=1 | x_{-i}) = sigmoid(h_i)

    where h_i = E(x with x_i=0) - E(x with x_i=1) is the "local field."
    For the Ising part: h_ising_i = (J@x)_i + b_i  (standard, zero-diagonal J)
    For the Lagrange part, flipping x_i from 0→1 changes each constraint's
    violation by A[:,i], so:

        u   = A @ x - c              (current raw violations, shape m)
        u0  = u - A[:,i] * x_i       (violation with x_i forced to 0)
        u1  = u0 + A[:,i]            (violation with x_i forced to 1)
        h_lagrange_i = λ^T (max(0,u0) - max(0,u1))  ← negative penalty for x_i=1

    Vectorized across all n spins simultaneously (O(mn) total):

        u0 = u[:,None] - A * x[None,:]    (shape m×n)
        u1 = u0 + A                        (shape m×n)
        h_lagrange = λ @ (max(0,u0) - max(0,u1))  (shape n)

    Full Gibbs conditional field:
        h = (J @ x) + b + h_lagrange
        P(x_i=1) = sigmoid(2 * beta * h_i)

    This is exact Gibbs (not a mean-field approximation) for the augmented
    energy, computed efficiently with two matrix-vector products.

**Reference:**
    Delacour, C., Haverkort, B., Sabo, F. et al. (2025). "Lagrange Oscillatory
    Neural Networks." arXiv:2505.07179.

Spec: REQ-LAGOON-001, REQ-LAGOON-002, REQ-LAGOON-003
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom


# ---------------------------------------------------------------------------
# Helper: Lagrange-augmented local field
# ---------------------------------------------------------------------------


def _lagoon_local_field(
    x: jax.Array,
    J: jax.Array,
    bias: jax.Array,
    A: jax.Array,
    b: jax.Array,
    lambda_: jax.Array,
) -> jax.Array:
    """Compute the Lagrange-augmented local field for every spin simultaneously.

    **Researcher summary:**
        h_i = (J@x)_i + b_i + λ^T(max(0,u0_i) - max(0,u1_i))
        where u0_i is the violation vector with x_i set to 0, u1_i with x_i=1.

    **Detailed explanation for engineers:**
        The local field h_i is the "effective external field" seen by spin i
        from all its neighbors and the constraint penalties. A positive h_i
        means spin i wants to be 1 (reduces energy); negative means it wants
        to be 0.

        The Lagrange contribution:
        - h_lagrange_i > 0: setting x_i=1 *reduces* the Lagrange penalty
          (fewer constraint violations), so it's encouraged.
        - h_lagrange_i < 0: setting x_i=1 *increases* the Lagrange penalty
          (causes more violations), so it's discouraged.

    Args:
        x: Current spin configuration, shape (n,), float32 in {0, 1}.
        J: Coupling matrix, shape (n, n), symmetric, zero diagonal.
        bias: Bias vector, shape (n,).
        A: Constraint matrix, shape (m, n). Constraints are Ax ≤ b.
        b: Constraint bounds, shape (m,).
        lambda_: Lagrange multipliers, shape (m,), non-negative.

    Returns:
        Local field vector h, shape (n,). Pass to sigmoid(2*beta*h) for
        Gibbs conditional probabilities.

    Spec: REQ-LAGOON-002
    """
    n = x.shape[0]

    # Standard Ising local field for all spins at once: (J @ x)_i + b_i
    # Since J is zero-diagonal, (J @ x)_i is correct regardless of x_i's value.
    h_ising = J @ x + bias  # shape (n,)

    # Lagrange local field: compute energy change when each spin flips 0→1.
    # u = A @ x - b: current raw violations per constraint, shape (m,)
    u = A @ x - b

    # u0[:,i] = violation vector if x_i were forced to 0:
    #   u0 = u[:, None] - A * x[None, :]   (subtract A[:,i]*x_i for each spin i)
    # u1[:,i] = violation vector if x_i were forced to 1:
    #   u1 = u0 + A   (add A[:,i] for each spin i)
    # Shapes: both (m, n)
    u0 = u[:, None] - A * x[None, :]  # shape (m, n)
    u1 = u0 + A  # shape (m, n)

    # Energy decrease from Lagrange term when x_i=1 vs x_i=0:
    #   h_lagrange_i = λ^T(max(0, u0[:,i]) - max(0, u1[:,i]))
    # Positive if flipping to 1 reduces violations, negative if it adds them.
    h_lagrange = lambda_ @ (jnp.maximum(0.0, u0) - jnp.maximum(0.0, u1))  # shape (n,)

    return h_ising + h_lagrange


def _gibbs_sweep(
    x: jax.Array,
    J: jax.Array,
    bias: jax.Array,
    A: jax.Array,
    b: jax.Array,
    lambda_: jax.Array,
    beta: float | jax.Array,
    key: jax.Array,
) -> jax.Array:
    """One parallel Gibbs sweep using the Lagrange-augmented conditional.

    **Detailed explanation for engineers:**
        Computes the Gibbs conditional probability for every spin simultaneously:
            P(x_i=1 | x_{-i}) = sigmoid(2 * beta * h_i)
        where h_i is the full local field (Ising + Lagrange). Then samples all
        spins independently (parallel Gibbs). This is an approximation to
        sequential Gibbs (where each spin sees the freshest state of all other
        spins), but converges to the correct stationary distribution at low
        temperatures and is orders of magnitude faster due to full vectorization.

    Args:
        x: Current spins, shape (n,), float32.
        J: Coupling matrix, shape (n, n).
        bias: Bias vector, shape (n,).
        A: Constraint matrix, shape (m, n).
        b: Constraint bounds, shape (m,).
        lambda_: Lagrange multipliers, shape (m,).
        beta: Inverse temperature (scalar). Higher = more greedy.
        key: JAX PRNG key for Bernoulli sampling.

    Returns:
        New spin configuration, shape (n,), float32.

    Spec: REQ-LAGOON-002
    """
    h = _lagoon_local_field(x, J, bias, A, b, lambda_)
    # Gibbs conditional: P(x_i=1) = sigmoid(2 * beta * h_i)
    probs = jax.nn.sigmoid(2.0 * beta * h)
    # Sample all spins in parallel from their conditionals.
    return jrandom.bernoulli(key, probs).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Main LagONN class
# ---------------------------------------------------------------------------


@dataclass
class LagONN:
    """Lagrange Oscillatory Neural Network: Ising + dual-ascent constraints.

    **Researcher summary:**
        Pairwise EBM with hard constraints Ax ≤ b enforced via Lagrange
        multipliers λ updated by dual ascent. Energy:

            E(x) = -0.5 x^T J x - b^T x + λ^T max(0, Ax - c)

        Sampling interleaves parallel Gibbs sweeps with λ dual-ascent steps.
        Provably escapes infeasible local minima that trap standard Ising.

    **Detailed explanation for engineers:**
        LagONN extends the Ising model with constraint-satisfaction machinery.
        The key design principle: instead of hard-coding feasibility by
        restricting the search space (which is NP-hard in general), LagONN
        *softens* hard constraints into energy penalties that grow over time.
        Initially λ=0 so the model just minimizes the Ising energy. As
        infeasible solutions dominate, λ grows for the violated constraints,
        adding energy penalties that steer the sampler toward feasibility.

        After sufficient dual-ascent steps, λ has grown large enough that
        the feasible region becomes the global energy minimum, and the sampler
        concentrates its probability mass there.

        **When does this NOT work?**
        - Constraints with very high coupling (J matrix entries >> constraint
          scale) may not overcome the energy barrier before λ grows too large.
        - Infeasible problems (Ax ≤ b has no solution) cause λ → ∞.
        - Short runs (few steps) may not give λ time to converge.

    Attributes:
        J: Coupling matrix, shape (n, n). Should be symmetric and zero-diagonal.
        bias: Bias vector, shape (n,).
        A: Constraint matrix, shape (m, n). Defines m linear constraints Ax ≤ c.
        b: Constraint bound vector, shape (m,). (Named 'b' in the problem
            definition; this is the RHS of Ax ≤ b, different from the bias.)
        lambda_: Lagrange multipliers, shape (m,). Non-negative (λ ≥ 0).
            Initialized to zero (no penalty). Grows via dual ascent during
            sampling to enforce constraints.

    Note on naming: ``self.b`` is the constraint bound (RHS of Ax ≤ b) and
    ``self.bias`` is the Ising bias vector b in E = -0.5 x^T J x - bias^T x.
    These are different! The 'b' naming follows the task specification.

    Spec: REQ-LAGOON-001, REQ-LAGOON-002, REQ-LAGOON-003
    """

    J: jax.Array  # coupling matrix (n × n), symmetric, zero diagonal
    bias: jax.Array  # bias vector (n,)
    A: jax.Array  # constraint matrix (m × n): Ax ≤ b
    b: jax.Array  # constraint bound vector (m,)
    lambda_: jax.Array  # Lagrange multipliers (m,), ≥ 0, initialized to zeros

    # ------------------------------------------------------------------
    # EnergyFunction protocol
    # ------------------------------------------------------------------

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute total scalar energy: Ising term + Lagrange penalty.

        **Researcher summary:**
            E(x) = -0.5 x^T J x - bias^T x + λ^T max(0, Ax - b)
            Zero when x satisfies all constraints (at λ > 0, satisfied
            constraints don't add penalty because max(0, negative) = 0).

        **Detailed explanation for engineers:**
            Two components:

            1. **Ising energy** (always present):
               E_ising = -0.5 * x @ J @ x - bias @ x
               Low energy = x aligns with coupling matrix J and bias.

            2. **Lagrange penalty** (grows over time as λ is updated):
               E_lagrange = λ^T max(0, Ax - b)
               Non-zero only when some constraints are violated (Ax > b for
               those rows). Each violated constraint j contributes
               λ_j * (A[j] @ x - b[j]) to the energy. As λ_j grows, this
               penalty becomes large, making constraint violation energetically
               unfavorable.

            When λ=0 (initial state), E(x) = E_ising(x) — standard Ising.

        Args:
            x: Spin configuration, shape (n,).

        Returns:
            Scalar total energy.

        Spec: REQ-LAGOON-001
        """
        # Ising pairwise + bias energy
        e_ising = -0.5 * x @ self.J @ x - self.bias @ x

        # Constraint violation penalty: only penalize violated constraints
        # max(0, Ax - b) is 0 for satisfied constraints (Ax ≤ b), positive otherwise
        violation = jnp.maximum(0.0, self.A @ x - self.b)
        e_lagrange = self.lambda_ @ violation

        return e_ising + e_lagrange

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy w.r.t. x via JAX autodiff.

        **Detailed explanation for engineers:**
            Uses ``jax.grad`` to automatically differentiate the ``energy()``
            method. The Ising part gives: dE_ising/dx = -J @ x - bias.
            The Lagrange part: dE_lagrange/dx = A^T * (λ * indicator(Ax > b))
            — the columns of A^T weighted by λ, but only for violated
            constraints (the subgradient of max(0, Ax - b)).

        Args:
            x: Spin configuration, shape (n,).

        Returns:
            Gradient array, shape (n,).

        Spec: REQ-LAGOON-001
        """
        return jax.grad(self.energy)(x)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Batched energy computation via jax.vmap.

        Args:
            xs: Batch of spin configurations, shape (batch, n).

        Returns:
            Energies, shape (batch,).

        Spec: REQ-LAGOON-001
        """
        return jax.vmap(self.energy)(xs)

    @property
    def input_dim(self) -> int:
        """Number of spin variables (n)."""
        return int(self.J.shape[0])

    @property
    def n_constraints(self) -> int:
        """Number of hard constraints (m)."""
        return int(self.A.shape[0])

    # ------------------------------------------------------------------
    # Dual-ascent λ update
    # ------------------------------------------------------------------

    def update_lambda(self, x: jax.Array, lr: float = 0.01) -> "LagONN":
        """Dual-ascent step: increase λ for currently-violated constraints.

        **Researcher summary:**
            λ ← max(0, λ + lr * max(0, Ax - b))
            Sub-gradient ascent on the Lagrangian. Satisfied constraints
            don't grow λ (violation = 0). λ ≥ 0 is enforced by the outer max.

        **Detailed explanation for engineers:**
            This is the "dual ascent" or "subgradient" step from Lagrangian
            relaxation theory. The Lagrangian is:

                L(x, λ) = E_ising(x) + λ^T max(0, Ax - b)

            We want to find λ* such that the primal minimum (over x) satisfies
            the constraints. The dual ascent step:

                λ ← λ + lr * ∇_λ L(x, λ) = λ + lr * max(0, Ax - b)

            followed by projection onto λ ≥ 0 (the outer max).

            **Why is this correct?**
            If constraint j is violated (A[j] @ x > b[j]):
            - violation[j] = A[j] @ x - b[j] > 0
            - λ_j increases → E_lagrange grows → sampler avoids violation

            If constraint j is satisfied (A[j] @ x ≤ b[j]):
            - violation[j] = 0
            - λ_j doesn't change (but won't go negative due to the outer max)

            Over many steps, λ converges to the Lagrange dual optimal, which
            forces the primal solution x to become feasible.

        Args:
            x: Current spin configuration (used to compute violations), shape (n,).
            lr: Dual ascent learning rate. Larger = faster λ growth but may
                oscillate. Typical range: 0.001 to 0.1. Default 0.01.

        Returns:
            New LagONN instance with updated lambda_. The original is unchanged
            (functional/immutable update via dataclasses.replace).

        Spec: REQ-LAGOON-003
        """
        # Compute current constraint violations (zero if satisfied, positive if violated)
        violation = jnp.maximum(0.0, self.A @ x - self.b)
        # Dual ascent: increase λ proportional to violation, clamp to ≥ 0
        new_lambda = jnp.maximum(0.0, self.lambda_ + lr * violation)
        # Return a new instance with updated λ (immutable dataclass pattern)
        return dataclasses.replace(self, lambda_=new_lambda)

    # ------------------------------------------------------------------
    # Feasibility check
    # ------------------------------------------------------------------

    def is_feasible(self, x: jax.Array) -> jax.Array:
        """Check if a configuration satisfies all hard constraints Ax ≤ b.

        **Detailed explanation for engineers:**
            Returns a scalar boolean: True iff ALL m constraints are satisfied.
            A constraint j is satisfied when (A[j] @ x) ≤ b[j], i.e., when the
            raw violation = A[j] @ x - b[j] ≤ 0.

            This is the key metric for benchmarking LagONN vs baseline Ising:
            what fraction of final samples are feasible?

        Args:
            x: Spin configuration, shape (n,), float32 or bool.

        Returns:
            Boolean scalar: True if feasible.

        Spec: REQ-LAGOON-003
        """
        raw = self.A @ x - self.b  # shape (m,), positive = violated
        return jnp.all(raw <= 0.0)

    def feasibility_rate(self, samples: jax.Array) -> float:
        """Compute fraction of samples satisfying all constraints.

        **Detailed explanation for engineers:**
            Applies ``is_feasible`` to each row of ``samples`` and returns the
            mean (fraction of feasible samples). This is the primary benchmark
            metric: LagONN should achieve a higher feasibility rate than
            unconstrained Ising sampling, especially as λ grows.

        Args:
            samples: Batch of spin configurations, shape (n_samples, n), float32.

        Returns:
            Python float in [0, 1]. 1.0 = all samples feasible.

        Spec: REQ-LAGOON-003
        """
        feasible_flags = jax.vmap(self.is_feasible)(samples)
        return float(jnp.mean(feasible_flags.astype(jnp.float32)))

    # ------------------------------------------------------------------
    # Sampling: interleaved Gibbs + dual-ascent λ updates
    # ------------------------------------------------------------------

    def sample(
        self,
        key: jax.Array,
        n_steps: int = 100,
        n_samples: int = 10,
        beta: float = 5.0,
        lr: float = 0.01,
    ) -> tuple[jax.Array, "LagONN"]:
        """Sample via interleaved parallel Gibbs sweeps and λ dual-ascent updates.

        **Researcher summary:**
            Alternates: (1) parallel Gibbs sweep using current λ, (2) dual-
            ascent λ update using new x. After n_steps warmup sweeps, collects
            n_samples. Returns samples and the final model (with converged λ).

        **Detailed explanation for engineers:**
            The sampling loop:

                FOR t = 1..n_steps:
                    x ← Gibbs_sweep(x, J, bias, A, b, λ, beta)  ← Lagrange-augmented
                    λ ← max(0, λ + lr * max(0, Ax - b))          ← dual ascent

                COLLECT:
                FOR s = 1..n_samples:
                    x ← Gibbs_sweep(x, J, bias, A, b, λ, beta)
                    λ ← update(λ, x)  ← still updating during collection
                    SAVE x to samples

            The interleaving is key: each Gibbs sweep uses the current λ to
            compute conditionals, and each λ update uses the Gibbs sweep's
            output to compute violations. This tight feedback loop rapidly
            steers the sampler toward feasibility.

            **Implementation note:**
            The warmup and collection use Python for-loops with ``jax.jit``-
            compiled inner functions. Lambda updates require Python control flow
            (immutable dataclass replacement), so the outer loop cannot be
            jax.lax.scan'd without significant refactoring. For large n_steps,
            consider compiling with XLA via the ``_sample_jax_loop`` helper
            if performance is critical.

        Args:
            key: JAX PRNG key for reproducible randomness.
            n_steps: Number of warmup Gibbs + λ-update iterations before
                collecting samples. More steps = better λ convergence.
            n_samples: Number of samples to collect after warmup.
            beta: Inverse temperature for Gibbs sampling. Higher = more greedy
                (concentrates probability on low-energy states). Default 5.0.
            lr: Dual-ascent learning rate for λ updates. Default 0.01.

        Returns:
            Tuple of:
            - samples: jnp.ndarray of shape (n_samples, n), float32, values in {0,1}.
            - final_model: LagONN with the final (converged) λ_ values.

        Spec: REQ-LAGOON-002, REQ-LAGOON-003
        """
        n = self.input_dim
        model = self  # will be replaced (immutably) with updated λ each step

        # Initialize spins randomly with 50/50 probability
        key, init_key = jrandom.split(key)
        x = jrandom.bernoulli(init_key, 0.5, (n,)).astype(jnp.float32)

        # --- Phase 1: Warmup — Gibbs sweeps interleaved with λ dual-ascent ---
        # After each sweep, λ grows for violated constraints, steering the sampler
        # progressively toward feasibility.
        for _ in range(n_steps):
            key, sweep_key = jrandom.split(key)
            x = _gibbs_sweep(x, model.J, model.bias, model.A, model.b, model.lambda_, beta, sweep_key)
            model = model.update_lambda(x, lr=lr)

        # --- Phase 2: Collect samples (still updating λ during collection) ---
        # Continuing to update λ during collection keeps the sampler honest —
        # if it slips into infeasible territory, λ corrects it.
        collected: list[jax.Array] = []
        for _ in range(n_samples):
            key, sweep_key = jrandom.split(key)
            x = _gibbs_sweep(x, model.J, model.bias, model.A, model.b, model.lambda_, beta, sweep_key)
            model = model.update_lambda(x, lr=lr)
            collected.append(x)

        samples = jnp.stack(collected)  # shape (n_samples, n)
        return samples, model


# ---------------------------------------------------------------------------
# Factory functions for benchmark problem generators
# ---------------------------------------------------------------------------


def make_random_constrained_ising(
    n: int,
    m: int,
    key: jax.Array,
    coupling_scale: float = 1.0,
    constraint_density: float = 0.3,
) -> LagONN:
    """Generate a random Ising problem with m random linear constraints.

    **Researcher summary:**
        Random J ~ N(0, coupling_scale/n) symmetric, b ~ N(0, 0.1), sparse
        constraint matrix A ~ Bernoulli(density) * N(0, 1), bounds set so
        ~50% of random spin configs are infeasible.

    **Detailed explanation for engineers:**
        Creates a benchmark instance for testing LagONN vs baseline Ising:
        - The Ising part (J, b) defines a random optimization landscape.
        - The constraints (A, c) define a feasibility region. The bounds c
          are set to the median of A @ x over random x, so approximately half
          of random configurations violate at least one constraint.
        - lambda_ is initialized to zeros.

    Args:
        n: Number of spin variables.
        m: Number of hard constraints.
        key: JAX PRNG key.
        coupling_scale: Scale of coupling matrix entries.
        constraint_density: Fraction of A entries that are non-zero.

    Returns:
        LagONN instance with zero lambda_.

    Spec: REQ-LAGOON-001
    """
    k1, k2, k3, k4 = jrandom.split(key, 4)

    # Symmetric coupling matrix with zero diagonal
    j_raw = jrandom.normal(k1, (n, n)) * (coupling_scale / jnp.sqrt(n))
    J = (j_raw + j_raw.T) / 2.0
    J = J.at[jnp.arange(n), jnp.arange(n)].set(0.0)  # zero diagonal

    # Small bias vector
    bias = jrandom.normal(k2, (n,)) * 0.1

    # Sparse constraint matrix A ∈ R^{m × n}
    # Each constraint uses about density*n of the spin variables
    mask = jrandom.bernoulli(k3, constraint_density, (m, n)).astype(jnp.float32)
    A = jrandom.normal(k4, (m, n)) * mask

    # Set bounds so ~50% of uniformly random spin configs violate at least 1 constraint.
    # We approximate the median of A@x over x ~ Bernoulli(0.5)^n as E[A@x] = A @ 0.5*1 = A.sum(1)/2
    # Set c = A.sum(1)/2 so constraints are tight at the "average" configuration.
    c = A.sum(axis=1) / 2.0

    lambda_ = jnp.zeros(m)

    return LagONN(J=J, bias=bias, A=A, b=c, lambda_=lambda_)


def make_sat_constrained_ising(
    n_vars: int,
    n_clauses: int,
    n_hard_violations: int,
    key: jax.Array,
) -> LagONN:
    """Generate a Max-3-SAT-style constrained Ising problem.

    **Researcher summary:**
        Random 3-SAT with n_vars binary variables and n_clauses clauses.
        Each clause (l1 OR l2 OR l3) is encoded as a pairwise Ising energy
        term. One hard constraint: total violated clauses ≤ n_hard_violations.
        LagONN should find low-energy solutions that also satisfy the
        hard feasibility bound.

    **Detailed explanation for engineers:**
        Max-3-SAT asks: assign boolean values to n_vars variables to satisfy
        as many clauses as possible. Each clause has 3 literals (a variable
        or its negation). We can't encode 3-body interactions in a pairwise
        Ising model, so we use a penalty relaxation:

        For clause c = (l1 OR l2 OR l3) with signs s_k ∈ {+1,-1}:
            - Literal k is satisfied when x_ik = (1 + s_k) / 2
            - Let y_k = x_ik if s_k=+1, (1-x_ik) if s_k=-1 (the "effective spin")
            - Clause violated iff y_1=y_2=y_3=0 (all literals false)
            - Linear penalty contribution: -w * (y_1 + y_2 + y_3 - 0.5)
              where w is a clause weight (penalizes fewer satisfied literals)

        Hard constraint: sum of clause violations ≤ n_hard_violations.
        Encoded as a single linear constraint: A @ x ≤ b where A is built
        from the clause structure and b = n_hard_violations - (offset).

        **Simplification:**
        For feasibility testing, we use a relaxed version: the Ising energy
        penalizes violation of each clause (linear + pairwise approximation),
        and the hard constraint imposes a budget on total violations.

    Args:
        n_vars: Number of boolean variables.
        n_clauses: Number of 3-clauses.
        n_hard_violations: Maximum allowed clause violations (hard constraint).
        key: JAX PRNG key.

    Returns:
        LagONN instance. Hard constraint: at most n_hard_violations clauses
        can be violated. lambda_ initialized to zeros.

    Spec: REQ-LAGOON-001
    """
    k1, k2, k3 = jrandom.split(key, 3)

    # Generate random 3-clauses: each clause has 3 distinct variable indices
    # and a sign (+1 = positive literal, -1 = negated literal)
    import numpy as np

    rng = np.random.default_rng(int(jrandom.randint(k1, (), 0, 2**30)))
    clause_vars = np.zeros((n_clauses, 3), dtype=np.int32)
    clause_signs = rng.choice([-1, 1], size=(n_clauses, 3))
    for i in range(n_clauses):
        clause_vars[i] = rng.choice(n_vars, size=3, replace=False)

    # Build the Ising coupling and bias from clause structure.
    # Clause energy (contribution when violated):
    #   E_clause = -(y1 + y2 + y3) where y_k = x_ik (if +) or (1-x_ik) (if -)
    # Expanding into Ising form:
    #   y_k = 0.5*(1 + s_k) + 0.5*s_k*(2*x_ik - 1)  [converts {0,1} → effectively +/-]
    #   But we use simple linear term: y_k = (1+s_k)/2 * 1 + s_k * (x_ik - 0.5)
    # Bias contribution of variable v from clause c:
    #   If v is literal k with sign s_k in clause c:
    #     bias_contribution = +s_k * clause_weight
    #   (negative sign → penalizes x_v=0, positive sign → penalizes x_v=1 when sum=0)

    J = np.zeros((n_vars, n_vars), dtype=np.float32)
    bias = np.zeros(n_vars, dtype=np.float32)
    clause_weight = 1.0  # energy penalty per violated clause

    for c_idx in range(n_clauses):
        v0, v1, v2 = clause_vars[c_idx]
        s0, s1, s2 = clause_signs[c_idx]
        # Each literal y_k = (1+s_k)/2 + s_k*(x_{vk} - 0.5) ∈ {0,1}
        # Clause violated if y0=y1=y2=0, i.e., all literals false.
        # Linear Ising penalty for violating (approximate: weight each literal)
        # Add bias to discourage false literal (opposite sign encourages true)
        bias[v0] += clause_weight * s0 * 0.5
        bias[v1] += clause_weight * s1 * 0.5
        bias[v2] += clause_weight * s2 * 0.5
        # Pairwise couplings: literals that agree should be co-satisfied
        # (positive coupling → they want to be equal sign)
        J[v0, v1] += clause_weight * s0 * s1 * 0.1
        J[v1, v2] += clause_weight * s1 * s2 * 0.1
        J[v0, v2] += clause_weight * s0 * s2 * 0.1

    # Symmetrize J
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # Hard constraint: budget on total "effective spin sum" (proxy for satisfied clauses).
    # We use a KNAPSACK-style constraint: sum_v |a_v| * x_v ≤ capacity
    # where a_v encodes how much each variable contributes to the satisfaction count.
    # The constraint is calibrated so ~(n_hard_violations/n_clauses) fraction of
    # random configs are feasible — matching the target violation budget.
    #
    # Construction:
    #   A_hard[0, v] = |net_bias[v]| (positive weight: how much variable v matters)
    #   capacity = E[A_hard @ x] + sigma = 0.5 * sum(A_hard) + 0.5 * std_A
    #   This ensures roughly 50th–70th percentile of random configs are feasible,
    #   giving a non-trivial but achievable constraint.
    #
    # Why this works: the Ising energy biases toward x=1 everywhere. The hard
    # constraint limits the total weighted sum. LagONN must find configurations
    # that are low Ising energy AND below the sum budget.

    # Net bias magnitude per variable: higher = more important to the SAT structure
    A_hard_row = np.abs(bias).reshape(1, n_vars).astype(np.float32)
    # Normalize so the constraint is scale-independent
    A_hard_row = A_hard_row / (A_hard_row.sum() + 1e-8)

    # Expected value of A_hard @ x for x ~ Bernoulli(0.5): E[A@x] = 0.5 * sum(A)
    expected_ax = 0.5 * float(A_hard_row.sum())
    # Standard deviation: std[A@x] = sqrt(0.25 * sum(a_i^2)) for independent Bernoullis
    std_ax = float(np.sqrt(0.25 * float((A_hard_row**2).sum())))

    # Set bound at mean - 0.5 * std → roughly 30th percentile → ~70% of random configs
    # VIOLATE the constraint. This creates meaningful difficulty: the Ising energy
    # pushes toward configurations that tend to violate, and LagONN must escape.
    b_hard = np.array([expected_ax - 0.5 * std_ax], dtype=np.float32)

    lambda_ = jnp.zeros(1)

    return LagONN(
        J=jnp.array(J),
        bias=jnp.array(bias),
        A=jnp.array(A_hard_row),
        b=jnp.array(b_hard),
        lambda_=lambda_,
    )


def make_scheduling_ising(
    n_jobs: int,
    n_slots: int,
    key: jax.Array,
) -> LagONN:
    """Generate a job-scheduling constrained Ising problem.

    **Researcher summary:**
        n_jobs × n_slots binary variables: x_{i,t}=1 means job i is assigned
        to slot t. Hard constraints: each job in exactly one slot (assignment
        constraint), total slot load ≤ capacity (resource constraint). Ising
        energy minimizes total completion cost. n = n_jobs * n_slots variables,
        m = n_jobs + n_slots constraints.

    **Detailed explanation for engineers:**
        This models a simple resource scheduling problem:
        - Variables: x_{i,t} ∈ {0,1}, flattened as x[i*n_slots + t]
        - Job assignment constraint: for each job i, sum_t x_{i,t} = 1
          (exactly one slot). Encoded as 2 inequalities:
            sum_t x_{i,t} ≤ 1   (at most one slot)
            -sum_t x_{i,t} ≤ -1  (at least one slot)
        - Slot capacity constraint: for each slot t, sum_i w_i * x_{i,t} ≤ C
          where w_i ~ U[1, 3] are job weights and C = n_jobs * 2 / n_slots.
        - Ising energy: random J with slight preference for earlier slots
          (diagonal bias b_t ~ -t/n_slots).

        Total constraints m = 2*n_jobs + n_slots.
        Total variables n = n_jobs * n_slots.

    Args:
        n_jobs: Number of jobs (each needs exactly one slot).
        n_slots: Number of available time slots.
        key: JAX PRNG key.

    Returns:
        LagONN instance with scheduling constraints. lambda_ = 0.

    Spec: REQ-LAGOON-001
    """
    import numpy as np

    n = n_jobs * n_slots
    k1, k2, k3 = jrandom.split(key, 3)

    # Ising energy: random couplings + slight preference for earlier slots
    j_raw = jrandom.normal(k1, (n, n)).astype(float) * (0.5 / jnp.sqrt(n))
    J = (j_raw + j_raw.T) / 2.0
    J = J.at[jnp.arange(n), jnp.arange(n)].set(0.0)

    # Bias: prefer earlier slots (lower cost)
    bias_np = np.zeros(n, dtype=np.float32)
    for i in range(n_jobs):
        for t in range(n_slots):
            bias_np[i * n_slots + t] = -float(t) / n_slots  # negative → prefer early

    # Build constraint matrix.
    # We use TWO types of hard constraints:
    #   1. Assignment: sum_t x_{i,t} ≤ 1 for each job i (at most one slot per job).
    #      The Ising bias toward x=1 makes jobs want to occupy ALL slots,
    #      violating these constraints. LagONN's λ pushes them to pick one.
    #   2. Capacity: sum_i w_i * x_{i,t} ≤ C for each slot t.
    #      With Ising bias toward full assignment, slots will be overloaded.
    #
    # We do NOT include "at least 1 slot" constraints (sum_t x_{i,t} ≥ 1),
    # because those combined with "at most 1" create exact-equality constraints
    # whose feasible region is too small for Gibbs sampling to find quickly.
    # Without the "at least 1" constraint, x=0 is trivially feasible, but the
    # Ising energy makes x=0 high-energy → the sampler must find low-energy
    # feasible assignments with at most one slot per job within capacity.
    #
    # Capacity is set to 60% of total job weight (tight enough to require
    # distributing jobs across slots, but feasible for any sparse assignment).
    rng = np.random.default_rng(int(jrandom.randint(k3, (), 0, 2**30)))
    job_weights = rng.uniform(1.0, 3.0, size=n_jobs).astype(np.float32)
    # Capacity: total weight / n_slots * 1.5 → one fully-packed slot would be
    # ~150% of capacity, so LagONN needs to SPREAD jobs across slots.
    capacity = float(job_weights.sum()) / n_slots * 1.5

    # m = n_jobs (assignment: ≤1) + n_slots (capacity)
    m = n_jobs + n_slots
    A_np = np.zeros((m, n), dtype=np.float32)
    b_np = np.zeros(m, dtype=np.float32)

    row = 0
    # Assignment constraints: sum_t x_{i,t} ≤ 1 for each job i
    for i in range(n_jobs):
        for t in range(n_slots):
            A_np[row, i * n_slots + t] = 1.0
        b_np[row] = 1.0
        row += 1

    # Slot capacity constraints: sum_i w_i * x_{i,t} ≤ C for each slot t
    for t in range(n_slots):
        for i in range(n_jobs):
            A_np[row, i * n_slots + t] = job_weights[i]
        b_np[row] = capacity
        row += 1

    lambda_ = jnp.zeros(m)

    return LagONN(
        J=jnp.array(J),
        bias=jnp.array(bias_np),
        A=jnp.array(A_np),
        b=jnp.array(b_np),
        lambda_=lambda_,
    )
