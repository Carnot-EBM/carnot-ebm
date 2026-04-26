"""LagrangeAdaptiveIsingConstraints — violation-driven lambda weight updates (arXiv 2501.04971).

**Researcher summary:**
    arXiv 2501.04971 (Self-Adaptive Ising Machine for Constrained Optimization) proposes
    Lagrange relaxation of Ising constraint weights: when a constraint is violated, its
    coupling weight lambda_k increases proportionally to the violation severity.  After
    enough iterations the energy landscape is shaped so that the minimum-energy state
    satisfies all constraints.  This is adaptive self-learning: the system modifies its
    OWN coupling matrix based on what it gets wrong.

**How it works step by step:**
    1. Build J matrix from binary constraints, each constraint weighted by its current lambda.
    2. Run InertiaIsingSampler on the weighted J to collect n_samples spin configurations.
    3. For each constraint, measure how often samples violate it — the violation rate.
    4. Increase lambda proportionally to the violation rate (Lagrange update rule).
    5. Repeat: next session uses the updated lambdas, so violated constraints receive
       stronger coupling weights and the energy landscape shifts to penalise violations more.

**Why this matters for FR-11 (Autonomous Self-Learning):**
    FR-11 requires that the system improves its own performance across sessions without
    human intervention.  LagrangeAdaptiveIsingConstraints realises this for the Ising
    constraint layer: the sampler's energy landscape is modified by its own past failures,
    which is a pure self-learning signal with no external supervision required.

Spec: REQ-FR11-020
"""

from __future__ import annotations

import numpy as np

from carnot.samplers.inertia_ising import InertiaIsingSampler


class LagrangeAdaptiveIsingConstraints:
    """Multi-session Ising constraint solver with violation-driven lambda adaptation.

    **Detailed explanation for engineers:**
        Each constraint k specifies a pair of spins (i, j) that MUST be aligned
        (sign=+1: they should be equal) or anti-aligned (sign=-1: they should differ).
        The coupling J[i,j] encodes this preference; the penalty coefficient scales
        how strongly the constraint is enforced.

        ``lambda_k`` is the Lagrange multiplier for constraint k.  When the sampler
        violates constraint k, lambda_k grows.  A larger lambda_k increases the
        coupling strength for (i,j), making the energy penalty for violating that
        constraint steeper.  Over sessions, the sampler is forced to satisfy the
        constraint to stay near the energy minimum.

        The update rule is:
            lambda_k ← lambda_k + lambda_lr * avg_violation_rate_k

        where avg_violation_rate_k is the fraction of collected samples that violate
        constraint k.

    Args:
        n_spins: Number of binary spin variables (dimension of the Ising problem).
        n_constraints: Number of constraints to manage.
        lambda_init: Initial Lagrange multiplier for all constraints.  Default 1.0.
        lambda_lr: Learning rate for the lambda update.  Default 0.1.
            Larger values make the system react faster to violations but can overshoot.

    Spec: REQ-FR11-020
    """

    def __init__(
        self,
        n_spins: int,
        n_constraints: int,
        lambda_init: float = 1.0,
        lambda_lr: float = 0.1,
    ) -> None:
        self.n_spins = n_spins
        self.n_constraints = n_constraints
        self.lambda_lr = lambda_lr
        # Per-constraint Lagrange multipliers — grow when constraints are violated.
        self.lambdas: np.ndarray = np.ones(n_constraints, dtype=np.float64) * lambda_init
        # Base coupling matrix (zero here; constraints are injected via build_J).
        self.J_base: np.ndarray = np.zeros((n_spins, n_spins), dtype=np.float64)
        self.h: np.ndarray = np.zeros(n_spins, dtype=np.float64)

    def build_J(self, constraints: list[dict]) -> np.ndarray:
        """Build the coupling matrix J, weighted by the current lambda values.

        **Detailed explanation:**
            Each constraint specifies a pair of spins (i, j) and a sign (+1 or -1).
            - sign=+1 ("agree"): coupling J[i,j] is positive, so the sampler prefers
              s_i == s_j (both +1 or both -1 — low energy).
            - sign=-1 ("disagree"): coupling J[i,j] is negative, so the sampler prefers
              s_i != s_j — one +1, one -1.

            The lambda_k multiplier scales the coupling strength for constraint k.
            Higher lambda → stronger preference → harder to violate.

        Args:
            constraints: List of dicts, each with:
                - "spins": [i, j] spin indices (0-indexed)
                - "sign": +1 for agree constraint, -1 for disagree constraint
                - "penalty": float coupling magnitude (how strongly to enforce)

        Returns:
            np.ndarray of shape (n_spins, n_spins) — the weighted symmetric J matrix.

        Spec: REQ-FR11-020
        """
        J = self.J_base.copy()
        for k, c in enumerate(constraints):
            i, j = c["spins"]
            coupling = self.lambdas[k] * c["sign"] * c["penalty"]
            J[i, j] += coupling
            J[j, i] += coupling
        return J

    def run_session(
        self,
        constraints: list[dict],
        n_sweeps: int = 100,
        n_samples: int = 10,
    ) -> dict:
        """Run one adaptive session: sample, measure violations, update lambdas.

        **Detailed explanation:**
            One "session" corresponds to one round of the multi-session relay.
            The sequence is:
              1. Build J from current lambdas.
              2. Run InertiaIsingSampler to collect n_samples spin configurations.
              3. Count how often each constraint is violated across the samples.
              4. Update each lambda proportionally to its per-sample violation rate.

            After calling this method, self.lambdas has been updated.  The next session
            starts from the new lambdas, so the energy landscape is steeper around
            previously-violated constraints.

        Args:
            constraints: Same format as build_J — list of constraint dicts.
            n_sweeps: Number of Gibbs sweeps per sample run.  Default 100.
            n_samples: Number of spin configurations to collect.  Default 10.

        Returns:
            dict with keys:
              - "violation_rate": float in [0, 1] — fraction of (sample, constraint)
                pairs that violated at least one constraint (averaged over all constraints).
              - "lambdas": list of updated lambda values (one per constraint).
              - "per_constraint_violation_rates": list of per-constraint violation rates.

        Spec: REQ-FR11-020
        """
        J = self.build_J(constraints)
        sampler = InertiaIsingSampler(J, self.h)
        samples = sampler.sample(n_sweeps=n_sweeps, n_samples=n_samples)

        # Compute per-constraint violation rates.
        per_constraint_vr: list[float] = []
        for k, c in enumerate(constraints):
            avg_viol = self._constraint_violation(samples, c)
            per_constraint_vr.append(float(avg_viol))
            # Lagrange update: grow lambda for violated constraints.
            self.lambdas[k] += self.lambda_lr * avg_viol

        overall_violation_rate = float(np.mean(per_constraint_vr))

        return {
            "violation_rate": overall_violation_rate,
            "lambdas": self.lambdas.tolist(),
            "per_constraint_violation_rates": per_constraint_vr,
        }

    def _count_violations(self, samples: np.ndarray, constraints: list[dict]) -> np.ndarray:
        """Count total constraint violations per sample.

        **Detailed explanation:**
            For each spin configuration, counts how many of the K constraints are violated.
            A constraint (i, j, sign=+1) is violated when s_i != s_j.
            A constraint (i, j, sign=-1) is violated when s_i == s_j.

            The violation check: s_i * s_j == sign means they are in the preferred
            alignment.  If s_i * s_j != sign, the constraint is violated.

        Args:
            samples: (n_samples, n_spins) array of spin configurations in {-1, +1}.
            constraints: List of constraint dicts.

        Returns:
            np.ndarray of shape (n_samples,) — integer count of violations per sample.

        Spec: REQ-FR11-020
        """
        n_samples = samples.shape[0]
        violation_counts = np.zeros(n_samples, dtype=np.int64)
        for c in constraints:
            i, j = c["spins"]
            sign = c["sign"]
            # s_i * s_j == sign: preferred alignment.  != sign: violated.
            product = samples[:, i] * samples[:, j]
            violated = product != sign
            violation_counts += violated.astype(np.int64)
        return violation_counts

    def _constraint_violation(self, samples: np.ndarray, constraint: dict) -> float:
        """Compute the average violation rate of a single constraint across samples.

        **Detailed explanation:**
            Returns the fraction of samples in which this specific constraint is violated.
            This is the per-constraint signal used to update lambda.

        Args:
            samples: (n_samples, n_spins) array of spin configurations in {-1, +1}.
            constraint: Single constraint dict with "spins" and "sign".

        Returns:
            float in [0, 1] — fraction of samples violating this constraint.

        Spec: REQ-FR11-020
        """
        i, j = constraint["spins"]
        sign = constraint["sign"]
        product = samples[:, i] * samples[:, j]
        violated = product != sign
        return float(violated.mean())
