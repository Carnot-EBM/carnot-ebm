"""GibbsWarmStart — Gibbs MCMC sampler with mean-field warm-start for Ising models.

**Why does cold-start fail for arbitration?**
    A Gibbs chain initialized from all-zero spins sits at the point of maximum
    symmetry: every spin is equally likely to be +1 or -1, so the chain measures
    nothing but initialization noise for the first ~100 sweeps.  At beta=1.0 and
    coupling std=0.01 (the default in MultiAgentArbiter), the mixing time is slow
    relative to the number of sweeps used before measurement (previously 0 — no
    MCMC was used at all, just direct energy evaluation).

    Mean-field warm-start fixes this by jumping directly to a spin configuration
    near a low-energy basin: each spin i is initialized to sign(h_i), which is the
    best single-spin decision given only the external field and ignoring coupling.
    From this starting point, 500 Gibbs sweeps efficiently explore the energy
    landscape near the initial basin, producing Boltzmann-distributed samples
    instead of initialization noise.

**Why 500 sweeps?**
    For n_spins=16 at beta=1.0, the theoretical mixing time is O(n log n) ≈ 45
    sweeps.  500 sweeps provides a 10x safety margin to ensure the chain has
    fully equilibrated regardless of the coupling strength.

Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-032
"""
from __future__ import annotations

import numpy as np


class GibbsWarmStart:
    """Gibbs sampler with mean-field warm-start for Ising energy landscapes.

    Implements the warm-start protocol from Exp 846: initialize spins from
    the mean-field approximation (sign of the external field h), then run
    Gibbs sweeps to reach equilibrium before measuring energy.

    The Ising Hamiltonian used here is:
        E(s) = -0.5 * s^T J s + h^T s

    Note the POSITIVE sign on h^T s.  This matches IsingConstraintInjector's
    convention where h >= 0 penalizes violation spins (s_i = +1), which is the
    opposite of the standard physics convention.

    Attributes:
        beta: Inverse temperature (default 1.0). Higher beta = sharper minima.
        seed: RNG seed for reproducibility.

    Spec: REQ-SAMPLE-020
    """

    def __init__(self, beta: float = 1.0, seed: int = 42) -> None:
        """Create a GibbsWarmStart sampler.

        Args:
            beta: Inverse temperature for the Boltzmann distribution. beta=1.0
                is the standard setting; higher values sharpen the minima.
            seed: RNG seed so results are reproducible across experiment runs.

        Spec: REQ-SAMPLE-020
        """
        self.beta = beta
        self.rng = np.random.default_rng(seed)

    def mf_init(self, h: np.ndarray) -> np.ndarray:
        """Initialize spin configuration from the mean-field approximation.

        Each spin is set to sign(h_i), which is the optimal single-spin value
        given only the external field (ignoring coupling to other spins).

        Why this choice: at zero magnetization (all spins = 0), the effective
        field on spin i is exactly h_i.  Setting s_i = sign(h_i) minimizes the
        field energy h_i * s_i in one shot, without any iterative solving.

        For h_i = 0: spin is assigned randomly (+1 or -1 with equal probability),
        since there is no preference from the external field alone.

        Args:
            h: External field vector of shape (n_spins,).

        Returns:
            Spin vector of shape (n_spins,) with values in {-1.0, +1.0}.

        Spec: REQ-SAMPLE-020
        """
        n = len(h)
        nonzero_mask = h != 0
        # sign(h_i) where h_i != 0, random ±1 where h_i = 0
        random_signs = self.rng.choice(np.array([-1.0, 1.0]), size=n)
        spins = np.where(nonzero_mask, np.sign(h), random_signs)
        return spins.astype(np.float64)

    def sweep(
        self, spins: np.ndarray, J: np.ndarray, h: np.ndarray
    ) -> np.ndarray:
        """Execute one Gibbs sweep: update every spin sequentially.

        For each spin i, we compute the conditional probability P(s_i = +1 | s_{-i})
        and sample from it.

        Derivation of the conditional:
            E(s) = -0.5 s^T J s + h^T s
            Energy contribution from spin i (all others fixed):
                E_i(s_i) = s_i * (h_i - sum_{j != i} J_ij * s_j) + constant
            Let h_eff_i = sum_{j != i} J_ij * s_j  (coupling field from neighbors)
            E_i = s_i * (h_i - h_eff_i) + constant
            P(s_i = +1) / P(s_i = -1) = exp(-2 * beta * (h_i - h_eff_i))
            P(s_i = +1) = sigmoid(2 * beta * (h_eff_i - h_i))

        Args:
            spins: Current spin configuration, shape (n_spins,), values ±1.
                   Modified in-place.
            J: Symmetric coupling matrix, shape (n_spins, n_spins).
            h: External field vector, shape (n_spins,).  Positive h_i PENALISES
               s_i = +1 (consistent with IsingConstraintInjector's convention).

        Returns:
            The updated spin array (same object as input, modified in-place).

        Spec: REQ-SAMPLE-020
        """
        n = len(spins)
        for i in range(n):
            # h_eff = coupling from all other spins (exclude self-coupling J[i,i])
            h_eff = float(J[i] @ spins) - J[i, i] * spins[i]
            # local_field > 0 favors s_i = +1; < 0 favors s_i = -1
            local_field = h_eff - h[i]
            prob_plus = 1.0 / (1.0 + np.exp(-2.0 * self.beta * local_field))
            spins[i] = 1.0 if self.rng.random() < prob_plus else -1.0
        return spins

    def warmup(
        self,
        J: np.ndarray,
        h: np.ndarray,
        n_sweeps: int = 500,
        s_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run Gibbs warm-up and return equilibrated spins + energy.

        Initializes from mean-field (or provided s_init), runs n_sweeps Gibbs
        sweeps to approach the stationary distribution, then measures the energy
        of the final configuration.

        Args:
            J: Coupling matrix, shape (n_spins, n_spins).
            h: External field, shape (n_spins,).
            n_sweeps: Number of Gibbs sweeps before measurement (default 500,
                per REQ-SAMPLE-020).  Set to 0 to measure from initial config
                without any sweeps (cold-start comparison baseline).
            s_init: Optional custom initial spins of shape (n_spins,).
                If None, uses mf_init(h) (mean-field initialization).

        Returns:
            Tuple (spins, energy):
                spins: Final spin configuration after n_sweeps.
                energy: E = -0.5 * s^T J s + h^T s at the final configuration.

        Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-032
        """
        spins = s_init.copy() if s_init is not None else self.mf_init(h)
        for _ in range(n_sweeps):
            self.sweep(spins, J, h)
        energy = float(-0.5 * spins @ J @ spins + h @ spins)
        return spins.copy(), energy
