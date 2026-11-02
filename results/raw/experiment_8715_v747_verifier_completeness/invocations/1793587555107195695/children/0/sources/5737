"""InertiaIsingSampler — Ising sampler with EMA inertia term (arXiv 2604.17109).

**Researcher summary:**
    Standard Metropolis-Hastings on Ising problems suffers from slow mixing and
    spin-glass freezing: spins get trapped in local minima and the chain barely
    moves. The Fully Parallel Inertia Ising Machine (arXiv 2604.17109) introduces
    a per-spin exponential moving average (EMA) of each spin's recent history.
    This "inertia" term biases each spin's flip probability toward its recent
    trajectory, smoothing the energy landscape and allowing escapes from shallow
    local minima. This reduces mixing sweeps by 15-25x in simulation.

    Optionally, Mpemba initialization (arXiv 2603.24183) seeds spins from the
    leading eigenvector of J, the spectral-optimal magnetization. This starts
    the chain closer to a good solution, reducing thermalization time further.

**The inertia flip probability formula:**
    p_flip(s_i) = sigmoid(-2 * s_i * (h_eff_i + alpha * m_i))
    where:
      h_eff_i = h[i] + sum_j J[i,j] * s[j]   (effective local field)
      m_i     = EMA of s_i over recent sweeps  (inertia momentum)
      alpha   = EMA decay factor (0.5 typical)

    When alpha=0 this reduces to standard Gibbs/Metropolis-Hastings.

**Why this matters for code verification:**
    Carnot uses Ising EBMs to encode constraint satisfaction: spins represent
    variable assignments and J encodes which assignments are jointly valid.
    If the sampler mixes poorly, it cannot distinguish valid (low-energy)
    from invalid (high-energy) configurations — the discrimination_delta is
    near zero. InertiaIsingSampler fixes this by exploring the energy landscape
    more aggressively.

Spec: REQ-ISING-020
"""

from __future__ import annotations

import numpy as np


class InertiaIsingSampler:
    """Ising sampler augmented with EMA inertia term and optional Mpemba init.

    **Researcher summary:**
        Implements the per-spin inertia update from arXiv 2604.17109 with
        optional Mpemba spectral initialization from arXiv 2603.24183.

    **Detailed explanation for engineers:**
        Each spin s_i in {-1, +1} is updated by computing:
          1. h_eff_i: the effective local field (how much neighboring spins
             and the bias "push" on spin i).
          2. m_i: the EMA of spin i's recent values. If spin i has been +1
             for many sweeps, m_i is close to +1. This "inertia" makes spin i
             less likely to flip — it resists reversing a sustained trajectory,
             which prevents the random oscillations that cause slow mixing.
          3. p_flip: the probability that spin i flips. Computed with the
             inertia term added to the effective field.

        **Mpemba initialization:**
        Normally spins start random. The Mpemba effect (arXiv 2603.24183)
        shows that initializing spins from the leading eigenvector of J
        (the direction of maximum coupling energy) dramatically shortens
        the thermalization phase. This is like starting a greedy search
        at a good initial guess rather than a random point.

    Args:
        J: NxN coupling matrix (symmetric). J[i,j] encodes the energy cost
           of spins i and j being opposite.
        h: N-dimensional bias vector. h[i] encodes the energy bonus for
           spin i being +1.
        alpha: EMA momentum coefficient in [0, 1). Default 0.5. Higher values
               give more inertia (more memory of past spin values). alpha=0
               is equivalent to standard Gibbs sampling.
        use_mpemba: If True, initialize spins from the leading eigenvector
                    of J (spectral-optimal). If False, use random ±1 spins.

    Spec: REQ-ISING-020
    """

    def __init__(
        self,
        J: np.ndarray,
        h: np.ndarray,
        alpha: float = 0.5,
        use_mpemba: bool = True,
    ) -> None:
        self.J = np.asarray(J, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.alpha = float(alpha)
        self.use_mpemba = use_mpemba
        # EMA momentum per spin — starts at zero (no prior history)
        self.m = np.zeros(len(h), dtype=np.float64)

    def _mpemba_init(self) -> np.ndarray:
        """Initialize spins from the leading eigenvector of J (Mpemba init).

        **Detailed explanation:**
            The leading eigenvector of J points in the direction of maximum
            coupling energy. Taking the sign of each component gives a binary
            configuration that is spectrally close to the ground state.
            np.linalg.eigh returns eigenvalues in ascending order, so the
            last column is the leading (largest) eigenvector.

        Returns:
            np.ndarray of shape (N,) with values in {-1, +1}.

        Spec: REQ-ISING-020
        """
        _eigvals, eigvecs = np.linalg.eigh(self.J)
        leading = eigvecs[:, -1]  # leading eigenvector (largest eigenvalue)
        signs = np.sign(leading)
        # If any component is exactly zero (degenerate), assign +1 arbitrarily
        signs[signs == 0] = 1.0
        return signs.astype(np.float64)

    def sample(self, n_sweeps: int = 100, n_samples: int = 1) -> np.ndarray:
        """Run the inertia-augmented Gibbs sweep and collect samples.

        **Detailed explanation for engineers:**
            One "sweep" visits every spin exactly once in a random order.
            For each spin i:
              - Compute h_eff_i = h[i] + J[i,:] @ s (local field from neighbors)
              - Compute p_flip = sigmoid(-2 * s[i] * (h_eff_i + alpha * m[i]))
              - Flip spin with probability p_flip
              - Update EMA: m[i] = alpha * s[i] + (1 - alpha) * m[i]

            The last n_samples sweeps are stored as output samples. For most
            convergence benchmarking purposes n_samples=1 suffices.

        Args:
            n_sweeps: Total number of sweeps to run.
            n_samples: Number of samples to collect (from the final sweeps).

        Returns:
            np.ndarray of shape (n_samples, N) containing collected spin configs.

        Spec: REQ-ISING-020
        """
        N = len(self.h)
        rng = np.random.default_rng()

        # Reset EMA momentum for a fresh sampling run
        self.m = np.zeros(N, dtype=np.float64)

        if self.use_mpemba:
            s = self._mpemba_init()
        else:
            # Uniform random ±1 initialization
            s = (2 * rng.integers(0, 2, size=N) - 1).astype(np.float64)

        samples: list[np.ndarray] = []
        collect_from = n_sweeps - n_samples

        for sweep in range(n_sweeps):
            for i in rng.permutation(N):
                # Effective local field: bias + sum of neighbor coupling contributions
                h_eff = self.h[i] + self.J[i] @ s

                # Inertia augmented flip probability (arXiv 2604.17109, Eq. 3)
                # The -2 * s[i] factor means: if s[i] and (h_eff + inertia) agree
                # in sign, the argument to exp is large and negative → p_flip is small.
                logit = -2.0 * s[i] * (h_eff + self.alpha * self.m[i])
                p_flip = 1.0 / (1.0 + np.exp(-logit))  # sigmoid(logit)

                if rng.random() < p_flip:
                    s[i] = -s[i]

                # Update EMA: blends current spin value into running average
                self.m[i] = self.alpha * s[i] + (1.0 - self.alpha) * self.m[i]

            if sweep >= collect_from:
                samples.append(s.copy())

        return np.stack(samples)

    def energy(self, s: np.ndarray) -> float:
        """Compute Ising energy E(s) = -0.5 * s^T J s - h^T s.

        **Detailed explanation:**
            Low energy means the configuration is "valid" or preferred under the
            model. A correct code constraint encoding should have lower energy
            than an incorrect one when the sampler has mixed to the right region.

        Args:
            s: Spin configuration of shape (N,) with values in {-1, +1}.

        Returns:
            Scalar float energy.

        Spec: REQ-ISING-020
        """
        s = np.asarray(s, dtype=np.float64)
        return float(-0.5 * s @ self.J @ s - self.h @ s)

    def sweeps_to_convergence(self, threshold: float = 0.01) -> int:
        """Run until per-sweep magnetization change < threshold; return sweep count.

        **Detailed explanation:**
            "Magnetization" is the mean spin value: mean(s). When the chain has
            converged, spins stop changing significantly sweep-over-sweep and
            the magnetization stabilizes. This is a standard mixing diagnostic.

            We run at most 1000 sweeps to avoid infinite loops if the chain
            never converges (e.g., frustrated system). In that case we return
            the cap so the caller can detect it.

        Args:
            threshold: Maximum allowed change in mean magnetization per sweep.
                       Default 0.01 (1% of the ±1 range).

        Returns:
            Number of sweeps until convergence, or 1000 if cap reached.

        Spec: REQ-ISING-020
        """
        N = len(self.h)
        rng = np.random.default_rng(seed=42)

        self.m = np.zeros(N, dtype=np.float64)
        if self.use_mpemba:
            s = self._mpemba_init()
        else:
            s = (2 * rng.integers(0, 2, size=N) - 1).astype(np.float64)

        max_sweeps = 1000
        prev_mag = float(np.mean(s))

        for sweep in range(1, max_sweeps + 1):
            for i in rng.permutation(N):
                h_eff = self.h[i] + self.J[i] @ s
                logit = -2.0 * s[i] * (h_eff + self.alpha * self.m[i])
                p_flip = 1.0 / (1.0 + np.exp(-logit))
                if rng.random() < p_flip:
                    s[i] = -s[i]
                self.m[i] = self.alpha * s[i] + (1.0 - self.alpha) * self.m[i]

            mag = float(np.mean(s))
            if abs(mag - prev_mag) < threshold:
                return sweep
            prev_mag = mag

        return max_sweeps
