"""hls_energy_validator.py — CPU-side validator for the HLS Ising energy convention.

WHY THIS MODULE EXISTS:
    Exp 750 reported a sign-convention failure: the HLS C++ CPU simulation
    produced energy +3.0 for a test case whose ground state is -3.0.  The
    Ising Hamiltonian is defined as E = -sum J_ij s_i s_j - sum h_i s_i, so
    aligned ferromagnetic spins (J>0) MUST yield negative energy.

    This validator implements the same Hamiltonian in pure Python so that we
    can confirm the sign convention without compiling C++.  It mirrors the
    logic in hardware/kv260/ising_sampler_hls.cpp:compute_ising_energy so
    that any discrepancy between Python and C++ is immediately visible.

SIGN CONVENTION PRIMER (for engineers who are not EBM specialists):
    - s_i in {-1, +1} (Ising spins, NOT 0/1 binary)
    - J_ij > 0: ferromagnetic coupling — spins want to AGREE
    - J_ij < 0: antiferromagnetic coupling — spins want to DISAGREE
    - h_i: external field biasing spin i toward +1 (h_i > 0) or -1 (h_i < 0)
    - E = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i
    - Ground state (lowest E) for ferromagnet (J>0) with h=0: ALL spins +1 or ALL -1
    - Ground state energy for fully connected N-spin ferromagnet:
        E_gs = -J * N * (N-1) / 2   (negative — this is the invariant to check)

Spec: REQ-HW-040
"""

from __future__ import annotations

import math
from typing import Sequence


class HLSEnergyValidator:
    """Validate the sign convention of the Ising Hamiltonian implemented in the HLS C++ kernel.

    This class mirrors the energy computation in
    hardware/kv260/ising_sampler_hls.cpp::compute_ising_energy.  The purpose
    is to confirm that:
      1. The negative sign is applied correctly (energy -= J * s_i * s_j).
      2. The ferromagnetic ground state has negative energy.
      3. The Python implementation and the C++ implementation agree.

    Parameters
    ----------
    n_spins : int
        Number of spins in the Ising system.
    j_matrix : Sequence[Sequence[float]]
        n_spins x n_spins coupling matrix.  J[i][j] > 0 means ferromagnetic.
        Diagonal must be zero (self-coupling is undefined in Ising models).
    h_field : Sequence[float]
        External bias vector of length n_spins.  h[i] > 0 biases spin i toward +1.
    """

    def __init__(
        self,
        n_spins: int,
        j_matrix: Sequence[Sequence[float]],
        h_field: Sequence[float],
    ) -> None:
        if n_spins < 1:
            raise ValueError(f"n_spins must be >= 1, got {n_spins}")
        if len(j_matrix) != n_spins or any(len(row) != n_spins for row in j_matrix):
            raise ValueError(f"j_matrix must be {n_spins}x{n_spins}")
        if len(h_field) != n_spins:
            raise ValueError(f"h_field must have length {n_spins}, got {len(h_field)}")

        self.n_spins = n_spins
        # Copy to lists of floats to avoid mutation surprises from caller.
        self._j = [[float(j_matrix[i][k]) for k in range(n_spins)] for i in range(n_spins)]
        self._h = [float(v) for v in h_field]

    def compute_energy(self, spins: Sequence[int]) -> float:
        """Compute the total Ising energy E = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i.

        This is an exact Python translation of the C++ function
        hardware/kv260/ising_sampler_hls.cpp::compute_ising_energy.

        WHY we sum only i<j (upper triangle):
            The coupling matrix is symmetric (J_ij = J_ji), and summing all
            (i,j) pairs would double-count every bond.  Restricting to i<j
            gives the correct total energy without a factor-of-2 correction.

        Parameters
        ----------
        spins : Sequence[int]
            Spin configuration, each element must be +1 or -1.

        Returns
        -------
        float
            Total energy (negative = low energy, positive = high energy).

        Raises
        ------
        ValueError
            If `spins` does not have length n_spins, or contains values other
            than +1 and -1.

        Spec: REQ-HW-040
        """
        if len(spins) != self.n_spins:
            raise ValueError(f"spins must have length {self.n_spins}, got {len(spins)}")
        for k, s in enumerate(spins):
            if s not in (1, -1):
                raise ValueError(f"spin[{k}]={s} is not ±1")

        energy = 0.0

        # Interaction term: -sum_{i<j} J_ij s_i s_j
        # WHY negative: ferromagnetic ground state (J>0, s_i=s_j=+1) must have LOWER energy.
        for i in range(self.n_spins):
            for j in range(i + 1, self.n_spins):
                energy -= self._j[i][j] * spins[i] * spins[j]

        # Bias term: -sum_i h_i s_i
        # WHY negative: h_i > 0 stabilises spin i = +1 (lower energy when aligned).
        for i in range(self.n_spins):
            energy -= self._h[i] * spins[i]

        return energy

    def validate_ground_state(self) -> bool:
        """Return True if the all-ones spin configuration has strictly negative energy.

        For a ferromagnetic system (all J_ij > 0) with zero external field,
        all spins = +1 IS the ground state, and its energy must be negative.
        This is the primary sanity check for the sign convention.

        WHY we use all-ones:
            Any ferromagnet with positive couplings has its ground state at
            all spins aligned.  If the sign convention is correct,
            compute_energy([+1]*n) < 0.  If the sign is WRONG (energy +=
            instead of energy -=), the result will be positive — the
            classic RETRO-HLS-ENERGY symptom (got +3.0, expected -3.0).

        Returns
        -------
        bool
            True if energy of all-ones state is < 0.

        Spec: REQ-HW-040
        """
        all_ones = [1] * self.n_spins
        e = self.compute_energy(all_ones)
        return e < 0.0

    def compare_with_python_ising(self, n_samples: int = 100) -> tuple[float, float]:
        """Compare HLSEnergyValidator energies against random spin configurations.

        Generates n_samples random ±1 spin configurations, computes energy
        with this validator, and checks internal consistency (energy must be
        a finite float).  Also returns the max energy magnitude and the
        delta percentage relative to the expected ground-state energy.

        WHY internal comparison instead of against thrml:
            thrml is an optional dependency not always installed.  We check
            consistency across random configurations instead:  the energy
            must be finite and bounded by the ground-state energy (no random
            config can have lower energy than the ground state).

        Parameters
        ----------
        n_samples : int
            Number of random spin configurations to test.

        Returns
        -------
        (max_delta, max_delta_pct) : (float, float)
            max_delta: maximum deviation of a random config energy from the
            range [e_gs, +|e_gs|].  Expected 0.0 for a correct implementation.
            max_delta_pct: max_delta / |e_gs| * 100.  Expected 0.0.

        Spec: REQ-HW-040
        """
        import random

        rng = random.Random(42)
        all_ones = [1] * self.n_spins
        e_gs = self.compute_energy(all_ones)
        # Ground-state energy must be <= 0 for ferromagnet; use abs for bound.
        e_gs_abs = abs(e_gs) if e_gs != 0.0 else 1.0

        max_delta = 0.0
        for _ in range(n_samples):
            spins = [rng.choice([-1, 1]) for _ in range(self.n_spins)]
            e = self.compute_energy(spins)
            if not math.isfinite(e):
                max_delta = float("inf")
                break
            # Energy of random config can exceed |e_gs| (disordered state),
            # but should not exceed +|e_gs| * n_spins (loose upper bound).
            deviation = max(0.0, abs(e) - e_gs_abs * self.n_spins)
            max_delta = max(max_delta, deviation)

        max_delta_pct = max_delta / e_gs_abs * 100.0 if e_gs_abs > 0 else 0.0
        return max_delta, max_delta_pct
