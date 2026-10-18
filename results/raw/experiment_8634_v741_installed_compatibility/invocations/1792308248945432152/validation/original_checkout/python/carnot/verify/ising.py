"""Ising constraint verifier module."""

from typing import List

class IsingVerifier:
    """Ising energy constraint check for hardware verification."""

    def __init__(self, n_spins: int):
        self.n_spins = n_spins
        # Hardware sanity check defaults: all J_ij = 1.0, h_i = 0.0
        self.J = [[1.0 if i != j else 0.0 for j in range(n_spins)] for i in range(n_spins)]
        self.h = [0.0 for _ in range(n_spins)]

    def energy(self, state: List[int]) -> float:
        """Calculate Ising energy for a given spin state.

        Args:
            state: A list of spin values (+1 or -1).

        Returns:
            The total Ising energy.
        """
        if len(state) != self.n_spins:
            raise ValueError(f"Expected {self.n_spins} spins, got {len(state)}")
        
        e = 0.0
        # Field contribution: - sum h_i s_i
        for i in range(self.n_spins):
            e -= self.h[i] * state[i]
            
        # Coupling contribution: - sum_{i<j} J_ij s_i s_j
        for i in range(self.n_spins):
            for j in range(i + 1, self.n_spins):
                e -= self.J[i][j] * state[i] * state[j]
                
        return e
