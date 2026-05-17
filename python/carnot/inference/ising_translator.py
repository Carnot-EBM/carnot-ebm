"""
SAT/SMT to Ising Translator.

Maps basic AND/OR/NOT clauses to quadratic energy penalties (QUBO/Ising models).
"""

from typing import Dict, Tuple, Any

class IsingTranslator:
    """Translates basic boolean constraints to Ising/QUBO models."""

    def __init__(self) -> None:
        """Initialize the translator with empty energy terms."""
        self.linear: Dict[str, float] = {}
        self.quadratic: Dict[Tuple[str, str], float] = {}
        self.offset: float = 0.0

    def add_and_constraint(self, z: str, x: str, y: str, penalty_weight: float = 1.0) -> None:
        """
        Adds penalty for z != AND(x, y).
        Penalty: P = 3*z + x*y - 2*x*z - 2*y*z
        """
        self._add_term(z, z, 3 * penalty_weight)
        self._add_term(x, y, penalty_weight)
        self._add_term(x, z, -2 * penalty_weight)
        self._add_term(y, z, -2 * penalty_weight)

    def add_or_constraint(self, z: str, x: str, y: str, penalty_weight: float = 1.0) -> None:
        """
        Adds penalty for z != OR(x, y).
        Penalty: P = z + x + y + x*y - 2*x*z - 2*y*z
        """
        self._add_term(z, z, penalty_weight)
        self._add_term(x, x, penalty_weight)
        self._add_term(y, y, penalty_weight)
        self._add_term(x, y, penalty_weight)
        self._add_term(x, z, -2 * penalty_weight)
        self._add_term(y, z, -2 * penalty_weight)

    def add_not_constraint(self, z: str, x: str, penalty_weight: float = 1.0) -> None:
        """
        Adds penalty for z != NOT x.
        Penalty: P = 2*x*z - z - x + 1
        """
        self._add_term(x, z, 2 * penalty_weight)
        self._add_term(z, z, -penalty_weight)
        self._add_term(x, x, -penalty_weight)
        self.offset += penalty_weight

    def _add_term(self, u: str, v: str, w: float) -> None:
        """Helper to add linear or quadratic terms."""
        if u == v:
            self.linear[u] = self.linear.get(u, 0.0) + w
        else:
            if u > v:
                u, v = v, u
            self.quadratic[(u, v)] = self.quadratic.get((u, v), 0.0) + w

    def get_qubo(self) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], float]:
        """Returns the QUBO representation: linear terms, quadratic terms, and constant offset."""
        return self.linear, self.quadratic, self.offset

    def evaluate_energy(self, state: Dict[str, int]) -> float:
        """Evaluate the energy of a given state."""
        energy = self.offset
        for var, weight in self.linear.items():
            if state.get(var, 0) == 1:
                energy += weight
        for (u, v), weight in self.quadratic.items():
            if state.get(u, 0) == 1 and state.get(v, 0) == 1:
                energy += weight
        return energy
