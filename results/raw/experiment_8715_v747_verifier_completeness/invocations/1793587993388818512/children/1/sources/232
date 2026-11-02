"""Multilinear relaxation for finite ASP energy tables.

Spec refs: REQ-KONA-6287, SCENARIO-KONA-6287-VERTEX-PARITY,
SCENARIO-KONA-6287-GRADIENT-CHECK, SCENARIO-KONA-6287-CONTROLS.

The relaxation is deliberately table based. Exp6274 already gives the trusted
finite discrete energy, so this module extends that finite function to
probabilities without learning a new model. The extension equals the discrete
energy at every binary vertex by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any

from carnot import asp_energy


class UnsupportedRelaxationFixture(ValueError):
    """Raised when a fixture exceeds the bounded finite relaxation contract."""

    def __init__(self, reason: str, fixture_id: str, atom_count: int, vertex_count: int) -> None:
        self.reason = reason
        self.fixture_id = fixture_id
        self.atom_count = atom_count
        self.vertex_count = vertex_count
        super().__init__(f"{reason}:{fixture_id}:atoms={atom_count}:vertices={vertex_count}")


@dataclass(frozen=True)
class VertexEnergyTable:
    """Finite discrete energy table used by the multilinear extension."""

    fixture_id: str
    atoms: tuple[str, ...]
    energies: tuple[int, ...]
    vertex_states: tuple[tuple[str, ...], ...]

    @property
    def atom_count(self) -> int:
        return len(self.atoms)

    @property
    def vertex_count(self) -> int:
        return len(self.energies)

    @property
    def best_discrete_energy(self) -> int:
        return min(self.energies)

    def state_for_mask(self, mask: int) -> tuple[str, ...]:
        return self.vertex_states[mask]

    def mask_for_state(self, state: Iterable[str]) -> int:
        atom_to_index = {atom: index for index, atom in enumerate(self.atoms)}
        mask = 0
        for atom in asp_energy.canonical_state(state):
            if atom not in atom_to_index:
                raise ValueError(f"unknown_atom:{atom}")
            mask |= 1 << atom_to_index[atom]
        return mask

    def discrete_energy(self, state: Iterable[str]) -> int:
        return self.energies[self.mask_for_state(state)]


def build_energy_table(
    compiled: asp_energy.CompiledASPProgram,
    *,
    fixture_id: str,
    max_atoms: int = 12,
    max_vertices: int = 4096,
) -> VertexEnergyTable:
    """Enumerate the finite ASP energy before any continuous relaxation."""

    atoms = tuple(compiled.program.atoms)
    vertex_count = 2 ** len(atoms)
    if len(atoms) > max_atoms or vertex_count > max_vertices:
        raise UnsupportedRelaxationFixture("vertex_bound", fixture_id, len(atoms), vertex_count)
    states: list[tuple[str, ...]] = []
    energies: list[int] = []
    for mask in range(vertex_count):
        state = tuple(atom for index, atom in enumerate(atoms) if mask & (1 << index))
        receipt = compiled.decompose_state(state)
        states.append(state)
        energies.append(int(receipt["total_energy"]))
    return VertexEnergyTable(
        fixture_id=fixture_id,
        atoms=atoms,
        energies=tuple(energies),
        vertex_states=tuple(states),
    )


def energy_at(probabilities: Sequence[float], table: VertexEnergyTable) -> float:
    """Evaluate the multilinear extension at one probability vector."""

    p = _checked_probabilities(probabilities, table)
    total = 0.0
    for mask, energy in enumerate(table.energies):
        total += float(energy) * _basis_weight(p, mask)
    return total


def gradient_at(probabilities: Sequence[float], table: VertexEnergyTable) -> list[float]:
    """Return the analytic gradient of the multilinear extension."""

    p = _checked_probabilities(probabilities, table)
    gradient: list[float] = []
    for target in range(table.atom_count):
        bit = 1 << target
        component = 0.0
        for mask in range(table.vertex_count):
            if mask & bit:
                continue
            weight = _basis_weight_without_index(p, mask, target)
            component += float(table.energies[mask | bit] - table.energies[mask]) * weight
        gradient.append(component)
    return gradient


def finite_difference_gradient(
    table: VertexEnergyTable,
    probabilities: Sequence[float],
    *,
    epsilon: float,
) -> list[float]:
    """Approximate the gradient with central differences inside the box."""

    p = list(_checked_probabilities(probabilities, table))
    if epsilon <= 0:
        raise ValueError("finite_difference_epsilon")
    if any(value - epsilon < 0.0 or value + epsilon > 1.0 for value in p):
        raise ValueError("finite_difference_boundary")
    gradient: list[float] = []
    for index in range(table.atom_count):
        plus = list(p)
        minus = list(p)
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient.append((energy_at(plus, table) - energy_at(minus, table)) / (2.0 * epsilon))
    return gradient


def check_gradient(
    table: VertexEnergyTable,
    probabilities: Sequence[float],
    *,
    epsilon: float,
    tolerance: float,
) -> dict[str, Any]:
    """Compare analytic gradients to finite differences for audit receipts."""

    analytic = gradient_at(probabilities, table)
    finite_difference = finite_difference_gradient(table, probabilities, epsilon=epsilon)
    errors = [abs(left - right) for left, right in zip(analytic, finite_difference, strict=True)]
    max_abs_error = max(errors, default=0.0)
    return {
        "fixture_id": table.fixture_id,
        "probabilities": [float(value) for value in probabilities],
        "analytic": analytic,
        "finite_difference": finite_difference,
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "passed": max_abs_error <= tolerance,
    }


def verify_vertex_parity(
    compiled: asp_energy.CompiledASPProgram,
    table: VertexEnergyTable,
) -> dict[str, Any]:
    """Check continuous and discrete energies on every binary vertex."""

    failures: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    for mask in range(table.vertex_count):
        state = table.state_for_mask(mask)
        probabilities = vertex_probability_vector(table, state)
        continuous = energy_at(probabilities, table)
        discrete = int(compiled.decompose_state(state)["total_energy"])
        delta = abs(continuous - discrete)
        max_abs_delta = max(max_abs_delta, delta)
        if delta != 0.0:
            failures.append(
                {
                    "state": list(state),
                    "continuous_energy": continuous,
                    "discrete_energy": discrete,
                    "abs_delta": delta,
                }
            )
    return {
        "fixture_id": table.fixture_id,
        "checked_vertices": table.vertex_count,
        "max_abs_delta": max_abs_delta,
        "failure_count": len(failures),
        "failures": failures,
        "parity_passed": not failures,
    }


def vertex_probability_vector(table: VertexEnergyTable, state: Iterable[str]) -> list[float]:
    """Convert a discrete atom state into a binary probability vector."""

    state_set = set(asp_energy.canonical_state(state))
    unknown = sorted(state_set - set(table.atoms))
    if unknown:
        raise ValueError(f"unknown_atom:{unknown[0]}")
    return [1.0 if atom in state_set else 0.0 for atom in table.atoms]


def round_probabilities(
    table: VertexEnergyTable,
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> list[str]:
    """Round probabilities to the nearest thresholded atom set."""

    p = _checked_probabilities(probabilities, table)
    return [atom for atom, value in zip(table.atoms, p, strict=True) if value >= threshold]


def refine(
    table: VertexEnergyTable,
    start: Sequence[float],
    *,
    steps: int,
    step_size: float,
) -> dict[str, Any]:
    """Run fixed-budget projected gradient descent in the probability box."""

    if steps < 0:
        raise ValueError("steps")
    if step_size <= 0.0:
        raise ValueError("step_size")
    current = list(_checked_probabilities(start, table))
    initial_energy = energy_at(current, table)
    energy_evaluations = 1
    for _ in range(steps):
        gradient = gradient_at(current, table)
        current = [_clip01(value - step_size * delta) for value, delta in zip(current, gradient)]
        energy_evaluations += 1
    final_gradient = gradient_at(current, table)
    final_energy = energy_at(current, table)
    energy_evaluations += 1
    return {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "final_probabilities": current,
        "final_gradient": final_gradient,
        "gradient_norm": _l2_norm(final_gradient),
        "projected_gradient_norm": projected_gradient_norm(current, final_gradient),
        "steps": steps,
        "step_size": step_size,
        "energy_evaluations": energy_evaluations,
    }


def stationary_point_record(
    table: VertexEnergyTable,
    probabilities: Sequence[float],
    *,
    gradient_tolerance: float,
    box_tolerance: float,
) -> dict[str, Any]:
    """Classify whether an interior fractional point is stationary."""

    p = list(_checked_probabilities(probabilities, table))
    gradient = gradient_at(p, table)
    gradient_norm = _l2_norm(gradient)
    fractional = all(box_tolerance < value < 1.0 - box_tolerance for value in p)
    return {
        "fixture_id": table.fixture_id,
        "probabilities": p,
        "energy": energy_at(p, table),
        "gradient": gradient,
        "gradient_norm": gradient_norm,
        "fractional": fractional,
        "stationary": fractional and gradient_norm <= gradient_tolerance,
    }


def projected_gradient_norm(
    probabilities: Sequence[float],
    gradient: Sequence[float],
    *,
    box_tolerance: float = 1e-9,
) -> float:
    """Measure first-order stationarity after accounting for box projection."""

    projected: list[float] = []
    for value, delta in zip(probabilities, gradient, strict=True):
        if value <= box_tolerance and delta > 0.0:
            projected.append(0.0)
        elif value >= 1.0 - box_tolerance and delta < 0.0:
            projected.append(0.0)
        else:
            projected.append(float(delta))
    return _l2_norm(projected)


def _checked_probabilities(
    probabilities: Sequence[float],
    table: VertexEnergyTable,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in probabilities)
    if len(values) != table.atom_count:
        raise ValueError("probability_length")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probability_bounds")
    return values


def _basis_weight(probabilities: Sequence[float], mask: int) -> float:
    weight = 1.0
    for index, value in enumerate(probabilities):
        weight *= value if mask & (1 << index) else 1.0 - value
    return weight


def _basis_weight_without_index(
    probabilities: Sequence[float],
    mask: int,
    target: int,
) -> float:
    weight = 1.0
    for index, value in enumerate(probabilities):
        if index == target:
            continue
        weight *= value if mask & (1 << index) else 1.0 - value
    return weight


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _l2_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))
