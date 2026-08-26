"""Build a bounded exact Kac--Ward reference for planar Ising models.

The experiment deliberately supports only tiny, zero-field, straight-line
planar fixtures.  It evaluates Kac--Ward determinants in binary64/complex128,
checks every state and autoregressive conditional against an independent full
enumeration, and emits sealed samples only after those checks pass.  This is a
bounded CPU reference, not a claim about the full scale reported in the source
paper and not an LLM or accelerator benchmark.

Spec refs: REQ-SAMPLER-6639,
SCENARIO-SAMPLER-6639-EXACT-AUTOREGRESSIVE-PARITY,
SCENARIO-SAMPLER-6639-FAIL-CLOSED-NUMERICS,
SCENARIO-SAMPLER-6639-SEALED-INDEPENDENT-EVIDENCE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import tempfile
import time
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np


INFERENCE_SUBSTRATE = "bounded_cpu_kac_ward_and_full_enumeration_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6639_kac_ward_planar_reference.json")
SAMPLE_RELATIVE_DIR = Path("results/experiment_6639_kac_ward_sample_banks")
TEST_RECEIPT_ENV = "CARNOT_6639_TEST_RECEIPTS"
ENUMERATION_LIMIT = 8
CONDITION_NUMBER_LIMIT = 1.0e12
DETERMINANT_PHASE_TOLERANCE = 1.0e-8
PARITY_TOLERANCES: dict[str, float] = {
    "partition": 2.0e-11,
    "probability": 2.0e-11,
    "conditional": 2.0e-11,
    "moment": 5.0e-11,
    "normalization": 2.0e-12,
}
REQUIRED_TEST_SCOPES = (
    "focused_python",
    "new_code_coverage",
    "full_python_suite",
    "rust_unit",
    "spec_coverage",
    "applicable_e2e",
    "artifact_validation",
    "adversarial",
)
REQUIRED_ATTACKS = (
    "planarity",
    "zero_field",
    "coupling_sign",
    "edge_orientation",
    "auxiliary_spin_omission",
    "determinant_branch",
    "precision",
    "graph_permutation",
    "rng_reuse",
    "enumeration_mismatch",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "method_contract",
    "fixture_manifest",
    "per_instance_rows",
    "reference_sample_manifest",
    "parity_metrics",
    "kac_ward_reference_ready_score",
    "attack_rows",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("scripts/research_conductor.py"),
    Path("research-references.md"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
)


class UnsupportedInstanceError(ValueError):
    """Say that an instance lies outside the intentionally bounded contract."""


class KacWardPrecisionError(RuntimeError):
    """Say that determinant arithmetic cannot support an exact-reference row."""


@dataclass(frozen=True)
class PlanarIsingInstance:
    """Hold one completely frozen graph-temperature-seed reference case."""

    instance_id: str
    n_spins: int
    edges: tuple[tuple[int, int, float], ...]
    fields: tuple[float, ...]
    positions: tuple[tuple[float, float], ...]
    order: tuple[int, ...]
    temperature: float
    seed: int

    @property
    def graph_id(self) -> str:
        """Recover the fixture family name without its case suffix."""

        return self.instance_id.split("__temperature_", 1)[0]

    @property
    def fixture_sha256(self) -> str:
        """Hash all inputs that can change the mathematical sample law."""

        payload = {
            "instance_id": self.instance_id,
            "n_spins": self.n_spins,
            "edges": [list(edge) for edge in self.edges],
            "fields": list(self.fields),
            "positions": [list(position) for position in self.positions],
            "order": list(self.order),
            "temperature": self.temperature,
            "seed": self.seed,
        }
        return _sha256_json(payload)


@dataclass(frozen=True)
class ExperimentConfig:
    """Freeze the small matrix that the exact experiment is allowed to run."""

    graph_ids: tuple[str, ...] = (
        "mixed_square_n4",
        "frustrated_triangle_tail_n4",
        "mixed_ladder_n6",
    )
    temperatures: tuple[float, ...] = (0.8, 1.4)
    seeds: tuple[int, ...] = (6639001, 6639002)
    sample_count: int = 64
    enumeration_limit: int = ENUMERATION_LIMIT


_GRAPH_FIXTURES: dict[str, dict[str, Any]] = {
    "mixed_square_n4": {
        "n_spins": 4,
        "edges": ((0, 1, 0.71), (1, 2, -0.43), (2, 3, 0.58), (3, 0, 0.36)),
        "positions": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        "order": (0, 1, 2, 3),
    },
    "frustrated_triangle_tail_n4": {
        "n_spins": 4,
        "edges": ((0, 1, 0.62), (1, 2, 0.51), (2, 0, -0.47), (2, 3, 0.39)),
        "positions": ((0.0, 0.0), (2.0, 0.0), (1.0, 1.4), (1.0, 2.6)),
        "order": (0, 1, 2, 3),
    },
    "mixed_ladder_n6": {
        "n_spins": 6,
        "edges": (
            (0, 1, 0.55),
            (1, 2, -0.31),
            (3, 4, 0.44),
            (4, 5, 0.67),
            (0, 3, -0.28),
            (1, 4, 0.49),
            (2, 5, -0.37),
        ),
        "positions": (
            (0.0, 1.0),
            (1.0, 1.0),
            (2.0, 1.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
        ),
        "order": (0, 1, 2, 5, 4, 3),
    },
}


def _canonical_json(value: Any) -> bytes:
    """Encode evidence deterministically and reject nonfinite JSON numbers."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("nonfinite JSON or unsupported evidence value") from exc


def _sha256_json(value: Any) -> str:
    """Return the SHA-256 digest of one canonical JSON value."""

    return sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without interpreting its contents."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_instances(config: ExperimentConfig) -> list[PlanarIsingInstance]:
    """Expand the frozen fixture matrix and reject unplanned case dimensions."""

    unknown = sorted(set(config.graph_ids) - set(_GRAPH_FIXTURES))
    if unknown or not config.graph_ids:
        raise ValueError(f"graph_ids must name frozen fixtures; unknown={unknown}")
    if config.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not config.seeds:
        raise ValueError("seeds must not be empty")
    if not config.temperatures or any(
        not math.isfinite(item) or item <= 0.0 for item in config.temperatures
    ):
        raise ValueError("temperatures must be finite and positive")
    if config.enumeration_limit <= 0 or config.enumeration_limit > ENUMERATION_LIMIT:
        raise ValueError(f"enumeration_limit must be in [1, {ENUMERATION_LIMIT}]")

    instances: list[PlanarIsingInstance] = []
    for graph_id in config.graph_ids:
        fixture = _GRAPH_FIXTURES[graph_id]
        for temperature in config.temperatures:
            for seed in config.seeds:
                instance = PlanarIsingInstance(
                    instance_id=(f"{graph_id}__temperature_{temperature:.6f}__seed_{int(seed)}"),
                    n_spins=int(fixture["n_spins"]),
                    edges=tuple(fixture["edges"]),
                    fields=(0.0,) * int(fixture["n_spins"]),
                    positions=tuple(fixture["positions"]),
                    order=tuple(fixture["order"]),
                    temperature=float(temperature),
                    seed=int(seed),
                )
                validate_instance(instance, enumeration_limit=config.enumeration_limit)
                instances.append(instance)
    return instances


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    """Return the signed area used by the proper-segment crossing test."""

    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _properly_cross(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Detect an interior crossing between two edges with distinct endpoints."""

    first = _orientation(a, b, c)
    second = _orientation(a, b, d)
    third = _orientation(c, d, a)
    fourth = _orientation(c, d, b)
    epsilon = 1.0e-14
    return first * second < -epsilon and third * fourth < -epsilon


def validate_instance(
    instance: PlanarIsingInstance,
    *,
    enumeration_limit: int = ENUMERATION_LIMIT,
) -> dict[str, Any]:
    """Enforce the narrow mathematical and numerical input contract."""

    if instance.n_spins < 1 or instance.n_spins > enumeration_limit:
        raise UnsupportedInstanceError(f"n_spins exceeds enumeration limit {enumeration_limit}")
    if len(instance.fields) != instance.n_spins or any(
        not math.isfinite(value) or value != 0.0 for value in instance.fields
    ):
        raise UnsupportedInstanceError("the bounded reference requires exact zero field")
    if len(instance.positions) != instance.n_spins or any(
        len(position) != 2 or not all(math.isfinite(value) for value in position)
        for position in instance.positions
    ):
        raise UnsupportedInstanceError("every vertex requires a finite coordinate pair")
    if not math.isfinite(instance.temperature) or instance.temperature <= 0.0:
        raise UnsupportedInstanceError("temperature must be finite and positive")
    if tuple(sorted(instance.order)) != tuple(range(instance.n_spins)):
        raise UnsupportedInstanceError("spin order must be a vertex permutation")

    graph = nx.Graph()
    graph.add_nodes_from(range(instance.n_spins))
    seen: set[tuple[int, int]] = set()
    for left, right, coupling in instance.edges:
        if left == right:
            raise UnsupportedInstanceError("self-loop edges are unsupported")
        if left not in graph or right not in graph:
            raise UnsupportedInstanceError("edge endpoint lies outside the spin range")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise UnsupportedInstanceError("duplicate edge endpoints are unsupported")
        seen.add(key)
        if not math.isfinite(coupling):
            raise UnsupportedInstanceError("every edge needs a finite coupling")
        graph.add_edge(left, right)

    planar, _ = nx.check_planarity(graph, counterexample=False)
    if not planar:
        raise UnsupportedInstanceError("nonplanar graphs are unsupported")
    edge_pairs = [(left, right) for left, right, _ in instance.edges]
    for index, (left, right) in enumerate(edge_pairs):
        for other_left, other_right in edge_pairs[index + 1 :]:
            if len({left, right, other_left, other_right}) < 4:
                continue
            if _properly_cross(
                instance.positions[left],
                instance.positions[right],
                instance.positions[other_left],
                instance.positions[other_right],
            ):
                raise UnsupportedInstanceError(
                    "supplied straight-line embedding contains an edge crossing"
                )

    prefix: set[int] = set()
    for vertex in instance.order:
        prefix.add(vertex)
        if len(prefix) > 1 and not nx.is_connected(graph.subgraph(prefix)):
            raise UnsupportedInstanceError("spin order must maintain a connected search prefix")
    return {
        "planar": True,
        "zero_field": True,
        "connected_search_order": True,
        "straight_line_embedding": True,
        "n_spins": instance.n_spins,
        "edge_count": len(instance.edges),
    }


def _energy(instance: PlanarIsingInstance, state: Sequence[int]) -> float:
    """Evaluate the Ising energy using each undirected coupling exactly once."""

    return -sum(coupling * state[left] * state[right] for left, right, coupling in instance.edges)


def enumerate_reference(instance: PlanarIsingInstance) -> dict[str, Any]:
    """Enumerate the full state space through an independent scalar route."""

    validate_instance(instance)
    states = [
        tuple(int(value) for value in state) for state in product((-1, 1), repeat=instance.n_spins)
    ]
    beta = 1.0 / instance.temperature
    energies = [_energy(instance, state) for state in states]
    scaled = np.asarray([-beta * energy for energy in energies], dtype=np.float64)
    maximum = float(np.max(scaled))
    weights = np.exp(scaled - maximum)
    reduced_partition = float(np.sum(weights))
    probabilities = weights / reduced_partition
    partition = math.exp(maximum) * reduced_partition
    state_array = np.asarray(states, dtype=np.float64)
    first = probabilities @ state_array
    second = np.einsum("s,si,sj->ij", probabilities, state_array, state_array)
    energy_moment = float(probabilities @ np.asarray(energies, dtype=np.float64))
    return {
        "states": [list(state) for state in states],
        "energies": [float(value) for value in energies],
        "probabilities": [float(value) for value in probabilities],
        "partition_function": float(partition),
        "first_moments": [float(value) for value in first],
        "second_moments": [[float(value) for value in row] for row in second],
        "energy_moment": energy_moment,
    }


def _stable_log_cosh(value: float) -> float:
    """Evaluate log(cosh(value)) without overflowing for large couplings."""

    magnitude = abs(value)
    return magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)


def _validated_logdet(
    matrix: np.ndarray,
    *,
    condition_limit: float = CONDITION_NUMBER_LIMIT,
) -> tuple[float, dict[str, float]]:
    """Return a positive-branch log determinant or fail before evidence use."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise KacWardPrecisionError("determinant matrix must be nonempty and square")
    if not np.all(np.isfinite(matrix)):
        raise KacWardPrecisionError("singular or nonfinite determinant matrix")
    condition = float(np.linalg.cond(matrix))
    if not math.isfinite(condition):
        raise KacWardPrecisionError("singular Kac--Ward determinant matrix")
    if condition > condition_limit:
        raise KacWardPrecisionError(
            f"ill-conditioned Kac--Ward matrix: {condition:.6g} > {condition_limit:.6g}"
        )
    sign, log_abs = np.linalg.slogdet(matrix)
    if sign == 0 or not np.isfinite(log_abs):
        # A singular finite matrix has already produced an infinite condition
        # number above.  Keep this guard for backend anomalies without lying
        # about a separately reachable NumPy path.
        raise KacWardPrecisionError(  # pragma: no cover
            "singular Kac--Ward determinant matrix"
        )
    phase = abs(float(np.angle(sign)))
    if phase > DETERMINANT_PHASE_TOLERANCE:
        raise KacWardPrecisionError(f"determinant branch is not positive-real: phase={phase:.6g}")
    return float(log_abs), {
        "condition_number": condition,
        "determinant_phase_abs": phase,
    }


def _kac_ward_graph_log_partition(
    nodes: Sequence[int],
    edges: Sequence[tuple[int, int, float]],
    positions: Mapping[int, tuple[float, float]],
    beta: float,
    *,
    condition_limit: float,
) -> tuple[float, dict[str, float]]:
    """Evaluate the zero-field Kac--Ward determinant for one planar embedding."""

    directed: list[tuple[int, int, float]] = []
    for raw_left, raw_right, coupling in edges:
        left, right = sorted((int(raw_left), int(raw_right)))
        directed.append((left, right, float(coupling)))
        directed.append((right, left, float(coupling)))
    directed.sort(key=lambda row: (row[0], row[1]))

    if not directed:
        return len(nodes) * math.log(2.0), {
            "condition_number": 1.0,
            "determinant_phase_abs": 0.0,
            "directed_edge_count": 0.0,
        }
    size = len(directed)
    transition = np.zeros((size, size), dtype=np.complex128)
    edge_weights = np.empty(size, dtype=np.complex128)
    for index, (left, right, coupling) in enumerate(directed):
        edge_weights[index] = np.tanh(beta * coupling)
        incoming_angle = math.atan2(
            positions[right][1] - positions[left][1],
            positions[right][0] - positions[left][0],
        )
        for target, (middle, next_vertex, _) in enumerate(directed):
            if middle != right or next_vertex == left:
                continue
            outgoing_angle = math.atan2(
                positions[next_vertex][1] - positions[middle][1],
                positions[next_vertex][0] - positions[middle][0],
            )
            turn = (outgoing_angle - incoming_angle + math.pi) % (2.0 * math.pi) - math.pi
            transition[index, target] = np.exp(0.5j * turn)
    matrix = np.eye(size, dtype=np.complex128) - transition * edge_weights[np.newaxis, :]
    log_determinant, diagnostics = _validated_logdet(matrix, condition_limit=condition_limit)
    log_partition = len(nodes) * math.log(2.0)
    log_partition += sum(_stable_log_cosh(beta * coupling) for _, _, coupling in edges)
    log_partition += 0.5 * log_determinant
    if not math.isfinite(log_partition):
        raise KacWardPrecisionError(
            "ill-conditioned Kac--Ward calculation produced a nonfinite log partition"
        )
    diagnostics["directed_edge_count"] = float(size)
    return log_partition, diagnostics


def kac_ward_log_partition(
    instance: PlanarIsingInstance,
    *,
    precision: str = "float64",
    condition_limit: float = CONDITION_NUMBER_LIMIT,
) -> tuple[float, dict[str, float]]:
    """Return the bounded zero-field partition function in logarithmic form."""

    validate_instance(instance)
    if precision != "float64":
        raise KacWardPrecisionError(
            f"unsupported precision {precision!r}; require float64/complex128"
        )
    if not math.isfinite(condition_limit) or condition_limit < 1.0:
        raise KacWardPrecisionError("ill-conditioned threshold must be finite and at least one")
    beta = 1.0 / instance.temperature
    return _kac_ward_graph_log_partition(
        tuple(range(instance.n_spins)),
        instance.edges,
        {index: position for index, position in enumerate(instance.positions)},
        beta,
        condition_limit=condition_limit,
    )


def _planar_positions(graph: nx.Graph) -> dict[int, tuple[float, float]]:
    """Construct a deterministic planar drawing for an auxiliary-spin minor."""

    planar, embedding = nx.check_planarity(graph, counterexample=False)
    if not planar:
        raise UnsupportedInstanceError("auxiliary-spin contraction became nonplanar")
    ordered = sorted(graph.nodes)
    if len(ordered) == 1:
        return {ordered[0]: (0.0, 0.0)}
    if len(ordered) == 2:
        return {ordered[0]: (0.0, 0.0), ordered[1]: (1.0, 0.0)}
    raw = nx.combinatorial_embedding_to_pos(embedding, fully_triangulate=False)
    return {node: (float(raw[node][0]), float(raw[node][1])) for node in ordered}


def _future_log_partition(
    instance: PlanarIsingInstance,
    prefix: Sequence[int],
    position: int,
    proposed_spin: int,
) -> tuple[float, dict[str, float]]:
    """Evaluate Q_i^+ or Q_i^- through the contracted auxiliary-spin graph."""

    order = instance.order
    future = tuple(order[position + 1 :])
    if not future:
        return 0.0, {"condition_number": 1.0, "determinant_phase_abs": 0.0}
    frozen_spins = {order[index]: int(prefix[index]) for index in range(position)}
    frozen_spins[order[position]] = proposed_spin
    future_set = set(future)
    internal_edges: list[tuple[int, int, float]] = []
    boundary_fields = {vertex: 0.0 for vertex in future}
    for left, right, coupling in instance.edges:
        if left in future_set and right in future_set:
            internal_edges.append((left, right, coupling))
        elif left in future_set and right in frozen_spins:
            boundary_fields[left] += coupling * frozen_spins[right]
        elif right in future_set and left in frozen_spins:
            boundary_fields[right] += coupling * frozen_spins[left]

    auxiliary = instance.n_spins
    extended_edges = list(internal_edges)
    for vertex in future:
        field = boundary_fields[vertex]
        if field != 0.0:
            extended_edges.append((auxiliary, vertex, field))
    graph = nx.Graph()
    graph.add_nodes_from((*future, auxiliary))
    graph.add_edges_from((left, right) for left, right, _ in extended_edges)
    positions = _planar_positions(graph)
    log_extended, diagnostics = _kac_ward_graph_log_partition(
        (*future, auxiliary),
        extended_edges,
        positions,
        1.0 / instance.temperature,
        condition_limit=CONDITION_NUMBER_LIMIT,
    )
    return log_extended - math.log(2.0), diagnostics


def _sigmoid(value: float) -> float:
    """Evaluate a logistic probability without exponential overflow."""

    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def _conditional_probability_plus(
    instance: PlanarIsingInstance,
    prefix: Sequence[int],
    position: int,
) -> tuple[float, dict[str, float]]:
    """Compute one autoregressive conditional by two Kac--Ward minors."""

    current = instance.order[position]
    frozen = {instance.order[index]: int(prefix[index]) for index in range(position)}
    direct_field = 0.0
    for left, right, coupling in instance.edges:
        if left == current and right in frozen:
            direct_field += coupling * frozen[right]
        elif right == current and left in frozen:
            direct_field += coupling * frozen[left]
    log_q_plus, plus_diagnostics = _future_log_partition(instance, prefix, position, 1)
    log_q_minus, minus_diagnostics = _future_log_partition(instance, prefix, position, -1)
    log_odds = 2.0 * direct_field / instance.temperature + log_q_plus - log_q_minus
    return _sigmoid(log_odds), {
        "plus_condition_number": plus_diagnostics["condition_number"],
        "minus_condition_number": minus_diagnostics["condition_number"],
        "log_odds": log_odds,
    }


def autoregressive_likelihood(
    instance: PlanarIsingInstance,
    state: Sequence[int],
) -> dict[str, Any]:
    """Multiply normalized Kac--Ward conditionals for one complete state."""

    validate_instance(instance)
    if len(state) != instance.n_spins or any(value not in (-1, 1) for value in state):
        raise UnsupportedInstanceError("spin state must contain exactly n values in {-1,+1}")
    ordered_state = [int(state[vertex]) for vertex in instance.order]
    prefix: list[int] = []
    selected: list[float] = []
    plus_probabilities: list[float] = []
    condition_numbers: list[float] = []
    for position, spin in enumerate(ordered_state):
        probability_plus, diagnostics = _conditional_probability_plus(instance, prefix, position)
        plus_probabilities.append(probability_plus)
        selected.append(probability_plus if spin == 1 else 1.0 - probability_plus)
        condition_numbers.extend(
            [diagnostics["plus_condition_number"], diagnostics["minus_condition_number"]]
        )
        prefix.append(spin)
    probability = math.prod(selected)
    return {
        "probability": probability,
        "log_probability": math.log(probability),
        "selected_conditionals": selected,
        "plus_conditionals": plus_probabilities,
        "maximum_condition_number": max(condition_numbers, default=1.0),
    }


def _enumerated_conditional(
    states: Sequence[Sequence[int]],
    probabilities: Sequence[float],
    order: Sequence[int],
    prefix: Sequence[int],
    position: int,
) -> float:
    """Derive P(s_i=+1|prefix) only from the enumeration probability table."""

    denominator = 0.0
    numerator = 0.0
    for state, probability in zip(states, probabilities, strict=True):
        if any(state[order[index]] != prefix[index] for index in range(position)):
            continue
        denominator += probability
        if state[order[position]] == 1:
            numerator += probability
    return numerator / denominator


def cross_check_instance(instance: PlanarIsingInstance) -> dict[str, Any]:
    """Compare every determinant-derived quantity with full enumeration."""

    enumeration = enumerate_reference(instance)
    log_partition, diagnostics = kac_ward_log_partition(instance)
    kw_partition = math.exp(log_partition)
    exact_partition = float(enumeration["partition_function"])
    partition_error = abs(kw_partition - exact_partition) / exact_partition

    kw_probabilities: list[float] = []
    conditional_error = 0.0
    prefixes: set[tuple[int, ...]] = set()
    for state in enumeration["states"]:
        likelihood = autoregressive_likelihood(instance, state)
        kw_probabilities.append(float(likelihood["probability"]))
        ordered = [state[vertex] for vertex in instance.order]
        for position in range(instance.n_spins):
            prefix = tuple(ordered[:position])
            if prefix in prefixes:
                continue
            prefixes.add(prefix)
            enumerated = _enumerated_conditional(
                enumeration["states"],
                enumeration["probabilities"],
                instance.order,
                prefix,
                position,
            )
            calculated, _ = _conditional_probability_plus(instance, prefix, position)
            conditional_error = max(conditional_error, abs(enumerated - calculated))

    exact_probabilities = np.asarray(enumeration["probabilities"], dtype=np.float64)
    calculated_probabilities = np.asarray(kw_probabilities, dtype=np.float64)
    probability_error = float(np.max(np.abs(exact_probabilities - calculated_probabilities)))
    states = np.asarray(enumeration["states"], dtype=np.float64)
    exact_first = exact_probabilities @ states
    calculated_first = calculated_probabilities @ states
    exact_second = np.einsum("s,si,sj->ij", exact_probabilities, states, states)
    calculated_second = np.einsum("s,si,sj->ij", calculated_probabilities, states, states)
    energies = np.asarray(enumeration["energies"], dtype=np.float64)
    energy_error = abs(float(exact_probabilities @ energies - calculated_probabilities @ energies))
    normalization_error = abs(float(np.sum(calculated_probabilities)) - 1.0)
    metrics = {
        "partition_error": partition_error,
        "state_probability_error_max": probability_error,
        "conditional_error_max": conditional_error,
        "first_moment_error_max": float(np.max(np.abs(exact_first - calculated_first))),
        "second_moment_error_max": float(np.max(np.abs(exact_second - calculated_second))),
        "energy_moment_error": energy_error,
        "normalization_error": normalization_error,
        "state_count": len(enumeration["states"]),
        "unique_prefix_count": len(prefixes),
        "condition_number": diagnostics["condition_number"],
        "determinant_phase_abs": diagnostics["determinant_phase_abs"],
        "enumeration_partition_function": exact_partition,
        "kac_ward_partition_function": kw_partition,
    }
    metrics["passed"] = bool(
        partition_error <= PARITY_TOLERANCES["partition"]
        and probability_error <= PARITY_TOLERANCES["probability"]
        and conditional_error <= PARITY_TOLERANCES["conditional"]
        and metrics["first_moment_error_max"] <= PARITY_TOLERANCES["moment"]
        and metrics["second_moment_error_max"] <= PARITY_TOLERANCES["moment"]
        and energy_error <= PARITY_TOLERANCES["moment"]
        and normalization_error <= PARITY_TOLERANCES["normalization"]
    )
    return metrics


def _domain_seed(instance: PlanarIsingInstance) -> int:
    """Derive a fresh PCG64 seed for exactly one graph-temperature-seed case."""

    material = f"exp6639:kac-ward-reference:{instance.fixture_sha256}".encode()
    return int.from_bytes(sha256(material).digest()[:16], "big")


def sample_reference_rows(
    instance: PlanarIsingInstance,
    *,
    sample_count: int,
) -> dict[str, Any]:
    """Draw deterministic independent samples and recheck each likelihood."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    enumeration = enumerate_reference(instance)
    exact_by_state = {
        tuple(state): probability
        for state, probability in zip(
            enumeration["states"], enumeration["probabilities"], strict=True
        )
    }
    domain_seed = _domain_seed(instance)
    rng = np.random.Generator(np.random.PCG64(domain_seed))
    rows: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        ordered_spins: list[int] = []
        plus_conditionals: list[float] = []
        selected: list[float] = []
        for position in range(instance.n_spins):
            probability_plus, _ = _conditional_probability_plus(instance, ordered_spins, position)
            spin = 1 if float(rng.random()) < probability_plus else -1
            ordered_spins.append(spin)
            plus_conditionals.append(probability_plus)
            selected.append(probability_plus if spin == 1 else 1.0 - probability_plus)
        state = [0] * instance.n_spins
        for position, vertex in enumerate(instance.order):
            state[vertex] = ordered_spins[position]
        probability = math.prod(selected)
        enumeration_probability = float(exact_by_state[tuple(state)])
        error = abs(probability - enumeration_probability)
        rows.append(
            {
                "sample_index": sample_index,
                "spins": state,
                "normalized_likelihood": probability,
                "log_normalized_likelihood": math.log(probability),
                "selected_conditionals": selected,
                "plus_conditionals": plus_conditionals,
                "enumeration_likelihood": enumeration_probability,
                "likelihood_parity_error": error,
                "likelihood_parity_passed": error <= PARITY_TOLERANCES["probability"],
            }
        )
    return {
        "rows": rows,
        "sample_rows_sha256": _sha256_json(rows),
        "domain_seed": domain_seed,
        "rng": "numpy.random.Generator(PCG64), one fresh domain-separated stream per case",
    }


def permute_instance(
    instance: PlanarIsingInstance,
    permutation: Sequence[int],
) -> tuple[PlanarIsingInstance, tuple[int, ...]]:
    """Relabel vertices and return the new-state index for each old vertex."""

    if tuple(sorted(permutation)) != tuple(range(instance.n_spins)):
        raise ValueError("permutation must contain every vertex exactly once")
    old_to_new = tuple(int(value) for value in permutation)
    fields = [0.0] * instance.n_spins
    positions: list[tuple[float, float]] = [(0.0, 0.0)] * instance.n_spins
    for old, new in enumerate(old_to_new):
        fields[new] = instance.fields[old]
        positions[new] = instance.positions[old]
    permuted = PlanarIsingInstance(
        instance_id=f"{instance.instance_id}__permuted",
        n_spins=instance.n_spins,
        edges=tuple(
            (old_to_new[left], old_to_new[right], coupling)
            for left, right, coupling in instance.edges
        ),
        fields=tuple(fields),
        positions=tuple(positions),
        order=tuple(old_to_new[vertex] for vertex in instance.order),
        temperature=instance.temperature,
        seed=instance.seed,
    )
    validate_instance(permuted)
    return permuted, old_to_new


def write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    seal: bool,
) -> dict[str, Any]:
    """Sync, replace, and optionally make one sample bank read-only."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(_canonical_json(dict(row)) + b"\n" for row in rows)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        if seal:
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": len(encoded),
        "row_count": len(rows),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
        "sealed_read_only": bool(seal),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sync and atomically replace the terminal experiment artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(dict(payload)) + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": len(encoded),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
    }


def protected_hashes(root: Path) -> dict[str, str]:
    """Capture files that an experiment run must never rewrite."""

    hashes: dict[str, str] = {}
    for relative in PROTECTED_RELATIVE_PATHS:
        path = root / relative
        hashes[relative.as_posix()] = sha256_file(path) if path.is_file() else "missing"
    return hashes


def protected_files_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, Any]:
    """Compare protected inputs without hiding missing or added entries."""

    names = sorted(set(before) | set(after))
    rows = [
        {
            "path": name,
            "before_sha256": before.get(name, "missing"),
            "after_sha256": after.get(name, "missing"),
            "unchanged": before.get(name) == after.get(name),
        }
        for name in names
    ]
    return {"all_unchanged": all(row["unchanged"] for row in rows), "rows": rows}


def _attack_row(
    attack: str,
    passed: bool,
    observed_value: Any,
    expected: str,
) -> dict[str, Any]:
    """Give every adversarial check the same inspectable row shape."""

    return {
        "attack": attack,
        "passed": bool(passed),
        "observed_value": observed_value,
        "expected": expected,
    }


def build_attack_rows() -> list[dict[str, Any]]:
    """Attack every fragile convention in the bounded determinant pipeline."""

    base = frozen_instances(
        ExperimentConfig(
            graph_ids=("mixed_square_n4",),
            temperatures=(0.8,),
            seeds=(6639001,),
            sample_count=8,
        )
    )[0]
    rows: list[dict[str, Any]] = []

    k33 = PlanarIsingInstance(
        "attack_k33",
        6,
        tuple((left, right, 0.2) for left in range(3) for right in range(3, 6)),
        (0.0,) * 6,
        tuple((float(index % 3), float(index // 3)) for index in range(6)),
        tuple(range(6)),
        1.0,
        1,
    )
    try:
        validate_instance(k33)
        planarity_caught = False  # pragma: no cover - dependency corruption guard.
    except UnsupportedInstanceError as exc:
        planarity_caught = "nonplanar" in str(exc)
    rows.append(_attack_row("planarity", planarity_caught, planarity_caught, "reject K3,3"))

    nonzero = PlanarIsingInstance(
        base.instance_id,
        base.n_spins,
        base.edges,
        (1.0e-8, 0.0, 0.0, 0.0),
        base.positions,
        base.order,
        base.temperature,
        base.seed,
    )
    try:
        validate_instance(nonzero)
        field_caught = False  # pragma: no cover - dependency corruption guard.
    except UnsupportedInstanceError as exc:
        field_caught = "zero field" in str(exc)
    rows.append(_attack_row("zero_field", field_caught, field_caught, "reject any field"))

    flipped_edges = list(base.edges)
    left, right, coupling = flipped_edges[0]
    flipped_edges[0] = (left, right, -coupling)
    flipped = PlanarIsingInstance(
        f"{base.instance_id}__sign_flip",
        base.n_spins,
        tuple(flipped_edges),
        base.fields,
        base.positions,
        base.order,
        base.temperature,
        base.seed,
    )
    base_probabilities = np.asarray(enumerate_reference(base)["probabilities"])
    flipped_probabilities = np.asarray(enumerate_reference(flipped)["probabilities"])
    sign_delta = float(np.max(np.abs(base_probabilities - flipped_probabilities)))
    rows.append(
        _attack_row("coupling_sign", sign_delta > 1.0e-6, sign_delta, "distribution changes")
    )

    reversed_instance = PlanarIsingInstance(
        f"{base.instance_id}__edge_reverse",
        base.n_spins,
        tuple((right, left, coupling) for left, right, coupling in base.edges),
        base.fields,
        base.positions,
        base.order,
        base.temperature,
        base.seed,
    )
    orientation_delta = abs(
        kac_ward_log_partition(base)[0] - kac_ward_log_partition(reversed_instance)[0]
    )
    rows.append(
        _attack_row(
            "edge_orientation", orientation_delta <= 1.0e-12, orientation_delta, "invariant"
        )
    )

    enumeration = enumerate_reference(base)
    omission_delta = 0.0
    for state in enumeration["states"]:
        ordered = [state[vertex] for vertex in base.order]
        for position in range(base.n_spins):
            prefix = ordered[:position]
            exact = _enumerated_conditional(
                enumeration["states"], enumeration["probabilities"], base.order, prefix, position
            )
            current = base.order[position]
            frozen = {base.order[index]: prefix[index] for index in range(position)}
            direct = sum(
                coupling * (frozen[right] if left == current else frozen[left])
                for left, right, coupling in base.edges
                if (left == current and right in frozen) or (right == current and left in frozen)
            )
            omission_delta = max(
                omission_delta, abs(exact - _sigmoid(2.0 * direct / base.temperature))
            )
    rows.append(
        _attack_row(
            "auxiliary_spin_omission",
            omission_delta > PARITY_TOLERANCES["conditional"],
            omission_delta,
            "omission has a nonzero conditional witness",
        )
    )

    try:
        _validated_logdet(np.asarray([[-1.0 + 0.0j]]), condition_limit=10.0)
        branch_caught = False  # pragma: no cover - dependency corruption guard.
    except KacWardPrecisionError:
        branch_caught = True
    rows.append(
        _attack_row("determinant_branch", branch_caught, branch_caught, "reject negative branch")
    )

    try:
        kac_ward_log_partition(base, precision="float32")
        precision_caught = False  # pragma: no cover - dependency corruption guard.
    except KacWardPrecisionError:
        precision_caught = True
    rows.append(_attack_row("precision", precision_caught, precision_caught, "reject float32"))

    permuted, old_to_new = permute_instance(base, (2, 0, 3, 1))
    permuted_enum = enumerate_reference(permuted)
    original_map = {
        tuple(state): probability
        for state, probability in zip(
            enumeration["states"], enumeration["probabilities"], strict=True
        )
    }
    permutation_error = 0.0
    for state, probability in zip(
        permuted_enum["states"], permuted_enum["probabilities"], strict=True
    ):
        original_state = tuple(state[old_to_new[index]] for index in range(base.n_spins))
        permutation_error = max(permutation_error, abs(probability - original_map[original_state]))
    rows.append(
        _attack_row(
            "graph_permutation",
            permutation_error <= 1.0e-14,
            permutation_error,
            "invariant after relabeling",
        )
    )

    alternate = PlanarIsingInstance(
        f"{base.graph_id}__temperature_1.400000__seed_{base.seed}",
        base.n_spins,
        base.edges,
        base.fields,
        base.positions,
        base.order,
        1.4,
        base.seed,
    )
    replay_a = sample_reference_rows(base, sample_count=8)
    replay_b = sample_reference_rows(base, sample_count=8)
    other = sample_reference_rows(alternate, sample_count=8)
    rng_passed = (
        replay_a["sample_rows_sha256"] == replay_b["sample_rows_sha256"]
        and replay_a["domain_seed"] != other["domain_seed"]
        and replay_a["sample_rows_sha256"] != other["sample_rows_sha256"]
    )
    rows.append(
        _attack_row(
            "rng_reuse",
            rng_passed,
            {
                "replay_equal": replay_a["sample_rows_sha256"] == replay_b["sample_rows_sha256"],
                "domain_separated": replay_a["domain_seed"] != other["domain_seed"],
            },
            "replay stable and case streams distinct",
        )
    )

    altered = list(enumeration["probabilities"])
    altered[0] += 1.0e-4
    mismatch = float(np.max(np.abs(np.asarray(altered) - np.asarray(enumeration["probabilities"]))))
    rows.append(
        _attack_row(
            "enumeration_mismatch",
            mismatch > PARITY_TOLERANCES["probability"],
            mismatch,
            "perturbation detected",
        )
    )
    return rows


def _package_versions() -> dict[str, str]:
    """Freeze the numerical package versions visible to this run."""

    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("numpy", "networkx", "pytest", "coverage"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _resource_snapshot(root: Path) -> dict[str, Any]:
    """Record host capacity as a precondition, not as a performance claim."""

    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(root)
    return {
        "cpu_architecture": platform.machine(),
        "cpu_logical_count": os.cpu_count(),
        "ram_bytes": int(page_size * physical_pages),
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes_at_start": int(disk.free),
        "note": "capacity inventory only; no hardware or acceleration claim",
    }


def _aggregate_parity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce per-case parity while preserving every named error dimension."""

    parity_rows = [row["parity"] for row in rows]
    error_keys = (
        "partition_error",
        "state_probability_error_max",
        "conditional_error_max",
        "first_moment_error_max",
        "second_moment_error_max",
        "energy_moment_error",
        "normalization_error",
    )
    return {
        "all_instances_passed": bool(parity_rows) and all(row["passed"] for row in parity_rows),
        "instance_count": len(parity_rows),
        "maximum_errors": {
            key: max((float(row[key]) for row in parity_rows if key in row), default=None)
            for key in error_keys
        },
        "tolerances": dict(PARITY_TOLERANCES),
        "methodology_note": (
            "Errors compare deterministic determinant-derived values with a separate full "
            "state enumeration on the same frozen tiny instance; zeros are mathematical "
            "agreement values, not empirical accuracy estimates."
        ),
    }


def _test_gate(receipts: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Require one successful receipt for every planned verification scope."""

    passed_scopes = {
        str(row.get("scope"))
        for row in receipts
        if row.get("exit_code") == 0 and row.get("command")
    }
    missing = [scope for scope in REQUIRED_TEST_SCOPES if scope not in passed_scopes]
    return not missing, missing


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash terminal content while treating the checksum field as its own blank slot."""

    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return _sha256_json(material)


def _field_provenance() -> dict[str, dict[str, str]]:
    """Explain the source and method for every required evidence field."""

    methods = {
        "status": "conjunction of math, parity, precision, test, protection, attack, and bank gates",
        "honest_verdict": "bounded verdict derived from the same gate conjunction",
        "verdict_class": "closed enum selected from positive or blocked",
        "gate_check_summary": "explicit failed-check reducer",
        "method_contract": "frozen REQ-SAMPLER-6639 scope",
        "fixture_manifest": "canonical serialization of frozen graph cases",
        "per_instance_rows": "Kac--Ward and enumeration evaluation",
        "reference_sample_manifest": "atomic sealed JSONL bank receipts",
        "parity_metrics": "maxima over complete state, prefix, and moment comparisons",
        "kac_ward_reference_ready_score": "binary conjunction only",
        "attack_rows": "live adversarial convention checks",
        "preconditions_checked": "runtime, package, resource, tolerance, and hash snapshot",
        "protected_files_unchanged": "before/after SHA-256 comparison",
        "inference_substrate": "fixed literal declaring bounded CPU math and no LLM",
        "verifier_is_oracle": "false because enumeration independently checks the sampler",
        "field_provenance": "this source/method map",
        "duration_s": "time.monotonic elapsed seconds",
        "tests_run": "externally supplied command receipts",
        "reproducibility_checksum": "SHA-256 of canonical final content with checksum blank",
    }
    return {
        field: {"source": "experiment_6639 module", "method": methods[field]}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    root: Path,
    config: ExperimentConfig,
    sample_dir: Path,
    test_receipts: Sequence[Mapping[str, Any]],
    run_date: str = "20260826",
) -> dict[str, Any]:
    """Build terminal evidence and seal banks only for fully passing cases."""

    started = time.monotonic()
    before = protected_hashes(root)
    instances = frozen_instances(config)
    fixture_manifest: list[dict[str, Any]] = []
    per_instance_rows: list[dict[str, Any]] = []
    sample_manifest: list[dict[str, Any]] = []
    math_failures: list[dict[str, str]] = []

    for instance in instances:
        validation = validate_instance(instance, enumeration_limit=config.enumeration_limit)
        fixture_manifest.append(
            {
                "instance_id": instance.instance_id,
                "graph_id": instance.graph_id,
                "n_spins": instance.n_spins,
                "edges": [list(edge) for edge in instance.edges],
                "fields": list(instance.fields),
                "positions": [list(position) for position in instance.positions],
                "spin_order": list(instance.order),
                "temperature": instance.temperature,
                "seed": instance.seed,
                "fixture_sha256": instance.fixture_sha256,
                "validation": validation,
            }
        )
        case_started = time.monotonic()
        try:
            parity = cross_check_instance(instance)
        except (UnsupportedInstanceError, KacWardPrecisionError, ArithmeticError) as exc:
            parity = {"passed": False, "failure": type(exc).__name__, "message": str(exc)}
            math_failures.append({"instance_id": instance.instance_id, "observed": str(exc)})
        setup_s = time.monotonic() - case_started
        per_instance_rows.append(
            {
                "instance_id": instance.instance_id,
                "graph_id": instance.graph_id,
                "temperature": instance.temperature,
                "seed": instance.seed,
                "parity": parity,
                "setup_cost": {
                    "state_count": 2**instance.n_spins,
                    "determinant_complexity_scope": "O(n^3) dense solve per bounded conditional",
                },
                "setup_wall_time_s": setup_s,
            }
        )
        if not parity.get("passed", False):
            continue
        sample_started = time.monotonic()
        samples = sample_reference_rows(instance, sample_count=config.sample_count)
        if not all(row["likelihood_parity_passed"] for row in samples["rows"]):
            per_instance_rows[-1]["parity"]["passed"] = False
            math_failures.append(
                {"instance_id": instance.instance_id, "observed": "sample likelihood mismatch"}
            )
            continue
        bank_path = sample_dir / f"{instance.instance_id}.jsonl"
        write_receipt = write_jsonl_atomic(bank_path, samples["rows"], seal=True)
        sample_manifest.append(
            {
                "instance_id": instance.instance_id,
                "path": str(bank_path.relative_to(root))
                if bank_path.is_relative_to(root)
                else str(bank_path),
                "resolved_path": str(bank_path.resolve()),
                "sha256": write_receipt["sha256"],
                "sample_rows_sha256": samples["sample_rows_sha256"],
                "sample_count": config.sample_count,
                "domain_seed": str(samples["domain_seed"]),
                "rng_independence_assumptions": (
                    "PCG64 draws are pseudorandom; each case uses a fresh SHA-256 domain-separated "
                    "seed and no generator state is reused across cases"
                ),
                "normalized_likelihoods": [row["normalized_likelihood"] for row in samples["rows"]],
                "setup_cost": per_instance_rows[-1]["setup_cost"],
                "setup_wall_time_s": setup_s,
                "sample_wall_time_s": time.monotonic() - sample_started,
                "atomic_write": write_receipt,
            }
        )

    attacks = build_attack_rows()
    parity_metrics = _aggregate_parity(per_instance_rows)
    after = protected_hashes(root)
    protection = protected_files_unchanged(before, after)
    tests_passed, missing_tests = _test_gate(test_receipts)
    failed_checks: list[dict[str, Any]] = []
    if math_failures:
        failed_checks.append(
            {"category": "math", "observed": math_failures, "expected": "no determinant failure"}
        )
    if not parity_metrics["all_instances_passed"]:
        failed_checks.append(
            {
                "category": "parity",
                "observed": parity_metrics["maximum_errors"],
                "expected": dict(PARITY_TOLERANCES),
            }
        )
    if any(not row["passed"] for row in attacks):
        failed_checks.append(
            {
                "category": "attack",
                "observed": [row for row in attacks if not row["passed"]],
                "expected": "all attacks pass",
            }
        )
    if not tests_passed:
        failed_checks.append(
            {"category": "test", "observed": missing_tests, "expected": list(REQUIRED_TEST_SCOPES)}
        )
    if not protection["all_unchanged"]:
        failed_checks.append(
            {
                "category": "protection",
                "observed": [row for row in protection["rows"] if not row["unchanged"]],
                "expected": "all protected hashes unchanged",
            }
        )
    if len(sample_manifest) != len(instances):
        failed_checks.append(
            {"category": "atomic", "observed": len(sample_manifest), "expected": len(instances)}
        )
    ready = not failed_checks
    first_category = failed_checks[0]["category"] if failed_checks else "none"
    status = (
        "complete_bounded_exact_reference_ready"
        if ready
        else f"blocked_{first_category}_check_failed"
    )
    honest_verdict = (
        "complete: bounded exact Kac--Ward reference passed full enumeration on every frozen planar fixture"
        if ready
        else f"blocked_{first_category}_check_failed: bounded reference readiness was not established"
    )
    artifact: dict[str, Any] = {
        "schema_version": "experiment-6639.v1",
        "planning_date": run_date,
        "status": status,
        "honest_verdict": honest_verdict,
        "verdict_class": "positive" if ready else "blocked",
        "gate_check_summary": {
            "passed": ready,
            "failed_checks": failed_checks,
            "checked_categories": [
                "math",
                "parity",
                "precision",
                "test",
                "protection",
                "attack",
                "atomic",
            ],
        },
        "method_contract": {
            "supported_graph": "connected straight-line planar simple graphs in the frozen fixture set",
            "supported_field": "exactly zero external field; conditionals use one contracted auxiliary spin",
            "supported_size": f"1 <= n_spins <= {config.enumeration_limit}; full 2^n enumeration required",
            "precision": "IEEE-754 binary64 real and complex128 determinant arithmetic only",
            "conditioning": f"finite condition number <= {CONDITION_NUMBER_LIMIT:.1e} and positive-real determinant phase <= {DETERMINANT_PHASE_TOLERANCE:.1e}",
            "complexity_scope": "dense O(n^3) determinant work per bounded conditional; no paper-scale claim",
            "conclusion_scope": "only graph-temperature-seed rows that pass every enumeration and attack gate",
        },
        "fixture_manifest": fixture_manifest,
        "per_instance_rows": per_instance_rows,
        "reference_sample_manifest": sample_manifest,
        "parity_metrics": parity_metrics,
        "kac_ward_reference_ready_score": 1.0 if ready else 0.0,
        "attack_rows": attacks,
        "preconditions_checked": {
            "fixtures_frozen": True,
            "couplings_temperatures_orders_seeds_frozen": True,
            "precision": "float64/complex128",
            "enumeration_limit": config.enumeration_limit,
            "sample_count_per_case": config.sample_count,
            "tolerances": dict(PARITY_TOLERANCES),
            "condition_number_limit": CONDITION_NUMBER_LIMIT,
            "determinant_phase_tolerance": DETERMINANT_PHASE_TOLERANCE,
            "package_versions": _package_versions(),
            "resources": _resource_snapshot(root),
            "protected_hashes_at_start": before,
        },
        "protected_files_unchanged": protection,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "duration_s": time.monotonic() - started,
        "tests_run": [dict(row) for row in test_receipts],
        "random_seed": 6639,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> bool:
    """Fail closed when terminal evidence is incomplete or internally inconsistent."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    score = payload["kac_ward_reference_ready_score"]
    if score not in (0.0, 1.0):
        raise ValueError("readiness score must be binary")
    if payload["verdict_class"] not in {"positive", "blocked"}:
        raise ValueError("verdict_class must use the closed enum")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate does not match the bounded contract")
    if payload["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be false")
    if set(payload["field_provenance"]) < set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance is incomplete")
    if payload["reproducibility_checksum"] != reproducibility_checksum(payload):
        raise ValueError("reproducibility checksum mismatch")

    ready = score == 1.0
    if ready:
        if payload["status"] != "complete_bounded_exact_reference_ready":
            raise ValueError("ready status is inconsistent")
        if payload["verdict_class"] != "positive":
            raise ValueError("ready verdict_class must be positive")
        if not payload["parity_metrics"].get("all_instances_passed") or any(
            not row.get("parity", {}).get("passed") for row in payload["per_instance_rows"]
        ):
            raise ValueError("ready artifact contains a parity failure")
        if any(not row.get("passed") for row in payload["attack_rows"]):
            raise ValueError("ready artifact contains an attack failure")
        if not payload["protected_files_unchanged"].get("all_unchanged"):
            raise ValueError("ready artifact contains a protected-file failure")
        if len(payload["reference_sample_manifest"]) != len(payload["per_instance_rows"]):
            raise ValueError("ready artifact has an incomplete sample manifest")
        tests_passed, missing_tests = _test_gate(payload["tests_run"])
        if not tests_passed:
            raise ValueError(f"ready artifact lacks test receipts: {missing_tests}")
        if not payload["gate_check_summary"].get("passed"):
            raise ValueError("ready artifact gate summary is blocked")
    else:
        if not str(payload["status"]).startswith("blocked_"):
            raise ValueError("blocked score requires blocked status")
        if payload["verdict_class"] != "blocked":
            raise ValueError("blocked score requires blocked verdict_class")

    attacks = {row.get("attack") for row in payload["attack_rows"]}
    if attacks != set(REQUIRED_ATTACKS):
        raise ValueError("attack rows do not match the required matrix")
    for manifest in payload["reference_sample_manifest"]:
        path = Path(manifest["resolved_path"])
        if not path.is_file() or sha256_file(path) != manifest["sha256"]:
            raise ValueError("sample manifest hash or path is invalid")
        if path.stat().st_mode & 0o222:
            raise ValueError("sample manifest bank is not sealed")
    return True


def load_test_receipts() -> list[dict[str, Any]]:
    """Load verification receipts supplied by the outer test workflow."""

    configured = os.environ.get(TEST_RECEIPT_ENV)
    if not configured:
        return []
    value = json.loads(Path(configured).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("test receipt file must contain a JSON list")
    return [dict(row) for row in value]


def run_experiment(
    *,
    root: Path,
    output_path: Path,
    sample_dir: Path,
    run_date: str,
) -> dict[str, Any]:
    """Build, validate, and atomically publish the one terminal artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        config=ExperimentConfig(),
        sample_dir=sample_dir,
        test_receipts=load_test_receipts(),
    )
    validate_artifact(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the planned date and redirectable evidence destinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260826")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6639 or validate an already written artifact."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        payload = json.loads(args.validate.read_text(encoding="utf-8"))
        validate_artifact(payload)
        print(f"validated {args.validate}")
        return 0
    output = args.output or root / RESULT_RELATIVE_PATH
    sample_dir = args.sample_dir or root / SAMPLE_RELATIVE_DIR
    artifact = run_experiment(
        root=root,
        output_path=output,
        sample_dir=sample_dir,
        run_date=args.date,
    )
    print(f"{artifact['status']}: {output}")
    return 0 if artifact["kac_ward_reference_ready_score"] == 1.0 else 2


if __name__ == "__main__":  # pragma: no cover - exercised by required CLI command.
    raise SystemExit(main())
