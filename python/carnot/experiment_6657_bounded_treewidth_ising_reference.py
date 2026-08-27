"""Build a bounded exact Ising reference with variable elimination.

The module accepts only tiny simple graphs with a validated tree-decomposition
width of at most four. It computes exact probabilities in log space and draws
independent ancestral samples from the elimination trace. Full enumeration is
used only as an independent small-state verifier. Timings are evidence receipts,
not a speed claim.

Spec refs: REQ-SAMPLER-6657, REQ-REPORT-6657,
SCENARIO-SAMPLER-6657-EXACT-PARITY,
SCENARIO-SAMPLER-6657-ANCESTRAL-SAMPLING,
SCENARIO-SAMPLER-6657-FAIL-CLOSED,
SCENARIO-REPORT-6657-ATOMIC-CHECKSUM.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import metadata
from itertools import product
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np


INFERENCE_SUBSTRATE = "cpu_bounded_treewidth_junction_tree_exact_sampling_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6657_bounded_treewidth_ising_reference.json")
TEST_RECEIPT_ENV = "CARNOT_6657_TEST_RECEIPTS"
MAX_TREEWIDTH = 4
MAX_SPINS = 10
DEFAULT_SAMPLE_COUNT = 4096
DECOMPOSITION_SEED = 6657
EXACT_TOLERANCES: dict[str, float] = {
    "partition": 2.0e-11,
    "probability": 2.0e-12,
    "log_probability": 2.0e-11,
    "marginal": 2.0e-11,
    "normalization": 2.0e-12,
}
SAMPLE_TOLERANCES: dict[str, float] = {"state": 0.06, "node": 0.05, "edge": 0.06}
REQUIRED_TEST_SCOPES = (
    "focused_python",
    "scoped_coverage",
    "full_python_suite",
    "spec_coverage",
    "row_artifact_checks",
    "adversarial",
    "sampling_e2e",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "gate_check_summary",
    "prior_failure_receipt",
    "supported_domain_contract",
    "fixture_manifest",
    "decomposition_rows",
    "exact_parity_rows",
    "exact_sample_rows",
    "normalized_mass_receipts",
    "timing_rows",
    "ising_reference_ready",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)
PROTECTED_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-roadmap.yaml"),
    Path("scripts/research_conductor.py"),
    Path("research-references.md"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("_bmad/traceability.md"),
)
SPINS = (-1, 1)


class UnsupportedGraphError(ValueError):
    """Identify an input that lies outside the frozen exact-reference contract."""


@dataclass(frozen=True)
class IsingInstance:
    """Store one finite Ising graph and its expected fixture outcome."""

    instance_id: str
    n_spins: int
    edges: tuple[tuple[int, int, float], ...]
    fields: tuple[float, ...]
    temperature: float
    seed: int
    family: str = "custom"
    expected_supported: bool = True
    expected_rejection: str | None = None

    @property
    def fixture_sha256(self) -> str:
        """Bind every value that can change the target distribution."""

        return _sha256_json(
            {
                "instance_id": self.instance_id,
                "n_spins": self.n_spins,
                "edges": [list(edge) for edge in self.edges],
                "fields": list(self.fields),
                "temperature": self.temperature,
                "seed": self.seed,
                "family": self.family,
                "expected_supported": self.expected_supported,
                "expected_rejection": self.expected_rejection,
            }
        )


@dataclass(frozen=True)
class TreeDecomposition:
    """Store deterministic bags, their tree, and the elimination certificate."""

    bags: tuple[tuple[int, ...], ...]
    tree_edges: tuple[tuple[int, int], ...]
    elimination_order: tuple[int, ...]
    width: int


@dataclass(frozen=True)
class _Factor:
    """Store one log-space factor over a stable variable order."""

    scope: tuple[int, ...]
    values: np.ndarray = field(compare=False, repr=False)


@dataclass(frozen=True)
class _Conditional:
    """Store one exact ancestral conditional for a removed variable."""

    variable: int
    context: tuple[int, ...]
    probability_plus: Mapping[tuple[int, ...], float] = field(compare=False, repr=False)


@dataclass(frozen=True)
class ExactSolution:
    """Expose the validated decomposition, partition, and sampling trace."""

    decomposition: TreeDecomposition
    log_partition: float
    conditionals: tuple[_Conditional, ...] = field(compare=False, repr=False)


@dataclass(frozen=True)
class ExperimentConfig:
    """Freeze the only sample budget and fixture matrix used for readiness."""

    sample_count: int = DEFAULT_SAMPLE_COUNT


class DeterministicEvidenceClock:
    """Give unit tests stable nonzero timing receipts without hiding real timing in production."""

    def __init__(self) -> None:
        self._value = 0.0

    def __call__(self) -> float:
        self._value += 0.001
        return self._value


def _canonical_json(value: Any) -> bytes:
    """Encode evidence deterministically and reject nonfinite JSON values."""

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
    """Hash one canonical JSON value."""

    return sha256(_canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without interpreting its content."""

    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_fixtures() -> tuple[IsingInstance, ...]:
    """Return twelve supported cases and three explicit rejection controls."""

    fixtures = (
        IsingInstance("singleton_field", 1, (), (0.31,), 1.0, 6657001, "tree"),
        IsingInstance("edge_ferro", 2, ((0, 1, 0.73),), (0.0, 0.0), 0.9, 6657002, "ferromagnetic"),
        IsingInstance(
            "path4_field",
            4,
            ((0, 1, 0.4), (1, 2, -0.3), (2, 3, 0.6)),
            (0.2, -0.1, 0.0, 0.3),
            1.2,
            6657003,
            "field",
        ),
        IsingInstance(
            "star5_antiferro",
            5,
            ((0, 1, -0.4), (0, 2, -0.5), (0, 3, -0.6), (0, 4, -0.3)),
            (0.0,) * 5,
            1.1,
            6657004,
            "antiferromagnetic",
        ),
        IsingInstance(
            "tree6_mixed",
            6,
            ((0, 1, 0.5), (1, 2, -0.4), (1, 3, 0.2), (3, 4, -0.7), (3, 5, 0.35)),
            (0.1, 0.0, -0.2, 0.0, 0.15, -0.05),
            0.85,
            6657005,
            "tree",
        ),
        IsingInstance(
            "triangle_ferro",
            3,
            ((0, 1, 0.6), (1, 2, 0.55), (0, 2, 0.45)),
            (0.0,) * 3,
            1.3,
            6657006,
            "cycle",
        ),
        IsingInstance(
            "frustrated_triangle",
            3,
            ((0, 1, 0.7), (1, 2, 0.5), (0, 2, -0.65)),
            (0.05, -0.08, 0.02),
            0.95,
            6657007,
            "frustrated",
        ),
        IsingInstance(
            "cycle4_antiferro",
            4,
            ((0, 1, -0.5), (1, 2, -0.4), (2, 3, -0.6), (0, 3, -0.45)),
            (0.0,) * 4,
            1.4,
            6657008,
            "antiferromagnetic",
        ),
        IsingInstance(
            "cycle4_field",
            4,
            ((0, 1, 0.4), (1, 2, -0.5), (2, 3, 0.35), (0, 3, 0.25)),
            (0.2, -0.15, 0.1, 0.0),
            1.0,
            6657009,
            "field",
        ),
        IsingInstance(
            "cycle5_frustrated",
            5,
            ((0, 1, 0.5), (1, 2, 0.5), (2, 3, 0.5), (3, 4, 0.5), (0, 4, -0.5)),
            (0.0,) * 5,
            1.15,
            6657010,
            "frustrated",
        ),
        IsingInstance(
            "ladder6_ferro",
            6,
            (
                (0, 1, 0.35),
                (1, 2, 0.45),
                (3, 4, 0.4),
                (4, 5, 0.5),
                (0, 3, 0.3),
                (1, 4, 0.55),
                (2, 5, 0.25),
            ),
            (0.0,) * 6,
            0.9,
            6657011,
            "ferromagnetic",
        ),
        IsingInstance(
            "complete_k5_tw4",
            5,
            tuple(
                (left, right, 0.18 * (1 if (left + right) % 3 else -1))
                for left in range(5)
                for right in range(left + 1, 5)
            ),
            (0.1, -0.05, 0.0, 0.08, -0.12),
            1.25,
            6657012,
            "frustrated",
        ),
        IsingInstance(
            "unsupported_k6_tw5",
            6,
            tuple((left, right, 0.2) for left in range(6) for right in range(left + 1, 6)),
            (0.0,) * 6,
            1.0,
            6657091,
            "unsupported",
            False,
            "treewidth",
        ),
        IsingInstance(
            "unsupported_self_loop",
            3,
            ((0, 0, 0.2),),
            (0.0,) * 3,
            1.0,
            6657092,
            "unsupported",
            False,
            "self-loop",
        ),
        IsingInstance(
            "unsupported_duplicate_edge",
            3,
            ((0, 1, 0.2), (1, 0, -0.3)),
            (0.0,) * 3,
            1.0,
            6657093,
            "unsupported",
            False,
            "duplicate edge",
        ),
    )
    return fixtures


def validate_instance(instance: IsingInstance) -> dict[str, Any]:
    """Validate graph syntax before decomposition or numerical work starts."""

    if not isinstance(instance.n_spins, int) or not 1 <= instance.n_spins:
        raise UnsupportedGraphError("spin count must be a positive integer")
    if instance.n_spins > MAX_SPINS:
        raise UnsupportedGraphError(f"spin count exceeds supported limit {MAX_SPINS}")
    if len(instance.fields) != instance.n_spins:
        raise UnsupportedGraphError("field count must match spin count")
    if not all(math.isfinite(float(value)) for value in instance.fields):
        raise UnsupportedGraphError("fields must be finite")
    if not math.isfinite(float(instance.temperature)) or instance.temperature <= 0.0:
        raise UnsupportedGraphError("temperature must be finite and positive")
    normalized: set[tuple[int, int]] = set()
    for left, right, coupling in instance.edges:
        if not isinstance(left, int) or not isinstance(right, int):
            raise UnsupportedGraphError("edge endpoint must be an integer")
        if not 0 <= left < instance.n_spins or not 0 <= right < instance.n_spins:
            raise UnsupportedGraphError("edge endpoint lies outside the graph")
        if left == right:
            raise UnsupportedGraphError("self-loop is unsupported")
        edge = tuple(sorted((left, right)))
        if edge in normalized:
            raise UnsupportedGraphError("duplicate edge is unsupported")
        normalized.add(edge)
        if not math.isfinite(float(coupling)):
            raise UnsupportedGraphError("couplings must be finite")
    return {
        "valid": True,
        "simple_graph": True,
        "finite_fields_and_couplings": True,
        "n_spins": instance.n_spins,
        "edge_count": len(instance.edges),
    }


def deterministic_tree_decomposition(instance: IsingInstance) -> TreeDecomposition:
    """Build a stable min-fill order and the matching elimination bags."""

    validate_instance(instance)
    adjacency = {vertex: set() for vertex in range(instance.n_spins)}
    for left, right, _ in instance.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(adjacency)
    bags: list[tuple[int, ...]] = []
    order: list[int] = []
    while remaining:

        def key(vertex: int) -> tuple[int, int, int]:
            neighbors = sorted(adjacency[vertex] & remaining)
            missing = sum(
                second not in adjacency[first]
                for offset, first in enumerate(neighbors)
                for second in neighbors[offset + 1 :]
            )
            return missing, len(neighbors), vertex

        vertex = min(remaining, key=key)
        neighbors = sorted(adjacency[vertex] & remaining - {vertex})
        bags.append(tuple(sorted((vertex, *neighbors))))
        order.append(vertex)
        for offset, first in enumerate(neighbors):
            for second in neighbors[offset + 1 :]:
                adjacency[first].add(second)
                adjacency[second].add(first)
        remaining.remove(vertex)

    tree_edges: list[tuple[int, int]] = []
    for index in range(len(bags) - 1):
        separator = set(bags[index]) - {order[index]}
        parent = next(
            (later for later in range(index + 1, len(bags)) if separator <= set(bags[later])),
            index + 1,
        )
        tree_edges.append((index, parent))
    width = max(len(bag) - 1 for bag in bags)
    return TreeDecomposition(tuple(bags), tuple(tree_edges), tuple(order), width)


def validate_tree_decomposition(
    instance: IsingInstance, decomposition: TreeDecomposition
) -> dict[str, Any]:
    """Check the certificate as a tree decomposition, not only as a bag list."""

    validate_instance(instance)
    bags = decomposition.bags
    if not bags or any(not bag for bag in bags):
        raise UnsupportedGraphError("vertex coverage requires nonempty decomposition bags")
    for bag in bags:
        if len(set(bag)) != len(bag) or any(
            not isinstance(vertex, int) or not 0 <= vertex < instance.n_spins for vertex in bag
        ):
            raise UnsupportedGraphError("bag vertices must be unique graph vertices")
    graph = {index: set() for index in range(len(bags))}
    normalized_tree_edges: set[tuple[int, int]] = set()
    for left, right in decomposition.tree_edges:
        if (
            not 0 <= left < len(bags)
            or not 0 <= right < len(bags)
            or left == right
            or tuple(sorted((left, right))) in normalized_tree_edges
        ):
            raise UnsupportedGraphError("decomposition tree has an invalid edge")
        edge = tuple(sorted((left, right)))
        normalized_tree_edges.add(edge)
        graph[left].add(right)
        graph[right].add(left)
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in graph[current] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    if len(normalized_tree_edges) != len(bags) - 1 or len(seen) != len(bags):
        raise UnsupportedGraphError("decomposition tree must be connected and acyclic")
    covered = set().union(*(set(bag) for bag in bags))
    if covered != set(range(instance.n_spins)):
        raise UnsupportedGraphError("vertex coverage is incomplete")
    for left, right, _ in instance.edges:
        if not any(left in bag and right in bag for bag in bags):
            raise UnsupportedGraphError("edge coverage is incomplete")
    for vertex in range(instance.n_spins):
        holders = {index for index, bag in enumerate(bags) if vertex in bag}
        reached = {next(iter(holders))}
        pending = list(reached)
        while pending:
            current = pending.pop()
            for neighbor in (graph[current] & holders) - reached:
                reached.add(neighbor)
                pending.append(neighbor)
        if reached != holders:
            raise UnsupportedGraphError("running intersection is invalid")
    if tuple(sorted(decomposition.elimination_order)) != tuple(range(instance.n_spins)):
        raise UnsupportedGraphError("elimination order must contain every vertex once")
    observed_width = max(len(bag) - 1 for bag in bags)
    if decomposition.width != observed_width:
        raise UnsupportedGraphError("declared width does not match the bags")
    if observed_width > MAX_TREEWIDTH:
        raise UnsupportedGraphError(
            f"certified treewidth {observed_width} exceeds supported treewidth {MAX_TREEWIDTH}"
        )
    return {
        "valid": True,
        "width": observed_width,
        "tree": True,
        "vertex_coverage": True,
        "edge_coverage": True,
        "running_intersection": True,
    }


def _initial_factors(instance: IsingInstance, evidence: Mapping[int, int]) -> list[_Factor]:
    factors: list[_Factor] = []
    inverse_temperature = 1.0 / instance.temperature
    for vertex, field_value in enumerate(instance.fields):
        factors.append(
            _Factor(
                (vertex,),
                np.asarray([-field_value, field_value], dtype=np.float64) * inverse_temperature,
            )
        )
    for left, right, coupling in instance.edges:
        first, second = sorted((left, right))
        values = (
            np.asarray([[coupling, -coupling], [-coupling, coupling]], dtype=np.float64)
            * inverse_temperature
        )
        factors.append(_Factor((first, second), values))
    for vertex, spin in evidence.items():
        if not isinstance(vertex, int) or not 0 <= vertex < instance.n_spins or spin not in SPINS:
            raise UnsupportedGraphError("evidence must map graph vertices to -1 or +1")
        values = np.asarray([0.0, -np.inf] if spin == -1 else [-np.inf, 0.0])
        factors.append(_Factor((vertex,), values))
    return factors


def _combine(involved: Sequence[_Factor], scope: tuple[int, ...]) -> np.ndarray:
    values = np.empty((2,) * len(scope), dtype=np.float64)
    positions = {variable: index for index, variable in enumerate(scope)}
    for index in np.ndindex(values.shape):
        total = 0.0
        for factor in involved:
            factor_index = tuple(index[positions[variable]] for variable in factor.scope)
            total += float(factor.values[factor_index])
        values[index] = total
    return values


def _eliminate(
    instance: IsingInstance,
    decomposition: TreeDecomposition,
    evidence: Mapping[int, int],
    keep_conditionals: bool,
) -> tuple[float, tuple[_Conditional, ...]]:
    factors = _initial_factors(instance, evidence)
    conditionals: list[_Conditional] = []
    for variable in decomposition.elimination_order:
        involved = [factor for factor in factors if variable in factor.scope]
        factors = [factor for factor in factors if variable not in factor.scope]
        context = tuple(sorted({item for factor in involved for item in factor.scope} - {variable}))
        scope = (variable, *context)
        joint = _combine(involved, scope)
        with np.errstate(invalid="ignore"):
            reduced = np.logaddexp(joint[0], joint[1])
        if keep_conditionals:
            probability_plus: dict[tuple[int, ...], float] = {}
            context_shape = (2,) * len(context)
            for context_index in np.ndindex(context_shape):
                denominator = float(reduced[context_index])
                probability_plus[tuple(SPINS[index] for index in context_index)] = float(
                    math.exp(float(joint[(1, *context_index)]) - denominator)
                )
            conditionals.append(_Conditional(variable, context, probability_plus))
        factors.append(_Factor(context, np.asarray(reduced)))
    log_partition = sum(float(np.asarray(factor.values)) for factor in factors)
    if not math.isfinite(log_partition):
        raise UnsupportedGraphError("evidence has zero or nonfinite probability mass")
    return log_partition, tuple(conditionals)


def solve_exact(instance: IsingInstance) -> ExactSolution:
    """Validate the decomposition and run exact log-space factor elimination."""

    decomposition = deterministic_tree_decomposition(instance)
    validate_tree_decomposition(instance, decomposition)
    log_partition, conditionals = _eliminate(instance, decomposition, {}, True)
    return ExactSolution(decomposition, log_partition, conditionals)


def _validate_state(instance: IsingInstance, state: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(value) for value in state)
    if len(normalized) != instance.n_spins or any(value not in SPINS for value in normalized):
        raise UnsupportedGraphError("spin state must match the graph and contain only -1 or +1")
    return normalized


def _log_weight(instance: IsingInstance, state: Sequence[int]) -> float:
    spins = _validate_state(instance, state)
    favorable = sum(field * spins[index] for index, field in enumerate(instance.fields))
    favorable += sum(
        coupling * spins[left] * spins[right] for left, right, coupling in instance.edges
    )
    return float(favorable / instance.temperature)


def configuration_log_probability(
    instance: IsingInstance, state: Sequence[int], solution: ExactSolution | None = None
) -> float:
    """Return one normalized log probability from the exact DP partition."""

    active = solution or solve_exact(instance)
    return _log_weight(instance, state) - active.log_partition


def configuration_probability(
    instance: IsingInstance, state: Sequence[int], solution: ExactSolution | None = None
) -> float:
    """Return one normalized probability without an unnormalized shortcut."""

    return math.exp(configuration_log_probability(instance, state, solution))


def _evidence_log_partition(
    instance: IsingInstance, solution: ExactSolution, evidence: Mapping[int, int]
) -> float:
    return _eliminate(instance, solution.decomposition, evidence, False)[0]


def _edge_id(left: int, right: int) -> str:
    return f"{min(left, right)}-{max(left, right)}"


def _pair_id(left_spin: int, right_spin: int) -> str:
    return f"{left_spin},{right_spin}"


def exact_marginals(
    instance: IsingInstance, solution: ExactSolution | None = None
) -> dict[str, Any]:
    """Compute exact node and edge marginals with evidence partitions."""

    active = solution or solve_exact(instance)
    node_plus = [
        math.exp(_evidence_log_partition(instance, active, {vertex: 1}) - active.log_partition)
        for vertex in range(instance.n_spins)
    ]
    edge_joint: dict[str, dict[str, float]] = {}
    for left, right, _ in instance.edges:
        first, second = sorted((left, right))
        edge_joint[_edge_id(first, second)] = {
            _pair_id(first_spin, second_spin): math.exp(
                _evidence_log_partition(instance, active, {first: first_spin, second: second_spin})
                - active.log_partition
            )
            for first_spin in SPINS
            for second_spin in SPINS
        }
    return {"node_plus": node_plus, "edge_joint": edge_joint}


def brute_force_reference(instance: IsingInstance) -> dict[str, Any]:
    """Enumerate scalar energies independently of the factor implementation."""

    validate_instance(instance)
    states = tuple(product(SPINS, repeat=instance.n_spins))
    log_weights = np.asarray([_log_weight(instance, state) for state in states])
    maximum = float(np.max(log_weights))
    partition = math.exp(maximum) * float(np.sum(np.exp(log_weights - maximum)))
    probabilities = np.exp(log_weights) / partition
    node_plus = [
        float(
            sum(
                probability
                for state, probability in zip(states, probabilities, strict=True)
                if state[vertex] == 1
            )
        )
        for vertex in range(instance.n_spins)
    ]
    edge_joint: dict[str, dict[str, float]] = {}
    for left, right, _ in instance.edges:
        first, second = sorted((left, right))
        edge_joint[_edge_id(first, second)] = {
            _pair_id(first_spin, second_spin): float(
                sum(
                    probability
                    for state, probability in zip(states, probabilities, strict=True)
                    if state[first] == first_spin and state[second] == second_spin
                )
            )
            for first_spin in SPINS
            for second_spin in SPINS
        }
    return {
        "states": states,
        "probabilities": probabilities.tolist(),
        "partition_function": partition,
        "log_partition": math.log(partition),
        "node_plus": node_plus,
        "edge_joint": edge_joint,
    }


def cross_check_fixture(instance: IsingInstance) -> dict[str, Any]:
    """Compare every exact output with independent brute-force enumeration."""

    solution = solve_exact(instance)
    brute = brute_force_reference(instance)
    marginals = exact_marginals(instance, solution)
    log_errors: list[float] = []
    probability_errors: list[float] = []
    mass = 0.0
    for state, expected in zip(brute["states"], brute["probabilities"], strict=True):
        observed_log = configuration_log_probability(instance, state, solution)
        observed = math.exp(observed_log)
        log_errors.append(abs(observed_log - math.log(expected)))
        probability_errors.append(abs(observed - expected))
        mass += observed
    node_error = max(
        (
            abs(observed - expected)
            for observed, expected in zip(marginals["node_plus"], brute["node_plus"], strict=True)
        ),
        default=0.0,
    )
    edge_error = max(
        (
            abs(value - brute["edge_joint"][edge_id][pair])
            for edge_id, table in marginals["edge_joint"].items()
            for pair, value in table.items()
        ),
        default=0.0,
    )
    row = {
        "instance_id": instance.instance_id,
        "partition_function_dp": math.exp(solution.log_partition),
        "partition_function_brute_force": brute["partition_function"],
        "partition_error": abs(math.exp(solution.log_partition) - brute["partition_function"]),
        "log_probability_error_max": max(log_errors, default=0.0),
        "probability_error_max": max(probability_errors, default=0.0),
        "node_marginal_error_max": node_error,
        "edge_marginal_error_max": edge_error,
        "probability_mass": mass,
        "normalization_error": abs(mass - 1.0),
        "state_count": len(brute["states"]),
        "tolerances": dict(EXACT_TOLERANCES),
    }
    row["passed"] = bool(
        row["partition_error"] <= EXACT_TOLERANCES["partition"]
        and row["log_probability_error_max"] <= EXACT_TOLERANCES["log_probability"]
        and row["probability_error_max"] <= EXACT_TOLERANCES["probability"]
        and row["node_marginal_error_max"] <= EXACT_TOLERANCES["marginal"]
        and row["edge_marginal_error_max"] <= EXACT_TOLERANCES["marginal"]
        and row["normalization_error"] <= EXACT_TOLERANCES["normalization"]
    )
    return row


def independent_samples(
    instance: IsingInstance,
    sample_count: int,
    seed: int,
    solution: ExactSolution | None = None,
) -> dict[str, Any]:
    """Draw independent states from exact conditionals in reverse elimination order."""

    if not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    active = solution or solve_exact(instance)
    rng = np.random.default_rng(int(seed))
    rows: list[list[int]] = []
    for _ in range(sample_count):
        state: dict[int, int] = {}
        for conditional in reversed(active.conditionals):
            context = tuple(state[vertex] for vertex in conditional.context)
            state[conditional.variable] = (
                1 if rng.random() < conditional.probability_plus[context] else -1
            )
        rows.append([state[index] for index in range(instance.n_spins)])
    sample_bytes = np.asarray(rows, dtype=np.int8, order="C").tobytes()
    return {
        "samples": rows,
        "sample_count": sample_count,
        "seed": int(seed),
        "rng": "numpy.PCG64",
        "sample_sha256": sha256(sample_bytes).hexdigest(),
    }


def sample_check_fixture(instance: IsingInstance, sample_count: int, seed: int) -> dict[str, Any]:
    """Compare exact independent sample frequencies with the brute-force law."""

    solution = solve_exact(instance)
    brute = brute_force_reference(instance)
    sample_receipt = independent_samples(instance, sample_count, seed, solution)
    samples = [tuple(row) for row in sample_receipt["samples"]]
    counts = {state: 0 for state in brute["states"]}
    for state in samples:
        counts[state] += 1
    state_error = max(
        abs(counts[state] / sample_count - probability)
        for state, probability in zip(brute["states"], brute["probabilities"], strict=True)
    )
    node_error = max(
        abs(
            sum(state[vertex] == 1 for state in samples) / sample_count - brute["node_plus"][vertex]
        )
        for vertex in range(instance.n_spins)
    )
    edge_error = max(
        (
            abs(
                sum(
                    state[int(edge_id.split("-")[0])] == int(pair.split(",")[0])
                    and state[int(edge_id.split("-")[1])] == int(pair.split(",")[1])
                    for state in samples
                )
                / sample_count
                - probability
            )
            for edge_id, table in brute["edge_joint"].items()
            for pair, probability in table.items()
        ),
        default=0.0,
    )
    brute_map = {
        state: probability
        for state, probability in zip(brute["states"], brute["probabilities"], strict=True)
    }
    likelihood_error = max(
        (
            abs(
                configuration_log_probability(instance, state, solution)
                - math.log(brute_map[state])
            )
            for state in set(samples)
        ),
        default=0.0,
    )
    row = {
        "instance_id": instance.instance_id,
        "seed": int(seed),
        "sample_count": sample_count,
        "rng": sample_receipt["rng"],
        "sample_sha256": sample_receipt["sample_sha256"],
        "independent_draw_contract": "one fresh ancestral traversal per sample; no Markov-chain state",
        "likelihood_error_max": likelihood_error,
        "state_frequency_error_max": state_error,
        "node_frequency_error_max": node_error,
        "edge_frequency_error_max": edge_error,
        "tolerances": dict(SAMPLE_TOLERANCES),
        "sample_size_note": "4096 exact independent draws exceed the 1000-sample distributional floor for n<64.",
    }
    row["passed"] = bool(
        likelihood_error <= EXACT_TOLERANCES["log_probability"]
        and state_error <= SAMPLE_TOLERANCES["state"]
        and node_error <= SAMPLE_TOLERANCES["node"]
        and edge_error <= SAMPLE_TOLERANCES["edge"]
    )
    return row


def protected_hashes(root: Path) -> dict[str, str]:
    """Capture the protected repository inputs before evidence generation."""

    return {
        path.as_posix(): sha256_file(root / path) if (root / path).is_file() else "missing"
        for path in PROTECTED_RELATIVE_PATHS
    }


def protected_files_unchanged(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, Any]:
    """Compare the complete protected path set without hiding additions or removals."""

    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path) == after.get(path) and before.get(path) != "missing",
        }
        for path in sorted(set(before) | set(after))
    ]
    return {"all_unchanged": bool(rows) and all(row["unchanged"] for row in rows), "rows": rows}


def _package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "pytest", "coverage"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _resource_snapshot(root: Path) -> dict[str, Any]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    physical_pages = os.sysconf("SC_PHYS_PAGES")
    disk = shutil.disk_usage(root)
    return {
        "cpu_architecture": platform.machine(),
        "cpu_logical_count": os.cpu_count(),
        "ram_bytes": int(page_size * physical_pages),
        "disk_free_bytes_at_start": int(disk.free),
        "resource_boundary": "CPU capacity inventory only; no performance claim",
    }


def _test_gate(receipts: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    passed = {
        str(row.get("scope"))
        for row in receipts
        if row.get("exit_code") == 0 and row.get("command") and row.get("summary")
    }
    missing = [scope for scope in REQUIRED_TEST_SCOPES if scope not in passed]
    return not missing, missing


def _prior_failure_receipt(root: Path) -> dict[str, Any]:
    path = root / "ops/conductor-log.md"
    records = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if "Kac-Ward exact planar-Ising reference" in line and "Hard wall-clock cap" in line
    ]
    return {
        "experiment_id": "exp6639-kac-ward-planar-reference",
        "terminal_verdict": "no_terminal_artifact_after_three_hard_wall_clock_caps",
        "source_path": "ops/conductor-log.md",
        "source_sha256": sha256_file(path),
        "terminal_record_count": len(records),
        "terminal_records": records,
        "changed_technique": "Kac-Ward determinants replaced by log-space factor elimination.",
        "changed_scope": "All planar graphs replaced by preregistered n<=10 graphs with certified treewidth<=4.",
    }


def _aggregate_rows(
    decomposition_rows: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]],
    sample_rows: Sequence[Mapping[str, Any]],
    mass_rows: Sequence[Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    protection: Mapping[str, Any],
) -> dict[str, Any]:
    supported_decompositions = [row for row in decomposition_rows if row.get("accepted")]
    rejected = [row for row in decomposition_rows if not row.get("accepted")]
    tests_passed, missing_tests = _test_gate(tests_run)
    values = {
        "supported_fixture_count": len(supported_decompositions),
        "rejection_fixture_count": len(rejected),
        "decomposition_all_passed": len(supported_decompositions) >= 12
        and all(row.get("passed") for row in decomposition_rows),
        "parity_all_passed": len(parity_rows) == len(supported_decompositions)
        and all(row.get("passed") for row in parity_rows),
        "normalized_mass_all_passed": len(mass_rows) == len(supported_decompositions)
        and all(row.get("passed") for row in mass_rows),
        "sampling_all_passed": len(sample_rows) == len(supported_decompositions)
        and all(row.get("passed") for row in sample_rows),
        "rejection_all_passed": len(rejected) >= 3 and all(row.get("passed") for row in rejected),
        "tests_all_passed": tests_passed,
        "missing_test_scopes": missing_tests,
        "protection_all_passed": protection.get("all_unchanged") is True,
        "maximum_errors": {
            "partition": max((float(row["partition_error"]) for row in parity_rows), default=None),
            "log_probability": max(
                (float(row["log_probability_error_max"]) for row in parity_rows), default=None
            ),
            "marginal": max(
                (
                    max(
                        float(row["node_marginal_error_max"]), float(row["edge_marginal_error_max"])
                    )
                    for row in parity_rows
                ),
                default=None,
            ),
            "normalization": max(
                (float(row["normalization_error"]) for row in parity_rows), default=None
            ),
            "sample_state_frequency": max(
                (float(row["state_frequency_error_max"]) for row in sample_rows), default=None
            ),
        },
    }
    values["ready"] = all(
        values[key]
        for key in (
            "decomposition_all_passed",
            "parity_all_passed",
            "normalized_mass_all_passed",
            "sampling_all_passed",
            "rejection_all_passed",
            "tests_all_passed",
            "protection_all_passed",
        )
    )
    return values


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Hash final content after blanking only the checksum field."""

    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return _sha256_json(material)


def _field_provenance(root: Path) -> dict[str, dict[str, str]]:
    source_path = Path(__file__)
    source_hash = sha256_file(source_path)
    return {
        field_name: {
            "source": str(source_path.relative_to(root)),
            "source_sha256": source_hash,
            "reducer": "experiment_6657 row reducer",
            "schema_lineage": "REQ-SAMPLER-6657 and REQ-REPORT-6657",
        }
        for field_name in REQUIRED_ARTIFACT_FIELDS
    }


def build_artifact(
    *,
    root: Path,
    config: ExperimentConfig,
    test_receipts: Sequence[Mapping[str, Any]],
    run_date: str = "20260827",
    timing_clock: Callable[[], float] | None = None,
    output_parent: Path | None = None,
) -> dict[str, Any]:
    """Build one terminal artifact entirely from visible fixture rows."""

    del output_parent
    if config.sample_count != DEFAULT_SAMPLE_COUNT:
        raise ValueError(f"sample_count must equal frozen count {DEFAULT_SAMPLE_COUNT}")
    clock = timing_clock or time.monotonic
    started = clock()
    before = protected_hashes(root)
    fixtures = frozen_fixtures()
    fixture_manifest: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for instance in fixtures:
        fixture_manifest.append(
            {
                "instance_id": instance.instance_id,
                "family": instance.family,
                "n_spins": instance.n_spins,
                "edges": [list(edge) for edge in instance.edges],
                "fields": list(instance.fields),
                "temperature": instance.temperature,
                "decomposition_seed": DECOMPOSITION_SEED,
                "sampling_seed": instance.seed,
                "expected_supported": instance.expected_supported,
                "expected_rejection": instance.expected_rejection,
                "fixture_sha256": instance.fixture_sha256,
            }
        )
        setup_started = clock()
        decomposition: TreeDecomposition | None = None
        try:
            decomposition = deterministic_tree_decomposition(instance)
            validation = validate_tree_decomposition(instance, decomposition)
            if not instance.expected_supported:
                raise UnsupportedGraphError("unsupported fixture was unexpectedly accepted")
            decomposition_rows.append(
                {
                    "instance_id": instance.instance_id,
                    "accepted": True,
                    "passed": True,
                    "width": decomposition.width,
                    "bags": [list(bag) for bag in decomposition.bags],
                    "tree_edges": [list(edge) for edge in decomposition.tree_edges],
                    "elimination_order": list(decomposition.elimination_order),
                    "running_intersection_checks": {"passed": validation["running_intersection"]},
                    "rejection_status": "accepted",
                }
            )
        except UnsupportedGraphError as exc:
            expected = instance.expected_rejection or "not rejected"
            passed = not instance.expected_supported and expected in str(exc)
            decomposition_rows.append(
                {
                    "instance_id": instance.instance_id,
                    "accepted": False,
                    "passed": passed,
                    "width": decomposition.width if decomposition else None,
                    "bags": [list(bag) for bag in decomposition.bags] if decomposition else [],
                    "tree_edges": [list(edge) for edge in decomposition.tree_edges]
                    if decomposition
                    else [],
                    "elimination_order": list(decomposition.elimination_order)
                    if decomposition
                    else [],
                    "running_intersection_checks": {
                        "passed": False,
                        "not_applicable_after_rejection": True,
                    },
                    "rejection_status": "rejected_as_expected"
                    if passed
                    else "unexpected_rejection",
                    "expected_rejection": expected,
                    "observed_error": str(exc),
                }
            )
            continue

        setup_s = max(clock() - setup_started, 0.0)
        parity = cross_check_fixture(instance)
        parity_rows.append(parity)
        mass_rows.append(
            {
                "instance_id": instance.instance_id,
                "probability_mass": parity["probability_mass"],
                "normalization_error": parity["normalization_error"],
                "tolerance": EXACT_TOLERANCES["normalization"],
                "passed": parity["normalization_error"] <= EXACT_TOLERANCES["normalization"],
            }
        )
        sample_started = clock()
        sample_row = sample_check_fixture(instance, config.sample_count, instance.seed)
        sampling_s = max(clock() - sample_started, 0.0)
        sample_rows.append(sample_row)
        timing_rows.append(
            {
                "instance_id": instance.instance_id,
                "setup_wall_time_s": setup_s,
                "sampling_wall_time_s": sampling_s,
                "sample_count": config.sample_count,
                "claim_boundary": "Timing is a reproducibility receipt with no performance claim.",
            }
        )

    after = protected_hashes(root)
    protection = protected_files_unchanged(before, after)
    aggregate = _aggregate_rows(
        decomposition_rows, parity_rows, sample_rows, mass_rows, test_receipts, protection
    )
    failed_checks: list[dict[str, Any]] = []
    checks = (
        ("decomposition", aggregate["decomposition_all_passed"]),
        ("parity", aggregate["parity_all_passed"]),
        ("normalization", aggregate["normalized_mass_all_passed"]),
        ("sampling", aggregate["sampling_all_passed"]),
        ("rejection", aggregate["rejection_all_passed"]),
        ("tests", aggregate["tests_all_passed"]),
        ("protected_files", aggregate["protection_all_passed"]),
    )
    for check, passed in checks:
        if not passed:
            observed: Any = aggregate.get("missing_test_scopes") if check == "tests" else False
            failed_checks.append({"check": check, "observed_value": observed, "expected": True})
    ready = bool(aggregate["ready"])
    first_failure = failed_checks[0]["check"] if failed_checks else "none"
    status = (
        "complete_bounded_treewidth_exact_reference_ready"
        if ready
        else f"blocked_{first_failure}_check_failed"
    )
    per_unit_rows = (
        [{"unit_type": "decomposition_or_rejection", **row} for row in decomposition_rows]
        + [{"unit_type": "exact_parity", **row} for row in parity_rows]
        + [{"unit_type": "normalized_mass", **row} for row in mass_rows]
        + [{"unit_type": "exact_sampling", **row} for row in sample_rows]
    )
    artifact: dict[str, Any] = {
        "schema_version": "experiment-6657.v1",
        "planning_date": run_date,
        "status": status,
        "honest_verdict": (
            "complete: bounded exact Ising reference is ready; setup and sampling timings support no speed claim"
            if ready
            else f"blocked_{first_failure}_check_failed: bounded exact-reference readiness was not established"
        ),
        "verdict_class": None if ready else "blocked",
        "gate_check_summary": {
            "passed": ready,
            "failed_checks": failed_checks,
            "checked_categories": [item[0] for item in checks],
        },
        "prior_failure_receipt": _prior_failure_receipt(root),
        "supported_domain_contract": {
            "spin_encoding": "{-1,+1}",
            "energy": "E(s)=-sum_(i,j in unique undirected edges) J_ij*s_i*s_j-sum_i h_i*s_i",
            "fields": "finite real values are supported",
            "couplings": "finite real values on unique simple undirected edges; ferro, antiferro, and frustration supported",
            "temperature": "finite and positive; probabilities are exp(-E/temperature)/Z",
            "treewidth": f"deterministic min-fill certificate must validate at width <= {MAX_TREEWIDTH}",
            "size": f"1 <= n_spins <= {MAX_SPINS}",
            "rejection_rules": [
                "self-loop",
                "duplicate edge",
                "invalid endpoint",
                "nonfinite value",
                "nonpositive temperature",
                "malformed decomposition",
                "treewidth above four",
                "oversized graph",
            ],
            "claim_boundary": "bounded CPU correctness reference only; no speed, asymptotic, hardware, or planar-general claim",
        },
        "fixture_manifest": fixture_manifest,
        "decomposition_rows": decomposition_rows,
        "exact_parity_rows": parity_rows,
        "exact_sample_rows": sample_rows,
        "normalized_mass_receipts": mass_rows,
        "timing_rows": timing_rows,
        "ising_reference_ready": ready,
        "per_unit_rows": per_unit_rows,
        "aggregate_row_recomputation": aggregate,
        "preconditions_checked": {
            "inputs_frozen": True,
            "spin_energy_convention_inventoried": True,
            "fixture_count_supported": sum(item.expected_supported for item in fixtures),
            "fixture_count_rejected": sum(not item.expected_supported for item in fixtures),
            "exact_tolerances": dict(EXACT_TOLERANCES),
            "sample_tolerances": dict(SAMPLE_TOLERANCES),
            "sample_count": config.sample_count,
            "decomposition_seed": DECOMPOSITION_SEED,
            "package_versions": _package_versions(),
            "resources": _resource_snapshot(root),
            "source_hashes": {
                path.as_posix(): sha256_file(root / path)
                for path in (
                    Path("python/carnot/experiment_6639_kac_ward_planar_reference.py"),
                    Path("python/carnot/samplers/spectral_k_block.py"),
                    Path("python/carnot/experiment_6597_spectral_k_block_ising_canary.py"),
                    Path("python/carnot/models/ising/__init__.py"),
                    Path("python/carnot/verify/ising.py"),
                    Path("openspec/capabilities/samplers/spec.md"),
                    Path("openspec/capabilities/research-reporting/spec.md"),
                )
            },
            "protected_hashes_at_start": before,
        },
        "protected_files_unchanged": protection,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": _field_provenance(root),
        "random_seed": {
            "decomposition_seed": DECOMPOSITION_SEED,
            "sampling_seed_schedule": {
                item.instance_id: item.seed for item in fixtures if item.expected_supported
            },
            "rng": "fresh NumPy PCG64 generator per fixture",
        },
        "duration_s": max(clock() - started, 0.0),
        "tests_run": [dict(row) for row in test_receipts],
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(payload: Mapping[str, Any]) -> bool:
    """Reject incomplete, drifted, unsupported, or over-claimed evidence."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(payload))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if payload["verdict_class"] not in (None, "blocked"):
        raise ValueError("verdict_class must be null or blocked")
    if payload["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate does not match the bounded contract")
    if payload["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true")
    if set(payload["field_provenance"]) < set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance is incomplete")
    if payload["reproducibility_checksum"] != reproducibility_checksum(payload):
        raise ValueError("reproducibility checksum mismatch")
    if payload["prior_failure_receipt"].get("terminal_record_count") != 3:
        raise ValueError("prior failure receipt must contain three terminal records")
    ready = payload["ising_reference_ready"] is True
    tests_passed, missing_tests = _test_gate(payload["tests_run"])
    if ready:
        if payload["status"] != "complete_bounded_treewidth_exact_reference_ready":
            raise ValueError("ready status is inconsistent")
        if payload["verdict_class"] is not None:
            raise ValueError("ready verdict_class must be null")
        if not payload["protected_files_unchanged"].get("all_unchanged"):
            raise ValueError("ready artifact contains a protected-file failure")
        if any(not row.get("passed") for row in payload["exact_sample_rows"]):
            raise ValueError("ready artifact contains a sample failure")
        if any(not row.get("passed") for row in payload["exact_parity_rows"]):
            raise ValueError("ready artifact contains a parity failure")
        if any(not row.get("passed") for row in payload["normalized_mass_receipts"]):
            raise ValueError("ready artifact contains a normalization failure")
        if any(not row.get("passed") for row in payload["decomposition_rows"]):
            raise ValueError("ready artifact contains a decomposition or rejection failure")
        if not tests_passed:
            raise ValueError(f"ready artifact lacks test receipts: {missing_tests}")
        if not payload["gate_check_summary"].get("passed"):
            raise ValueError("ready gate summary is blocked")
    else:
        if not str(payload["status"]).startswith("blocked_"):
            raise ValueError("blocked readiness requires blocked status")
        if payload["verdict_class"] != "blocked":
            raise ValueError("blocked readiness requires blocked verdict_class")
        if payload["gate_check_summary"].get("passed") or not payload["gate_check_summary"].get(
            "failed_checks"
        ):
            raise ValueError("blocked gate summary must name failures")
    recomputed = _aggregate_rows(
        payload["decomposition_rows"],
        payload["exact_parity_rows"],
        payload["exact_sample_rows"],
        payload["normalized_mass_receipts"],
        payload["tests_run"],
        payload["protected_files_unchanged"],
    )
    if recomputed != payload["aggregate_row_recomputation"] or recomputed["ready"] != ready:
        raise ValueError("aggregate row recomputation does not match the rows")
    supported_count = sum(
        bool(row.get("expected_supported")) for row in payload["fixture_manifest"]
    )
    if supported_count < 12 or len(payload["fixture_manifest"]) < 15:
        raise ValueError("fixture manifest is incomplete")
    expected_units = (
        len(payload["decomposition_rows"])
        + len(payload["exact_parity_rows"])
        + len(payload["exact_sample_rows"])
        + len(payload["normalized_mass_receipts"])
    )
    if len(payload["per_unit_rows"]) != expected_units:
        raise ValueError("per-unit rows are incomplete")
    if any(
        "no performance claim" not in row.get("claim_boundary", "")
        for row in payload["timing_rows"]
    ):
        raise ValueError("timing rows must preserve the no-performance-claim boundary")
    return True


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write one complete JSON document through same-directory replacement."""

    encoded = _canonical_json(payload) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "atomic_replace": True,
        "file_fsync": True,
        "directory_fsync": True,
    }


def load_test_receipts() -> list[dict[str, Any]]:
    """Load outer verification commands without writing repository state."""

    configured = os.environ.get(TEST_RECEIPT_ENV)
    if not configured:
        return []
    value = json.loads(Path(configured).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("test receipt file must contain a JSON list")
    return [dict(row) for row in value]


def run_experiment(*, root: Path, output_path: Path, run_date: str) -> dict[str, Any]:
    """Build, validate, and atomically publish the terminal artifact."""

    artifact = build_artifact(
        root=root,
        run_date=run_date,
        config=ExperimentConfig(),
        test_receipts=load_test_receipts(),
    )
    validate_artifact(artifact)
    write_json_atomic(output_path, artifact)
    return artifact


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260827")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6657 or validate a redirected artifact."""

    args = _parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.validate is not None:
        payload = json.loads(args.validate.read_text(encoding="utf-8"))
        validate_artifact(payload)
        print(f"validated {args.validate}")
        return 0
    output = args.output or root / RESULT_RELATIVE_PATH
    artifact = run_experiment(root=root, output_path=output, run_date=args.date)
    print(f"{artifact['status']}: {output}")
    return 0 if artifact["ising_reference_ready"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised by the required CLI command.
    raise SystemExit(main())
