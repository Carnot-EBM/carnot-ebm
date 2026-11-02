"""Unsupervised RUN-CSP-style message passing for binary validator graphs.

The solver maps Carnot validator rows into a bipartite variable-constraint
graph. It then trains a small set of shared message-passing parameters using
only the graph energy as a loss, matching the core RUN-CSP idea of learning a
size-independent constraint solver without labeled assignments.

Spec: REQ-SAMPLE-1972, SCENARIO-SAMPLE-1972
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_ID = 1972
RUN_DATE = "20260512"
SPEC_REFS = ["REQ-SAMPLE-1972", "SCENARIO-SAMPLE-1972"]
PAPER_REFERENCE = "arXiv:1909.08387"
DEFAULT_RESULT_PATH = (
    Path(__file__).resolve().parents[3] / "results/experiment_1972_run_csp_unsupervised.json"
)

BinaryValue = int
ProbabilityVector = Sequence[float]
EnergyCallback = Callable[[ProbabilityVector], float]


@dataclass(frozen=True)
class BinaryCSPConstraint:
    """A pairwise binary-CSP constraint represented by an allowed table."""

    name: str
    scope: tuple[int, int]
    allowed: tuple[tuple[BinaryValue, BinaryValue], ...]

    def __post_init__(self) -> None:
        if len(self.scope) != 2 or self.scope[0] == self.scope[1]:
            raise ValueError("scope must contain exactly two distinct variable indices")
        if not self.allowed:
            raise ValueError("allowed table must contain at least one binary pair")
        for pair in self.allowed:
            if len(pair) != 2 or any(value not in (0, 1) for value in pair):
                raise ValueError("allowed table entries must be binary values")

    @property
    def allowed_set(self) -> frozenset[tuple[BinaryValue, BinaryValue]]:
        """Return allowed tuples as a set for fast membership checks."""

        return frozenset(self.allowed)

    def is_satisfied(self, assignment: Sequence[int | bool]) -> bool:
        """Return whether the discrete assignment satisfies this constraint."""

        left, right = self.scope
        return (int(bool(assignment[left])), int(bool(assignment[right]))) in self.allowed_set

    def violation_probability(self, probabilities: ProbabilityVector) -> float:
        """Expected violation under independent Bernoulli variable probabilities."""

        left, right = self.scope
        p_left = _clamp_probability(probabilities[left])
        p_right = _clamp_probability(probabilities[right])
        loss = 0.0
        for left_value in (0, 1):
            left_prob = p_left if left_value else 1.0 - p_left
            for right_value in (0, 1):
                if (left_value, right_value) not in self.allowed_set:
                    right_prob = p_right if right_value else 1.0 - p_right
                    loss += left_prob * right_prob
        return loss

    def message_to(self, variable_index: int, probabilities: ProbabilityVector) -> float:
        """Constraint-to-variable warning message.

        Positive values favor assigning the variable to 1; negative values favor
        0. The message is the difference between expected violation if the
        variable were clamped to 0 versus clamped to 1.
        """

        loss_if_zero = self._expected_violation_if(variable_index, 0, probabilities)
        loss_if_one = self._expected_violation_if(variable_index, 1, probabilities)
        return loss_if_zero - loss_if_one

    def to_validator_row(self) -> dict[str, Any]:
        """Return the Carnot validator-row shape accepted by the graph mapper."""

        return {
            "name": self.name,
            "scope": list(self.scope),
            "allowed": [list(pair) for pair in self.allowed],
        }

    def _expected_violation_if(
        self,
        variable_index: int,
        value: BinaryValue,
        probabilities: ProbabilityVector,
    ) -> float:
        left, right = self.scope
        if variable_index == left:
            other_index = right
            fixed_position = 0
        elif variable_index == right:
            other_index = left
            fixed_position = 1
        else:
            raise ValueError("variable_index must be part of the constraint scope")

        other_probability = _clamp_probability(probabilities[other_index])
        loss = 0.0
        for other_value in (0, 1):
            other_mass = other_probability if other_value else 1.0 - other_probability
            pair = (value, other_value) if fixed_position == 0 else (other_value, value)
            if pair not in self.allowed_set:
                loss += other_mass
        return loss


@dataclass(frozen=True)
class RunCSPGraph:
    """RUN-CSP-compatible bipartite graph plus an optional Carnot energy hook."""

    num_variables: int
    constraints: tuple[BinaryCSPConstraint, ...]
    energy_fn: EnergyCallback | None = None

    def __post_init__(self) -> None:
        if self.num_variables < 1:
            raise ValueError("num_variables must be positive")
        if not self.constraints:
            raise ValueError("at least one constraint is required")
        for constraint in self.constraints:
            for variable in constraint.scope:
                if variable < 0 or variable >= self.num_variables:
                    raise ValueError(f"constraint variable {variable} out of range")

    @property
    def num_constraints(self) -> int:
        """Number of constraint nodes in the bipartite architecture."""

        return len(self.constraints)

    @property
    def variable_degrees(self) -> tuple[int, ...]:
        """Return per-variable degree for normalized message aggregation."""

        degrees = [0 for _ in range(self.num_variables)]
        for constraint in self.constraints:
            left, right = constraint.scope
            degrees[left] += 1
            degrees[right] += 1
        return tuple(degrees)

    def message_passing_architecture(self) -> dict[str, Any]:
        """Describe the variable-constraint bipartite message architecture."""

        edges: list[tuple[int, int]] = []
        for constraint_index, constraint in enumerate(self.constraints):
            for variable_index in constraint.scope:
                edges.append((constraint_index, variable_index))
        return {
            "architecture": "run_csp_bipartite_message_passing",
            "num_variable_nodes": self.num_variables,
            "num_constraint_nodes": self.num_constraints,
            "bipartite_edges": edges,
            "message_types": ["constraint_to_variable", "variable_to_constraint"],
        }

    def table_energy(self, probabilities: ProbabilityVector) -> float:
        """Carnot-compatible energy from expected allowed-table violations."""

        if len(probabilities) != self.num_variables:
            raise ValueError("probabilities length must match num_variables")
        return float(sum(constraint.violation_probability(probabilities) for constraint in self.constraints))

    def energy(self, probabilities: ProbabilityVector) -> float:
        """Evaluate the supplied Carnot energy callback or the table energy."""

        if self.energy_fn is not None:
            return float(self.energy_fn(probabilities))
        return self.table_energy(probabilities)

    def discrete_energy(self, assignment: Sequence[int | bool]) -> float:
        """Count violated constraints for a discrete assignment."""

        if len(assignment) != self.num_variables:
            raise ValueError("assignment length must match num_variables")
        return float(sum(0 if constraint.is_satisfied(assignment) else 1 for constraint in self.constraints))

    def satisfaction_rate(self, assignment: Sequence[int | bool]) -> float:
        """Fraction of constraints satisfied by a discrete assignment."""

        return 1.0 - self.discrete_energy(assignment) / self.num_constraints

    def metadata(self) -> dict[str, Any]:
        """Return JSON-serializable graph metadata."""

        return {
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "num_bipartite_edges": 2 * self.num_constraints,
            "constraint_family": "pairwise_binary_allowed_table",
        }


@dataclass(frozen=True)
class RUNCSPSolverConfig:
    """Shared message-passing parameters for the unsupervised RUN-CSP solver."""

    epochs: int = 5
    message_steps: int = 32
    seed: int = 1972
    candidate_gains: tuple[float, ...] = (0.4, 0.8, 1.6, 2.4)
    damping: float = 0.7
    initial_noise: float = 0.35
    logit_clip: float = 8.0
    repair_sweeps: int = 8

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.message_steps < 1:
            raise ValueError("message_steps must be at least 1")
        if not self.candidate_gains or any(gain <= 0.0 for gain in self.candidate_gains):
            raise ValueError("candidate_gains must contain positive values")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if self.initial_noise < 0.0:
            raise ValueError("initial_noise must be non-negative")
        if self.logit_clip <= 0.0:
            raise ValueError("logit_clip must be positive")
        if self.repair_sweeps < 0:
            raise ValueError("repair_sweeps must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-facing copy of the solver configuration."""

        return {
            "epochs": self.epochs,
            "message_steps": self.message_steps,
            "seed": self.seed,
            "candidate_gains": list(self.candidate_gains),
            "damping": self.damping,
            "initial_noise": self.initial_noise,
            "logit_clip": self.logit_clip,
            "repair_sweeps": self.repair_sweeps,
        }


@dataclass(frozen=True)
class RunCSPResult:
    """Training or evaluation result for one binary-CSP graph."""

    num_variables: int
    num_constraints: int
    parameters: dict[str, float]
    probabilities: list[float]
    assignment: list[int]
    initial_energy: float
    final_energy: float
    continuous_energy: float
    normalized_energy: float
    satisfaction_rate: float
    history: list[dict[str, float]]
    labels_used: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable result."""

        return {
            "num_variables": self.num_variables,
            "num_constraints": self.num_constraints,
            "parameters": dict(self.parameters),
            "initial_energy": round(self.initial_energy, 6),
            "final_energy": round(self.final_energy, 6),
            "continuous_energy": round(self.continuous_energy, 6),
            "normalized_energy": round(self.normalized_energy, 6),
            "satisfaction_rate": round(self.satisfaction_rate, 6),
            "history": self.history,
            "labels_used": self.labels_used,
            "assignment_prefix": self.assignment[:16],
            "probability_prefix": [round(value, 6) for value in self.probabilities[:16]],
        }


class RUNCSPSolver:
    """Unsupervised RUN-CSP-style solver with shared message parameters."""

    def __init__(self, config: RUNCSPSolverConfig | None = None) -> None:
        self.config = config or RUNCSPSolverConfig()

    def train(self, graph: RunCSPGraph) -> RunCSPResult:
        """Train shared message gain using graph energy only, no labels."""

        uniform = [0.5 for _ in range(graph.num_variables)]
        initial_energy = graph.energy(uniform)
        best_result: RunCSPResult | None = None
        best_gain = self.config.candidate_gains[0]
        history = [
            {
                "epoch": 0.0,
                "message_gain": 0.0,
                "energy": round(initial_energy, 6),
                "satisfaction_rate": 0.0,
            }
        ]

        for epoch in range(1, self.config.epochs + 1):
            epoch_best: RunCSPResult | None = None
            for offset, gain in enumerate(self._candidate_gains_around(best_gain)):
                result = self._run_with_gain(
                    graph,
                    gain=gain,
                    seed=self.config.seed + epoch * 10_003 + offset,
                    initial_energy=initial_energy,
                    history=[],
                )
                if epoch_best is None or result.continuous_energy < epoch_best.continuous_energy:
                    epoch_best = result
            if best_result is None or epoch_best.continuous_energy <= best_result.continuous_energy:
                best_result = epoch_best
                best_gain = epoch_best.parameters["message_gain"]
            history.append(
                {
                    "epoch": float(epoch),
                    "message_gain": round(best_gain, 6),
                    "energy": round(best_result.continuous_energy, 6),
                    "satisfaction_rate": round(best_result.satisfaction_rate, 6),
                }
            )

        return self._with_history(best_result, history)

    def evaluate(self, graph: RunCSPGraph, parameters: Mapping[str, float]) -> RunCSPResult:
        """Evaluate learned shared parameters on another problem size."""

        uniform = [0.5 for _ in range(graph.num_variables)]
        return self._run_with_gain(
            graph,
            gain=float(parameters["message_gain"]),
            seed=self.config.seed + graph.num_variables * 997,
            initial_energy=graph.energy(uniform),
            history=[],
        )

    def _candidate_gains_around(self, center: float) -> tuple[float, ...]:
        configured = set(float(gain) for gain in self.config.candidate_gains)
        configured.update((center * 0.75, center, center * 1.25))
        return tuple(sorted(gain for gain in configured if gain > 0.0))

    def _run_with_gain(
        self,
        graph: RunCSPGraph,
        *,
        gain: float,
        seed: int,
        initial_energy: float,
        history: list[dict[str, float]],
    ) -> RunCSPResult:
        probabilities = self._message_pass(graph, gain=gain, seed=seed)
        continuous_energy = graph.energy(probabilities)
        assignment = [1 if probability >= 0.5 else 0 for probability in probabilities]
        assignment = self._repair_assignment(graph, assignment)
        final_energy = graph.discrete_energy(assignment)
        return RunCSPResult(
            num_variables=graph.num_variables,
            num_constraints=graph.num_constraints,
            parameters={"message_gain": round(gain, 6), "damping": self.config.damping},
            probabilities=probabilities,
            assignment=assignment,
            initial_energy=initial_energy,
            final_energy=final_energy,
            continuous_energy=continuous_energy,
            normalized_energy=final_energy / graph.num_constraints,
            satisfaction_rate=graph.satisfaction_rate(assignment),
            history=history,
            labels_used=False,
        )

    def _message_pass(self, graph: RunCSPGraph, *, gain: float, seed: int) -> list[float]:
        rng = random.Random(seed)
        logits = [
            rng.uniform(-self.config.initial_noise, self.config.initial_noise)
            for _ in range(graph.num_variables)
        ]
        degrees = graph.variable_degrees
        for _ in range(self.config.message_steps):
            probabilities = [_sigmoid(logit) for logit in logits]
            messages = [0.0 for _ in range(graph.num_variables)]
            for constraint in graph.constraints:
                left, right = constraint.scope
                messages[left] += constraint.message_to(left, probabilities)
                messages[right] += constraint.message_to(right, probabilities)
            for index, message in enumerate(messages):
                normalized = message / max(degrees[index], 1)
                logits[index] = _clip(
                    logits[index] + self.config.damping * gain * normalized,
                    -self.config.logit_clip,
                    self.config.logit_clip,
                )
        return [_sigmoid(logit) for logit in logits]

    def _repair_assignment(self, graph: RunCSPGraph, assignment: list[int]) -> list[int]:
        repaired = self._functional_projection(graph, assignment)
        current_energy = graph.discrete_energy(repaired)
        for _ in range(self.config.repair_sweeps):
            improved = False
            for variable_index in range(graph.num_variables):
                candidate = list(repaired)
                candidate[variable_index] = 1 - candidate[variable_index]
                candidate_energy = graph.discrete_energy(candidate)
                if candidate_energy < current_energy:
                    repaired = candidate
                    current_energy = candidate_energy
                    improved = True
            if not improved:
                break
        return repaired

    def _functional_projection(self, graph: RunCSPGraph, assignment: list[int]) -> list[int]:
        projected = list(assignment)
        adjacency = [[] for _ in range(graph.num_variables)]
        for constraint in graph.constraints:
            relation = _functional_relation(constraint)
            if relation is None:
                continue
            left, right = constraint.scope
            adjacency[left].append((right, relation[(left, right)]))
            adjacency[right].append((left, relation[(right, left)]))

        seen = [False for _ in range(graph.num_variables)]
        for root in range(graph.num_variables):
            if seen[root]:
                continue
            candidates = [
                self._project_component(root, value, adjacency, projected)
                for value in (projected[root], 1 - projected[root])
            ]
            projected = min(candidates, key=graph.discrete_energy)
            stack = [root]
            seen[root] = True
            while stack:
                current = stack.pop()
                for neighbor, _mapping in adjacency[current]:
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        stack.append(neighbor)
        return projected

    @staticmethod
    def _project_component(
        root: int,
        root_value: int,
        adjacency: Sequence[Sequence[tuple[int, Mapping[int, int]]]],
        assignment: list[int],
    ) -> list[int]:
        candidate = list(assignment)
        candidate[root] = root_value
        stack = [root]
        visited = {root}
        while stack:
            current = stack.pop()
            for neighbor, mapping in adjacency[current]:
                implied = mapping[candidate[current]]
                if neighbor not in visited:
                    candidate[neighbor] = implied
                    visited.add(neighbor)
                    stack.append(neighbor)
        return candidate

    @staticmethod
    def _with_history(result: RunCSPResult | None, history: list[dict[str, float]]) -> RunCSPResult:
        if result is None:
            raise RuntimeError("RUN-CSP training did not evaluate any candidate")
        return RunCSPResult(
            num_variables=result.num_variables,
            num_constraints=result.num_constraints,
            parameters=result.parameters,
            probabilities=result.probabilities,
            assignment=result.assignment,
            initial_energy=result.initial_energy,
            final_energy=result.final_energy,
            continuous_energy=result.continuous_energy,
            normalized_energy=result.normalized_energy,
            satisfaction_rate=result.satisfaction_rate,
            history=history,
            labels_used=result.labels_used,
        )


@dataclass(frozen=True)
class RunCSPExperimentConfig:
    """Configuration for the Exp 1972 train-on-40 evaluate-on-1000 run."""

    train_variables: int = 40
    eval_variable_counts: tuple[int, ...] = (40, 1000)
    edge_factor: int = 2
    train_seed: int = 1972
    eval_seed: int = 1973
    solver: RUNCSPSolverConfig = RUNCSPSolverConfig()

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-facing experiment controls."""

        return {
            "train_variables": self.train_variables,
            "eval_variable_counts": list(self.eval_variable_counts),
            "edge_factor": self.edge_factor,
            "train_seed": self.train_seed,
            "eval_seed": self.eval_seed,
            "solver": self.solver.as_dict(),
        }


def validator_graph_to_run_csp(
    num_variables: int,
    validator_constraints: Sequence[Mapping[str, Any] | BinaryCSPConstraint],
    *,
    energy_fn: EnergyCallback | None = None,
) -> RunCSPGraph:
    """Map Carnot validator rows into a RUN-CSP bipartite graph."""

    constraints: list[BinaryCSPConstraint] = []
    for index, row in enumerate(validator_constraints):
        if isinstance(row, BinaryCSPConstraint):
            constraints.append(row)
            continue
        scope = row.get("scope", row.get("variables"))
        if scope is None:
            raise ValueError("validator constraint must include scope or variables")
        allowed = row.get("allowed")
        if allowed is None:
            raise ValueError("validator constraint must include allowed table")
        constraints.append(
            BinaryCSPConstraint(
                name=str(row.get("name", f"constraint_{index}")),
                scope=_binary_scope(scope),
                allowed=tuple(_binary_pair(pair) for pair in allowed),
            )
        )
    return RunCSPGraph(
        num_variables=num_variables,
        constraints=tuple(constraints),
        energy_fn=energy_fn,
    )


def build_planted_binary_csp(
    *,
    num_variables: int,
    edge_factor: int = 2,
    seed: int = 1972,
) -> RunCSPGraph:
    """Build a deterministic planted pairwise binary-CSP graph."""

    if num_variables < 2:
        raise ValueError("num_variables must be at least 2")
    if edge_factor < 1:
        raise ValueError("edge_factor must be positive")
    rng = random.Random(seed)
    planted = [rng.randrange(2) for _ in range(num_variables)]
    edges = {(index, index + 1) for index in range(num_variables - 1)}
    target_edges = max(num_variables - 1, num_variables * edge_factor)
    while len(edges) < target_edges:
        left = rng.randrange(num_variables)
        right = rng.randrange(num_variables)
        if left == right:
            continue
        edges.add((min(left, right), max(left, right)))

    constraints = []
    for constraint_index, (left, right) in enumerate(sorted(edges)):
        parity = planted[left] ^ planted[right]
        allowed = ((0, 0), (1, 1)) if parity == 0 else ((0, 1), (1, 0))
        constraints.append(
            BinaryCSPConstraint(
                name=f"xor_{constraint_index}_{left}_{right}",
                scope=(left, right),
                allowed=allowed,
            )
        )
    return RunCSPGraph(num_variables=num_variables, constraints=tuple(constraints))


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_RESULT_PATH,
    config: RunCSPExperimentConfig | None = None,
) -> dict[str, Any]:
    """Run Exp 1972 and write the terminal RUN-CSP results artifact."""

    cfg = config or RunCSPExperimentConfig()
    output = Path(output_path)
    train_graph = build_planted_binary_csp(
        num_variables=cfg.train_variables,
        edge_factor=cfg.edge_factor,
        seed=cfg.train_seed,
    )
    solver = RUNCSPSolver(cfg.solver)
    trained = solver.train(train_graph)

    evaluations: dict[str, Any] = {}
    for offset, num_variables in enumerate(cfg.eval_variable_counts):
        graph = build_planted_binary_csp(
            num_variables=num_variables,
            edge_factor=cfg.edge_factor,
            seed=cfg.eval_seed + offset,
        )
        evaluations[str(num_variables)] = solver.evaluate(graph, trained.parameters).as_dict()

    artifact = {
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "run_date": RUN_DATE,
        "solver_name": "unsupervised_run_csp_message_passing",
        "paper_reference": PAPER_REFERENCE,
        "problem_family": "planted_pairwise_binary_csp",
        "config": cfg.as_dict(),
        "training": {
            "graph": train_graph.metadata(),
            "result": trained.as_dict(),
            "history": trained.history,
            "learned_parameters": trained.parameters,
        },
        "evaluations": evaluations,
        "labels_used": False,
        "cpu_only": True,
        "hardware_execution_performed": False,
        "network_access_used": False,
        "artifact_path": str(output),
        "honest_verdict": _honest_verdict(evaluations),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _binary_scope(scope: Any) -> tuple[int, int]:
    values = tuple(int(value) for value in scope)
    if len(values) != 2:
        raise ValueError("scope must contain exactly two variables")
    return values


def _binary_pair(pair: Any) -> tuple[int, int]:
    values = tuple(int(value) for value in pair)
    if len(values) != 2 or any(value not in (0, 1) for value in values):
        raise ValueError("allowed table entries must be binary values")
    return values


def _functional_relation(
    constraint: BinaryCSPConstraint,
) -> dict[tuple[int, int], dict[int, int]] | None:
    by_left: dict[int, list[int]] = {0: [], 1: []}
    by_right: dict[int, list[int]] = {0: [], 1: []}
    for left_value, right_value in constraint.allowed:
        by_left[left_value].append(right_value)
        by_right[right_value].append(left_value)
    if any(len(values) != 1 for values in by_left.values()):
        return None
    if any(len(values) != 1 for values in by_right.values()):
        return None
    left, right = constraint.scope
    return {
        (left, right): {value: values[0] for value, values in by_left.items()},
        (right, left): {value: values[0] for value, values in by_right.items()},
    }


def _clamp_probability(value: float) -> float:
    return _clip(float(value), 1e-9, 1.0 - 1e-9)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _honest_verdict(evaluations: Mapping[str, Mapping[str, Any]]) -> str:
    thousand = evaluations.get("1000")
    if thousand and float(thousand["satisfaction_rate"]) >= 0.95:
        return "complete_generalized_40_to_1000_cpu_only"
    return "complete_partial_generalization_cpu_only"
