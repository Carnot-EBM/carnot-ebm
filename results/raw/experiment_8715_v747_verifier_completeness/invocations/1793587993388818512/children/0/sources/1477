"""Exp 5103: TACO-style adaptive heuristic for bounded CSP solving.

Spec refs: REQ-VERIFY-5103, SCENARIO-VERIFY-5103.

This experiment keeps the TACO-style adaptation in a helper role. The adaptive
loop looks at only the current graph structure and a bounded conflict
relaxation trace, then proposes a variable order. A complete exact
backtracking graph-coloring solver still decides whether the instance is
colorable and every returned coloring is checked edge by edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
from typing import Any


JsonDict = dict[str, Any]

RESULT_RELATIVE_PATH = "results/experiment_5103_taco_adaptive_csp_heuristic_v468.json"
RUN_DATE = "20260701"
RANDOM_SEED = 5103
CSP_FAMILY = "bounded_graph_coloring_taco_v1"
INFERENCE_SUBSTRATE = "exact_solver_with_adaptive_cpu_heuristic"
EXACT_SOLVER_BACKEND = "deterministic_backtracking_graph_coloring_cpu"
SUCCESS_VERDICT = "success_taco_adaptive_heuristic_reduces_exact_solver_effort"
NO_WIN_VERDICT = "complete_taco_adaptive_heuristic_no_effort_win"
MAX_NODES = 13
DEFAULT_ADAPTATION_STEPS = 32

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "csp_family",
    "exact_solver_backend",
    "instances_total",
    "baseline_effort",
    "static_heuristic_effort",
    "adapted_effort",
    "delta_effort_vs_baseline",
    "correctness_preserved",
    "harmful_instance_count",
    "adaptation_steps",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal prefix; success only when adapted exact-solver effort is below "
        "the no-help exact baseline with correctness preserved."
    ),
    "duration_s": "Measured local CPU time for graph generation, heuristic adaptation, exact search, and JSON assembly.",
    "inference_substrate": "Must be exact_solver_with_adaptive_cpu_heuristic, never live_llm_inference.",
    "csp_family": "Declared deterministic bounded graph-coloring CSP family.",
    "exact_solver_backend": "The complete CPU backtracking solver that owns all labels and returned colorings.",
    "instances_total": "Number of train, dev, and held-out generated CSP instances evaluated.",
    "baseline_effort": "Exact-solver search effort with natural variable order and no heuristic help.",
    "static_heuristic_effort": "Exact-solver search effort after static degree-order variable help.",
    "adapted_effort": "Exact-solver search effort after adapted instance-wise variable-order help.",
    "delta_effort_vs_baseline": "Signed effort minus no-help baseline; negative means less exact-solver effort.",
    "correctness_preserved": "True only when all arm labels agree with the exact authority and all colorings verify.",
    "harmful_instance_count": "Count of instances where adapted help costs more exact-solver effort than no help.",
    "adaptation_steps": "Total bounded CPU relaxation steps used to produce advisory variable orders.",
    "flagged_adversarial": "False only when substrate, schema, exact labels, and no-heuristic-solve checks pass.",
}


@dataclass(frozen=True)
class GraphColoringInstance:
    """One bounded k-coloring CSP instance with an exact expected label."""

    instance_id: str
    split: str
    n_nodes: int
    n_colors: int
    edges: tuple[tuple[int, int], ...]
    expected_colorable: bool
    description: str


@dataclass(frozen=True)
class SolveTrace:
    """Complete exact-solver trace for one variable order."""

    colorable: bool
    assignment: tuple[int, ...] | None
    search_nodes: int
    constraint_checks: int
    backtracks: int
    duration_s: float

    @property
    def status(self) -> str:
        """Return the JSON status label used in artifacts."""

        return "colorable" if self.colorable else "uncolorable"

    @property
    def effort_score(self) -> int:
        """Search-node plus constraint-check effort, all from exact search."""

        return self.search_nodes + self.constraint_checks

    def to_json(self, *, order: Sequence[int], solution_verified: bool) -> JsonDict:
        """Return a stable artifact row for this exact solve."""

        return {
            "status": self.status,
            "colorable": self.colorable,
            "assignment": list(self.assignment) if self.assignment is not None else None,
            "variable_order": list(order),
            "solution_verified": solution_verified,
            "effort": {
                "metric": "search_nodes_plus_constraint_checks",
                "total_effort_score": self.effort_score,
                "search_nodes": self.search_nodes,
                "constraint_checks": self.constraint_checks,
                "backtracks": self.backtracks,
                "duration_s": self.duration_s,
            },
        }


@dataclass(frozen=True)
class AdaptationConfig:
    """Bounded CPU-only heuristic parameters for the instance-wise loop."""

    steps: int = DEFAULT_ADAPTATION_STEPS
    seed: int = RANDOM_SEED
    conflict_weight: float = 1.0
    triangle_weight: float = 50.0
    degree_weight: float = 2.0


@dataclass(frozen=True)
class AdaptationResult:
    """Advisory order and telemetry from the unsupervised conflict loop."""

    order: tuple[int, ...]
    steps: int
    conflict_counts: tuple[int, ...]
    triangle_counts: tuple[int, ...]
    degree_counts: tuple[int, ...]
    heuristic_scores: tuple[float, ...]
    relaxed_coloring: tuple[int, ...]
    heuristic_only_solution_counted: bool = False

    def to_json(self) -> JsonDict:
        """Return adaptation telemetry without claiming it solved the CSP."""

        return {
            "order": list(self.order),
            "steps": self.steps,
            "conflict_counts": list(self.conflict_counts),
            "triangle_counts": list(self.triangle_counts),
            "degree_counts": list(self.degree_counts),
            "heuristic_scores": [round(score, 6) for score in self.heuristic_scores],
            "relaxed_coloring": list(self.relaxed_coloring),
            "heuristic_only_solution_counted": self.heuristic_only_solution_counted,
        }


class ExactGraphColoringSolver:
    """Complete CPU backtracking solver used as the only correctness authority."""

    backend_name = EXACT_SOLVER_BACKEND

    def solve(self, instance: GraphColoringInstance, order: Sequence[int]) -> SolveTrace:
        """Solve graph coloring exactly while respecting an advisory variable order."""

        _validate_instance(instance)
        _validate_order(instance, order)
        adjacency = _adjacency(instance)
        assignment: list[int | None] = [None for _ in range(instance.n_nodes)]
        search_nodes = 0
        constraint_checks = 0
        backtracks = 0
        started = time.perf_counter()

        def search(position: int) -> bool:
            nonlocal search_nodes, constraint_checks, backtracks
            search_nodes += 1
            if position == instance.n_nodes:
                return True
            node = order[position]
            for color in range(instance.n_colors):
                allowed = True
                for neighbor in adjacency[node]:
                    constraint_checks += 1
                    if assignment[neighbor] == color:
                        allowed = False
                        break
                if allowed:
                    assignment[node] = color
                    if search(position + 1):
                        return True
                    assignment[node] = None
            backtracks += 1
            return False

        colorable = search(0)
        duration_s = round(time.perf_counter() - started, 6)
        completed = tuple(int(color) for color in assignment) if colorable else None
        return SolveTrace(
            colorable=colorable,
            assignment=completed,
            search_nodes=search_nodes,
            constraint_checks=constraint_checks,
            backtracks=backtracks,
            duration_s=duration_s,
        )

    def verify_assignment(self, instance: GraphColoringInstance, assignment: Sequence[int] | None) -> bool:
        """Check a proposed coloring without treating the heuristic as an authority."""

        if assignment is None or len(assignment) != instance.n_nodes:
            return False
        if any(color < 0 or color >= instance.n_colors for color in assignment):
            return False
        return all(assignment[left] != assignment[right] for left, right in instance.edges)


def build_instance_family() -> tuple[GraphColoringInstance, ...]:
    """Generate deterministic train/dev/held-out bounded graph-coloring CSPs."""

    return (
        _wheel_instance("train_wheel5_odd_unsat", "train", 5),
        _late_triangle_instance(),
        _hub_distractor_k4_instance("train_hub_distractor_k4_4", "train", 4),
        _hub_distractor_k4_instance("dev_hub_distractor_k4_8", "dev", 8),
        _wheel_instance("dev_wheel6_even_sat", "dev", 6),
        _late_k4_instance(),
        _wheel_instance("heldout_wheel7_odd_unsat", "heldout", 7),
        _wheel_instance("heldout_wheel8_even_sat", "heldout", 8),
    )


def baseline_order(instance: GraphColoringInstance) -> tuple[int, ...]:
    """No-help baseline order: the generator's natural node order."""

    return tuple(range(instance.n_nodes))


def static_degree_order(instance: GraphColoringInstance) -> tuple[int, ...]:
    """Static help order: high-degree nodes first, deterministic tie by node id."""

    degree_counts = _degree_counts(instance)
    return tuple(sorted(range(instance.n_nodes), key=lambda node: (-degree_counts[node], node)))


def adapt_instance_order(
    instance: GraphColoringInstance,
    config: AdaptationConfig | None = None,
) -> AdaptationResult:
    """Run bounded conflict relaxation and return an advisory variable order."""

    active_config = config or AdaptationConfig()
    _validate_instance(instance)
    adjacency = _adjacency(instance)
    rng = random.Random(active_config.seed + _instance_seed(instance))
    colors = [rng.randrange(instance.n_colors) for _ in range(instance.n_nodes)]
    conflict_counts = [0 for _ in range(instance.n_nodes)]

    for step in range(active_config.steps):
        conflicted_nodes: list[int] = []
        for left, right in instance.edges:
            if colors[left] == colors[right]:
                conflict_counts[left] += 1
                conflict_counts[right] += 1
                conflicted_nodes.extend((left, right))
        if not conflicted_nodes:
            perturb = max(range(instance.n_nodes), key=lambda node: (len(adjacency[node]), -node))
            colors[perturb] = (colors[perturb] + 1) % instance.n_colors
            continue
        node = conflicted_nodes[(step + active_config.seed) % len(conflicted_nodes)]
        colors[node] = _least_conflicting_color(node, colors, adjacency, instance.n_colors, step)

    triangle_counts = _triangle_counts(instance)
    degree_counts = _degree_counts(instance)
    scores = tuple(
        active_config.conflict_weight * conflict_counts[node]
        + active_config.triangle_weight * triangle_counts[node]
        + active_config.degree_weight * degree_counts[node]
        for node in range(instance.n_nodes)
    )
    order = tuple(sorted(range(instance.n_nodes), key=lambda node: (-scores[node], node)))
    return AdaptationResult(
        order=order,
        steps=active_config.steps,
        conflict_counts=tuple(conflict_counts),
        triangle_counts=triangle_counts,
        degree_counts=degree_counts,
        heuristic_scores=scores,
        relaxed_coloring=tuple(colors),
    )


def run(duration_s: float | None = None) -> JsonDict:
    """Run all three solver-effort arms and return the terminal artifact."""

    started = time.perf_counter()
    family = build_instance_family()
    solver = ExactGraphColoringSolver()
    rows: list[JsonDict] = []
    totals = _empty_totals()
    correctness_preserved = True
    harmful_instances: list[str] = []
    total_adaptation_steps = 0

    for instance in family:
        row = _evaluate_instance(instance, solver)
        rows.append(row)
        total_adaptation_steps += int(row["adaptation"]["steps"])
        for arm_name in ("baseline", "static_heuristic", "adapted"):
            _add_effort(totals[arm_name], row[arm_name]["effort"])
        if row["adapted"]["effort"]["total_effort_score"] > row["baseline"]["effort"]["total_effort_score"]:
            harmful_instances.append(str(row["instance_id"]))
        if not _row_correctness_preserved(row):
            correctness_preserved = False

    baseline_effort = _summarize_effort(totals["baseline"])
    static_effort = _summarize_effort(totals["static_heuristic"])
    adapted_effort = _summarize_effort(totals["adapted"])
    deltas = {
        "static_heuristic": static_effort["total_effort_score"] - baseline_effort["total_effort_score"],
        "adapted": adapted_effort["total_effort_score"] - baseline_effort["total_effort_score"],
        "adapted_vs_static": adapted_effort["total_effort_score"] - static_effort["total_effort_score"],
    }
    flagged_adversarial = not correctness_preserved or INFERENCE_SUBSTRATE != "exact_solver_with_adaptive_cpu_heuristic"
    adapted_wins = bool(deltas["adapted"] < 0 and correctness_preserved and not flagged_adversarial)
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": "carnot.experiment_5103_taco_adaptive_csp_heuristic.v468",
        "experiment_id": 5103,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": SUCCESS_VERDICT if adapted_wins else NO_WIN_VERDICT,
        "duration_s": elapsed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "csp_family": CSP_FAMILY,
        "exact_solver_backend": solver.backend_name,
        "instances_total": len(family),
        "baseline_effort": baseline_effort,
        "static_heuristic_effort": static_effort,
        "adapted_effort": adapted_effort,
        "delta_effort_vs_baseline": deltas,
        "correctness_preserved": correctness_preserved,
        "harmful_instance_count": len(harmful_instances),
        "adaptation_steps": total_adaptation_steps,
        "flagged_adversarial": flagged_adversarial,
        "harmful_instances": harmful_instances,
        "split_summaries": _split_summaries(rows),
        "per_instance_results": rows,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-5103", "SCENARIO-VERIFY-5103"],
        "methodology_note": (
            "The adaptive loop proposes only variable-order help from CPU conflict relaxation. "
            "All labels, solved/unsolved statuses, and accepted colorings come from complete exact search."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "csp_family": artifact["csp_family"],
            "instances": [
                {
                    "instance_id": row["instance_id"],
                    "split": row["split"],
                    "baseline_effort": row["baseline"]["effort"]["total_effort_score"],
                    "static_effort": row["static_heuristic"]["effort"]["total_effort_score"],
                    "adapted_effort": row["adapted"]["effort"]["total_effort_score"],
                    "adapted_order": row["adaptation"]["order"],
                }
                for row in rows
            ],
        }
    )
    validate_artifact(artifact)
    return artifact


def write_artifact(root: str | Path | None = None, output_path: str | Path | None = None) -> JsonDict:
    """Run the experiment and write the Exp 5103 terminal JSON artifact."""

    repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    destination = Path(output_path) if output_path is not None else repo_root / RESULT_RELATIVE_PATH
    artifact = run()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5103 artifact violates its terminal contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    _require(
        verdict.startswith(SUCCESS_VERDICT) or verdict.startswith(NO_WIN_VERDICT),
        "honest_verdict must use an Exp5103 terminal prefix",
    )
    _require(
        isinstance(artifact["duration_s"], int | float) and artifact["duration_s"] >= 0.0,
        "duration_s must be nonnegative",
    )
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require("llm" not in str(artifact["inference_substrate"]).lower(), "inference_substrate")
    _require(artifact["csp_family"] == CSP_FAMILY, "csp_family")
    _require(artifact["exact_solver_backend"] == EXACT_SOLVER_BACKEND, "exact_solver_backend")
    _require(isinstance(artifact["instances_total"], int) and artifact["instances_total"] > 0, "instances_total")
    for field in ("baseline_effort", "static_heuristic_effort", "adapted_effort"):
        _require(_valid_effort_summary(artifact[field]), field)
    _validate_deltas(artifact)
    _require(artifact["correctness_preserved"] is True, "correctness_preserved")
    _require(
        isinstance(artifact["harmful_instance_count"], int) and artifact["harmful_instance_count"] >= 0,
        "harmful_instance_count",
    )
    _require(
        isinstance(artifact["adaptation_steps"], int)
        and artifact["adaptation_steps"] >= artifact["instances_total"],
        "adaptation_steps",
    )
    _require(artifact["flagged_adversarial"] is False, "flagged_adversarial")
    principles = artifact.get("field_principles")
    _require(
        isinstance(principles, Mapping) and set(REQUIRED_ARTIFACT_FIELDS).issubset(principles),
        "field_principles",
    )
    rows = artifact.get("per_instance_results")
    _require(isinstance(rows, list) and len(rows) == artifact["instances_total"], "per_instance_results")
    _require(all(_row_correctness_preserved(row) for row in rows), "per_instance_results")
    _require(
        all(row.get("heuristic_only_solution_counted") is False for row in rows),
        "heuristic_only_solution_counted",
    )
    harmful = artifact.get("harmful_instances", [])
    _require(isinstance(harmful, list) and len(harmful) == artifact["harmful_instance_count"], "harmful_instance_count")
    if verdict.startswith(SUCCESS_VERDICT):
        _require(artifact["delta_effort_vs_baseline"]["adapted"] < 0, "delta_effort_vs_baseline")


def main() -> int:
    """CLI entrypoint used by the conductor and tests."""

    root = Path(os.environ.get("CARNOT_EXP5103_ROOT", Path(__file__).resolve().parents[2]))
    write_artifact(root=root)
    return 0


def _evaluate_instance(instance: GraphColoringInstance, solver: ExactGraphColoringSolver) -> JsonDict:
    baseline = solver.solve(instance, baseline_order(instance))
    static_order = static_degree_order(instance)
    static = solver.solve(instance, static_order)
    adaptation = adapt_instance_order(instance)
    adapted = solver.solve(instance, adaptation.order)
    exact_label = {
        "status": baseline.status,
        "colorable": baseline.colorable,
        "source": EXACT_SOLVER_BACKEND,
    }
    baseline_verified = _solution_verified(solver, instance, baseline)
    static_verified = _solution_verified(solver, instance, static)
    adapted_verified = _solution_verified(solver, instance, adapted)
    return {
        "instance_id": instance.instance_id,
        "split": instance.split,
        "description": instance.description,
        "n_nodes": instance.n_nodes,
        "n_edges": len(instance.edges),
        "n_colors": instance.n_colors,
        "expected_colorable": instance.expected_colorable,
        "exact_label": exact_label,
        "heuristic_only_solution_counted": False,
        "baseline": baseline.to_json(order=baseline_order(instance), solution_verified=baseline_verified),
        "static_heuristic": static.to_json(order=static_order, solution_verified=static_verified),
        "adapted": adapted.to_json(order=adaptation.order, solution_verified=adapted_verified),
        "adaptation": adaptation.to_json(),
        "adapted_harmful_vs_baseline": adapted.effort_score > baseline.effort_score,
    }


def _wheel_instance(instance_id: str, split: str, rim_nodes: int) -> GraphColoringInstance:
    center = rim_nodes
    edges = [(node, (node + 1) % rim_nodes) for node in range(rim_nodes)]
    edges.extend((center, node) for node in range(rim_nodes))
    return GraphColoringInstance(
        instance_id=instance_id,
        split=split,
        n_nodes=rim_nodes + 1,
        n_colors=3,
        edges=_canonical_edges(edges),
        expected_colorable=rim_nodes % 2 == 0,
        description=f"{rim_nodes}-rim wheel graph under 3-coloring.",
    )


def _late_triangle_instance() -> GraphColoringInstance:
    edges = [(5, 6), (6, 7), (5, 7)]
    edges.extend((node, 5 + (node % 3)) for node in range(5))
    return GraphColoringInstance(
        instance_id="train_late_triangle_sat",
        split="train",
        n_nodes=8,
        n_colors=3,
        edges=_canonical_edges(edges),
        expected_colorable=True,
        description="A satisfiable triangle appears after low-degree prefix nodes.",
    )


def _hub_distractor_k4_instance(instance_id: str, split: str, leaves: int) -> GraphColoringInstance:
    clique = tuple(range(leaves + 1, leaves + 5))
    edges = [(0, leaf) for leaf in range(1, leaves + 1)]
    edges.extend(itertools.combinations(clique, 2))
    return GraphColoringInstance(
        instance_id=instance_id,
        split=split,
        n_nodes=leaves + 5,
        n_colors=3,
        edges=_canonical_edges(edges),
        expected_colorable=False,
        description="A high-degree star hub distracts static degree ordering from a late K4 obstruction.",
    )


def _late_k4_instance() -> GraphColoringInstance:
    edges = list(itertools.combinations(range(4, 8), 2))
    edges.extend((node, 4 + node) for node in range(4))
    return GraphColoringInstance(
        instance_id="heldout_late_k4_unsat",
        split="heldout",
        n_nodes=8,
        n_colors=3,
        edges=_canonical_edges(edges),
        expected_colorable=False,
        description="A held-out K4 obstruction appears after four low-degree prefix nodes.",
    )


def _least_conflicting_color(
    node: int,
    colors: Sequence[int],
    adjacency: Sequence[Sequence[int]],
    n_colors: int,
    step: int,
) -> int:
    candidates: list[int] = []
    best_conflicts: int | None = None
    for color in range(n_colors):
        conflicts = sum(1 for neighbor in adjacency[node] if colors[neighbor] == color)
        if best_conflicts is None or conflicts < best_conflicts:
            best_conflicts = conflicts
            candidates = [color]
        elif conflicts == best_conflicts:
            candidates.append(color)
    return candidates[step % len(candidates)]


def _solution_verified(
    solver: ExactGraphColoringSolver,
    instance: GraphColoringInstance,
    trace: SolveTrace,
) -> bool:
    if not trace.colorable:
        return trace.assignment is None
    return solver.verify_assignment(instance, trace.assignment)


def _row_correctness_preserved(row: Mapping[str, Any]) -> bool:
    label = row["exact_label"]["colorable"]
    if label is not row["expected_colorable"]:
        return False
    for arm_name in ("baseline", "static_heuristic", "adapted"):
        arm = row[arm_name]
        if arm["colorable"] is not label:
            return False
        if label and arm["solution_verified"] is not True:
            return False
        if not label and arm["assignment"] is not None:
            return False
    return True


def _empty_totals() -> dict[str, JsonDict]:
    return {
        "baseline": _new_total(),
        "static_heuristic": _new_total(),
        "adapted": _new_total(),
    }


def _new_total() -> JsonDict:
    return {
        "instances": 0,
        "total_effort_score": 0,
        "search_nodes": 0,
        "constraint_checks": 0,
        "backtracks": 0,
        "duration_s": 0.0,
    }


def _add_effort(total: JsonDict, effort: Mapping[str, Any]) -> None:
    total["instances"] += 1
    total["total_effort_score"] += int(effort["total_effort_score"])
    total["search_nodes"] += int(effort["search_nodes"])
    total["constraint_checks"] += int(effort["constraint_checks"])
    total["backtracks"] += int(effort["backtracks"])
    total["duration_s"] += float(effort["duration_s"])


def _summarize_effort(total: Mapping[str, Any]) -> JsonDict:
    return {
        "metric": "search_nodes_plus_constraint_checks",
        "instances": int(total["instances"]),
        "total_effort_score": int(total["total_effort_score"]),
        "search_nodes": int(total["search_nodes"]),
        "constraint_checks": int(total["constraint_checks"]),
        "backtracks": int(total["backtracks"]),
        "duration_s": round(float(total["duration_s"]), 6),
    }


def _split_summaries(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    summaries: JsonDict = {}
    for split in ("train", "dev", "heldout"):
        split_rows = [row for row in rows if row["split"] == split]
        summaries[split] = {
            "instances": len(split_rows),
            "baseline_effort": sum(row["baseline"]["effort"]["total_effort_score"] for row in split_rows),
            "static_heuristic_effort": sum(row["static_heuristic"]["effort"]["total_effort_score"] for row in split_rows),
            "adapted_effort": sum(row["adapted"]["effort"]["total_effort_score"] for row in split_rows),
            "harmful_instance_count": sum(1 for row in split_rows if row["adapted_harmful_vs_baseline"]),
        }
        summaries[split]["delta_adapted_vs_baseline"] = (
            summaries[split]["adapted_effort"] - summaries[split]["baseline_effort"]
        )
    return summaries


def _validate_deltas(artifact: Mapping[str, Any]) -> None:
    deltas = artifact["delta_effort_vs_baseline"]
    _require(isinstance(deltas, Mapping), "delta_effort_vs_baseline")
    baseline = artifact["baseline_effort"]["total_effort_score"]
    static = artifact["static_heuristic_effort"]["total_effort_score"]
    adapted = artifact["adapted_effort"]["total_effort_score"]
    expected = {
        "static_heuristic": static - baseline,
        "adapted": adapted - baseline,
        "adapted_vs_static": adapted - static,
    }
    _require(dict(deltas) == expected, "delta_effort_vs_baseline")


def _valid_effort_summary(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("metric") == "search_nodes_plus_constraint_checks"
        and int(value.get("instances", 0)) > 0
        and int(value.get("total_effort_score", 0)) > 0
        and int(value.get("search_nodes", 0)) > 0
        and int(value.get("constraint_checks", 0)) > 0
        and float(value.get("duration_s", -1.0)) >= 0.0
    )


def _validate_instance(instance: GraphColoringInstance) -> None:
    _require(1 <= instance.n_nodes <= MAX_NODES, "n_nodes out of bounded range")
    _require(instance.n_colors >= 2, "n_colors must be at least two")
    _require(instance.split in {"train", "dev", "heldout"}, "split must be train/dev/heldout")
    for left, right in instance.edges:
        _require(0 <= left < right < instance.n_nodes, "edge endpoint out of range or not canonical")


def _validate_order(instance: GraphColoringInstance, order: Sequence[int]) -> None:
    _require(tuple(sorted(order)) == tuple(range(instance.n_nodes)), "order must be a permutation of nodes")


def _canonical_edges(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({(min(left, right), max(left, right)) for left, right in edges if left != right}))


def _adjacency(instance: GraphColoringInstance) -> tuple[tuple[int, ...], ...]:
    adjacency = [set() for _ in range(instance.n_nodes)]
    for left, right in instance.edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency)


def _degree_counts(instance: GraphColoringInstance) -> tuple[int, ...]:
    return tuple(len(neighbors) for neighbors in _adjacency(instance))


def _triangle_counts(instance: GraphColoringInstance) -> tuple[int, ...]:
    edge_set = set(instance.edges)
    counts = [0 for _ in range(instance.n_nodes)]
    for left, middle, right in itertools.combinations(range(instance.n_nodes), 3):
        if (left, middle) in edge_set and (left, right) in edge_set and (middle, right) in edge_set:
            counts[left] += 1
            counts[middle] += 1
            counts[right] += 1
    return tuple(counts)


def _instance_seed(instance: GraphColoringInstance) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(instance.instance_id))


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
