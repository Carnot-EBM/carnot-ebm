"""Exp 3211 ConstraintBench-style feasibility/objective exact pilot.

Spec refs: REQ-BENCH-3211, SCENARIO-BENCH-3211.

This module is deliberately smaller than ConstraintBench itself. It builds a
local fixture with exact bounded references, then scores deterministic candidate
answers with feasibility and objective quality kept separate. That separation
matters because a model can output a high-value answer that is unusable once a
capacity, assignment, or coloring constraint is checked.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.constraintbench_feasibility_objective_pilot.v1"
EXPERIMENT_ID = "exp3211"
MILESTONE = "2026.05.297"
FIXTURE_REL_PATH = Path("data/research/constraintbench_feasibility_objective_pilot_v1.jsonl")
ARTIFACT_REL_PATH = Path(
    "results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json"
)

MANDATED_OPTIONAL_SMOKE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REFERENCE_PAPERS: tuple[JsonDict, ...] = (
    {
        "title": "ConstraintBench: Benchmarking LLM Constraint Reasoning on Direct Optimization",
        "source": "https://arxiv.org/abs/2602.22465",
        "used_for": "separate feasibility and objective-quality scoring",
    },
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "experiment_id",
    "milestone",
    "reference_papers",
    "fixture_path",
    "fixture_count",
    "optimization_families",
    "exact_solver_backends",
    "feasibility_metric_defined",
    "objective_gap_metric_defined",
    "hallucinated_entity_metric_defined",
    "optional_llm_smoke",
    "ready_for_clean_verifier",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)


def build_fixture_rows() -> list[JsonDict]:
    """REQ-BENCH-3211: materialize 15 exact rows across three small families."""

    rows: list[JsonDict] = []
    for builder in (_knapsack_rows, _assignment_rows, _graph_coloring_rows):
        for row in builder():
            solved = solve_row(row)
            rows.append(
                {
                    **row,
                    "schema_version": "carnot.constraintbench_feasibility_objective.row.v1",
                    "exact_reference": solved,
                    "checker": {
                        "authority": "local_exhaustive_enumeration",
                        "backend": row["checker_backend"],
                        "candidate_schema": row["candidate_schema"],
                    },
                }
            )
    return rows


def write_fixture(rows: Sequence[Mapping[str, Any]], path: Path | str) -> JsonDict:
    """Persist the JSONL fixture and return a small manifest summary."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(_canonical_json(dict(row)) + "\n" for row in rows)
    output.write_text(text, encoding="utf-8")
    return {
        "schema_version": "carnot.constraintbench_feasibility_objective.fixture.v1",
        "fixture_path": str(output),
        "fixture_count": len(rows),
        "optimization_families": sorted({str(row["family"]) for row in rows}),
        "exact_solver_backends": sorted({str(row["checker_backend"]) for row in rows}),
    }


def solve_row(row: Mapping[str, Any]) -> JsonDict:
    """Solve one fixture row with its exact bounded checker."""

    family = row.get("family")
    if family == "knapsack":
        return _solve_knapsack(row)
    if family == "assignment":
        return _solve_assignment(row)
    if family == "graph_coloring":
        return _solve_graph_coloring(row)
    return {"feasible": False, "objective_value": None, "solution": {}, "feasible_count": 0}


def feasible_nonoptimal_solution(row: Mapping[str, Any]) -> JsonDict:
    """Return a feasible answer with worse objective value than the exact reference."""

    exact = row["exact_reference"]
    optimum = int(exact["objective_value"])
    sense = str(row["objective"]["sense"])
    for solution in _enumerate_feasible_solutions(row):
        value = _objective_value(row, solution)
        if (sense == "max" and value < optimum) or (sense == "min" and value > optimum):
            return solution
    raise ValueError("fixture row has no feasible nonoptimal solution")  # pragma: no cover


def score_candidate(row: Mapping[str, Any], candidate_text: str) -> JsonDict:
    """SCENARIO-BENCH-3211: score format, entity, feasibility, and objective gap separately."""

    if row.get("family") not in {"knapsack", "assignment", "graph_coloring"}:
        return _score_failure(row, False, "invalid_format", ["unknown_family"])

    try:
        parsed = json.loads(candidate_text)
    except json.JSONDecodeError:
        return _score_failure(row, False, "invalid_format", ["invalid_json"])
    if not isinstance(parsed, Mapping):
        return _score_failure(row, False, "invalid_format", ["candidate_not_object"])

    normalized, format_error = _normalize_candidate(row, parsed)
    if format_error is not None:
        return _score_failure(row, False, "invalid_format", [format_error])

    hallucinated, hallucination_reasons = _hallucinated_entities(row, normalized)
    if hallucinated:
        return _score_failure(row, True, "hallucinated_entity", hallucination_reasons)

    complete, completeness_reasons = _entity_slots_complete(row, normalized)
    if not complete:
        return _score_failure(row, True, "entity_omission", completeness_reasons)

    feasible, reasons = _check_feasibility(row, normalized)
    if not feasible:
        return _score_failure(row, True, "missing_constraint", reasons)

    objective_value = _objective_value(row, normalized)
    gap = _objective_gap(row, objective_value)
    return _score_base(row) | {
        "valid_format": True,
        "invalid_format": False,
        "hallucinated_entity": False,
        "missing_constraint": False,
        "feasibility_pass": True,
        "objective_value": objective_value,
        "objective_gap": float(gap),
        "reasons": [],
        "parsed_solution": normalized,
    }


def aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate exact-pilot metrics without collapsing feasibility into format."""

    total = len(rows)
    if total == 0:
        return {
            "candidate_count": 0,
            "valid_format_rate": 0.0,
            "feasibility_pass_rate": 0.0,
            "objective_gap_mean_feasible": None,
            "hallucinated_entity_rate": 0.0,
            "missing_constraint_rate": 0.0,
            "invalid_format_rate": 0.0,
        }
    gaps = [float(row["objective_gap"]) for row in rows if row.get("objective_gap") is not None]
    return {
        "candidate_count": total,
        "valid_format_rate": _rate(sum(row.get("valid_format") is True for row in rows), total),
        "feasibility_pass_rate": _rate(
            sum(row.get("feasibility_pass") is True for row in rows), total
        ),
        "objective_gap_mean_feasible": round(sum(gaps) / len(gaps), 6) if gaps else None,
        "hallucinated_entity_rate": _rate(
            sum(row.get("hallucinated_entity") is True for row in rows), total
        ),
        "missing_constraint_rate": _rate(
            sum(row.get("missing_constraint") is True for row in rows), total
        ),
        "invalid_format_rate": _rate(sum(row.get("invalid_format") is True for row in rows), total),
    }


def build_artifact(root: Path | str = REPO_ROOT, tests_run: Sequence[str] = ()) -> JsonDict:
    """Build the terminal artifact and write the fixture JSONL it references."""

    root_path = Path(root)
    fixture_rows = build_fixture_rows()
    fixture_path = root_path / FIXTURE_REL_PATH
    write_fixture(fixture_rows, fixture_path)
    candidate_scores = [
        score_candidate(row, _candidate_text_for_fixture_index(row, index))
        for index, row in enumerate(fixture_rows)
    ]
    metric_summary = aggregate_scores(candidate_scores)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "reference_papers": list(REFERENCE_PAPERS),
        "fixture_path": FIXTURE_REL_PATH.as_posix(),
        "fixture_count": len(fixture_rows),
        "optimization_families": sorted({str(row["family"]) for row in fixture_rows}),
        "exact_solver_backends": sorted(
            {str(row["checker_backend"]) for row in fixture_rows}
        ),
        "feasibility_metric_defined": True,
        "objective_gap_metric_defined": True,
        "hallucinated_entity_metric_defined": True,
        "missing_constraint_metric_defined": True,
        "invalid_format_metric_defined": True,
        "metric_definitions": _metric_definitions(),
        "metric_summary": metric_summary,
        "candidate_scores": candidate_scores,
        "optional_llm_smoke": None,
        "mandated_optional_smoke_model_ids": list(MANDATED_OPTIONAL_SMOKE_MODEL_IDS),
        "ready_for_clean_verifier": _ready_for_clean_verifier(fixture_rows, metric_summary),
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "tests_run": list(tests_run),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifacts(
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Write the Exp 3211 fixture JSONL and terminal JSON artifact."""

    root_path = Path(root)
    artifact = build_artifact(root_path, tests_run=tests_run)
    output = root_path / ARTIFACT_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _knapsack_rows() -> list[JsonDict]:
    raw_rows = [
        (
            "cbpilot-3211-knapsack-001",
            7,
            [("map", 2, 5), ("rope", 3, 6), ("lamp", 4, 7), ("kit", 1, 3)],
            ["kit"],
            [("lamp", "rope")],
        ),
        (
            "cbpilot-3211-knapsack-002",
            9,
            [("sensor", 3, 8), ("battery", 4, 9), ("case", 2, 4), ("radio", 3, 7)],
            [],
            [("battery", "radio")],
        ),
        (
            "cbpilot-3211-knapsack-003",
            6,
            [("book", 2, 4), ("medkit", 3, 9), ("water", 3, 8), ("snack", 1, 2)],
            ["book"],
            [],
        ),
        (
            "cbpilot-3211-knapsack-004",
            8,
            [("cpu", 3, 10), ("gpu", 5, 13), ("fan", 2, 4), ("ssd", 2, 6)],
            ["fan"],
            [("cpu", "gpu")],
        ),
        (
            "cbpilot-3211-knapsack-005",
            5,
            [("alpha", 1, 3), ("beta", 2, 4), ("gamma", 4, 9), ("delta", 2, 5)],
            [],
            [("gamma", "delta")],
        ),
    ]
    return [
        {
            "row_id": row_id,
            "family": "knapsack",
            "instance_data": {
                "capacity": capacity,
                "items": [
                    {"name": name, "weight": weight, "value": value}
                    for name, weight, value in items
                ],
                "items_by_name": {
                    name: {"name": name, "weight": weight, "value": value}
                    for name, weight, value in items
                },
                "required_items": required,
                "incompatible_pairs": [list(pair) for pair in incompatible],
            },
            "constraints": [
                "selected_items must name only listed items",
                "total selected weight must be <= capacity",
                "all required_items must be selected",
                "no incompatible pair may be selected together",
            ],
            "objective": {"sense": "max", "name": "total_value"},
            "candidate_schema": {"selected_items": "list[str]"},
            "checker_backend": "exact_knapsack_subset_enumerator",
        }
        for row_id, capacity, items, required, incompatible in raw_rows
    ]


def _assignment_rows() -> list[JsonDict]:
    raw_rows = [
        (
            "cbpilot-3211-assignment-001",
            ["pack", "ship", "audit"],
            ["amy", "bo", "cy"],
            {("pack", "amy"): 7, ("pack", "bo"): 4, ("pack", "cy"): 6, ("ship", "amy"): 5, ("ship", "bo"): 8, ("ship", "cy"): 4, ("audit", "amy"): 6, ("audit", "bo"): 3, ("audit", "cy"): 9},
            [],
            [],
        ),
        (
            "cbpilot-3211-assignment-002",
            ["triage", "repair", "verify"],
            ["ivy", "jo", "kai"],
            {("triage", "ivy"): 8, ("triage", "jo"): 5, ("triage", "kai"): 7, ("repair", "ivy"): 4, ("repair", "jo"): 9, ("repair", "kai"): 6, ("verify", "ivy"): 6, ("verify", "jo"): 4, ("verify", "kai"): 8},
            [["repair", "kai"]],
            [],
        ),
        (
            "cbpilot-3211-assignment-003",
            ["red", "blue", "green"],
            ["nora", "omar", "paz"],
            {("red", "nora"): 6, ("red", "omar"): 9, ("red", "paz"): 5, ("blue", "nora"): 8, ("blue", "omar"): 4, ("blue", "paz"): 7, ("green", "nora"): 5, ("green", "omar"): 6, ("green", "paz"): 9},
            [],
            [["blue", "omar"]],
        ),
        (
            "cbpilot-3211-assignment-004",
            ["extract", "label", "review"],
            ["ren", "sue", "tay"],
            {("extract", "ren"): 8, ("extract", "sue"): 6, ("extract", "tay"): 4, ("label", "ren"): 5, ("label", "sue"): 9, ("label", "tay"): 7, ("review", "ren"): 6, ("review", "sue"): 4, ("review", "tay"): 8},
            [["review", "tay"]],
            [],
        ),
        (
            "cbpilot-3211-assignment-005",
            ["north", "south", "west"],
            ["uma", "vic", "wes"],
            {("north", "uma"): 9, ("north", "vic"): 5, ("north", "wes"): 6, ("south", "uma"): 6, ("south", "vic"): 8, ("south", "wes"): 4, ("west", "uma"): 4, ("west", "vic"): 7, ("west", "wes"): 9},
            [],
            [["south", "wes"]],
        ),
    ]
    rows: list[JsonDict] = []
    for row_id, tasks, workers, scores, required, forbidden in raw_rows:
        rows.append(
            {
                "row_id": row_id,
                "family": "assignment",
                "instance_data": {
                    "tasks": tasks,
                    "workers": workers,
                    "scores": {f"{task}|{worker}": score for (task, worker), score in scores.items()},
                    "required_assignments": required,
                    "forbidden_assignments": forbidden,
                },
                "constraints": [
                    "every task must be assigned exactly once",
                    "each worker may be assigned to at most one task",
                    "required_assignments must be present",
                    "forbidden_assignments must be absent",
                ],
                "objective": {"sense": "max", "name": "total_score"},
                "candidate_schema": {"assignment": "dict[task, worker]"},
                "checker_backend": "exact_assignment_permutation_enumerator",
            }
        )
    return rows


def _graph_coloring_rows() -> list[JsonDict]:
    raw_rows = [
        ("cbpilot-3211-coloring-001", [0, 1, 2], [[0, 1]], [0, 1, 2]),
        ("cbpilot-3211-coloring-002", [0, 1, 2, 3], [[0, 1], [1, 2], [2, 3]], [0, 1, 2]),
        ("cbpilot-3211-coloring-003", [0, 1, 2, 3], [[0, 1], [1, 2], [0, 2]], [0, 1, 2, 3]),
        ("cbpilot-3211-coloring-004", [0, 1, 2, 3], [[0, 1], [0, 2], [0, 3]], [0, 1, 2]),
        ("cbpilot-3211-coloring-005", [0, 1, 2, 3], [[0, 1], [1, 2], [2, 3], [3, 0]], [0, 1, 2]),
    ]
    return [
        {
            "row_id": row_id,
            "family": "graph_coloring",
            "instance_data": {"nodes": nodes, "edges": edges, "colors": colors},
            "constraints": [
                "every listed node must receive exactly one listed color",
                "adjacent nodes must receive different colors",
            ],
            "objective": {"sense": "min", "name": "used_color_count"},
            "candidate_schema": {"colors": "dict[node, color]"},
            "checker_backend": "exact_graph_coloring_enumerator",
        }
        for row_id, nodes, edges, colors in raw_rows
    ]


def _solve_knapsack(row: Mapping[str, Any]) -> JsonDict:
    feasible = list(_enumerate_feasible_solutions(row))
    best = max(feasible, key=lambda solution: (_objective_value(row, solution), _canonical_json(solution)))
    return _reference(row, best, len(feasible))


def _solve_assignment(row: Mapping[str, Any]) -> JsonDict:
    feasible = list(_enumerate_feasible_solutions(row))
    best = max(feasible, key=lambda solution: (_objective_value(row, solution), _canonical_json(solution)))
    return _reference(row, best, len(feasible))


def _solve_graph_coloring(row: Mapping[str, Any]) -> JsonDict:
    feasible = list(_enumerate_feasible_solutions(row))
    best = min(feasible, key=lambda solution: (_objective_value(row, solution), _canonical_json(solution)))
    return _reference(row, best, len(feasible))


def _reference(row: Mapping[str, Any], solution: Mapping[str, Any], feasible_count: int) -> JsonDict:
    return {
        "feasible": True,
        "objective_sense": row["objective"]["sense"],
        "objective_name": row["objective"]["name"],
        "objective_value": _objective_value(row, solution),
        "solution": dict(solution),
        "feasible_count": feasible_count,
    }


def _enumerate_feasible_solutions(row: Mapping[str, Any]) -> list[JsonDict]:
    family = row["family"]
    if family == "knapsack":
        items = [item["name"] for item in row["instance_data"]["items"]]
        solutions = []
        for size in range(len(items) + 1):
            for subset in itertools.combinations(items, size):
                solution = {"selected_items": sorted(subset)}
                if _check_feasibility(row, solution)[0]:
                    solutions.append(solution)
        return solutions
    if family == "assignment":
        tasks = list(row["instance_data"]["tasks"])
        workers = list(row["instance_data"]["workers"])
        solutions = []
        for assigned_workers in itertools.permutations(workers, len(tasks)):
            solution = {"assignment": dict(zip(tasks, assigned_workers, strict=True))}
            if _check_feasibility(row, solution)[0]:
                solutions.append(solution)
        return solutions
    nodes = list(row["instance_data"]["nodes"])
    colors = list(row["instance_data"]["colors"])
    solutions = []
    for assignment in itertools.product(colors, repeat=len(nodes)):
        solution = {"colors": {str(node): color for node, color in zip(nodes, assignment, strict=True)}}
        if _check_feasibility(row, solution)[0]:
            solutions.append(solution)
    return solutions


def _normalize_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[JsonDict, str | None]:
    family = row["family"]
    if family == "knapsack":
        selected = candidate.get("selected_items")
        if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
            return {}, "selected_items_not_string_list"
        return {"selected_items": sorted(selected)}, None
    if family == "assignment":
        assignment = candidate.get("assignment")
        if not isinstance(assignment, Mapping) or not all(
            isinstance(task, str) and isinstance(worker, str) for task, worker in assignment.items()
        ):
            return {}, "assignment_not_string_mapping"
        return {"assignment": dict(assignment)}, None
    colors = candidate.get("colors")
    if not isinstance(colors, Mapping) or not all(
        isinstance(node, str) and isinstance(color, int) for node, color in colors.items()
    ):
        return {}, "colors_not_node_integer_mapping"
    return {"colors": dict(colors)}, None


def _hallucinated_entities(row: Mapping[str, Any], solution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    data = row["instance_data"]
    if row["family"] == "knapsack":
        allowed = set(data["items_by_name"])
        reasons = [f"unknown_item:{item}" for item in solution["selected_items"] if item not in allowed]
    elif row["family"] == "assignment":
        tasks = set(data["tasks"])
        workers = set(data["workers"])
        reasons = [
            *(f"unknown_task:{task}" for task in solution["assignment"] if task not in tasks),
            *(f"unknown_worker:{worker}" for worker in solution["assignment"].values() if worker not in workers),
        ]
    else:
        nodes = {str(node) for node in data["nodes"]}
        colors = set(data["colors"])
        reasons = [
            *(f"unknown_node:{node}" for node in solution["colors"] if node not in nodes),
            *(f"unknown_color:{color}" for color in solution["colors"].values() if color not in colors),
        ]
    return bool(reasons), reasons


def _entity_slots_complete(row: Mapping[str, Any], solution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if row["family"] == "assignment":
        missing = [
            f"missing_task:{task}"
            for task in row["instance_data"]["tasks"]
            if task not in solution["assignment"]
        ]
        return not missing, missing
    if row["family"] == "graph_coloring":
        missing = [
            f"missing_node:{node}"
            for node in row["instance_data"]["nodes"]
            if str(node) not in solution["colors"]
        ]
        return not missing, missing
    return True, []


def _check_feasibility(row: Mapping[str, Any], solution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if row["family"] == "knapsack":
        return _check_knapsack(row, solution)
    if row["family"] == "assignment":
        return _check_assignment(row, solution)
    return _check_graph_coloring(row, solution)


def _check_knapsack(row: Mapping[str, Any], solution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    data = row["instance_data"]
    selected = set(solution["selected_items"])
    reasons = []
    weight = sum(data["items_by_name"][item]["weight"] for item in selected if item in data["items_by_name"])
    if weight > int(data["capacity"]):
        reasons.append(f"capacity_exceeded:{weight}>{data['capacity']}")
    reasons.extend(
        f"required_missing:{item}" for item in data["required_items"] if item not in selected
    )
    reasons.extend(
        f"incompatible_pair:{a}|{b}"
        for a, b in data["incompatible_pairs"]
        if a in selected and b in selected
    )
    return not reasons, reasons


def _check_assignment(row: Mapping[str, Any], solution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    data = row["instance_data"]
    assignment = solution["assignment"]
    workers = list(assignment.values())
    reasons = []
    if len(set(workers)) != len(workers):
        reasons.append("worker_reused")
    reasons.extend(
        f"required_assignment_missing:{task}->{worker}"
        for task, worker in data["required_assignments"]
        if assignment.get(task) != worker
    )
    reasons.extend(
        f"forbidden_assignment:{task}->{worker}"
        for task, worker in data["forbidden_assignments"]
        if assignment.get(task) == worker
    )
    return not reasons, reasons


def _check_graph_coloring(row: Mapping[str, Any], solution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    colors = solution["colors"]
    reasons = [
        f"edge_conflict:{a}-{b}"
        for a, b in row["instance_data"]["edges"]
        if colors[str(a)] == colors[str(b)]
    ]
    return not reasons, reasons


def _objective_value(row: Mapping[str, Any], solution: Mapping[str, Any]) -> int:
    data = row["instance_data"]
    if row["family"] == "knapsack":
        return sum(data["items_by_name"][item]["value"] for item in solution["selected_items"])
    if row["family"] == "assignment":
        return sum(data["scores"][f"{task}|{worker}"] for task, worker in solution["assignment"].items())
    return len(set(solution["colors"].values()))


def _objective_gap(row: Mapping[str, Any], objective_value: int) -> int:
    optimum = int(row["exact_reference"]["objective_value"])
    if row["objective"]["sense"] == "max":
        return max(0, optimum - objective_value)
    return max(0, objective_value - optimum)


def _candidate_text_for_fixture_index(row: Mapping[str, Any], index: int) -> str:
    exact_solution = row["exact_reference"]["solution"]
    if index in {0, 5, 8, 11, 14}:
        return json.dumps(exact_solution, sort_keys=True)
    if index in {1, 6, 9, 12}:
        return json.dumps(feasible_nonoptimal_solution(row), sort_keys=True)
    if index == 2:
        return json.dumps({"selected_items": ["ghost-item"]})
    if index == 3:
        return json.dumps({"selected_items": [item["name"] for item in row["instance_data"]["items"]]})
    if index == 4:
        return "not-json"
    if index == 7:
        worker = row["instance_data"]["workers"][0]
        return json.dumps(
            {"assignment": {task: worker for task in row["instance_data"]["tasks"]}}
        )
    if index == 10:
        return json.dumps({"colors": {str(row["instance_data"]["nodes"][0]): 0}})
    return json.dumps({"colors": {str(node): 0 for node in row["instance_data"]["nodes"]}})


def _score_base(row: Mapping[str, Any]) -> JsonDict:
    return {
        "row_id": str(row.get("row_id")),
        "family": str(row.get("family")),
        "checker_backend": row.get("checker_backend"),
        "optimum_value": row.get("exact_reference", {}).get("objective_value"),
        "objective_sense": row.get("objective", {}).get("sense"),
    }


def _score_failure(
    row: Mapping[str, Any],
    valid_format: bool,
    violation_class: str,
    reasons: Sequence[str],
) -> JsonDict:
    return _score_base(row) | {
        "valid_format": valid_format,
        "invalid_format": violation_class == "invalid_format",
        "hallucinated_entity": violation_class == "hallucinated_entity",
        "missing_constraint": violation_class == "missing_constraint",
        "feasibility_pass": False,
        "objective_value": None,
        "objective_gap": None,
        "reasons": list(reasons),
        "parsed_solution": None,
    }


def _ready_for_clean_verifier(
    fixture_rows: Sequence[Mapping[str, Any]],
    metric_summary: Mapping[str, Any],
) -> bool:
    return (
        len(fixture_rows) >= 15
        and {row["family"] for row in fixture_rows} >= {"knapsack", "assignment", "graph_coloring"}
        and metric_summary.get("candidate_count") == len(fixture_rows)
    )


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: exact ConstraintBench-style pilot only; no full ConstraintBench "
        f"coverage claimed; fixture_count={artifact['fixture_count']}"
    )


def _metric_definitions() -> JsonDict:
    return {
        "valid_format": "candidate JSON matches the family-specific solution schema",
        "feasibility_pass": "candidate satisfies hard constraints under the exact checker",
        "objective_gap": "absolute optimality gap among feasible candidates only",
        "hallucinated_entity": "candidate names an item, task, worker, node, or color absent from the instance",
        "missing_constraint": "candidate has valid entities but violates a hard constraint",
        "invalid_format": "candidate cannot be parsed or does not match the required schema",
    }


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
