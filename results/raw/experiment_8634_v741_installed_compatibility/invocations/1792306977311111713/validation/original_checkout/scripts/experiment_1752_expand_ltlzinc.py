#!/usr/bin/env python3
"""Exp 1752 LTLZinc spatial reasoning benchmark expansion.

This script turns the existing temporal LTLZinc benchmark into provenance for
100 deterministic topological-map routing cases.  It writes the reusable spatial
dataset and a terminal experiment artifact without calling MiniZinc.

Spec: REQ-LEARN-1752, SCENARIO-LEARN-1752.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


JsonDict = dict[str, Any]

EXPERIMENT_ID = 1752
EXPERIMENT = "1752_ltlzinc_spatial_reasoning_expansion"
SCHEMA = "carnot.ltlzinc_spatial_topological_routing_benchmark.v1"
ARTIFACT_SCHEMA = "carnot.experiment_1752_ltlzinc_spatial.v1"
BENCHMARK_ID = "ltlzinc_spatial_topological_routing_v1"
GENERATOR = "scripts/experiment_1752_expand_ltlzinc.py"
RUN_DATE = "20260510"
DEFAULT_BASE_BENCHMARK_PATH = REPO_ROOT / "data" / "ltlzinc_benchmark.json"
DEFAULT_SPATIAL_OUTPUT_PATH = REPO_ROOT / "data" / "ltlzinc_spatial_benchmark.json"
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1752_spatial.json"
VERIFIER_PATH = "scripts.experiment_1752_expand_ltlzinc.verify_spatial_case"
SPATIAL_FAMILIES = (
    "route_exists",
    "avoid_blocked_edge",
    "visit_waypoint",
    "avoid_zone",
    "simple_path",
)
CASES_PER_FAMILY_STATE = 10
REQUIRED_BENCHMARK_FIELDS = (
    "schema",
    "benchmark_id",
    "generator",
    "run_date",
    "spec",
    "source",
    "case_count",
    "map_count",
    "sat_case_count",
    "repair_hint_case_count",
    "supported_spatial_families",
    "family_counts",
    "cases",
)
REQUIRED_CASE_FIELDS = (
    "case_id",
    "source_benchmark",
    "nonforgetting_phase",
    "spatial_family",
    "topological_map",
    "map_id",
    "nodes",
    "edges",
    "blocked_edges",
    "start",
    "goal",
    "route",
    "waypoint",
    "forbidden_node",
    "ltlzinc_formula",
    "minizinc_constraint",
    "expected_satisfied",
    "label",
    "certificate_state",
    "dvi_label",
    "fr11_memory_hint",
    "evaluation",
    "retention",
    "tags",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "schema",
    "experiment_id",
    "benchmark_path",
    "spatial_case_count",
    "validated_case_count",
    "sat_case_count",
    "repair_hint_case_count",
    "family_counts",
    "commands_run",
    "honest_verdict",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = dict(payload)
    destination.write_text(
        json.dumps(written, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return written


def load_base_benchmark(path: Path | str = DEFAULT_BASE_BENCHMARK_PATH) -> JsonDict:
    """REQ-LEARN-1752-2: read the existing LTLZinc benchmark as provenance."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "base benchmark must be a mapping")
    _require("benchmark_id" in payload, "base benchmark must include benchmark_id")
    return dict(payload)


def _source_summary(base_benchmark: Mapping[str, Any], base_path: Path) -> JsonDict:
    return {
        "base_benchmark_path": _repo_relative(base_path),
        "base_benchmark_id": str(base_benchmark.get("benchmark_id", "")),
        "base_schema": str(base_benchmark.get("schema", "")),
        "base_case_count": int(base_benchmark.get("case_count", 0)),
        "base_spec": list(base_benchmark.get("spec", [])),
    }


def _case_source_benchmark(source: Mapping[str, Any]) -> JsonDict:
    return {
        "path": str(source["base_benchmark_path"]),
        "benchmark_id": str(source["base_benchmark_id"]),
        "schema": str(source["base_schema"]),
        "case_count": int(source["base_case_count"]),
    }


def _node(map_id: str, row: int, col: int) -> str:
    return f"{map_id}_r{row}c{col}"


def _edge_key(left: str, right: str) -> str:
    a, b = sorted((str(left), str(right)))
    return f"{a}--{b}"


def _grid_edges(map_id: str) -> list[list[str]]:
    edges: list[list[str]] = []
    for row in range(3):
        for col in range(3):
            if col < 2:
                edges.append([_node(map_id, row, col), _node(map_id, row, col + 1)])
            if row < 2:
                edges.append([_node(map_id, row, col), _node(map_id, row + 1, col)])
    return edges


def _topological_map(seed: int) -> JsonDict:
    map_id = f"spatial-map-{seed:02d}"
    nodes = [_node(map_id, row, col) for row in range(3) for col in range(3)]
    return {
        "map_id": map_id,
        "topology": "3x3_grid",
        "nodes": nodes,
        "edges": _grid_edges(map_id),
    }


def _routes_for_family(family: str, map_id: str) -> JsonDict:
    start = _node(map_id, 0, 0)
    goal = _node(map_id, 2, 2)
    top_route = [start, _node(map_id, 0, 1), _node(map_id, 0, 2), _node(map_id, 1, 2), goal]
    down_route = [start, _node(map_id, 1, 0), _node(map_id, 2, 0), _node(map_id, 2, 1), goal]
    center_route = [
        start,
        _node(map_id, 0, 1),
        _node(map_id, 1, 1),
        _node(map_id, 2, 1),
        goal,
    ]
    loop_route = [
        start,
        _node(map_id, 0, 1),
        start,
        _node(map_id, 1, 0),
        _node(map_id, 2, 0),
        _node(map_id, 2, 1),
        goal,
    ]
    jump_route = [start, _node(map_id, 0, 2), _node(map_id, 1, 2), goal]
    blocked_first_edge = [[start, _node(map_id, 0, 1)]]
    route_specs = {
        "route_exists": {
            "sat_route": top_route,
            "repair_route": jump_route,
            "blocked_edges": [],
            "waypoint": None,
            "forbidden_node": None,
        },
        "avoid_blocked_edge": {
            "sat_route": down_route,
            "repair_route": top_route,
            "blocked_edges": blocked_first_edge,
            "waypoint": None,
            "forbidden_node": None,
        },
        "visit_waypoint": {
            "sat_route": center_route,
            "repair_route": top_route,
            "blocked_edges": [],
            "waypoint": _node(map_id, 1, 1),
            "forbidden_node": None,
        },
        "avoid_zone": {
            "sat_route": top_route,
            "repair_route": center_route,
            "blocked_edges": [],
            "waypoint": None,
            "forbidden_node": _node(map_id, 1, 1),
        },
        "simple_path": {
            "sat_route": down_route,
            "repair_route": loop_route,
            "blocked_edges": [],
            "waypoint": None,
            "forbidden_node": None,
        },
    }
    return dict(route_specs[family])


def _ltlzinc_formula(
    family: str, start: str, goal: str, waypoint: str | None, forbidden: str | None
) -> str:
    if family == "route_exists":
        return f"reachable({start}, {goal})"
    if family == "avoid_blocked_edge":
        return f"route({start}, {goal}) && G not blocked_edge"
    if family == "visit_waypoint":
        return f"route({start}, {goal}) && F at({waypoint})"
    if family == "avoid_zone":
        return f"route({start}, {goal}) && G not at({forbidden})"
    if family == "simple_path":
        return f"route({start}, {goal}) && all_distinct(route)"
    raise ValueError(f"unsupported spatial family: {family}")  # pragma: no cover


def _minizinc_constraint(
    family: str,
    waypoint: str | None,
    forbidden: str | None,
) -> str:
    if family == "route_exists":
        return "constraint route_starts_at_start /\\ route_ends_at_goal /\\ adjacent_steps;"
    if family == "avoid_blocked_edge":
        return "constraint forall(i in 1..ROUTE_LEN-1)(not blocked[route[i], route[i+1]]);"
    if family == "visit_waypoint":
        return f"constraint exists(i in ROUTE_INDEX)(route[i] = {waypoint});"
    if family == "avoid_zone":
        return f"constraint forall(i in ROUTE_INDEX)(route[i] != {forbidden});"
    if family == "simple_path":
        return "constraint all_different(route);"
    raise ValueError(f"unsupported spatial family: {family}")  # pragma: no cover


def make_spatial_case(
    *,
    family: str,
    seed: int,
    expected_satisfied: bool,
    source_benchmark: Mapping[str, Any],
) -> JsonDict:
    """REQ-LEARN-1752-3: build one topological-map routing case."""

    base_map = _topological_map(seed)
    map_id = str(base_map["map_id"])
    route_spec = _routes_for_family(family, map_id)
    route_key = "sat_route" if expected_satisfied else "repair_route"
    route = list(route_spec[route_key])
    blocked_edges = [list(edge) for edge in route_spec["blocked_edges"]]
    waypoint = route_spec["waypoint"]
    forbidden = route_spec["forbidden_node"]
    topological_map = dict(base_map)
    topological_map["blocked_edges"] = blocked_edges
    certificate_state = "SAT" if expected_satisfied else "REPAIR_HINT"
    state_slug = "sat" if expected_satisfied else "repair-hint"
    case_id = f"ltlzinc-spatial-{family}-{state_slug}-{seed:02d}"
    return {
        "case_id": case_id,
        "source": "exp1752_ltlzinc_spatial_synthetic",
        "source_benchmark": dict(source_benchmark),
        "nonforgetting_phase": "spatial",
        "spatial_family": family,
        "constraint_family": family,
        "topological_map": topological_map,
        "map_id": map_id,
        "nodes": list(base_map["nodes"]),
        "edges": list(base_map["edges"]),
        "blocked_edges": blocked_edges,
        "start": _node(map_id, 0, 0),
        "goal": _node(map_id, 2, 2),
        "route": route,
        "waypoint": waypoint,
        "forbidden_node": forbidden,
        "ltlzinc_formula": _ltlzinc_formula(family, route[0], route[-1], waypoint, forbidden),
        "minizinc_constraint": _minizinc_constraint(family, waypoint, forbidden),
        "expected_satisfied": bool(expected_satisfied),
        "label": "accepted" if expected_satisfied else "rejected",
        "certificate_state": certificate_state,
        "dvi_label": 0 if expected_satisfied else 1,
        "fr11_memory_hint": (
            "promote_spatial_constraint_success"
            if expected_satisfied
            else "promote_spatial_constraint_violation"
        ),
        "evaluation": {
            "verifier_path": VERIFIER_PATH,
            "expected_verifier_result": bool(expected_satisfied),
        },
        "retention": {
            "phase": "spatial",
            "anchor_case_id": case_id,
            "must_retrieve_after_updates": True,
            "nonforgetting_check": "retrieve_same_spatial_case_after_curriculum_updates",
        },
        "tags": [
            "ltlzinc",
            "spatial",
            "topological-map",
            f"operator:{family}",
            "phase:spatial",
        ],
    }


def _edge_set(edges: Sequence[Sequence[str]]) -> set[str]:
    return {_edge_key(edge[0], edge[1]) for edge in edges}


def _route_is_valid(case: Mapping[str, Any]) -> bool:
    route = [str(node) for node in case["route"]]
    edges = _edge_set(case["edges"])
    blocked = _edge_set(case["blocked_edges"])
    endpoint_ok = bool(route) and route[0] == str(case["start"]) and route[-1] == str(case["goal"])
    transition_ok = all(
        _edge_key(left, right) in edges and _edge_key(left, right) not in blocked
        for left, right in zip(route, route[1:], strict=False)
    )
    return endpoint_ok and transition_ok


def verify_spatial_case(case: Mapping[str, Any]) -> bool:
    """REQ-LEARN-1752-6: locally evaluate the supported spatial families."""

    family = str(case["spatial_family"])
    route = [str(node) for node in case["route"]]
    valid_route = _route_is_valid(case)
    if family == "route_exists":
        return valid_route
    if family == "avoid_blocked_edge":
        return valid_route
    if family == "visit_waypoint":
        return valid_route and str(case["waypoint"]) in route
    if family == "avoid_zone":
        return valid_route and str(case["forbidden_node"]) not in route
    if family == "simple_path":
        return valid_route and len(set(route)) == len(route)
    raise ValueError(f"unsupported spatial family: {family}")  # pragma: no cover


def validate_spatial_case_schema(case: Mapping[str, Any]) -> None:
    """REQ-LEARN-1752-3/6: enforce one reusable spatial case contract."""

    missing = sorted(set(REQUIRED_CASE_FIELDS).difference(case))
    _require(not missing, f"missing spatial case fields: {missing}")
    _require(case["spatial_family"] in SPATIAL_FAMILIES, "unsupported spatial family")
    _require(case["certificate_state"] in {"SAT", "REPAIR_HINT"}, "unsupported certificate_state")
    _require(isinstance(case["topological_map"], Mapping), "topological_map must be a mapping")
    _require(isinstance(case["evaluation"], Mapping), "evaluation must be a mapping")
    _require(isinstance(case["retention"], Mapping), "retention must be a mapping")
    _require(case["evaluation"].get("verifier_path") == VERIFIER_PATH, "unsupported verifier")
    _require(
        bool(case["evaluation"].get("expected_verifier_result"))
        is bool(case["expected_satisfied"]),
        "expected verifier result must match expected_satisfied",
    )
    _require(
        verify_spatial_case(case) is bool(case["expected_satisfied"]),
        "spatial verifier disagrees with expected label",
    )


def _count_cases(cases: Sequence[Mapping[str, Any]], key: str, value: Any) -> int:
    return sum(1 for case in cases if case.get(key) == value)


def _family_counts(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: JsonDict = {}
    for family in SPATIAL_FAMILIES:
        family_cases = [case for case in cases if case["spatial_family"] == family]
        certificate_counts = Counter(str(case["certificate_state"]) for case in family_cases)
        counts[family] = {
            "case_count": len(family_cases),
            "sat_case_count": int(certificate_counts.get("SAT", 0)),
            "repair_hint_case_count": int(certificate_counts.get("REPAIR_HINT", 0)),
        }
    return counts


def _spatial_cases(source_benchmark: Mapping[str, Any]) -> list[JsonDict]:
    cases: list[JsonDict] = []
    for family_index, family in enumerate(SPATIAL_FAMILIES):
        for offset in range(CASES_PER_FAMILY_STATE):
            seed = family_index * CASES_PER_FAMILY_STATE + offset
            cases.append(
                make_spatial_case(
                    family=family,
                    seed=seed,
                    expected_satisfied=True,
                    source_benchmark=source_benchmark,
                )
            )
            cases.append(
                make_spatial_case(
                    family=family,
                    seed=seed,
                    expected_satisfied=False,
                    source_benchmark=source_benchmark,
                )
            )
    return cases


def build_spatial_benchmark(
    base_benchmark_path: Path | str = DEFAULT_BASE_BENCHMARK_PATH,
) -> JsonDict:
    """REQ-LEARN-1752-2: build the full deterministic spatial benchmark."""

    base_path = Path(base_benchmark_path)
    base_benchmark = load_base_benchmark(base_path)
    source = _source_summary(base_benchmark, base_path)
    cases = _spatial_cases(_case_source_benchmark(source))
    certificate_counts = Counter(str(case["certificate_state"]) for case in cases)
    payload: JsonDict = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "generator": GENERATOR,
        "run_date": RUN_DATE,
        "spec": ["REQ-LEARN-1752", "SCENARIO-LEARN-1752"],
        "source": source,
        "case_count": len(cases),
        "map_count": len({str(case["map_id"]) for case in cases}),
        "sat_case_count": int(certificate_counts.get("SAT", 0)),
        "repair_hint_case_count": int(certificate_counts.get("REPAIR_HINT", 0)),
        "supported_spatial_families": list(SPATIAL_FAMILIES),
        "family_counts": _family_counts(cases),
        "cases": cases,
    }
    validate_spatial_benchmark(payload)
    return payload


def validate_spatial_benchmark(payload: Mapping[str, Any]) -> None:
    """REQ-LEARN-1752-2/4/6: enforce benchmark count and schema invariants."""

    missing = sorted(set(REQUIRED_BENCHMARK_FIELDS).difference(payload))
    _require(not missing, f"missing spatial benchmark fields: {missing}")
    _require(payload["schema"] == SCHEMA, "unsupported schema")
    cases = payload["cases"]
    _require(isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)), "cases invalid")
    for case in cases:
        validate_spatial_case_schema(case)
    case_ids = [str(case["case_id"]) for case in cases]
    _require(len(set(case_ids)) == len(case_ids), "case_id values must be unique")
    _require(payload["case_count"] == len(cases), "case_count must match cases")
    _require(payload["map_count"] == len({str(case["map_id"]) for case in cases}), "map_count")
    _require(
        payload["sat_case_count"] == _count_cases(cases, "certificate_state", "SAT"),
        "sat_case_count must match cases",
    )
    _require(
        payload["repair_hint_case_count"]
        == _count_cases(cases, "certificate_state", "REPAIR_HINT"),
        "repair_hint_case_count must match cases",
    )
    _require(
        set(payload["supported_spatial_families"]) == set(SPATIAL_FAMILIES),
        "supported_spatial_families mismatch",
    )
    _require(payload["family_counts"] == _family_counts(cases), "family_counts mismatch")


def write_spatial_benchmark(
    *,
    output_path: Path | str = DEFAULT_SPATIAL_OUTPUT_PATH,
    base_benchmark_path: Path | str = DEFAULT_BASE_BENCHMARK_PATH,
) -> JsonDict:
    """REQ-LEARN-1752-2: write `data/ltlzinc_spatial_benchmark.json`."""

    benchmark = build_spatial_benchmark(base_benchmark_path=base_benchmark_path)
    return _write_json(output_path, benchmark)


def _case_results(cases: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    results: list[JsonDict] = []
    for case in cases:
        local_satisfied = verify_spatial_case(case)
        expected = bool(case["expected_satisfied"])
        results.append(
            {
                "case_id": str(case["case_id"]),
                "spatial_family": str(case["spatial_family"]),
                "expected_satisfied": expected,
                "local_satisfied": local_satisfied,
                "validated": local_satisfied is expected,
            }
        )
    return results


def build_artifact(
    *,
    benchmark: Mapping[str, Any],
    benchmark_path: Path | str,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    commands_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-LEARN-1752-5/6: build the terminal Exp 1752 artifact."""

    validate_spatial_benchmark(benchmark)
    cases = benchmark["cases"]
    case_results = _case_results(cases)
    validated_case_count = sum(1 for result in case_results if result["validated"])
    complete = (
        int(benchmark["case_count"]) == 100
        and validated_case_count == 100
        and int(benchmark["sat_case_count"]) == 50
        and int(benchmark["repair_hint_case_count"]) == 50
    )
    artifact: JsonDict = {
        "status": "complete" if complete else "blocked",
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec": ["REQ-LEARN-1752", "SCENARIO-LEARN-1752"],
        "artifact_metadata": {"project_root": str(project_root), "run_date": run_date},
        "run_date": run_date,
        "started_at": started_at or _timestamp(),
        "finished_at": _timestamp(),
        "duration_s": round(float(duration_s), 3),
        "benchmark_id": str(benchmark["benchmark_id"]),
        "benchmark_path": str(benchmark_path),
        "source": dict(benchmark["source"]),
        "spatial_case_count": int(benchmark["case_count"]),
        "validated_case_count": validated_case_count,
        "sat_case_count": int(benchmark["sat_case_count"]),
        "repair_hint_case_count": int(benchmark["repair_hint_case_count"]),
        "map_count": int(benchmark["map_count"]),
        "supported_spatial_families": list(benchmark["supported_spatial_families"]),
        "family_counts": dict(benchmark["family_counts"]),
        "case_results": case_results,
        "commands_run": list(commands_run or []),
        "honest_verdict": (
            "complete: ltlzinc_spatial_benchmark_ready"
            if complete
            else "blocked: ltlzinc_spatial_benchmark_incomplete"
        ),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1752-5/6: enforce the terminal artifact contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS).difference(artifact))
    _require(not missing, f"missing artifact fields: {missing}")
    _require(artifact["schema"] == ARTIFACT_SCHEMA, "unsupported artifact schema")
    _require(artifact["status"] in {"complete", "blocked"}, "unsupported status")
    case_results = artifact.get("case_results", [])
    _require(
        isinstance(case_results, Sequence) and not isinstance(case_results, (str, bytes)),
        "case_results must be rows",
    )
    validated_case_count = sum(1 for result in case_results if result.get("validated"))
    _require(
        int(artifact["validated_case_count"]) == validated_case_count,
        "validated_case_count must match case_results",
    )
    if artifact["status"] == "complete":
        _require(int(artifact["spatial_case_count"]) == 100, "complete requires 100 cases")
        _require(int(artifact["validated_case_count"]) == 100, "complete requires validation")
        _require(int(artifact["sat_case_count"]) == 50, "complete requires 50 SAT cases")
        _require(
            int(artifact["repair_hint_case_count"]) == 50,
            "complete requires 50 REPAIR_HINT cases",
        )
        _require(
            str(artifact["honest_verdict"]).startswith("complete:"),
            "complete verdict must be explicit",
        )


def run_experiment(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    benchmark_path: Path | str = DEFAULT_SPATIAL_OUTPUT_PATH,
    base_benchmark_path: Path | str = DEFAULT_BASE_BENCHMARK_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    commands_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1752 and write both the spatial benchmark and terminal JSON."""

    started_at = _timestamp()
    t0 = time.perf_counter()
    benchmark = write_spatial_benchmark(
        output_path=benchmark_path,
        base_benchmark_path=base_benchmark_path,
    )
    recorded_commands = list(commands_run or [f"python {GENERATOR}"])
    artifact = build_artifact(
        benchmark=benchmark,
        benchmark_path=benchmark_path,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        commands_run=recorded_commands,
    )
    return _write_json(output_path, artifact)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = list(sys.argv[1:] if argv is None else argv)
    output_path = Path(args[0]) if args else DEFAULT_ARTIFACT_PATH
    benchmark_path = Path(args[1]) if len(args) > 1 else DEFAULT_SPATIAL_OUTPUT_PATH
    artifact = run_experiment(output_path=output_path, benchmark_path=benchmark_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
