"""Exp 3224 Logitext-style partial SMT coverage pilot.

Spec refs: REQ-VERIFY-3224, SCENARIO-VERIFY-3224.

This module does not try to understand arbitrary natural language.  It replays
two already-structured `.297` fixture banks and records which constraint pieces
are exact enough to hand to a deterministic solver today.  The boundary is
important: answer equality and bounded optimization instances are solver-ready;
free-form extraction of a new local arithmetic rule from prose is explicitly
left as a future Logitext-style extraction problem.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
SCHEMA_VERSION = "carnot.logitext_partial_smt_context_coverage_pilot.v1"
EXPERIMENT_ID = "exp3224"
MILESTONE = "2026.05.298"
OUTPUT_REL_PATH = Path(
    "results/experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.json"
)

CONTEXT_ARTIFACT_REL_PATH = Path(
    "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
)
CONTEXT_FIXTURE_REL_PATH = Path("data/research/context_cot_clbench_parametric_shortcut_v1.jsonl")
CONSTRAINT_ARTIFACT_REL_PATH = Path(
    "results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json"
)
CONSTRAINT_FIXTURE_REL_PATH = Path("data/research/constraintbench_feasibility_objective_pilot_v1.jsonl")

INFERENCE_SUBSTRATE = "deterministic_artifact_replay_no_llm"
FULLY_FORMALIZABLE = "fully_formalizable"
PARTIALLY_FORMALIZABLE = "partially_formalizable"
NOT_FORMALIZABLE = "not_formalizable_without_extraction"
FORMALIZABILITY_LABELS = (FULLY_FORMALIZABLE, PARTIALLY_FORMALIZABLE, NOT_FORMALIZABLE)

TAXONOMY_FRAGMENT_TYPES = (
    "boolean",
    "string_equality",
    "arithmetic_interval",
    "all_different",
    "graph_relation",
    "feasibility",
    "objective_bound",
)
CONSTRAINT_TAXONOMY: tuple[JsonDict, ...] = (
    {
        "fragment_type": "boolean",
        "solver_encoding": "Bool",
        "description": "Binary decision flags such as selected items or rejected prior-bait answers.",
    },
    {
        "fragment_type": "string_equality",
        "solver_encoding": "String equality over canonical answer literals",
        "description": "Exact answer matching when the fixture already supplies the target string.",
    },
    {
        "fragment_type": "arithmetic_interval",
        "solver_encoding": "QF_LIA integer equality or closed interval bound",
        "description": "Integer answer checks, capacities, domains, and finite numeric bounds.",
    },
    {
        "fragment_type": "all_different",
        "solver_encoding": "Distinct finite-domain integer variables",
        "description": "Assignment-style one-worker-per-task constraints.",
    },
    {
        "fragment_type": "graph_relation",
        "solver_encoding": "Finite-domain edge inequality constraints",
        "description": "Graph coloring adjacency relations and node-color domains.",
    },
    {
        "fragment_type": "feasibility",
        "solver_encoding": "Conjunction of hard constraints",
        "description": "A row is feasible only when every listed hard constraint is satisfied.",
    },
    {
        "fragment_type": "objective_bound",
        "solver_encoding": "Exact optimum equality or inequality bound from the reference solver",
        "description": "The fixture's exact reference objective remains the optimization authority.",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema_version",
    "experiment_id",
    "milestone",
    "source_fixture_artifacts",
    "fixture_row_count",
    "fully_formalizable_count",
    "partially_formalizable_count",
    "not_formalizable_count",
    "constraint_taxonomy",
    "smt_rows",
    "partial_smt_coverage",
    "exact_solver_row_count",
    "coverage_ready",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
)
SMT_ROW_FIELDS = (
    "row_id",
    "artifact_source",
    "fixture_family",
    "formalizability",
    "constraint_fragments",
    "formalized_fragments",
    "unformalized_requirements",
    "solver_ready_representation",
    "exact_solver_pointer",
    "exp3225_priority_score",
    "exp3225_priority_reason",
)
SOURCE_SPECS = (
    ("exp3210_context_artifact", CONTEXT_ARTIFACT_REL_PATH, "json"),
    ("exp3210_context_fixture", CONTEXT_FIXTURE_REL_PATH, "jsonl"),
    ("exp3211_constraint_artifact", CONSTRAINT_ARTIFACT_REL_PATH, "json"),
    ("exp3211_constraint_fixture", CONSTRAINT_FIXTURE_REL_PATH, "jsonl"),
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run -m pytest -o addopts='' tests/python/test_experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/logitext_partial_smt_context_coverage_pilot_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3224: build the partial SMT coverage artifact.

    The artifact is a coverage map, not a new verifier.  It says which checked-in
    fixture rows are already exact enough for SMT-style scoring and which rows
    still need a natural-language extraction layer before their full semantics
    can be trusted.
    """

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else float(started_s)
    sources = source_fixture_artifacts(root_path)
    source_errors = [source["id"] for source in sources if source["exists"] is not True]
    smt_rows: list[JsonDict] = []
    if not source_errors:
        context_rows = read_jsonl_objects(root_path / CONTEXT_FIXTURE_REL_PATH)
        constraint_rows = read_jsonl_objects(root_path / CONSTRAINT_FIXTURE_REL_PATH)
        smt_rows = [
            *[context_smt_row(row) for row in context_rows],
            *[constraintbench_smt_row(row) for row in constraint_rows],
        ]

    counts = formalizability_counts(smt_rows)
    family_coverage = coverage_by_fixture_family(smt_rows)
    readiness = readiness_checks(source_errors, smt_rows, counts)
    ready = all(readiness.values())
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "source_fixture_artifacts": sources,
        "source_errors": source_errors,
        "fixture_row_count": len(smt_rows),
        "fully_formalizable_count": counts[FULLY_FORMALIZABLE],
        "partially_formalizable_count": counts[PARTIALLY_FORMALIZABLE],
        "not_formalizable_count": counts[NOT_FORMALIZABLE],
        "constraint_taxonomy": [dict(entry) for entry in CONSTRAINT_TAXONOMY],
        "smt_rows": smt_rows,
        "coverage_by_fixture_family": family_coverage,
        "partial_smt_coverage": safe_rate(
            counts[FULLY_FORMALIZABLE] + counts[PARTIALLY_FORMALIZABLE],
            len(smt_rows),
        ),
        "fully_smt_coverage": safe_rate(counts[FULLY_FORMALIZABLE], len(smt_rows)),
        "exact_solver_row_count": sum(
            bool(row.get("exact_solver_pointer")) for row in smt_rows
        ),
        "highest_value_rows_for_exp3225": highest_value_rows(smt_rows),
        "coverage_method": {
            "row_denominator": "Exp 3210 context JSONL rows plus Exp 3211 ConstraintBench JSONL rows",
            "partial_smt_coverage": "(fully_formalizable + partially_formalizable) / fixture_row_count",
            "full_context_rule_extraction_claimed": False,
            "exact_checker_authority_preserved": True,
        },
        "coverage_ready": ready,
        "readiness_checks": readiness,
        "blocked_reasons": [name for name, passed in readiness.items() if passed is not True],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_details": {
            "executes_models": False,
            "new_live_model_calls": 0,
            "training_performed": False,
            "offline_exact_artifact_replay": True,
        },
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the requested Exp 3224 JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def context_smt_row(row: Mapping[str, Any]) -> JsonDict:
    """Map one Exp 3210 context row to exact answer-check fragments."""

    checker = str(row.get("exact_checker_type") or "")
    expected = str(row.get("expected_answer") or "")
    prior = str(row.get("prior_bait_answer") or "")
    if checker == "exact_integer_string":
        representation = {
            "kind": "answer_integer_equality",
            "logic": "QF_LIA",
            "variables": {"answer": "Int"},
            "assertions": [{"op": "=", "left": "answer", "right": parse_int_or_string(expected)}],
            "rejected_prior_bait": parse_int_or_string(prior),
        }
        fragments = [
            fragment("arithmetic_interval", "answer equals the fixture's extracted integer result"),
            fragment("boolean", "prior-bait answer is rejected by exact integer checker"),
        ]
        return base_smt_row(
            row_id=str(row.get("fixture_id") or ""),
            artifact_source="exp3210_context",
            fixture_family=str(row.get("family") or ""),
            formalizability=PARTIALLY_FORMALIZABLE,
            fragments=fragments,
            representation=representation,
            pointer={
                "source_fixture_artifact": CONTEXT_FIXTURE_REL_PATH.as_posix(),
                "checker_type": checker,
                "row_id": str(row.get("fixture_id") or ""),
            },
            unformalized=["natural_language_local_rule_semantics"],
            priority_score=1.0,
            priority_reason="local arithmetic rows expose the highest-value extraction gap for exp3225",
        )

    representation = {
        "kind": "answer_string_equality",
        "logic": "QF_S",
        "variables": {"answer": "String"},
        "assertions": [
            {"op": "=", "left": "answer", "right": expected},
            {"op": "!=", "left": "answer", "right": prior},
        ],
    }
    fragments = [
        fragment("string_equality", "answer equals the fixture's extracted local-context string"),
        fragment("boolean", "prior-bait answer is rejected by exact string checker"),
    ]
    return base_smt_row(
        row_id=str(row.get("fixture_id") or ""),
        artifact_source="exp3210_context",
        fixture_family=str(row.get("family") or ""),
        formalizability=FULLY_FORMALIZABLE if checker else NOT_FORMALIZABLE,
        fragments=fragments if checker else [],
        representation=representation if checker else None,
        pointer={
            "source_fixture_artifact": CONTEXT_FIXTURE_REL_PATH.as_posix(),
            "checker_type": checker,
            "row_id": str(row.get("fixture_id") or ""),
        }
        if checker
        else None,
        unformalized=[] if checker else ["missing_exact_checker_type"],
        priority_score=0.74 if row.get("family") == "context_defined_entity_facts" else 0.70,
        priority_reason="context answer equality is exact once the fixture-supplied target is trusted",
    )


def constraintbench_smt_row(row: Mapping[str, Any]) -> JsonDict:
    """Map one Exp 3211 structured optimization row to finite SMT fragments."""

    family = str(row.get("family") or "")
    if family == "knapsack":
        return knapsack_smt_row(row)
    if family == "assignment":
        return assignment_smt_row(row)
    if family == "graph_coloring":
        return graph_coloring_smt_row(row)
    return base_smt_row(
        row_id=str(row.get("row_id") or ""),
        artifact_source="exp3211_constraintbench",
        fixture_family=family,
        formalizability=NOT_FORMALIZABLE,
        fragments=[],
        representation=None,
        pointer=None,
        unformalized=["unknown_constraintbench_family"],
        priority_score=0.0,
        priority_reason="unknown family cannot be mapped without additional extraction",
    )


def knapsack_smt_row(row: Mapping[str, Any]) -> JsonDict:
    """Represent a bounded knapsack fixture as Boolean choices and linear bounds."""

    data = mapping(row.get("instance_data"))
    exact = mapping(row.get("exact_reference"))
    items = [mapping(item) for item in list_value(data.get("items"))]
    variables = {str(item.get("name")): "Bool" for item in items}
    value_terms = {str(item.get("name")): int_value(item.get("value")) for item in items}
    weight_terms = {str(item.get("name")): int_value(item.get("weight")) for item in items}
    representation = {
        "kind": "knapsack_bv_linear",
        "logic": "QF_LIA",
        "variables": variables,
        "capacity": int_value(data.get("capacity")),
        "weight_terms": weight_terms,
        "value_terms": value_terms,
        "required_items": list_value(data.get("required_items")),
        "incompatible_pairs": list_value(data.get("incompatible_pairs")),
        "objective": objective_bound(row),
    }
    fragments = [
        fragment("boolean", "one Boolean variable per listed item"),
        fragment("arithmetic_interval", "total selected weight is bounded by capacity"),
        fragment("feasibility", "required and incompatible-item constraints are hard constraints"),
        fragment("objective_bound", "total value is compared with the exact reference optimum"),
    ]
    return base_constraint_row(row, fragments, representation, 0.86, "knapsack combines feasibility and objective-gap checks")


def assignment_smt_row(row: Mapping[str, Any]) -> JsonDict:
    """Represent an assignment fixture as finite-domain worker variables."""

    data = mapping(row.get("instance_data"))
    tasks = [str(task) for task in list_value(data.get("tasks"))]
    workers = [str(worker) for worker in list_value(data.get("workers"))]
    worker_index = {worker: index for index, worker in enumerate(workers)}
    representation = {
        "kind": "assignment_finite_domain",
        "logic": "QF_FD",
        "variables": {task: {"domain": worker_index} for task in tasks},
        "all_different": tasks,
        "required_assignments": list_value(data.get("required_assignments")),
        "forbidden_assignments": list_value(data.get("forbidden_assignments")),
        "scores": mapping(data.get("scores")),
        "objective": objective_bound(row),
    }
    fragments = [
        fragment("all_different", "each task receives one worker and workers are not reused"),
        fragment("boolean", "required and forbidden assignments are hard Boolean clauses"),
        fragment("feasibility", "assignment validity is checked before objective quality"),
        fragment("objective_bound", "total score is compared with the exact reference optimum"),
    ]
    return base_constraint_row(row, fragments, representation, 0.90, "assignment rows stress all-different plus objective scoring")


def graph_coloring_smt_row(row: Mapping[str, Any]) -> JsonDict:
    """Represent a graph-coloring fixture as finite-domain edge inequalities."""

    data = mapping(row.get("instance_data"))
    nodes = [str(node) for node in list_value(data.get("nodes"))]
    colors = [int_value(color) for color in list_value(data.get("colors"))]
    representation = {
        "kind": "graph_coloring_finite_domain",
        "logic": "QF_FD",
        "variables": {node: {"domain": colors} for node in nodes},
        "edge_inequalities": list_value(data.get("edges")),
        "objective": objective_bound(row),
    }
    fragments = [
        fragment("graph_relation", "adjacent nodes must have different finite-domain colors"),
        fragment("arithmetic_interval", "each color variable is bounded to the listed color domain"),
        fragment("feasibility", "node coverage and edge conflicts are hard constraints"),
        fragment("objective_bound", "used-color count is compared with the exact reference optimum"),
    ]
    return base_constraint_row(row, fragments, representation, 0.94, "graph-coloring rows exercise relation constraints cleanly")


def base_constraint_row(
    row: Mapping[str, Any],
    fragments: list[JsonDict],
    representation: JsonDict,
    priority_score: float,
    priority_reason: str,
) -> JsonDict:
    """Attach common exact-solver metadata for ConstraintBench rows."""

    return base_smt_row(
        row_id=str(row.get("row_id") or ""),
        artifact_source="exp3211_constraintbench",
        fixture_family=str(row.get("family") or ""),
        formalizability=FULLY_FORMALIZABLE,
        fragments=fragments,
        representation=representation,
        pointer={
            "source_fixture_artifact": CONSTRAINT_FIXTURE_REL_PATH.as_posix(),
            "checker_backend": str(row.get("checker_backend") or ""),
            "row_id": str(row.get("row_id") or ""),
            "authority": "local_exhaustive_enumeration",
        },
        unformalized=[],
        priority_score=priority_score,
        priority_reason=priority_reason,
    )


def base_smt_row(
    *,
    row_id: str,
    artifact_source: str,
    fixture_family: str,
    formalizability: str,
    fragments: list[JsonDict],
    representation: JsonDict | None,
    pointer: JsonDict | None,
    unformalized: list[str],
    priority_score: float,
    priority_reason: str,
) -> JsonDict:
    """Create a row with stable keys so downstream scoring can diff artifacts."""

    return {
        "row_id": row_id,
        "artifact_source": artifact_source,
        "fixture_family": fixture_family,
        "formalizability": formalizability,
        "constraint_fragments": sorted({frag["fragment_type"] for frag in fragments}),
        "formalized_fragments": fragments,
        "unformalized_requirements": unformalized,
        "solver_ready_representation": representation,
        "exact_solver_pointer": pointer,
        "solver_authority_preserved": True,
        "logitext_boundary": (
            "natural-language extraction is not used as solver evidence in this pilot"
        ),
        "exp3225_priority_score": round(unit_interval(priority_score), 6),
        "exp3225_priority_reason": priority_reason,
    }


def objective_bound(row: Mapping[str, Any]) -> JsonDict:
    """Return the exact reference objective in a solver-friendly shape."""

    exact = mapping(row.get("exact_reference"))
    objective = mapping(row.get("objective"))
    return {
        "sense": str(exact.get("objective_sense") or objective.get("sense") or ""),
        "name": str(exact.get("objective_name") or objective.get("name") or ""),
        "exact_value": exact.get("objective_value"),
        "feasible_reference": bool(exact.get("feasible") is True),
        "solution": exact.get("solution") or {},
        "feasible_count": exact.get("feasible_count"),
    }


def source_fixture_artifacts(root: Path) -> list[JsonDict]:
    """Describe the four source artifacts, including presence and checksums."""

    sources: list[JsonDict] = []
    for source_id, rel_path, source_type in SOURCE_SPECS:
        path = root / rel_path
        exists = path.exists()
        sources.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "type": source_type,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
                "row_count": source_row_count(path, source_type) if exists else 0,
            }
        )
    return sources


def source_row_count(path: Path, source_type: str) -> int:
    """Count JSONL rows or read fixture_count from source artifacts."""

    if source_type == "jsonl":
        return len(read_jsonl_objects(path))
    payload = read_json_object(path)
    return int_value(payload.get("fixture_count"))


def formalizability_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count the three formalizability labels with stable zero defaults."""

    return {
        label: sum(row.get("formalizability") == label for row in rows)
        for label in FORMALIZABILITY_LABELS
    }


def coverage_by_fixture_family(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize full, partial, and missing coverage per fixture family."""

    coverage: dict[str, JsonDict] = {}
    for row in rows:
        family = str(row.get("fixture_family") or "")
        entry = coverage.setdefault(
            family,
            {
                "row_count": 0,
                FULLY_FORMALIZABLE: 0,
                PARTIALLY_FORMALIZABLE: 0,
                NOT_FORMALIZABLE: 0,
                "partial_smt_coverage": 0.0,
                "fragment_types": [],
                "exact_solver_row_count": 0,
            },
        )
        entry["row_count"] += 1
        entry[str(row.get("formalizability"))] += 1
        entry["exact_solver_row_count"] += int(bool(row.get("exact_solver_pointer")))
        fragments = set(entry["fragment_types"])
        fragments.update(row.get("constraint_fragments") or [])
        entry["fragment_types"] = sorted(fragments)
    for entry in coverage.values():
        entry["partial_smt_coverage"] = safe_rate(
            entry[FULLY_FORMALIZABLE] + entry[PARTIALLY_FORMALIZABLE],
            entry["row_count"],
        )
    return dict(sorted(coverage.items()))


def highest_value_rows(rows: Sequence[Mapping[str, Any]], limit: int = 12) -> list[JsonDict]:
    """Choose rows that give Exp 3225 the most useful clean verifier coverage."""

    ranked = sorted(
        rows,
        key=lambda row: (-float(row.get("exp3225_priority_score") or 0.0), str(row.get("row_id") or "")),
    )
    return [
        {
            "row_id": str(row.get("row_id") or ""),
            "artifact_source": str(row.get("artifact_source") or ""),
            "fixture_family": str(row.get("fixture_family") or ""),
            "formalizability": str(row.get("formalizability") or ""),
            "priority_score": row.get("exp3225_priority_score"),
            "reason": row.get("exp3225_priority_reason"),
            "use_for": "exp3225 clean verifier scoring row selection",
        }
        for row in ranked[:limit]
    ]


def readiness_checks(
    source_errors: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
) -> dict[str, bool]:
    """Compute the explicit gates behind `coverage_ready`."""

    row_count = len(rows)
    formalizable = [
        row
        for row in rows
        if row.get("formalizability") in {FULLY_FORMALIZABLE, PARTIALLY_FORMALIZABLE}
    ]
    return {
        "all_sources_present": not source_errors,
        "has_context_rows": any(row.get("artifact_source") == "exp3210_context" for row in rows),
        "has_constraintbench_rows": any(
            row.get("artifact_source") == "exp3211_constraintbench" for row in rows
        ),
        "counts_match_denominator": sum(counts.values()) == row_count,
        "all_rows_labeled": all(row.get("formalizability") in FORMALIZABILITY_LABELS for row in rows),
        "formalizable_rows_have_solver_material": all(row_has_solver_material(row) for row in formalizable),
        "deterministic_no_llm_substrate": True,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal artifact before it is written or reported.

    Validation keeps this pilot honest.  A row can be useful with only partial
    SMT coverage, but the artifact must not present live model output or
    natural-language extraction as if it were exact solver evidence.
    """

    for field in REQUIRED_ARTIFACT_FIELDS:
        require(field in artifact, f"missing required artifact field: {field}")

    rows = list_value(artifact.get("smt_rows"))
    fixture_count = int_value(artifact.get("fixture_row_count"))
    full = int_value(artifact.get("fully_formalizable_count"))
    partial = int_value(artifact.get("partially_formalizable_count"))
    not_formalizable = int_value(artifact.get("not_formalizable_count"))

    require(len(rows) == fixture_count, "smt row count does not match fixture_row_count")
    require(
        full + partial + not_formalizable == fixture_count,
        "formalizability counts do not match fixture-row denominator",
    )
    require(
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        "inference substrate must be deterministic artifact replay",
    )
    require(artifact.get("conductor_file_modified") is False, "conductor file must not be modified")
    require(artifact.get("active_roadmap_modified") is False, "active roadmap must not be modified")
    require(
        metric_is_unit_interval(artifact.get("partial_smt_coverage")),
        "partial_smt_coverage must be a unit-interval metric",
    )
    require(
        int_value(artifact.get("exact_solver_row_count"))
        == sum(bool(mapping(row).get("exact_solver_pointer")) for row in rows),
        "exact_solver_row_count does not match row pointers",
    )
    for row in rows:
        row_map = mapping(row)
        for field in SMT_ROW_FIELDS:
            require(field in row_map, f"missing SMT row field: {field}")
        require(
            row_map.get("formalizability") in FORMALIZABILITY_LABELS,
            "invalid formalizability label",
        )
        if row_map.get("formalizability") in {FULLY_FORMALIZABLE, PARTIALLY_FORMALIZABLE}:
            require(row_has_solver_material(row_map), "formalizable row lacks solver material")
    if artifact.get("coverage_ready") is True:
        readiness = mapping(artifact.get("readiness_checks"))
        require(readiness and all(readiness.values()), "coverage_ready overclaims failed gates")


def row_has_solver_material(row: Mapping[str, Any]) -> bool:
    """Return true when a formalizable row has exact evidence to point at."""

    return bool(
        row.get("exact_solver_pointer")
        or row.get("solver_ready_representation")
        or row.get("formalized_fragments")
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a conductor-friendly verdict that does not overclaim NLU."""

    if artifact.get("coverage_ready") is True:
        return (
            "complete: partial SMT coverage ready; "
            f"rows={artifact['fixture_row_count']} "
            f"full={artifact['fully_formalizable_count']} "
            f"partial={artifact['partially_formalizable_count']}"
        )
    return "blocked_missing_sources: partial SMT coverage not ready"


def read_json_object(path: Path | str) -> JsonDict:
    """Read a JSON object and fail closed on non-object payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def read_jsonl_objects(path: Path | str) -> list[JsonDict]:
    """Read a JSONL file as objects, ignoring blank lines only."""

    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSONL object row at {path}")
        rows.append(payload)
    return rows


def sha256_file(path: Path) -> str:
    """Return a stable checksum for source artifact provenance."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def fragment(fragment_type: str, description: str) -> JsonDict:
    """Build a typed fragment record from the declared taxonomy."""

    return {"fragment_type": fragment_type, "description": description}


def mapping(value: Any) -> JsonDict:
    """Return a shallow dict for mapping values and an empty dict otherwise."""

    return dict(value) if isinstance(value, Mapping) else {}


def list_value(value: Any) -> list[Any]:
    """Return list values while keeping malformed fields from crashing coverage."""

    return list(value) if isinstance(value, list) else []


def int_value(value: Any) -> int:
    """Convert numeric fixture fields into integers with a safe zero fallback."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_int_or_string(value: Any) -> int | str:
    """Use integer SMT constants when possible, otherwise preserve the literal."""

    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def duration(started_s: float, now_s: float | None) -> float:
    """Return rounded elapsed wall time."""

    now = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, now - started_s), 6)


def safe_rate(numerator: int, denominator: int) -> float:
    """Return a rounded rate with a zero-denominator fallback."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def unit_interval(value: float) -> float:
    """Clamp a score into the closed unit interval."""

    return max(0.0, min(1.0, float(value)))


def numeric_or_none(value: Any) -> float | None:
    """Convert finite numeric values while preserving missing values."""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def metric_is_unit_interval(value: Any) -> bool:
    """Return true when a value is numeric and lies in [0, 1]."""

    numeric = numeric_or_none(value)
    return numeric is not None and 0.0 <= numeric <= 1.0


def require(condition: bool, message: str) -> None:
    """Raise a validation error with a stable message."""

    if not condition:
        raise ValueError(message)


def main() -> None:  # pragma: no cover - CLI exercised by the conductor.
    """Write the requested artifact and print a compact run receipt."""

    output = write_artifact()
    artifact = read_json_object(output)
    print(
        json.dumps(
            {
                "artifact": output.as_posix(),
                "coverage_ready": artifact["coverage_ready"],
                "fixture_row_count": artifact["fixture_row_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
