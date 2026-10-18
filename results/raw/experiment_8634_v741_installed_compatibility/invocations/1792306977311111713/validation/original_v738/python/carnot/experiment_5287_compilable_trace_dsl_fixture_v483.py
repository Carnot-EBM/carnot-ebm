#!/usr/bin/env python3
"""Exp 5287: deterministic compilable trace DSL fixture.

Spec refs: REQ-VERIFY-5287, SCENARIO-VERIFY-5287.

This module turns the Exp 5273 solver cases into a tiny trace DSL fixture. The
DSL is deliberately small: it records claims, dependency links, executable
expressions, compiled constraints, deduction labels, counterexample labels, and
localized repairs. The solver remains the authority, so a valid-looking trace
cannot be accepted unless its compiled constraints match the Exp 5273 label or
its localized repair is applied and checked again.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_5273_solver_fixture_rebuild_v482 as fixture_mod

try:  # pragma: no cover - tests inject the unavailable-solver branch directly.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5287
EXPERIMENT_NAME = "experiment_5287_compilable_trace_dsl_fixture_v483"
RESULT_RELATIVE_PATH = Path("results/experiment_5287_compilable_trace_dsl_fixture_v483.json")
SCHEMA = "carnot.experiment_5287.compilable_trace_dsl_fixture.v483"
TRACE_SCHEMA_VERSION = "compilable_trace_dsl_v1"
SPEC_REFS = ("REQ-VERIFY-5287", "SCENARIO-VERIFY-5287")
INFERENCE_SUBSTRATE = "offline_deterministic_fixture_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")
CASE_TYPES = ("positive", "negative", "malformed", "semantic-error", "repair")
ALLOWED_DEDUCTION_SCHEMAS = (
    "constraints_from_claims",
    "solver_label_from_constraints",
)
ALLOWED_REPAIR_LABELS = ("localized_expression_replacement",)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5287 verdict; starts with complete: or blocked_ and states "
        "whether the trace DSL fixture is usable."
    ),
    "inference_substrate": (
        "Declares offline deterministic fixture compilation and solver checking with no "
        "LLM, GGUF, API, or external judge dependency."
    ),
    "trace_dsl_ready": (
        "Bare gate for exp5288; true only when all trace DSL fixture families compile or "
        "reject as intended, solver correctness is separate from format validity, "
        "localized repairs recheck cleanly, and unsafe false accepts are zero."
    ),
    "trace_dsl_ready_principle": (
        "Explains why the compilable trace DSL fixture can or cannot gate downstream "
        "SOTA extraction."
    ),
    "dsl_schema_summary": (
        "Summarizes the minimal trace DSL fields for claims, dependency links, "
        "executable expressions, constraints, deduction steps, counterexample labels, "
        "and localized repairs."
    ),
    "fixture_case_counts": (
        "Counts positive, negative, malformed, semantic-error, and repair trace cases "
        "so downstream pilots cannot silently drop a failure mode."
    ),
    "solver_correctness_metrics": (
        "Records deterministic solver/verifier outcomes after DSL compilation, "
        "including correct accepts, semantic-error rejections, repair successes, and "
        "solver false-accept candidates."
    ),
    "format_vs_semantic_split": (
        "Shows that schema/format validity is measured separately from solver "
        "correctness, including format-valid traces that remain semantically wrong."
    ),
    "unsafe_false_accepts": (
        "Counts semantically wrong or unrepaired solver-false-accept traces that were "
        "accepted; must be zero for trace_dsl_ready."
    ),
    "tests_run": (
        "Commands run to validate the trace DSL fixture module, artifact schema, "
        "new-code coverage, and repository test status."
    ),
}
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "trace_dsl_ready",
    "trace_dsl_ready_principle",
    "dsl_schema_summary",
    "fixture_case_counts",
    "solver_correctness_metrics",
    "format_vs_semantic_split",
    "unsafe_false_accepts",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "dsl_schema_summary",
    "solver_correctness_metrics",
    "format_vs_semantic_split",
    "unsafe_false_accepts",
)


@dataclass(frozen=True)
class TraceValidation:
    """Schema validation result before a trace is allowed to reach the solver."""

    ok: bool
    errors: tuple[str, ...]


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _utc_run_date() -> str:
    return time.strftime("%Y%m%d", time.gmtime())


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixtures_by_id() -> dict[str, fixture_mod.SolverFixture]:
    return {fixture.fixture_id: fixture for fixture in fixture_mod.fixture_set()}


def trace_cases(
    fixtures: Sequence[fixture_mod.SolverFixture] | None = None,
) -> list[JsonDict]:
    """Return deterministic trace DSL records seeded from Exp 5273 fixtures."""

    active_fixtures = list(fixture_mod.fixture_set() if fixtures is None else fixtures)
    by_id = {fixture.fixture_id: fixture for fixture in active_fixtures}
    cases = [
        _trace_from_fixture(fixture, case_type="positive", suffix="reference")
        for fixture in active_fixtures
    ]
    cases.extend(
        [
            _trace_from_fixture(
                by_id["single_even_high"], case_type="negative", suffix="counterexample"
            ),
            _trace_from_fixture(
                by_id["even_and_odd"], case_type="negative", suffix="counterexample"
            ),
            _trace_from_fixture(
                by_id["even_and_odd"],
                case_type="semantic-error",
                suffix="empty_false_accept",
                drop_constraints=True,
                include_counterexamples=False,
            ),
            _trace_from_fixture(
                by_id["small_pair_sum"],
                case_type="semantic-error",
                suffix="overconstrained",
                expression_overrides={"sum_is_five": "a + b == 6"},
                include_counterexamples=False,
            ),
            _trace_from_fixture(
                by_id["small_pair_sum"],
                case_type="repair",
                suffix="repair_sum",
                expression_overrides={"sum_is_five": "a + b == 6"},
                repairs={"sum_is_five": "a + b == 5"},
            ),
            _trace_from_fixture(
                by_id["too_large_sum"],
                case_type="repair",
                suffix="repair_missing_conflict",
                expression_overrides={"sum_is_five": "p + q <= 5"},
                repairs={"sum_is_five": "p + q == 5"},
            ),
        ]
    )
    malformed_dependency = _trace_from_fixture(
        by_id["single_even_high"], case_type="malformed", suffix="unknown_dependency"
    )
    malformed_dependency["expressions"][0]["depends_on"] = ["missing_claim"]
    malformed_expression = _trace_from_fixture(
        by_id["single_even_high"], case_type="malformed", suffix="bad_expression"
    )
    malformed_expression["expressions"][0]["expr"] = "x is even"
    cases.extend([malformed_dependency, malformed_expression])
    return cases


def _trace_from_fixture(
    fixture: fixture_mod.SolverFixture,
    *,
    case_type: str,
    suffix: str,
    expression_overrides: Mapping[str, str] | None = None,
    drop_constraints: bool = False,
    include_counterexamples: bool = True,
    repairs: Mapping[str, str] | None = None,
) -> JsonDict:
    variables = copy.deepcopy(fixture.reference_encoding["variables"])
    source_constraints = [] if drop_constraints else list(fixture.reference_encoding["constraints"])
    expressions = [
        {
            "id": f"expr_{constraint['id']}",
            "expr": (expression_overrides or {}).get(
                str(constraint["id"]), str(constraint["expr"])
            ),
            "depends_on": ["claim_requirements"],
        }
        for constraint in source_constraints
    ]
    constraints = [
        {
            "id": str(constraint["id"]),
            "expression_id": f"expr_{constraint['id']}",
            "depends_on": [f"expr_{constraint['id']}"],
        }
        for constraint in source_constraints
    ]
    trace_repairs = [
        {
            "id": f"repair_{constraint_id}",
            "label": "localized_expression_replacement",
            "target_id": f"expr_{constraint_id}",
            "replacement_expr": replacement_expr,
            "depends_on": ["deduce_solver_label"],
        }
        for constraint_id, replacement_expr in (repairs or {}).items()
    ]
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": f"trace_{fixture.fixture_id}_{suffix}",
        "fixture_id": fixture.fixture_id,
        "case_type": case_type,
        "variables": variables,
        "claims": [{"id": "claim_requirements", "text": fixture.natural_language}],
        "expressions": expressions,
        "constraints": constraints,
        "deductions": [
            {
                "id": "deduce_constraints",
                "schema": "constraints_from_claims",
                "premises": ["claim_requirements"],
                "conclusion": "constraints_compiled",
            },
            {
                "id": "deduce_solver_label",
                "schema": "solver_label_from_constraints",
                "premises": [str(constraint["id"]) for constraint in source_constraints],
                "conclusion": fixture.expected_status,
            },
        ],
        "counterexamples": _counterexample_labels(fixture) if include_counterexamples else [],
        "repairs": trace_repairs,
    }


def _counterexample_labels(fixture: fixture_mod.SolverFixture) -> list[JsonDict]:
    return [
        {
            "id": f"counterexample_{index}",
            "assignment": row["assignment"],
            "violated_constraints": row["violated_constraints"],
            "label": "violates_listed_constraints",
            "depends_on": list(row["violated_constraints"]),
        }
        for index, row in enumerate(fixture_mod.fixture_counterexample_rows(fixture))
    ]


def fixture_case_counts(cases: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count required trace fixture families without the artifact principle field."""

    counts = Counter(str(case.get("case_type")) for case in cases)
    return {case_type: counts.get(case_type, 0) for case_type in CASE_TYPES}


def validate_trace_schema(record: Mapping[str, Any]) -> TraceValidation:
    """Validate DSL shape, dependency links, and executable expression syntax."""

    errors: list[str] = []
    if record.get("schema_version") != TRACE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TRACE_SCHEMA_VERSION}")
    if record.get("fixture_id") not in _fixtures_by_id():
        errors.append("fixture_id must reference an Exp 5273 solver fixture")
    if record.get("case_type") not in CASE_TYPES:
        errors.append("case_type must be a required trace fixture family")

    variables = _validate_variables(record.get("variables"), errors)
    ids: set[str] = set()
    claims = _object_list(record.get("claims"), "claims", errors)
    expressions = _object_list(record.get("expressions"), "expressions", errors)
    constraints = _object_list(record.get("constraints"), "constraints", errors)
    deductions = _object_list(record.get("deductions"), "deductions", errors)
    counterexamples = _object_list(record.get("counterexamples"), "counterexamples", errors)
    repairs = _object_list(record.get("repairs"), "repairs", errors)

    _collect_claims(claims, ids, errors)
    expression_ids = _collect_expressions(expressions, ids, variables, errors)
    constraint_ids = _collect_constraints(constraints, ids, expression_ids, errors)
    _collect_deductions(deductions, ids, constraint_ids, errors)
    _collect_counterexamples(counterexamples, ids, constraint_ids, errors)
    _collect_repairs(repairs, ids, expression_ids, variables, errors)
    _validate_dependency_links(
        [*expressions, *constraints, *deductions, *counterexamples, *repairs],
        ids,
        errors,
    )
    return TraceValidation(ok=not errors, errors=tuple(errors))


def _validate_variables(raw: Any, errors: list[str]) -> JsonDict:
    if not isinstance(raw, Mapping):
        errors.append("variables must be an object")
        return {}
    variables: JsonDict = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not _IDENTIFIER.match(name):
            errors.append(f"invalid variable name {name!r}")
            continue
        if not isinstance(spec, Mapping) or spec.get("type") != "int":
            errors.append(f"variable {name} must declare type int")
            continue
        variables[name] = {"type": "int"}
    return variables


def _object_list(raw: Any, field: str, errors: list[str]) -> list[JsonDict]:
    if not isinstance(raw, list):
        errors.append(f"{field} must be a list")
        return []
    objects = []
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            objects.append(dict(item))
        else:
            errors.append(f"{field}[{index}] must be an object")
    return objects


def _add_id(row: Mapping[str, Any], ids: set[str], errors: list[str], field: str) -> str | None:
    row_id = row.get("id")
    if not isinstance(row_id, str) or not _IDENTIFIER.match(row_id):
        errors.append(f"{field} has invalid id")
        return None
    if row_id in ids:
        errors.append(f"duplicate id {row_id}")
        return None
    ids.add(row_id)
    return row_id


def _collect_claims(claims: Sequence[Mapping[str, Any]], ids: set[str], errors: list[str]) -> None:
    for claim in claims:
        _add_id(claim, ids, errors, "claim")
        if not isinstance(claim.get("text"), str) or not claim["text"].strip():
            errors.append("claim text must be non-empty")


def _collect_expressions(
    expressions: Sequence[Mapping[str, Any]],
    ids: set[str],
    variables: Mapping[str, Any],
    errors: list[str],
) -> set[str]:
    expression_ids: set[str] = set()
    for expression in expressions:
        expression_id = _add_id(expression, ids, errors, "expression")
        expr = expression.get("expr")
        if expression_id is not None:
            expression_ids.add(expression_id)
        if not isinstance(expr, str) or not expr.strip():
            errors.append("expression expr must be non-empty")
            continue
        errors.extend(_expression_errors(expr, variables))
    return expression_ids


def _collect_constraints(
    constraints: Sequence[Mapping[str, Any]],
    ids: set[str],
    expression_ids: set[str],
    errors: list[str],
) -> set[str]:
    constraint_ids: set[str] = set()
    for constraint in constraints:
        constraint_id = _add_id(constraint, ids, errors, "constraint")
        expression_id = constraint.get("expression_id")
        if constraint_id is not None:
            constraint_ids.add(constraint_id)
        if expression_id not in expression_ids:
            errors.append(f"constraint {constraint_id} references unknown expression_id")
    return constraint_ids


def _collect_deductions(
    deductions: Sequence[Mapping[str, Any]],
    ids: set[str],
    constraint_ids: set[str],
    errors: list[str],
) -> None:
    for deduction in deductions:
        _add_id(deduction, ids, errors, "deduction")
        if deduction.get("schema") not in ALLOWED_DEDUCTION_SCHEMAS:
            errors.append("deduction schema is not allowed")
        if not isinstance(deduction.get("conclusion"), str):
            errors.append("deduction conclusion must be a string")
        premises = deduction.get("premises")
        if not isinstance(premises, list):
            errors.append("deduction premises must be a list")
        elif deduction.get("schema") == "solver_label_from_constraints":
            unknown = [premise for premise in premises if premise not in constraint_ids]
            if unknown:
                errors.append("solver deduction premises must be constraint ids")


def _collect_counterexamples(
    counterexamples: Sequence[Mapping[str, Any]],
    ids: set[str],
    constraint_ids: set[str],
    errors: list[str],
) -> None:
    for counterexample in counterexamples:
        _add_id(counterexample, ids, errors, "counterexample")
        if not _int_mapping(counterexample.get("assignment")):
            errors.append("counterexample assignment must map variables to ints")
        violated = counterexample.get("violated_constraints")
        if not isinstance(violated, list):
            errors.append("counterexample violated_constraints must be a list")
        elif any(constraint_id not in constraint_ids for constraint_id in violated):
            errors.append("counterexample references unknown violated constraint")


def _collect_repairs(
    repairs: Sequence[Mapping[str, Any]],
    ids: set[str],
    expression_ids: set[str],
    variables: Mapping[str, Any],
    errors: list[str],
) -> None:
    for repair in repairs:
        _add_id(repair, ids, errors, "repair")
        if repair.get("label") not in ALLOWED_REPAIR_LABELS:
            errors.append("repair label is not allowed")
        if repair.get("target_id") not in expression_ids:
            errors.append("repair target_id must reference an expression")
        replacement_expr = repair.get("replacement_expr")
        if not isinstance(replacement_expr, str) or not replacement_expr.strip():
            errors.append("repair replacement_expr must be non-empty")
        else:
            errors.extend(_expression_errors(replacement_expr, variables))


def _validate_dependency_links(
    rows: Sequence[Mapping[str, Any]], known_ids: set[str], errors: list[str]
) -> None:
    for row in rows:
        for field in ("depends_on", "premises"):
            links = row.get(field, [])
            if not isinstance(links, list):
                errors.append(f"{field} must be a list")
                continue
            for link in links:
                if not isinstance(link, str) or link not in known_ids:
                    errors.append(f"unknown dependency {link}")


def _expression_errors(expr: str, variables: Mapping[str, Any]) -> list[str]:
    payload = {
        "schema_version": fixture_mod.IR_SCHEMA_VERSION,
        "variables": dict(variables),
        "constraints": [{"id": "expr_check", "expr": expr}],
    }
    validation = fixture_mod.validate_extracted_constraints(payload)
    return list(validation.errors)


def _int_mapping(raw: Any) -> bool:
    return isinstance(raw, Mapping) and all(
        isinstance(key, str) and isinstance(value, int) for key, value in raw.items()
    )


def compile_trace_to_constraint_ir(record: Mapping[str, Any]) -> JsonDict:
    """Compile a schema-valid trace DSL record into Exp 5273 constraint IR."""

    validation = validate_trace_schema(record)
    if not validation.ok:
        raise ValueError(validation.errors[0])
    expressions = {str(row["id"]): str(row["expr"]) for row in record["expressions"]}
    return {
        "schema_version": fixture_mod.IR_SCHEMA_VERSION,
        "variables": copy.deepcopy(record["variables"]),
        "constraints": [
            {"id": str(row["id"]), "expr": expressions[str(row["expression_id"])]}
            for row in record["constraints"]
        ],
    }


def apply_localized_repairs(record: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    """Apply declared local expression replacements and return changed IDs."""

    repaired = copy.deepcopy(dict(record))
    changed: list[str] = []
    expressions = {str(row["id"]): row for row in repaired.get("expressions", [])}
    for repair in repaired.get("repairs", []):
        target_id = str(repair.get("target_id"))
        target = expressions.get(target_id)
        if target is not None and target.get("expr") != repair.get("replacement_expr"):
            target["expr"] = str(repair["replacement_expr"])
            changed.append(target_id)
    return repaired, changed


def check_trace(record: Mapping[str, Any], *, z3_module: Any = _z3) -> JsonDict:
    """Validate, compile, solver-check, and optionally repair one trace record."""

    fixtures = _fixtures_by_id()
    fixture = fixtures.get(str(record.get("fixture_id")))
    expected_status = fixture.expected_status if fixture is not None else "unknown"
    base = _base_row(record, expected_status)
    validation = validate_trace_schema(record)
    if not validation.ok:
        base["schema_errors"] = list(validation.errors)
        return base
    base["format_valid"] = True
    if not _solver_available(z3_module):
        base["schema_errors"] = ["z3_unavailable"]
        return base

    assert fixture is not None
    compiled = compile_trace_to_constraint_ir(record)
    score = fixture_mod.score_candidate(fixture, compiled, z3_module=z3_module)
    deduction_valid = _deduction_labels_valid(record, score.solver_status)
    counterexample_valid = _counterexample_labels_valid(record, compiled)
    semantic_correct = bool(score.matches_expected and deduction_valid and counterexample_valid)
    base.update(
        {
            "solver_was_run": True,
            "solver_status": score.solver_status,
            "solver_false_accept": score.false_accept,
            "semantic_correct": semantic_correct,
            "deduction_labels_valid": deduction_valid,
            "counterexample_labels_valid": counterexample_valid,
            "final_solver_status": score.solver_status,
        }
    )

    if record.get("case_type") == "repair":
        repaired, changed = apply_localized_repairs(record)
        repaired_score = fixture_mod.score_candidate(
            fixture, compile_trace_to_constraint_ir(repaired), z3_module=z3_module
        )
        repair_success = bool(
            changed
            and repaired_score.matches_expected
            and _deduction_labels_valid(repaired, repaired_score.solver_status)
            and _counterexample_labels_valid(repaired, compile_trace_to_constraint_ir(repaired))
        )
        base.update(
            {
                "repair_success": repair_success,
                "repair_target_id": str(record["repairs"][0]["target_id"]),
                "repair_changed_expression_ids": changed,
                "final_solver_status": repaired_score.solver_status,
                "accepted": repair_success,
            }
        )
    elif record.get("case_type") == "semantic-error":
        base["accepted"] = False
    else:
        base["accepted"] = semantic_correct
    return base


def _base_row(record: Mapping[str, Any], expected_status: str) -> JsonDict:
    repair_target = ""
    repairs = record.get("repairs")
    if isinstance(repairs, list) and repairs:
        repair_target = str(repairs[0].get("target_id", ""))
    return {
        "trace_id": str(record.get("trace_id", "")),
        "fixture_id": str(record.get("fixture_id", "")),
        "case_type": str(record.get("case_type", "")),
        "expected_status": expected_status,
        "format_valid": False,
        "schema_errors": [],
        "solver_was_run": False,
        "solver_status": "not_run",
        "solver_false_accept": False,
        "semantic_correct": False,
        "deduction_labels_valid": False,
        "counterexample_labels_valid": False,
        "repair_success": False,
        "repair_target_id": repair_target,
        "repair_changed_expression_ids": [],
        "final_solver_status": "not_run",
        "accepted": False,
    }


def _deduction_labels_valid(record: Mapping[str, Any], solver_status: str) -> bool:
    solver_deductions = [
        row
        for row in record.get("deductions", [])
        if row.get("schema") == "solver_label_from_constraints"
    ]
    return bool(solver_deductions) and all(
        row.get("conclusion") == solver_status for row in solver_deductions
    )


def _counterexample_labels_valid(record: Mapping[str, Any], compiled_ir: Mapping[str, Any]) -> bool:
    by_id = {str(row["id"]): str(row["expr"]) for row in compiled_ir.get("constraints", [])}
    for counterexample in record.get("counterexamples", []):
        assignment = counterexample.get("assignment", {})
        violated = counterexample.get("violated_constraints", [])
        if not violated:
            return False
        for constraint_id in violated:
            expr = by_id.get(str(constraint_id))
            if expr is None:
                return False
            try:
                if fixture_mod._eval_formula(expr, assignment):
                    return False
            except Exception:
                return False
    return True


def evaluate_trace_cases(cases: Sequence[Mapping[str, Any]], *, z3_module: Any = _z3) -> JsonDict:
    """Evaluate all trace DSL cases and compute readiness metrics."""

    rows = [check_trace(case, z3_module=z3_module) for case in cases]
    counts = fixture_case_counts(cases)
    solver_available = _solver_available(z3_module)
    semantic_error_rejections = sum(
        1
        for row in rows
        if row["case_type"] == "semantic-error" and row["format_valid"] and not row["accepted"]
    )
    repair_successes = sum(
        1 for row in rows if row["case_type"] == "repair" and row["repair_success"]
    )
    unsafe_false_accepts = sum(
        1
        for row in rows
        if row["accepted"]
        and row["expected_status"] == "unsat"
        and row["final_solver_status"] == "sat"
    )
    format_valid_semantic_wrong = [
        row["trace_id"] for row in rows if row["format_valid"] and not row["semantic_correct"]
    ]
    metrics = {
        "total_cases": len(rows),
        "solver_available": solver_available,
        "solver_checked_cases": sum(1 for row in rows if row["solver_was_run"]),
        "accepted_cases": sum(1 for row in rows if row["accepted"]),
        "positive_accepts": _accepted_count(rows, "positive"),
        "negative_accepts": _accepted_count(rows, "negative"),
        "malformed_rejections": sum(
            1 for row in rows if row["case_type"] == "malformed" and not row["format_valid"]
        ),
        "semantic_error_rejections": semantic_error_rejections,
        "repair_successes": repair_successes,
        "solver_false_accept_candidates": sum(1 for row in rows if row["solver_false_accept"]),
    }
    split = {
        "format_valid": sum(1 for row in rows if row["format_valid"]),
        "format_invalid": sum(1 for row in rows if not row["format_valid"]),
        "format_valid_semantic_wrong": len(format_valid_semantic_wrong),
        "format_valid_semantic_wrong_trace_ids": format_valid_semantic_wrong,
        "semantic_error_trace_ids": [
            row["trace_id"] for row in rows if row["case_type"] == "semantic-error"
        ],
    }
    ready = _ready(counts, metrics, unsafe_false_accepts)
    return {
        "rows": rows,
        "fixture_case_counts": counts,
        "solver_correctness_metrics": metrics,
        "format_vs_semantic_split": split,
        "unsafe_false_accepts": unsafe_false_accepts,
        "trace_dsl_ready": ready,
        "trace_dsl_ready_principle": _ready_principle(
            ready=ready,
            counts=counts,
            metrics=metrics,
            unsafe_false_accepts=unsafe_false_accepts,
        ),
    }


def _accepted_count(rows: Sequence[Mapping[str, Any]], case_type: str) -> int:
    return sum(1 for row in rows if row["case_type"] == case_type and row["accepted"])


def _ready(
    counts: Mapping[str, int], metrics: Mapping[str, Any], unsafe_false_accepts: int
) -> bool:
    return bool(
        metrics["solver_available"]
        and all(counts[case_type] > 0 for case_type in CASE_TYPES)
        and metrics["positive_accepts"] == counts["positive"]
        and metrics["negative_accepts"] == counts["negative"]
        and metrics["malformed_rejections"] == counts["malformed"]
        and metrics["semantic_error_rejections"] == counts["semantic-error"]
        and metrics["repair_successes"] == counts["repair"]
        and unsafe_false_accepts == 0
    )


def _ready_principle(
    *,
    ready: bool,
    counts: Mapping[str, int],
    metrics: Mapping[str, Any],
    unsafe_false_accepts: int,
) -> str:
    if ready:
        return (
            "ready: Exp5273 solver cases compile through the trace DSL, malformed records "
            "reject before solver scoring, format-valid semantic errors are rejected, "
            "localized repairs recheck cleanly, and unsafe_false_accepts=0 for exp5288."
        )
    blockers = [
        f"solver_available={metrics['solver_available']}",
        "missing_case_types="
        + ",".join(case_type for case_type in CASE_TYPES if counts[case_type] == 0),
        f"semantic_error_rejections={metrics['semantic_error_rejections']}",
        f"repair_successes={metrics['repair_successes']}",
        f"unsafe_false_accepts={unsafe_false_accepts}",
    ]
    return "blocked: " + "; ".join(blockers)


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
    z3_module: Any = _z3,
    write: bool = True,
) -> JsonDict:
    """Build the offline trace DSL fixture artifact and optionally write it."""

    started = time.perf_counter()
    cases = trace_cases()
    summary = evaluate_trace_cases(cases, z3_module=z3_module)
    artifact: JsonDict = {
        "experiment": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "run_date": _utc_run_date(),
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(time.perf_counter() - started, 6),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(summary)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "trace_dsl_ready": bool(summary["trace_dsl_ready"]),
        "trace_dsl_ready_principle": str(summary["trace_dsl_ready_principle"]),
        "dsl_schema_summary": _wrap("dsl_schema_summary", dsl_schema_summary()),
        "fixture_case_counts": dict(summary["fixture_case_counts"])
        | {"principle": FIELD_PRINCIPLES["fixture_case_counts"]},
        "solver_correctness_metrics": _wrap(
            "solver_correctness_metrics", summary["solver_correctness_metrics"]
        ),
        "format_vs_semantic_split": _wrap(
            "format_vs_semantic_split", summary["format_vs_semantic_split"]
        ),
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", summary["unsafe_false_accepts"]),
        "tests_run": [dict(row) for row in tests_run],
        "case_results": summary["rows"],
        "trace_fixture_checksum": _stable_json(cases),
        "source_fixture": {
            "experiment": 5273,
            "path": str(fixture_mod.RESULT_RELATIVE_PATH),
            "fixture_ids": sorted(_fixtures_by_id()),
        },
    }
    validate_artifact(artifact)
    if write:
        write_json(result_path, artifact)
    return artifact


def dsl_schema_summary() -> JsonDict:
    """Return the compact schema description embedded in the result artifact."""

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "compiler_target": fixture_mod.IR_SCHEMA_VERSION,
        "top_level_fields": [
            "schema_version",
            "trace_id",
            "fixture_id",
            "case_type",
            "variables",
            "claims",
            "expressions",
            "constraints",
            "deductions",
            "counterexamples",
            "repairs",
        ],
        "dependency_fields": ["depends_on", "premises", "expression_id", "target_id"],
        "deduction_schemas": list(ALLOWED_DEDUCTION_SCHEMAS),
        "repair_labels": list(ALLOWED_REPAIR_LABELS),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 5287 artifact violates the required schema."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field {field}"
    for field in WRAPPED_FIELDS:
        value = artifact[field]
        assert isinstance(value, Mapping), f"{field} must be principle-wrapped"
        assert "value" in value and "principle" in value, f"{field} must be principle-wrapped"
        assert value["principle"] == FIELD_PRINCIPLES[field], f"{field} principle mismatch"

    verdict = artifact["honest_verdict"]["value"]
    assert isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), (
        "honest_verdict.value must start with complete: or blocked_"
    )
    assert "usable" in verdict, "honest_verdict.value must state whether fixture is usable"
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, (
        f"inference_substrate.value must be {INFERENCE_SUBSTRATE}"
    )
    assert isinstance(artifact["trace_dsl_ready"], bool), "trace_dsl_ready must be a bare bool"
    assert (
        isinstance(artifact["trace_dsl_ready_principle"], str)
        and artifact["trace_dsl_ready_principle"]
    ), "trace_dsl_ready_principle must be non-empty"
    counts = artifact["fixture_case_counts"]
    assert isinstance(counts, Mapping), "fixture_case_counts must be object"
    assert counts.get("principle") == FIELD_PRINCIPLES["fixture_case_counts"], (
        "fixture_case_counts principle mismatch"
    )
    for case_type in CASE_TYPES:
        assert isinstance(counts.get(case_type), int), f"fixture_case_counts missing {case_type}"
    assert isinstance(artifact["unsafe_false_accepts"]["value"], int), (
        "unsafe_false_accepts.value must be int"
    )
    assert isinstance(artifact["tests_run"], list), "tests_run must be a list"
    if artifact["trace_dsl_ready"]:
        assert artifact["unsafe_false_accepts"]["value"] == 0, (
            "ready trace DSL requires zero unsafe false accepts"
        )
        metrics = artifact["solver_correctness_metrics"]["value"]
        assert metrics["semantic_error_rejections"] == counts["semantic-error"], (
            "ready trace DSL requires semantic-error rejections"
        )
        assert metrics["repair_successes"] == counts["repair"], (
            "ready trace DSL requires repair successes"
        )


def _honest_verdict(summary: Mapping[str, Any]) -> str:
    if summary["trace_dsl_ready"]:
        return "complete: trace DSL fixture usable for exp5288 solver-checked extraction"
    return "blocked_trace_dsl_unusable: trace DSL fixture usable=false; see ready principle"


def _solver_available(z3_module: Any) -> bool:
    return z3_module is not None and hasattr(z3_module, "Solver") and hasattr(z3_module, "Int")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument(
        "--tests-run",
        default="[]",
        help="JSON list of {command, outcome} records to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(result_path=Path(args.output), tests_run=json.loads(args.tests_run), write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
