"""Exp 5273: deterministic solver-labeled fixture rebuild for V482.

Spec refs: REQ-VERIFY-5273, SCENARIO-VERIFY-5273.

This module rebuilds the constraint-extraction fixture under solver authority.
It does not call a model. The goal is to make the next extraction retry fair:
natural-language prompts, executable constraint encodings, labels, controls,
counterexamples, and checksums are all fixed before any LLM is allowed to
propose constraints.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

try:  # pragma: no cover - the blocked run path covers injected unavailability.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5273_solver_fixture_rebuild_v482.json")
V481_RESULT_RELATIVE_PATH = Path(
    "results/experiment_5262_solver_grounded_constraint_extraction_v481.json"
)
SCHEMA = "carnot.experiment_5273.solver_fixture_rebuild.v482"
SPEC_REFS = ("REQ-VERIFY-5273", "SCENARIO-VERIFY-5273")
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
IR_SCHEMA_VERSION = "solver_constraint_ir_v1"
TERMINAL_PREFIXES = ("complete:", "blocked_")
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5273 verdict; starts with complete: or blocked_ and states "
        "whether the deterministic solver fixture is ready for Exp 5274."
    ),
    "inference_substrate": (
        "Declares an offline deterministic certificate with no LLM call, preventing "
        "fixture readiness from being mistaken for live extraction quality."
    ),
    "solver_fixture_ready": (
        "Bare gate for Exp 5274; true only when schema validation, exact solver "
        "labels, reference baseline validity, controls, counterexamples, and "
        "checksums all pass."
    ),
    "solver_fixture_ready_principle": (
        "Explains the exact deterministic checks that opened or blocked the "
        "downstream extraction retry gate."
    ),
    "fixture_count": (
        "Counts solver-labeled natural-language fixtures so downstream extraction "
        "cannot silently shrink the evaluation panel."
    ),
    "baseline_validity": (
        "Reference-copy baseline validity proves the executable labels are "
        "internally consistent before any model is scored."
    ),
    "counterexample_coverage": (
        "Fraction of fixtures with deterministic negative assignments or "
        "counterexamples, preventing SAT-only or empty-extraction baselines from "
        "looking useful."
    ),
    "schema_checks_passed": (
        "Records whether malformed extracted constraints were rejected before "
        "solver scoring."
    ),
    "fixture_checksums": (
        "Content-addressed receipts for prompts, labels, reference encodings, "
        "counterexamples, schema, and the fixture set prevent silent fixture drift."
    ),
    "tests_run": (
        "Commands run to validate fixture generation, schema validation, solver "
        "scoring, new-code coverage, and repository test status."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "solver_fixture_ready",
    "solver_fixture_ready_principle",
    "fixture_count",
    "baseline_validity",
    "counterexample_coverage",
    "schema_checks_passed",
    "fixture_checksums",
    "tests_run",
)
WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "fixture_count",
    "baseline_validity",
    "counterexample_coverage",
    "schema_checks_passed",
    "fixture_checksums",
)


@dataclass(frozen=True)
class SolverFixture:
    """A natural-language constraint case with a solver-checked label."""

    fixture_id: str
    family: str
    natural_language: str
    expected_status: str
    gold_assignment: JsonDict
    reference_encoding: JsonDict
    negative_assignments: tuple[JsonDict, ...]


@dataclass(frozen=True)
class ConstraintExpression:
    """One executable expression from the extracted-constraint schema."""

    constraint_id: str
    expr: str


@dataclass(frozen=True)
class ConstraintCandidate:
    """Schema-normalized extracted constraints ready for exact solver scoring."""

    variables: JsonDict
    constraints: tuple[ConstraintExpression, ...]
    raw_payload: JsonDict


@dataclass(frozen=True)
class SchemaValidation:
    """Result of validating extraction schema before solver scoring."""

    ok: bool
    candidate: ConstraintCandidate | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ScoreResult:
    """Solver result for one candidate against one fixture label."""

    fixture_id: str
    schema_valid: bool
    solver_status: str
    expected_status: str
    matches_expected: bool
    false_accept: bool
    assignment: JsonDict
    counterexample: JsonDict
    errors: tuple[str, ...]

    def to_dict(self) -> JsonDict:
        return {
            "fixture_id": self.fixture_id,
            "schema_valid": self.schema_valid,
            "solver_status": self.solver_status,
            "expected_status": self.expected_status,
            "matches_expected": self.matches_expected,
            "false_accept": self.false_accept,
            "assignment": self.assignment,
            "counterexample": self.counterexample,
            "errors": list(self.errors),
        }


def sha16(value: str | bytes) -> str:
    """Return a short stable checksum for compact receipts."""

    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def fixture_set() -> list[SolverFixture]:
    """Return the fixed SAT/UNSAT panel for Exp 5274 extraction retries."""

    return [
        SolverFixture(
            fixture_id="single_even_high",
            family="single_integer_parity",
            natural_language=(
                "Choose one integer x. It must be at least 0 and at most 5, "
                "it must be even, and it must be greater than 3."
            ),
            expected_status="sat",
            gold_assignment={"x": 4},
            reference_encoding=_encoding(
                variables=("x",),
                constraints=(
                    ("x_low", "x >= 0"),
                    ("x_high", "x <= 5"),
                    ("x_even", "x % 2 == 0"),
                    ("x_gt_three", "x > 3"),
                ),
            ),
            negative_assignments=({"x": 5},),
        ),
        SolverFixture(
            fixture_id="small_pair_sum",
            family="bounded_pair_sum",
            natural_language=(
                "Choose integers a and b. Each must be between 0 and 3 "
                "inclusive. Their sum must equal 5, and a must be less than b."
            ),
            expected_status="sat",
            gold_assignment={"a": 2, "b": 3},
            reference_encoding=_encoding(
                variables=("a", "b"),
                constraints=(
                    ("a_low", "a >= 0"),
                    ("a_high", "a <= 3"),
                    ("b_low", "b >= 0"),
                    ("b_high", "b <= 3"),
                    ("sum_is_five", "a + b == 5"),
                    ("a_less_than_b", "a < b"),
                ),
            ),
            negative_assignments=({"a": 3, "b": 2},),
        ),
        SolverFixture(
            fixture_id="fixed_schedule_window",
            family="linear_schedule",
            natural_language=(
                "Choose integer start s, duration d, and end e. The start must "
                "be exactly 1, the duration must be exactly 4, the end must "
                "equal start plus duration, and the end must be no later than 5."
            ),
            expected_status="sat",
            gold_assignment={"d": 4, "e": 5, "s": 1},
            reference_encoding=_encoding(
                variables=("s", "d", "e"),
                constraints=(
                    ("start_fixed", "s == 1"),
                    ("duration_fixed", "d == 4"),
                    ("end_sum", "s + d == e"),
                    ("end_bound", "e <= 5"),
                ),
            ),
            negative_assignments=({"s": 1, "d": 3, "e": 4},),
        ),
        SolverFixture(
            fixture_id="even_and_odd",
            family="parity_contradiction",
            natural_language=(
                "Choose one integer y. It must be between 1 and 4 inclusive, "
                "it must be even, and it must also be odd."
            ),
            expected_status="unsat",
            gold_assignment={},
            reference_encoding=_encoding(
                variables=("y",),
                constraints=(
                    ("y_low", "y >= 1"),
                    ("y_high", "y <= 4"),
                    ("y_even", "y % 2 == 0"),
                    ("y_odd", "y % 2 == 1"),
                ),
            ),
            negative_assignments=({"y": 2},),
        ),
        SolverFixture(
            fixture_id="too_large_sum",
            family="bounded_sum_contradiction",
            natural_language=(
                "Choose integers p and q. Each must be between 0 and 2 "
                "inclusive. The sum p plus q must equal 5."
            ),
            expected_status="unsat",
            gold_assignment={},
            reference_encoding=_encoding(
                variables=("p", "q"),
                constraints=(
                    ("p_low", "p >= 0"),
                    ("p_high", "p <= 2"),
                    ("q_low", "q >= 0"),
                    ("q_high", "q <= 2"),
                    ("sum_is_five", "p + q == 5"),
                ),
            ),
            negative_assignments=({"p": 2, "q": 2},),
        ),
        SolverFixture(
            fixture_id="bounded_gap_conflict",
            family="bounded_gap_contradiction",
            natural_language=(
                "Choose integers u and v. Both must be between 0 and 2 "
                "inclusive. The value u must be more than 2 greater than v."
            ),
            expected_status="unsat",
            gold_assignment={},
            reference_encoding=_encoding(
                variables=("u", "v"),
                constraints=(
                    ("u_low", "u >= 0"),
                    ("u_high", "u <= 2"),
                    ("v_low", "v >= 0"),
                    ("v_high", "v <= 2"),
                    ("gap_gt_two", "u - v > 2"),
                ),
            ),
            negative_assignments=({"u": 2, "v": 0},),
        ),
    ]


def validate_extracted_constraints(payload: Mapping[str, Any]) -> SchemaValidation:
    """Reject malformed extracted constraints before any solver receives them."""

    errors: list[str] = []
    if payload.get("schema_version") != IR_SCHEMA_VERSION:
        errors.append("schema_version must be solver_constraint_ir_v1")

    variables_raw = payload.get("variables")
    variables: JsonDict = {}
    if not isinstance(variables_raw, Mapping):
        errors.append("variables must be an object")
    else:
        for name, spec in variables_raw.items():
            if not isinstance(name, str) or not _IDENTIFIER.match(name):
                errors.append(f"invalid variable name {name!r}")
                continue
            if not isinstance(spec, Mapping) or spec.get("type") != "int":
                errors.append(f"variable {name} must declare type int")
                continue
            variables[name] = {"type": "int"}

    constraints_raw = payload.get("constraints")
    constraints: list[ConstraintExpression] = []
    if not isinstance(constraints_raw, list):
        errors.append("constraints must be a list")
    else:
        for index, row in enumerate(constraints_raw):
            if not isinstance(row, Mapping):
                errors.append(f"constraint {index} must be an object")
                continue
            constraint_id = row.get("id")
            expr = row.get("expr")
            if not isinstance(constraint_id, str) or not _IDENTIFIER.match(constraint_id):
                errors.append(f"constraint {index} has invalid id")
                continue
            if not isinstance(expr, str) or not expr.strip():
                errors.append(f"constraint {constraint_id} has invalid expr")
                continue
            try:
                _validate_expression(expr, set(variables))
            except ValueError as exc:
                errors.append(f"constraint {constraint_id}: {exc}")
                continue
            constraints.append(ConstraintExpression(constraint_id=constraint_id, expr=expr.strip()))

    if errors:
        return SchemaValidation(ok=False, candidate=None, errors=tuple(errors))
    return SchemaValidation(
        ok=True,
        candidate=ConstraintCandidate(
            variables=variables,
            constraints=tuple(constraints),
            raw_payload=dict(payload),
        ),
        errors=(),
    )


def score_candidate(
    fixture: SolverFixture,
    payload: Mapping[str, Any],
    *,
    z3_module: Any = _z3,
) -> ScoreResult:
    """Schema-check and solver-score one candidate against a fixture label."""

    schema = validate_extracted_constraints(payload)
    if not schema.ok:
        return ScoreResult(
            fixture_id=fixture.fixture_id,
            schema_valid=False,
            solver_status="schema_error",
            expected_status=fixture.expected_status,
            matches_expected=False,
            false_accept=False,
            assignment={},
            counterexample={"schema_errors": list(schema.errors)},
            errors=schema.errors,
        )
    if not _checker_available(z3_module):  # pragma: no cover - run() blocks before scoring.
        return ScoreResult(
            fixture_id=fixture.fixture_id,
            schema_valid=True,
            solver_status="solver_unavailable",
            expected_status=fixture.expected_status,
            matches_expected=False,
            false_accept=False,
            assignment={},
            counterexample={"solver": "z3_unavailable"},
            errors=("z3_unavailable",),
        )

    assert schema.candidate is not None
    env = {name: z3_module.Int(name) for name in sorted(schema.candidate.variables)}
    solver = z3_module.Solver()
    solver.set(timeout=2000)
    for constraint in schema.candidate.constraints:
        solver.add(_compile_formula(constraint.expr, env, z3_module))
    status = solver.check()
    if status == z3_module.sat:
        solver_status = "sat"
        assignment = _model_assignment(solver.model(), env)
    elif status == z3_module.unsat:
        solver_status = "unsat"
        assignment = {}
    else:  # pragma: no cover - tiny deterministic integer fixtures should decide.
        solver_status = "unknown"
        assignment = {}

    matches = solver_status == fixture.expected_status
    false_accept = fixture.expected_status == "unsat" and solver_status == "sat"
    counterexample = _counterexample_for_result(fixture, solver_status, assignment, matches)
    return ScoreResult(
        fixture_id=fixture.fixture_id,
        schema_valid=True,
        solver_status=solver_status,
        expected_status=fixture.expected_status,
        matches_expected=matches,
        false_accept=false_accept,
        assignment=assignment,
        counterexample=counterexample,
        errors=(),
    )


def fixture_counterexample_rows(fixture: SolverFixture) -> list[JsonDict]:
    """Return deterministic negative assignments and the constraints they violate."""

    rows: list[JsonDict] = []
    constraints = [
        ConstraintExpression(constraint_id=str(row["id"]), expr=str(row["expr"]))
        for row in fixture.reference_encoding["constraints"]
    ]
    for assignment in fixture.negative_assignments:
        violated = [
            constraint.constraint_id
            for constraint in constraints
            if not bool(_eval_formula(constraint.expr, assignment))
        ]
        rows.append({"assignment": dict(assignment), "violated_constraints": violated})
    return rows


def counterexample_coverage(fixtures: Sequence[SolverFixture]) -> float:
    """Return the fraction of fixtures with at least one rejecting assignment."""

    if not fixtures:
        return 0.0
    covered = sum(1 for fixture in fixtures if fixture_counterexample_rows(fixture))
    return covered / len(fixtures)


def score_baselines(
    fixtures: Sequence[SolverFixture],
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Score deterministic controls that separate fixture sanity from extraction."""

    arms: dict[str, list[tuple[SolverFixture, JsonDict, str | None]]] = {
        "reference_copy": [
            (fixture, fixture.reference_encoding, fixture.fixture_id) for fixture in fixtures
        ],
        "empty_extraction": [
            (fixture, _empty_encoding(), None) for fixture in fixtures
        ],
        "deterministic_shuffled_reference": [
            (
                fixture,
                fixtures[(index + 2) % len(fixtures)].reference_encoding,
                fixtures[(index + 2) % len(fixtures)].fixture_id,
            )
            for index, fixture in enumerate(fixtures)
        ],
    }
    baselines: JsonDict = {}
    for arm, items in arms.items():
        rows: list[JsonDict] = []
        for fixture, payload, source_fixture_id in items:
            score = score_candidate(fixture, payload, z3_module=z3_module)
            row = score.to_dict()
            row["source_fixture_id"] = source_fixture_id
            rows.append(row)
        baselines[arm] = _aggregate_scores(rows) | {"rows": rows}
    return baselines


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
    z3_module: Any = _z3,
    root: Path = REPO_ROOT,
    write: bool = True,
) -> JsonDict:
    """Build the offline fixture certificate and optionally write the artifact."""

    started = time.perf_counter()
    fixtures = fixture_set()
    checksums = fixture_checksums(fixtures)
    checker_available = _checker_available(z3_module)
    schema_report = run_schema_checks()
    schema_checks_passed = bool(schema_report["passed"] and checker_available)
    coverage = counterexample_coverage(fixtures)

    if checker_available:
        baselines = score_baselines(fixtures, z3_module=z3_module)
        baseline_validity = float(baselines["reference_copy"]["validity_rate"])
    else:
        baselines = {}
        baseline_validity = 0.0

    ready = (
        checker_available
        and schema_checks_passed
        and baseline_validity == 1.0
        and coverage == 1.0
        and bool(checksums["fixture_set_sha256"])
        and "empty_extraction" in baselines
        and "deterministic_shuffled_reference" in baselines
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "duration_s": round(time.perf_counter() - started, 6),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(ready, checker_available)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "solver_fixture_ready": bool(ready),
        "solver_fixture_ready_principle": _ready_principle(
            ready=ready,
            checker_available=checker_available,
            schema_checks_passed=schema_checks_passed,
            baseline_validity=baseline_validity,
            coverage=coverage,
        ),
        "fixture_count": _wrap("fixture_count", len(fixtures)),
        "baseline_validity": _wrap("baseline_validity", baseline_validity),
        "counterexample_coverage": _wrap("counterexample_coverage", coverage),
        "schema_checks_passed": _wrap("schema_checks_passed", schema_checks_passed),
        "fixture_checksums": _wrap("fixture_checksums", checksums),
        "tests_run": [dict(row) for row in tests_run],
        "fixtures": [_fixture_to_dict(fixture) for fixture in fixtures],
        "baselines": baselines,
        "schema_check_report": schema_report,
        "prior_v481_diagnosis": diagnose_v481(root),
        "no_llm_invoked": True,
    }
    validate_artifact(artifact)
    if write:
        write_json(result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 5273 artifact violates the required schema."""

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
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, (
        f"inference_substrate.value must be {INFERENCE_SUBSTRATE}"
    )
    assert isinstance(artifact["solver_fixture_ready"], bool), (
        "solver_fixture_ready must be a bare bool"
    )
    assert isinstance(artifact["solver_fixture_ready_principle"], str) and artifact[
        "solver_fixture_ready_principle"
    ], "solver_fixture_ready_principle must be non-empty"
    assert isinstance(artifact["fixture_count"]["value"], int), (
        "fixture_count.value must be int"
    )
    assert _rate_ok(artifact["baseline_validity"]["value"]), (
        "baseline_validity.value must be numeric in [0, 1]"
    )
    assert _rate_ok(artifact["counterexample_coverage"]["value"]), (
        "counterexample_coverage.value must be numeric in [0, 1]"
    )
    assert isinstance(artifact["schema_checks_passed"]["value"], bool), (
        "schema_checks_passed.value must be bool"
    )
    assert isinstance(artifact["fixture_checksums"]["value"], Mapping), (
        "fixture_checksums.value must be object"
    )
    assert isinstance(artifact["tests_run"], list), "tests_run must be a list"
    if artifact["solver_fixture_ready"]:
        assert artifact["baseline_validity"]["value"] == 1.0, (
            "ready fixture requires reference baseline validity 1.0"
        )
        assert artifact["counterexample_coverage"]["value"] == 1.0, (
            "ready fixture requires full counterexample coverage"
        )
        assert artifact["schema_checks_passed"]["value"] is True, (
            "ready fixture requires schema checks"
        )
        assert artifact["fixture_checksums"]["value"].get("fixture_set_sha256"), (
            "ready fixture requires fixture_set_sha256"
        )


def run_schema_checks() -> JsonDict:
    """Exercise schema acceptance and rejection cases used by the gate."""

    valid = validate_extracted_constraints(fixture_set()[0].reference_encoding).ok
    malformed = [
        {"variables": {"x": {"type": "int"}}, "constraints": []},
        {"schema_version": IR_SCHEMA_VERSION, "variables": ["x"], "constraints": []},
        {
            "schema_version": IR_SCHEMA_VERSION,
            "variables": {"x": {"type": "int"}},
            "constraints": [{"id": "bad", "expr": "x is even"}],
        },
    ]
    rejections = [not validate_extracted_constraints(payload).ok for payload in malformed]
    return {
        "valid_reference_accepted": valid,
        "malformed_rejections": rejections,
        "passed": bool(valid and all(rejections)),
    }


def fixture_checksums(fixtures: Sequence[SolverFixture]) -> JsonDict:
    """Return content receipts for fixture prompts, labels, encodings, and schema."""

    fixture_receipts: JsonDict = {}
    for fixture in fixtures:
        fixture_receipts[fixture.fixture_id] = {
            "natural_language_sha256": _sha256(fixture.natural_language),
            "label_sha256": _sha256(fixture.expected_status),
            "reference_encoding_sha256": _sha256(_stable_json(fixture.reference_encoding)),
            "counterexamples_sha256": _sha256(_stable_json(list(fixture.negative_assignments))),
        }
    schema_receipt = {
        "schema_version": IR_SCHEMA_VERSION,
        "required_top_level_fields": ("schema_version", "variables", "constraints"),
        "constraint_fields": ("id", "expr"),
        "variable_type": "int",
    }
    return {
        "schema_sha256": _sha256(_stable_json(schema_receipt)),
        "fixture_set_sha256": _sha256(_stable_json([_fixture_to_dict(fixture) for fixture in fixtures])),
        "fixtures": fixture_receipts,
    }


def diagnose_v481(root: Path = REPO_ROOT) -> JsonDict:
    """Summarize why the prior V481 pilot did not provide a fair extraction gate."""

    path = root / V481_RESULT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("extraction_results", [])
    parseable = sum(1 for row in rows if row.get("parseable"))
    total = len(rows)
    baseline = payload.get("baseline", {})
    baseline_validity = float(baseline.get("validity_rate", 0.0))
    model_validity = float(payload.get("constraint_validity_rate", {}).get("value", 0.0))
    return {
        "source_artifact": str(path),
        "baseline_validity": baseline_validity,
        "baseline_false_accepts": int(baseline.get("false_accepts", 0)),
        "model_validity": model_validity,
        "parseable_rows": parseable,
        "total_rows": total,
        "baseline_validity_not_useful": baseline_validity == 0.5
        and int(baseline.get("false_accepts", 0)) > 0,
        "model_validity_not_useful": parseable < total or model_validity < baseline_validity,
        "diagnosis": (
            "V481 mixed malformed extraction with solver validity: only "
            f"{parseable}/{total} model rows were parseable, while the empty "
            f"baseline scored {baseline_validity} by accepting all SAT-shaped "
            "cases and false-accepting UNSAT contradictions."
        ),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _encoding(
    *,
    variables: Sequence[str],
    constraints: Sequence[tuple[str, str]],
) -> JsonDict:
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "variables": {name: {"type": "int"} for name in variables},
        "constraints": [{"id": constraint_id, "expr": expr} for constraint_id, expr in constraints],
    }


def _empty_encoding() -> JsonDict:
    return {"schema_version": IR_SCHEMA_VERSION, "variables": {}, "constraints": []}


def _validate_expression(expr: str, variables: set[str]) -> None:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"syntax error: {exc.msg}") from exc
    _validate_ast(tree.body, variables)


def _validate_ast(node: ast.AST, variables: set[str]) -> None:
    if isinstance(node, ast.Compare):
        _validate_ast(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if not isinstance(op, ast.Eq | ast.Lt | ast.LtE | ast.Gt | ast.GtE):
                raise ValueError("unsupported comparison operator")
            _validate_ast(comparator, variables)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mod):
            raise ValueError("unsupported arithmetic operator")
        _validate_ast(node.left, variables)
        _validate_ast(node.right, variables)
        return
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"unknown variable {node.id}")
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return
    raise ValueError("unsupported expression")


def _compile_formula(expression: str, env: Mapping[str, Any], z3_module: Any) -> Any:
    tree = ast.parse(expression.strip(), mode="eval")
    return _compile_ast(tree.body, env, z3_module)


def _compile_ast(node: ast.AST, env: Mapping[str, Any], z3_module: Any) -> Any:
    if isinstance(node, ast.Compare):
        left = _compile_ast(node.left, env, z3_module)
        pieces = []
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _compile_ast(comparator, env, z3_module)
            pieces.append(_compare(left, op, right))
            left = right
        return z3_module.And(*pieces) if len(pieces) > 1 else pieces[0]
    if isinstance(node, ast.BinOp):
        left = _compile_ast(node.left, env, z3_module)
        right = _compile_ast(node.right, env, z3_module)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left % right
    if isinstance(node, ast.Name):
        return env[node.id]
    return z3_module.IntVal(node.value)


def _compare(left: Any, op: ast.cmpop, right: Any) -> Any:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    return left >= right


def _eval_formula(expression: str, assignment: Mapping[str, int]) -> bool:
    tree = ast.parse(expression.strip(), mode="eval")
    return bool(_eval_ast(tree.body, assignment))


def _eval_ast(node: ast.AST, assignment: Mapping[str, int]) -> Any:
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, assignment)
        result = True
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval_ast(comparator, assignment)
            result = result and _eval_compare(left, op, right)
            left = right
        return result
    if isinstance(node, ast.BinOp):
        left = int(_eval_ast(node.left, assignment))
        right = int(_eval_ast(node.right, assignment))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left % right
    if isinstance(node, ast.Name):
        return int(assignment[node.id])
    return int(node.value)


def _eval_compare(left: int, op: ast.cmpop, right: int) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    return left >= right


def _model_assignment(model: Any, env: Mapping[str, Any]) -> JsonDict:
    assignment: JsonDict = {}
    for name, var in env.items():
        assignment[name] = int(model.evaluate(var, model_completion=True).as_long())
    return assignment


def _counterexample_for_result(
    fixture: SolverFixture,
    solver_status: str,
    assignment: Mapping[str, Any],
    matches: bool,
) -> JsonDict:
    if matches:
        return {}
    if fixture.expected_status == "unsat" and solver_status == "sat":
        return {
            "satisfying_assignment": dict(assignment),
            "principle": "Candidate missed a contradiction and accepted an expected-UNSAT fixture.",
        }
    if fixture.expected_status == "sat":
        return {
            "gold_assignment": dict(fixture.gold_assignment),
            "principle": "Reference SAT witness shows the candidate overconstrained a satisfiable fixture.",
        }
    return {"expected_status": fixture.expected_status, "solver_status": solver_status}


def _aggregate_scores(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    correct = sum(1 for row in rows if row["matches_expected"])
    return {
        "validity_rate": correct / total if total else 0.0,
        "false_accepts": sum(1 for row in rows if row["false_accept"]),
        "counterexamples_found": sum(1 for row in rows if row["counterexample"]),
    }


def _fixture_to_dict(fixture: SolverFixture) -> JsonDict:
    return {
        "fixture_id": fixture.fixture_id,
        "family": fixture.family,
        "natural_language": fixture.natural_language,
        "expected_status": fixture.expected_status,
        "gold_assignment": fixture.gold_assignment,
        "reference_encoding": fixture.reference_encoding,
        "counterexamples": fixture_counterexample_rows(fixture),
    }


def _honest_verdict(ready: bool, checker_available: bool) -> str:
    if not checker_available:
        return "blocked_z3_unavailable: solver_fixture_ready false because exact checker is unavailable"
    if ready:
        return "complete: solver_fixture_ready true for Exp 5274 deterministic gated retry"
    return "complete: solver_fixture_ready false because deterministic fixture checks did not all pass"


def _ready_principle(
    *,
    ready: bool,
    checker_available: bool,
    schema_checks_passed: bool,
    baseline_validity: float,
    coverage: float,
) -> str:
    if ready:
        return (
            "ready=true because Z3 is available, malformed schemas reject before "
            "solver scoring, reference baseline validity is 1.0, counterexample "
            "coverage is 1.0, empty/shuffled controls are recorded, and checksums exist."
        )
    return (
        "ready=false because checker_available=%s, schema_checks_passed=%s, "
        "baseline_validity=%s, counterexample_coverage=%s."
        % (checker_available, schema_checks_passed, baseline_validity, coverage)
    )


def _rate_ok(value: Any) -> bool:
    return isinstance(value, int | float) and 0.0 <= float(value) <= 1.0


def _checker_available(z3_module: Any) -> bool:
    return z3_module is not None and hasattr(z3_module, "Solver") and hasattr(z3_module, "Int")


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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
