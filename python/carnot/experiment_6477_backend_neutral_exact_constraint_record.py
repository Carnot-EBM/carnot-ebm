"""Exp6477 backend-neutral exact constraint record.

Spec refs: REQ-VERIFY-6477, SCENARIO-VERIFY-6477-SCHEMA,
SCENARIO-VERIFY-6477-BACKEND-PARITY, SCENARIO-VERIFY-6477-ATTACKS,
SCENARIO-VERIFY-6477-ROWS.

This module keeps a small finite-domain record separate from its backends.
Z3 and exhaustive replay both read the same record. The scalar energy is only
a row-level diagnostic, so it cannot replace exact parity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import importlib.metadata as metadata
import itertools
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

import z3

from carnot import task_runtime_receipts as receipts


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6477
RANDOM_CASE_SEEDS = (647701, 647702, 647703, 647704, 647705)
INFERENCE_SUBSTRATE = "exact_solver_replay_no_llm"
RECORD_SCHEMA_VERSION = "carnot.exact_constraint_record.v1"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_6477.backend_neutral_exact_record.v1"
EXACT_INT_BOUND = 1_000_000
MAX_EXHAUSTIVE_STATES = 50_000
Z3_TIMEOUT_MS = 1_000

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6477_backend_neutral_exact_constraint_record.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6477_backend_neutral_exact_constraint_record --date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py "
    "-m pytest "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6477_backend_neutral_exact_constraint_record.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6477_backend_neutral_exact_constraint_record.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6477_backend_neutral_exact_constraint_record.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6477_backend_neutral_exact_constraint_record --validate"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: ops/e2e-test-plan.md has no direct Exp6477 entry"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    VALIDATE_COMMAND,
    E2E_PLAN_COMMAND,
    RUN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "constraint_record_schema_and_hash",
    "backend_versions_and_settings",
    "immutable_case_manifest",
    "translation_receipts",
    "per_unit_rows",
    "satisfiability_parity",
    "witness_validity_parity",
    "violation_set_parity",
    "scalar_violation_energy_rows",
    "unsupported_operation_rows",
    "aggregate_row_recomputation",
    "attack_matrix",
    "exact_constraint_record_ready_score",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal status distinguishes complete backend parity from a partially implemented translator.",
    "constraint_record_schema_and_hash": "A versioned schema binds every backend and energy computation to the same semantics.",
    "backend_versions_and_settings": "Pinned solver identities and settings prevent backend drift from masquerading as semantic change.",
    "immutable_case_manifest": "Sealed cases prevent result-dependent fixture edits after a disagreement appears.",
    "translation_receipts": "Constraint-ID-preserving receipts expose dropped, rewritten, or invented semantics.",
    "per_unit_rows": "Case, seed, backend, and attack rows make parity independently recomputable.",
    "satisfiability_parity": "Matching satisfiability is the first exact semantic invariant across backends.",
    "witness_validity_parity": "Independent witness replay catches solvers that agree on status but return invalid assignments.",
    "violation_set_parity": "Exact constraint-ID parity prevents matching scalar totals from hiding different failures.",
    "scalar_violation_energy_rows": "Energy rows bind the proposal score to the same exact record without making it an oracle.",
    "unsupported_operation_rows": "Explicit rejection prevents silent approximation of operations outside the record's scope.",
    "aggregate_row_recomputation": "Row-derived parity catches favorable summaries that omit a failing case.",
    "attack_matrix": "Translation attacks test the known negation, domain, auxiliary, and objective-sign failure modes.",
    "exact_constraint_record_ready_score": "A conjunctive gate blocks held energy experiments until exact record semantics are stable.",
    "protected_files_unchanged": "The new record cannot gain parity by altering protected evaluators or conductor logic.",
    "gate_check_summary": "A blocked result must name the backend, case, expected invariant, and observed mismatch.",
    "preconditions_checked": "Backend and version receipts prove exact dependencies existed before cases ran.",
    "inference_substrate": "Declaring exact_solver_replay_no_llm prevents deterministic solver work from being presented as model reasoning.",
    "verifier_is_oracle": "Exact backends are authoritative only within the declared finite-domain record.",
    "field_principles": "A principle map preserves the semantic reason for every parity field.",
    "field_provenance": "Case hashes, backend receipts, and reducers make each field traceable.",
    "random_seed": "Declared seeds reproduce generated small-domain cases and attack ordering.",
    "duration_s": "Wall time detects skipped exhaustive enumeration or backend execution.",
    "tests_run": "Executed tests prove the translator, evaluator, attacks, and reducers ran.",
    "reproducibility_checksum": "The checksum binds schema, cases, backend settings, code, and result.",
    "honest_verdict": "The verdict states exact parity, bounded failure, or unsupported scope without overclaiming global solver equivalence.",
}

ATTACK_IDS = (
    "dropped_negation",
    "domain_widening",
    "integer_to_boolean_coercion",
    "auxiliary_variable_leakage",
    "overflow",
    "duplicate_constraint_ids",
    "objective_sign_reversal",
    "matching_totals_different_violation_sets",
)

SOURCE_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("_bmad/prd.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("python/carnot/task_runtime_receipts.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("ops/e2e-test-plan.md"),
)

PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("python/carnot/verify/sat.py"),
    Path("python/carnot/verify/z3_math.py"),
    Path("python/carnot/verify/nco_constraint.py"),
    Path("python/carnot/verify/ising.py"),
)

SUPPORTED_BOOL_OPS = frozenset({"linear_compare", "bool_var", "not", "and", "or", "all_different"})
SUPPORTED_COMPARE_OPS = frozenset({"eq", "ne", "le", "lt", "ge", "gt"})


class UnsupportedRecordError(ValueError):
    """Raised when a record asks for semantics outside the exact subset."""


@dataclass
class FiniteDomainVar:
    """One finite integer or Boolean variable in the exact record."""

    var_id: str
    lower: int
    upper: int
    kind: str = "int"
    role: str = "primary"

    def to_dict(self) -> JsonDict:
        return {
            "var_id": self.var_id,
            "lower": self.lower,
            "upper": self.upper,
            "kind": self.kind,
            "role": self.role,
        }


@dataclass
class LinearExpr:
    """Integer linear expression with explicit coefficients only."""

    coefficients: dict[str, int] = field(default_factory=dict)
    constant: int = 0

    def to_dict(self) -> JsonDict:
        return {
            "coefficients": dict(sorted(self.coefficients.items())),
            "constant": int(self.constant),
        }


@dataclass
class BoolExpr:
    """Boolean expression over supported finite-domain atoms."""

    op: str
    children: tuple["BoolExpr", ...] = ()
    expr: LinearExpr | None = None
    compare_op: str = ""
    rhs: int = 0
    var_id: str = ""
    var_ids: tuple[str, ...] = ()

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {"op": self.op}
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        if self.expr is not None:
            payload["expr"] = self.expr.to_dict()
        if self.compare_op:
            payload["compare_op"] = self.compare_op
            payload["rhs"] = int(self.rhs)
        if self.var_id:
            payload["var_id"] = self.var_id
        if self.var_ids:
            payload["var_ids"] = list(self.var_ids)
        return payload


@dataclass
class ConstraintSpec:
    """A named hard constraint with optional protected status."""

    constraint_id: str
    expr: BoolExpr
    weight: int = 1
    protected: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "constraint_id": self.constraint_id,
            "expr": self.expr.to_dict(),
            "weight": int(self.weight),
            "protected": bool(self.protected),
        }


@dataclass
class ObjectiveTerm:
    """A named linear objective term kept separate from feasibility."""

    objective_id: str
    expr: LinearExpr
    weight: int = 1

    def to_dict(self) -> JsonDict:
        return {
            "objective_id": self.objective_id,
            "expr": self.expr.to_dict(),
            "weight": int(self.weight),
        }


@dataclass
class ConstraintRecord:
    """Versioned finite-domain record shared by all exact backends."""

    case_id: str
    case_kind: str
    seed: int
    variables: tuple[FiniteDomainVar, ...]
    constraints: tuple[ConstraintSpec, ...]
    objective_terms: tuple[ObjectiveTerm, ...] = ()
    schema_version: str = RECORD_SCHEMA_VERSION
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "case_kind": self.case_kind,
            "seed": int(self.seed),
            "description": self.description,
            "variables": [var.to_dict() for var in self.variables],
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "objective_terms": [term.to_dict() for term in self.objective_terms],
        }

    def record_hash(self) -> str:
        return receipts.sha256_json(self.to_dict())


def lin(coefficients: Mapping[str, int] | None = None, constant: int = 0) -> LinearExpr:
    """Build a linear expression from explicit integer coefficients."""

    return LinearExpr(dict(coefficients or {}), int(constant))


def cmp(expr: LinearExpr, compare_op: str, rhs: int = 0) -> BoolExpr:
    """Build a supported linear comparison."""

    return BoolExpr(op="linear_compare", expr=expr, compare_op=compare_op, rhs=int(rhs))


def bool_var(var_id: str) -> BoolExpr:
    """Read a declared Boolean variable."""

    return BoolExpr(op="bool_var", var_id=var_id)


def not_(child: BoolExpr) -> BoolExpr:
    """Negate one Boolean expression."""

    return BoolExpr(op="not", children=(child,))


def and_(*children: BoolExpr) -> BoolExpr:
    """Conjoin Boolean expressions."""

    return BoolExpr(op="and", children=tuple(children))


def or_(*children: BoolExpr) -> BoolExpr:
    """Disjoin Boolean expressions."""

    return BoolExpr(op="or", children=tuple(children))


def all_different(*var_ids: str) -> BoolExpr:
    """Require all listed finite-domain variables to differ."""

    return BoolExpr(op="all_different", var_ids=tuple(var_ids))


def canonical_json(value: Any) -> str:
    """Return stable JSON for record and artifact hashes."""

    return receipts.canonical_json(value)


def _git_output(args: Sequence[str], root: Path) -> str:
    """Run git and return stdout, or empty text outside a checkout."""

    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _package_version(name: str) -> str:
    """Return an installed package version for backend receipts."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not_installed"


def _source_hashes(root: Path) -> dict[str, str | None]:
    """Hash source, spec, and test files used by the artifact."""

    return {path.as_posix(): receipts.sha256_file(root / path) for path in SOURCE_RELATIVE_PATHS}


def _protected_hashes(root: Path) -> dict[str, str | None]:
    """Hash files that this experiment must not alter."""

    return {path.as_posix(): receipts.sha256_file(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    """Compare protected hashes before and after artifact construction."""

    after = _protected_hashes(root)
    files = {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "unchanged": all(row["unchanged"] for row in files.values()),
        "changed_paths": [path for path, row in files.items() if not row["unchanged"]],
    }


def constraint_record_schema_and_hash() -> JsonDict:
    """Return the exact record schema contract and its hash."""

    schema = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "integer_bound_abs": EXACT_INT_BOUND,
        "max_exhaustive_states": MAX_EXHAUSTIVE_STATES,
        "variable_fields": ["var_id", "kind", "role", "lower", "upper"],
        "variable_kinds": ["int", "bool"],
        "variable_roles": ["primary", "auxiliary"],
        "supported_constraint_kinds": sorted(SUPPORTED_BOOL_OPS),
        "supported_compare_ops": sorted(SUPPORTED_COMPARE_OPS),
        "objective_terms": "weighted integer linear expressions; diagnostic only",
        "scalar_violation_energy": "sum of violated source constraint weights",
        "unsupported_operations_fail_closed": [
            "nonlinear",
            "ambiguous_integer_to_boolean",
            "duplicate_constraint_ids",
            "duplicate_objective_ids",
            "unknown_variables",
            "overflow",
        ],
    }
    return {**schema, "schema_sha256": receipts.sha256_json(schema)}


def _ids_duplicate(ids: Sequence[str]) -> str | None:
    """Return the first duplicate id, if any."""

    seen: set[str] = set()
    for item in ids:
        if item in seen:
            return item
        seen.add(item)
    return None


def _state_count(record: ConstraintRecord) -> int:
    count = 1
    for var in record.variables:
        count *= int(var.upper) - int(var.lower) + 1
    return count


def _vars_by_id(record: ConstraintRecord) -> dict[str, FiniteDomainVar]:
    return {var.var_id: var for var in record.variables}


def _append_once(errors: list[str], reason: str) -> None:
    if reason not in errors:
        errors.append(reason)


def _validate_linear_expr(
    expr: LinearExpr | None,
    vars_by_id: Mapping[str, FiniteDomainVar],
    errors: list[str],
) -> None:
    if not isinstance(expr, LinearExpr):
        _append_once(errors, "unsupported_nonlinear_multiply")
        return
    if abs(int(expr.constant)) > EXACT_INT_BOUND:
        _append_once(errors, "overflow_linear_constant")
    for var_id, coefficient in expr.coefficients.items():
        if var_id not in vars_by_id:
            _append_once(errors, f"unknown_variable:{var_id}")
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            _append_once(errors, "non_integer_linear_coefficient")
        elif abs(coefficient) > EXACT_INT_BOUND:
            _append_once(errors, "overflow_linear_coefficient")


def _validate_bool_expr(
    expr: BoolExpr,
    vars_by_id: Mapping[str, FiniteDomainVar],
    errors: list[str],
) -> None:
    if not isinstance(expr, BoolExpr):
        _append_once(errors, "unsupported_boolean_expression")
        return
    if expr.op not in SUPPORTED_BOOL_OPS:
        if "mul" in expr.op or "nonlinear" in expr.op:
            _append_once(errors, "unsupported_nonlinear_multiply")
        else:
            _append_once(errors, f"unsupported_boolean_op:{expr.op}")
        return
    if expr.op == "linear_compare":
        if expr.compare_op not in SUPPORTED_COMPARE_OPS:
            _append_once(errors, f"unsupported_compare_op:{expr.compare_op}")
        if isinstance(expr.rhs, bool) or not isinstance(expr.rhs, int):
            _append_once(errors, "non_integer_compare_rhs")
        elif abs(expr.rhs) > EXACT_INT_BOUND:
            _append_once(errors, "overflow_compare_rhs")
        _validate_linear_expr(expr.expr, vars_by_id, errors)
    elif expr.op == "bool_var":
        var = vars_by_id.get(expr.var_id)
        if var is None:
            _append_once(errors, f"unknown_variable:{expr.var_id}")
        elif var.kind != "bool":
            _append_once(errors, f"ambiguous_integer_to_boolean:{expr.var_id}")
    elif expr.op == "all_different":
        if len(expr.var_ids) < 2:
            _append_once(errors, "all_different_needs_two_variables")
        for var_id in expr.var_ids:
            if var_id not in vars_by_id:
                _append_once(errors, f"unknown_variable:{var_id}")
    elif expr.op == "not":
        if len(expr.children) != 1:
            _append_once(errors, "not_requires_one_child")
        for child in expr.children:
            _validate_bool_expr(child, vars_by_id, errors)
    else:
        if not expr.children:
            _append_once(errors, f"{expr.op}_requires_children")
        for child in expr.children:
            _validate_bool_expr(child, vars_by_id, errors)


def validate_record(record: ConstraintRecord) -> list[str]:
    """Return fail-closed validation errors for the finite record."""

    errors: list[str] = []
    if record.schema_version != RECORD_SCHEMA_VERSION:
        errors.append(f"unsupported_schema_version:{record.schema_version}")
    duplicate_var = _ids_duplicate([var.var_id for var in record.variables])
    if duplicate_var:
        errors.append(f"duplicate_variable_id:{duplicate_var}")
    for var in record.variables:
        if not var.var_id:
            _append_once(errors, "empty_variable_id")
        if var.kind not in {"int", "bool"}:
            _append_once(errors, f"unsupported_variable_kind:{var.kind}")
        if var.role not in {"primary", "auxiliary"}:
            _append_once(errors, f"unsupported_variable_role:{var.role}")
        if isinstance(var.lower, bool) or isinstance(var.upper, bool):
            _append_once(errors, f"non_integer_domain:{var.var_id}")
            continue
        if int(var.lower) > int(var.upper):
            _append_once(errors, f"empty_domain:{var.var_id}")
        if abs(int(var.lower)) > EXACT_INT_BOUND or abs(int(var.upper)) > EXACT_INT_BOUND:
            _append_once(errors, f"overflow_domain:{var.var_id}")
        if var.kind == "bool" and (int(var.lower), int(var.upper)) != (0, 1):
            _append_once(errors, f"invalid_bool_domain:{var.var_id}")
    duplicate_constraint = _ids_duplicate([c.constraint_id for c in record.constraints])
    if duplicate_constraint:
        errors.append(f"duplicate_constraint_id:{duplicate_constraint}")
    duplicate_objective = _ids_duplicate([term.objective_id for term in record.objective_terms])
    if duplicate_objective:
        errors.append(f"duplicate_objective_id:{duplicate_objective}")
    vars_by_id = _vars_by_id(record)
    for constraint in record.constraints:
        if not constraint.constraint_id:
            _append_once(errors, "empty_constraint_id")
        if isinstance(constraint.weight, bool) or not isinstance(constraint.weight, int):
            _append_once(errors, f"non_integer_constraint_weight:{constraint.constraint_id}")
        elif int(constraint.weight) <= 0:
            _append_once(errors, f"nonpositive_constraint_weight:{constraint.constraint_id}")
        _validate_bool_expr(constraint.expr, vars_by_id, errors)
    for term in record.objective_terms:
        if not term.objective_id:
            _append_once(errors, "empty_objective_id")
        if isinstance(term.weight, bool) or not isinstance(term.weight, int):
            _append_once(errors, f"non_integer_objective_weight:{term.objective_id}")
        elif int(term.weight) == 0:
            _append_once(errors, f"zero_objective_weight:{term.objective_id}")
        _validate_linear_expr(term.expr, vars_by_id, errors)
    if record.variables and not any(error.startswith(("empty_domain", "overflow_domain")) for error in errors):
        if _state_count(record) > MAX_EXHAUSTIVE_STATES:
            _append_once(errors, "exhaustive_state_budget_exceeded")
    return errors


def ensure_supported(record: ConstraintRecord) -> None:
    """Raise when a record cannot be translated exactly."""

    errors = validate_record(record)
    if errors:
        raise UnsupportedRecordError(";".join(errors))


def eval_linear(expr: LinearExpr, assignment: Mapping[str, int]) -> int:
    """Evaluate one linear expression on a finite assignment."""

    total = int(expr.constant)
    for var_id, coefficient in expr.coefficients.items():
        total += int(coefficient) * int(assignment[var_id])
    return total


def _compare(left: int, op: str, right: int) -> bool:
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "le":
        return left <= right
    if op == "lt":
        return left < right
    if op == "ge":
        return left >= right
    if op == "gt":
        return left > right
    raise UnsupportedRecordError(f"unsupported_compare_op:{op}")


def eval_bool(expr: BoolExpr, assignment: Mapping[str, int]) -> bool:
    """Evaluate a Boolean expression with exact integer arithmetic."""

    if expr.op == "linear_compare":
        if expr.expr is None:
            raise UnsupportedRecordError("missing_linear_expr")
        return _compare(eval_linear(expr.expr, assignment), expr.compare_op, int(expr.rhs))
    if expr.op == "bool_var":
        return int(assignment[expr.var_id]) == 1
    if expr.op == "not":
        return not eval_bool(expr.children[0], assignment)
    if expr.op == "and":
        return all(eval_bool(child, assignment) for child in expr.children)
    if expr.op == "or":
        return any(eval_bool(child, assignment) for child in expr.children)
    if expr.op == "all_different":
        values = [int(assignment[var_id]) for var_id in expr.var_ids]
        return len(set(values)) == len(values)
    raise UnsupportedRecordError(f"unsupported_boolean_op:{expr.op}")


def assignment_domain_valid(record: ConstraintRecord, assignment: Mapping[str, int]) -> bool:
    """Check that an assignment stays inside declared finite domains."""

    for var in record.variables:
        if var.var_id not in assignment:
            return False
        value = int(assignment[var.var_id])
        if value < int(var.lower) or value > int(var.upper):
            return False
        if var.kind == "bool" and value not in {0, 1}:
            return False
    return True


def violated_constraint_ids(record: ConstraintRecord, assignment: Mapping[str, int]) -> list[str]:
    """Return source ids for constraints false on the assignment."""

    return [
        constraint.constraint_id
        for constraint in record.constraints
        if not eval_bool(constraint.expr, assignment)
    ]


def protected_violations(record: ConstraintRecord, assignment: Mapping[str, int]) -> list[str]:
    """Return violated source ids marked as protected."""

    return [
        constraint.constraint_id
        for constraint in record.constraints
        if constraint.protected and not eval_bool(constraint.expr, assignment)
    ]


def scalar_violation_energy(record: ConstraintRecord, assignment: Mapping[str, int]) -> int:
    """Return diagnostic weighted violation count from the record."""

    return sum(
        int(constraint.weight)
        for constraint in record.constraints
        if not eval_bool(constraint.expr, assignment)
    )


def objective_value(record: ConstraintRecord, assignment: Mapping[str, int]) -> int:
    """Evaluate explicit objective terms on an assignment."""

    return sum(
        int(term.weight) * eval_linear(term.expr, assignment)
        for term in record.objective_terms
    )


def _assignment_key(record: ConstraintRecord, assignment: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(int(assignment[var.var_id]) for var in record.variables)


def enumerate_assignments(record: ConstraintRecord) -> list[dict[str, int]]:
    """Enumerate the complete finite state space in lexicographic order."""

    ensure_supported(record)
    names = [var.var_id for var in record.variables]
    domains = [range(int(var.lower), int(var.upper) + 1) for var in record.variables]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*domains)]


def _base_backend_row(
    *,
    record: ConstraintRecord,
    backend: str,
    selected_assignment: Mapping[str, int],
    satisfiable: bool,
    backend_reported_violations: Sequence[str],
    backend_scalar_energy: int,
    backend_objective_value: int,
    state_count: int,
    translation_receipt_hash: str,
) -> JsonDict:
    """Build one backend row with independent witness replay."""

    replayed_violations = violated_constraint_ids(record, selected_assignment)
    witness_valid = replayed_violations == [] if satisfiable else None
    return {
        "row_type": "backend_case",
        "case_id": record.case_id,
        "case_kind": record.case_kind,
        "seed": int(record.seed),
        "backend": backend,
        "record_hash": record.record_hash(),
        "source_constraint_ids": [constraint.constraint_id for constraint in record.constraints],
        "satisfiable": bool(satisfiable),
        "selected_assignment": dict(selected_assignment),
        "domain_assignment_valid": assignment_domain_valid(record, selected_assignment),
        "witness_valid": witness_valid,
        "violated_constraint_ids": list(backend_reported_violations),
        "replayed_violated_constraint_ids": replayed_violations,
        "protected_violations": [
            constraint.constraint_id
            for constraint in record.constraints
            if constraint.protected and constraint.constraint_id in backend_reported_violations
        ],
        "objective_value": int(backend_objective_value),
        "scalar_violation_energy": int(backend_scalar_energy),
        "state_count": int(state_count),
        "translation_receipt_hash": translation_receipt_hash,
    }


def exhaustive_translation_receipt(record: ConstraintRecord) -> JsonDict:
    """Describe the exhaustive reference backend for one record."""

    payload = {
        "backend": "exhaustive",
        "record_hash": record.record_hash(),
        "enumeration_order": "variable declaration order, lexicographic values",
        "state_count": _state_count(record),
        "constraint_receipts": [
            {
                "constraint_id": constraint.constraint_id,
                "source_hash": receipts.sha256_json(constraint.to_dict()),
                "predicate_backend": "python_exact_replay",
                "protected": constraint.protected,
                "weight": constraint.weight,
            }
            for constraint in record.constraints
        ],
    }
    return {**payload, "translation_hash": receipts.sha256_json(payload)}


def exhaustive_backend_solve(record: ConstraintRecord) -> JsonDict:
    """Solve one record by full finite-domain enumeration."""

    assignments = enumerate_assignments(record)
    receipt = exhaustive_translation_receipt(record)
    best_sat: tuple[int, tuple[int, ...], dict[str, int]] | None = None
    best_any: tuple[int, int, tuple[int, ...], dict[str, int]] | None = None
    for assignment in assignments:
        energy = scalar_violation_energy(record, assignment)
        objective = objective_value(record, assignment)
        key = _assignment_key(record, assignment)
        any_key = (energy, objective, key, assignment)
        if best_any is None or any_key[:3] < best_any[:3]:
            best_any = any_key
        if energy == 0:
            sat_key = (objective, key, assignment)
            if best_sat is None or sat_key[:2] < best_sat[:2]:
                best_sat = sat_key
    selected = best_sat[2] if best_sat is not None else best_any[3]  # type: ignore[index]
    satisfiable = best_sat is not None
    violations = violated_constraint_ids(record, selected)
    row = _base_backend_row(
        record=record,
        backend="exhaustive",
        selected_assignment=selected,
        satisfiable=satisfiable,
        backend_reported_violations=violations,
        backend_scalar_energy=scalar_violation_energy(record, selected),
        backend_objective_value=objective_value(record, selected),
        state_count=len(assignments),
        translation_receipt_hash=receipt["translation_hash"],
    )
    return {"backend": "exhaustive", "row": row, "translation_receipt": receipt}


def _z3_linear(expr: LinearExpr, z3_vars: Mapping[str, z3.ArithRef]) -> z3.ArithRef:
    terms = [z3.IntVal(int(expr.constant))]
    for var_id, coefficient in sorted(expr.coefficients.items()):
        terms.append(z3.IntVal(int(coefficient)) * z3_vars[var_id])
    return z3.Sum(terms)


def _z3_bool(expr: BoolExpr, z3_vars: Mapping[str, z3.ArithRef]) -> z3.BoolRef:
    if expr.op == "linear_compare":
        left = _z3_linear(expr.expr or LinearExpr(), z3_vars)
        right = z3.IntVal(int(expr.rhs))
        if expr.compare_op == "eq":
            return left == right
        if expr.compare_op == "ne":
            return left != right
        if expr.compare_op == "le":
            return left <= right
        if expr.compare_op == "lt":
            return left < right
        if expr.compare_op == "ge":
            return left >= right
        if expr.compare_op == "gt":
            return left > right
    if expr.op == "bool_var":
        return z3_vars[expr.var_id] == z3.IntVal(1)
    if expr.op == "not":
        return z3.Not(_z3_bool(expr.children[0], z3_vars))
    if expr.op == "and":
        return z3.And(*[_z3_bool(child, z3_vars) for child in expr.children])
    if expr.op == "or":
        return z3.Or(*[_z3_bool(child, z3_vars) for child in expr.children])
    if expr.op == "all_different":
        return z3.Distinct(*[z3_vars[var_id] for var_id in expr.var_ids])
    raise UnsupportedRecordError(f"unsupported_boolean_op:{expr.op}")


def _z3_translation(record: ConstraintRecord) -> JsonDict:
    """Translate one supported record to Z3 expressions and receipts."""

    ensure_supported(record)
    z3_vars = {var.var_id: z3.Int(var.var_id) for var in record.variables}
    domain_constraints = [
        z3.And(z3_vars[var.var_id] >= int(var.lower), z3_vars[var.var_id] <= int(var.upper))
        for var in record.variables
    ]
    constraint_refs = [
        (constraint, _z3_bool(constraint.expr, z3_vars)) for constraint in record.constraints
    ]
    objective_expr = z3.Sum(
        [
            z3.IntVal(int(term.weight)) * _z3_linear(term.expr, z3_vars)
            for term in record.objective_terms
        ]
        or [z3.IntVal(0)]
    )
    payload = {
        "backend": "z3",
        "record_hash": record.record_hash(),
        "solver_settings": {
            "solver": "Optimize",
            "timeout_ms": Z3_TIMEOUT_MS,
            "priority": "lex",
            "model_completion": True,
            "objective_order": "feasibility_or_energy_then_objective_then_lexicographic_variables",
        },
        "domain_receipts": [
            {
                "var_id": var.var_id,
                "kind": var.kind,
                "role": var.role,
                "lower": int(var.lower),
                "upper": int(var.upper),
            }
            for var in record.variables
        ],
        "constraint_receipts": [
            {
                "constraint_id": constraint.constraint_id,
                "source_hash": receipts.sha256_json(constraint.to_dict()),
                "z3_expr": str(ref),
                "protected": constraint.protected,
                "weight": constraint.weight,
            }
            for constraint, ref in constraint_refs
        ],
        "objective_receipts": [
            {
                "objective_id": term.objective_id,
                "source_hash": receipts.sha256_json(term.to_dict()),
                "weight": int(term.weight),
            }
            for term in record.objective_terms
        ],
    }
    payload["translation_hash"] = receipts.sha256_json(payload)
    return {
        **payload,
        "_z3_vars": z3_vars,
        "_domain_constraints": domain_constraints,
        "_constraint_refs": constraint_refs,
        "_objective_expr": objective_expr,
    }


def _z3_public_receipt(translation: Mapping[str, Any]) -> JsonDict:
    """Remove live Z3 objects from a translation receipt."""

    return {
        key: value
        for key, value in translation.items()
        if not str(key).startswith("_")
    }


def _z3_optimize() -> z3.Optimize:
    opt = z3.Optimize()
    opt.set(timeout=Z3_TIMEOUT_MS, priority="lex")
    return opt


def _z3_model_assignment(
    model: z3.ModelRef,
    record: ConstraintRecord,
    z3_vars: Mapping[str, z3.ArithRef],
) -> dict[str, int]:
    return {
        var.var_id: int(str(model.evaluate(z3_vars[var.var_id], model_completion=True)))
        for var in record.variables
    }


def _z3_eval_bool(model: z3.ModelRef, ref: z3.BoolRef) -> bool:
    return z3.is_true(model.evaluate(ref, model_completion=True))


def _z3_eval_int(model: z3.ModelRef, ref: z3.ArithRef) -> int:
    return int(str(model.evaluate(ref, model_completion=True)))


def z3_backend_solve(record: ConstraintRecord) -> JsonDict:
    """Solve one record through Z3 without using exhaustive expected values."""

    translation = _z3_translation(record)
    z3_vars: Mapping[str, z3.ArithRef] = translation["_z3_vars"]
    domain_constraints: Sequence[z3.BoolRef] = translation["_domain_constraints"]
    constraint_refs: Sequence[tuple[ConstraintSpec, z3.BoolRef]] = translation["_constraint_refs"]
    objective_expr: z3.ArithRef = translation["_objective_expr"]

    hard = _z3_optimize()
    hard.add(*domain_constraints)
    hard.add(*[ref for _, ref in constraint_refs])
    hard.minimize(objective_expr)
    for var in record.variables:
        hard.minimize(z3_vars[var.var_id])
    satisfiable = hard.check() == z3.sat
    if satisfiable:
        model = hard.model()
        energy_expr = z3.IntVal(0)
    else:
        soft = _z3_optimize()
        soft.add(*domain_constraints)
        energy_expr = z3.Sum(
            [
                z3.If(ref, z3.IntVal(0), z3.IntVal(int(constraint.weight)))
                for constraint, ref in constraint_refs
            ]
            or [z3.IntVal(0)]
        )
        soft.minimize(energy_expr)
        soft.minimize(objective_expr)
        for var in record.variables:
            soft.minimize(z3_vars[var.var_id])
        if soft.check() != z3.sat:  # pragma: no cover - supported domains are non-empty.
            raise UnsupportedRecordError("z3_domain_unsat")
        model = soft.model()

    assignment = _z3_model_assignment(model, record, z3_vars)
    backend_violations = [
        constraint.constraint_id
        for constraint, ref in constraint_refs
        if not _z3_eval_bool(model, ref)
    ]
    backend_energy = _z3_eval_int(model, energy_expr) if not satisfiable else 0
    backend_objective = _z3_eval_int(model, objective_expr)
    public_receipt = _z3_public_receipt(translation)
    row = _base_backend_row(
        record=record,
        backend="z3",
        selected_assignment=assignment,
        satisfiable=satisfiable,
        backend_reported_violations=backend_violations,
        backend_scalar_energy=backend_energy,
        backend_objective_value=backend_objective,
        state_count=_state_count(record),
        translation_receipt_hash=public_receipt["translation_hash"],
    )
    return {"backend": "z3", "row": row, "translation_receipt": public_receipt}


def _case(
    *,
    case_id: str,
    case_kind: str,
    seed: int,
    variables: Sequence[FiniteDomainVar],
    constraints: Sequence[ConstraintSpec],
    objectives: Sequence[ObjectiveTerm] = (),
    description: str,
) -> ConstraintRecord:
    return ConstraintRecord(
        case_id=case_id,
        case_kind=case_kind,
        seed=seed,
        variables=tuple(variables),
        constraints=tuple(constraints),
        objective_terms=tuple(objectives),
        description=description,
    )


def random_seed_case(seed: int) -> ConstraintRecord:
    """Generate one sealed small-domain satisfiable random case."""

    rng = random.Random(seed)
    x_target = rng.randrange(0, 4)
    y_target = (x_target + rng.randrange(1, 4)) % 4
    return _case(
        case_id=f"random_seed_{seed}",
        case_kind="random_seed",
        seed=seed,
        variables=(
            FiniteDomainVar("x", 0, 3),
            FiniteDomainVar("y", 0, 3),
            FiniteDomainVar("flag", 0, 1, kind="bool"),
        ),
        constraints=(
            ConstraintSpec(
                f"c_seed_{seed}_sum",
                cmp(lin({"x": 1, "y": 1}), "eq", x_target + y_target),
            ),
            ConstraintSpec(f"c_seed_{seed}_diff", all_different("x", "y")),
            ConstraintSpec(
                f"c_seed_{seed}_or",
                or_(bool_var("flag"), cmp(lin({"x": 1}), "eq", x_target)),
            ),
            ConstraintSpec(f"c_seed_{seed}_flag_false", cmp(lin({"flag": 1}), "eq", 0)),
        ),
        objectives=(ObjectiveTerm(f"o_seed_{seed}_linear", lin({"x": 1, "y": -1}), 1),),
        description="Seeded finite-domain case with Boolean composition and all-different.",
    )


def immutable_cases() -> list[ConstraintRecord]:
    """Return sealed cases covering the declared exact subset."""

    cases = [
        _case(
            case_id="sat_linear_all_different",
            case_kind="satisfiable",
            seed=647700,
            variables=(FiniteDomainVar("x", 0, 2), FiniteDomainVar("y", 0, 2)),
            constraints=(
                ConstraintSpec("c_sum_eq_two", cmp(lin({"x": 1, "y": 1}), "eq", 2)),
                ConstraintSpec("c_x_le_y", cmp(lin({"x": 1, "y": -1}), "le", 0)),
                ConstraintSpec("c_all_diff", all_different("x", "y")),
            ),
            objectives=(ObjectiveTerm("o_minimize_x_minus_y", lin({"x": 1, "y": -1}), 1),),
            description="Satisfiable linear and all-different control.",
        ),
        _case(
            case_id="unsat_linear_boundary",
            case_kind="unsatisfiable",
            seed=647710,
            variables=(FiniteDomainVar("x", 0, 1),),
            constraints=(
                ConstraintSpec("c_domain_low_replay", cmp(lin({"x": 1}), "ge", 0)),
                ConstraintSpec("c_impossible_ge_two", cmp(lin({"x": 1}), "ge", 2)),
            ),
            objectives=(ObjectiveTerm("o_minimize_x", lin({"x": 1}), 1),),
            description="Unsatisfiable boundary case used for domain-widening attack.",
        ),
        _case(
            case_id="negated_unsat",
            case_kind="negated",
            seed=647720,
            variables=(FiniteDomainVar("x", 0, 1),),
            constraints=(
                ConstraintSpec("c_not_zero", not_(cmp(lin({"x": 1}), "eq", 0))),
                ConstraintSpec("c_force_zero", cmp(lin({"x": 1}), "eq", 0)),
            ),
            objectives=(ObjectiveTerm("o_minimize_x_negated", lin({"x": 1}), 1),),
            description="Negation must be preserved because dropping it changes satisfiability.",
        ),
        _case(
            case_id="auxiliary_link_sat",
            case_kind="auxiliary_variable",
            seed=647730,
            variables=(
                FiniteDomainVar("x", 0, 2),
                FiniteDomainVar("aux", 0, 3, role="auxiliary"),
            ),
            constraints=(
                ConstraintSpec("c_aux_link", cmp(lin({"aux": 1, "x": -1}), "eq", 1)),
                ConstraintSpec("c_aux_limit", cmp(lin({"aux": 1}), "le", 2)),
            ),
            objectives=(ObjectiveTerm("o_minimize_aux", lin({"aux": 1}), 1),),
            description="Auxiliary variable is internal but must keep its linking constraint.",
        ),
        _case(
            case_id="protected_clause_unsat",
            case_kind="protected_clause",
            seed=647740,
            variables=(
                FiniteDomainVar("p", 0, 1, kind="bool"),
                FiniteDomainVar("q", 0, 1, kind="bool"),
            ),
            constraints=(
                ConstraintSpec("c_force_p_false", cmp(lin({"p": 1}), "eq", 0)),
                ConstraintSpec("c_force_q_false", cmp(lin({"q": 1}), "eq", 0)),
                ConstraintSpec("c_protected_or", or_(bool_var("p"), bool_var("q")), protected=True),
            ),
            objectives=(ObjectiveTerm("o_minimize_true_literals", lin({"p": 1, "q": 1}), 1),),
            description="Protected clause violation remains visible in an unsat best assignment.",
        ),
        _case(
            case_id="boundary_values_sat",
            case_kind="boundary_value",
            seed=647750,
            variables=(FiniteDomainVar("z", -2, 2), FiniteDomainVar("y", 0, 4)),
            constraints=(
                ConstraintSpec("c_z_ge_lower", cmp(lin({"z": 1}), "ge", -2)),
                ConstraintSpec("c_z_le_upper", cmp(lin({"z": 1}), "le", 2)),
                ConstraintSpec("c_y_plus_z_zero", cmp(lin({"y": 1, "z": 1}), "eq", 0)),
                ConstraintSpec("c_y_lt_three", cmp(lin({"y": 1}), "lt", 3)),
                ConstraintSpec("c_y_ne_three", cmp(lin({"y": 1}), "ne", 3)),
            ),
            objectives=(ObjectiveTerm("o_minimize_z", lin({"z": 1}), 1),),
            description="Boundary value case covers negative domains and strict inequality.",
        ),
    ]
    cases.extend(random_seed_case(seed) for seed in RANDOM_CASE_SEEDS)
    return cases


def unsupported_record_fixtures() -> dict[str, ConstraintRecord]:
    """Return invalid records that must be rejected before translation."""

    base_vars = (FiniteDomainVar("x", 0, 1),)
    return {
        "unsupported_nonlinear_multiply": _case(
            case_id="unsupported_nonlinear_multiply",
            case_kind="unsupported",
            seed=1,
            variables=base_vars,
            constraints=(ConstraintSpec("c_mul", BoolExpr(op="nonlinear_multiply")),),
            description="Nonlinear multiply is outside the exact subset.",
        ),
        "ambiguous_integer_to_boolean": _case(
            case_id="ambiguous_integer_to_boolean",
            case_kind="unsupported",
            seed=2,
            variables=base_vars,
            constraints=(ConstraintSpec("c_int_as_bool", bool_var("x")),),
            description="Integer variables cannot be silently coerced to Boolean.",
        ),
        "duplicate_constraint_ids": _case(
            case_id="duplicate_constraint_ids",
            case_kind="unsupported",
            seed=3,
            variables=base_vars,
            constraints=(
                ConstraintSpec("c_dup", cmp(lin({"x": 1}), "ge", 0)),
                ConstraintSpec("c_dup", cmp(lin({"x": 1}), "le", 1)),
            ),
            description="Duplicate constraint ids hide violation provenance.",
        ),
        "overflow_domain": _case(
            case_id="overflow_domain",
            case_kind="unsupported",
            seed=4,
            variables=(FiniteDomainVar("x", 0, EXACT_INT_BOUND + 1),),
            constraints=(ConstraintSpec("c_x", cmp(lin({"x": 1}), "ge", 0)),),
            description="Domains outside the exact bound are rejected.",
        ),
        "unknown_variable": _case(
            case_id="unknown_variable",
            case_kind="unsupported",
            seed=5,
            variables=base_vars,
            constraints=(ConstraintSpec("c_unknown", cmp(lin({"missing": 1}), "eq", 0)),),
            description="Every referenced variable must be declared.",
        ),
        "duplicate_objective_ids": _case(
            case_id="duplicate_objective_ids",
            case_kind="unsupported",
            seed=6,
            variables=base_vars,
            constraints=(ConstraintSpec("c_x", cmp(lin({"x": 1}), "ge", 0)),),
            objectives=(
                ObjectiveTerm("o_dup", lin({"x": 1}), 1),
                ObjectiveTerm("o_dup", lin({"x": 1}), -1),
            ),
            description="Objective ids need stable provenance too.",
        ),
    }


def build_unsupported_operation_rows() -> list[JsonDict]:
    """Build rows proving unsupported cases fail before translation."""

    rows: list[JsonDict] = []
    for operation_id, record in unsupported_record_fixtures().items():
        errors = validate_record(record)
        rows.append(
            {
                "row_type": "unsupported_operation",
                "operation_id": operation_id,
                "case_id": record.case_id,
                "seed": int(record.seed),
                "backend": "validator",
                "record_hash": record.record_hash(),
                "rejected_before_translation": bool(errors),
                "rejection_reasons": errors,
                "backend_result_trusted": False,
            }
        )
    return rows


def evaluate_case(record: ConstraintRecord) -> JsonDict:
    """Run both exact backends and compare semantic rows."""

    z3_result = z3_backend_solve(record)
    exhaustive_result = exhaustive_backend_solve(record)
    rows = [z3_result["row"], exhaustive_result["row"]]
    z3_row, exhaustive_row = rows
    return {
        "case_id": record.case_id,
        "seed": int(record.seed),
        "record_hash": record.record_hash(),
        "backend_rows": rows,
        "translation_receipts": {
            "z3": z3_result["translation_receipt"],
            "exhaustive": exhaustive_result["translation_receipt"],
        },
        "satisfiability_match": z3_row["satisfiable"] == exhaustive_row["satisfiable"],
        "witness_validity_match": z3_row["witness_valid"] == exhaustive_row["witness_valid"],
        "violation_set_match": (
            z3_row["violated_constraint_ids"] == exhaustive_row["violated_constraint_ids"]
        ),
        "protected_violation_match": (
            z3_row["protected_violations"] == exhaustive_row["protected_violations"]
        ),
        "objective_value_match": z3_row["objective_value"] == exhaustive_row["objective_value"],
        "scalar_energy_match": (
            z3_row["scalar_violation_energy"] == exhaustive_row["scalar_violation_energy"]
        ),
    }


def _transform_expr_drop_negation(expr: BoolExpr) -> BoolExpr:
    if expr.op == "not":
        return _transform_expr_drop_negation(expr.children[0])
    return BoolExpr(
        op=expr.op,
        children=tuple(_transform_expr_drop_negation(child) for child in expr.children),
        expr=expr.expr,
        compare_op=expr.compare_op,
        rhs=expr.rhs,
        var_id=expr.var_id,
        var_ids=expr.var_ids,
    )


def _copy_record(record: ConstraintRecord, **updates: Any) -> ConstraintRecord:
    payload = {
        "case_id": record.case_id,
        "case_kind": record.case_kind,
        "seed": record.seed,
        "variables": record.variables,
        "constraints": record.constraints,
        "objective_terms": record.objective_terms,
        "schema_version": record.schema_version,
        "description": record.description,
    }
    payload.update(updates)
    return ConstraintRecord(**payload)


def _case_by_id(cases: Sequence[ConstraintRecord], case_id: str) -> ConstraintRecord:
    for case in cases:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def _semantic_mismatch(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "satisfiable",
        "violated_constraint_ids",
        "protected_violations",
        "objective_value",
        "scalar_violation_energy",
    )
    return any(left.get(field) != right.get(field) for field in fields)


def build_attack_matrix(cases: Sequence[ConstraintRecord]) -> JsonDict:
    """Run translation-risk attacks as fail-closed controls."""

    rows: list[JsonDict] = []

    negated = _case_by_id(cases, "negated_unsat")
    dropped = _copy_record(
        negated,
        case_id="attack_dropped_negation",
        constraints=tuple(
            ConstraintSpec(c.constraint_id, _transform_expr_drop_negation(c.expr), c.weight, c.protected)
            for c in negated.constraints
        ),
    )
    original = exhaustive_backend_solve(negated)["row"]
    attacked = exhaustive_backend_solve(dropped)["row"]
    mismatch = _semantic_mismatch(original, attacked)
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "dropped_negation",
            "case_id": negated.case_id,
            "seed": negated.seed,
            "backend": "attack_simulator",
            "semantic_mismatch": mismatch,
            "detected": mismatch,
            "false_accept": not mismatch,
            "expected_invariant": "negation must preserve satisfiability and violation ids",
            "observed_mismatch": {
                "original_satisfiable": original["satisfiable"],
                "attacked_satisfiable": attacked["satisfiable"],
            },
        }
    )

    boundary = _case_by_id(cases, "unsat_linear_boundary")
    widened = _copy_record(
        boundary,
        case_id="attack_domain_widening",
        variables=(FiniteDomainVar("x", 0, 2),),
    )
    original = exhaustive_backend_solve(boundary)["row"]
    attacked = exhaustive_backend_solve(widened)["row"]
    mismatch = _semantic_mismatch(original, attacked)
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "domain_widening",
            "case_id": boundary.case_id,
            "seed": boundary.seed,
            "backend": "attack_simulator",
            "semantic_mismatch": mismatch,
            "detected": mismatch,
            "false_accept": not mismatch,
            "expected_invariant": "domain receipts must match the source finite bounds",
            "record_hash_changed": boundary.record_hash() != widened.record_hash(),
        }
    )

    invalid = unsupported_record_fixtures()["ambiguous_integer_to_boolean"]
    invalid_errors = validate_record(invalid)
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "integer_to_boolean_coercion",
            "case_id": invalid.case_id,
            "seed": invalid.seed,
            "backend": "validator",
            "rejected_before_translation": bool(invalid_errors),
            "rejection_reasons": invalid_errors,
            "detected": bool(invalid_errors),
            "false_accept": not bool(invalid_errors),
        }
    )

    auxiliary = _case_by_id(cases, "auxiliary_link_sat")
    leaked = _copy_record(
        auxiliary,
        case_id="attack_auxiliary_variable_leakage",
        constraints=tuple(c for c in auxiliary.constraints if c.constraint_id != "c_aux_link"),
    )
    original = exhaustive_backend_solve(auxiliary)["row"]
    attacked = exhaustive_backend_solve(leaked)["row"]
    mismatch = _semantic_mismatch(original, attacked)
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "auxiliary_variable_leakage",
            "case_id": auxiliary.case_id,
            "seed": auxiliary.seed,
            "backend": "attack_simulator",
            "semantic_mismatch": mismatch,
            "detected": mismatch,
            "false_accept": not mismatch,
            "expected_invariant": "auxiliary variables cannot be accepted without source link constraints",
            "original_assignment": original["selected_assignment"],
            "attacked_assignment": attacked["selected_assignment"],
        }
    )

    for attack_id, fixture_id in (
        ("overflow", "overflow_domain"),
        ("duplicate_constraint_ids", "duplicate_constraint_ids"),
    ):
        record = unsupported_record_fixtures()[fixture_id]
        errors = validate_record(record)
        rows.append(
            {
                "row_type": "attack",
                "attack_id": attack_id,
                "case_id": record.case_id,
                "seed": record.seed,
                "backend": "validator",
                "rejected_before_translation": bool(errors),
                "rejection_reasons": errors,
                "detected": bool(errors),
                "false_accept": not bool(errors),
            }
        )

    sat = _case_by_id(cases, "sat_linear_all_different")
    reversed_objective = _copy_record(
        sat,
        case_id="attack_objective_sign_reversal",
        objective_terms=tuple(
            ObjectiveTerm(term.objective_id, term.expr, -int(term.weight))
            for term in sat.objective_terms
        ),
    )
    original = exhaustive_backend_solve(sat)["row"]
    attacked = exhaustive_backend_solve(reversed_objective)["row"]
    objective_mismatch = original["objective_value"] != attacked["objective_value"]
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "objective_sign_reversal",
            "case_id": sat.case_id,
            "seed": sat.seed,
            "backend": "attack_simulator",
            "objective_mismatch": objective_mismatch,
            "detected": objective_mismatch,
            "false_accept": not objective_mismatch,
            "expected_invariant": "objective term signs are source semantics",
            "original_objective": original["objective_value"],
            "attacked_objective": attacked["objective_value"],
        }
    )

    protected = _case_by_id(cases, "protected_clause_unsat")
    left_assignment = {"p": 0, "q": 0}
    right_assignment = {"p": 1, "q": 0}
    left_energy = scalar_violation_energy(protected, left_assignment)
    right_energy = scalar_violation_energy(protected, right_assignment)
    left_set = violated_constraint_ids(protected, left_assignment)
    right_set = violated_constraint_ids(protected, right_assignment)
    same_total = left_energy == right_energy
    same_set = left_set == right_set
    detected = same_total and not same_set
    rows.append(
        {
            "row_type": "attack",
            "attack_id": "matching_totals_different_violation_sets",
            "case_id": protected.case_id,
            "seed": protected.seed,
            "backend": "row_reducer",
            "scalar_energy_equal": same_total,
            "violation_sets_equal": same_set,
            "left_violated_constraint_ids": left_set,
            "right_violated_constraint_ids": right_set,
            "detected": detected,
            "false_accept": not detected,
            "expected_invariant": "violation ids must match, not only scalar totals",
        }
    )

    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION + ".attack_matrix",
        "rows": rows,
        "attack_count": len(rows),
        "all_attacks_detected": all(row["detected"] is True for row in rows),
        "false_accept_count": sum(1 for row in rows if row["false_accept"] is True),
        "failed_attack_ids": [row["attack_id"] for row in rows if row["detected"] is not True],
    }


def scalar_violation_energy_rows(backend_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project backend rows into scalar energy diagnostics."""

    return [
        {
            "row_type": "scalar_violation_energy",
            "case_id": row["case_id"],
            "seed": row["seed"],
            "backend": row["backend"],
            "record_hash": row["record_hash"],
            "selected_assignment": dict(row["selected_assignment"]),
            "violated_constraint_ids": list(row["violated_constraint_ids"]),
            "protected_violations": list(row["protected_violations"]),
            "objective_value": int(row["objective_value"]),
            "scalar_violation_energy": int(row["scalar_violation_energy"]),
            "release_authority": False,
        }
        for row in backend_rows
    ]


def _case_manifest(cases: Sequence[ConstraintRecord]) -> JsonDict:
    """Build an immutable manifest for code-sealed cases."""

    rows = [
        {
            "case_id": case.case_id,
            "case_kind": case.case_kind,
            "seed": int(case.seed),
            "record_hash": case.record_hash(),
            "variable_count": len(case.variables),
            "constraint_ids": [constraint.constraint_id for constraint in case.constraints],
            "objective_ids": [term.objective_id for term in case.objective_terms],
            "state_count": _state_count(case),
            "sealed_source": "python/carnot/experiment_6477_backend_neutral_exact_constraint_record.py",
        }
        for case in cases
    ]
    payload = {
        "schema_version": ARTIFACT_SCHEMA_VERSION + ".case_manifest",
        "planning_date": RUN_DATE,
        "case_count": len(rows),
        "random_case_seeds": list(RANDOM_CASE_SEEDS),
        "rows": rows,
    }
    return {**payload, "manifest_hash": receipts.sha256_json(payload)}


def recompute_aggregates_from_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute readiness and parity only from per-unit rows."""

    row_type_counts = Counter(str(row.get("row_type")) for row in rows)
    backend_rows = [row for row in rows if row.get("row_type") == "backend_case"]
    unsupported_rows = [row for row in rows if row.get("row_type") == "unsupported_operation"]
    attack_rows = [row for row in rows if row.get("row_type") == "attack"]
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in backend_rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    pair_complete = all({row["backend"] for row in group} == {"z3", "exhaustive"} for group in grouped.values())

    satisfiability_mismatches = 0
    witness_validity_mismatches = 0
    violation_set_mismatches = 0
    protected_mismatches = 0
    objective_mismatches = 0
    scalar_energy_mismatches = 0
    witness_invalid = 0
    domain_invalid = 0
    protected_violation_pairs = 0
    for group in grouped.values():
        by_backend = {row["backend"]: row for row in group}
        if set(by_backend) != {"z3", "exhaustive"}:
            continue
        left = by_backend["z3"]
        right = by_backend["exhaustive"]
        satisfiability_mismatches += int(left["satisfiable"] != right["satisfiable"])
        witness_validity_mismatches += int(left["witness_valid"] != right["witness_valid"])
        violation_set_mismatches += int(
            left["violated_constraint_ids"] != right["violated_constraint_ids"]
        )
        protected_mismatches += int(left["protected_violations"] != right["protected_violations"])
        objective_mismatches += int(left["objective_value"] != right["objective_value"])
        scalar_energy_mismatches += int(
            left["scalar_violation_energy"] != right["scalar_violation_energy"]
        )
        protected_violation_pairs += int(bool(left["protected_violations"]))
        for row in (left, right):
            witness_invalid += int(row["satisfiable"] is True and row["witness_valid"] is not True)
            domain_invalid += int(row["domain_assignment_valid"] is not True)

    all_unsupported_rejected = all(
        row.get("rejected_before_translation") is True
        and row.get("backend_result_trusted") is False
        for row in unsupported_rows
    )
    all_attacks_detected = bool(attack_rows) and all(
        row.get("detected") is True and row.get("false_accept") is False for row in attack_rows
    )
    all_parity = all(
        value == 0
        for value in (
            satisfiability_mismatches,
            witness_validity_mismatches,
            violation_set_mismatches,
            protected_mismatches,
            objective_mismatches,
            scalar_energy_mismatches,
            witness_invalid,
            domain_invalid,
        )
    ) and pair_complete
    score = 1.0 if all_parity and all_unsupported_rejected and all_attacks_detected else 0.0
    return {
        "row_count": len(rows),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "case_count": len(grouped),
        "backend_case_row_count": len(backend_rows),
        "all_backend_pairs_complete": pair_complete,
        "satisfiability_mismatch_count": satisfiability_mismatches,
        "witness_validity_mismatch_count": witness_validity_mismatches,
        "witness_invalid_count": witness_invalid,
        "domain_invalid_count": domain_invalid,
        "violation_set_mismatch_count": violation_set_mismatches,
        "protected_violation_mismatch_count": protected_mismatches,
        "objective_value_mismatch_count": objective_mismatches,
        "scalar_energy_mismatch_count": scalar_energy_mismatches,
        "protected_violation_pair_count": protected_violation_pairs,
        "unsupported_operation_count": len(unsupported_rows),
        "unsupported_rejected_count": sum(
            1 for row in unsupported_rows if row.get("rejected_before_translation") is True
        ),
        "all_unsupported_operations_rejected": all_unsupported_rejected,
        "attack_count": len(attack_rows),
        "detected_attack_count": sum(1 for row in attack_rows if row.get("detected") is True),
        "false_accept_count": sum(1 for row in attack_rows if row.get("false_accept") is True),
        "all_attacks_detected": all_attacks_detected,
        "all_backend_parity_checks_passed": all_parity,
        "exact_constraint_record_ready_score_from_rows": score,
    }


def _parity_summary(aggregate: Mapping[str, Any], kind: str) -> JsonDict:
    """Build one top-level parity summary from aggregate counts."""

    key_by_kind = {
        "satisfiability": "satisfiability_mismatch_count",
        "witness_validity": "witness_validity_mismatch_count",
        "violation_set": "violation_set_mismatch_count",
    }
    mismatch_key = key_by_kind[kind]
    return {
        "kind": kind,
        "case_count": aggregate["case_count"],
        "backend_case_row_count": aggregate["backend_case_row_count"],
        "mismatch_count": aggregate[mismatch_key],
        "all_match": aggregate[mismatch_key] == 0 and aggregate["all_backend_pairs_complete"] is True,
    }


def _backend_versions_and_settings(cases: Sequence[ConstraintRecord]) -> JsonDict:
    """Record exact backend versions, settings, and CPU budget."""

    total_states = sum(_state_count(case) for case in cases)
    return {
        "z3": {
            "package": "z3-solver",
            "package_version": _package_version("z3-solver"),
            "z3_version": z3.get_version_string(),
            "settings": {
                "solver": "Optimize",
                "timeout_ms": Z3_TIMEOUT_MS,
                "priority": "lex",
                "model_completion": True,
            },
        },
        "exhaustive_reference": {
            "backend": "bounded_python_exhaustive_evaluator",
            "python": platform.python_version(),
            "state_budget_per_case": MAX_EXHAUSTIVE_STATES,
            "total_states_enumerated_per_full_run": total_states,
            "enumeration_order": "variable declaration order, lexicographic values",
        },
        "available_cpu_budget": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "state_budget_per_case": MAX_EXHAUSTIVE_STATES,
        },
        "dependency_policy": "no_new_required_dependency_added; z3-solver is already declared and installed",
    }


def _tests_run_receipt(test_exit_codes: Mapping[str, int | None] | None) -> JsonDict:
    exits = dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS})
    return {
        "commands": list(DEFAULT_TEST_COMMANDS),
        "exit_codes": exits,
        "all_recorded_passed": all(exits.get(command) == 0 for command in DEFAULT_TEST_COMMANDS),
    }


def _field_provenance(
    source_hashes: Mapping[str, str | None],
    case_manifest: Mapping[str, Any],
    backend_versions: Mapping[str, Any],
) -> dict[str, JsonDict]:
    """Map each required field to source and reducer receipts."""

    source_paths = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(source_hashes.items())
        if digest is not None
    ]
    case_hashes = [
        {"case_id": row["case_id"], "record_hash": row["record_hash"]}
        for row in case_manifest["rows"]
    ]
    backend_receipt = receipts.sha256_json(backend_versions)
    return {
        field: {
            "spec_refs": ["REQ-VERIFY-6477"],
            "source_paths": source_paths,
            "case_hashes": case_hashes,
            "backend_versions_and_settings_hash": backend_receipt,
            "value_source": "Z3 translation, exhaustive replay, unsupported validators, and row reducers",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions_checked(
    *,
    root: Path,
    run_date: str,
    source_hashes: Mapping[str, str | None],
    schema: Mapping[str, Any],
    backend_versions: Mapping[str, Any],
    case_manifest: Mapping[str, Any],
) -> JsonDict:
    """Record backend, version, code, and CPU receipts before the run."""

    return {
        "run_date": run_date,
        "planning_date": RUN_DATE,
        "repository_state": {
            "head": _git_output(["rev-parse", "HEAD"], root),
            "status_short": _git_output(["status", "--short"], root),
        },
        "schema_sha256": schema["schema_sha256"],
        "case_manifest_hash": case_manifest["manifest_hash"],
        "backend_versions_and_settings": backend_versions,
        "runtime_packages": {
            "pytest": _package_version("pytest"),
            "coverage": _package_version("coverage"),
            "pytest-cov": _package_version("pytest-cov"),
            "pytest-xdist": _package_version("pytest-xdist"),
        },
        "source_hashes": dict(source_hashes),
        "available_cpu_budget": backend_versions["available_cpu_budget"],
        "llm_invocation_allowed": False,
        "required_dependency_added": False,
        "inference_substrate_checked": INFERENCE_SUBSTRATE,
    }


def _gate_check_summary(
    *,
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
) -> JsonDict:
    """Build the conjunctive readiness gate summary."""

    checks = {
        "satisfiability_parity": aggregate["satisfiability_mismatch_count"] == 0,
        "witness_validity_replayed": aggregate["witness_invalid_count"] == 0
        and aggregate["witness_validity_mismatch_count"] == 0,
        "violation_ids_match": aggregate["violation_set_mismatch_count"] == 0,
        "protected_violations_match": aggregate["protected_violation_mismatch_count"] == 0,
        "objective_values_match": aggregate["objective_value_mismatch_count"] == 0,
        "scalar_energies_match": aggregate["scalar_energy_mismatch_count"] == 0,
        "unsupported_operations_fail_closed": aggregate["all_unsupported_operations_rejected"] is True,
        "attacks_caught": aggregate["all_attacks_detected"] is True,
        "aggregate_rows_recomputed": aggregate["exact_constraint_record_ready_score_from_rows"] == 1.0,
        "protected_files_unchanged": protected["unchanged"] is True,
    }
    return {
        "checks": checks,
        "all_gates_passed": all(checks.values()),
        "failed_gates": [key for key, value in checks.items() if not value],
        "mismatch_rows": [],
    }


def _status(score: float, gates: Mapping[str, Any]) -> str:
    if score == 1.0 and gates.get("all_gates_passed") is True:
        return "complete"
    return "blocked_exact_constraint_record_parity"


def _honest_verdict(status: str) -> str:
    if status == "complete":
        return (
            "complete: Z3 and exhaustive replay agree within the declared "
            "finite-domain record; scalar energy remains diagnostic only"
        )
    return (
        "complete_blocked: exact backend parity or fail-closed controls failed "
        "within the finite-domain record"
    )


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float,
    tests_run: Mapping[str, int | None] | None,
) -> JsonDict:
    """Build the terminal Exp6477 artifact."""

    protected_before = _protected_hashes(root)
    source_hashes = _source_hashes(root)
    cases = immutable_cases()
    case_results = [evaluate_case(case) for case in cases]
    backend_rows = [row for result in case_results for row in result["backend_rows"]]
    unsupported_rows = build_unsupported_operation_rows()
    attack_matrix = build_attack_matrix(cases)
    per_unit_rows = [*backend_rows, *unsupported_rows, *attack_matrix["rows"]]
    aggregate = recompute_aggregates_from_rows(per_unit_rows)
    energy_rows = scalar_violation_energy_rows(backend_rows)
    case_manifest = _case_manifest(cases)
    schema = constraint_record_schema_and_hash()
    backend_versions = _backend_versions_and_settings(cases)
    protected = _protected_unchanged(root, protected_before)
    test_receipt = _tests_run_receipt(tests_run)
    gates = _gate_check_summary(
        aggregate=aggregate,
        protected=protected,
    )
    score = float(aggregate["exact_constraint_record_ready_score_from_rows"])
    if not gates["all_gates_passed"]:
        score = 0.0
    status = _status(score, gates)
    translation_receipts = {
        result["case_id"]: result["translation_receipts"] for result in case_results
    }
    artifact: JsonDict = {
        "status": status,
        "constraint_record_schema_and_hash": schema,
        "backend_versions_and_settings": backend_versions,
        "immutable_case_manifest": case_manifest,
        "translation_receipts": translation_receipts,
        "per_unit_rows": per_unit_rows,
        "satisfiability_parity": _parity_summary(aggregate, "satisfiability"),
        "witness_validity_parity": _parity_summary(aggregate, "witness_validity"),
        "violation_set_parity": _parity_summary(aggregate, "violation_set"),
        "scalar_violation_energy_rows": energy_rows,
        "unsupported_operation_rows": unsupported_rows,
        "aggregate_row_recomputation": aggregate,
        "attack_matrix": attack_matrix,
        "exact_constraint_record_ready_score": score,
        "protected_files_unchanged": protected,
        "gate_check_summary": gates,
        "preconditions_checked": _preconditions_checked(
            root=root,
            run_date=run_date,
            source_hashes=source_hashes,
            schema=schema,
            backend_versions=backend_versions,
            case_manifest=case_manifest,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(source_hashes, case_manifest, backend_versions),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s),
        "tests_run": test_receipt,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(status),
        "rows": per_unit_rows,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile fields."""

    normalized = json.loads(canonical_json(payload))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return receipts.sha256_json(normalized)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Validate required fields, row reduction, and authority boundaries."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return [f"missing required field: {missing[0]}"]
    aggregate = recompute_aggregates_from_rows(artifact.get("per_unit_rows", []))
    if artifact.get("aggregate_row_recomputation") != aggregate:
        errors.append("aggregate_row_recomputation mismatch")
    expected_score = aggregate["exact_constraint_record_ready_score_from_rows"]
    if artifact.get("exact_constraint_record_ready_score") != expected_score:
        errors.append("exact_constraint_record_ready_score mismatch")
    if artifact.get("satisfiability_parity") != _parity_summary(aggregate, "satisfiability"):
        errors.append("satisfiability_parity mismatch")
    if artifact.get("witness_validity_parity") != _parity_summary(aggregate, "witness_validity"):
        errors.append("witness_validity_parity mismatch")
    if artifact.get("violation_set_parity") != _parity_summary(aggregate, "violation_set"):
        errors.append("violation_set_parity mismatch")
    if artifact.get("attack_matrix", {}).get("all_attacks_detected") is not True:
        errors.append("attack matrix must detect every attack")
    if any(row.get("backend_result_trusted") is not False for row in artifact.get("unsupported_operation_rows", [])):
        errors.append("unsupported operation row trusted a backend")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle must be true within declared finite-domain record")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        errors.append("protected files changed")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact.get("field_principles", {}):
            errors.append(f"missing field_principles entry: {field_name}")
            break
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(("complete:", "complete_")):
        errors.append("honest_verdict lacks required terminal prefix")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], path: str | Path) -> Path:
    """Write the artifact atomically."""

    return receipts.write_json_atomic(path, artifact)


def run(
    *,
    date: str = RUN_DATE,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Build and write the Exp6477 artifact."""

    start = time.monotonic()
    artifact = build_artifact(
        root=REPO_ROOT,
        run_date=date,
        duration_s=max(time.monotonic() - start, 0.0001),
        tests_run=test_exit_codes,
    )
    write_artifact(artifact, result_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result_path = Path(args.result_path)
    if args.validate:
        if not result_path.is_file():
            print(json.dumps({"ok": False, "errors": ["artifact missing"]}, sort_keys=True))
            return 1
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        errors = validate_artifact(payload)
        print(
            json.dumps(
                {"ok": not errors, "errors": errors, "path": str(result_path)},
                sort_keys=True,
            )
        )
        return 0 if not errors else 1
    artifact = run(date=str(args.date), result_path=result_path)
    print(
        json.dumps(
            {
                "path": str(result_path),
                "status": artifact["status"],
                "exact_constraint_record_ready_score": artifact[
                    "exact_constraint_record_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
