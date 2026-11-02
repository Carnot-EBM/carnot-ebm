"""ROCE-to-validator-tree compiler for deterministic local checks.

Spec: REQ-VERIFY-1878, SCENARIO-VERIFY-1878.

This module bridges ROCE prompt constraints into the same trust boundary used
by Carnot's NSVIF validators: extracted constraints are data, and only fixed
local Python functions evaluate candidate outputs.  The compiled tree also
emits PySAT-compatible and Z3-compatible hard-conjunction metadata for the
same leaf IDs.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import operator
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.roce_extractor import ROCEExtractor
from carnot.verifiers import dsl

JsonDict = dict[str, Any]

RUN_DATE = "20260511"
EXPERIMENT_ID = 1878
EXPERIMENT = "1878_roce_validator_tree"
TREE_SCHEMA_VERSION = "carnot.roce_validator_tree.v1"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1878_roce_validator_tree.json")
SPEC_TRACES = ["REQ-VERIFY-1878", "SCENARIO-VERIFY-1878"]
COMPILED_BACKENDS = ["python", "pysat_cnf", "z3"]
SUPPORTED_PREDICATES = frozenset(
    {
        "format_json",
        "json_required_keys",
        "required_text",
        "forbidden_text",
        "word_count_at_most",
        "word_count_at_least",
        "single_line",
        "arithmetic_equality",
        "conditional_required_text",
    }
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "honest_verdict",
    "validator_tree_compiler_ready",
    "zero_false_accepts",
    "false_accept_count",
    "constraint_coverage_rate",
    "unsupported_constraint_types",
    "tests_run",
)


@dataclass(frozen=True)
class GuardSpec:
    """Predicate that controls whether a guarded validator leaf is active."""

    predicate: str
    arguments: JsonDict

    def evaluate(self, output_text: str, parsed_json: Any) -> bool:
        if self.predicate == "contains_text":
            return _contains_text(output_text, str(self.arguments["term"]))
        if self.predicate == "json_key_equals":
            key = str(self.arguments["key"])
            value = self.arguments["value"]
            return isinstance(parsed_json, dict) and parsed_json.get(key) == value
        raise ValueError(f"unsupported guard predicate:{self.predicate}")

    def to_dict(self) -> JsonDict:
        return {"predicate": self.predicate, "arguments": dict(self.arguments)}


@dataclass(frozen=True)
class ValidatorLeaf:
    """One local Python validation leaf compiled from a ROCE predicate."""

    id: str
    predicate: str
    arguments: JsonDict
    description: str
    source_constraint_type: str
    guard: GuardSpec | None = None

    def validate(self, output_text: str, parsed_json: Any, json_error: str | None) -> "LeafResult":
        if self.guard is not None and not self.guard.evaluate(output_text, parsed_json):
            return LeafResult(self.id, self.predicate, accepted=True, active=False)
        issue = _evaluate_leaf(self, output_text, parsed_json, json_error)
        return LeafResult(
            self.id,
            self.predicate,
            accepted=issue is None,
            active=True,
            issue=issue,
        )

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "predicate": self.predicate,
            "arguments": dict(self.arguments),
            "description": self.description,
            "source_constraint_type": self.source_constraint_type,
            "guard": None if self.guard is None else self.guard.to_dict(),
        }


@dataclass(frozen=True)
class LeafResult:
    """Validation outcome for one compiled leaf."""

    leaf_id: str
    predicate: str
    accepted: bool
    active: bool
    issue: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        return {
            "leaf_id": self.leaf_id,
            "predicate": self.predicate,
            "accepted": self.accepted,
            "active": self.active,
            "issue": self.issue,
        }


@dataclass(frozen=True)
class TreeValidationResult:
    """Whole-tree validation result with failures, skipped leaves, and coverage."""

    accepted: bool
    leaf_results: tuple[LeafResult, ...]
    unsupported_constraint_types: list[str]

    @property
    def failure_ids(self) -> list[str]:
        return [result.leaf_id for result in self.leaf_results if result.active and not result.accepted]

    @property
    def skipped_ids(self) -> list[str]:
        return [result.leaf_id for result in self.leaf_results if not result.active]

    def to_dict(self) -> JsonDict:
        return {
            "accepted": self.accepted,
            "failure_ids": self.failure_ids,
            "skipped_ids": self.skipped_ids,
            "unsupported_constraint_types": list(self.unsupported_constraint_types),
            "leaf_results": [result.to_dict() for result in self.leaf_results],
        }


@dataclass(frozen=True)
class Z3TreeProblem:
    """Z3-compatible hard-conjunction descriptor for the validator tree."""

    variables: dict[str, str]
    assertions: list[str]
    backend: str = "z3-compatible-hard-conjunction"

    def to_dict(self) -> JsonDict:
        return {
            "backend": self.backend,
            "variables": dict(self.variables),
            "assertions": list(self.assertions),
            "z3_backend_available": z3_backend_available(),
        }


@dataclass(frozen=True)
class ROCEValidatorTree:
    """Compiled local evaluation tree for ROCE constraints."""

    leaves: tuple[ValidatorLeaf, ...]
    unsupported_constraint_types: list[str]
    total_constraint_count: int
    case_id: str = ""

    @property
    def supported_constraint_count(self) -> int:
        return len(self.leaves)

    @property
    def constraint_coverage_rate(self) -> float:
        return _rate(self.supported_constraint_count, self.total_constraint_count)

    @property
    def pysat_problem(self) -> dsl.PySatProblem:
        variables: dict[str, int] = {}
        clauses: list[list[int]] = []
        next_var = 1
        for leaf in self.leaves:
            leaf_var = next_var
            variables[leaf.id] = leaf_var
            next_var += 1
            if leaf.guard is None:
                clauses.append([leaf_var])
                continue
            guard_id = f"guard-{leaf.id}"
            guard_var = next_var
            variables[guard_id] = guard_var
            next_var += 1
            clauses.append([-guard_var, leaf_var])
        return dsl.PySatProblem(variables=variables, clauses=clauses)

    @property
    def z3_problem(self) -> Z3TreeProblem:
        variables: dict[str, str] = {}
        assertions: list[str] = []
        for leaf in self.leaves:
            leaf_symbol = _z3_symbol(leaf.id)
            variables[leaf.id] = leaf_symbol
            if leaf.guard is None:
                assertions.append(leaf_symbol)
                continue
            guard_id = f"guard-{leaf.id}"
            guard_symbol = _z3_symbol(guard_id)
            variables[guard_id] = guard_symbol
            assertions.append(f"Implies({guard_symbol}, {leaf_symbol})")
        return Z3TreeProblem(variables=variables, assertions=assertions)

    def validate(self, output_text: str) -> TreeValidationResult:
        parsed_json, json_error = _parse_json(output_text)
        leaf_results = tuple(
            leaf.validate(output_text, parsed_json, json_error) for leaf in self.leaves
        )
        accepted = (
            not self.unsupported_constraint_types
            and all(result.accepted for result in leaf_results)
        )
        return TreeValidationResult(
            accepted=accepted,
            leaf_results=leaf_results,
            unsupported_constraint_types=self.unsupported_constraint_types,
        )

    def to_dict(self) -> JsonDict:
        return {
            "case_id": self.case_id,
            "tree_schema_version": TREE_SCHEMA_VERSION,
            "compiled_backends": list(COMPILED_BACKENDS),
            "leaves": [leaf.to_dict() for leaf in self.leaves],
            "supported_constraint_count": self.supported_constraint_count,
            "total_constraint_count": self.total_constraint_count,
            "constraint_coverage_rate": self.constraint_coverage_rate,
            "unsupported_constraint_types": list(self.unsupported_constraint_types),
            "pysat_problem": self.pysat_problem.to_dict(),
            "z3_problem": self.z3_problem.to_dict(),
        }


def z3_backend_available() -> bool:
    """Return whether the optional Z3 Python package is importable."""

    return importlib.util.find_spec("z3") is not None


def compile_roce_validator_tree(
    source: str | Sequence[ConstraintResult],
    *,
    case_id: str = "",
) -> ROCEValidatorTree:
    """Compile prompt text or ROCE `ConstraintResult` rows into a validator tree."""

    constraints = ROCEExtractor().extract(source, domain="roce") if isinstance(source, str) else list(source)
    leaves: list[ValidatorLeaf] = []
    unsupported: list[str] = []
    for index, constraint in enumerate(constraints, start=1):
        predicate = _predicate(constraint)
        if predicate not in SUPPORTED_PREDICATES:
            unsupported.append(predicate)
            continue
        leaves.append(_compile_leaf(index, constraint, predicate))
    return ROCEValidatorTree(
        leaves=tuple(leaves),
        unsupported_constraint_types=sorted(set(unsupported)),
        total_constraint_count=len(constraints),
        case_id=case_id,
    )


def default_roce_fixture_cases() -> list[JsonDict]:
    """Return deterministic valid and adversarial invalid ROCE fixture cases."""

    prompt = (
        "Return a single-line JSON object only. "
        'Use strict key order {"answer": ..., "sum": ...} and no other top-level keys. '
        'Include "approved". Do not mention "secret". Keep under 20 words. '
        "The response must state 2 + 3 = 5. "
        'If the response contains "approved", it must also contain "audited".'
    )
    return [
        {
            "case_id": "roce-json-guard-arithmetic",
            "prompt": prompt,
            "known_good": '{"answer":"approved audited","sum":"2 + 3 = 5"}',
            "known_bad": [
                '{"answer":"approved","sum":"2 + 3 = 5"}',
                '{"sum":"2 + 3 = 5","answer":"approved audited"}',
                '{"answer":"approved audited secret","sum":"2 + 3 = 5"}',
                '{"answer":"approved audited","sum":"2 + 3 = 6"}',
                "approved audited 2 + 3 = 5",
            ],
        }
    ]


def evaluate_fixture_case(case: Mapping[str, Any]) -> JsonDict:
    """Compile and evaluate one Exp 1878 fixture case."""

    case_id = str(case.get("case_id") or "unknown")
    tree = compile_roce_validator_tree(str(case.get("prompt") or ""), case_id=case_id)
    good = tree.validate(str(case.get("known_good") or ""))
    bad_results = [tree.validate(str(output)) for output in case.get("known_bad", [])]
    false_accept_count = sum(1 for result in bad_results if result.accepted)
    return {
        "case_id": case_id,
        "validator_tree": tree.to_dict(),
        "known_good": good.to_dict(),
        "known_bad": [result.to_dict() for result in bad_results],
        "false_accept_count": false_accept_count,
        "zero_false_accepts": false_accept_count == 0,
    }


def build_artifact(
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Build the Exp 1878 artifact without writing it."""

    rows = [evaluate_fixture_case(case) for case in (cases or default_roce_fixture_cases())]
    total_constraints = sum(row["validator_tree"]["total_constraint_count"] for row in rows)
    supported_constraints = sum(row["validator_tree"]["supported_constraint_count"] for row in rows)
    unsupported = sorted(
        {
            item
            for row in rows
            for item in row["validator_tree"]["unsupported_constraint_types"]
        }
    )
    false_accept_count = sum(row["false_accept_count"] for row in rows)
    known_good_passes = sum(1 for row in rows if row["known_good"]["accepted"])
    constraint_coverage_rate = _rate(supported_constraints, total_constraints)
    zero_false_accepts = false_accept_count == 0
    ready = (
        bool(rows)
        and known_good_passes == len(rows)
        and zero_false_accepts
        and constraint_coverage_rate == 1.0
        and not unsupported
    )
    return {
        "status": "complete" if ready else "partial",
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "timestamp": _timestamp(),
        "spec_traces": list(SPEC_TRACES),
        "tree_schema_version": TREE_SCHEMA_VERSION,
        "compiled_backends": list(COMPILED_BACKENDS),
        "validator_tree_compiler_ready": ready,
        "zero_false_accepts": zero_false_accepts,
        "false_accept_count": false_accept_count,
        "constraint_coverage_rate": constraint_coverage_rate,
        "unsupported_constraint_types": unsupported,
        "fixture_cases": len(rows),
        "known_good_pass_rate": _rate(known_good_passes, len(rows)),
        "case_results": rows,
        "tests_run": list(tests_run or []),
        "honest_verdict": _honest_verdict(ready, constraint_coverage_rate, false_accept_count),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the Exp 1878 artifact has the required completion schema."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert 0.0 <= artifact["constraint_coverage_rate"] <= 1.0, "coverage out of range"
    assert artifact["false_accept_count"] >= 0, "false_accept_count must be nonnegative"
    if artifact["status"] == "complete":
        assert artifact["validator_tree_compiler_ready"] is True, "complete requires ready"
        assert artifact["zero_false_accepts"] is True, "complete requires zero false accepts"
        assert artifact["false_accept_count"] == 0, "complete requires false_accept_count=0"
        assert artifact["constraint_coverage_rate"] == 1.0, "complete requires full coverage"
        assert artifact["unsupported_constraint_types"] == [], "complete requires no unsupported types"


def run_experiment(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    cases: Iterable[Mapping[str, Any]] | None = None,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run Exp 1878 and write `results/experiment_1878_roce_validator_tree.json`."""

    artifact = build_artifact(cases=cases, tests_run=tests_run)
    artifact["artifact_path"] = str(output_path)
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def _compile_leaf(index: int, constraint: ConstraintResult, predicate: str) -> ValidatorLeaf:
    arguments = dict(constraint.metadata.get("arguments") or {})
    guard = None
    if predicate == "conditional_required_text":
        guard_args = dict(arguments["guard"])
        guard_predicate = str(guard_args.pop("predicate"))
        guard = GuardSpec(predicate=guard_predicate, arguments=guard_args)
        arguments = dict(arguments["then"])
    return ValidatorLeaf(
        id=f"c{index:03d}-{predicate}",
        predicate=predicate,
        arguments=arguments,
        description=constraint.description,
        source_constraint_type=constraint.constraint_type,
        guard=guard,
    )


def _evaluate_leaf(
    leaf: ValidatorLeaf,
    output_text: str,
    parsed_json: Any,
    json_error: str | None,
) -> JsonDict | None:
    predicate = leaf.predicate
    args = leaf.arguments
    if predicate == "format_json":
        return None if json_error is None else _issue(leaf, "valid JSON", json_error)
    if predicate == "json_required_keys":
        return _json_required_keys_issue(leaf, parsed_json)
    if predicate in {"required_text", "conditional_required_text"}:
        term = str(args["term"])
        return None if _contains_text(output_text, term) else _issue(leaf, term, "not found")
    if predicate == "forbidden_text":
        term = str(args["term"])
        return _issue(leaf, f"not {term}", "found") if _contains_text(output_text, term) else None
    if predicate == "word_count_at_most":
        count = _word_count(output_text)
        limit = int(args["limit"])
        return None if count <= limit else _issue(leaf, f"<={limit}", count)
    if predicate == "word_count_at_least":
        count = _word_count(output_text)
        limit = int(args["limit"])
        return None if count >= limit else _issue(leaf, f">={limit}", count)
    if predicate == "single_line":
        return None if len(output_text.splitlines()) <= 1 else _issue(leaf, "single line", "multi-line")
    if predicate == "arithmetic_equality":
        return None if _output_has_required_equation(output_text, args) else _issue(
            leaf,
            f"{args['left']} = {args['right']}",
            "not found or false",
        )
    raise AssertionError(f"unsupported compiled predicate:{predicate}")  # pragma: no cover


def _json_required_keys_issue(leaf: ValidatorLeaf, parsed_json: Any) -> JsonDict | None:
    keys = [str(key) for key in leaf.arguments["keys"]]
    if not isinstance(parsed_json, dict):
        return _issue(leaf, keys, "json value is not object")
    actual_keys = list(parsed_json)
    missing = [key for key in keys if key not in parsed_json]
    if missing:
        return _issue(leaf, keys, {"missing": missing, "actual_keys": actual_keys})
    if leaf.arguments.get("ordered") and actual_keys[: len(keys)] != keys:
        return _issue(leaf, keys, {"actual_order": actual_keys})
    if leaf.arguments.get("no_extra_keys") and actual_keys != keys:
        return _issue(leaf, keys, {"actual_keys": actual_keys})
    return None


def _issue(leaf: ValidatorLeaf, expected: Any, observed: Any) -> JsonDict:
    return {
        "constraint_id": leaf.id,
        "message": f"{leaf.predicate} constraint failed",
        "expected": expected,
        "observed": observed,
    }


def _parse_json(output_text: str) -> tuple[Any, str | None]:
    try:
        return json.loads(output_text), None
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"


def _output_has_required_equation(output_text: str, args: Mapping[str, Any]) -> bool:
    required_left = _normalize_expression(str(args["left"]))
    required_right = _normalize_number(str(args["right"]))
    for left, right in _equation_claims(output_text):
        if _normalize_expression(left) != required_left:
            continue
        if _normalize_number(right) != required_right:
            continue
        return _arithmetic_equal(left, right)
    return False


def _equation_claims(text: str) -> list[tuple[str, str]]:
    pattern = r"(-?\d+(?:\.\d+)?(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+)\s*=\s*(-?\d+(?:\.\d+)?)"
    return [(match.group(1), match.group(2)) for match in re.finditer(pattern, text)]


def _arithmetic_equal(left: str, right: str) -> bool:
    return _eval_arithmetic(left) == _eval_arithmetic(right)


def _eval_arithmetic(expression: str) -> float:
    return float(_eval_ast(ast.parse(expression, mode="eval").body))


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_ast(node.operand)
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPS:
        return _ARITHMETIC_OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    raise ValueError(f"unsupported arithmetic expression:{ast.dump(node)}")


_ARITHMETIC_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _predicate(constraint: ConstraintResult) -> str:
    return str(constraint.metadata.get("predicate") or constraint.constraint_type)


def _contains_text(output_text: str, term: str) -> bool:
    return term.lower() in output_text.lower()


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _normalize_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _normalize_number(number: str) -> str:
    value = float(number)
    return str(int(value)) if value.is_integer() else str(value)


def _z3_symbol(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _honest_verdict(ready: bool, coverage: float, false_accept_count: int) -> str:
    if ready:
        return (
            "complete: ROCE constraints compiled to guarded Python, "
            "PySAT-compatible, and Z3-compatible validator leaves with zero false accepts"
        )
    return (
        "partial: ROCE validator-tree gates not all satisfied; "
        f"constraint_coverage_rate={coverage}, false_accept_count={false_accept_count}"
    )


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dict(payload)


__all__ = [
    "COMPILED_BACKENDS",
    "DEFAULT_ARTIFACT_PATH",
    "EXPERIMENT_ID",
    "REQUIRED_ARTIFACT_FIELDS",
    "ROCEValidatorTree",
    "SPEC_TRACES",
    "TREE_SCHEMA_VERSION",
    "TreeValidationResult",
    "ValidatorLeaf",
    "build_artifact",
    "compile_roce_validator_tree",
    "default_roce_fixture_cases",
    "evaluate_fixture_case",
    "run_experiment",
    "validate_artifact",
    "z3_backend_available",
]
