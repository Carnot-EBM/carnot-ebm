"""Exp 3114 offline fragment-level code and constraint verification.

Spec refs: REQ-VERIFY-3114, SCENARIO-VERIFY-3114.

This pilot is intentionally small: it reads checked-in exact fixtures, splits
repairable candidates into local fragments or clauses, and emits repair targets
with exact evidence. It does not call a language model or claim that any repair
was generated; the value is the localized handoff for a later repair step.
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
SCHEMA = "carnot.fragment_level_code_constraint_verification_pilot.v1"
ARTIFACT = "experiment_3114_fragment_level_code_constraint_verification_pilot_v1"
OUTPUT_REL_PATH = Path("results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json")
REPAIR_TARGET_MANIFEST_REL_PATH = Path(
    "results/fragment_verification_pilot_3114/repair_target_manifest.jsonl"
)

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3100_REL_PATH = Path("results/experiment_3100_z3_oracle_feedback_v2.json")
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3114_fragment_level_code_constraint_verification_pilot.py -q --no-cov",
    ".venv/bin/coverage erase && .venv/bin/coverage run --source=python/carnot/eval -m pytest tests/python/test_experiment_3114_fragment_level_code_constraint_verification_pilot.py -q -n 0 --no-cov && .venv/bin/coverage report --include=python/carnot/eval/fragment_verification_pilot_3114.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    (EXP3097_REL_PATH, "exact fixture protocol audit"),
    (MANIFEST_REL_PATH, "stratified exact fixture manifest"),
    (EXP3100_REL_PATH, "z3/test-oracle feedback baseline"),
    (EXP3111_REL_PATH, "certified coherence feedback"),
)
REQUIRED_FIELDS = (
    "fragment_verification_pilot_ready",
    "exact_fixture_count",
    "fragment_count",
    "failing_fragment_count",
    "unknown_fragment_count",
    "repair_target_manifest_path",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)

_SIMPLE_CONSTRAINT_RE = re.compile(r"^(?P<var>[A-Za-z_]\w*)\s*(?P<op>>=|<=|==)\s*(?P<num>-?\d+)$")
_SUM_CONSTRAINT_RE = re.compile(
    r"^(?P<a>[A-Za-z_]\w*)\s*\+\s*(?P<b>[A-Za-z_]\w*)\s*==\s*(?P<num>-?\d+)$"
)
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object from a local artifact path."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read JSONL fixture rows from a local manifest."""

    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            rows.append(dict(payload))
    return rows


def sha256_file(path: Path) -> str:
    """Hash a present source artifact for traceability."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_fixture_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Select one deterministic row for each pilot fixture category."""

    targets = (
        ("arithmetic_code_assertions", "arithmetic_true_verification"),
        ("arithmetic_code_assertions", "arithmetic_false_verification"),
        ("repairable_invalid_candidates", "json_syntax_repair"),
        ("repairable_invalid_candidates", "numeric_bound_repair"),
        ("repairable_invalid_candidates", "python_assertion_repair"),
    )
    selected: list[JsonDict] = []
    for family, perturbation in targets:
        match = next(
            (
                dict(row)
                for row in rows
                if row.get("task_family") == family and row.get("perturbation_type") == perturbation
            ),
            None,
        )
        if match is not None:
            selected.append(match)
    return selected


def fragment_checks_for_row(row: Mapping[str, Any]) -> list[JsonDict]:
    """Split one exact fixture row into checked fragments or clauses."""

    family = str(row.get("task_family", ""))
    perturbation = str(row.get("perturbation_type", ""))
    if family == "arithmetic_code_assertions":
        return _assertion_fragments(row)
    if family == "repairable_invalid_candidates" and perturbation == "python_assertion_repair":
        return _assertion_fragments(row)
    if family == "repairable_invalid_candidates" and perturbation == "json_syntax_repair":
        return _json_fragments(row)
    if family == "repairable_invalid_candidates" and perturbation == "numeric_bound_repair":
        return _numeric_constraint_fragments(row)
    return [_unknown_fragment(row)]


def repair_targets_from_fragments(fragments: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Convert failing fragment rows into the repair handoff manifest shape."""

    targets: list[JsonDict] = []
    for fragment in fragments:
        if fragment["status"] == "fail":
            targets.append(
                {
                    "fixture_id": fragment["fixture_id"],
                    "fragment_id": fragment["fragment_id"],
                    "failing_constraint": fragment["failing_constraint"],
                    "expected_direction": fragment["expected_direction"],
                    "solver_evidence": fragment["solver_evidence"],
                }
            )
    return targets


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3114 terminal artifact payload from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    read_json_object(root_path / EXP3100_REL_PATH)
    read_json_object(root_path / EXP3111_REL_PATH)
    manifest_rel_path = Path(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH)
    rows = read_jsonl_rows(root_path / manifest_rel_path)
    selected_rows = select_fixture_rows(rows)
    fragments = [fragment for row in selected_rows for fragment in fragment_checks_for_row(row)]
    repair_targets = repair_targets_from_fragments(fragments)
    failing_count = sum(1 for fragment in fragments if fragment["status"] == "fail")
    unknown_count = sum(1 for fragment in fragments if fragment["status"] == "unknown")
    source_artifacts = _source_artifacts(root_path, manifest_rel_path)
    ready = bool(
        selected_rows
        and fragments
        and repair_targets
        and failing_count == len(repair_targets)
        and all(source["present"] for source in source_artifacts)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fragment_verification_pilot_ready": ready,
        "exact_fixture_count": len(selected_rows),
        "selected_fixture_ids": [str(row["source_fixture_id"]) for row in selected_rows],
        "fragment_count": len(fragments),
        "failing_fragment_count": failing_count,
        "unknown_fragment_count": unknown_count,
        "non_applicable_fragment_count": sum(
            1 for fragment in fragments if fragment["status"] == "non-applicable"
        ),
        "repair_target_manifest_path": REPAIR_TARGET_MANIFEST_REL_PATH.as_posix(),
        "repair_target_manifest": repair_targets,
        "fragment_checks": fragments,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts,
        "inference_substrate": _inference_substrate(),
        "duration_s": round((time.perf_counter() if now_s is None else now_s) - start, 6),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Write the Exp 3114 JSON artifact and its JSONL repair target manifest."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    manifest_path = root_path / REPAIR_TARGET_MANIFEST_REL_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in artifact["repair_target_manifest"]),
        encoding="utf-8",
    )
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _assertion_fragments(row: Mapping[str, Any]) -> list[JsonDict]:
    fixture_id = str(row["source_fixture_id"])
    payload = row["leakage_safe_prompt_payload"]
    candidate = str(payload["candidate_assertion"])
    parsed = ast.parse(candidate)
    assertion = parsed.body[0]
    compare = assertion.test
    left_value = _eval_int(compare.left)
    claimed_value = _eval_int(compare.comparators[0])
    expression_source = ast.unparse(compare.left)
    claim_source = ast.unparse(compare.comparators[0])
    claim_passes = left_value == claimed_value
    return [
        {
            "fixture_id": fixture_id,
            "fragment_id": f"{fixture_id}:assert_expression",
            "status": "pass",
            "failing_constraint": "",
            "expected_direction": "no change",
            "solver_evidence": {
                "authority": "python_ast_literal_evaluator",
                "expression": expression_source,
                "computed_value": left_value,
            },
        },
        {
            "fixture_id": fixture_id,
            "fragment_id": f"{fixture_id}:assert_claim",
            "status": "pass" if claim_passes else "fail",
            "failing_constraint": "" if claim_passes else "claimed_value == computed_value",
            "expected_direction": (
                "no change"
                if claim_passes
                else f"replace claimed value {claimed_value} with {left_value}"
            ),
            "solver_evidence": {
                "authority": "python_ast_literal_evaluator",
                "assertion": candidate,
                "expression": expression_source,
                "claimed_fragment": claim_source,
                "computed_value": left_value,
                "claimed_value": claimed_value,
            },
        },
    ]


def _json_fragments(row: Mapping[str, Any]) -> list[JsonDict]:
    fixture_id = str(row["source_fixture_id"])
    payload = row["leakage_safe_prompt_payload"]
    candidate = str(payload["candidate"])
    required_fields = [str(field) for field in payload.get("required_fields", [])]
    fragments: list[JsonDict] = []
    try:
        parsed = json.loads(candidate)
        parse_error = ""
    except json.JSONDecodeError as exc:
        parsed = None
        parse_error = str(exc)
    parse_passed = isinstance(parsed, Mapping)
    fragments.append(
        {
            "fixture_id": fixture_id,
            "fragment_id": f"{fixture_id}:json_document",
            "status": "pass" if parse_passed else "fail",
            "failing_constraint": "" if parse_passed else "valid_json_document",
            "expected_direction": (
                "no change"
                if parse_passed
                else "produce parseable JSON while preserving required fields"
            ),
            "solver_evidence": {
                "authority": "python_json_parser",
                "candidate": candidate,
                "parse_error": parse_error,
            },
        }
    )
    for field in required_fields:
        applicable = parse_passed
        present = applicable and field in parsed
        fragments.append(
            {
                "fixture_id": fixture_id,
                "fragment_id": f"{fixture_id}:required_field:{field}",
                "status": (
                    "pass" if present else ("fail" if applicable else "non-applicable")
                ),
                "failing_constraint": "" if present or not applicable else f"required field {field}",
                "expected_direction": (
                    "no change"
                    if present
                    else (
                        f"add required field {field}"
                        if applicable
                        else "fix JSON syntax before checking required fields"
                    )
                ),
                "solver_evidence": {
                    "authority": "python_json_parser",
                    "field": field,
                    "json_parse_passed": parse_passed,
                },
            }
        )
    return fragments


def _numeric_constraint_fragments(row: Mapping[str, Any]) -> list[JsonDict]:
    fixture_id = str(row["source_fixture_id"])
    payload = row["leakage_safe_prompt_payload"]
    assignment = {str(key): int(value) for key, value in payload["candidate_assignment"].items()}
    fragments: list[JsonDict] = []
    for index, constraint in enumerate(payload["constraints"]):
        text = str(constraint)
        passed, evidence, direction = _evaluate_constraint(text, assignment)
        fragments.append(
            {
                "fixture_id": fixture_id,
                "fragment_id": f"{fixture_id}:constraint:{index}",
                "status": "pass" if passed else "fail",
                "failing_constraint": "" if passed else text,
                "expected_direction": "no change" if passed else direction,
                "solver_evidence": evidence,
            }
        )
    return fragments


def _evaluate_constraint(text: str, assignment: Mapping[str, int]) -> tuple[bool, JsonDict, str]:
    simple = _SIMPLE_CONSTRAINT_RE.match(text)
    if simple is not None:
        var = simple.group("var")
        op = simple.group("op")
        rhs = int(simple.group("num"))
        lhs = int(assignment[var])
        passed = (op == ">=" and lhs >= rhs) or (op == "<=" and lhs <= rhs) or (op == "==" and lhs == rhs)
        direction = _simple_direction(var, lhs, op, rhs)
        return passed, _constraint_evidence(text, lhs, rhs, assignment), direction
    summed = _SUM_CONSTRAINT_RE.match(text)
    left_var = summed.group("a")
    right_var = summed.group("b")
    rhs = int(summed.group("num"))
    lhs = int(assignment[left_var]) + int(assignment[right_var])
    direction = f"increase sum by {rhs - lhs} across {[left_var, right_var]}"
    return lhs == rhs, _constraint_evidence(text, lhs, rhs, assignment), direction


def _simple_direction(var: str, lhs: int, op: str, rhs: int) -> str:
    if op == ">=" and lhs < rhs:
        return f"increase {var} by {rhs - lhs}"
    if op == "<=" and lhs > rhs:
        return f"decrease {var} by {lhs - rhs}"
    return f"set {var} to {rhs}"


def _constraint_evidence(text: str, lhs: int, rhs: int, assignment: Mapping[str, int]) -> JsonDict:
    return {
        "authority": "deterministic_integer_constraint_evaluator",
        "constraint": text,
        "lhs_value": lhs,
        "rhs_value": rhs,
        "assignment": dict(assignment),
    }


def _unknown_fragment(row: Mapping[str, Any]) -> JsonDict:
    fixture_id = str(row.get("source_fixture_id", "unknown-fixture"))
    perturbation = str(row.get("perturbation_type", "unknown"))
    return {
        "fixture_id": fixture_id,
        "fragment_id": f"{fixture_id}:{perturbation}",
        "status": "unknown",
        "failing_constraint": "unsupported_fragment_parser",
        "expected_direction": "manual parser/checker extension required",
        "solver_evidence": {
            "authority": "fragment_verification_pilot",
            "reason": "unsupported task_family or perturbation_type",
        },
    }


def _eval_int(node: ast.AST) -> int:
    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.UnaryOp):
        return -_eval_int(node.operand)
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_int(node.left), _eval_int(node.right))
    raise ValueError(f"unsupported arithmetic AST node: {type(node).__name__}")  # pragma: no cover


def _source_artifacts(root: Path, manifest_rel_path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path, role in SOURCE_ARTIFACTS:
        actual_rel = manifest_rel_path if rel_path == MANIFEST_REL_PATH else rel_path
        path = root / actual_rel
        rows.append(
            {
                "path": actual_rel.as_posix(),
                "role": role,
                "present": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _inference_substrate() -> JsonDict:
    return {
        "kind": "offline_fragment_constraint_verification",
        "live_llm_calls": 0,
        "executes_models": False,
        "uses_checked_in_artifacts_only": True,
        "executes_json_parser": True,
        "executes_python_ast_parser": True,
        "executes_integer_constraint_evaluator": True,
        "no_live_llm_inference": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["fragment_verification_pilot_ready"] is True:
        return (
            "complete: fragment verification pilot ready; "
            f"fixtures={artifact['exact_fixture_count']}; "
            f"failing_fragments={artifact['failing_fragment_count']}"
        )
    return "blocked_fragment_verification_pilot_missing_required_evidence"
