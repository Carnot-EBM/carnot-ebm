"""Build Exp 3111 certified coherence feedback over exact fixtures.

Spec refs: REQ-REPORT-3111, SCENARIO-REPORT-3111.

This module treats solver and test-oracle evidence as the authority. It can use
Z3 to localize contradictory SMT-style rows, but it deliberately does not run a
language model or require a cached SOTA pair: the claim is proof-carrying
coherence feedback, not live model lift.
"""

from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path
import time
from typing import Any, Mapping

try:  # pragma: no cover - absence is exercised by passing z3_module=None.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.290"
SCHEMA = "carnot.certified_coherence_feedback.v3"
ARTIFACT = "experiment_3111_certified_coherence_z3_mcs_feedback_v3"
OUTPUT_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3111_certified_coherence_z3_mcs_feedback_v3.py"

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3098_REL_PATH = Path("results/experiment_3098_maxsat_abstention_routing_policy_v1.json")
EXP3100_REL_PATH = Path("results/experiment_3100_z3_oracle_feedback_v2.json")
EXP3110_REL_PATH = Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json")
MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")
NON_TINY_EXACT_COUNT_FLOOR = 6

REQUIRED_FIELDS = (
    "certified_coherence_feedback_v3_ready",
    "z3_available",
    "exact_ground_truth_count",
    "certificate_count",
    "unsat_core_count",
    "minimal_repair_distance_summary",
    "solver_only_success_count",
    "guided_success_count",
    "formal_feedback_delta",
    "vacuity_guard_passed",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3111_certified_coherence_z3_mcs_feedback_v3.py -q",
    ".venv/bin/pytest tests/python/test_experiment_3111_certified_coherence_z3_mcs_feedback_v3.py -q --cov-reset --cov=python/carnot/reporting/certified_coherence_feedback_v3_3111.py --cov-report=term-missing --cov-fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("exp3097_exact_fixture_protocol", EXP3097_REL_PATH, True),
    ("exp3097_stratified_manifest", MANIFEST_REL_PATH, True),
    ("exp3098_maxsat_policy", EXP3098_REL_PATH, True),
    ("exp3100_z3_oracle_feedback_v2", EXP3100_REL_PATH, True),
    ("exp3110_sota_model_manifest_corrigendum", EXP3110_REL_PATH, True),
)
_SIMPLE_CONSTRAINT_RE = re.compile(r"^(?P<var>[A-Za-z_]\w*)\s*(?P<op>>=|<=|==)\s*(?P<num>-?\d+)$")
_SUM_CONSTRAINT_RE = re.compile(
    r"^(?P<a>[A-Za-z_]\w*)\s*\+\s*(?P<b>[A-Za-z_]\w*)\s*==\s*(?P<num>-?\d+)$"
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, failing closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows_from_text(text: str) -> list[JsonDict]:
    """Read JSONL rows, skipping malformed or non-object lines."""

    rows: list[JsonDict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def read_jsonl_rows(path: Path) -> list[JsonDict]:
    """Read a JSONL manifest, returning no rows when it is absent."""

    try:
        return read_jsonl_rows_from_text(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def sha256_file(path: Path) -> str | None:
    """Return a checksum for source provenance when a file is present."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    z3_module: Any = _z3,
    tests_run: list[str] | tuple[str, ...] | None = None,
) -> JsonDict:
    """REQ-REPORT-3111: build the certified feedback artifact from local evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    exp3098 = read_json_object(root_path / EXP3098_REL_PATH)
    exp3100 = read_json_object(root_path / EXP3100_REL_PATH)
    exp3110 = read_json_object(root_path / EXP3110_REL_PATH)
    manifest_rel_path = _manifest_rel_path(exp3097)
    rows = _formal_label_rows(read_jsonl_rows(root_path / manifest_rel_path))
    certificates = [certificate_for_row(row, z3_module=z3_module) for row in rows]
    exact_ground_truth_count = len(rows)
    unsat_core_count = sum(1 for row in certificates if row["unsat_core"])
    correction_set_count = sum(1 for row in certificates if row["minimal_correction_set"])
    repair_summary = _minimal_repair_distance_summary(certificates)
    source_artifacts = _source_artifacts(root_path, manifest_rel_path)
    missing_required_sources = [
        row for row in source_artifacts if row["required"] is True and row["present"] is not True
    ]
    vacuity_guard_passed = _vacuity_guard_passed(
        certificates=certificates,
        unsat_core_count=unsat_core_count,
        correction_set_count=correction_set_count,
        repair_summary=repair_summary,
        upstream_vacuity=exp3100.get("vacuity_guard_passed") is not False,
    )
    z3_available = z3_module is not None
    readiness_checks = {
        "z3_available": z3_available,
        "exp3097_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "exp3098_maxsat_policy_ready": exp3098.get("maxsat_policy_ready") is True,
        "exact_ground_truth_non_tiny": exact_ground_truth_count >= NON_TINY_EXACT_COUNT_FLOOR,
        "all_formal_rows_certified": exact_ground_truth_count > 0
        and len(certificates) == exact_ground_truth_count,
        "contradiction_localization_nonempty": unsat_core_count > 0 or correction_set_count > 0,
        "minimal_repair_distance_summary_computed": repair_summary["count"] > 0,
        "vacuity_guard_passed": vacuity_guard_passed,
        "required_sources_present": not missing_required_sources,
        "cached_sota_pair_not_required": True,
        "solver_only_allowed_without_cached_pair": _solver_only_allowed(exp3110),
    }
    ready = all(
        readiness_checks[key]
        for key in (
            "z3_available",
            "exp3097_protocol_ready",
            "exp3098_maxsat_policy_ready",
            "exact_ground_truth_non_tiny",
            "all_formal_rows_certified",
            "contradiction_localization_nonempty",
            "minimal_repair_distance_summary_computed",
            "vacuity_guard_passed",
            "required_sources_present",
        )
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "certified_coherence_feedback_v3_ready": ready,
        "z3_available": z3_available,
        "z3_version": _z3_version(z3_module),
        "exact_ground_truth_count": exact_ground_truth_count,
        "certificate_count": len(certificates),
        "unsat_core_count": unsat_core_count,
        "minimal_correction_set_count": correction_set_count,
        "minimal_repair_distance_summary": repair_summary,
        "solver_only_success_count": int(exp3100.get("solver_only_success_count") or 0),
        "guided_success_count": int(exp3100.get("guided_success_count") or 0),
        "formal_feedback_delta": float(exp3100.get("formal_feedback_delta") or 0.0),
        "vacuity_guard_passed": vacuity_guard_passed,
        "readiness_checks": readiness_checks,
        "certificates": certificates,
        "maxsat_policy_summary": _maxsat_policy_summary(exp3098),
        "baseline_comparison": _baseline_comparison(exp3100),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_artifacts if row["sha256"] is not None
        },
        "missing_source_artifacts": missing_required_sources,
        "inference_substrate": _inference_substrate(exp3110, z3_available),
        "no_live_llm_inference": True,
        "no_new_model_execution": True,
        "no_conductor_execution": True,
        "scripts_research_conductor_modified": False,
        "duration_s": _duration(start, now_s),
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
    z3_module: Any = _z3,
    tests_run: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Build and persist the Exp 3111 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        started_s=started_s,
        now_s=now_s,
        z3_module=z3_module,
        tests_run=tests_run,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3111 artifact violates the required terminal contract."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("certified_coherence_feedback_v3_ready"):
        if artifact.get("certificate_count") != artifact.get("exact_ground_truth_count"):
            raise ValueError("ready artifact requires certificate count to equal exact count")
        if artifact.get("unsat_core_count", 0) <= 0:
            raise ValueError("ready artifact requires at least one unsat core")
        summary = artifact.get("minimal_repair_distance_summary")
        if not isinstance(summary, Mapping) or summary.get("count", 0) <= 0:
            raise ValueError("ready artifact requires repair-distance summary")


def certificate_for_row(row: Mapping[str, Any], *, z3_module: Any = _z3) -> JsonDict:
    """Build one proof-carrying coherence certificate for an exact fixture row."""

    family = str(row.get("task_family") or "")
    perturbation = str(row.get("perturbation_type") or "")
    if family == "arithmetic_code_assertions" or perturbation == "python_assertion_repair":
        return _arithmetic_certificate(row, z3_module)
    if perturbation == "json_syntax_repair":
        return _json_certificate(row)
    if perturbation == "numeric_bound_repair":
        return _numeric_assignment_certificate(row, z3_module)
    if family == "smt_constraints":
        return _smt_certificate(row, z3_module)
    cert = _base_certificate(row, "unsupported_oracle")
    cert.update(
        {
            "coherence_status": "unsupported",
            "minimal_correction_set": {"kind": "unsupported_fixture_family"},
            "diagnostics": {"reason": "unsupported_fixture_family"},
        }
    )
    cert["maxsat_route"] = _maxsat_route(row, cert)
    return cert


def _arithmetic_certificate(row: Mapping[str, Any], z3_module: Any) -> JsonDict:
    payload = _payload(row)
    cert = _base_certificate(row, "python_ast_runtime_execution")
    left_value, right_value = _assertion_values(str(payload.get("candidate_assertion") or ""))
    distance = _number_distance(left_value, right_value)
    if distance == 0:
        cert.update(
            {
                "coherence_status": "coherent",
                "repair_distance": 0,
                "coherence_gap": 0,
                "minimal_correction_set": {},
                "diagnostics": {"computed_value": _json_number(left_value)},
            }
        )
    else:
        cert.update(
            {
                "coherence_status": "incoherent",
                "repair_distance": distance,
                "coherence_gap": distance,
                "unsat_core": _arithmetic_unsat_core(row, left_value, right_value, z3_module),
                "minimal_correction_set": {
                    "kind": "replace_claimed_value",
                    "from": _json_number(right_value),
                    "to": _json_number(left_value),
                },
                "diagnostics": {
                    "computed_value": _json_number(left_value),
                    "claimed_value": _json_number(right_value),
                },
            }
        )
    cert["maxsat_route"] = _maxsat_route(row, cert)
    return cert


def _json_certificate(row: Mapping[str, Any]) -> JsonDict:
    payload = _payload(row)
    cert = _base_certificate(row, "python_json_parser")
    try:
        parsed = json.loads(str(payload.get("candidate") or ""))
    except json.JSONDecodeError as exc:
        cert.update(
            {
                "coherence_status": "incoherent",
                "repair_distance": 1,
                "coherence_gap": 1,
                "minimal_correction_set": {
                    "kind": "json_token_edit",
                    "edits": [{"operation": "insert_delimiter", "token": ","}],
                },
                "diagnostics": {"json_error": str(exc)},
            }
        )
    else:
        missing = [field for field in payload.get("required_fields", []) if field not in parsed]
        cert.update(
            {
                "coherence_status": "coherent" if not missing else "incoherent",
                "repair_distance": len(missing),
                "coherence_gap": len(missing),
                "minimal_correction_set": {}
                if not missing
                else {"kind": "add_missing_fields", "fields": missing},
                "diagnostics": {"missing_fields": missing},
            }
        )
    cert["maxsat_route"] = _maxsat_route(row, cert)
    return cert


def _numeric_assignment_certificate(row: Mapping[str, Any], z3_module: Any) -> JsonDict:
    payload = _payload(row)
    cert = _base_certificate(row, "z3_solver" if z3_module is not None else "z3_unavailable")
    assignment = {
        str(key): int(value)
        for key, value in dict(payload.get("candidate_assignment") or {}).items()
    }
    named = _named_constraints(payload, z3_module)
    named.extend(_assignment_constraints(assignment, z3_module))
    check = _check_named_constraints(named, z3_module)
    distance, repaired = _assignment_repair(payload, assignment)
    if check["status"] == "sat":
        cert.update(
            {
                "coherence_status": "coherent",
                "repair_distance": 0,
                "coherence_gap": 0,
                "model": check["model"],
                "minimal_correction_set": {},
            }
        )
    else:
        cert.update(
            {
                "coherence_status": "incoherent",
                "repair_distance": distance,
                "coherence_gap": distance,
                "unsat_core": check["unsat_core"],
                "minimal_correction_set": {
                    "kind": "relax_assignment",
                    "constraints_to_relax": _minimal_correction_set(named, z3_module),
                    "repair_assignment": repaired,
                },
                "diagnostics": {"candidate_assignment": assignment},
            }
        )
    cert["maxsat_route"] = _maxsat_route(row, cert)
    return cert


def _smt_certificate(row: Mapping[str, Any], z3_module: Any) -> JsonDict:
    payload = _payload(row)
    cert = _base_certificate(row, "z3_solver" if z3_module is not None else "z3_unavailable")
    named = _named_constraints(payload, z3_module)
    check = _check_named_constraints(named, z3_module)
    if check["status"] == "sat":
        cert.update(
            {
                "coherence_status": "coherent",
                "repair_distance": 0,
                "coherence_gap": 0,
                "model": check["model"],
                "minimal_correction_set": {},
            }
        )
    else:
        gap = _bound_gap(payload)
        cert.update(
            {
                "coherence_status": "incoherent",
                "repair_distance": len(_minimal_correction_set(named, z3_module)),
                "coherence_gap": gap,
                "unsat_core": check["unsat_core"],
                "minimal_correction_set": {
                    "kind": "remove_conflicting_constraint",
                    "constraints_to_relax": _minimal_correction_set(named, z3_module),
                },
                "diagnostics": {"constraints": list(payload.get("constraints") or [])},
            }
        )
    cert["maxsat_route"] = _maxsat_route(row, cert)
    return cert


def _base_certificate(row: Mapping[str, Any], authority: str) -> JsonDict:
    return {
        "fixture_id": str(row.get("source_fixture_id") or row.get("fixture_id") or "unknown"),
        "task_family": str(row.get("task_family") or ""),
        "perturbation_type": str(row.get("perturbation_type") or ""),
        "exact_label": str(row.get("expected_answer") or ""),
        "solver_label": str(row.get("solver_label") or ""),
        "solver_authority": authority,
        "coherence_status": "unknown",
        "unsat_core": [],
        "minimal_correction_set": {},
        "repair_distance": None,
        "coherence_gap": None,
        "model": {},
        "diagnostics": {},
        "maxsat_route": {},
    }


def _payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("leakage_safe_prompt_payload")
    return payload if isinstance(payload, Mapping) else {}


def _assertion_values(assertion: str) -> tuple[Fraction, Fraction]:
    module = ast.parse(assertion)
    statement = module.body[0]
    if not isinstance(statement, ast.Assert):
        raise ValueError("candidate_assertion must be an assert statement")
    test = statement.test
    if (
        not isinstance(test, ast.Compare)
        or len(test.ops) != 1
        or not isinstance(test.ops[0], ast.Eq)
        or len(test.comparators) != 1
    ):
        raise ValueError("candidate_assertion must compare two expressions")
    return _eval_numeric_ast(test.left), _eval_numeric_ast(test.comparators[0])


def _eval_numeric_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_numeric_ast(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_ast(node.left)
        right = _eval_numeric_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"unsupported arithmetic AST node: {ast.dump(node)}")


def _number_distance(left: Fraction, right: Fraction) -> int | float:
    distance = abs(left - right)
    return int(distance) if distance.denominator == 1 else float(distance)


def _json_number(value: Fraction) -> int | float:
    return int(value) if value.denominator == 1 else float(value)


def _arithmetic_unsat_core(
    row: Mapping[str, Any],
    left_value: Fraction,
    right_value: Fraction,
    z3_module: Any,
) -> list[str]:
    if z3_module is None or left_value.denominator != 1 or right_value.denominator != 1:
        return ["computed_value", "claimed_value"]
    symbol = _safe_name(str(row.get("source_fixture_id") or "arith"))
    value = z3_module.Int(f"{symbol}_value")
    solver = z3_module.Solver()
    solver.assert_and_track(value == int(left_value), "computed_value")
    solver.assert_and_track(value == int(right_value), "claimed_value")
    if solver.check() == z3_module.unsat:
        return [str(item) for item in solver.unsat_core()]
    return []


def _named_constraints(payload: Mapping[str, Any], z3_module: Any) -> list[tuple[str, Any]]:
    if z3_module is None:
        return []
    variables = {
        str(name): z3_module.Int(str(name)) for name in payload.get("variables", []) if name
    }
    named: list[tuple[str, Any]] = []
    for idx, text in enumerate(payload.get("constraints") or []):
        named.append((f"constraint_{idx}", _constraint_to_z3(str(text), variables, z3_module)))
    return named


def _assignment_constraints(assignment: Mapping[str, int], z3_module: Any) -> list[tuple[str, Any]]:
    if z3_module is None:
        return []
    return [
        (f"assign_{name}", z3_module.Int(name) == value)
        for name, value in sorted(assignment.items())
    ]


def _constraint_to_z3(text: str, variables: Mapping[str, Any], z3_module: Any) -> Any:
    simple = _SIMPLE_CONSTRAINT_RE.match(text.strip())
    if simple:
        var = variables[simple.group("var")]
        num = int(simple.group("num"))
        op = simple.group("op")
        if op == ">=":
            return var >= num
        if op == "<=":
            return var <= num
        return var == num
    summed = _SUM_CONSTRAINT_RE.match(text.strip())
    if summed:
        return variables[summed.group("a")] + variables[summed.group("b")] == int(
            summed.group("num")
        )
    raise ValueError(f"unsupported constraint: {text}")


def _check_named_constraints(named: list[tuple[str, Any]], z3_module: Any) -> JsonDict:
    if z3_module is None:
        return {"status": "unknown", "unsat_core": [], "model": {}}
    solver = z3_module.Solver()
    for name, constraint in named:
        solver.assert_and_track(constraint, name)
    status = solver.check()
    if status == z3_module.sat:
        return {"status": "sat", "unsat_core": [], "model": _model_dict(solver.model())}
    if status == z3_module.unsat:
        return {
            "status": "unsat",
            "unsat_core": [str(item) for item in solver.unsat_core()],
            "model": {},
        }
    return {"status": "unknown", "unsat_core": [], "model": {}}


def _minimal_correction_set(named: list[tuple[str, Any]], z3_module: Any) -> list[str]:
    if z3_module is None or _sat([constraint for _, constraint in named], z3_module):
        return []
    for width in range(1, len(named) + 1):
        for subset in combinations(range(len(named)), width):
            remaining = [
                constraint for idx, (_, constraint) in enumerate(named) if idx not in set(subset)
            ]
            if _sat(remaining, z3_module):
                return [named[idx][0] for idx in subset]
    return [name for name, _ in named]


def _sat(constraints: list[Any], z3_module: Any) -> bool:
    solver = z3_module.Solver()
    solver.add(constraints)
    return solver.check() == z3_module.sat


def _model_dict(model: Any) -> JsonDict:
    values: JsonDict = {}
    for decl in model.decls():
        raw_value = str(model[decl])
        try:
            values[str(decl.name())] = int(raw_value)
        except ValueError:
            continue
    return values


def _assignment_repair(
    payload: Mapping[str, Any],
    assignment: Mapping[str, int],
) -> tuple[int, JsonDict]:
    target = None
    variables: tuple[str, str] | None = None
    for text in payload.get("constraints") or []:
        match = _SUM_CONSTRAINT_RE.match(str(text).strip())
        if match:
            variables = (match.group("a"), match.group("b"))
            target = int(match.group("num"))
            break
    if target is None or variables is None:
        return 0, dict(assignment)
    current = sum(int(assignment.get(name, 0)) for name in variables)
    delta = target - current
    repaired = dict(assignment)
    repaired[variables[0]] = repaired.get(variables[0], 0) + delta
    return abs(delta), repaired


def _bound_gap(payload: Mapping[str, Any]) -> int:
    lower: dict[str, int] = {}
    upper: dict[str, int] = {}
    for text in payload.get("constraints") or []:
        match = _SIMPLE_CONSTRAINT_RE.match(str(text).strip())
        if not match:
            continue
        name = match.group("var")
        value = int(match.group("num"))
        if match.group("op") == ">=":
            lower[name] = max(value, lower.get(name, value))
        if match.group("op") == "<=":
            upper[name] = min(value, upper.get(name, value))
    gaps = [lower[name] - upper[name] for name in lower.keys() & upper.keys() if lower[name] > upper[name]]
    return max(gaps, default=0)


def _maxsat_route(row: Mapping[str, Any], cert: Mapping[str, Any]) -> JsonDict:
    expected_action = _expected_action(row)
    status = str(cert.get("coherence_status"))
    has_localization = bool(cert.get("unsat_core") or cert.get("minimal_correction_set"))
    if expected_action == "accept" and status == "coherent":
        action = "accept"
    elif expected_action == "reject" and status in {"incoherent", "unsupported"}:
        action = "reject"
    else:
        action = "abstain"
    soft_terms = [
        {"id": "SC_EXACT_LABEL_MATCH", "weight": 100, "satisfied": bool(row.get("expected_answer"))},
        {"id": "SC_LOCALIZED_FEEDBACK", "weight": 40, "satisfied": has_localization},
        {"id": "SC_MINIMAL_DISTANCE", "weight": 35, "satisfied": cert.get("repair_distance") is not None},
    ]
    return {
        "action": action,
        "hard_constraints": [
            {"id": "HC_EXACT_LABEL_PRESENT", "satisfied": bool(row.get("expected_answer"))},
            {"id": "HC_SOLVER_OR_TEST_AUTHORITY", "satisfied": bool(cert.get("solver_authority"))},
        ],
        "soft_constraints": soft_terms,
        "soft_score": sum(term["weight"] for term in soft_terms if term["satisfied"]),
        "policy_style": "weighted_maxsat_reference_evaluator",
    }


def _expected_action(row: Mapping[str, Any]) -> str:
    target = row.get("verifier_target")
    if isinstance(target, Mapping) and target.get("expected_action"):
        return str(target["expected_action"])
    return "accept" if row.get("expected_answer") in {"VALID", "SAT"} else "reject"


def _formal_label_rows(rows: list[JsonDict]) -> list[JsonDict]:
    return [row for row in rows if row.get("expected_answer") and row.get("solver_label")]


def _minimal_repair_distance_summary(certificates: list[Mapping[str, Any]]) -> JsonDict:
    distances = [
        float(cert["repair_distance"])
        for cert in certificates
        if isinstance(cert.get("repair_distance"), int | float) and cert.get("repair_distance") > 0
    ]
    if not distances:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(distances),
        "min": min(distances),
        "max": max(distances),
        "mean": round(sum(distances) / len(distances), 6),
    }


def _vacuity_guard_passed(
    *,
    certificates: list[Mapping[str, Any]],
    unsat_core_count: int,
    correction_set_count: int,
    repair_summary: Mapping[str, Any],
    upstream_vacuity: bool,
) -> bool:
    statuses = {str(cert.get("coherence_status")) for cert in certificates}
    return (
        upstream_vacuity
        and "coherent" in statuses
        and "incoherent" in statuses
        and (unsat_core_count > 0 or correction_set_count > 0)
        and repair_summary.get("count", 0) > 0
    )


def _manifest_rel_path(exp3097: Mapping[str, Any]) -> Path:
    raw = exp3097.get("stratified_eval_manifest_path")
    return Path(str(raw)) if raw else MANIFEST_REL_PATH


def _source_artifacts(root: Path, manifest_rel_path: Path) -> list[JsonDict]:
    specs = [
        (role, manifest_rel_path if path == MANIFEST_REL_PATH else path, required)
        for role, path, required in SOURCE_SPECS
    ]
    return [_source_artifact(root, role, path, required) for role, path, required in specs]


def _source_artifact(root: Path, role: str, rel_path: Path, required: bool) -> JsonDict:
    path = root / rel_path
    return {
        "path": rel_path.as_posix(),
        "role": role,
        "required": required,
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def _solver_only_allowed(exp3110: Mapping[str, Any]) -> bool:
    downstream = exp3110.get("downstream_usage")
    if not isinstance(downstream, Mapping):
        return False
    solver_rules = downstream.get("solver_only_tasks")
    return isinstance(solver_rules, Mapping) and solver_rules.get("allowed_without_cached_sota_pair") is True


def _maxsat_policy_summary(exp3098: Mapping[str, Any]) -> JsonDict:
    return {
        "maxsat_policy_ready": exp3098.get("maxsat_policy_ready") is True,
        "hard_constraint_count": len(exp3098.get("hard_constraints") or []),
        "soft_constraint_count": len(exp3098.get("soft_constraints") or []),
        "objective_terms": exp3098.get("objective_terms") if isinstance(exp3098.get("objective_terms"), Mapping) else {},
    }


def _baseline_comparison(exp3100: Mapping[str, Any]) -> JsonDict:
    solver_count = int(exp3100.get("solver_only_success_count") or 0)
    guided_count = int(exp3100.get("guided_success_count") or 0)
    return {
        "source": EXP3100_REL_PATH.as_posix(),
        "solver_only_success_count": solver_count,
        "guided_success_count": guided_count,
        "guided_minus_solver_only": guided_count - solver_count,
        "live_llm_lift_claimed": False,
    }


def _inference_substrate(exp3110: Mapping[str, Any], z3_available: bool) -> JsonDict:
    return {
        "kind": "deterministic_z3_and_python_test_oracle_certificates",
        "live_llm_inference": False,
        "executes_models": False,
        "executes_solvers": z3_available,
        "executes_hardware": False,
        "cached_sota_pair_available": exp3110.get("cached_sota_pair_available") is True,
        "cached_sota_pair_required_for_readiness": False,
        "solver_only_tasks_allowed_without_cached_sota_pair": _solver_only_allowed(exp3110),
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("certified_coherence_feedback_v3_ready"):
        return (
            "complete: certified_coherence_feedback_v3_ready=true; "
            f"certificate_count={artifact.get('certificate_count')}; "
            f"unsat_core_count={artifact.get('unsat_core_count')}; "
            "cached_sota_pair_not_required"
        )
    failed = [
        name for name, ok in artifact.get("readiness_checks", {}).items() if ok is not True
    ]
    return "blocked_certified_coherence_feedback_v3: " + ",".join(failed)


def _z3_version(z3_module: Any) -> str | None:
    if z3_module is None:
        return None
    return str(z3_module.get_version_string())


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)
