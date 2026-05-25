"""Build the Exp 3084 ReSyn exact fixture bank.

Spec refs: REQ-VERIFY-3084, SCENARIO-VERIFY-3084.

The bank is deliberately small, exact, and CPU-only. The rows are meant for
later verifier, abstention, and repair tasks that need labels from local
authorities rather than model judgment. Z3 handles SMT satisfiability,
bounded Python arithmetic handles assertion truth, and local parser/runtime
checks handle repairable-invalid candidates.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

try:  # pragma: no cover - absence is covered through z3_module=None.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
SCHEMA = "carnot.resyn_exact_fixture_bank.v1"
FIXTURE_SCHEMA = "carnot.resyn_exact_fixture.v1"
OUTPUT_REL_PATH = Path("results/experiment_3084_resyn_exact_fixture_bank_generator_v1.json")
MANIFEST_REL_PATH = Path("results/resyn_exact_fixture_bank_3084/fixture_manifest.jsonl")
EXP3070_REL_PATH = Path("results/experiment_3070_first_token_abstention_sota_panel_v1.json")
EXP3083_REL_PATH = Path("results/experiment_3083_verifier_hardness_autopsy_protocol_v1.json")

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "resyn_fixture_bank_ready",
    "exact_fixture_count",
    "family_count",
    "fixture_manifest_path",
    "exact_label_sources",
    "perturbation_families",
    "tests_added_or_reused",
    "preconditions_checked",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
REQUIRED_FIXTURE_FIELDS: tuple[str, ...] = (
    "fixture_id",
    "family",
    "task_axis",
    "perturbation_family",
    "leakage_safe_prompt_payload",
    "prompt_payload_sha256",
    "exact_label",
    "label_source",
)
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
SOURCE_REL_PATHS: tuple[tuple[str, Path, str], ...] = (
    ("exp3070", EXP3070_REL_PATH, "first-token abstention exact-label pressure"),
    ("exp3083", EXP3083_REL_PATH, "verifier-hardness recovery protocol"),
    ("codex", Path("CODEX.md"), "repo spec-first and tests-first workflow"),
    ("research_references", Path("research-references.md"), "ReSyn/verifier context"),
)


@dataclass(frozen=True)
class FixtureBankConfig:
    """Runtime paths and clock injection for the deterministic generator."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_path: Path | None = None
    started_s: float | None = None
    clock: ClockFn = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_s is None else float(self.started_s)

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / MANIFEST_REL_PATH


def write_artifact(
    config: FixtureBankConfig | None = None,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Generate fixtures, write the JSONL manifest, and write the terminal artifact."""

    active = config or FixtureBankConfig()
    started_s = active.start_time()
    output_path = active.resolved_output_path()
    manifest_path = active.resolved_manifest_path()
    if z3_module is None:
        artifact = _blocked_artifact(active, started_s)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact

    rows = generate_fixture_rows(z3_module=z3_module)
    validate_fixture_rows(rows, z3_module=z3_module)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    artifact = _complete_artifact(
        active=active,
        started_s=started_s,
        rows=rows,
        manifest_path=manifest_path,
        z3_module=z3_module,
    )
    validate_artifact(artifact, rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def generate_fixture_rows(*, z3_module: Any = _z3) -> list[JsonDict]:
    """Return the deterministic 72-row exact fixture bank."""

    if z3_module is None:
        raise ValueError("z3_module is required for SMT fixtures")
    rows: list[JsonDict] = []
    rows.extend(_smt_fixture_rows(z3_module))
    rows.extend(_arithmetic_fixture_rows())
    rows.extend(_repair_fixture_rows(z3_module))
    return rows


def validate_fixture_rows(rows: Sequence[Mapping[str, Any]], *, z3_module: Any = _z3) -> None:
    """Re-run local authorities and reject fixture rows whose labels drift."""

    if z3_module is None:
        raise ValueError("z3_module is required for fixture validation")
    for row in rows:
        missing = sorted(set(REQUIRED_FIXTURE_FIELDS) - set(row))
        if missing:
            raise ValueError(f"fixture {row.get('fixture_id')} missing required fields: {missing}")
        payload = row["leakage_safe_prompt_payload"]
        if row["prompt_payload_sha256"] != hash_prompt_payload(payload):
            raise ValueError(f"prompt hash mismatch for {row['fixture_id']}")
        prompt_text = _canonical_json(payload).lower()
        if "answer" in prompt_text:
            raise ValueError(f"prompt payload leaks answer field for {row['fixture_id']}")
        family = str(row["family"])
        if family == "smt_constraints":
            _validate_smt_row(row, z3_module)
        elif family == "arithmetic_code_assertions":
            _validate_arithmetic_row(row)
        elif family == "repairable_invalid_candidates":
            _validate_repair_row(row, z3_module)
        else:
            raise ValueError(f"unknown fixture family: {family}")


def validate_artifact(
    artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None = None
) -> None:
    """Reject terminal artifacts that overstate fixture-bank readiness."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("no_live_llm_inference") is not True:
        raise ValueError("inference_substrate must declare no live LLM inference")
    if substrate.get("llm_used_for_labels") is not False:
        raise ValueError("inference_substrate must declare LLM labels were not used")
    if artifact.get("resyn_fixture_bank_ready") is not True:
        verdict = str(artifact.get("honest_verdict", ""))
        if not verdict.startswith("blocked_exact_label_tooling_missing"):
            raise ValueError("blocked artifact must disclose exact-label tooling failure")
        return
    if int(artifact.get("exact_fixture_count", 0)) < 64:
        raise ValueError("ready fixture bank requires at least 64 exact fixtures")
    if int(artifact.get("family_count", 0)) < 3:
        raise ValueError("ready fixture bank requires at least three families")
    if rows is not None and int(artifact["exact_fixture_count"]) != len(rows):
        raise ValueError("exact_fixture_count must equal manifest row count")
    if not str(artifact.get("honest_verdict", "")).startswith(SUCCESS_PREFIXES):
        raise ValueError("ready artifact honest_verdict must use a terminal success prefix")


def safe_eval_arithmetic(expression: str) -> int:
    """Evaluate a tiny integer arithmetic expression without Python builtins."""

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("unsupported arithmetic expression") from exc
    return _eval_arithmetic_node(parsed.body)


def hash_prompt_payload(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for the leakage-safe prompt payload."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum for a written artifact or manifest file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smt_fixture_rows(z3_module: Any) -> list[JsonDict]:
    rows = []
    for index in range(24):
        if index % 2 == 0:
            authority = {
                "variables": [f"x_{index}", f"y_{index}"],
                "constraints": [
                    {"op": "ge", "var": f"x_{index}", "value": index + 2},
                    {"op": "le", "var": f"x_{index}", "value": index + 1},
                    {"op": "ge", "var": f"y_{index}", "value": 0},
                ],
            }
            perturbation = "smt_unsat_abstention"
            task_axis = "abstaining"
        else:
            lower = index % 5
            upper = lower + 6
            authority = {
                "variables": [f"x_{index}", f"y_{index}"],
                "constraints": [
                    {"op": "ge", "var": f"x_{index}", "value": lower},
                    {"op": "le", "var": f"x_{index}", "value": upper},
                    {"op": "ge", "var": f"y_{index}", "value": 0},
                    {"op": "le", "var": f"y_{index}", "value": upper},
                    {"op": "eq_sum", "vars": [f"x_{index}", f"y_{index}"], "value": lower + 3},
                ],
            }
            perturbation = "smt_sat_solving"
            task_axis = "solving"
        status = _smt_status(authority, z3_module)
        prompt_payload = {
            "task": "Classify the integer constraints as SAT or UNSAT.",
            "variables": authority["variables"],
            "constraints": _constraint_strings(authority["constraints"]),
            "response_schema": {"verdict": "SAT_OR_UNSAT"},
        }
        rows.append(
            _fixture_row(
                fixture_id=f"resyn-3084-smt-{index:03d}",
                family="smt_constraints",
                task_axis=task_axis,
                perturbation_family=perturbation,
                prompt_payload=prompt_payload,
                exact_label={
                    "kind": "smt_satisfiability",
                    "solver_status": status,
                    "is_satisfiable": status == "sat",
                },
                label_source="z3_solver",
                authority_payload=authority,
            )
        )
    return rows


def _arithmetic_fixture_rows() -> list[JsonDict]:
    rows = []
    for index in range(24):
        a = index + 3
        b = index % 7 + 2
        c = index % 5 + 1
        d = index // 3
        expression = f"(({a} + {b}) * {c}) - {d}"
        computed = safe_eval_arithmetic(expression)
        claim = computed if index % 2 == 0 else computed + (index % 4 + 1)
        assertion_passes = computed == claim
        prompt_payload = {
            "task": "Classify the candidate arithmetic assertion.",
            "expression": expression,
            "candidate_assertion": f"assert ({expression}) == {claim}",
            "response_schema": {"verdict": "VALID_OR_INVALID"},
        }
        rows.append(
            _fixture_row(
                fixture_id=f"resyn-3084-arith-{index:03d}",
                family="arithmetic_code_assertions",
                task_axis="verifying",
                perturbation_family=(
                    "arithmetic_true_verification"
                    if assertion_passes
                    else "arithmetic_false_verification"
                ),
                prompt_payload=prompt_payload,
                exact_label={
                    "kind": "arithmetic_assertion",
                    "assertion_passes": assertion_passes,
                    "computed_value": computed,
                    "claimed_value": claim,
                },
                label_source="python_ast_runtime_execution",
                authority_payload={"expression": expression, "claimed_value": claim},
            )
        )
    return rows


def _repair_fixture_rows(z3_module: Any) -> list[JsonDict]:
    rows = []
    for index in range(24):
        if index % 3 == 0:
            rows.append(_json_repair_fixture(index))
        elif index % 3 == 1:
            rows.append(_numeric_repair_fixture(index, z3_module))
        else:
            rows.append(_python_assertion_repair_fixture(index))
    return rows


def _json_repair_fixture(index: int) -> JsonDict:
    limit = index + 2
    candidate = f'{{"mode": "bounded" "limit": {limit}}}'
    repair = json.dumps({"mode": "bounded", "limit": limit}, sort_keys=True)
    exact_label = _json_repair_label(candidate, repair)
    return _fixture_row(
        fixture_id=f"resyn-3084-repair-json-{index:03d}",
        family="repairable_invalid_candidates",
        task_axis="repairing",
        perturbation_family="json_syntax_repair",
        prompt_payload={
            "task": "Repair the candidate so the object parses and preserves fields.",
            "candidate": candidate,
            "required_fields": ["mode", "limit"],
        },
        exact_label=exact_label,
        label_source="json_parser",
        authority_payload={"repair_kind": "json", "candidate": candidate, "repair": repair},
    )


def _numeric_repair_fixture(index: int, z3_module: Any) -> JsonDict:
    x_name = f"rx_{index}"
    y_name = f"ry_{index}"
    target = index + 10
    candidate = {x_name: index + 1, y_name: 1}
    repair = {x_name: index + 4, y_name: target - (index + 4)}
    authority = {
        "variables": [x_name, y_name],
        "constraints": [
            {"op": "ge", "var": x_name, "value": 0},
            {"op": "ge", "var": y_name, "value": 0},
            {"op": "eq_sum", "vars": [x_name, y_name], "value": target},
        ],
        "candidate": candidate,
        "repair": repair,
    }
    exact_label = _numeric_repair_label(authority, z3_module)
    return _fixture_row(
        fixture_id=f"resyn-3084-repair-smt-{index:03d}",
        family="repairable_invalid_candidates",
        task_axis="repairing",
        perturbation_family="numeric_bound_repair",
        prompt_payload={
            "task": "Repair the candidate integer assignment while preserving variable names.",
            "variables": authority["variables"],
            "constraints": _constraint_strings(authority["constraints"]),
            "candidate_assignment": candidate,
        },
        exact_label=exact_label,
        label_source="z3_solver",
        authority_payload=authority,
    )


def _python_assertion_repair_fixture(index: int) -> JsonDict:
    expression = f"({index + 5} * 2) - {index % 3}"
    computed = safe_eval_arithmetic(expression)
    candidate_claim = computed + 1
    repair_claim = computed
    authority = {
        "repair_kind": "python_assertion",
        "expression": expression,
        "candidate_claim": candidate_claim,
        "repair_claim": repair_claim,
    }
    exact_label = _python_assertion_repair_label(authority)
    return _fixture_row(
        fixture_id=f"resyn-3084-repair-py-{index:03d}",
        family="repairable_invalid_candidates",
        task_axis="repairing",
        perturbation_family="python_assertion_repair",
        prompt_payload={
            "task": "Repair the candidate assertion while keeping the expression fixed.",
            "expression": expression,
            "candidate_assertion": f"assert ({expression}) == {candidate_claim}",
        },
        exact_label=exact_label,
        label_source="python_ast_runtime_execution",
        authority_payload=authority,
    )


def _fixture_row(
    *,
    fixture_id: str,
    family: str,
    task_axis: str,
    perturbation_family: str,
    prompt_payload: JsonDict,
    exact_label: JsonDict,
    label_source: str,
    authority_payload: JsonDict,
) -> JsonDict:
    return {
        "schema": FIXTURE_SCHEMA,
        "fixture_id": fixture_id,
        "family": family,
        "task_axis": task_axis,
        "perturbation_family": perturbation_family,
        "leakage_safe_prompt_payload": prompt_payload,
        "prompt_payload_sha256": hash_prompt_payload(prompt_payload),
        "exact_label": exact_label,
        "label_source": label_source,
        "authority_payload": authority_payload,
    }


def _validate_smt_row(row: Mapping[str, Any], z3_module: Any) -> None:
    status = _smt_status(row["authority_payload"], z3_module)
    if status != row["exact_label"].get("solver_status"):
        raise ValueError(f"SMT label mismatch for {row['fixture_id']}")


def _validate_arithmetic_row(row: Mapping[str, Any]) -> None:
    authority = row["authority_payload"]
    assertion_passes = _arithmetic_assertion_passes(
        str(authority["expression"]),
        int(authority["claimed_value"]),
    )
    if assertion_passes != row["exact_label"].get("assertion_passes"):
        raise ValueError(f"arithmetic assertion label mismatch for {row['fixture_id']}")


def _validate_repair_row(row: Mapping[str, Any], z3_module: Any) -> None:
    authority = row["authority_payload"]
    label_source = str(row["label_source"])
    if label_source == "json_parser":
        exact_label = _json_repair_label(str(authority["candidate"]), str(authority["repair"]))
    elif label_source == "z3_solver":
        exact_label = _numeric_repair_label(authority, z3_module)
    elif label_source == "python_ast_runtime_execution":
        exact_label = _python_assertion_repair_label(authority)
    else:
        raise ValueError(f"unknown repair label source: {label_source}")
    if (
        exact_label["candidate_valid"] != row["exact_label"].get("candidate_valid")
        or exact_label["repairable"] != row["exact_label"].get("repairable")
        or exact_label["failure_kind"] != row["exact_label"].get("failure_kind")
    ):
        raise ValueError(f"repair fixture label mismatch for {row['fixture_id']}")


def _json_repair_label(candidate: str, repair: str) -> JsonDict:
    candidate_valid = _json_loads_ok(candidate)
    repair_valid = _json_loads_ok(repair)
    return {
        "kind": "repairability",
        "candidate_valid": candidate_valid,
        "repairable": (not candidate_valid) and repair_valid,
        "failure_kind": "json_decode_error" if not candidate_valid else "none",
        "repair_validation": "passed" if repair_valid else "failed",
    }


def _numeric_repair_label(authority: Mapping[str, Any], z3_module: Any) -> JsonDict:
    candidate_valid = _assignment_satisfies(authority, authority["candidate"], z3_module)
    repair_valid = _assignment_satisfies(authority, authority["repair"], z3_module)
    return {
        "kind": "repairability",
        "candidate_valid": candidate_valid,
        "repairable": (not candidate_valid) and repair_valid,
        "failure_kind": "constraint_violation" if not candidate_valid else "none",
        "repair_validation": "passed" if repair_valid else "failed",
    }


def _python_assertion_repair_label(authority: Mapping[str, Any]) -> JsonDict:
    candidate_valid = _arithmetic_assertion_passes(
        str(authority["expression"]),
        int(authority["candidate_claim"]),
    )
    repair_valid = _arithmetic_assertion_passes(
        str(authority["expression"]),
        int(authority["repair_claim"]),
    )
    return {
        "kind": "repairability",
        "candidate_valid": candidate_valid,
        "repairable": (not candidate_valid) and repair_valid,
        "failure_kind": "assertion_failure" if not candidate_valid else "none",
        "repair_validation": "passed" if repair_valid else "failed",
    }


def _smt_status(authority: Mapping[str, Any], z3_module: Any) -> str:
    solver = z3_module.Solver()
    variables = {name: z3_module.Int(name) for name in authority["variables"]}
    for constraint in authority["constraints"]:
        solver.add(_z3_constraint(constraint, variables))
    return str(solver.check())


def _assignment_satisfies(
    authority: Mapping[str, Any],
    assignment: Mapping[str, int],
    z3_module: Any,
) -> bool:
    solver = z3_module.Solver()
    variables = {name: z3_module.Int(name) for name in authority["variables"]}
    for constraint in authority["constraints"]:
        solver.add(_z3_constraint(constraint, variables))
    for name, value in assignment.items():
        solver.add(variables[name] == int(value))
    return str(solver.check()) == "sat"


def _z3_constraint(constraint: Mapping[str, Any], variables: Mapping[str, Any]) -> Any:
    op = str(constraint["op"])
    if op == "ge":
        return variables[str(constraint["var"])] >= int(constraint["value"])
    if op == "le":
        return variables[str(constraint["var"])] <= int(constraint["value"])
    if op == "eq_sum":
        left, right = [variables[str(name)] for name in constraint["vars"]]
        return left + right == int(constraint["value"])
    raise ValueError(f"unsupported SMT constraint op: {op}")


def _constraint_strings(constraints: Sequence[Mapping[str, Any]]) -> list[str]:
    rendered = []
    for constraint in constraints:
        op = str(constraint["op"])
        if op == "ge":
            rendered.append(f"{constraint['var']} >= {constraint['value']}")
        elif op == "le":
            rendered.append(f"{constraint['var']} <= {constraint['value']}")
        elif op == "eq_sum":
            left, right = constraint["vars"]
            rendered.append(f"{left} + {right} == {constraint['value']}")
        else:
            raise ValueError(f"unsupported SMT constraint op: {op}")
    return rendered


def _arithmetic_assertion_passes(expression: str, claimed_value: int) -> bool:
    return safe_eval_arithmetic(expression) == claimed_value


def _eval_arithmetic_node(node: ast.AST) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise ValueError("unsupported arithmetic expression")


def _json_loads_ok(payload: str) -> bool:
    try:
        json.loads(payload)
    except json.JSONDecodeError:
        return False
    return True


def _complete_artifact(
    *,
    active: FixtureBankConfig,
    started_s: float,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    z3_module: Any,
) -> JsonDict:
    family_counts = dict(sorted(Counter(str(row["family"]) for row in rows).items()))
    label_sources = sorted({str(row["label_source"]) for row in rows})
    perturbations = sorted({str(row["perturbation_family"]) for row in rows})
    task_axes = sorted({str(row["task_axis"]) for row in rows})
    manifest_rel = _relative_path(active.repo_root, manifest_path)
    ready = (
        manifest_path.is_file()
        and len(rows) >= 64
        and len(family_counts) >= 3
        and len(label_sources) >= 3
        and {"solving", "verifying", "abstaining", "repairing"} <= set(task_axes)
    )
    return {
        "schema": SCHEMA,
        "artifact": "experiment_3084_resyn_exact_fixture_bank_generator_v1",
        "run_date": RUN_DATE,
        "resyn_fixture_bank_ready": ready,
        "exact_fixture_count": len(rows),
        "family_count": len(family_counts),
        "family_counts": family_counts,
        "fixture_manifest_path": manifest_rel,
        "fixture_manifest_sha256": sha256_file(manifest_path),
        "exact_label_sources": label_sources,
        "perturbation_families": perturbations,
        "task_axes": task_axes,
        "tests_added_or_reused": [
            "tests/python/test_experiment_3084_resyn_exact_fixture_bank_generator.py"
        ],
        "preconditions_checked": _preconditions_checked(z3_module=z3_module),
        "source_artifacts": _source_artifacts(active.repo_root),
        "inference_substrate": _inference_substrate(z3_available=True),
        "duration_s": active.clock() - started_s,
        "honest_verdict": (
            "complete: resyn_fixture_bank_ready=true; exact_fixture_count=72; family_count=3; no_llm_labels"
            if ready
            else "blocked_exact_label_tooling_missing: generated fixture bank failed readiness gates"
        ),
    }


def _blocked_artifact(active: FixtureBankConfig, started_s: float) -> JsonDict:
    return {
        "schema": SCHEMA,
        "artifact": "experiment_3084_resyn_exact_fixture_bank_generator_v1",
        "run_date": RUN_DATE,
        "resyn_fixture_bank_ready": False,
        "exact_fixture_count": 0,
        "family_count": 0,
        "family_counts": {},
        "fixture_manifest_path": _relative_path(active.repo_root, active.resolved_manifest_path()),
        "fixture_manifest_sha256": None,
        "exact_label_sources": [],
        "perturbation_families": [],
        "task_axes": [],
        "tests_added_or_reused": [
            "tests/python/test_experiment_3084_resyn_exact_fixture_bank_generator.py"
        ],
        "preconditions_checked": _preconditions_checked(z3_module=None),
        "source_artifacts": _source_artifacts(active.repo_root),
        "inference_substrate": _inference_substrate(z3_available=False),
        "duration_s": active.clock() - started_s,
        "honest_verdict": "blocked_exact_label_tooling_missing: z3 unavailable for SMT fixture authority",
    }


def _preconditions_checked(*, z3_module: Any) -> JsonDict:
    z3_ok = z3_module is not None
    return {
        "z3_import": {
            "ok": z3_ok,
            "detail": _z3_version(z3_module) if z3_ok else "z3_module_unavailable",
        },
        "python_runtime_validation": {"ok": True, "detail": sys.executable},
        "json_parser_validation": {"ok": True, "detail": "stdlib json"},
        "llm_labeling": {"ok": True, "detail": "not used"},
    }


def _inference_substrate(*, z3_available: bool) -> JsonDict:
    return {
        "kind": "deterministic_z3_python_json_fixture_generation",
        "z3_available": z3_available,
        "executes_z3": z3_available,
        "executes_python_runtime": True,
        "executes_json_parser": True,
        "executes_models": False,
        "live_llm_calls": 0,
        "no_live_llm_inference": True,
        "llm_used_for_labels": False,
    }


def _source_artifacts(repo_root: Path) -> list[JsonDict]:
    rows = []
    for source_id, rel_path, role in SOURCE_REL_PATHS:
        path = repo_root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "role": role,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _z3_version(z3_module: Any) -> str:
    getter = getattr(z3_module, "get_version_string", None)
    return str(getter()) if callable(getter) else "unknown"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def main() -> None:  # pragma: no cover - thin manual entrypoint.
    write_artifact()


if __name__ == "__main__":  # pragma: no cover
    main()
