"""Experiment 4865: hostile audit of the Exp 4861 A1 fork probe.

Spec refs: REQ-ARC-WMTE-4865,
SCENARIO-ARC-WMTE-4865-A1-FORK-AUDIT,
SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from statistics import median
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):  # pragma: no cover - direct script guard.
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

EXPERIMENT = "experiment_4865_fork_probe_audit"
EXPERIMENT_ID = 4865
SCHEMA = "carnot.arc.a1_fork_probe_audit_4865.v1"
SOURCE_ARTIFACT_RELATIVE_PATH = "results/experiment_4861_generation_wall_fork_probe.json"
SOURCE_SCRIPT_RELATIVE_PATH = "python/carnot/experiment_4861_generation_wall_fork_probe.py"
RESULT_RELATIVE_PATH = "results/experiment_4865_fork_probe_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DURATION_FLOOR_S = 0.0001
RANDOM_SEED = 4865
HIGH_ACCURACY_THRESHOLD = 0.5
TERMINAL_PREFIXES = ("complete_", "blocked_", "success_")
BUCKETS = ("COVERED", "ENUMERATED_BUT_LOST", "NEVER_ENUMERATED")
FORK_VERDICTS = ("GUIDANCE_WALL", "PLANNER_GAP", "INDUCER_CEILING")

SPEC_REFS = [
    "REQ-ARC-WMTE-4865",
    "SCENARIO-ARC-WMTE-4865-A1-FORK-AUDIT",
    "SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_a1_fork_probe_audited."
    },
    "a1_genuinely_diagnostic": {
        "principle": (
            "the load-bearing check -- planner-blind AND positive-control-migrated AND "
            "numbers-match-fork AND live-path-reachable; else A1 is a tautology/harness "
            "non-test and its fork verdict is void."
        )
    },
    "planner_blind_confirmed": {
        "principle": (
            "true iff the banked winner was used only to classify, never to seed "
            "induction/planning (the tautology trap)."
        )
    },
    "positive_control_confirmed": {
        "principle": "true iff tu93 really came out HIGH accuracy + COVERED."
    },
    "numbers_match_fork": {
        "principle": (
            "true iff the per-game accuracy x bucket numbers support the claimed fork_verdict."
        )
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts (0.0001s floor)."},
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "source_artifact_path",
    "source_script_path",
    "source_artifact_checksum",
    "source_script_checksum",
    "source_honest_verdict",
    "source_fork_verdict",
    "source_n_games_measured",
    "source_flagged_adversarial",
    "field_principles",
    "live_path_reachable_confirmed",
    "solve_provenance_confirmed",
    "checks",
    "non_diagnostic_reasons",
    "summarizer_result",
    "adversarial_result",
    "live_lint_result",
    "preconditions_checked",
    "audit_report_path",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

JsonDict = dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _safe_suffix(reasons: list[str]) -> str:
    if not reasons:
        return "audited"
    joined = "_".join(reasons[:3])
    return re.sub(r"[^a-z0-9_]+", "_", joined.lower()).strip("_") or "failed_checks"


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean["reproducibility_checksum"] = ""
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_summarizer(path: Path) -> JsonDict:
    from scripts import summarize_artifact

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        returncode = summarize_artifact.summarize(path)
    return {"returncode": int(returncode), "stdout": buffer.getvalue(), "stderr": ""}


def run_adversarial_verify(path: Path) -> JsonDict:
    from scripts import adversarial_verify

    return dict(adversarial_verify.verify_artifact(path))


def run_arc_orphan_solver_lint(root: Path) -> JsonDict:
    command = [sys.executable, str(root / "scripts" / "arc_orphan_solver_lint.py")]
    proc = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def _call_name(callable_node: ast.AST) -> str:
    if isinstance(callable_node, ast.Name):
        return callable_node.id
    if isinstance(callable_node, ast.Attribute):
        return callable_node.attr
    return ""


def _full_call_name(callable_node: ast.AST) -> str:
    if isinstance(callable_node, ast.Name):
        return callable_node.id
    if isinstance(callable_node, ast.Attribute):
        prefix = _full_call_name(callable_node.value)
        return f"{prefix}.{callable_node.attr}" if prefix else callable_node.attr
    return ""


def _attach_ast_parents(node: ast.AST) -> None:
    for parent in ast.walk(node):
        for child in ast.iter_child_nodes(parent):
            child.__dict__["_parent"] = parent


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _first_parent_call(node: ast.AST) -> ast.Call | None:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.Call):
            return parent
        parent = getattr(parent, "_parent", None)
    return None


def _source_checks(
    artifact: Mapping[str, Any],
    source_text: str,
) -> tuple[JsonDict, list[str], JsonDict, list[str]]:
    planner_reasons: list[str] = []
    live_reasons: list[str] = []
    if artifact.get("planner_blind_to_banked_answer") is not True:
        planner_reasons.append("artifact_planner_blind_flag_false")
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        detail = {
            "passed": False,
            "artifact_flag": artifact.get("planner_blind_to_banked_answer"),
            "parse_error": str(exc),
            "winning_prefix_refs": 0,
            "disallowed_refs": [],
        }
        live_detail = {"passed": False, "parse_error": str(exc), "calls": []}
        return (
            detail,
            planner_reasons + ["a1_source_not_parseable"],
            live_detail,
            ["source_live_path_uninspectable"],
        )

    _attach_ast_parents(tree)
    function = _find_function(tree, "measure_game_with_live_induce_plan")
    if function is None:
        detail = {
            "passed": False,
            "artifact_flag": artifact.get("planner_blind_to_banked_answer"),
            "function_present": False,
            "winning_prefix_refs": 0,
            "disallowed_refs": [],
        }
        live_detail = {"passed": False, "function_present": False, "calls": []}
        return (
            detail,
            planner_reasons + ["measure_game_with_live_induce_plan_missing"],
            live_detail,
            ["measure_game_with_live_induce_plan_missing"],
        )

    refs: list[JsonDict] = []
    disallowed: list[JsonDict] = []
    calls: list[str] = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call):
                calls.append(_full_call_name(node.func))
            if isinstance(node, ast.Name) and node.id == "winning_prefix":
                call = _first_parent_call(node)
                call_name = _call_name(call.func) if call is not None else ""
                record = {"line": int(getattr(node, "lineno", 0)), "call": call_name or None}
                refs.append(record)
                if call_name != "classify_planned_pool":
                    disallowed.append(record)

    if not refs:
        planner_reasons.append("winning_prefix_not_used_for_classification")
    if disallowed:
        planner_reasons.append("banked_answer_used_before_classification")

    has_induce = any(name.endswith("._induce_and_plan") for name in calls)
    has_load = any(name.endswith(".load_engine") or name == "load_engine" for name in calls)
    has_plan = any(name.endswith(".plan_in_model") or name == "plan_in_model" for name in calls)
    if not has_induce:
        live_reasons.append("induce_and_plan_not_called")
    if not has_load:
        live_reasons.append("load_engine_not_called")
    if not has_plan:
        live_reasons.append("plan_in_model_not_called")

    return (
        {
            "passed": not planner_reasons,
            "artifact_flag": artifact.get("planner_blind_to_banked_answer"),
            "function_present": True,
            "winning_prefix_refs": len(refs),
            "disallowed_refs": disallowed,
            "allowed_only_in_classify_planned_pool": not disallowed and bool(refs),
        },
        planner_reasons,
        {
            "passed": not live_reasons,
            "function_present": True,
            "calls": calls,
            "induce_and_plan_called": has_induce,
            "load_engine_called": has_load,
            "plan_in_model_called": has_plan,
        },
        live_reasons,
    )


def _positive_control_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    row = _mapping(artifact.get("positive_control_fork"))
    reasons: list[str] = []
    if artifact.get("positive_control_migrated") is not True:
        reasons.append("positive_control_not_migrated")
    if not row:
        reasons.append("positive_control_row_missing")
    game = str(row.get("game") or artifact.get("positive_control_game") or "")
    if game != "tu93" or artifact.get("positive_control_game") != "tu93":
        reasons.append("positive_control_not_tu93")
    if row.get("planned_bucket") != "COVERED":
        reasons.append("positive_control_not_covered")
    accuracy = _finite_float(row.get("engine_heldout_accuracy"))
    if accuracy is None or accuracy < HIGH_ACCURACY_THRESHOLD:
        reasons.append("positive_control_low_accuracy")
    return (
        {
            "passed": not reasons,
            "positive_control_game": artifact.get("positive_control_game"),
            "artifact_positive_control_migrated": artifact.get("positive_control_migrated"),
            "row_game": row.get("game"),
            "planned_bucket": row.get("planned_bucket"),
            "engine_heldout_accuracy": row.get("engine_heldout_accuracy"),
            "high_accuracy_threshold": HIGH_ACCURACY_THRESHOLD,
        },
        reasons,
    )


def _valid_row_values(row: Mapping[str, Any]) -> tuple[str | None, float | None, bool | None]:
    bucket = str(row.get("planned_bucket")) if row.get("planned_bucket") in BUCKETS else None
    accuracy = _finite_float(row.get("engine_heldout_accuracy"))
    migrated = row.get("migrated") if isinstance(row.get("migrated"), bool) else None
    return bucket, accuracy, migrated


def _computed_fork_verdict(per_game_fork: Mapping[str, Any]) -> str | None:
    accuracies: list[float] = []
    migrations = 0
    valid_rows = 0
    for row in per_game_fork.values():
        if not isinstance(row, Mapping):
            continue
        bucket, accuracy, _migrated = _valid_row_values(row)
        if bucket is None or accuracy is None:
            continue
        valid_rows += 1
        accuracies.append(accuracy)
        if bucket == "COVERED":
            migrations += 1
    if valid_rows < 3 or not accuracies:
        return None
    high_accuracy = float(median(accuracies)) >= HIGH_ACCURACY_THRESHOLD
    if high_accuracy and migrations >= 1:
        return "GUIDANCE_WALL"
    if high_accuracy:
        return "PLANNER_GAP"
    return "INDUCER_CEILING"


def _numbers_match_fork_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    per_game = _mapping(artifact.get("per_game_fork"))
    reasons: list[str] = []
    valid_accuracies: list[float] = []
    computed_migrations = 0
    row_count = 0
    invalid_games: list[str] = []
    for game, row in per_game.items():
        row_count += 1
        if not isinstance(row, Mapping):
            reasons.append("invalid_per_game_row")
            invalid_games.append(str(game))
            continue
        bucket, accuracy, migrated = _valid_row_values(row)
        if bucket is None:
            reasons.append("invalid_planned_bucket")
            invalid_games.append(str(game))
        if accuracy is None or accuracy < 0.0 or accuracy > 1.0:
            reasons.append("invalid_engine_heldout_accuracy")
            invalid_games.append(str(game))
        else:
            valid_accuracies.append(accuracy)
        if migrated is None:
            reasons.append("invalid_migrated_flag")
            invalid_games.append(str(game))
        elif bucket in BUCKETS and migrated != (bucket == "COVERED"):
            reasons.append("row_migrated_mismatch")
            invalid_games.append(str(game))
        if bucket == "COVERED":
            computed_migrations += 1

    try:
        n_games = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        n_games = -1
        reasons.append("n_games_measured_not_integer")
    if n_games != row_count:
        reasons.append("n_games_measured_mismatch")
    if n_games < 3 or row_count < 3:
        reasons.append("n_games_measured_below_3")

    computed_median = float(median(valid_accuracies)) if valid_accuracies else None
    computed_fork = _computed_fork_verdict(per_game)
    reported_fork = artifact.get("fork_verdict")
    if reported_fork is not None and reported_fork not in FORK_VERDICTS:
        reasons.append("invalid_fork_verdict")
    if computed_fork is not None and reported_fork != computed_fork:
        reasons.append("fork_verdict_mismatch")
    if artifact.get("coverage_migration_count") != computed_migrations:
        reasons.append("coverage_migration_count_mismatch")
    reported_median = _finite_float(artifact.get("median_engine_heldout_accuracy"))
    if computed_median is None:
        if artifact.get("median_engine_heldout_accuracy") is not None:
            reasons.append("median_engine_heldout_accuracy_mismatch")
    elif reported_median is None or abs(reported_median - computed_median) > 1e-9:
        reasons.append("median_engine_heldout_accuracy_mismatch")

    unique_reasons = list(dict.fromkeys(reasons))
    return (
        {
            "passed": not unique_reasons,
            "reported_fork_verdict": reported_fork,
            "computed_fork_verdict": computed_fork,
            "reported_n_games_measured": artifact.get("n_games_measured"),
            "per_game_count": row_count,
            "reported_coverage_migration_count": artifact.get("coverage_migration_count"),
            "computed_coverage_migration_count": computed_migrations,
            "reported_median_engine_heldout_accuracy": artifact.get(
                "median_engine_heldout_accuracy"
            ),
            "computed_median_engine_heldout_accuracy": computed_median,
            "invalid_games": sorted(set(invalid_games)),
        },
        unique_reasons,
    )


def _live_path_and_provenance_check(
    artifact: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
    source_live_check: Mapping[str, Any],
    source_live_reasons: list[str],
) -> tuple[JsonDict, list[str], bool, bool]:
    lint_passed = live_lint_result.get("passed") is True
    artifact_live = artifact.get("live_path_reachable") is True
    source_live = source_live_check.get("passed") is True
    provenance = artifact.get("solve_provenance")
    live_confirmed = lint_passed and artifact_live and source_live
    provenance_confirmed = provenance == "development_proxy"
    reasons: list[str] = []
    if not live_confirmed:
        reasons.append("live_path_unreachable")
    reasons.extend(source_live_reasons)
    if not provenance_confirmed:
        reasons.append("solve_provenance_not_development_proxy")
    unique_reasons = list(dict.fromkeys(reasons))
    return (
        {
            "passed": not unique_reasons,
            "arc_orphan_solver_lint_passed": lint_passed,
            "artifact_live_path_reachable": artifact_live,
            "source_live_path_called": source_live,
            "source_live_path": dict(source_live_check),
            "solve_provenance": provenance,
            "development_proxy": provenance_confirmed,
        },
        unique_reasons,
        live_confirmed,
        provenance_confirmed,
    )


def _tool_cleanliness_check(
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
) -> JsonDict:
    summarizer_clean = summarizer_result.get("returncode") == 0
    adversarial_clean = (
        adversarial_result.get("loaded") is not False and adversarial_result.get("flag_count") == 0
    )
    return {
        "passed": summarizer_clean and adversarial_clean,
        "summarizer_returncode": summarizer_result.get("returncode"),
        "adversarial_flag_count": adversarial_result.get("flag_count"),
        "adversarial_loaded": adversarial_result.get("loaded"),
    }


def audit_a1_artifact(
    artifact: Mapping[str, Any],
    *,
    source_text: str,
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
) -> JsonDict:
    planner_check, planner_reasons, source_live_check, source_live_reasons = _source_checks(
        artifact, source_text
    )
    positive_check, positive_reasons = _positive_control_check(artifact)
    numbers_check, number_reasons = _numbers_match_fork_check(artifact)
    live_check, live_reasons, live_confirmed, provenance_confirmed = (
        _live_path_and_provenance_check(
            artifact,
            live_lint_result,
            source_live_check,
            source_live_reasons,
        )
    )
    tool_check = _tool_cleanliness_check(summarizer_result, adversarial_result)

    reasons = planner_reasons + positive_reasons + number_reasons + live_reasons
    reasons = list(dict.fromkeys(reasons))
    genuinely_diagnostic = not reasons
    return {
        "honest_verdict": (
            "complete_a1_fork_probe_audited"
            if genuinely_diagnostic
            else f"complete_a1_fork_probe_non_test_{_safe_suffix(reasons)}"
        ),
        "a1_genuinely_diagnostic": genuinely_diagnostic,
        "planner_blind_confirmed": not planner_reasons,
        "positive_control_confirmed": not positive_reasons,
        "numbers_match_fork": not number_reasons,
        "live_path_reachable_confirmed": live_confirmed,
        "solve_provenance_confirmed": provenance_confirmed,
        "non_diagnostic_reasons": reasons,
        "checks": {
            "planner_blind_to_banked_answer": planner_check,
            "positive_control": positive_check,
            "numbers_match_fork": numbers_check,
            "live_path_and_provenance": live_check,
            "summarizer_and_adversarial_verify": tool_check,
        },
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    source = repo / SOURCE_ARTIFACT_RELATIVE_PATH
    script = repo / SOURCE_SCRIPT_RELATIVE_PATH
    spec = repo / SPEC_RELATIVE_PATH
    spec_text = spec.read_text(encoding="utf-8") if spec.exists() else ""
    return {
        "ok": (
            source.exists()
            and script.exists()
            and "REQ-ARC-WMTE-4865" in spec_text
            and (repo / "scripts/summarize_artifact.py").exists()
            and (repo / "scripts/adversarial_verify.py").exists()
            and (repo / "scripts/arc_orphan_solver_lint.py").exists()
        ),
        "source_artifact_present": source.exists(),
        "source_script_present": script.exists(),
        "spec_has_req_4865": "REQ-ARC-WMTE-4865" in spec_text,
        "summarizer_script_present": (repo / "scripts/summarize_artifact.py").exists(),
        "adversarial_verify_script_present": (repo / "scripts/adversarial_verify.py").exists(),
        "arc_orphan_solver_lint_present": (repo / "scripts/arc_orphan_solver_lint.py").exists(),
    }


def _blocked_artifact(checks: Mapping[str, Any]) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_script_path": SOURCE_SCRIPT_RELATIVE_PATH,
        "source_artifact_checksum": None,
        "source_script_checksum": None,
        "source_honest_verdict": None,
        "source_fork_verdict": None,
        "source_n_games_measured": None,
        "source_flagged_adversarial": None,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_a1_artifact_missing",
        "a1_genuinely_diagnostic": False,
        "planner_blind_confirmed": False,
        "positive_control_confirmed": False,
        "numbers_match_fork": False,
        "live_path_reachable_confirmed": False,
        "solve_provenance_confirmed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": {},
        "non_diagnostic_reasons": ["missing_a1_artifact_or_script"],
        "summarizer_result": {},
        "adversarial_result": {},
        "live_lint_result": {},
        "preconditions_checked": dict(checks),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": DURATION_FLOOR_S,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    source_path: Path,
    source_script_path: Path,
    source_artifact: Mapping[str, Any],
    audit: Mapping[str, Any],
    summarizer_result: Mapping[str, Any],
    adversarial_result: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "source_artifact_path": SOURCE_ARTIFACT_RELATIVE_PATH,
        "source_script_path": SOURCE_SCRIPT_RELATIVE_PATH,
        "source_artifact_checksum": file_checksum(source_path),
        "source_script_checksum": file_checksum(source_script_path),
        "source_honest_verdict": source_artifact.get("honest_verdict"),
        "source_fork_verdict": source_artifact.get("fork_verdict"),
        "source_n_games_measured": source_artifact.get("n_games_measured"),
        "source_flagged_adversarial": source_artifact.get("flagged_adversarial") is True,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": audit.get("honest_verdict"),
        "a1_genuinely_diagnostic": audit.get("a1_genuinely_diagnostic"),
        "planner_blind_confirmed": audit.get("planner_blind_confirmed"),
        "positive_control_confirmed": audit.get("positive_control_confirmed"),
        "numbers_match_fork": audit.get("numbers_match_fork"),
        "live_path_reachable_confirmed": audit.get("live_path_reachable_confirmed"),
        "solve_provenance_confirmed": audit.get("solve_provenance_confirmed"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": dict(_mapping(audit.get("checks"))),
        "non_diagnostic_reasons": list(audit.get("non_diagnostic_reasons") or []),
        "summarizer_result": dict(summarizer_result),
        "adversarial_result": dict(adversarial_result),
        "live_lint_result": dict(live_lint_result),
        "preconditions_checked": dict(preconditions_checked),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(DURATION_FLOOR_S, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "a1_genuinely_diagnostic",
        "planner_blind_confirmed",
        "positive_control_confirmed",
        "numbers_match_fork",
        "live_path_reachable_confirmed",
        "solve_provenance_confirmed",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field}_must_be_bool")
    if not isinstance(artifact.get("checks"), dict):
        errors.append("checks_must_be_dict")
    if not isinstance(artifact.get("non_diagnostic_reasons"), list):
        errors.append("non_diagnostic_reasons_must_be_list")
    if artifact.get("a1_genuinely_diagnostic") is True and artifact.get("non_diagnostic_reasons"):
        errors.append("diagnostic_artifact_has_failure_reasons")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    expected = "sha256:" + payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected:
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def render_markdown_section(artifact: Mapping[str, Any]) -> str:
    checks = _mapping(artifact.get("checks"))
    rows = [
        "",
        "## Experiment 4865 .448 A1 Fork Probe Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- a1_genuinely_diagnostic: `{artifact.get('a1_genuinely_diagnostic')}`",
        f"- Non-diagnostic reasons: `{', '.join(artifact.get('non_diagnostic_reasons') or []) or '-'}`",
        f"- Inference substrate: `{artifact.get('inference_substrate')}`",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for name, check in checks.items():
        if not isinstance(check, Mapping):
            continue
        detail = {key: value for key, value in check.items() if key != "passed"}
        rows.append(
            f"| `{name}` | `{check.get('passed')}` | `{json.dumps(detail, sort_keys=True)}` |"
        )
    rows.extend(
        [
            "",
            f"- Source artifact checksum: `{artifact.get('source_artifact_checksum')}`",
            f"- Source script checksum: `{artifact.get('source_script_checksum')}`",
            "",
        ]
    )
    return "\n".join(rows)


def append_markdown_report(
    artifact: Mapping[str, Any],
    *,
    root: Path | str = REPO_ROOT,
) -> Path:
    report_path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    marker = "## Experiment 4865 .448 A1 Fork Probe Audit"
    if report_path.exists():
        current = report_path.read_text(encoding="utf-8")
        if marker in current:
            return report_path
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        current = "# ARC Null Silent-Bug Audit\n"
    report_path.write_text(current.rstrip() + render_markdown_section(artifact), encoding="utf-8")
    return report_path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    now: Callable[[], float] | None = None,
) -> JsonDict:
    repo = Path(root)
    clock = now or time.monotonic
    start = clock()
    checks = check_preconditions(repo)
    source_path = repo / SOURCE_ARTIFACT_RELATIVE_PATH
    source_script_path = repo / SOURCE_SCRIPT_RELATIVE_PATH
    if not checks["ok"] or not source_path.exists() or not source_script_path.exists():
        checks = dict(checks)
        checks["ok"] = False
        checks["source_artifact_present"] = source_path.exists()
        checks["source_script_present"] = source_script_path.exists()
        artifact = _blocked_artifact(checks)
        if write:
            write_artifact(artifact, root=repo)
            append_markdown_report(artifact, root=repo)
        return artifact

    source_artifact = _read_json(source_path)
    source_text = source_script_path.read_text(encoding="utf-8")
    summarizer_result = run_summarizer(source_path)
    adversarial_result = run_adversarial_verify(source_path)
    live_lint_result = run_arc_orphan_solver_lint(repo)
    audit = audit_a1_artifact(
        source_artifact,
        source_text=source_text,
        summarizer_result=summarizer_result,
        adversarial_result=adversarial_result,
        live_lint_result=live_lint_result,
    )
    artifact = build_artifact(
        source_path=source_path,
        source_script_path=source_script_path,
        source_artifact=source_artifact,
        audit=audit,
        summarizer_result=summarizer_result,
        adversarial_result=adversarial_result,
        live_lint_result=live_lint_result,
        preconditions_checked=checks,
        duration_s=clock() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=repo)
        append_markdown_report(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "a1_genuinely_diagnostic": artifact["a1_genuinely_diagnostic"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
