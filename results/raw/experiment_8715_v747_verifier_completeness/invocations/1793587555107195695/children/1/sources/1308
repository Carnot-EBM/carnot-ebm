"""Exp 4918: adversarial audit of Exp 4914 causal-abstraction diagnostic.

Spec refs: REQ-ARC-WMTE-4918,
SCENARIO-ARC-WMTE-4918-A1-DIAGNOSTIC-AUDIT,
SCENARIO-ARC-WMTE-4918-NAMED-FAILURES,
SCENARIO-ARC-WMTE-4918-BLOCKED-A1-ARTIFACT.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT / "python"))


EXPERIMENT = "experiment_4918_causal_abstraction_audit"
EXPERIMENT_ID = 4918
SCHEMA = "carnot.arc.causal_abstraction_audit_4918.v1"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4914_causal_abstraction_wall_diagnostic.json"
A1_SCRIPT_RELATIVE_PATH = "python/carnot/experiment_4914_causal_abstraction_wall_diagnostic.py"
EXP4903_ARTIFACT_RELATIVE_PATH = (
    "results/experiment_4903_env_grounded_location_pruned_search.json"
)
RESULT_RELATIVE_PATH = "results/experiment_4918_causal_abstraction_audit.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DURATION_FLOOR_S = 1.0
RANDOM_SEED = 4918

SPEC_REFS = [
    "REQ-ARC-WMTE-4918",
    "SCENARIO-ARC-WMTE-4918-A1-DIAGNOSTIC-AUDIT",
    "SCENARIO-ARC-WMTE-4918-NAMED-FAILURES",
    "SCENARIO-ARC-WMTE-4918-BLOCKED-A1-ARTIFACT",
]

CHECK_NAMES = (
    "real_transitions",
    "not_value_table",
    "observable_claims_verified",
    "positive_control_observable",
    "oracle_distinct_planner_blind",
    "numbers_match_fork",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_a1_causal_abstraction_audited "
            "(trusted or with named failures)."
        )
    },
    "a1_diagnostic_trustworthy": {
        "principle": (
            "AND of the 6 checks -- the capstone trusts A1's closure verdict ONLY if this is true."
        )
    },
    "checks": {
        "principle": (
            "per-check booleans {real_transitions, not_value_table, "
            "observable_claims_verified, positive_control_observable, "
            "oracle_distinct_planner_blind, numbers_match_fork}."
        )
    },
    "a1_failure_reasons": {
        "principle": (
            "list of named failures (empty if trusted) -- the audit reports honestly, no rubber-stamp."
        )
    },
    "observable_claims_spot_checked": {
        "principle": (
            ">=2 claimed-observable variables verified readable from frame/env state "
            "(the load-bearing honesty check)."
        )
    },
    "inference_substrate": {
        "principle": "verifier_ensemble_against_cached_candidates (reads cached artifacts; 1s floor)."
    },
    "preconditions_checked": {
        "principle": (
            "records exp4914/exp4903 presence; a missing input emits blocked_."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "source_a1_artifact",
    "source_a1_script",
    "source_exp4903_artifact",
    "transition_cross_checks",
    "not_value_table_evidence",
    "positive_control_evidence",
    "oracle_distinct_planner_blind_evidence",
    "numbers_match_fork_evidence",
    "field_principles",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

JsonDict = dict[str, Any]
Clock = Callable[[], float]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _finite_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _attach_checksum(artifact: JsonDict) -> JsonDict:
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _duration(value: float) -> float:
    return max(float(value), DURATION_FLOOR_S)


def _full_call_name(callable_node: ast.AST) -> str:
    if isinstance(callable_node, ast.Name):
        return callable_node.id
    if isinstance(callable_node, ast.Attribute):
        prefix = _full_call_name(callable_node.value)
        return f"{prefix}.{callable_node.attr}" if prefix else callable_node.attr
    return ""


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _source_transition_evidence(source_text: str) -> tuple[JsonDict, list[str]]:
    reasons: list[str] = []
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return {"passed": False, "parse_error": str(exc)}, ["a1_script_not_parseable"]

    function = _find_function(tree, "default_game_classifier")
    if function is None:
        return (
            {
                "passed": False,
                "default_game_classifier_present": False,
                "collect_transitions_called": False,
                "placeholder_transition_literals": [],
            },
            ["default_game_classifier_missing"],
        )

    calls = [_full_call_name(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)]
    collect_called = any(call.endswith("collect_transitions") for call in calls)
    load_engine_called = any(call.endswith("load_engine") for call in calls)
    placeholders: list[JsonDict] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Dict):
            keys = [key.value for key in node.keys if isinstance(key, ast.Constant)]
            if "placeholder" in keys or "synthetic" in keys:
                placeholders.append({"line": int(getattr(node, "lineno", 0)), "keys": keys})

    if not collect_called:
        reasons.append("a1_script_collect_transitions_missing")
    if not load_engine_called:
        reasons.append("a1_script_load_engine_missing")
    if placeholders:
        reasons.append("a1_script_placeholder_transitions_present")
    return (
        {
            "passed": not reasons,
            "default_game_classifier_present": True,
            "collect_transitions_called": collect_called,
            "load_engine_called": load_engine_called,
            "calls": calls,
            "placeholder_transition_literals": placeholders,
        },
        reasons,
    )


def _failed_a1_rows(a1_artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = _mapping(a1_artifact.get("per_game_causal_abstraction"))
    config_games = {
        str(game)
        for game in _sequence(_mapping(a1_artifact.get("causal_abstraction_config")).get("failed_games"))
    }
    out: dict[str, Mapping[str, Any]] = {}
    for game, row in rows.items():
        row_map = _mapping(row)
        if row_map.get("role") == "failed" or str(game) in config_games:
            out[str(game)] = row_map
    return out


def _row_bool(row: Mapping[str, Any], key: str) -> bool:
    return row.get(key) is True


def _exp4903_failed_row_passes(
    *, game: str, exp4903_row: Mapping[str, Any], a1_row: Mapping[str, Any]
) -> JsonDict:
    evidence = _mapping(a1_row.get("evidence"))
    transition_count = _finite_int(evidence.get("transition_count"))
    real_reads = _finite_int(exp4903_row.get("real_env_value_reads"))
    value_predictions = _finite_int(exp4903_row.get("change_value_predictions_used"))
    best_path_len = _finite_int(exp4903_row.get("best_path_len"))
    first_win = _finite_float(exp4903_row.get("first_win_env_grounded"))
    methods = {str(item) for item in _sequence(exp4903_row.get("live_path_methods_called"))}
    checks = {
        "row_present": bool(exp4903_row),
        "same_game": exp4903_row.get("game") in (None, game),
        "never_enumerated_bucket": exp4903_row.get("bucket") == "NEVER_ENUMERATED",
        "never_enumerated_baseline": exp4903_row.get("baseline_bucket") == "NEVER_ENUMERATED",
        "failed_first_win": first_win is not None and first_win <= 0.0,
        "not_migrated": exp4903_row.get("migrated") is not True,
        "multi_step_failed_prefix": best_path_len is not None and best_path_len > 1,
        "real_env_reads_cover_samples": (
            real_reads is not None
            and transition_count is not None
            and transition_count > 0
            and real_reads >= transition_count
        ),
        "no_change_value_predictions": value_predictions == 0,
        "live_transition_path_called": {
            "arc_executable_world_model.load_engine",
            "arc_executable_world_model.plan_in_model",
        }.issubset(methods),
    }
    return {
        "game": game,
        "passed": all(checks.values()),
        "checks": checks,
        "transition_count_cited_by_4914": transition_count,
        "real_env_value_reads_in_4903": real_reads,
        "change_value_predictions_used_in_4903": value_predictions,
        "exp4903_bucket": exp4903_row.get("bucket"),
        "exp4903_baseline_bucket": exp4903_row.get("baseline_bucket"),
    }


def _check_real_transitions(
    a1_artifact: Mapping[str, Any], a1_source_text: str, exp4903_artifact: Mapping[str, Any]
) -> tuple[bool, list[JsonDict], JsonDict, list[str]]:
    reasons: list[str] = []
    source_evidence, source_reasons = _source_transition_evidence(a1_source_text)
    reasons.extend(source_reasons)

    failed_rows = _failed_a1_rows(a1_artifact)
    exp_rows = _mapping(exp4903_artifact.get("per_game_first_win"))
    cross_checks = [
        _exp4903_failed_row_passes(
            game=game,
            exp4903_row=_mapping(exp_rows.get(game)),
            a1_row=row,
        )
        for game, row in sorted(failed_rows.items())
    ]
    if len(cross_checks) < 3:
        reasons.append("fewer_than_three_failed_transitions_cross_checked")
    for row in cross_checks:
        if not row["passed"]:
            reasons.append(f"exp4903_failed_transition_mismatch:{row['game']}")
    return not reasons, cross_checks, source_evidence, reasons


def _classification_rows_valid(rows: Mapping[str, Any]) -> bool:
    if not rows:
        return False
    for row in rows.values():
        row_map = _mapping(row)
        if row_map.get("classification") not in {"OBSERVABLE_GAP", "HIDDEN_STATE"}:
            return False
        if not _sequence(row_map.get("required_variables")):
            return False
        if not isinstance(row_map.get("observable_from_interface"), Mapping):
            return False
    return True


def _contains_decision_need_marker(value: Any, path: tuple[str, ...] = ()) -> bool:
    if path and path[0] == "field_principles":
        return False
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"fails_when", "decision_need_targets"}:
                return True
            if "value_accuracy" in key_text or key_text.startswith("value_acc"):
                return True
            if _contains_decision_need_marker(child, (*path, str(key))):
                return True
    elif isinstance(value, list):
        return any(_contains_decision_need_marker(child, path) for child in value)
    return False


def _check_not_value_table(a1_artifact: Mapping[str, Any]) -> tuple[bool, JsonDict, list[str]]:
    rows = _mapping(a1_artifact.get("per_game_causal_abstraction"))
    config = _mapping(a1_artifact.get("causal_abstraction_config"))
    evidence = {
        "is_decision_need_table_in_disguise": a1_artifact.get(
            "is_decision_need_table_in_disguise"
        ),
        "classification_only": config.get("classification_only"),
        "classification_rows_valid": _classification_rows_valid(rows),
        "decision_need_markers_present": _contains_decision_need_marker(a1_artifact),
        "targets": config.get("targets"),
    }
    reasons: list[str] = []
    if evidence["is_decision_need_table_in_disguise"] is not False:
        reasons.append("decision_need_table_in_disguise")
    if evidence["classification_only"] is not True:
        reasons.append("classification_only_not_declared")
    if evidence["classification_rows_valid"] is not True:
        reasons.append("classification_report_missing")
    if evidence["decision_need_markers_present"]:
        reasons.append("decision_need_markers_present")
    return not reasons, evidence, reasons


def _extractor_readable(variable: str, extractor: Any) -> bool:
    text = str(extractor or "").lower()
    if not text:
        return False
    specific_markers = {
        "visible_grid_hash": ("frame.grid",),
        "action_id": ("candidate.action_id",),
        "action_data": ("candidate.data",),
        "changed_cell_value_basis": ("frame grid", "transition delta"),
        "visible_level_before": ("env", "frame"),
    }
    markers = specific_markers.get(variable)
    if markers is not None:
        return all(marker in text for marker in markers)
    return any(marker in text for marker in ("frame", "candidate", "env"))


def _check_observable_claims(a1_artifact: Mapping[str, Any]) -> tuple[bool, list[JsonDict], list[str]]:
    rows: dict[str, Mapping[str, Any]] = {}
    rows.update({str(key): _mapping(value) for key, value in _mapping(a1_artifact.get("per_game_causal_abstraction")).items()})
    rows.update({str(key): _mapping(value) for key, value in _mapping(a1_artifact.get("positive_control_rows")).items()})
    verified: list[JsonDict] = []
    failures: list[str] = []
    seen_variables: set[str] = set()
    for game, row in sorted(rows.items()):
        observable = _mapping(row.get("observable_from_interface"))
        proofs = _mapping(_mapping(row.get("evidence")).get("observability_proofs"))
        for variable in _sequence(row.get("required_variables")):
            variable_name = str(variable)
            if observable.get(variable_name) is not True:
                continue
            proof = _mapping(proofs.get(variable_name))
            extractor = proof.get("extractor")
            passed = proof.get("observable") is True and _extractor_readable(variable_name, extractor)
            record = {
                "game": game,
                "variable": variable_name,
                "extractor": extractor,
                "proof": proof.get("proof"),
                "passed": passed,
            }
            if passed and variable_name not in seen_variables:
                verified.append(record)
                seen_variables.add(variable_name)
            if not passed:
                failures.append(f"observable_claim_not_readable:{game}:{variable_name}")
    if len(verified) < 2:
        failures.append("fewer_than_two_observable_claims_verified")
    return not failures, verified, failures


def _row_observable(row: Mapping[str, Any]) -> bool:
    observable = _mapping(row.get("observable_from_interface"))
    required = _sequence(row.get("required_variables"))
    return row.get("classification") == "OBSERVABLE_GAP" and bool(required) and all(
        observable.get(str(variable)) is True for variable in required
    )


def _check_positive_controls(a1_artifact: Mapping[str, Any]) -> tuple[bool, JsonDict, list[str]]:
    rows = _mapping(a1_artifact.get("positive_control_rows"))
    row_evidence = [
        {
            "game": str(game),
            "classification": _mapping(row).get("classification"),
            "row_observable": _row_observable(_mapping(row)),
        }
        for game, row in sorted(rows.items())
    ]
    declared = a1_artifact.get("positive_control_classifies_observable")
    passed = bool(rows) and declared is True and all(row["row_observable"] for row in row_evidence)
    reasons = [] if passed else ["positive_control_not_observable"]
    return passed, {"declared": declared, "rows": row_evidence}, reasons


def _check_oracle_planner(a1_artifact: Mapping[str, Any]) -> tuple[bool, JsonDict, list[str]]:
    evidence = {
        "verifier_is_oracle": a1_artifact.get("verifier_is_oracle"),
        "planner_blind_to_banked_answer": a1_artifact.get("planner_blind_to_banked_answer"),
        "precondition_planner_blind": _mapping(a1_artifact.get("preconditions_checked")).get(
            "planner_blind_to_banked_answer"
        ),
    }
    passed = (
        evidence["verifier_is_oracle"] is False
        and evidence["planner_blind_to_banked_answer"] is True
    )
    return passed, evidence, [] if passed else ["oracle_or_planner_blind_failed"]


def _compute_fork(
    failed_rows: Mapping[str, Mapping[str, Any]], positive_rows: Mapping[str, Mapping[str, Any]]
) -> str | None:
    if not positive_rows or not all(_row_observable(row) for row in positive_rows.values()):
        return "DIAGNOSTIC_DEGENERATE_RETIRED"
    if len(failed_rows) < 3:
        return None
    if all(_row_observable(row) for row in failed_rows.values()):
        return "WALL_IS_OBSERVABLE_VARIABLE_GAP"
    return "WALL_IS_HIDDEN_STATE"


def _check_numbers_match_fork(a1_artifact: Mapping[str, Any]) -> tuple[bool, JsonDict, list[str]]:
    failed_rows = _failed_a1_rows(a1_artifact)
    positive_rows = {
        str(game): _mapping(row)
        for game, row in _mapping(a1_artifact.get("positive_control_rows")).items()
    }
    computed = _compute_fork(failed_rows, positive_rows)
    declared = a1_artifact.get("fork_verdict")
    failed_observable_subset = bool(failed_rows) and all(
        _row_observable(row) for row in failed_rows.values()
    )
    positive_observable = bool(positive_rows) and all(
        _row_observable(row) for row in positive_rows.values()
    )
    evidence = {
        "declared_fork_verdict": declared,
        "computed_fork_verdict": computed,
        "declared_minimal_abstraction_is_observable_subset": a1_artifact.get(
            "minimal_abstraction_is_observable_subset"
        ),
        "computed_minimal_abstraction_is_observable_subset": failed_observable_subset,
        "declared_positive_control_classifies_observable": a1_artifact.get(
            "positive_control_classifies_observable"
        ),
        "computed_positive_control_classifies_observable": positive_observable,
        "per_game_classifications": {
            game: row.get("classification") for game, row in sorted(failed_rows.items())
        },
        "positive_control_classifications": {
            game: row.get("classification") for game, row in sorted(positive_rows.items())
        },
    }
    passed = (
        declared == computed
        and a1_artifact.get("minimal_abstraction_is_observable_subset")
        == failed_observable_subset
        and a1_artifact.get("positive_control_classifies_observable") == positive_observable
    )
    return passed, evidence, [] if passed else ["numbers_do_not_match_fork"]


def _source_descriptor(path: str, checksum: str | None = None) -> JsonDict:
    return {"path": path, "checksum": checksum}


def audit_sources(
    *,
    a1_artifact: Mapping[str, Any],
    a1_source_text: str,
    exp4903_artifact: Mapping[str, Any],
    duration_s: float = DURATION_FLOOR_S,
    preconditions_checked: Mapping[str, Any] | None = None,
    source_a1_artifact: Mapping[str, Any] | None = None,
    source_a1_script: Mapping[str, Any] | None = None,
    source_exp4903_artifact: Mapping[str, Any] | None = None,
) -> JsonDict:
    real_ok, transition_rows, source_evidence, real_reasons = _check_real_transitions(
        a1_artifact, a1_source_text, exp4903_artifact
    )
    value_ok, value_evidence, value_reasons = _check_not_value_table(a1_artifact)
    observable_ok, observable_spots, observable_reasons = _check_observable_claims(a1_artifact)
    positive_ok, positive_evidence, positive_reasons = _check_positive_controls(a1_artifact)
    oracle_ok, oracle_evidence, oracle_reasons = _check_oracle_planner(a1_artifact)
    numbers_ok, numbers_evidence, numbers_reasons = _check_numbers_match_fork(a1_artifact)

    checks = {
        "real_transitions": real_ok,
        "not_value_table": value_ok,
        "observable_claims_verified": observable_ok,
        "positive_control_observable": positive_ok,
        "oracle_distinct_planner_blind": oracle_ok,
        "numbers_match_fork": numbers_ok,
    }
    reason_by_check = {
        "real_transitions": "real_transitions_failed",
        "not_value_table": "decision_need_table_in_disguise",
        "observable_claims_verified": "observable_claims_unverified",
        "positive_control_observable": "positive_control_not_observable",
        "oracle_distinct_planner_blind": "oracle_or_planner_blind_failed",
        "numbers_match_fork": "numbers_do_not_match_fork",
    }
    detailed_reasons = (
        real_reasons
        + value_reasons
        + observable_reasons
        + positive_reasons
        + oracle_reasons
        + numbers_reasons
    )
    failure_reasons = [
        reason_by_check[check] for check, passed in checks.items() if passed is not True
    ]
    failure_reasons.extend(detailed_reasons)
    failure_reasons = list(dict.fromkeys(failure_reasons))

    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "complete_a1_causal_abstraction_audited",
        "a1_diagnostic_trustworthy": all(checks.values()),
        "checks": checks,
        "a1_failure_reasons": [] if all(checks.values()) else failure_reasons,
        "observable_claims_spot_checked": observable_spots,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked or {}),
        "source_a1_artifact": dict(
            source_a1_artifact or _source_descriptor(A1_ARTIFACT_RELATIVE_PATH)
        ),
        "source_a1_script": dict(source_a1_script or _source_descriptor(A1_SCRIPT_RELATIVE_PATH)),
        "source_exp4903_artifact": dict(
            source_exp4903_artifact or _source_descriptor(EXP4903_ARTIFACT_RELATIVE_PATH)
        ),
        "transition_cross_checks": transition_rows,
        "source_transition_evidence": source_evidence,
        "not_value_table_evidence": value_evidence,
        "positive_control_evidence": positive_evidence,
        "oracle_distinct_planner_blind_evidence": oracle_evidence,
        "numbers_match_fork_evidence": numbers_evidence,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(duration_s),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def build_blocked_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    missing_reasons: list[str] = []
    if not _row_bool(_mapping(preconditions_checked.get("experiment_4914_artifact")), "present"):
        missing_reasons.append("missing_experiment_4914_artifact")
    if not _row_bool(_mapping(preconditions_checked.get("experiment_4914_script")), "present"):
        missing_reasons.append("missing_experiment_4914_script")
    if not _row_bool(_mapping(preconditions_checked.get("experiment_4903_artifact")), "present"):
        missing_reasons.append("missing_experiment_4903_artifact")
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": "blocked_a1_artifact_missing",
        "a1_diagnostic_trustworthy": False,
        "checks": {check: False for check in CHECK_NAMES},
        "a1_failure_reasons": missing_reasons or ["missing_required_precondition"],
        "observable_claims_spot_checked": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "source_a1_artifact": _source_descriptor(A1_ARTIFACT_RELATIVE_PATH),
        "source_a1_script": _source_descriptor(A1_SCRIPT_RELATIVE_PATH),
        "source_exp4903_artifact": _source_descriptor(EXP4903_ARTIFACT_RELATIVE_PATH),
        "transition_cross_checks": [],
        "not_value_table_evidence": {},
        "positive_control_evidence": {},
        "oracle_distinct_planner_blind_evidence": {},
        "numbers_match_fork_evidence": {},
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(duration_s),
        "reproducibility_checksum": "",
    }
    return _attach_checksum(artifact)


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing_field:{field}")
    if errors:
        return errors

    verdict = str(artifact.get("honest_verdict"))
    if not verdict.startswith(("complete_", "blocked_")):
        errors.append("honest_verdict_terminal_prefix")
    checks = artifact.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(CHECK_NAMES):
        errors.append("checks")
        checks = {}
    elif any(value not in (True, False) for value in checks.values()):
        errors.append("checks_bool")
    trustworthy = artifact.get("a1_diagnostic_trustworthy")
    if trustworthy is not all(bool(checks.get(check)) for check in CHECK_NAMES):
        errors.append("a1_diagnostic_trustworthy")
    reasons = artifact.get("a1_failure_reasons")
    if not isinstance(reasons, list):
        errors.append("a1_failure_reasons")
        reasons = []
    if trustworthy is True and reasons:
        errors.append("a1_failure_reasons")
    if trustworthy is False and not reasons:
        errors.append("a1_failure_reasons")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    try:
        if float(artifact.get("duration_s")) < DURATION_FLOOR_S:
            errors.append("duration_s")
    except (TypeError, ValueError):
        errors.append("duration_s")

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles")
    else:
        for field, principle in FIELD_PRINCIPLES.items():
            if principles.get(field) != principle:
                errors.append(f"field_principles.{field}")

    if checks.get("observable_claims_verified") is True:
        spots = artifact.get("observable_claims_spot_checked")
        if not isinstance(spots, list) or len(spots) < 2:
            errors.append("observable_claims_spot_checked")
    if checks.get("real_transitions") is True:
        rows = artifact.get("transition_cross_checks")
        if not isinstance(rows, list) or not rows or any(
            _mapping(row).get("passed") is not True for row in rows
        ):
            errors.append("transition_cross_checks")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked")
    for field in (
        "source_a1_artifact",
        "source_a1_script",
        "source_exp4903_artifact",
        "not_value_table_evidence",
        "positive_control_evidence",
        "oracle_distinct_planner_blind_evidence",
        "numbers_match_fork_evidence",
    ):
        if not isinstance(artifact.get(field), Mapping):
            errors.append(field)
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _precondition_record(path: Path) -> JsonDict:
    record: JsonDict = {"path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path), "present": path.exists()}
    if path.exists():
        record["checksum"] = file_checksum(path)
    return record


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    now: Clock = time.time,
) -> JsonDict:
    root_path = Path(root)
    started = now()
    a1_path = root_path / A1_ARTIFACT_RELATIVE_PATH
    script_path = root_path / A1_SCRIPT_RELATIVE_PATH
    exp4903_path = root_path / EXP4903_ARTIFACT_RELATIVE_PATH
    preconditions = {
        "experiment_4914_artifact": _precondition_record(a1_path),
        "experiment_4914_script": _precondition_record(script_path),
        "experiment_4903_artifact": _precondition_record(exp4903_path),
    }

    if not all(_mapping(row).get("present") is True for row in preconditions.values()):
        artifact = build_blocked_artifact(
            preconditions_checked=preconditions,
            duration_s=now() - started,
        )
    else:
        artifact = audit_sources(
            a1_artifact=_read_json(a1_path),
            a1_source_text=script_path.read_text(encoding="utf-8"),
            exp4903_artifact=_read_json(exp4903_path),
            duration_s=now() - started,
            preconditions_checked=preconditions,
            source_a1_artifact=preconditions["experiment_4914_artifact"],
            source_a1_script=preconditions["experiment_4914_script"],
            source_exp4903_artifact=preconditions["experiment_4903_artifact"],
        )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError(";".join(errors))
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "a1_diagnostic_trustworthy": artifact["a1_diagnostic_trustworthy"],
                "checks": artifact["checks"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main(sys.argv[1:]))
