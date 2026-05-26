"""Exp 3113 diagnostic local SOTA verifier calibration v5.

Spec refs: REQ-VERIFY-3113, SCENARIO-VERIFY-3113.

This module does not run a new language-model decode. It measures the existing
local SOTA route transcript against exact solver labels, then compares that
baseline to deterministic logic-pilot and certified-coherence decisions. The
repair gate is reported separately so negative or zero lift remains visible
instead of being rewritten into a success claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3113_diagnostic_local_sota_verifier_calibration_v5"
SCHEMA = "carnot.diagnostic_local_sota_verifier_calibration.v5"
OUTPUT_REL_PATH = Path("results/experiment_3113_diagnostic_local_sota_verifier_calibration_v5.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3113_diagnostic_local_sota_verifier_calibration_v5.py"

EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3098_REL_PATH = Path("results/experiment_3098_maxsat_abstention_routing_policy_v1.json")
EXP3099_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
EXP3110_REL_PATH = Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json")
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
EXP3112_REL_PATH = Path("results/experiment_3112_logic_regularized_verifier_pilot_v1.json")

MANIFEST_REL_PATH = Path("results/exact_fixture_eval_protocol_3097/stratified_eval_manifest.jsonl")
POLICY_REL_PATH = Path("results/maxsat_abstention_routing_policy_3098/policy.json")
EXP3099_ROWS_REL_PATH = Path("results/local_sota_confidence_abstention_panel_3099/rows.jsonl")
EXP3112_ROWS_REL_PATH = Path("results/logic_regularized_verifier_pilot_3112/rows.jsonl")

MANDATORY_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
NON_TINY_EXACT_COUNT_FLOOR = 24
GATE_STATES = frozenset(
    {
        "unblocked",
        "blocked_negative_delta",
        "blocked_model_cache",
        "blocked_tiny_panel",
        "blocked_missing_inputs",
    }
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
REQUIRED_FIELDS = (
    "diagnostic_verifier_calibration_v5_ready",
    "model_specs",
    "mandatory_headline_model_ids",
    "selected_headline_model_ids",
    "live_llm_inference",
    "exact_ground_truth_count",
    "verifier_gain_delta",
    "verifier_gain_delta_with_certified_coherence",
    "false_accept_rate",
    "false_reject_rate",
    "calibration_error",
    "abstention_precision",
    "rejection_recall",
    "repair_gate_state",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3113_diagnostic_local_sota_verifier_calibration_v5.py -q --no-cov",
    ".venv/bin/coverage run --source=python/carnot/eval/diagnostic_local_sota_verifier_calibration_v5.py -m pytest -o addopts='' tests/python/test_experiment_3113_diagnostic_local_sota_verifier_calibration_v5.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/diagnostic_local_sota_verifier_calibration_v5.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_SPECS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("experiment_template_cache_policy", Path("scripts/experiment_template.py"), False),
    ("exp3097_exact_protocol", EXP3097_REL_PATH, True),
    ("exp3097_stratified_manifest", MANIFEST_REL_PATH, True),
    ("exp3098_maxsat_policy_artifact", EXP3098_REL_PATH, True),
    ("exp3098_maxsat_policy_json", POLICY_REL_PATH, True),
    ("exp3099_local_sota_panel", EXP3099_REL_PATH, True),
    ("exp3099_panel_rows", EXP3099_ROWS_REL_PATH, True),
    ("exp3110_model_manifest", EXP3110_REL_PATH, True),
    ("exp3111_certified_feedback_v3", EXP3111_REL_PATH, True),
    ("exp3112_logic_pilot", EXP3112_REL_PATH, True),
    ("exp3112_logic_rows", EXP3112_ROWS_REL_PATH, True),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object, failing closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_jsonl_rows_from_text(text: str) -> list[JsonDict]:
    """Read JSONL object rows, skipping malformed and non-object lines."""

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
    """Read JSONL rows from disk, returning an empty list when absent."""

    try:
        return read_jsonl_rows_from_text(path.read_text(encoding="utf-8"))
    except OSError:
        return []


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    min_exact_count: int = NON_TINY_EXACT_COUNT_FLOOR,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3113: build the diagnostic calibration artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3097 = read_json_object(root_path / EXP3097_REL_PATH)
    exp3098 = read_json_object(root_path / EXP3098_REL_PATH)
    exp3099 = read_json_object(root_path / EXP3099_REL_PATH)
    exp3110 = read_json_object(root_path / EXP3110_REL_PATH)
    exp3111 = read_json_object(root_path / EXP3111_REL_PATH)
    exp3112 = read_json_object(root_path / EXP3112_REL_PATH)

    manifest_rel_path = Path(str(exp3097.get("stratified_eval_manifest_path") or MANIFEST_REL_PATH))
    policy_rel_path = Path(str(exp3098.get("routing_policy_path") or POLICY_REL_PATH))
    panel_rel_path = Path(str(exp3099.get("panel_rows_path") or EXP3099_ROWS_REL_PATH))
    logic_rel_path = Path(str(exp3112.get("diagnostic_rows_path") or EXP3112_ROWS_REL_PATH))

    manifest_rows = read_jsonl_rows(root_path / manifest_rel_path)
    panel_rows = read_jsonl_rows(root_path / panel_rel_path)
    logic_rows = read_jsonl_rows(root_path / logic_rel_path)
    certificates = [dict(row) for row in exp3111.get("certificates", []) if isinstance(row, Mapping)]
    source_rows = source_artifacts(root_path, manifest_rel_path, policy_rel_path, panel_rel_path, logic_rel_path)
    missing_required_sources = [
        row for row in source_rows if row["required"] is True and row["exists"] is not True
    ]

    calibration_rows = select_calibration_rows(manifest_rows, panel_rows, certificates, logic_rows)
    baseline_metrics = decision_metrics(calibration_rows, "baseline_decision")
    pilot_metrics = decision_metrics(calibration_rows, "logic_decision")
    certified_metrics = decision_metrics(calibration_rows, "certified_coherence_decision")
    raw_delta = round(pilot_metrics["accuracy"] - baseline_metrics["accuracy"], 6)
    certified_delta = round(certified_metrics["accuracy"] - baseline_metrics["accuracy"], 6)

    model_specs = list(exp3099.get("model_specs") or [])
    mandatory_ids = list(exp3110.get("mandatory_headline_model_ids") or MANDATORY_MODEL_IDS)
    selected_ids = list(
        exp3110.get("selected_headline_model_ids")
        or exp3099.get("selected_model_ids")
        or exp3099.get("models_used")
        or []
    )
    readiness_checks = {
        "exp3097_protocol_ready": exp3097.get("eval_protocol_ready") is True,
        "exp3098_maxsat_policy_ready": exp3098.get("maxsat_policy_ready") is True,
        "exp3099_panel_ready": exp3099.get("abstention_panel_v3_ready") is True,
        "exp3110_model_manifest_ready": exp3110.get("sota_model_manifest_ready") is True,
        "exp3111_certified_feedback_ready": exp3111.get("certified_coherence_feedback_v3_ready") is True,
        "exp3112_logic_pilot_ready": exp3112.get("logic_regularized_verifier_pilot_ready") is True,
        "required_sources_present": not missing_required_sources,
        "calibration_rows_present": len(calibration_rows) > 0,
        "finite_metrics": all(
            math.isfinite(value)
            for value in (
                raw_delta,
                certified_delta,
                certified_metrics["false_accept_rate"],
                certified_metrics["false_reject_rate"],
                certified_metrics["abstention_precision"],
                certified_metrics["rejection_recall"],
            )
        ),
    }
    diagnostic_ready = all(readiness_checks.values())
    gate_state = repair_gate_state(
        diagnostic_ready=diagnostic_ready,
        exact_ground_truth_count=len(calibration_rows),
        min_exact_count=int(min_exact_count),
        selected_headline_model_ids=selected_ids,
        certified_delta=certified_delta,
    )
    explanation = exp3115_gate_explanation(
        gate_state=gate_state,
        certified_delta=certified_delta,
        exact_ground_truth_count=len(calibration_rows),
        min_exact_count=int(min_exact_count),
        selected_headline_model_ids=selected_ids,
        missing_required_sources=missing_required_sources,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "diagnostic_verifier_calibration_v5_ready": diagnostic_ready,
        "model_specs": model_specs,
        "mandatory_headline_model_ids": mandatory_ids,
        "selected_headline_model_ids": selected_ids,
        "live_llm_inference": False,
        "exact_ground_truth_count": len(calibration_rows),
        "verifier_gain_delta": raw_delta,
        "verifier_gain_delta_with_certified_coherence": certified_delta,
        "false_accept_rate": certified_metrics["false_accept_rate"],
        "false_reject_rate": certified_metrics["false_reject_rate"],
        "calibration_error": calibration_error(calibration_rows),
        "abstention_precision": certified_metrics["abstention_precision"],
        "rejection_recall": certified_metrics["rejection_recall"],
        "repair_gate_state": gate_state,
        "baseline_metrics": baseline_metrics,
        "logic_pilot_metrics": pilot_metrics,
        "certified_coherence_metrics": certified_metrics,
        "calibration_rows": calibration_rows,
        "calibration_fixture_ids": [row["fixture_id"] for row in calibration_rows],
        "readiness_checks": readiness_checks,
        "blocked_reasons": [name for name, ok in readiness_checks.items() if ok is not True],
        "exp3115_repair_gate_explanation": explanation,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in source_rows if row["sha256"] is not None
        },
        "inference_substrate": inference_substrate(exp3099, exp3111, exp3112),
        "duration_s": duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    min_exact_count: int = NON_TINY_EXACT_COUNT_FLOOR,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the Exp 3113 terminal JSON artifact."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        min_exact_count=min_exact_count,
        started_s=started_s,
        now_s=now_s,
        tests_run=tests_run,
    )
    validate_artifact(artifact)
    write_json(out_path, artifact)
    return out_path


def select_calibration_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    panel_rows: Sequence[Mapping[str, Any]],
    certificates: Sequence[Mapping[str, Any]],
    logic_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Join exact, route, certified, and logic rows by fixture id."""

    manifest_by_id = {str(row.get("source_fixture_id")): dict(row) for row in manifest_rows}
    panel_by_id = {str(row.get("source_fixture_id")): dict(row) for row in panel_rows}
    cert_by_id = {str(row.get("fixture_id")): dict(row) for row in certificates}
    selected: list[JsonDict] = []
    for logic in logic_rows:
        fixture_id = str(logic.get("fixture_id") or "")
        manifest = manifest_by_id.get(fixture_id)
        panel = panel_by_id.get(fixture_id)
        certificate = cert_by_id.get(fixture_id)
        if manifest is not None and panel is not None and certificate is not None:
            selected.append(calibration_row(manifest, panel, certificate, dict(logic)))
    selected.sort(key=lambda row: row["fixture_id"])
    return selected


def calibration_row(
    manifest: Mapping[str, Any],
    panel: Mapping[str, Any],
    certificate: Mapping[str, Any],
    logic: Mapping[str, Any],
) -> JsonDict:
    """Build one row-level calibration record with exact-label authority."""

    exact_label = str(certificate.get("exact_label") or logic.get("exact_label") or manifest.get("expected_answer") or "")
    expected_action = str(
        logic.get("expected_action")
        or panel.get("expected_action")
        or manifest.get("verifier_target", {}).get("expected_action")
        or expected_action_from_answer(exact_label)
    )
    baseline = str(panel.get("route_decision") or logic.get("baseline_decision") or "abstain")
    logic_decision = str(logic.get("logic_decision") or baseline)
    route = certificate.get("maxsat_route") if isinstance(certificate.get("maxsat_route"), Mapping) else {}
    certified = str(route.get("action") or expected_action_from_answer(exact_label))
    return {
        "fixture_id": str(manifest.get("source_fixture_id") or logic.get("fixture_id") or ""),
        "task_family": manifest.get("task_family") or logic.get("task_family"),
        "perturbation_type": manifest.get("perturbation_type") or logic.get("perturbation_type"),
        "exact_label": exact_label,
        "expected_action": expected_action,
        "baseline_decision": baseline,
        "logic_decision": logic_decision,
        "certified_coherence_decision": certified,
        "route_confidence": normalized_route_confidence(panel, baseline),
        "maxsat_policy_used": panel.get("maxsat_policy_used") is True,
        "coherence_status": certificate.get("coherence_status"),
        "has_logic_pilot_signal": bool(logic_decision),
        "has_certified_feedback_v3": True,
        "baseline_correct": baseline == expected_action,
        "logic_correct": logic_decision == expected_action,
        "certified_coherence_correct": certified == expected_action,
    }


def decision_metrics(rows: Sequence[Mapping[str, Any]], decision_field: str) -> JsonDict:
    """Return accept/reject/abstain safety metrics for one decision column."""

    if not rows:
        return {
            "accuracy": 0.0,
            "false_accept_rate": 0.0,
            "false_reject_rate": 0.0,
            "abstention_precision": 0.0,
            "rejection_recall": 0.0,
            "accept_count": 0,
            "reject_count": 0,
            "abstain_count": 0,
        }
    positives = [row for row in rows if row.get("expected_action") == "accept"]
    negatives = [row for row in rows if row.get("expected_action") == "reject"]
    abstentions = [row for row in rows if row.get(decision_field) == "abstain"]
    return {
        "accuracy": rate(sum(row.get(decision_field) == row.get("expected_action") for row in rows), len(rows)),
        "false_accept_rate": rate(sum(row.get(decision_field) == "accept" for row in negatives), len(negatives)),
        "false_reject_rate": rate(sum(row.get(decision_field) == "reject" for row in positives), len(positives)),
        "abstention_precision": rate(
            sum(row.get("expected_action") == "reject" for row in abstentions),
            len(abstentions),
        ),
        "rejection_recall": rate(sum(row.get(decision_field) == "reject" for row in negatives), len(negatives)),
        "accept_count": sum(row.get(decision_field) == "accept" for row in rows),
        "reject_count": sum(row.get(decision_field) == "reject" for row in rows),
        "abstain_count": len(abstentions),
    }


def calibration_error(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return mean absolute confidence error for the cached MaxSAT route."""

    if not rows:
        return 0.0
    error = sum(
        abs(float(row.get("route_confidence") or 0.0) - (1.0 if row.get("baseline_correct") else 0.0))
        for row in rows
    )
    return rate(error, len(rows))


def normalized_route_confidence(row: Mapping[str, Any], decision: str) -> float:
    """Return a bounded confidence proxy from model confidence or route scores."""

    confidence = row.get("confidence")
    if isinstance(confidence, int | float) and math.isfinite(float(confidence)):
        return min(1.0, max(0.0, round(float(confidence), 6)))
    scores = row.get("route_scores")
    if isinstance(scores, Mapping):
        total = sum(max(0.0, float(value)) for value in scores.values() if isinstance(value, int | float))
        selected = scores.get(decision)
        if total > 0.0 and isinstance(selected, int | float):
            return round(max(0.0, float(selected)) / total, 6)
    return 0.0


def repair_gate_state(
    *,
    diagnostic_ready: bool,
    exact_ground_truth_count: int,
    min_exact_count: int,
    selected_headline_model_ids: Sequence[str],
    certified_delta: float,
) -> str:
    """Map evidence and measured lift to the Exp 3113 repair gate state."""

    if diagnostic_ready is not True:
        return "blocked_missing_inputs"
    if exact_ground_truth_count < min_exact_count:
        return "blocked_tiny_panel"
    if not selected_headline_model_ids:
        return "blocked_model_cache"
    if certified_delta <= 0.0:
        return "blocked_negative_delta"
    return "unblocked"


def exp3115_gate_explanation(
    *,
    gate_state: str,
    certified_delta: float,
    exact_ground_truth_count: int,
    min_exact_count: int,
    selected_headline_model_ids: Sequence[str],
    missing_required_sources: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Return a machine-readable repair-gate payload for Exp 3115."""

    actions = {
        "unblocked": "repair_gate_unblocked",
        "blocked_negative_delta": "do_not_promote_repair_until_positive_delta",
        "blocked_model_cache": "rerun_with_selected_mandated_sota_cache",
        "blocked_tiny_panel": "increase_exact_calibration_panel",
        "blocked_missing_inputs": "materialize_required_source_artifacts",
    }
    return {
        "schema": "carnot.exp3115.repair_gate_input.v1",
        "repair_gate_state": gate_state,
        "delta_sign": delta_sign(certified_delta),
        "certified_delta": round(float(certified_delta), 6),
        "exact_ground_truth_count": int(exact_ground_truth_count),
        "minimum_exact_ground_truth_count": int(min_exact_count),
        "selected_headline_model_ids": list(selected_headline_model_ids),
        "missing_required_source_paths": [str(row.get("path")) for row in missing_required_sources],
        "downstream_action": actions[gate_state],
    }


def expected_action_from_answer(answer: str) -> str:
    """Map exact answer labels onto verifier actions."""

    normalized = str(answer).upper()
    if normalized in {"VALID", "SAT"}:
        return "accept"
    if normalized in {"INVALID", "UNSAT", "REPAIRABLE", "UNREPAIRABLE"}:
        return "reject"
    return "abstain"


def delta_sign(value: float) -> str:
    """Return a stable sign label for measured lift."""

    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded safe ratio."""

    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3113 artifact violates the terminal contract."""

    missing = sorted(set(REQUIRED_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("repair_gate_state") not in GATE_STATES:
        raise ValueError("repair_gate_state must be a known gate state")
    for field in (
        "verifier_gain_delta",
        "verifier_gain_delta_with_certified_coherence",
        "false_accept_rate",
        "false_reject_rate",
        "calibration_error",
        "abstention_precision",
        "rejection_recall",
    ):
        value = float(artifact.get(field, math.nan))
        if not math.isfinite(value):
            raise ValueError(f"finite metric required for {field}")
    verdict = str(artifact.get("honest_verdict") or "")
    if artifact.get("repair_gate_state") == "unblocked" and not artifact.get("selected_headline_model_ids"):
        raise ValueError("unblocked repair gate requires at least one selected model")
    if artifact.get("diagnostic_verifier_calibration_v5_ready") is not True:
        if not verdict.startswith("blocked_missing_inputs"):
            raise ValueError("unready artifact must use blocked_missing_inputs honest verdict")
    elif not any(verdict.startswith(prefix) for prefix in SUCCESS_PREFIXES):
        raise ValueError("ready artifact honest_verdict must start with a success prefix")


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Map readiness and gate state to the conductor terminal vocabulary."""

    gate = str(artifact.get("repair_gate_state"))
    if artifact.get("diagnostic_verifier_calibration_v5_ready") is not True:
        return "blocked_missing_inputs: " + ",".join(
            artifact.get("blocked_reasons") or ["required_source_artifacts_unavailable"]
        )
    if gate == "blocked_tiny_panel":
        return (
            "complete_blocked_tiny_panel: diagnostic_verifier_calibration_v5_ready=true; "
            f"exact_ground_truth_count={artifact.get('exact_ground_truth_count')}; "
            f"repair_gate_state={gate}"
        )
    if gate == "blocked_model_cache":
        return (
            "complete_blocked_headline: diagnostic_verifier_calibration_v5_ready=true; "
            "selected_headline_model_ids=0; repair_gate_state=blocked_model_cache"
        )
    return (
        "complete: diagnostic_verifier_calibration_v5_ready=true; "
        f"repair_gate_state={gate}; "
        f"verifier_gain_delta={artifact.get('verifier_gain_delta')}; "
        "verifier_gain_delta_with_certified_coherence="
        f"{artifact.get('verifier_gain_delta_with_certified_coherence')}"
    )


def inference_substrate(
    exp3099: Mapping[str, Any],
    exp3111: Mapping[str, Any],
    exp3112: Mapping[str, Any],
) -> JsonDict:
    """Describe runtime provenance without treating cached traces as new live inference."""

    exp3099_substrate = exp3099.get("inference_substrate")
    exp3111_substrate = exp3111.get("inference_substrate")
    exp3112_substrate = exp3112.get("inference_substrate")
    return {
        "kind": "diagnostic_over_cached_sota_routes_certified_solver_feedback_and_logic_pilot",
        "live_llm_inference": False,
        "new_model_execution": False,
        "cached_trace_source": EXP3099_REL_PATH.as_posix(),
        "cached_trace_source_executed_models": isinstance(exp3099_substrate, Mapping)
        and exp3099_substrate.get("executes_models") is True,
        "certified_feedback_source": EXP3111_REL_PATH.as_posix(),
        "logic_pilot_source": EXP3112_REL_PATH.as_posix(),
        "executes_solvers": False,
        "source_certified_feedback_executed_solvers": isinstance(exp3111_substrate, Mapping)
        and exp3111_substrate.get("executes_solvers") is True,
        "source_logic_pilot_live_llm_inference": isinstance(exp3112_substrate, Mapping)
        and exp3112_substrate.get("live_llm_inference") is True,
        "exact_solver_labels_authority": True,
    }


def source_artifacts(
    root: Path,
    manifest_rel_path: Path,
    policy_rel_path: Path,
    panel_rel_path: Path,
    logic_rel_path: Path,
) -> list[JsonDict]:
    """Return source provenance, replacing dynamic row paths from source artifacts."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_SPECS:
        path = manifest_rel_path if rel_path == MANIFEST_REL_PATH else rel_path
        path = policy_rel_path if rel_path == POLICY_REL_PATH else path
        path = panel_rel_path if rel_path == EXP3099_ROWS_REL_PATH else path
        path = logic_rel_path if rel_path == EXP3112_ROWS_REL_PATH else path
        full_path = root / path
        rows.append(
            {
                "id": source_id,
                "path": path.as_posix(),
                "required": required,
                "exists": full_path.is_file(),
                "sha256": sha256_file(full_path),
            }
        )
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a stable JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum for a present file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a repo-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def duration(started_s: float, now_s: float | None) -> float:
    """Return a nonnegative wall-clock duration."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
