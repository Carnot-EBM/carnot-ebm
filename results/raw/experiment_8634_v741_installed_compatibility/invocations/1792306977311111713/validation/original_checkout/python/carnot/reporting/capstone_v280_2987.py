"""Build the Exp 2987 milestone .280 capstone artifact.

Spec refs: REQ-REPORT-2987, SCENARIO-REPORT-2987.

This module is a terminal aggregation layer. It reads the checked-in .280 JSON
artifacts and matrix v14, then decides paper readiness from local evidence and
claim-boundary compliance only. It does not run models, solvers, synthesis,
board commands, smoke tests, the conductor, or documentation reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260524"
MILESTONE = "2026.05.280"
SCHEMA = "carnot.milestone_capstone.v280_terminal.v1"
ARTIFACT = "experiment_2987_capstone_v280"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2987_capstone_v280.json")

EXP2975_REL_PATH = Path("results/experiment_2975_archive_v279_activate_v280.json")
EXP2976_REL_PATH = Path("results/experiment_2976_dccd_adaptrack_tracecoder_protocol_v1.json")
EXP2977_REL_PATH = Path("results/experiment_2977_sota_intent_preserving_code_repair_v1.json")
EXP2978_REL_PATH = Path("results/experiment_2978_first_step_semantic_energy_repair_telemetry_v1.json")
EXP2979_REL_PATH = Path("results/experiment_2979_solver_feedback_mcs_frontier_v1.json")
EXP2980_REL_PATH = Path("results/experiment_2980_sota_solver_formalization_feedback_v2.json")
EXP2981_REL_PATH = Path("results/experiment_2981_interwhen_partial_monitor_promotion_v2.json")
EXP2982_REL_PATH = Path("results/experiment_2982_fr11_independent_metric_utility_gate_v4.json")
EXP2983_REL_PATH = Path("results/experiment_2983_trace_to_skill_repair_memory_pilot_v1.json")
EXP2984_REL_PATH = Path("results/experiment_2984_gatemate_readback_smoke_vector_v4.json")
EXP2985_REL_PATH = Path("results/experiment_2985_ssqa_dual_bram_register_map_plan_v1.json")
MATRIX_V14_REL_PATH = Path("results/experiment_2986_cross_corpus_matrix_v14.json")
EXP2986_REL_PATH = MATRIX_V14_REL_PATH

CLASSIFICATIONS = (
    "clean",
    "flagged",
    "blocked",
    "missing",
    "gated-skipped",
    "pilot-only",
    "projection-only",
)

HARDWARE_CLAIM_FIELDS = (
    "sampler_claim_allowed",
    "speedup_claim_allowed",
    "thermodynamic_claim_allowed",
)

MATRIX_ROW_IDS = {
    "exp2975": "exp2975_archive_activation",
    "exp2976": "exp2976_repair_protocol",
    "exp2977": "exp2977_intent_preserving_repair",
    "exp2978": "exp2978_semantic_energy_telemetry",
    "exp2979": "exp2979_solver_feedback_frontier",
    "exp2980": "exp2980_solver_formalization_feedback",
    "exp2981": "exp2981_partial_monitor_promotion",
    "exp2982": "exp2982_fr11_independent_metric_gate",
    "exp2983": "exp2983_trace_to_skill_memory_pilot",
    "exp2984": "exp2984_gatemate_readback_smoke",
    "exp2985": "exp2985_ssqa_register_map_plan",
}

REQUIRED_CLAIM_FIELDS = {
    "exp2975": ("archive_ready",),
    "exp2976": (
        "intent_preserving_repair_protocol_ready",
        "trace_execution_plan_ready",
        "prior_failure_addressed",
    ),
    "exp2977": (
        "repair_rerun_clean",
        "headline_result",
        "n_tasks",
        "pass_at_1_delta",
        "pass_at_k_delta",
        "schema_failure_rate_delta",
        "syntax_failure_rate_delta",
        "false_accept_delta",
        "runtime_trace_coverage",
    ),
    "exp2978": (
        "telemetry_panel_ready",
        "semantic_energy_signal_usable",
        "first_step_signal_usable",
        "no_headline_verifier_claim",
    ),
    "exp2979": (
        "mcs_feedback_schema_ready",
        "frontier_upgrade_ready",
        "reference_solver_verified_accuracy",
        "reference_z3_execution_rate",
    ),
    "exp2980": (
        "formalization_feedback_clean",
        "headline_result",
        "n_items",
        "parseability_rate",
        "solver_verified_accuracy",
        "z3_execution_rate",
        "tautology_flag_rate",
        "feedback_repair_delta",
    ),
    "exp2981": (
        "partial_monitor_promoted",
        "fixture_count",
        "live_trace_count",
        "prefix_failure_localization_rate",
        "false_alarm_rate",
        "full_streaming_verification_claim",
    ),
    "exp2982": (
        "continuous_self_learning_task",
        "fr11_independent_metrics_evaluated",
        "fr11_independent_self_learning_ready",
        "no_identical_metric_flag",
        "forgetting_guard_passed",
        "heldout_independent_delta_vs_random",
        "negative_control_delta",
    ),
    "exp2983": (
        "trace_to_skill_memory_ready",
        "continuous_self_learning_task",
        "heldout_skill_reuse_delta",
        "negative_control_delta",
        "leakage_flag",
        "headline_result",
        "fresh_live_llm_inference_used",
    ),
    "exp2984": (
        "board_detected",
        "flash_succeeded",
        "readback_attempted",
        "readback_supported",
        "readback_hash",
        "smoke_vector_attempted",
        "smoke_vector_passed",
        "sampler_claim_allowed",
        "speedup_claim_allowed",
        "thermodynamic_claim_allowed",
    ),
    "exp2985": (
        "register_map_plan_ready",
        "projection_only",
        "sampler_claim_allowed",
        "speedup_claim_allowed",
        "thermodynamic_claim_allowed",
    ),
    "exp2986": (
        "matrix_v14_ready",
        "claim_boundary_violations",
        "repair_claim_status",
        "solver_claim_status",
        "fr11_claim_status",
        "hardware_claim_status",
    ),
}


@dataclass(frozen=True)
class SourceSpec:
    """A milestone source artifact that the capstone audits."""

    experiment_id: str
    path: Path


SOURCE_SPECS = (
    SourceSpec("exp2975", EXP2975_REL_PATH),
    SourceSpec("exp2976", EXP2976_REL_PATH),
    SourceSpec("exp2977", EXP2977_REL_PATH),
    SourceSpec("exp2978", EXP2978_REL_PATH),
    SourceSpec("exp2979", EXP2979_REL_PATH),
    SourceSpec("exp2980", EXP2980_REL_PATH),
    SourceSpec("exp2981", EXP2981_REL_PATH),
    SourceSpec("exp2982", EXP2982_REL_PATH),
    SourceSpec("exp2983", EXP2983_REL_PATH),
    SourceSpec("exp2984", EXP2984_REL_PATH),
    SourceSpec("exp2985", EXP2985_REL_PATH),
    SourceSpec("exp2986", EXP2986_REL_PATH),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk, returning an empty object on any failure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source-file digest so the capstone can prove what it read."""

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
) -> dict[str, Any]:
    """REQ-REPORT-2987: synthesize the terminal .280 capstone."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    matrix = payloads.get("exp2986", {})
    matrix_rows = _matrix_rows_by_experiment(matrix)
    audit_rows = _artifact_audit(root_path, payloads, matrix_rows)
    buckets = _classification_buckets(audit_rows)
    hardware_summary = _hardware_claim_boundary_summary(audit_rows, matrix, payloads)

    repair_ready = _repair_ready(payloads.get("exp2977", {}), audit_rows)
    solver_ready = _solver_ready(payloads.get("exp2979", {}), payloads.get("exp2980", {}), audit_rows)
    fr11_ready = _fr11_ready(payloads.get("exp2982", {}), audit_rows)
    hardware_ready = _hardware_ready(payloads.get("exp2984", {}), audit_rows, hardware_summary)
    status_words = {
        "repair": _readiness_word(repair_ready, _classification_for("exp2977", audit_rows)),
        "solver": _readiness_word(solver_ready, _classification_for("exp2980", audit_rows)),
        "fr11": _readiness_word(fr11_ready, _classification_for("exp2982", audit_rows)),
        "hardware": _readiness_word(hardware_ready, _classification_for("exp2984", audit_rows)),
    }

    paper_blockers = _paper_ready_blockers(
        repair_ready,
        solver_ready,
        fr11_ready,
        hardware_ready,
        buckets,
        hardware_summary,
        bool(matrix.get("matrix_v14_ready")),
    )
    paper_ready = not paper_blockers
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)
    if started_s is None and now_s is None and duration_s == 0.0:  # pragma: no cover
        duration_s = 0.000001

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": _honest_verdict(paper_ready, buckets),
        "milestone": MILESTONE,
        "paper_ready": paper_ready,
        "headline_outcome": _headline_outcome(
            paper_ready,
            status_words,
        ),
        "clean_artifacts": buckets["clean"],
        "flagged_artifacts": buckets["flagged"],
        "blocked_artifacts": buckets["blocked"],
        "missing_artifacts": buckets["missing"],
        "gated_skipped_artifacts": buckets["gated-skipped"],
        "pilot_only_artifacts": buckets["pilot-only"],
        "projection_only_artifacts": buckets["projection-only"],
        "artifact_classification_counts": {name: len(buckets[name]) for name in CLASSIFICATIONS},
        "artifact_audit": audit_rows,
        "matrix_v14_ready": bool(matrix.get("matrix_v14_ready")),
        "matrix_v14_honest_verdict": str(matrix.get("honest_verdict") or ""),
        "matrix_row_classification_counts": _matrix_counts(matrix),
        "repair_ready": repair_ready,
        "solver_ready": solver_ready,
        "fr11_ready": fr11_ready,
        "hardware_ready": hardware_ready,
        "paper_ready_blockers": paper_blockers,
        "gaps_closed": _gaps_closed(repair_ready, solver_ready, fr11_ready, hardware_ready, buckets),
        "gaps_remaining": _gaps_remaining(
            repair_ready,
            solver_ready,
            fr11_ready,
            hardware_ready,
            buckets,
            matrix,
        ),
        "model_compliance_summary": _model_compliance_summary(audit_rows, matrix),
        "hardware_claim_boundary_summary": hardware_summary,
        "retirement_recommendations": _retirement_recommendations(
            repair_ready,
            solver_ready,
            hardware_ready,
            buckets,
        ),
        "next_milestone_recommendations": _next_milestone_recommendations(matrix),
        "source_artifacts_read": _source_rows(root_path),
        "source_checksums": {
            spec.path.as_posix(): sha256_file(root_path / spec.path) for spec in SOURCE_SPECS
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": duration_s,
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2987 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, dict[str, Any]]:
    return {spec.experiment_id: read_json_object(root / spec.path) for spec in SOURCE_SPECS}


def _source_rows(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": spec.experiment_id,
            "path": spec.path.as_posix(),
            "present": (root / spec.path).is_file(),
            "readable_json_object": bool(read_json_object(root / spec.path)),
            "sha256": sha256_file(root / spec.path),
        }
        for spec in SOURCE_SPECS
    ]


def _matrix_rows_by_experiment(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = matrix.get("rows")
    if not isinstance(rows, list):
        return {}
    by_experiment: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        experiment_id = str(row.get("source_experiment_id") or "")
        if experiment_id and experiment_id not in by_experiment:
            by_experiment[experiment_id] = row
    return by_experiment


def _artifact_audit(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    matrix_rows: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        payload = payloads.get(spec.experiment_id, {})
        matrix_row = matrix_rows.get(spec.experiment_id, {})
        present = (root / spec.path).is_file()
        classification = _classify_artifact(spec.experiment_id, payload, matrix_row, present)
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": present,
                "readable_json_object": bool(payload),
                "classification": classification,
                "classification_source": "matrix_v14_row" if matrix_row else "local_capstone_rule",
                "honest_verdict": str(payload.get("honest_verdict") or ""),
                "required_claim_fields": _required_claim_fields(spec.experiment_id, payload),
                "model_compliance": _row_mapping(matrix_row, "model_compliance")
                or _model_compliance_from_payload(payload),
                "hardware_compliance": _row_mapping(matrix_row, "hardware_compliance")
                or _hardware_compliance_from_payload(payload),
                "hardware_claim_fields": {
                    field: payload.get(field) for field in HARDWARE_CLAIM_FIELDS if field in payload
                },
                "prior_failure_outcome": _prior_failure_outcome(spec.experiment_id, payload, matrix_row),
                "matrix_row_id": str(matrix_row.get("row_id") or ""),
                "claim_boundary_guard_passed": matrix_row.get("claim_boundary_guard_passed", True)
                is True,
                "claim_boundary_violations": _list_of_dicts(
                    matrix_row.get("claim_boundary_violations")
                ),
                "upstream_flags": _string_list(matrix_row.get("upstream_flags"))
                or _flag_kinds(payload),
                "sha256": sha256_file(root / spec.path),
            }
        )
    return rows


def _classify_artifact(
    experiment_id: str,
    payload: Mapping[str, Any],
    matrix_row: Mapping[str, Any],
    present: bool,
) -> str:
    if not present or not payload:
        matrix_status = str(matrix_row.get("status") or "")
        return "gated-skipped" if matrix_status == "gated-skipped" else "missing"
    matrix_status = str(matrix_row.get("status") or "")
    if matrix_status in CLASSIFICATIONS and matrix_status != "missing":
        return matrix_status
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    return _status_from_payload(experiment_id, payload)


def _status_from_payload(experiment_id: str, payload: Mapping[str, Any]) -> str:
    if experiment_id == "exp2975" and payload.get("archive_ready") is True:
        return "projection-only"
    if experiment_id == "exp2976" and payload.get("intent_preserving_repair_protocol_ready") is True:
        return "projection-only"
    if experiment_id == "exp2977" and payload.get("repair_rerun_clean") is True:
        return "clean"
    if experiment_id == "exp2978" and payload.get("telemetry_panel_ready") is True:
        return "pilot-only"
    if experiment_id == "exp2979" and payload.get("mcs_feedback_schema_ready") is True:
        return "clean"
    if experiment_id == "exp2980" and payload.get("formalization_feedback_clean") is True:
        return "clean"
    if experiment_id == "exp2981" and payload.get("partial_monitor_promoted") is True:
        return "clean"
    if experiment_id == "exp2982" and payload.get("fr11_independent_self_learning_ready") is True:
        return "clean"
    if experiment_id == "exp2983" and payload.get("trace_to_skill_memory_ready") is True:
        return "pilot-only"
    if experiment_id == "exp2984" and (
        bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True
    ):
        return "clean"
    if experiment_id == "exp2985" and payload.get("register_map_plan_ready") is True:
        return "projection-only"
    if experiment_id == "exp2986" and payload.get("matrix_v14_ready") is True:
        return "projection-only"
    return "blocked"


def _classification_buckets(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    buckets = {name: [] for name in CLASSIFICATIONS}
    for row in rows:
        classification = str(row["classification"])
        if classification in buckets:
            buckets[classification].append(str(row["experiment_id"]))
    return buckets


def _required_claim_fields(experiment_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {field: payload.get(field) for field in REQUIRED_CLAIM_FIELDS.get(experiment_id, ())}


def _prior_failure_outcome(
    experiment_id: str,
    payload: Mapping[str, Any],
    matrix_row: Mapping[str, Any],
) -> str:
    if matrix_row.get("prior_failure_outcome"):
        return str(matrix_row["prior_failure_outcome"])
    if experiment_id == "exp2986" and payload:
        return "matrix_v14_terminal_aggregation"
    return "matrix_v14_missing"


def _row_mapping(row: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _model_compliance_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    mandatory = _string_list(payload.get("mandatory_headline_model_ids"))
    models = _string_list(payload.get("models_used"))
    if not mandatory:
        return {"status": "not_applicable"}
    if payload.get("legacy_model_used_only_for_smoke") is True:
        return {"status": "legacy_smoke_only", "models_used": models, "mandatory": mandatory}
    if not set(models).intersection(mandatory):
        return {
            "status": "non_compliant_missing_mandated_model",
            "models_used": models,
            "mandatory": mandatory,
        }
    if _has_flags(payload):
        return {
            "status": "flagged_mandated_model_evidence",
            "models_used": models,
            "mandatory": mandatory,
        }
    return {"status": "compliant", "models_used": models, "mandatory": mandatory}


def _hardware_compliance_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    disallowed = _unsupported_hardware_claim_fields(payload)
    if disallowed:
        return {"status": "claim_boundary_violation", "fields": disallowed}
    if payload.get("projection_only") is True:
        return {"status": "projection_only"}
    if payload.get("inference_substrate") == "physical_gatemate_board":
        if bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True:
            return {"status": "compliant_hardware_readback_or_smoke"}
        return {"status": "blocked_no_readback_or_smoke_output"}
    return {"status": "not_applicable"}


def _repair_ready(payload: Mapping[str, Any], audit_rows: list[dict[str, Any]]) -> bool:
    return (
        _classification_for("exp2977", audit_rows) == "clean"
        and payload.get("repair_rerun_clean") is True
        and payload.get("headline_result") is True
        and (_coerce_int(payload.get("n_tasks")) or 0) >= 20
        and (_coerce_float(payload.get("pass_at_1_delta")) or 0.0) > 0.0
        and (_coerce_float(payload.get("pass_at_k_delta")) or 0.0) >= 0.0
        and (_coerce_float(payload.get("schema_failure_rate_delta")) or 0.0) <= 0.0
        and (_coerce_float(payload.get("syntax_failure_rate_delta")) or 0.0) <= 0.0
        and (_coerce_float(payload.get("false_accept_delta")) or 0.0) <= 0.0
        and (_coerce_float(payload.get("runtime_trace_coverage")) or 0.0) >= 0.8
        and not _has_flags(payload)
    )


def _solver_ready(
    frontier: Mapping[str, Any],
    formalization: Mapping[str, Any],
    audit_rows: list[dict[str, Any]],
) -> bool:
    tautology_flag_rate = _coerce_float(formalization.get("tautology_flag_rate"))
    return (
        _classification_for("exp2979", audit_rows) == "clean"
        and _classification_for("exp2980", audit_rows) == "clean"
        and frontier.get("mcs_feedback_schema_ready") is True
        and frontier.get("frontier_upgrade_ready") is True
        and formalization.get("formalization_feedback_clean") is True
        and (_coerce_float(formalization.get("parseability_rate")) or 0.0) >= 0.5
        and (_coerce_float(formalization.get("solver_verified_accuracy")) or 0.0) >= 0.4
        and (_coerce_float(formalization.get("z3_execution_rate")) or 0.0) >= 0.5
        and tautology_flag_rate == 0.0
        and not _has_flags(formalization)
    )


def _fr11_ready(payload: Mapping[str, Any], audit_rows: list[dict[str, Any]]) -> bool:
    return (
        _classification_for("exp2982", audit_rows) == "clean"
        and payload.get("fr11_independent_metrics_evaluated") is True
        and payload.get("fr11_independent_self_learning_ready") is True
        and payload.get("no_identical_metric_flag") is True
        and payload.get("forgetting_guard_passed") is True
        and not _has_flags(payload)
    )


def _hardware_ready(
    payload: Mapping[str, Any],
    audit_rows: list[dict[str, Any]],
    hardware_summary: Mapping[str, Any],
) -> bool:
    return (
        _classification_for("exp2984", audit_rows) == "clean"
        and (bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True)
        and not hardware_summary.get("claim_boundary_violations")
        and not hardware_summary.get("unsupported_claim_fields_by_artifact")
    )


def _classification_for(experiment_id: str, audit_rows: list[dict[str, Any]]) -> str:
    for row in audit_rows:
        if row.get("experiment_id") == experiment_id:
            return str(row.get("classification") or "")
    return ""


def _hardware_claim_boundary_summary(
    audit_rows: list[dict[str, Any]],
    matrix: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    unsupported = {
        experiment_id: fields
        for experiment_id, fields in (
            (spec.experiment_id, _unsupported_hardware_claim_fields(payloads.get(spec.experiment_id, {})))
            for spec in SOURCE_SPECS
        )
        if fields
    }
    violations = _list_of_dicts(matrix.get("claim_boundary_violations"))
    if not violations:
        for row in audit_rows:
            violations.extend(_list_of_dicts(row.get("claim_boundary_violations")))
    exp2984 = payloads.get("exp2984", {})
    return {
        "hardware_claim_status": str(matrix.get("hardware_claim_status") or ""),
        "claim_boundary_violations": violations,
        "unsupported_claim_fields_by_artifact": unsupported,
        "gatemate_readback_or_smoke_present": bool(exp2984.get("readback_hash"))
        or exp2984.get("smoke_vector_passed") is True,
        "gatemate_board_detected": bool(exp2984.get("board_detected")),
        "gatemate_flash_succeeded": bool(exp2984.get("flash_succeeded")),
        "hardware_rows": [
            row["experiment_id"]
            for row in audit_rows
            if row.get("experiment_id") in {"exp2984", "exp2985"}
        ],
    }


def _unsupported_hardware_claim_fields(payload: Mapping[str, Any]) -> list[str]:
    return [field for field in HARDWARE_CLAIM_FIELDS if payload.get(field) is True]


def _paper_ready_blockers(
    repair_ready: bool,
    solver_ready: bool,
    fr11_ready: bool,
    hardware_ready: bool,
    buckets: Mapping[str, list[str]],
    hardware_summary: Mapping[str, Any],
    matrix_ready: bool,
) -> list[str]:
    blockers: list[str] = []
    if not matrix_ready:
        blockers.append("matrix_v14_not_ready")
    if not repair_ready:
        blockers.append("repair_not_ready")
    if not solver_ready:
        blockers.append("solver_not_ready")
    if not fr11_ready:
        blockers.append("fr11_not_ready")
    if not hardware_ready:
        blockers.append("hardware_not_ready")
    if buckets["flagged"]:
        blockers.append("flagged_artifacts_present")
    if buckets["blocked"]:
        blockers.append("blocked_artifacts_present")
    if buckets["missing"]:
        blockers.append("missing_artifacts_present")
    if buckets["gated-skipped"]:
        blockers.append("gated_skipped_artifacts_present")
    if hardware_summary.get("claim_boundary_violations") or hardware_summary.get(
        "unsupported_claim_fields_by_artifact"
    ):
        blockers.append("claim_boundary_violations_present")
    return blockers


def _headline_outcome(
    paper_ready: bool,
    status_words: Mapping[str, str],
) -> str:
    if paper_ready:
        return "paper_ready: .280 cleared repair, solver, FR-11, and GateMate local-evidence gates"
    return (
        "not_paper_ready: "
        f"repair={status_words['repair']}, "
        f"solver={status_words['solver']}, "
        f"fr11={status_words['fr11']}, "
        f"hardware={status_words['hardware']}"
    )


def _readiness_word(ready: bool, classification: str) -> str:
    if ready:
        return "clean"
    if classification in {"flagged", "blocked", "missing", "gated-skipped", "pilot-only", "projection-only"}:
        return classification
    return "blocked"


def _honest_verdict(paper_ready: bool, buckets: Mapping[str, list[str]]) -> str:
    return (
        "complete: milestone_280_capstone; "
        f"paper_ready={str(paper_ready).lower()}; "
        f"clean={len(buckets['clean'])}; "
        f"flagged={len(buckets['flagged'])}; "
        f"blocked={len(buckets['blocked'])}; "
        f"missing={len(buckets['missing'])}"
    )


def _gaps_closed(
    repair_ready: bool,
    solver_ready: bool,
    fr11_ready: bool,
    hardware_ready: bool,
    buckets: Mapping[str, list[str]],
) -> list[str]:
    closed: list[str] = []
    if "exp2975" in buckets["projection-only"]:
        closed.append("Archive/activation bookkeeping for .280 is complete.")
    if repair_ready:
        closed.append("Intent-preserving repair cleared live headline gates.")
    if "exp2979" in buckets["clean"]:
        closed.append("Solver MCS/MUS feedback frontier is clean deterministic evidence.")
    if solver_ready:
        closed.append("Solver formalization feedback is paper-ready without unresolved flags.")
    if "exp2981" in buckets["clean"]:
        closed.append("Partial monitor promotion landed with measured localization and no full-streaming claim.")
    if fr11_ready:
        closed.append("FR-11 independent metric self-learning gate is clean.")
    if hardware_ready:
        closed.append("GateMate hardware readback or smoke-vector evidence is clean.")
    if "exp2985" in buckets["projection-only"]:
        closed.append("SSQA dual-BRAM register-map plan landed as projection-only evidence.")
    return closed


def _gaps_remaining(
    repair_ready: bool,
    solver_ready: bool,
    fr11_ready: bool,
    hardware_ready: bool,
    buckets: Mapping[str, list[str]],
    matrix: Mapping[str, Any],
) -> list[str]:
    gaps: list[str] = []
    if not repair_ready:
        gaps.append("Repair is not paper-ready: the intent-preserving rerun did not clear cached-SOTA, n>=20, and positive-delta gates.")
    if not solver_ready:
        gaps.append("Solver feedback is not paper-ready: numeric Z3 gains must be reproduced without artifact flags.")
    if not fr11_ready:
        gaps.append("FR-11 is not paper-ready: independent metrics or forgetting guards are missing.")
    if not hardware_ready:
        gaps.append("GateMate is not hardware-ready: readback hash or passed host-visible smoke vector is absent.")
    if buckets["flagged"]:
        gaps.append(f"Flagged .280 artifacts remain: {', '.join(buckets['flagged'])}.")
    if buckets["blocked"]:
        gaps.append(f"Blocked .280 artifacts remain: {', '.join(buckets['blocked'])}.")
    if buckets["missing"]:
        gaps.append(f"Missing .280 artifacts remain: {', '.join(buckets['missing'])}.")
    if buckets["gated-skipped"]:
        gaps.append(f"Gated-skipped .280 artifacts remain: {', '.join(buckets['gated-skipped'])}.")
    if not matrix.get("matrix_v14_ready"):
        gaps.append("Matrix v14 is missing or not ready, so cross-corpus closeout cannot be paper evidence.")
    elif matrix.get("flagged_count") or matrix.get("blocked_count"):
        gaps.append(
            "Matrix v14 still contains flagged or blocked rows: "
            f"flagged={matrix.get('flagged_count')}, blocked={matrix.get('blocked_count')}."
        )
    return _unique_strings(gaps)


def _retirement_recommendations(
    repair_ready: bool,
    solver_ready: bool,
    hardware_ready: bool,
    buckets: Mapping[str, list[str]],
) -> list[str]:
    recommendations: list[str] = []
    if not repair_ready:
        recommendations.append(
            "Retire CPU-smoke-only intent-preserving repair reruns as headline candidates; rerun only when cached_sota_pair supplies mandated local GGUF models and n>=20."
        )
        recommendations.append(
            "Retire schema-only DCCD promotion claims; keep protocol-only repair artifacts out of paper claims until methodology flags clear."
        )
    if not solver_ready:
        recommendations.append(
            "Retire Exp 2980 as paper evidence despite clean numeric Z3 metrics until duration provenance and reproducibility checksum flags are cleared."
        )
    if not hardware_ready:
        recommendations.append(
            "Retire GateMate sampler, speedup, thermodynamic, and Boltzmann claims until readback or host-visible smoke-vector IO exists."
        )
    if "exp2983" in buckets["flagged"] or "exp2983" in buckets["pilot-only"]:
        recommendations.append(
            "Retire trace-to-skill memory headline claims from artifact replay; keep the scope pilot-only until a fresh held-out live rerun clears flags."
        )
    return recommendations


def _next_milestone_recommendations(matrix: Mapping[str, Any]) -> list[str]:
    local = [
        "Repair: restore mandated cached SOTA availability before spending another milestone on live repair reruns.",
        "Solver: rerun the feedback formalization with reproducibility checksum and enough wall-clock provenance to clear adversarial flags.",
        "FR-11: preserve independent held-out metrics, random replay, negative controls, and forgetting guards as mandatory gates.",
        "GateMate: implement host-visible readback or smoke-vector IO before any sampler-facing claim.",
        "SSQA: move the dual-BRAM register-map plan to RTL, PnR, resource accounting, and readback evidence before promotion.",
    ]
    matrix_recs = _string_list(matrix.get("next_milestone_recommendations"))
    return _unique_strings([*matrix_recs, *local])


def _model_compliance_summary(audit_rows: list[dict[str, Any]], matrix: Mapping[str, Any]) -> dict[str, int]:
    matrix_summary = matrix.get("model_compliance_summary")
    if isinstance(matrix_summary, Mapping):
        return {str(key): int(value) for key, value in matrix_summary.items() if isinstance(value, int)}
    statuses = [
        str(row.get("model_compliance", {}).get("status"))
        for row in audit_rows
        if isinstance(row.get("model_compliance"), Mapping)
    ]
    return {status: statuses.count(status) for status in sorted(set(statuses))}


def _matrix_counts(matrix: Mapping[str, Any]) -> dict[str, int]:
    return {
        "row_count": int(matrix.get("row_count") or 0),
        "clean": int(matrix.get("clean_count") or 0),
        "flagged": int(matrix.get("flagged_count") or 0),
        "blocked": int(matrix.get("blocked_count") or 0),
        "gated-skipped": int(matrix.get("gated_skipped_count") or 0),
        "pilot-only": int(matrix.get("pilot_only_count") or 0),
        "projection-only": int(matrix.get("projection_only_count") or 0),
    }


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().lower().startswith(
        ("blocked", "gate_blocked")
    )


def _has_flags(payload: Mapping[str, Any]) -> bool:
    if payload.get("flagged_adversarial") is True:
        return True
    flags = payload.get("corrigendum_pending")
    return isinstance(flags, list) and bool(flags)


def _flag_kinds(payload: Mapping[str, Any]) -> list[str]:
    kinds: list[str] = []
    if payload.get("flagged_adversarial") is True:
        kinds.append("flagged_adversarial=true")
    flags = payload.get("corrigendum_pending")
    if isinstance(flags, list):
        for item in flags:
            if isinstance(item, Mapping):
                kinds.append(str(item.get("kind") or "unknown"))
    return _unique_strings(kinds)


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_path(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())
