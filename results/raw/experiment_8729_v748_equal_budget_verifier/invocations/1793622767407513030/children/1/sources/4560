"""Build the Exp 2986 cross-corpus matrix v14 artifact.

Spec refs: REQ-REPORT-2986, SCENARIO-REPORT-2986.

This module is a narrow aggregation layer. It reads matrix v13, the .279
capstone, and the checked-in .280 artifacts, then emits explicit claim-boundary
rows. It does not rerun model inference, verifier scoring, solver execution,
synthesis, board flashing, readback, or hardware smoke tests.
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
SCHEMA = "carnot.cross_corpus_matrix.v14_280_claim_boundary.v1"
ARTIFACT = "experiment_2986_cross_corpus_matrix_v14"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_2986_cross_corpus_matrix_v14.json")

MATRIX_V13_REL_PATH = Path("results/experiment_2973_cross_corpus_matrix_v13.json")
CAPSTONE_V279_REL_PATH = Path("results/experiment_2974_capstone_v279.json")
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

STATUSES = (
    "clean",
    "flagged",
    "blocked",
    "gated-skipped",
    "pilot-only",
    "projection-only",
)

PRIOR_BUCKETS = (
    ("clean_rows", "clean"),
    ("flagged_rows", "flagged"),
    ("blocked_rows", "blocked"),
    ("gated_skipped_rows", "gated-skipped"),
    ("pilot_only_rows", "pilot-only"),
    ("aggregation_only_rows", "projection-only"),
)

HARDWARE_CLAIM_FIELDS = (
    "sampler_claim_allowed",
    "speedup_claim_allowed",
    "thermodynamic_claim_allowed",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    path: Path
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp2973", MATRIX_V13_REL_PATH, required=True),
    SourceSpec("exp2974", CAPSTONE_V279_REL_PATH, required=True),
    SourceSpec("exp2975", EXP2975_REL_PATH),
    SourceSpec("exp2976", EXP2976_REL_PATH),
    SourceSpec("exp2977", EXP2977_REL_PATH),
    SourceSpec("exp2978", EXP2978_REL_PATH),
    SourceSpec("exp2979", EXP2979_REL_PATH),
    SourceSpec("exp2980", EXP2980_REL_PATH),
    SourceSpec("exp2981", EXP2981_REL_PATH),
    SourceSpec("exp2982", EXP2982_REL_PATH, required=True),
    SourceSpec("exp2983", EXP2983_REL_PATH),
    SourceSpec("exp2984", EXP2984_REL_PATH),
    SourceSpec("exp2985", EXP2985_REL_PATH),
)


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON artifact, returning `{}` when it is absent or malformed."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA256 digest for an artifact path, or `None` when absent."""

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
    """REQ-REPORT-2986: build matrix v14 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    payloads = _load_sources(root_path)
    source_artifacts = _source_artifacts_read(root_path, payloads)
    source_checksums = _source_checksums(source_artifacts)
    rows = [*_prior_rows(payloads.get("exp2973", {})), *_v14_rows(payloads)]
    counts = _status_counts(rows)
    violations = _claim_boundary_violations(rows)
    end = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, end - start), 6)

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": "blocked_required_upstream_missing",
        "matrix_v14_ready": False,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_artifacts_read": source_artifacts,
        "source_checksums": source_checksums,
        "rows": rows,
        "row_count": len(rows),
        "clean_count": counts["clean"],
        "flagged_count": counts["flagged"],
        "blocked_count": counts["blocked"],
        "gated_skipped_count": counts["gated-skipped"],
        "pilot_only_count": counts["pilot-only"],
        "projection_only_count": counts["projection-only"],
        "repair_claim_status": _repair_claim_status(rows),
        "solver_claim_status": _solver_claim_status(rows),
        "fr11_claim_status": _fr11_claim_status(rows),
        "hardware_claim_status": _hardware_claim_status(rows),
        "model_compliance_summary": _model_compliance_summary(rows),
        "claim_boundary_violations": violations,
        "next_milestone_recommendations": _next_milestone_recommendations(rows),
        "no_new_llm_call": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "duration_s": duration_s,
    }

    required_errors = _required_source_errors(payloads)
    if required_errors:
        artifact["required_upstream_errors"] = required_errors
        return artifact

    if payloads["exp2982"].get("fr11_independent_metrics_evaluated") is not True:
        artifact["honest_verdict"] = "blocked_fr11_independent_metrics_not_evaluated"
        return artifact

    artifact["matrix_v14_ready"] = True
    artifact["honest_verdict"] = _complete_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2986 deliverable JSON."""

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


def _source_artifacts_read(
    root: Path,
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "present": path.is_file(),
                "required": spec.required,
                "readable_json_object": bool(payloads.get(spec.experiment_id)),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _source_checksums(source_artifacts: list[dict[str, Any]]) -> dict[str, str | None]:
    return {str(row["path"]): row["sha256"] for row in source_artifacts}


def _required_source_errors(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if spec.required and not payloads.get(spec.experiment_id):
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _prior_rows(matrix_v13: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    verdict = str(matrix_v13.get("honest_verdict", ""))
    for bucket_key, status in PRIOR_BUCKETS:
        for row_id in _string_list(matrix_v13.get(bucket_key)):
            rows.append(
                _row(
                    row_id=f"carry_forward:{row_id}",
                    source_experiment_id="exp2973",
                    milestone="2026.05.279",
                    status=status,
                    claim_class="prior_279_carry_forward",
                    evidence_type="matrix_v13_bucket",
                    model_compliance={"status": "not_applicable"},
                    hardware_compliance={"status": "not_applicable"},
                    prior_failure_outcome="carried_forward_from_matrix_v13",
                    claim_boundary=(
                        "Prior .279 matrix bucket copied without metric recomputation "
                        "or claim promotion."
                    ),
                    claim_boundary_violations=[],
                    source_honest_verdict=verdict,
                    upstream_flags=[],
                    summary={"source_bucket": bucket_key, "source_row_id": row_id},
                )
            )
    return rows


def _v14_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        _exp2975_row(payloads.get("exp2975", {})),
        _exp2976_row(payloads.get("exp2976", {})),
        _exp2977_row(payloads.get("exp2977", {})),
        _exp2978_row(payloads.get("exp2978", {})),
        _exp2979_row(payloads.get("exp2979", {})),
        _exp2980_row(payloads.get("exp2980", {})),
        _exp2981_row(payloads.get("exp2981", {})),
        _exp2982_row(payloads.get("exp2982", {})),
        _exp2983_row(payloads.get("exp2983", {})),
        _exp2984_row(payloads.get("exp2984", {})),
        _exp2985_row(payloads.get("exp2985", {})),
    ]


def _exp2975_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = _status_with_flags(
        "projection-only" if payload.get("archive_ready") is True else "blocked",
        payload,
        payload.get("scripts_research_conductor_modified") is not True,
    )
    return _artifact_row(
        "exp2975_archive_activation",
        "exp2975",
        status,
        "archive_activation",
        "aggregation_only_archive_state",
        "Archive/activation bookkeeping is not new research evidence.",
        payload,
        {"archive_ready": bool(payload.get("archive_ready"))},
        "archive_completed_without_reclassifying_v279",
    )


def _exp2976_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        payload.get("intent_preserving_repair_protocol_ready") is True
        and payload.get("trace_execution_plan_ready") is True
    )
    status = _status_with_flags("projection-only" if ready else "blocked", payload, True)
    return _artifact_row(
        "exp2976_repair_protocol",
        "exp2976",
        status,
        "repair_protocol",
        "artifact_protocol",
        "Protocol evidence cannot claim a repair-rate improvement.",
        payload,
        {
            "intent_preserving_repair_protocol_ready": bool(
                payload.get("intent_preserving_repair_protocol_ready")
            ),
            "trace_execution_plan_ready": bool(payload.get("trace_execution_plan_ready")),
            "prior_failure_addressed": bool(payload.get("prior_failure_addressed")),
        },
        "protocol_addresses_prior_dccd_failures",
    )


def _exp2977_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("repair_rerun_clean") is True
        and _coerce_int(payload.get("n_tasks")) is not None
        and _coerce_int(payload.get("n_tasks")) >= 20
        and _coerce_float(payload.get("pass_at_1_delta")) is not None
        and _coerce_float(payload.get("pass_at_1_delta")) > 0.0
    )
    status = _status_with_flags("clean" if clean else "blocked", payload, True)
    return _artifact_row(
        "exp2977_intent_preserving_repair",
        "exp2977",
        status,
        "repair_eval",
        "live_or_smoke_code_repair_eval",
        "Repair row is clean only with cached SOTA evidence and positive repair deltas.",
        payload,
        _repair_summary(payload),
        "blocked_cached_sota_pair_unavailable"
        if _blocked_verdict(payload.get("honest_verdict"))
        else "repair_rerun_evaluated",
    )


def _exp2978_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        payload.get("telemetry_panel_ready") is True
        and payload.get("no_headline_verifier_claim") is True
    )
    status = _status_with_flags("pilot-only" if ready else "blocked", payload, True)
    return _artifact_row(
        "exp2978_semantic_energy_telemetry",
        "exp2978",
        status,
        "repair_telemetry",
        "diagnostic_proxy_panel",
        "Semantic-energy telemetry is diagnostic only and not a verifier claim.",
        payload,
        {
            "telemetry_panel_ready": bool(payload.get("telemetry_panel_ready")),
            "semantic_energy_signal_usable": bool(payload.get("semantic_energy_signal_usable")),
            "first_step_signal_usable": bool(payload.get("first_step_signal_usable")),
            "no_headline_verifier_claim": bool(payload.get("no_headline_verifier_claim")),
        },
        "diagnostic_signal_only",
    )


def _exp2979_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        payload.get("mcs_feedback_schema_ready") is True
        and payload.get("frontier_upgrade_ready") is True
    )
    status = _status_with_flags("clean" if ready else "blocked", payload, True)
    return _artifact_row(
        "exp2979_solver_feedback_frontier",
        "exp2979",
        status,
        "solver_feedback_frontier",
        "deterministic_z3_frontier",
        "MCS/MUS feedback frontier is deterministic source evidence for solver reruns.",
        payload,
        {
            "mcs_feedback_schema_ready": bool(payload.get("mcs_feedback_schema_ready")),
            "frontier_upgrade_ready": bool(payload.get("frontier_upgrade_ready")),
            "reference_solver_verified_accuracy": _coerce_float(
                payload.get("reference_solver_verified_accuracy")
            ),
            "reference_z3_execution_rate": _coerce_float(payload.get("reference_z3_execution_rate")),
        },
        "solver_feedback_frontier_prepared",
    )


def _exp2980_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("formalization_feedback_clean") is True
        and _coerce_float(payload.get("parseability_rate")) == 1.0
        and _coerce_float(payload.get("solver_verified_accuracy")) == 1.0
        and _coerce_float(payload.get("tautology_flag_rate")) == 0.0
    )
    status = _status_with_flags("clean" if clean else "flagged", payload, True)
    prior_outcome = (
        "solver_delta_clean_but_adversarially_flagged" if _has_flags(payload) and clean else "solver_feedback_rerun_evaluated"
    )
    return _artifact_row(
        "exp2980_solver_formalization_feedback",
        "exp2980",
        status,
        "solver_eval",
        "live_llm_plus_z3",
        "Solver row is clean only when Z3 gates pass and upstream artifact flags are absent.",
        payload,
        {
            "formalization_feedback_clean": bool(payload.get("formalization_feedback_clean")),
            "n_items": _coerce_int(payload.get("n_items")),
            "parseability_rate": _coerce_float(payload.get("parseability_rate")),
            "solver_verified_accuracy": _coerce_float(payload.get("solver_verified_accuracy")),
            "answer_accuracy": _coerce_float(payload.get("answer_accuracy")),
            "z3_execution_rate": _coerce_float(payload.get("z3_execution_rate")),
            "tautology_flag_rate": _coerce_float(payload.get("tautology_flag_rate")),
            "feedback_repair_delta": _coerce_float(payload.get("feedback_repair_delta")),
        },
        prior_outcome,
    )


def _exp2981_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _monitor_claim_violations("exp2981_partial_monitor_promotion", payload)
    clean = (
        payload.get("partial_monitor_promoted") is True
        and _coerce_float(payload.get("prefix_failure_localization_rate")) is not None
        and _coerce_float(payload.get("prefix_failure_localization_rate")) >= 0.8
        and _coerce_float(payload.get("false_alarm_rate")) is not None
        and _coerce_float(payload.get("false_alarm_rate")) <= 0.2
    )
    status = _status_with_flags("clean" if clean else "blocked", payload, not violations)
    return _artifact_row(
        "exp2981_partial_monitor_promotion",
        "exp2981",
        status,
        "partial_monitor",
        "deterministic_monitor_harness",
        "Partial monitor promotion does not imply full streaming verification.",
        payload,
        {
            "partial_monitor_promoted": bool(payload.get("partial_monitor_promoted")),
            "fixture_count": _coerce_int(payload.get("fixture_count")),
            "live_trace_count": _coerce_int(payload.get("live_trace_count")),
            "prefix_failure_localization_rate": _coerce_float(
                payload.get("prefix_failure_localization_rate")
            ),
            "false_alarm_rate": _coerce_float(payload.get("false_alarm_rate")),
            "full_streaming_verification_claim": bool(
                payload.get("full_streaming_verification_claim")
            ),
        },
        "partial_monitor_promoted_without_full_streaming_claim",
        claim_boundary_violations=violations,
    )


def _exp2982_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    clean = (
        payload.get("fr11_independent_metrics_evaluated") is True
        and payload.get("fr11_independent_self_learning_ready") is True
        and payload.get("no_identical_metric_flag") is True
        and payload.get("forgetting_guard_passed") is True
    )
    status = _status_with_flags("clean" if clean else "blocked", payload, True)
    return _artifact_row(
        "exp2982_fr11_independent_metric_gate",
        "exp2982",
        status,
        "fr11_self_learning",
        "aggregation_and_deterministic_replay",
        "FR-11 row uses independent heldout metrics, not the update-selection metric.",
        payload,
        {
            "continuous_self_learning_task": bool(payload.get("continuous_self_learning_task")),
            "fr11_independent_metrics_evaluated": bool(
                payload.get("fr11_independent_metrics_evaluated")
            ),
            "fr11_independent_self_learning_ready": bool(
                payload.get("fr11_independent_self_learning_ready")
            ),
            "no_identical_metric_flag": bool(payload.get("no_identical_metric_flag")),
            "forgetting_guard_passed": bool(payload.get("forgetting_guard_passed")),
            "heldout_independent_delta_vs_random": payload.get(
                "heldout_independent_delta_vs_random"
            )
            if isinstance(payload.get("heldout_independent_delta_vs_random"), Mapping)
            else {},
            "negative_control_delta": payload.get("negative_control_delta")
            if isinstance(payload.get("negative_control_delta"), Mapping)
            else payload.get("negative_control_delta"),
        },
        "fr11_prior_tautology_replaced_by_independent_metrics",
    )


def _exp2983_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    ready = (
        payload.get("trace_to_skill_memory_ready") is True
        and payload.get("headline_result") is not True
        and payload.get("leakage_flag") is not True
    )
    status = _status_with_flags("pilot-only" if ready else "blocked", payload, True)
    return _artifact_row(
        "exp2983_trace_to_skill_memory_pilot",
        "exp2983",
        status,
        "trace_to_skill_memory",
        "artifact_replay_pilot",
        "Trace-to-skill memory is pilot-only artifact replay unless clean live reruns exist.",
        payload,
        {
            "trace_to_skill_memory_ready": bool(payload.get("trace_to_skill_memory_ready")),
            "heldout_skill_reuse_delta": _coerce_float(payload.get("heldout_skill_reuse_delta")),
            "negative_control_delta": _coerce_float(payload.get("negative_control_delta")),
            "leakage_flag": bool(payload.get("leakage_flag")),
            "headline_result": bool(payload.get("headline_result")),
            "fresh_live_llm_inference_used": bool(payload.get("fresh_live_llm_inference_used")),
        },
        "pilot_memory_reuse_not_headline",
    )


def _exp2984_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _hardware_claim_violations("exp2984_gatemate_readback_smoke", payload)
    readback_or_smoke = bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True
    status = _status_with_flags("clean" if readback_or_smoke else "blocked", payload, not violations)
    return _artifact_row(
        "exp2984_gatemate_readback_smoke",
        "exp2984",
        status,
        "hardware_readback_smoke",
        "physical_gatemate_board",
        "GateMate row requires host-visible readback or smoke output for sampler-facing claims.",
        payload,
        {
            "board_detected": bool(payload.get("board_detected")),
            "flash_succeeded": bool(payload.get("flash_succeeded")),
            "readback_attempted": bool(payload.get("readback_attempted")),
            "readback_supported": bool(payload.get("readback_supported")),
            "readback_hash_present": bool(payload.get("readback_hash")),
            "smoke_vector_attempted": bool(payload.get("smoke_vector_attempted")),
            "smoke_vector_passed": bool(payload.get("smoke_vector_passed")),
        },
        "blocked_no_readback_or_host_visible_smoke_io",
        claim_boundary_violations=violations,
    )


def _exp2985_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    violations = _hardware_claim_violations("exp2985_ssqa_register_map_plan", payload)
    ready = payload.get("register_map_plan_ready") is True and payload.get("projection_only") is True
    status = _status_with_flags("projection-only" if ready else "blocked", payload, not violations)
    return _artifact_row(
        "exp2985_ssqa_register_map_plan",
        "exp2985",
        status,
        "hardware_register_map_plan",
        "architecture_projection",
        "SSQA register-map evidence is projection-only until RTL, PnR, and readback exist.",
        payload,
        {
            "register_map_plan_ready": bool(payload.get("register_map_plan_ready")),
            "projection_only": bool(payload.get("projection_only")),
        },
        "projection_register_map_plan_only",
        claim_boundary_violations=violations,
    )


def _artifact_row(
    row_id: str,
    source_experiment_id: str,
    status: str,
    claim_class: str,
    evidence_type: str,
    claim_boundary: str,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    prior_failure_outcome: str,
    *,
    claim_boundary_violations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not payload:
        status = "gated-skipped"
    return _row(
        row_id=row_id,
        source_experiment_id=source_experiment_id,
        milestone=MILESTONE,
        status=status,
        claim_class=claim_class,
        evidence_type=evidence_type,
        model_compliance=_model_compliance(payload),
        hardware_compliance=_hardware_compliance(payload),
        prior_failure_outcome=prior_failure_outcome if payload else "missing_or_malformed_artifact",
        claim_boundary=claim_boundary,
        claim_boundary_violations=claim_boundary_violations or [],
        source_honest_verdict=str(payload.get("honest_verdict", "")),
        upstream_flags=_flag_kinds(payload),
        summary=summary,
    )


def _row(
    *,
    row_id: str,
    source_experiment_id: str,
    milestone: str,
    status: str,
    claim_class: str,
    evidence_type: str,
    model_compliance: Mapping[str, Any],
    hardware_compliance: Mapping[str, Any],
    prior_failure_outcome: str,
    claim_boundary: str,
    claim_boundary_violations: list[dict[str, Any]],
    source_honest_verdict: str,
    upstream_flags: list[str],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "source_experiment_id": source_experiment_id,
        "milestone": milestone,
        "status": status,
        "claim_class": claim_class,
        "evidence_type": evidence_type,
        "model_compliance": dict(model_compliance),
        "hardware_compliance": dict(hardware_compliance),
        "prior_failure_outcome": prior_failure_outcome,
        "headline_eligible": status == "clean",
        "paper_claim_eligible": status == "clean",
        "claim_boundary": claim_boundary,
        "claim_boundary_guard_passed": not claim_boundary_violations,
        "claim_boundary_violations": claim_boundary_violations,
        "source_honest_verdict": source_honest_verdict,
        "upstream_flags": upstream_flags,
        "summary": dict(summary),
    }


def _repair_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "repair_rerun_clean": bool(payload.get("repair_rerun_clean")),
        "n_tasks": _coerce_int(payload.get("n_tasks")),
        "pass_at_1_delta": _coerce_float(payload.get("pass_at_1_delta")),
        "pass_at_k_delta": _coerce_float(payload.get("pass_at_k_delta")),
        "syntax_failure_rate_delta": _coerce_float(payload.get("syntax_failure_rate_delta")),
        "schema_failure_rate_delta": _coerce_float(payload.get("schema_failure_rate_delta")),
        "false_accept_delta": _coerce_float(payload.get("false_accept_delta")),
        "headline_result": bool(payload.get("headline_result")),
        "runtime_trace_coverage": _coerce_float(payload.get("runtime_trace_coverage")),
    }


def _model_compliance(payload: Mapping[str, Any]) -> dict[str, Any]:
    mandatory = _string_list(payload.get("mandatory_headline_model_ids"))
    models = _string_list(payload.get("models_used"))
    if not payload or not mandatory:
        return {"status": "not_applicable"}
    if payload.get("legacy_model_used_only_for_smoke") is True or payload.get(
        "legacy_models_only_for_smoke"
    ) is True:
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


def _hardware_compliance(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload:
        return {"status": "gated_skipped"}
    disallowed = [field for field in HARDWARE_CLAIM_FIELDS if payload.get(field) is True]
    if disallowed:
        return {"status": "claim_boundary_violation", "fields": disallowed}
    if payload.get("projection_only") is True:
        return {"status": "projection_only"}
    if payload.get("inference_substrate") == "physical_gatemate_board":
        if bool(payload.get("readback_hash")) or payload.get("smoke_vector_passed") is True:
            return {"status": "compliant_hardware_readback_or_smoke"}
        return {"status": "blocked_no_readback_or_smoke_output"}
    return {"status": "not_applicable"}


def _hardware_claim_violations(row_id: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = [field for field in HARDWARE_CLAIM_FIELDS if payload.get(field) is True]
    if not fields:
        return []
    return [
        {
            "row_id": row_id,
            "violation": "unsupported_hardware_claim_allowed",
            "fields": fields,
        }
    ]


def _monitor_claim_violations(row_id: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("full_streaming_verification_claim") is not True:
        return []
    supported = _get_path(payload, "promotion_gates.full_streaming_claim_supported.passed")
    if supported is True:
        return []
    return [
        {
            "row_id": row_id,
            "violation": "unsupported_full_streaming_verification_claim",
            "fields": ["full_streaming_verification_claim"],
        }
    ]


def _status_with_flags(base_status: str, payload: Mapping[str, Any], guard_passed: bool) -> str:
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if not guard_passed or _has_flags(payload):
        return "flagged"
    return base_status


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
                kind = str(item.get("kind") or "unknown")
                severity = str(item.get("severity") or "unknown")
                kinds.append(f"{kind}:{severity}")
    return _unique_strings(kinds)


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {status: sum(1 for row in rows if row.get("status") == status) for status in STATUSES}


def _claim_boundary_violations(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        row_violations = row.get("claim_boundary_violations")
        if isinstance(row_violations, list):
            violations.extend(item for item in row_violations if isinstance(item, dict))
    return violations


def _model_compliance_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    statuses = [
        str(row.get("model_compliance", {}).get("status"))
        for row in rows
        if isinstance(row.get("model_compliance"), Mapping)
    ]
    return {status: statuses.count(status) for status in sorted(set(statuses))}


def _repair_claim_status(rows: list[dict[str, Any]]) -> str:
    repair_rows = [row for row in rows if str(row.get("claim_class", "")).startswith("repair")]
    if any(row.get("status") == "blocked" for row in repair_rows):
        return "blocked: intent-preserving repair rerun did not clear cached-SOTA gates"
    if any(row.get("status") == "flagged" for row in repair_rows):
        return "flagged: repair evidence remains adversarially flagged"
    if any(row.get("status") == "clean" for row in repair_rows):
        return "clean: repair evidence cleared v14 gates"
    return "projection-only: repair protocol or telemetry only"


def _solver_claim_status(rows: list[dict[str, Any]]) -> str:
    solver_rows = [row for row in rows if str(row.get("claim_class", "")).startswith("solver")]
    if any(row.get("status") == "flagged" for row in solver_rows):
        return "flagged: solver feedback row has clean Z3 metrics but unresolved artifact flags"
    if any(row.get("status") == "blocked" for row in solver_rows):
        return "blocked: solver frontier evidence missing or gated"
    if solver_rows and all(row.get("status") == "clean" for row in solver_rows):
        return "clean: deterministic frontier and feedback formalization both clear"
    return "gated-skipped: no solver rows available"


def _fr11_claim_status(rows: list[dict[str, Any]]) -> str:
    fr11 = next(
        (row for row in rows if row.get("row_id") == "exp2982_fr11_independent_metric_gate"),
        {},
    )
    if fr11.get("status") == "clean":
        return "clean: independent FR-11 metrics evaluated with no identical-metric flag"
    if fr11.get("status") == "blocked":
        return "blocked: independent FR-11 metric precondition not satisfied"
    if fr11.get("status") == "flagged":
        return "flagged: independent FR-11 row has unresolved flags"
    return "gated-skipped: independent FR-11 row missing"


def _hardware_claim_status(rows: list[dict[str, Any]]) -> str:
    hardware_rows = [
        row
        for row in rows
        if str(row.get("claim_class", "")).startswith("hardware")
        or row.get("claim_class") == "hardware_readback_smoke"
    ]
    if any(row.get("status") == "flagged" for row in hardware_rows):
        return "flagged: unsupported hardware claim boundary violation present"
    if any(row.get("status") == "blocked" for row in hardware_rows):
        return "blocked: board contact exists but readback or smoke output is absent"
    if hardware_rows and all(row.get("status") in {"clean", "projection-only"} for row in hardware_rows):
        return "projection-only: hardware plan remains non-sampler evidence"
    return "gated-skipped: no hardware rows available"


def _next_milestone_recommendations(rows: list[dict[str, Any]]) -> list[str]:
    recommendations = [
        "Repair: rerun intent-preserving code repair with cached mandated SOTA models and no artifact flags.",
        "Solver: preserve Exp 2980 Z3 gains while adding reproducibility checksum and duration provenance.",
        "Hardware: add host-visible GateMate readback or smoke-output IO before sampler-facing claims.",
        "SSQA: convert the dual-BRAM register map from projection to RTL plus PnR/resource reports.",
    ]
    fr11 = _fr11_claim_status(rows)
    if fr11.startswith("clean:"):
        recommendations.append("FR-11: carry independent heldout metrics forward as the self-learning boundary.")
    return recommendations


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v14_ready=true; "
        f"clean={artifact['clean_count']}; "
        f"flagged={artifact['flagged_count']}; "
        f"blocked={artifact['blocked_count']}; "
        f"gated_skipped={artifact['gated_skipped_count']}; "
        f"pilot_only={artifact['pilot_only_count']}; "
        f"projection_only={artifact['projection_only_count']}"
    )


def _get_path(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


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


def _unique_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
