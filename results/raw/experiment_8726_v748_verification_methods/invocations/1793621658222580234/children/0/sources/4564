"""Build the Exp 3038 cross-corpus matrix v18 artifact.

Spec refs: REQ-REPORT-3038, SCENARIO-REPORT-3038.

This module is a milestone accounting layer. It reads checked-in JSON
artifacts, records what each .284 task can honestly support, and deliberately
does not rerun model inference, validators, synthesis, board flashing, or
hardware smoke checks. That distinction is important because this matrix is
allowed to cite source-model details only as provenance, not as a new live
claim made by the matrix itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
MILESTONE = "2026.05.284"
SCHEMA = "carnot.cross_corpus_matrix.v18_284_task_coverage.v1"
ARTIFACT = "experiment_3038_cross_corpus_matrix_v18"
OUTPUT_REL_PATH = Path("results/experiment_3038_cross_corpus_matrix_v18.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3038_cross_corpus_matrix_v18.py"

MATRIX_V17_REL_PATH = Path("results/experiment_3024_cross_corpus_matrix_v17.json")
EXP3026_REL_PATH = Path("results/experiment_3026_archive_v283_activate_v284.json")
EXP3027_REL_PATH = Path("results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json")
EXP3028_REL_PATH = Path("results/experiment_3028_sota_repair_clean_methodology_rerun_v2.json")
EXP3029_REL_PATH = Path("results/experiment_3029_repair_promotion_boundary_audit_v2.json")
EXP3030_REL_PATH = Path("results/experiment_3030_validator_frontier_corrigendum_v2.json")
EXP3031_REL_PATH = Path("results/experiment_3031_dccd_structured_repair_panel_v1.json")
EXP3032_REL_PATH = Path("results/experiment_3032_fr11_heldout_dvi_replay_v2.json")
EXP3033_REL_PATH = Path("results/experiment_3033_fr11_nonforgetting_negative_control_stress_v1.json")
EXP3034_REL_PATH = Path("results/experiment_3034_gatemate_output_contract_pinout_decision_v1.json")
EXP3035_REL_PATH = Path("results/experiment_3035_gatemate_output_shim_rtl_ccf_sim_v1.json")
EXP3035_GATE_CHECK_REL_PATH = Path("results/experiment_3035_gatemate_output_shim_rtl_ccf_sim.json")
EXP3036_REL_PATH = Path("results/experiment_3036_gatemate_host_visible_flash_smoke_v4.json")
EXP3037_REL_PATH = Path("results/experiment_3037_ssqa_bounded_rtl_pnr_gate_artifact_v2.json")
EXP3039_REL_PATH = Path("results/experiment_3039_capstone_v284.json")

STATUSES = (
    "clean",
    "flagged",
    "blocked",
    "gated_skipped",
    "projection_only",
    "pilot_only",
    "missing",
    "retired",
)

HARDWARE_FORBIDDEN_FIELDS = (
    "speedup_claim_made",
    "speedup_claimed",
    "sampler_claim_made",
    "thermodynamic_claim_made",
    "boltzmann_claim_made",
    "annealing_claim_made",
    "energy_claim_made",
    "hardware_performance_claim_made",
    "hardware_execution_claim_made",
)


@dataclass(frozen=True)
class SourceSpec:
    experiment_id: str
    planned_path: Path
    alternate_path: Path | None = None
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3024", MATRIX_V17_REL_PATH, required=True),
    SourceSpec("exp3026", EXP3026_REL_PATH),
    SourceSpec("exp3027", EXP3027_REL_PATH),
    SourceSpec("exp3028", EXP3028_REL_PATH),
    SourceSpec("exp3029", EXP3029_REL_PATH),
    SourceSpec("exp3030", EXP3030_REL_PATH),
    SourceSpec("exp3031", EXP3031_REL_PATH),
    SourceSpec("exp3032", EXP3032_REL_PATH),
    SourceSpec("exp3033", EXP3033_REL_PATH),
    SourceSpec("exp3034", EXP3034_REL_PATH),
    SourceSpec("exp3035", EXP3035_REL_PATH, alternate_path=EXP3035_GATE_CHECK_REL_PATH),
    SourceSpec("exp3036", EXP3036_REL_PATH),
    SourceSpec("exp3037", EXP3037_REL_PATH),
    SourceSpec("exp3039", EXP3039_REL_PATH),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while turning absence or malformed JSON into no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for an existing file."""

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
) -> JsonDict:
    """REQ-REPORT-3038: build matrix v18 from upstream artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = _load_sources(root_path)
    source_artifacts = _source_artifacts(root_path, loaded)
    payloads = {exp_id: row["payload"] for exp_id, row in loaded.items()}
    rows = _matrix_rows(payloads, loaded)
    counts = _status_counts(rows)
    duration_s = _duration(start, now_s)
    represented = {str(row["experiment_id"]) for row in rows}
    required_errors = _required_source_errors(loaded)
    coverage_errors = _coverage_errors(represented)
    ready = not required_errors and not coverage_errors and len(rows) == 14

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v18_ready": ready,
        "rows_total": len(rows),
        "clean": counts["clean"],
        "flagged": counts["flagged"],
        "blocked": counts["blocked"],
        "gated_skipped": counts["gated_skipped"],
        "projection_only": counts["projection_only"],
        "pilot_only": counts["pilot_only"],
        "missing": counts["missing"],
        "retired": counts["retired"],
        "matrix_rows": rows,
        "baseline_v17_summary": _baseline_v17_summary(payloads.get("exp3024", {})),
        "source_artifacts_read": _source_artifacts_public(source_artifacts),
        "source_checksums": _source_checksums(source_artifacts),
        "missing_artifacts": _missing_artifacts(source_artifacts),
        "cited_upstream_artifacts": _cited_upstream_artifacts(source_artifacts, payloads),
        "inference_substrate": _inference_substrate(),
        "recommended_next_actions": _recommended_next_actions(rows),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
        "duration_s": duration_s,
        "honest_verdict": "blocked_required_v17_baseline_missing",
    }
    if required_errors:
        artifact["required_upstream_errors"] = required_errors
        return artifact
    if coverage_errors:
        artifact["coverage_errors"] = coverage_errors
        artifact["honest_verdict"] = "blocked_matrix_v18_task_coverage_incomplete"
        return artifact
    artifact["honest_verdict"] = _complete_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3038 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_sources(root: Path) -> dict[str, JsonDict]:
    loaded: dict[str, JsonDict] = {}
    for spec in SOURCE_SPECS:
        planned = root / spec.planned_path
        alternate = root / spec.alternate_path if spec.alternate_path is not None else None
        actual_path = spec.planned_path
        payload = read_json_object(planned)
        if not payload and alternate is not None and alternate.is_file():
            actual_path = spec.alternate_path or spec.planned_path
            payload = read_json_object(root / actual_path)
        loaded[spec.experiment_id] = {
            "spec": spec,
            "payload": payload,
            "actual_path": actual_path,
            "planned_path_present": planned.is_file(),
            "actual_path_present": (root / actual_path).is_file(),
        }
    return loaded


def _source_artifacts(root: Path, loaded: Mapping[str, JsonDict]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        row = loaded[spec.experiment_id]
        actual_path = Path(row["actual_path"])
        alternate_path = spec.alternate_path
        rows.append(
            {
                "experiment_id": spec.experiment_id,
                "planned_path": spec.planned_path.as_posix(),
                "alternate_path": alternate_path.as_posix() if alternate_path else None,
                "actual_path": actual_path.as_posix(),
                "planned_path_present": bool(row["planned_path_present"]),
                "present": bool(row["actual_path_present"]),
                "required": spec.required,
                "readable_json_object": bool(row["payload"]),
                "sha256": sha256_file(root / actual_path),
            }
        )
    return rows


def _source_artifacts_public(source_artifacts: list[JsonDict]) -> list[JsonDict]:
    return [dict(row) for row in source_artifacts]


def _source_checksums(source_artifacts: list[JsonDict]) -> dict[str, str | None]:
    return {str(row["actual_path"]): row["sha256"] for row in source_artifacts}


def _missing_artifacts(source_artifacts: list[JsonDict]) -> list[str]:
    missing: list[str] = []
    for row in source_artifacts:
        if row.get("planned_path_present") is not True:
            missing.append(str(row["planned_path"]))
        elif row.get("present") is not True:
            missing.append(str(row["actual_path"]))
    return missing


def _required_source_errors(loaded: Mapping[str, JsonDict]) -> list[JsonDict]:
    errors: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        if spec.required and not loaded[spec.experiment_id]["payload"]:
            errors.append(
                {
                    "experiment_id": spec.experiment_id,
                    "path": spec.planned_path.as_posix(),
                    "reason": "missing_or_malformed_artifact",
                }
            )
    return errors


def _coverage_errors(represented: set[str]) -> list[JsonDict]:
    expected = {f"exp{number}" for number in range(3026, 3040)}
    missing = sorted(expected - represented)
    if not missing:
        return []
    return [{"reason": "missing_task_rows", "experiment_ids": missing}]


def _cited_upstream_artifacts(
    source_artifacts: list[JsonDict],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for source in source_artifacts:
        exp_id = str(source["experiment_id"])
        payload = payloads.get(exp_id, {})
        citation: JsonDict = {
            "experiment_id": exp_id,
            "planned_path": source["planned_path"],
            "actual_path": source["actual_path"],
            "present": source["present"],
            "planned_path_present": source["planned_path_present"],
            "readable_json_object": source["readable_json_object"],
            "sha256": source["sha256"],
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        }
        model_details = _source_model_details(payload)
        hardware_details = _source_hardware_details(payload)
        gate_details = _source_gate_details(payload)
        if model_details:
            citation["source_model_details"] = model_details
        if hardware_details:
            citation["source_hardware_details"] = hardware_details
        if gate_details:
            citation["source_gate_details"] = gate_details
        citations.append(citation)
    return citations


def _source_model_details(payload: Mapping[str, Any]) -> JsonDict:
    substrate = _mapping(payload.get("inference_substrate"))
    substrate_details = {
        key: substrate.get(key)
        for key in (
            "kind",
            "mode",
            "gguf_cache_paths",
            "gpu_inventory",
            "selected_headline_model",
            "model_checksum_feasibility",
            "loader",
            "model_load_attempted",
        )
        if substrate.get(key) not in (None, [], {})
    }
    fields = {
        "model_specs": payload.get("model_specs"),
        "target_model": payload.get("target_model"),
        "headline_models_used": payload.get("headline_models_used"),
        "headline_models_available": payload.get("headline_models_available"),
        "model_checksums": payload.get("model_checksums"),
        "inference_substrate": substrate_details,
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_hardware_details(payload: Mapping[str, Any]) -> JsonDict:
    fields = {
        "gatemate_output_contract_ready": payload.get("gatemate_output_contract_ready"),
        "host_visible_io_plan_ready": payload.get("host_visible_io_plan_ready"),
        "selected_output_path": payload.get("selected_output_path"),
        "host_reader_command": payload.get("host_reader_command"),
        "board_detected": payload.get("board_detected"),
        "bitstream_built": payload.get("bitstream_built"),
        "flash_attempted": payload.get("flash_attempted"),
        "host_visible_output_observed": payload.get("host_visible_output_observed"),
        "ssqa_gate_status": payload.get("ssqa_gate_status"),
        "resource_report_paths": payload.get("resource_report_paths"),
        "exact_blocker_or_next_action": payload.get("exact_blocker_or_next_action"),
        "exact_operator_action_required": payload.get("exact_operator_action_required"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _source_gate_details(payload: Mapping[str, Any]) -> JsonDict:
    fields = {
        "status": payload.get("status"),
        "blocked_at_layer": payload.get("blocked_at_layer"),
        "gate_check_summary": payload.get("gate_check_summary"),
        "gates_evaluated": payload.get("gates_evaluated"),
    }
    return {key: value for key, value in fields.items() if value not in (None, [], {})}


def _matrix_rows(payloads: Mapping[str, Mapping[str, Any]], loaded: Mapping[str, JsonDict]) -> list[JsonDict]:
    rows = [
        _exp3026_row(payloads.get("exp3026", {}), loaded["exp3026"]),
        _exp3027_row(payloads.get("exp3027", {}), loaded["exp3027"]),
        _exp3028_row(payloads.get("exp3028", {}), loaded["exp3028"]),
        _exp3029_row(payloads.get("exp3029", {}), loaded["exp3029"]),
        _exp3030_row(payloads.get("exp3030", {}), loaded["exp3030"]),
        _exp3031_row(payloads.get("exp3031", {}), loaded["exp3031"]),
        _exp3032_row(payloads.get("exp3032", {}), loaded["exp3032"]),
        _exp3033_row(payloads.get("exp3033", {}), loaded["exp3033"]),
        _exp3034_row(payloads.get("exp3034", {}), loaded["exp3034"]),
    ]
    exp3035 = _exp3035_row(payloads.get("exp3035", {}), loaded["exp3035"], rows[-1])
    rows.append(exp3035)
    rows.append(_exp3036_row(payloads.get("exp3036", {}), loaded["exp3036"], exp3035))
    rows.append(_exp3037_row(payloads.get("exp3037", {}), loaded["exp3037"]))
    rows.append(_exp3038_row())
    rows.append(_exp3039_row(payloads.get("exp3039", {}), loaded["exp3039"]))
    return rows


def _exp3026_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3026", loaded, "missing", "archive_activation", payload)
    ready = payload.get("milestone_archived") is True and payload.get("next_milestone") == MILESTONE
    status = "projection_only" if ready else _guarded_status(payload, "blocked")
    return _task_row(
        "exp3026",
        loaded,
        status,
        "archive_activation",
        payload,
        summary={
            "milestone_archived": bool(payload.get("milestone_archived")),
            "next_milestone": str(payload.get("next_milestone") or ""),
            "capstone_ready": bool(payload.get("capstone_ready")),
            "previous_paper_ready": bool(payload.get("previous_paper_ready")),
            "protected_files_unchanged": bool(payload.get("protected_files_unchanged")),
        },
    )


def _exp3027_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3027", loaded, "missing", "methodology_corrigendum", payload)
    clean = payload.get("methodology_corrigendum_ready") is True
    status = _guarded_status(payload, "clean" if clean else "blocked")
    return _task_row(
        "exp3027",
        loaded,
        status,
        "methodology_corrigendum",
        payload,
        repair_claim_status="rerun_required" if payload.get("repair_rerun_required") is True else None,
        summary={
            "methodology_corrigendum_ready": bool(payload.get("methodology_corrigendum_ready")),
            "repair_rerun_required": bool(payload.get("repair_rerun_required")),
            "flagged_rows_reviewed": _int_or_none(payload.get("flagged_rows_reviewed")),
        },
    )


def _exp3028_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3028", loaded, "missing", "repair_rerun", payload)
    clean = (
        payload.get("clean_repair_rerun_ready") is True
        and payload.get("repair_controller_clean") is True
        and payload.get("clean_repair_claim_promotable_candidate") is True
        and _int_or(payload.get("n_tasks"), 0) >= 20
        and _float_or(payload.get("pass_at_1_delta"), 0.0) > 0.0
        and payload.get("tautology_gate_clean") is True
    )
    status = _guarded_status(payload, "clean" if clean else "blocked")
    repair_status = "clean_candidate_flagged" if status == "flagged" and clean else "clean_candidate" if clean else "blocked"
    return _task_row(
        "exp3028",
        loaded,
        status,
        "repair_rerun",
        payload,
        repair_claim_status=repair_status,
        summary={
            "clean_repair_rerun_ready": bool(payload.get("clean_repair_rerun_ready")),
            "clean_repair_claim_promotable_candidate": bool(
                payload.get("clean_repair_claim_promotable_candidate")
            ),
            "n_tasks": _int_or_none(payload.get("n_tasks")),
            "n_live_transcripts": _int_or_none(payload.get("n_live_transcripts")),
            "pass_at_1_delta": _float_or_none(payload.get("pass_at_1_delta")),
            "pass_at_k_delta": _float_or_none(payload.get("pass_at_k_delta")),
            "false_accept_delta": _float_or_none(payload.get("false_accept_delta")),
            "tautology_gate_clean": bool(payload.get("tautology_gate_clean")),
        },
    )


def _exp3029_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3029", loaded, "missing", "repair_boundary_audit", payload)
    clean = payload.get("repair_promotion_boundary_ready") is True
    status = _guarded_status(payload, "clean" if clean else "blocked")
    return _task_row(
        "exp3029",
        loaded,
        status,
        "repair_boundary_audit",
        payload,
        repair_claim_status=str(payload.get("repair_claim_status") or "unknown"),
        summary={
            "repair_promotion_boundary_ready": bool(payload.get("repair_promotion_boundary_ready")),
            "repair_claim_status": str(payload.get("repair_claim_status") or "unknown"),
            "promotable_claim_count": len(_list_of_mappings(payload.get("promotable_claims"))),
            "bounded_claim_count": len(_list_of_mappings(payload.get("bounded_claims"))),
            "retired_or_blocked_claim_count": len(
                _list_of_mappings(payload.get("retired_or_blocked_claims"))
            ),
        },
    )


def _exp3030_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3030", loaded, "missing", "validator_frontier_corrigendum", payload)
    clean = payload.get("validator_frontier_corrigendum_ready") is True
    status = _guarded_status(payload, "clean" if clean else "blocked")
    return _task_row(
        "exp3030",
        loaded,
        status,
        "validator_frontier_corrigendum",
        payload,
        summary={
            "validator_frontier_corrigendum_ready": bool(
                payload.get("validator_frontier_corrigendum_ready")
            ),
            "verified_region_count": _int_or_none(payload.get("verified_region_count")),
            "unresolved_region_count": _int_or_none(payload.get("unresolved_region_count")),
            "fallback_only_count": _int_or_none(payload.get("fallback_only_count")),
            "missing_authority_count": _int_or_none(payload.get("missing_authority_count")),
        },
    )


def _exp3031_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3031", loaded, "missing", "dccd_structured_repair_panel", payload)
    ready = payload.get("dccd_panel_ready") is True or payload.get("dccd_structured_repair_panel_ready") is True
    status = _guarded_status(payload, "clean" if ready else "blocked")
    return _task_row(
        "exp3031",
        loaded,
        status,
        "dccd_structured_repair_panel",
        payload,
        repair_claim_status="pilot_panel_flagged" if status == "flagged" and ready else "pilot_panel",
        summary={
            "dccd_panel_ready": bool(payload.get("dccd_panel_ready") or payload.get("dccd_structured_repair_panel_ready")),
            "n_cases": _int_or_none(payload.get("n_cases")),
            "false_accept_delta": _float_or_none(payload.get("false_accept_delta")),
            "intent_drift_delta": _float_or_none(payload.get("intent_drift_delta")),
        },
    )


def _exp3032_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3032", loaded, "missing", "fr11_heldout_replay", payload)
    clean = (
        payload.get("fr11_heldout_replay_ready") is True
        and payload.get("continuous_self_learning_tested") is True
        and payload.get("tautology_risk_cleared") is True
        and payload.get("information_asymmetry_enforced") is True
        and not payload.get("invariant_violations")
    )
    status = _guarded_status(payload, "clean" if clean else "blocked")
    return _task_row(
        "exp3032",
        loaded,
        status,
        "fr11_heldout_replay",
        payload,
        fr11_self_learning_promotable=False,
        summary={
            "fr11_heldout_replay_ready": bool(payload.get("fr11_heldout_replay_ready")),
            "continuous_self_learning_tested": bool(payload.get("continuous_self_learning_tested")),
            "heldout_trace_count": _int_or_none(payload.get("heldout_trace_count")),
            "feasible_infeasible_auc_delta": _float_or_none(
                payload.get("feasible_infeasible_auc_delta")
            ),
            "shuffled_feedback_delta": _float_or_none(payload.get("shuffled_feedback_delta")),
            "tautology_risk_cleared": bool(payload.get("tautology_risk_cleared")),
        },
    )


def _exp3033_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3033", loaded, "missing", "fr11_nonforgetting_stress", payload)
    promotable = payload.get("fr11_self_learning_promotable") is True
    clean = (
        payload.get("fr11_nonforgetting_stress_ready") is True
        and promotable
        and not payload.get("drift_failures")
    )
    status = _guarded_status(payload, "clean" if clean else "flagged")
    return _task_row(
        "exp3033",
        loaded,
        status,
        "fr11_nonforgetting_stress",
        payload,
        fr11_self_learning_promotable=promotable,
        summary={
            "fr11_nonforgetting_stress_ready": bool(payload.get("fr11_nonforgetting_stress_ready")),
            "fr11_self_learning_promotable": promotable,
            "promotion_decision": str(payload.get("promotion_decision") or ""),
            "prior_retention_delta": _float_or_none(payload.get("prior_retention_delta")),
            "heldout_delta_after_update": _float_or_none(payload.get("heldout_delta_after_update")),
            "shuffled_control_delta": _float_or_none(payload.get("shuffled_control_delta")),
            "drift_failure_count": len(_as_list(payload.get("drift_failures"))),
        },
    )


def _exp3034_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3034", loaded, "missing", "gatemate_output_contract", payload)
    violation = _hardware_claim_fields(payload)
    if violation:
        status = "flagged"
    elif payload.get("gatemate_output_contract_ready") is True:
        status = "clean"
    else:
        status = "blocked"
    return _task_row(
        "exp3034",
        loaded,
        status,
        "gatemate_output_contract",
        payload,
        gatemate_output_contract_ready=_safe_bool(payload.get("gatemate_output_contract_ready")),
        host_visible_output_observed=False,
        summary={
            "gatemate_output_contract_ready": bool(payload.get("gatemate_output_contract_ready")),
            "host_visible_io_plan_ready": bool(payload.get("host_visible_io_plan_ready")),
            "selected_output_path": str(payload.get("selected_output_path") or ""),
            "operator_action_count": len(_as_list(payload.get("exact_operator_action_required"))),
            "unsupported_hardware_claim_fields": violation,
        },
    )


def _exp3035_row(payload: Mapping[str, Any], loaded: Mapping[str, Any], exp3034_row: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "gated_skipped" if exp3034_row.get("gatemate_output_contract_ready") is False else "missing"
        return _task_row(
            "exp3035",
            loaded,
            status,
            "gatemate_output_shim",
            payload,
            gatemate_output_contract_ready=_safe_bool(exp3034_row.get("gatemate_output_contract_ready")),
            host_visible_output_observed=False,
            summary={"gate_source": "exp3034", "gate_status": exp3034_row.get("status")},
        )
    verdict_status = _status_from_verdict(str(payload.get("honest_verdict") or payload.get("status") or ""))
    ready = payload.get("gatemate_output_shim_ready") is True or payload.get("gatemate_output_shim_sim_passed") is True
    status = "clean" if ready else verdict_status
    return _task_row(
        "exp3035",
        loaded,
        status,
        "gatemate_output_shim",
        payload,
        gatemate_output_contract_ready=_safe_bool(exp3034_row.get("gatemate_output_contract_ready")),
        host_visible_output_observed=False,
        summary={
            "gatemate_output_shim_ready": bool(payload.get("gatemate_output_shim_ready")),
            "gatemate_output_shim_sim_passed": bool(payload.get("gatemate_output_shim_sim_passed")),
            "blocked_at_layer": str(payload.get("blocked_at_layer") or ""),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
        },
    )


def _exp3036_row(payload: Mapping[str, Any], loaded: Mapping[str, Any], exp3035_row: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "gated_skipped" if exp3035_row.get("status") in {"gated_skipped", "blocked"} else "missing"
        return _task_row(
            "exp3036",
            loaded,
            status,
            "gatemate_host_visible_flash_smoke",
            payload,
            host_visible_output_observed=False,
            gatemate_output_contract_ready=_safe_bool(exp3035_row.get("gatemate_output_contract_ready")),
            summary={"gate_source": "exp3035", "gate_status": exp3035_row.get("status")},
        )
    observed = payload.get("host_visible_output_observed") is True
    ready = payload.get("gatemate_flash_smoke_ready") is True and observed
    status = _status_from_verdict(str(payload.get("honest_verdict") or "")) if not ready else "clean"
    if not ready and status == "clean":
        status = "blocked"
    return _task_row(
        "exp3036",
        loaded,
        status,
        "gatemate_host_visible_flash_smoke",
        payload,
        host_visible_output_observed=observed,
        gatemate_output_contract_ready=_safe_bool(exp3035_row.get("gatemate_output_contract_ready")),
        summary={
            "gatemate_flash_smoke_ready": bool(payload.get("gatemate_flash_smoke_ready")),
            "board_detected": bool(payload.get("board_detected")),
            "bitstream_built": bool(payload.get("bitstream_built")),
            "flash_attempted": bool(payload.get("flash_attempted")),
            "host_visible_output_observed": observed,
        },
    )


def _exp3037_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3037", loaded, "missing", "ssqa_boundary", payload)
    violation = _hardware_claim_fields(payload)
    gate_status = str(payload.get("ssqa_gate_status") or "")
    if violation:
        status = "flagged"
    elif gate_status in {"gate_skipped", "gated_skipped"}:
        status = "gated_skipped"
    elif payload.get("ssqa_boundary_ready") is True:
        status = "clean"
    else:
        status = "blocked"
    substrate = _mapping(payload.get("inference_substrate"))
    return _task_row(
        "exp3037",
        loaded,
        status,
        "ssqa_boundary",
        payload,
        ssqa_gate_status=gate_status or None,
        host_visible_output_observed=_safe_bool(substrate.get("host_visible_output_observed")),
        summary={
            "ssqa_boundary_ready": bool(payload.get("ssqa_boundary_ready")),
            "ssqa_gate_status": gate_status,
            "resource_report_count": len(_as_list(payload.get("resource_report_paths"))),
            "unsupported_hardware_claim_fields": violation,
        },
    )


def _exp3038_row() -> JsonDict:
    return _base_row(
        experiment_id="exp3038",
        status="clean",
        task_class="cross_corpus_matrix",
        planned_path=OUTPUT_REL_PATH.as_posix(),
        actual_path=OUTPUT_REL_PATH.as_posix(),
        planned_path_present=False,
        actual_path_present=False,
        source_honest_verdict="current aggregation task",
        summary={"matrix_task": "current_exp3038", "task_coverage_target": 14},
    )


def _exp3039_row(payload: Mapping[str, Any], loaded: Mapping[str, Any]) -> JsonDict:
    if not payload:
        return _task_row("exp3039", loaded, "missing", "capstone", payload)
    ready = payload.get("capstone_ready") is True
    status = _guarded_status(payload, "clean" if ready else "blocked")
    return _task_row(
        "exp3039",
        loaded,
        status,
        "capstone",
        payload,
        repair_claim_status=str(payload.get("repair_claim_status") or "not_applicable"),
        summary={
            "capstone_ready": bool(payload.get("capstone_ready")),
            "paper_ready": bool(payload.get("paper_ready")),
        },
    )


def _task_row(
    experiment_id: str,
    loaded: Mapping[str, Any],
    status: str,
    task_class: str,
    payload: Mapping[str, Any],
    *,
    repair_claim_status: str | None = None,
    fr11_self_learning_promotable: bool | None = None,
    gatemate_output_contract_ready: bool | None = None,
    host_visible_output_observed: bool | None = None,
    ssqa_gate_status: str | None = None,
    summary: Mapping[str, Any] | None = None,
) -> JsonDict:
    return _base_row(
        experiment_id=experiment_id,
        status=status if status in STATUSES else "missing",
        task_class=task_class,
        planned_path=loaded["spec"].planned_path.as_posix(),
        actual_path=Path(loaded["actual_path"]).as_posix(),
        planned_path_present=bool(loaded["planned_path_present"]),
        actual_path_present=bool(loaded["actual_path_present"]),
        source_honest_verdict=str(payload.get("honest_verdict") or ""),
        repair_claim_status=repair_claim_status,
        fr11_self_learning_promotable=fr11_self_learning_promotable,
        gatemate_output_contract_ready=gatemate_output_contract_ready,
        host_visible_output_observed=host_visible_output_observed,
        ssqa_gate_status=ssqa_gate_status,
        upstream_flags=_upstream_flags(payload),
        summary=summary,
    )


def _base_row(
    *,
    experiment_id: str,
    status: str,
    task_class: str,
    planned_path: str,
    actual_path: str,
    planned_path_present: bool,
    actual_path_present: bool,
    source_honest_verdict: str,
    repair_claim_status: str | None = None,
    fr11_self_learning_promotable: bool | None = None,
    gatemate_output_contract_ready: bool | None = None,
    host_visible_output_observed: bool | None = None,
    ssqa_gate_status: str | None = None,
    upstream_flags: list[str] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> JsonDict:
    return {
        "experiment_id": experiment_id,
        "milestone": MILESTONE,
        "status": status,
        "task_class": task_class,
        "planned_path": planned_path,
        "actual_path": actual_path,
        "planned_path_present": planned_path_present,
        "actual_path_present": actual_path_present,
        "source_honest_verdict": source_honest_verdict,
        "repair_claim_status": repair_claim_status or "not_applicable",
        "fr11_self_learning_promotable": fr11_self_learning_promotable,
        "gatemate_output_contract_ready": gatemate_output_contract_ready,
        "host_visible_output_observed": host_visible_output_observed,
        "ssqa_gate_status": ssqa_gate_status,
        "upstream_flags": upstream_flags or [],
        "summary": dict(summary or {}),
    }


def _guarded_status(payload: Mapping[str, Any], nominal: str) -> str:
    if _hardware_claim_fields(payload):
        return "flagged"
    if _has_flags(payload):
        return "flagged"
    verdict_status = _status_from_verdict(str(payload.get("honest_verdict") or ""))
    if verdict_status != "clean":
        return verdict_status
    return nominal


def _status_from_verdict(verdict: str) -> str:
    lowered = verdict.lower()
    if "gate_skipped" in lowered or "gated_skipped" in lowered or "blocked_gate" in lowered:
        return "gated_skipped"
    if "retired" in lowered:
        return "retired"
    if "flagged" in lowered:
        return "flagged"
    if lowered.startswith("blocked") or " blocked_" in lowered or ": blocked_" in lowered:
        return "blocked"
    return "clean"


def _has_flags(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True or bool(_as_list(payload.get("corrigendum_pending")))


def _hardware_claim_fields(payload: Mapping[str, Any]) -> list[str]:
    return [field for field in HARDWARE_FORBIDDEN_FIELDS if payload.get(field) is True]


def _upstream_flags(payload: Mapping[str, Any]) -> list[str]:
    flags: list[str] = []
    if payload.get("flagged_adversarial") is True:
        flags.append("flagged_adversarial=true")
    for row in _as_list(payload.get("corrigendum_pending")):
        if isinstance(row, Mapping):
            kind = str(row.get("kind") or "UNKNOWN")
            severity = str(row.get("severity") or "")
            flags.append(f"{kind}:{severity}" if severity else kind)
    return flags


def _status_counts(rows: list[JsonDict]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[str(row.get("status") or "missing")] += 1
    return counts


def _baseline_v17_summary(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_v17_ready": bool(payload.get("matrix_v17_ready")),
        "clean": _int_or(payload.get("clean_count"), 0),
        "flagged": _int_or(payload.get("flagged_count"), 0),
        "blocked": _int_or(payload.get("blocked_count"), 0),
        "gated_skipped": _int_or(payload.get("gated_skipped_count"), 0),
        "projection_only": _int_or(payload.get("projection_only_count"), 0),
        "pilot_only": _int_or(payload.get("pilot_only_count"), 0),
        "missing": _int_or(payload.get("missing_count"), 0),
    }


def _recommended_next_actions(rows: list[JsonDict]) -> list[str]:
    by_exp = {str(row["experiment_id"]): row for row in rows}
    actions: list[str] = []
    if by_exp.get("exp3029", {}).get("repair_claim_status") == "bounded":
        actions.append("Keep repair wording bounded until matrix and capstone repair blockers clear.")
    if by_exp.get("exp3033", {}).get("fr11_self_learning_promotable") is True:
        actions.append("Carry FR-11 forward as controller-only self-learning; do not claim model-weight learning.")
    if by_exp.get("exp3034", {}).get("gatemate_output_contract_ready") is False:
        actions.append("Resolve the GateMate host-visible output pinout before rerunning shim or flash smoke.")
    if by_exp.get("exp3037", {}).get("ssqa_gate_status") == "gate_skipped":
        actions.append("Keep SSQA as gate-skipped until GateMate host-visible output is observed.")
    if by_exp.get("exp3039", {}).get("status") == "missing":
        actions.append("Run the .284 capstone after matrix v18 is available.")
    return actions


def _complete_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: matrix_v18_ready=true; "
        f"rows_total={artifact['rows_total']}; "
        f"clean={artifact['clean']}; flagged={artifact['flagged']}; "
        f"blocked={artifact['blocked']}; gated_skipped={artifact['gated_skipped']}; "
        f"projection_only={artifact['projection_only']}; pilot_only={artifact['pilot_only']}; "
        f"missing={artifact['missing']}; retired={artifact['retired']}"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "executes_models": False,
        "source": "checked_in_artifacts",
    }


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _list_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in _as_list(value) if isinstance(row, Mapping)]


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
