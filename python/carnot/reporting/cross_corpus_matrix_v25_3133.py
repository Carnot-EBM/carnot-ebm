"""Build the Exp 3133 cross-corpus matrix v25 artifact.

Spec refs: REQ-REPORT-3133, SCENARIO-REPORT-3133.

Matrix v25 is an evidence ledger over checked-in `.291` artifacts. It does not
run models, repairs, solvers, the conductor, or hardware. The point is to keep
the publication blockers visible while making the new evidence chain
machine-readable for the next capstone.
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
RUN_DATE = "20260526"
MILESTONE = "2026.05.291"
SCHEMA = "carnot.cross_corpus_matrix.v25_291_artifact_aggregation.v1"
ARTIFACT = "experiment_3133_cross_corpus_matrix_v25"
OUTPUT_REL_PATH = Path("results/experiment_3133_cross_corpus_matrix_v25.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3133_cross_corpus_matrix_v25.py"

MATRIX_V24_REL_PATH = Path("results/experiment_3120_cross_corpus_matrix_v24.json")
CAPSTONE_V290_REL_PATH = Path("results/experiment_3121_capstone_v290.json")
EXP3122_REL_PATH = Path("results/experiment_3122_archive_v290_activate_v291.json")
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3124_REL_PATH = Path(
    "results/experiment_3124_difficulty_stratified_live_sota_verifier_panel_v6.json"
)
EXP3125_REL_PATH = Path(
    "results/experiment_3125_prefix_closed_deterministic_verifier_bound_pilot_v1.json"
)
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3127_REL_PATH = Path("results/experiment_3127_multi_turn_monitored_repair_ladder_v1.json")
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3129_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
EXP3130_REL_PATH = Path("results/experiment_3130_arm_ebt_energy_budget_sidecar_diagnostic_v2.json")
EXP3131_REL_PATH = Path("results/experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.json")
EXP3132_REL_PATH = Path("results/experiment_3132_hardware_evidence_sampler_boundary_v5.json")

STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "retired",
    "projection_only",
    "diagnostic_only",
    "model_spec_gap",
)
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "projection_only",
    "model_spec_gap",
}


@dataclass(frozen=True)
class SourceSpec:
    """One source artifact the matrix reads and cites."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    ready_field: str = ""


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3120", MATRIX_V24_REL_PATH, "matrix_v24_authority", True, "matrix_v24_ready"),
    SourceSpec(
        "exp3121", CAPSTONE_V290_REL_PATH, "capstone_v290_authority", True, "capstone_ready"
    ),
    SourceSpec(
        "exp3122",
        EXP3122_REL_PATH,
        "archive_v290_activate_v291",
        False,
        "archive_v290_activate_v291_ready",
    ),
    SourceSpec(
        "exp3123", EXP3123_REL_PATH, "sota_cache_coverage", False, "sota_cache_manifest_v2_ready"
    ),
    SourceSpec(
        "exp3124",
        EXP3124_REL_PATH,
        "live_verifier_lift",
        False,
        "difficulty_stratified_live_sota_panel_v6_ready",
    ),
    SourceSpec(
        "exp3125",
        EXP3125_REL_PATH,
        "prefix_closed_bounds",
        False,
        "prefix_closed_bound_pilot_ready",
    ),
    SourceSpec(
        "exp3126",
        EXP3126_REL_PATH,
        "fragment_time_monitors",
        False,
        "fragment_time_monitor_v1_ready",
    ),
    SourceSpec("exp3127", EXP3127_REL_PATH, "repair_ladder", False, "status"),
    SourceSpec("exp3128", EXP3128_REL_PATH, "fr11_evoenv", False, "fr11_evoenv_pilot_v1_ready"),
    SourceSpec(
        "exp3129",
        EXP3129_REL_PATH,
        "fr11_constraint_memory",
        False,
        "fr11_constraint_memory_audit_v1_ready",
    ),
    SourceSpec(
        "exp3130",
        EXP3130_REL_PATH,
        "arm_ebt_energy_budget",
        False,
        "arm_ebt_energy_budget_sidecar_v2_ready",
    ),
    SourceSpec("exp3131", EXP3131_REL_PATH, "kan_pwa_milp", False, "kan_pwa_milp_audit_v1_ready"),
    SourceSpec(
        "exp3132",
        EXP3132_REL_PATH,
        "hardware_sampler_boundary",
        False,
        "hardware_evidence_sampler_boundary_v5_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, returning empty evidence when it is absent or invalid."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source checksum for reproducible matrix provenance."""

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
    """REQ-REPORT-3133: aggregate matrix v25 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3120"]
    capstone = payloads["exp3121"]
    rows = _carry_forward_rows(matrix) + _dot291_rows(payloads) if matrix else []
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    prior_count = _prior_publication_blocker_count(matrix, capstone)
    missing_artifacts = _missing_artifacts(sources)
    required_source_errors = _required_source_errors(sources)
    invariant_violations = _invariant_violations(
        matrix,
        capstone,
        rows,
        status_counts,
        publication_blockers,
        required_source_errors,
    )
    ready = not invariant_violations
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v25_ready": ready,
        "rows_total": len(rows),
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v24": len(publication_blockers) - prior_count,
        "status_counts": status_counts,
        "publication_blockers": publication_blockers,
        "rows": rows,
        "missing_artifacts": missing_artifacts,
        "headline_claim_allowance_summary": _headline_claim_allowance_summary(payloads),
        "verifier_repair_summary": _verifier_repair_summary(payloads, rows),
        "fr11_summary": _fr11_summary(payloads, rows),
        "architecture_boundary_summary": _architecture_boundary_summary(payloads, rows),
        "diagnostic_only_rows": _row_ids_by_status(rows, "diagnostic_only"),
        "gated_skips": _gated_skips(rows),
        "architecture_boundary_rows": _architecture_boundary_rows(rows),
        "source_artifacts": _public_sources(sources),
        "source_checksums": {
            str(row["path"]): row.get("sha256") for row in _public_sources(sources)
        },
        "required_source_errors": required_source_errors,
        "invariant_violations": invariant_violations,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "status_updates_written": False,
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
) -> Path:
    """Build and persist the Exp 3133 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_payload(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "ready_field": spec.ready_field,
        "source_type": "json",
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "payload": payload,
        "sha256": sha256_file(path),
    }


def _carry_forward_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for raw in _as_list(matrix.get("rows")):
        if not isinstance(raw, Mapping):
            continue
        row = _claim_entry(raw)
        summary = _as_mapping(row.get("summary"))
        summary.setdefault("v25_status_rationale", "carried_forward_from_matrix_v24")
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v24")
        rows.append(row)
    return rows


def _dot291_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _archive_handoff_row(payloads["exp3122"]),
        _sota_cache_row(payloads["exp3123"]),
        _live_verifier_row(payloads["exp3124"]),
        _prefix_bounds_row(payloads["exp3125"]),
        _fragment_time_row(payloads["exp3126"]),
        _repair_ladder_row(payloads["exp3127"]),
        _fr11_evoenv_row(payloads["exp3128"]),
        _fr11_memory_row(payloads["exp3129"]),
        _arm_ebt_row(payloads["exp3130"]),
        _kan_row(payloads["exp3131"]),
        _hardware_row(payloads["exp3132"]),
    ]


def _archive_handoff_row(payload: Mapping[str, Any]) -> JsonDict:
    return _row(
        row_id="dot291:exp3122_archive_handoff",
        status=_ready_status(bool(payload), payload, "archive_v290_activate_v291_ready"),
        source_artifact=EXP3122_REL_PATH.as_posix(),
        source_field="archive_v290_activate_v291_ready",
        evidence_class="archive_v290_activate_v291",
        claim_scope="milestone_activation",
        summary={
            "archive_v290_activate_v291_ready": payload.get("archive_v290_activate_v291_ready")
            is True,
            "prior_publication_blocker_count": _int_or_none(
                payload.get("prior_publication_blocker_count")
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _sota_cache_row(payload: Mapping[str, Any]) -> JsonDict:
    present = _text_list(payload.get("present_model_ids"))
    missing = _text_list(payload.get("missing_model_ids"))
    selected = _text_list(payload.get("selected_headline_model_ids"))
    ready = payload.get("sota_cache_manifest_v2_ready") is True
    headline_allowed = payload.get("headline_claim_allowed") is True
    cached_pair = payload.get("cached_sota_pair_available") is True
    if not payload:
        status = "missing"
    elif not ready:
        status = "blocked"
    elif not headline_allowed or not selected:
        status = "model_spec_gap"
    elif missing or not cached_pair:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot291:exp3123_sota_cache_coverage",
        status=status,
        source_artifact=EXP3123_REL_PATH.as_posix(),
        source_field="sota_cache_manifest_v2_ready",
        evidence_class="sota_cache_preconditions_manifest",
        claim_scope="local_sota_model_cache_policy",
        summary={
            "sota_cache_manifest_v2_ready": ready,
            "cached_sota_pair_available": cached_pair,
            "headline_claim_allowed": headline_allowed,
            "present_model_ids": present,
            "missing_model_ids": missing,
            "selected_headline_model_ids": selected,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _live_verifier_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("difficulty_stratified_live_sota_panel_v6_ready") is True
    false_accept = _float_or_none(payload.get("false_accept_rate")) or 0.0
    gain = _float_or_none(payload.get("verifier_gain_delta")) or 0.0
    repair_gate = str(payload.get("repair_gate_state") or "")
    headline_allowed = payload.get("headline_claim_allowed") is True
    if not payload:
        status = "missing"
    elif not ready or not headline_allowed or repair_gate != "unblocked" or false_accept > 0.0:
        status = "blocked"
    elif gain <= 0.0:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot291:exp3124_live_verifier_lift",
        status=status,
        source_artifact=EXP3124_REL_PATH.as_posix(),
        source_field="difficulty_stratified_live_sota_panel_v6_ready",
        evidence_class="difficulty_stratified_live_sota_verifier_panel",
        claim_scope="live_sota_verifier_lift",
        summary={
            "difficulty_stratified_live_sota_panel_v6_ready": ready,
            "headline_claim_allowed": headline_allowed,
            "repair_gate_state": repair_gate,
            "verifier_gain_delta": gain,
            "false_accept_rate": false_accept,
            "false_reject_rate": _float_or_none(payload.get("false_reject_rate")) or 0.0,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "selected_model_ids": _text_list(payload.get("selected_model_ids")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _prefix_bounds_row(payload: Mapping[str, Any]) -> JsonDict:
    status = _ready_status(bool(payload), payload, "prefix_closed_bound_pilot_ready")
    if status == "clean":
        status = "bounded"
    return _row(
        row_id="dot291:exp3125_prefix_bounds",
        status=status,
        source_artifact=EXP3125_REL_PATH.as_posix(),
        source_field="prefix_closed_bound_pilot_ready",
        evidence_class="prefix_closed_deterministic_verifier_bound",
        claim_scope="bounded_prefix_correctness",
        summary={
            "prefix_closed_bound_pilot_ready": payload.get("prefix_closed_bound_pilot_ready")
            is True,
            "lower_bound": _float_or_none(payload.get("lower_bound")),
            "upper_bound": _float_or_none(payload.get("upper_bound")),
            "bound_width": _float_or_none(payload.get("bound_width")),
            "explored_prefix_count": _int_or_none(payload.get("explored_prefix_count")),
            "accepted_prefix_count": _int_or_none(payload.get("accepted_prefix_count")),
            "limitations": _text_list(payload.get("limitations")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fragment_time_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("fragment_time_monitor_v1_ready") is not True:
        status = "blocked"
    elif (
        (_float_or_none(payload.get("ledger_consistency_rate")) or 0.0) < 1.0
        or (_int_or_none(payload.get("monitor_violation_count")) or 0) > 0
        or (_int_or_none(payload.get("contradiction_count")) or 0) > 0
        or (_int_or_none(payload.get("satisfiable_drift_count")) or 0) > 0
    ):
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot291:exp3126_fragment_time_monitors",
        status=status,
        source_artifact=EXP3126_REL_PATH.as_posix(),
        source_field="fragment_time_monitor_v1_ready",
        evidence_class="fragment_time_monitor_satisfiable_drift_audit",
        claim_scope="fragment_time_monitor_boundary",
        summary={
            "fragment_time_monitor_v1_ready": payload.get("fragment_time_monitor_v1_ready") is True,
            "monitor_event_count": _int_or_none(payload.get("monitor_event_count")),
            "monitor_violation_count": _int_or_none(payload.get("monitor_violation_count")),
            "contradiction_count": _int_or_none(payload.get("contradiction_count")),
            "satisfiable_drift_count": _int_or_none(payload.get("satisfiable_drift_count")),
            "ledger_consistency_rate": _float_or_none(payload.get("ledger_consistency_rate")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_ladder_row(payload: Mapping[str, Any]) -> JsonDict:
    source_status = normal_status(str(payload.get("status") or "missing"))
    if not payload:
        status = "missing"
    elif source_status == "clean" or str(payload.get("status") or "").lower() == "success":
        status = "clean"
    elif str(payload.get("blocked_at_layer") or "") or source_status in {"blocked", "missing"}:
        status = "blocked"
    else:
        status = source_status
    return _row(
        row_id="dot291:exp3127_repair_ladder",
        status=status,
        source_artifact=EXP3127_REL_PATH.as_posix(),
        source_field="status",
        evidence_class="multi_turn_monitored_repair_ladder",
        claim_scope="repair_live_rerun",
        summary={
            "source_status": str(payload.get("status") or ""),
            "blocked_at_layer": str(payload.get("blocked_at_layer") or ""),
            "gated_skip": bool(payload.get("blocked_at_layer")),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
            "gates_evaluated_count": len(_as_list(payload.get("gates_evaluated"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_evoenv_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("fr11_evoenv_pilot_v1_ready") is not True:
        status = "blocked"
    elif (
        payload.get("no_weight_update_claim") is True
        or payload.get("live_model_environment_synthesis") is False
        or (_float_or_none(payload.get("retention_delta")) or 0.0) == 0.0
    ):
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot291:exp3128_fr11_evoenv",
        status=status,
        source_artifact=EXP3128_REL_PATH.as_posix(),
        source_field="fr11_evoenv_pilot_v1_ready",
        evidence_class="fr11_evoenv_verifiable_environment_synthesis",
        claim_scope="controller_only_environment_synthesis",
        summary={
            "fr11_evoenv_pilot_v1_ready": payload.get("fr11_evoenv_pilot_v1_ready") is True,
            "continuous_self_learning_targeted": payload.get("continuous_self_learning_targeted")
            is True,
            "live_model_environment_synthesis": payload.get("live_model_environment_synthesis")
            is True,
            "admitted_environment_count": _int_or_none(payload.get("admitted_environment_count")),
            "candidate_environment_count": _int_or_none(payload.get("candidate_environment_count")),
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "soundness_errors": _int_or_none(payload.get("soundness_errors")),
            "completeness_errors": _int_or_none(payload.get("completeness_errors")),
            "retention_delta": _float_or_none(payload.get("retention_delta")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_memory_row(payload: Mapping[str, Any]) -> JsonDict:
    ledger = _float_or_none(payload.get("ledger_consistency_rate"))
    recommendation = str(payload.get("promotion_recommendation") or "")
    if not payload:
        status = "missing"
    elif payload.get("fr11_constraint_memory_audit_v1_ready") is not True:
        status = "blocked"
    elif (
        payload.get("no_weight_update_claim") is True
        or (ledger is not None and ledger < 1.0)
        or "controller" in recommendation
        or "block_model_weight" in recommendation
    ):
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot291:exp3129_fr11_memory",
        status=status,
        source_artifact=EXP3129_REL_PATH.as_posix(),
        source_field="fr11_constraint_memory_audit_v1_ready",
        evidence_class="fr11_constraint_memory_retention_drift_audit",
        claim_scope="controller_only_constraint_memory",
        summary={
            "fr11_constraint_memory_audit_v1_ready": payload.get(
                "fr11_constraint_memory_audit_v1_ready"
            )
            is True,
            "admitted_environment_count": _int_or_none(payload.get("admitted_environment_count")),
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "promotion_recommendation": recommendation,
            "soundness_errors": _int_or_none(payload.get("soundness_errors")),
            "completeness_errors": _int_or_none(payload.get("completeness_errors")),
            "forgetting_regression_count": _int_or_none(payload.get("forgetting_regression_count")),
            "satisfiable_drift_count": _int_or_none(payload.get("satisfiable_drift_count")),
            "ledger_consistency_rate": ledger,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _arm_ebt_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("arm_ebt_energy_budget_sidecar_v2_ready") is not True:
        status = "blocked"
    elif payload.get("live_integration") is not True or _as_list(
        payload.get("integration_blockers")
    ):
        status = "projection_only"
    else:
        status = "clean"
    correlation = _as_mapping(_as_mapping(payload.get("correlation_metrics")).get("sidecar_energy"))
    return _row(
        row_id="dot291:exp3130_arm_ebt_energy_budget",
        status=status,
        source_artifact=EXP3130_REL_PATH.as_posix(),
        source_field="arm_ebt_energy_budget_sidecar_v2_ready",
        evidence_class="arm_ebt_energy_budget_sidecar_diagnostic",
        claim_scope="architecture_energy_budget_boundary",
        summary={
            "arm_ebt_energy_budget_sidecar_v2_ready": payload.get(
                "arm_ebt_energy_budget_sidecar_v2_ready"
            )
            is True,
            "live_integration": payload.get("live_integration") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "integration_blocker_count": len(_as_list(payload.get("integration_blockers"))),
            "sidecar_energy_pearson_reject_or_repair": _float_or_none(
                correlation.get("pearson_reject_or_repair")
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _kan_row(payload: Mapping[str, Any]) -> JsonDict:
    claim_boundary = _as_mapping(payload.get("claim_boundary"))
    does_not_prove = _text_list(claim_boundary.get("does_not_prove"))
    if not payload:
        status = "missing"
    elif payload.get("kan_pwa_milp_audit_v1_ready") is not True or _as_list(
        payload.get("implementation_blockers")
    ):
        status = "blocked"
    elif does_not_prove:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot291:exp3131_kan_pwa_milp",
        status=status,
        source_artifact=EXP3131_REL_PATH.as_posix(),
        source_field="kan_pwa_milp_audit_v1_ready",
        evidence_class="kan_pwa_milp_verifier_abstraction_audit",
        claim_scope="architecture_kan_verifier_boundary",
        summary={
            "kan_pwa_milp_audit_v1_ready": payload.get("kan_pwa_milp_audit_v1_ready") is True,
            "kan_code_present": payload.get("kan_code_present") is True,
            "abstraction_count": _int_or_none(payload.get("abstraction_count")),
            "milp_property_check_count": _int_or_none(payload.get("milp_property_check_count")),
            "milp_property_pass_count": _int_or_none(payload.get("milp_property_pass_count")),
            "claim_boundary_proves": str(claim_boundary.get("proves") or ""),
            "claim_boundary_does_not_prove": does_not_prove,
            "implementation_blocker_count": len(_as_list(payload.get("implementation_blockers"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _hardware_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("hardware_evidence_sampler_boundary_v5_ready") is True
    gatemate = payload.get("gatemate_evidence_complete") is True
    ssqa = payload.get("ssqa_readback_ready") is True
    speedup = payload.get("speedup_claim_allowed") is True
    missing_operator = _as_list(payload.get("missing_operator_evidence"))
    if not payload:
        status = "missing"
    elif not ready or not gatemate or not ssqa or missing_operator:
        status = "blocked"
    elif speedup:
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="dot291:exp3132_hardware_sampler_boundary",
        status=status,
        source_artifact=EXP3132_REL_PATH.as_posix(),
        source_field="hardware_evidence_sampler_boundary_v5_ready",
        evidence_class="hardware_evidence_sampler_boundary",
        claim_scope="architecture_hardware_sampler_boundary",
        summary={
            "hardware_evidence_sampler_boundary_v5_ready": ready,
            "gatemate_evidence_complete": gatemate,
            "ssqa_readback_ready": ssqa,
            "speedup_claim_allowed": speedup,
            "hardware_commands_run": _as_list(payload.get("hardware_commands_run")),
            "missing_operator_evidence_count": len(missing_operator),
            "sampler_boundary_decisions": _as_mapping(payload.get("sampler_boundary_decisions")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _claim_entry(row: Mapping[str, Any]) -> JsonDict:
    status = normal_status(str(row.get("status") or "missing"))
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": str(row.get("evidence_class") or ""),
        "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
        "claim_scope": str(row.get("claim_scope") or ""),
        "summary": _as_mapping(row.get("summary")),
        "row_origin": str(row.get("row_origin") or "matrix_v24"),
    }


def _row(
    *,
    row_id: str,
    status: str,
    source_artifact: str,
    source_field: str,
    evidence_class: str,
    claim_scope: str,
    summary: Mapping[str, Any],
) -> JsonDict:
    normalized = normal_status(status)
    return {
        "row_id": row_id,
        "status": normalized,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence_class": evidence_class,
        "blocker_class": blocker_class(normalized),
        "claim_scope": claim_scope,
        "summary": dict(summary),
        "row_origin": "milestone_291",
    }


def _ready_status(present: bool, payload: Mapping[str, Any], ready_field: str) -> str:
    if not present:
        return "missing"
    if payload.get(ready_field) is True or str(payload.get("status") or "").lower() == "success":
        return "clean"
    return "blocked"


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    for row in rows:
        status = normal_status(str(row.get("status") or "missing"))
        if status in PUBLICATION_BLOCKING_STATUSES:
            blockers.append(
                {
                    "row_id": str(row.get("row_id") or ""),
                    "status": status,
                    "blocker_class": str(row.get("blocker_class") or blocker_class(status)),
                    "source_artifact": str(row.get("source_artifact") or ""),
                    "source_field": str(row.get("source_field") or ""),
                    "claim_scope": str(row.get("claim_scope") or ""),
                }
            )
    return blockers


def _prior_publication_blocker_count(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> int:
    return (
        _int_or_none(matrix.get("publication_blocker_count"))
        or _int_or_none(capstone.get("publication_blocker_count"))
        or 0
    )


def _missing_artifacts(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "path": str(row["path"]),
            "experiment_id": str(row["experiment_id"]),
            "reason": "missing_or_malformed_required_artifact"
            if row.get("required") is True
            else "missing_or_malformed_dot291_artifact",
        }
        for row in sources
        if row.get("readable_json_object") is not True
    ]


def _required_source_errors(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {"path": str(row["path"]), "reason": "missing_or_malformed_required_artifact"}
        for row in sources
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _headline_claim_allowance_summary(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    cache = payloads["exp3123"]
    verifier = payloads["exp3124"]
    cached_pair = cache.get("cached_sota_pair_available") is True
    cache_headline = cache.get("headline_claim_allowed") is True
    live_headline = (
        verifier.get("headline_claim_allowed") is True
        and str(verifier.get("repair_gate_state") or "") == "unblocked"
        and (_float_or_none(verifier.get("false_accept_rate")) or 0.0) == 0.0
    )
    blocked: list[str] = []
    if not cached_pair:
        blocked.append("comparative_sota_pair")
    if not live_headline:
        blocked.append("live_verifier_lift")
    return {
        "sota_cache_headline_allowed": cache_headline,
        "comparative_sota_pair_allowed": cached_pair,
        "live_verifier_headline_allowed": live_headline,
        "cached_sota_pair_available": cached_pair,
        "present_model_ids": _text_list(cache.get("present_model_ids")),
        "missing_model_ids": _text_list(cache.get("missing_model_ids")),
        "selected_headline_model_ids": _text_list(cache.get("selected_headline_model_ids")),
        "repair_gate_state": str(verifier.get("repair_gate_state") or ""),
        "false_accept_rate": _float_or_none(verifier.get("false_accept_rate")),
        "blocked_headline_claims": blocked,
    }


def _verifier_repair_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    verifier = payloads["exp3124"]
    ladder = payloads["exp3127"]
    blocker_rows = [
        row["row_id"]
        for row in _publication_blockers(rows)
        if str(row.get("claim_scope") or "")
        in {"local_sota_model_cache_policy", "live_sota_verifier_lift", "repair_live_rerun"}
    ]
    return {
        "sota_cache_status": statuses.get("dot291:exp3123_sota_cache_coverage", "missing"),
        "live_verifier_status": statuses.get("dot291:exp3124_live_verifier_lift", "missing"),
        "prefix_bounds_status": statuses.get("dot291:exp3125_prefix_bounds", "missing"),
        "fragment_time_monitor_status": statuses.get(
            "dot291:exp3126_fragment_time_monitors", "missing"
        ),
        "repair_ladder_status": statuses.get("dot291:exp3127_repair_ladder", "missing"),
        "repair_gate_state": str(verifier.get("repair_gate_state") or ""),
        "verifier_gain_delta": _float_or_none(verifier.get("verifier_gain_delta")) or 0.0,
        "false_accept_rate": _float_or_none(verifier.get("false_accept_rate")) or 0.0,
        "false_reject_rate": _float_or_none(verifier.get("false_reject_rate")) or 0.0,
        "repair_ladder_blocked_at_layer": str(ladder.get("blocked_at_layer") or ""),
        "repair_ladder_gate_check_summary": str(ladder.get("gate_check_summary") or ""),
        "publication_blocker_row_ids": blocker_rows,
    }


def _fr11_summary(
    payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]
) -> JsonDict:
    statuses = _row_statuses(rows)
    evo = payloads["exp3128"]
    memory = payloads["exp3129"]
    no_weight_update = (
        evo.get("no_weight_update_claim") is True or memory.get("no_weight_update_claim") is True
    )
    ledger = _float_or_none(memory.get("ledger_consistency_rate"))
    return {
        "evoenv_status": statuses.get("dot291:exp3128_fr11_evoenv", "missing"),
        "memory_status": statuses.get("dot291:exp3129_fr11_memory", "missing"),
        "continuous_self_learning_targeted": evo.get("continuous_self_learning_targeted") is True,
        "admitted_environment_count": _int_or_none(memory.get("admitted_environment_count"))
        or _int_or_none(evo.get("admitted_environment_count"))
        or 0,
        "no_weight_update_claim": no_weight_update,
        "model_weight_learning_allowed": not no_weight_update and (ledger is None or ledger >= 1.0),
        "promotion_recommendation": str(memory.get("promotion_recommendation") or ""),
        "soundness_errors": _int_or_none(memory.get("soundness_errors"))
        or _int_or_none(evo.get("soundness_errors"))
        or 0,
        "completeness_errors": _int_or_none(memory.get("completeness_errors"))
        or _int_or_none(evo.get("completeness_errors"))
        or 0,
        "forgetting_regression_count": _int_or_none(memory.get("forgetting_regression_count")) or 0,
        "ledger_consistency_rate": ledger,
    }


def _architecture_boundary_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    arm = payloads["exp3130"]
    hardware = payloads["exp3132"]
    return {
        "arm_ebt_status": statuses.get("dot291:exp3130_arm_ebt_energy_budget", "missing"),
        "kan_pwa_milp_status": statuses.get("dot291:exp3131_kan_pwa_milp", "missing"),
        "hardware_sampler_status": statuses.get(
            "dot291:exp3132_hardware_sampler_boundary", "missing"
        ),
        "live_integration": arm.get("live_integration") is True,
        "integration_blocker_count": len(_as_list(arm.get("integration_blockers"))),
        "speedup_claim_allowed": hardware.get("speedup_claim_allowed") is True,
        "hardware_commands_run": _as_list(hardware.get("hardware_commands_run")),
        "gatemate_evidence_complete": hardware.get("gatemate_evidence_complete") is True,
        "ssqa_readback_ready": hardware.get("ssqa_readback_ready") is True,
        "missing_operator_evidence_count": len(_as_list(hardware.get("missing_operator_evidence"))),
        "architecture_boundary_row_ids": _architecture_boundary_rows(rows),
    }


def _row_ids_by_status(rows: list[Mapping[str, Any]], status: str) -> list[str]:
    return [
        str(row.get("row_id") or "")
        for row in rows
        if normal_status(str(row.get("status") or "")) == status
    ]


def _gated_skips(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    gated: list[JsonDict] = []
    for row in rows:
        summary = _as_mapping(row.get("summary"))
        status = normal_status(str(row.get("status") or ""))
        if status == "gated_skipped" or summary.get("gated_skip") is True:
            gated.append(
                {
                    "row_id": str(row.get("row_id") or ""),
                    "status": status,
                    "source_artifact": str(row.get("source_artifact") or ""),
                    "reason": str(
                        summary.get("gate_check_summary")
                        or summary.get("status_rationale")
                        or "gated_skip_status"
                    ),
                }
            )
    return gated


def _architecture_boundary_rows(rows: list[Mapping[str, Any]]) -> list[str]:
    boundary_scopes = {
        "architecture_energy_budget_boundary",
        "architecture_kan_verifier_boundary",
        "architecture_hardware_sampler_boundary",
    }
    return [
        str(row.get("row_id") or "")
        for row in rows
        if str(row.get("claim_scope") or "") in boundary_scopes
    ]


def _row_statuses(rows: list[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("row_id") or ""): normal_status(str(row.get("status") or "missing"))
        for row in rows
    }


def _public_sources(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row["experiment_id"]),
            "path": str(row["path"]),
            "role": str(row["role"]),
            "required": row.get("required") is True,
            "ready_field": str(row.get("ready_field") or ""),
            "present": row.get("present") is True,
            "readable_json_object": row.get("readable_json_object") is True,
            "sha256": row.get("sha256"),
            "source_type": str(row.get("source_type") or "json"),
        }
        for row in sources
    ]


def _invariant_violations(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blockers: list[Mapping[str, Any]],
    required_source_errors: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if required_source_errors:
        violations.append("required source artifacts missing or malformed")
    if matrix and matrix.get("matrix_v24_ready") is not True:
        violations.append("matrix v24 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone v290 authority is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v25 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to rows_total")
    if len(publication_blockers) != sum(
        count for status, count in status_counts.items() if status in PUBLICATION_BLOCKING_STATUSES
    ):
        violations.append("publication_blocker_count does not match row statuses")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot291_artifacts",
        "source": "matrix_v24_capstone_v290_and_dot291_artifacts",
        "executes_models": False,
        "executes_verifiers": False,
        "executes_repairs": False,
        "executes_solvers": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("matrix_v25_ready") is not True:
        return (
            "blocked_matrix_v25_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"invariant_violations={len(_as_list(artifact.get('invariant_violations')))}"
        )
    return (
        "complete: matrix_v25_ready=true; "
        f"rows_total={artifact.get('rows_total')}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v24={artifact.get('blocker_delta_from_v24')}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}"
    )


def normal_status(value: str) -> str:
    text = value.strip().lower().replace("-", "_")
    aliases = {
        "success": "clean",
        "passed": "clean",
        "complete": "clean",
        "gate_skipped": "gated_skipped",
        "gate_blocked": "gated_skipped",
        "diagnostic": "diagnostic_only",
        "pilot_only": "bounded",
        "projection": "projection_only",
    }
    text = aliases.get(text, text)
    return text if text in STATUSES else "missing"


def blocker_class(status: str) -> str:
    normalized = normal_status(status)
    if normalized in {"clean", "retired"}:
        return "none"
    if normalized == "diagnostic_only":
        return "diagnostic_only"
    if normalized in PUBLICATION_BLOCKING_STATUSES:
        return normalized
    return "unknown"  # pragma: no cover - all current status vocabulary is classified above.


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value)]


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)
