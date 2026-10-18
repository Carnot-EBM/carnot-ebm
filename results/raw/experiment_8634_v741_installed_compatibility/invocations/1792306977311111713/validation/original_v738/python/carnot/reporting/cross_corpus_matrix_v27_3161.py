"""Build the Exp 3161 cross-corpus matrix v27 artifact.

Spec refs: REQ-REPORT-3161, SCENARIO-REPORT-3161.

Matrix v27 is an evidence ledger over checked-in `.293` artifacts. It carries
the v26 matrix forward, then adds one row per .293 task so the next capstone
can see exactly which work is clean, blocked, flagged, gated, diagnostic,
projection-only, bounded, missing, or retired. The builder does not run live
models, verifier panels, repairs, solvers, hardware commands, the conductor,
or pushes; it only reads JSON artifacts and preserves their boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from carnot.reporting import cross_corpus_matrix_v26_3147 as v26


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.293"
SCHEMA = "carnot.cross_corpus_matrix.v27_293_artifact_aggregation.v1"
ARTIFACT = "experiment_3161_cross_corpus_matrix_v27"
OUTPUT_REL_PATH = Path("results/experiment_3161_cross_corpus_matrix_v27.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3161_cross_corpus_matrix_v27.py"

MATRIX_V26_REL_PATH = Path("results/experiment_3147_cross_corpus_matrix_v26.json")
CAPSTONE_V292_REL_PATH = Path("results/experiment_3148_capstone_v292.json")
EXP3149_REL_PATH = Path("results/experiment_3149_archive_v292_activate_v293.json")
EXP3150_REL_PATH = Path("results/experiment_3150_adversarial_verifier_evidence_corrigendum_v1.json")
EXP3151_REL_PATH = Path("results/experiment_3151_live_inference_authenticity_preflight_v1.json")
EXP3152_REL_PATH = Path("results/experiment_3152_clean_live_sota_verifier_rerun_v8.json")
EXP3153_REL_PATH = Path("results/experiment_3153_repair_gate_unlock_decision_v2.json")
EXP3154_REL_PATH = Path("results/experiment_3154_multi_turn_repair_ladder_v3.json")
EXP3155_REL_PATH = Path("results/experiment_3155_tracefix_counterexample_repair_pilot_v1.json")
EXP3155_ALIAS_REL_PATH = Path("results/experiment_3155_tracefix_counterexample_repair_pilot.json")
EXP3156_REL_PATH = Path("results/experiment_3156_fr11_ledger_consistency_closure_v1.json")
EXP3157_REL_PATH = Path("results/experiment_3157_fr11_attractor_residual_memory_audit_v1.json")
EXP3158_REL_PATH = Path("results/experiment_3158_ebcn_energy_sidecar_calibration_v1.json")
EXP3159_REL_PATH = Path("results/experiment_3159_kan_proof_carrying_monitor_expansion_v1.json")
EXP3160_REL_PATH = Path("results/experiment_3160_hardware_sampler_evidence_boundary_v7.json")

STATUSES = (
    "clean",
    "blocked",
    "flagged",
    "gated_skipped",
    "diagnostic_only",
    "projection_only",
    "bounded",
    "missing",
    "retired",
)
PUBLICATION_BLOCKING_STATUSES = {
    "blocked",
    "flagged",
    "gated_skipped",
    "projection_only",
    "bounded",
    "missing",
}

read_json_object = v26.read_json_object
sha256_file = v26.sha256_file
_as_mapping = v26._as_mapping
_as_list = v26._as_list
_text_list = v26._text_list
_int_or_none = v26._int_or_none
_float_or_none = v26._float_or_none


@dataclass(frozen=True)
class SourceSpec:
    """One JSON artifact that v27 reads and cites without mutating it."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    ready_field: str = ""
    aliases: tuple[Path, ...] = ()


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3147", MATRIX_V26_REL_PATH, "matrix_v26_authority", True, "matrix_v26_ready"),
    SourceSpec(
        "exp3148", CAPSTONE_V292_REL_PATH, "capstone_v292_authority", True, "capstone_ready"
    ),
    SourceSpec(
        "exp3149",
        EXP3149_REL_PATH,
        "archive_v292_activate_v293",
        False,
        "archive_v292_activate_v293_ready",
    ),
    SourceSpec(
        "exp3150",
        EXP3150_REL_PATH,
        "adversarial_corrigendum",
        False,
        "adversarial_corrigendum_v1_ready",
    ),
    SourceSpec(
        "exp3151",
        EXP3151_REL_PATH,
        "live_inference_authenticity_preflight",
        False,
        "live_inference_authenticity_preflight_ready",
    ),
    SourceSpec(
        "exp3152",
        EXP3152_REL_PATH,
        "clean_live_verifier_rerun",
        False,
        "clean_live_verifier_rerun_v8_ready",
    ),
    SourceSpec(
        "exp3153", EXP3153_REL_PATH, "repair_gate_v2", False, "repair_gate_decision_v2_ready"
    ),
    SourceSpec(
        "exp3154", EXP3154_REL_PATH, "repair_ladder_v3", False, "multi_turn_repair_ladder_v3_ready"
    ),
    SourceSpec(
        "exp3155",
        EXP3155_REL_PATH,
        "tracefix_counterexample_repair",
        False,
        "tracefix_counterexample_repair_pilot_v1_ready",
        (EXP3155_ALIAS_REL_PATH,),
    ),
    SourceSpec(
        "exp3156",
        EXP3156_REL_PATH,
        "fr11_ledger_consistency_closure",
        False,
        "fr11_ledger_consistency_closure_v1_ready",
    ),
    SourceSpec(
        "exp3157",
        EXP3157_REL_PATH,
        "fr11_attractor_residual_memory",
        False,
        "fr11_attractor_residual_memory_audit_v1_ready",
    ),
    SourceSpec(
        "exp3158",
        EXP3158_REL_PATH,
        "ebcn_energy_sidecar_calibration",
        False,
        "ebcn_energy_sidecar_calibration_v1_ready",
    ),
    SourceSpec(
        "exp3159",
        EXP3159_REL_PATH,
        "kan_monitor_expansion",
        False,
        "kan_proof_carrying_monitor_expansion_v1_ready",
    ),
    SourceSpec(
        "exp3160",
        EXP3160_REL_PATH,
        "hardware_sampler_boundary",
        False,
        "hardware_sampler_evidence_boundary_v7_ready",
    ),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3161: aggregate matrix v27 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3147"]
    capstone = payloads["exp3148"]
    rows = _carry_forward_rows(matrix) + _dot293_rows(payloads) if matrix else []
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    prior_count = _prior_publication_blocker_count(matrix, capstone)
    missing_artifacts = _missing_artifacts(matrix, sources)
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
        "matrix_v27_ready": ready,
        "rows_total": len(rows),
        "status_counts": status_counts,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v26": len(publication_blockers) - prior_count,
        "inherited_adversarial_flag_count": _inherited_adversarial_flag_count(payloads),
        "missing_artifacts": missing_artifacts,
        "publication_blockers": publication_blockers,
        "rows": rows,
        "false_accept_recovery_summary": _false_accept_recovery_summary(payloads, rows, matrix),
        "repair_summary": _repair_summary(payloads, rows),
        "fr11_summary": _fr11_summary(payloads, rows),
        "architecture_boundary_summary": _architecture_boundary_summary(payloads, rows),
        "paper_readiness_implications": _paper_readiness_implications(
            rows, len(publication_blockers), payloads
        ),
        "source_artifacts": _public_sources(sources),
        "source_checksums": {
            str(row["path"]): row.get("sha256") for row in _public_sources(sources)
        },
        "required_source_errors": required_source_errors,
        "invariant_violations": invariant_violations,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
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
    """Build and persist the Exp 3161 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def normal_status(status: str) -> str:
    """Normalize historical matrix statuses into the v27 closed vocabulary."""

    normalized = str(status or "").strip()
    if normalized == "model_spec_gap":
        return "blocked"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map row status to the blocker class used by publication summaries."""

    normalized = normal_status(status)
    if normalized == "clean":
        return "none"
    if normalized == "retired":
        return "retired_non_blocking"
    if normalized == "diagnostic_only":
        return "diagnostic_non_blocking"
    return f"publication_blocker_{normalized}"


def _source_payload(root: Path, spec: SourceSpec) -> JsonDict:
    primary = root / spec.path
    loaded_path = spec.path
    payload = read_json_object(primary)
    for alias in spec.aliases:
        if payload:
            break
        alias_payload = read_json_object(root / alias)
        if alias_payload:
            payload = alias_payload
            loaded_path = alias
    loaded_abs = root / loaded_path
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "loaded_path": loaded_path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "ready_field": spec.ready_field,
        "source_type": "json",
        "primary_present": primary.is_file(),
        "present": loaded_abs.is_file(),
        "readable_json_object": bool(payload),
        "alias_loaded": loaded_path != spec.path,
        "payload": payload,
        "sha256": sha256_file(loaded_abs),
    }


def _carry_forward_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for raw in _as_list(matrix.get("rows")):
        if not isinstance(raw, Mapping):
            continue
        row = _claim_entry(raw)
        summary = _as_mapping(row.get("summary"))
        summary.setdefault("v27_status_rationale", "carried_forward_from_matrix_v26")
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v26")
        rows.append(row)
    return rows


def _dot293_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _archive_row(payloads["exp3149"]),
        _corrigendum_row(payloads["exp3150"]),
        _preflight_row(payloads["exp3151"]),
        _clean_rerun_row(payloads["exp3152"], payloads["exp3151"], payloads["exp3150"]),
        _repair_gate_v2_row(payloads["exp3153"]),
        _repair_ladder_row(payloads["exp3154"], payloads["exp3153"]),
        _tracefix_row(payloads["exp3155"]),
        _fr11_ledger_row(payloads["exp3156"]),
        _fr11_residual_row(payloads["exp3157"]),
        _energy_row(payloads["exp3158"]),
        _kan_row(payloads["exp3159"]),
        _hardware_row(payloads["exp3160"]),
    ]


def _archive_row(payload: Mapping[str, Any]) -> JsonDict:
    status = _ready_status(payload, "archive_v292_activate_v293_ready")
    return _row(
        row_id="dot293:exp3149_archive_handoff",
        status=status,
        source_artifact=EXP3149_REL_PATH.as_posix(),
        source_field="archive_v292_activate_v293_ready",
        evidence_class="archive_v292_activate_v293",
        claim_scope="milestone_handoff",
        summary={
            "archive_v292_activate_v293_ready": payload.get("archive_v292_activate_v293_ready")
            is True,
            "prior_publication_blocker_count": _int_or_none(
                payload.get("prior_publication_blocker_count")
            ),
            "carry_forward_blocker_count": len(_as_list(payload.get("carry_forward_blockers"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _corrigendum_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("adversarial_corrigendum_v1_ready") is not True:
        status = "blocked"
    elif (
        payload.get("live_verifier_evidence_trusted") is False
        or (_int_or_none(payload.get("flagged_artifact_count")) or 0) > 0
        or bool(_as_list(payload.get("blocked_downstream_fields")))
    ):
        status = "flagged"
    else:
        status = "clean"
    return _row(
        row_id="dot293:exp3150_adversarial_corrigendum",
        status=status,
        source_artifact=EXP3150_REL_PATH.as_posix(),
        source_field="live_verifier_evidence_trusted",
        evidence_class="adversarial_verifier_evidence_corrigendum",
        claim_scope="live_verifier_source_trust",
        summary={
            "adversarial_corrigendum_v1_ready": payload.get("adversarial_corrigendum_v1_ready")
            is True,
            "flagged_artifact_count": _int_or_none(payload.get("flagged_artifact_count")) or 0,
            "inherited_adversarial_flag_count": _inherited_flags_from_corrigendum(payload),
            "known_false_accept_recovery_preserved": payload.get(
                "known_false_accept_recovery_preserved"
            )
            is True,
            "live_verifier_evidence_trusted": payload.get("live_verifier_evidence_trusted") is True,
            "repair_gate_implication": str(payload.get("repair_gate_implication") or ""),
            "blocked_downstream_field_count": len(
                _as_list(payload.get("blocked_downstream_fields"))
            ),
            "safe_downstream_field_count": len(_as_list(payload.get("safe_downstream_fields"))),
            "methodology_requirement_count": len(
                _as_list(payload.get("methodology_requirements_for_rerun"))
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _preflight_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("live_inference_authenticity_preflight_ready") is not True
        or payload.get("preflight_passed") is not True
    ):
        status = "blocked"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot293:exp3151_live_preflight",
        status=status,
        source_artifact=EXP3151_REL_PATH.as_posix(),
        source_field="preflight_passed",
        evidence_class="live_inference_authenticity_preflight",
        claim_scope="live_verifier_preflight",
        summary={
            "live_inference_authenticity_preflight_ready": payload.get(
                "live_inference_authenticity_preflight_ready"
            )
            is True,
            "preflight_passed": payload.get("preflight_passed") is True,
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "minimum_duration_requirement_s": _float_or_none(
                payload.get("minimum_duration_requirement_s")
            ),
            "duration_s": _float_or_none(payload.get("duration_s")),
            "blocked_reason": str(payload.get("blocked_reason") or ""),
            "selected_model_ids": _text_list(payload.get("selected_model_ids")),
            "locally_usable_model_ids": _text_list(payload.get("locally_usable_model_ids")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _clean_rerun_row(
    payload: Mapping[str, Any],
    preflight: Mapping[str, Any],
    corrigendum: Mapping[str, Any],
) -> JsonDict:
    gate_failed = preflight.get("preflight_passed") is False or str(
        corrigendum.get("repair_gate_implication") or ""
    ).startswith("blocked")
    if not payload and gate_failed:
        status = "gated_skipped"
    elif not payload:
        status = "missing"
    elif payload.get("clean_live_verifier_rerun_v8_ready") is not True:
        status = normal_status(str(payload.get("status") or "blocked"))
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif (
        payload.get("false_accept_gate_passed") is True
        and payload.get("headline_claim_allowed") is True
    ):
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="dot293:exp3152_clean_live_rerun",
        status=status,
        source_artifact=EXP3152_REL_PATH.as_posix(),
        source_field="clean_live_verifier_rerun_v8_ready",
        evidence_class="clean_live_sota_verifier_rerun_v8",
        claim_scope="live_verifier_headline_evidence",
        summary={
            "clean_live_verifier_rerun_v8_ready": payload.get("clean_live_verifier_rerun_v8_ready")
            is True,
            "gated_skip": status == "gated_skipped",
            "preflight_passed": preflight.get("preflight_passed") is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "false_accept_gate_passed": payload.get("false_accept_gate_passed") is True,
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "false_accept_rate": _float_or_none(payload.get("false_accept_rate")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_gate_v2_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("repair_gate_decision_v2_ready") is True:
        status = "clean" if payload.get("repair_gate_state") == "unblocked" else "blocked"
    else:
        status = normal_status(str(payload.get("status") or "blocked"))
    return _row(
        row_id="dot293:exp3153_repair_gate_v2",
        status=status,
        source_artifact=EXP3153_REL_PATH.as_posix(),
        source_field="repair_gate_decision_v2_ready",
        evidence_class="repair_gate_unlock_decision_v2",
        claim_scope="repair_gate_decision",
        summary={
            "repair_gate_decision_v2_ready": payload.get("repair_gate_decision_v2_ready") is True,
            "repair_gate_state": str(payload.get("repair_gate_state") or ""),
            "source_status": str(payload.get("status") or ""),
            "blocked_at_layer": str(payload.get("blocked_at_layer") or ""),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
            "gates_evaluated_count": len(_as_list(payload.get("gates_evaluated"))),
            "selected_repair_rows": _as_list(payload.get("selected_repair_rows")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_ladder_row(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> JsonDict:
    gate_state = str(gate.get("repair_gate_state") or gate.get("status") or "")
    if not payload and gate_state and gate_state != "unblocked":
        status = "gated_skipped"
    elif not payload:
        status = "missing"
    elif payload.get("multi_turn_repair_ladder_v3_ready") is True:
        status = "clean" if payload.get("headline_repair_claim_allowed") is True else "bounded"
    else:
        status = normal_status(str(payload.get("status") or "blocked"))
    return _row(
        row_id="dot293:exp3154_repair_ladder_v3",
        status=status,
        source_artifact=EXP3154_REL_PATH.as_posix(),
        source_field="multi_turn_repair_ladder_v3_ready",
        evidence_class="multi_turn_repair_ladder_v3",
        claim_scope="repair_execution",
        summary={
            "multi_turn_repair_ladder_v3_ready": payload.get("multi_turn_repair_ladder_v3_ready")
            is True,
            "gated_skip": status == "gated_skipped",
            "repair_gate_state": gate_state,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "repair_attempt_count": _int_or_none(payload.get("repair_attempt_count")) or 0,
            "headline_repair_claim_allowed": payload.get("headline_repair_claim_allowed") is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _tracefix_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("tracefix_counterexample_repair_pilot_v1_ready") is True:
        status = "clean"
    else:
        status = normal_status(str(payload.get("status") or "blocked"))
    return _row(
        row_id="dot293:exp3155_tracefix_repair",
        status=status,
        source_artifact=EXP3155_ALIAS_REL_PATH.as_posix()
        if payload.get("schema") == "blocked_gate_check_v1"
        else EXP3155_REL_PATH.as_posix(),
        source_field="tracefix_counterexample_repair_pilot_v1_ready",
        evidence_class="tracefix_counterexample_repair_pilot",
        claim_scope="formal_counterexample_repair",
        summary={
            "tracefix_counterexample_repair_pilot_v1_ready": payload.get(
                "tracefix_counterexample_repair_pilot_v1_ready"
            )
            is True,
            "source_status": str(payload.get("status") or ""),
            "blocked_at_layer": str(payload.get("blocked_at_layer") or ""),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
            "counterexample_count": _int_or_none(payload.get("counterexample_count")) or 0,
            "exact_replay_pass_count": _int_or_none(payload.get("exact_replay_pass_count")) or 0,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_ledger_row(payload: Mapping[str, Any]) -> JsonDict:
    ledger = _float_or_none(payload.get("ledger_consistency_rate"))
    if not payload:
        status = "missing"
    elif payload.get("fr11_ledger_consistency_closure_v1_ready") is not True:
        status = "blocked"
    elif (
        (ledger is not None and ledger < 1.0)
        or payload.get("no_weight_update_claim") is True
        or (_int_or_none(payload.get("soundness_errors")) or 0) > 0
        or (_int_or_none(payload.get("completeness_errors")) or 0) > 0
        or str(payload.get("promotion_recommendation") or "").startswith("block")
    ):
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot293:exp3156_fr11_ledger_closure",
        status=status,
        source_artifact=EXP3156_REL_PATH.as_posix(),
        source_field="ledger_consistency_rate",
        evidence_class="fr11_ledger_consistency_closure",
        claim_scope="fr11_self_learning_promotion_gate",
        summary={
            "fr11_ledger_consistency_closure_v1_ready": payload.get(
                "fr11_ledger_consistency_closure_v1_ready"
            )
            is True,
            "continuous_self_learning_targeted": payload.get("continuous_self_learning_targeted")
            is True,
            "replay_panel_count": _int_or_none(payload.get("replay_panel_count")) or 0,
            "ledger_consistency_rate": ledger,
            "ledger_consistent_count": _int_or_none(payload.get("ledger_consistent_count")) or 0,
            "soundness_errors": _int_or_none(payload.get("soundness_errors")) or 0,
            "completeness_errors": _int_or_none(payload.get("completeness_errors")) or 0,
            "methodology_complete": payload.get("methodology_complete") is True,
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "promotion_recommendation": str(payload.get("promotion_recommendation") or ""),
            "residual_mismatch_count": len(_as_list(payload.get("residual_mismatch_rows"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_residual_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("fr11_attractor_residual_memory_audit_v1_ready") is not True
        or (_int_or_none(payload.get("unsafe_skip_count")) or 0) > 0
    ):
        status = "blocked"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot293:exp3157_fr11_residual_memory",
        status=status,
        source_artifact=EXP3157_REL_PATH.as_posix(),
        source_field="fr11_attractor_residual_memory_audit_v1_ready",
        evidence_class="fr11_attractor_residual_memory_audit",
        claim_scope="fr11_controller_memory_diagnostic",
        summary={
            "fr11_attractor_residual_memory_audit_v1_ready": payload.get(
                "fr11_attractor_residual_memory_audit_v1_ready"
            )
            is True,
            "continuous_self_learning_targeted": payload.get("continuous_self_learning_targeted")
            is True,
            "replay_panel_count": _int_or_none(payload.get("replay_panel_count")) or 0,
            "ledger_consistency_rate": _float_or_none(payload.get("ledger_consistency_rate")),
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "promotion_recommendation": str(payload.get("promotion_recommendation") or ""),
            "redundant_check_suppression_rate": _float_or_none(
                payload.get("redundant_check_suppression_rate")
            ),
            "unsafe_skip_count": _int_or_none(payload.get("unsafe_skip_count")) or 0,
            "risky_family_escalation_rate": _float_or_none(
                payload.get("risky_family_escalation_rate")
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _energy_row(payload: Mapping[str, Any]) -> JsonDict:
    blockers = _as_list(payload.get("residual_blockers"))
    if not payload:
        status = "missing"
    elif payload.get("ebcn_energy_sidecar_calibration_v1_ready") is not True:
        status = "blocked"
    elif payload.get("live_integration_claim_allowed") is not True or blockers:
        status = "projection_only"
    else:
        status = "clean"
    return _row(
        row_id="dot293:exp3158_ebcn_energy_sidecar",
        status=status,
        source_artifact=EXP3158_REL_PATH.as_posix(),
        source_field="live_integration_claim_allowed",
        evidence_class="ebcn_energy_sidecar_calibration",
        claim_scope="architecture_energy_sidecar_boundary",
        summary={
            "ebcn_energy_sidecar_calibration_v1_ready": payload.get(
                "ebcn_energy_sidecar_calibration_v1_ready"
            )
            is True,
            "exact_labeled_row_count": _int_or_none(payload.get("exact_labeled_row_count")) or 0,
            "known_false_accept_rows_scored": _int_or_none(
                payload.get("known_false_accept_rows_scored")
            )
            or 0,
            "scalar_energy_auc": _float_or_none(payload.get("scalar_energy_auc")),
            "violation_localization_coverage": _float_or_none(
                payload.get("violation_localization_coverage")
            ),
            "live_integration_claim_allowed": payload.get("live_integration_claim_allowed") is True,
            "residual_blocker_count": len(blockers),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _kan_row(payload: Mapping[str, Any]) -> JsonDict:
    blockers = _as_list(payload.get("residual_blockers")) + _as_list(
        payload.get("implementation_blockers")
    )
    if not payload:
        status = "missing"
    elif payload.get("kan_proof_carrying_monitor_expansion_v1_ready") is not True:
        status = "blocked"
    elif payload.get("deployed_verifier_claim_allowed") is not True or blockers:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot293:exp3159_kan_monitor_expansion",
        status=status,
        source_artifact=EXP3159_REL_PATH.as_posix(),
        source_field="deployed_verifier_claim_allowed",
        evidence_class="kan_proof_carrying_monitor_expansion",
        claim_scope="architecture_kan_monitor_boundary",
        summary={
            "kan_proof_carrying_monitor_expansion_v1_ready": payload.get(
                "kan_proof_carrying_monitor_expansion_v1_ready"
            )
            is True,
            "monitor_record_count": _int_or_none(payload.get("monitor_record_count")) or 0,
            "new_monitor_record_count": _int_or_none(payload.get("new_monitor_record_count")) or 0,
            "exact_row_coverage_count": _int_or_none(payload.get("exact_row_coverage_count")) or 0,
            "deployed_verifier_claim_allowed": payload.get("deployed_verifier_claim_allowed")
            is True,
            "claim_boundary_proves": str(
                _as_mapping(payload.get("claim_boundary")).get("proves") or ""
            ),
            "claim_boundary_does_not_prove": _text_list(
                _as_mapping(payload.get("claim_boundary")).get("does_not_prove")
            ),
            "residual_blocker_count": len(_as_list(payload.get("residual_blockers"))),
            "implementation_blocker_count": len(_as_list(payload.get("implementation_blockers"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _hardware_row(payload: Mapping[str, Any]) -> JsonDict:
    missing_operator = _as_list(payload.get("missing_operator_evidence"))
    if not payload:
        status = "missing"
    elif payload.get("hardware_sampler_evidence_boundary_v7_ready") is not True:
        status = "blocked"
    elif payload.get("authenticated_speedup_claim_allowed") is True and not missing_operator:
        status = "clean"
    elif missing_operator or str(payload.get("gatemate_status") or "").startswith("blocked"):
        status = "blocked"
    else:
        status = "bounded"
    return _row(
        row_id="dot293:exp3160_hardware_boundary",
        status=status,
        source_artifact=EXP3160_REL_PATH.as_posix(),
        source_field="authenticated_speedup_claim_allowed",
        evidence_class="hardware_sampler_evidence_boundary_v7",
        claim_scope="architecture_hardware_sampler_boundary",
        summary={
            "hardware_sampler_evidence_boundary_v7_ready": payload.get(
                "hardware_sampler_evidence_boundary_v7_ready"
            )
            is True,
            "authenticated_speedup_claim_allowed": payload.get(
                "authenticated_speedup_claim_allowed"
            )
            is True,
            "no_hardware_commands_run": payload.get("no_hardware_commands_run") is True,
            "cuda_status": str(payload.get("cuda_status") or ""),
            "kv260_status": str(payload.get("kv260_status") or ""),
            "gatemate_status": str(payload.get("gatemate_status") or ""),
            "polarfire_status": str(payload.get("polarfire_status") or ""),
            "extropic_thrml_status": str(payload.get("extropic_thrml_status") or ""),
            "kona_status": str(payload.get("kona_status") or ""),
            "missing_operator_evidence_count": len(missing_operator),
            "missing_required_source_artifacts": _text_list(
                payload.get("missing_required_source_artifacts")
            ),
            "hardware_commands_run": _as_list(payload.get("hardware_commands_run")),
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
        "blocker_class": blocker_class(status),
        "claim_scope": str(row.get("claim_scope") or ""),
        "summary": _as_mapping(row.get("summary")),
        "row_origin": str(row.get("row_origin") or "matrix_v26"),
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
        "row_origin": "milestone_293",
    }


def _ready_status(payload: Mapping[str, Any], ready_field: str) -> str:
    if not payload:
        return "missing"
    return "clean" if payload.get(ready_field) is True else "blocked"


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


def _missing_artifacts(
    matrix: Mapping[str, Any],
    sources: list[Mapping[str, Any]],
) -> list[JsonDict]:
    missing: list[JsonDict] = []
    for row in _as_list(matrix.get("missing_artifacts")):
        if isinstance(row, Mapping):
            missing.append(
                {
                    "path": str(row.get("path") or ""),
                    "experiment_id": str(row.get("experiment_id") or ""),
                    "reason": "carried_forward_missing_artifact_from_v26",
                }
            )
    for row in sources:
        if row.get("required") is True:
            continue
        if row.get("readable_json_object") is not True:
            missing.append(
                {
                    "path": str(row["path"]),
                    "experiment_id": str(row["experiment_id"]),
                    "reason": "missing_or_gated_dot293_artifact",
                }
            )
        elif row.get("alias_loaded") is True and row.get("primary_present") is not True:
            missing.append(
                {
                    "path": str(row["path"]),
                    "experiment_id": str(row["experiment_id"]),
                    "reason": "missing_expected_dot293_deliverable_alias_loaded",
                    "loaded_alias_path": str(row["loaded_path"]),
                }
            )
    return missing


def _required_source_errors(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {"path": str(row["path"]), "reason": "missing_or_malformed_required_artifact"}
        for row in sources
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _inherited_adversarial_flag_count(payloads: Mapping[str, Mapping[str, Any]]) -> int:
    return _inherited_flags_from_corrigendum(payloads["exp3150"])


def _inherited_flags_from_corrigendum(payload: Mapping[str, Any]) -> int:
    counts = _as_mapping(payload.get("adversarial_flag_counts"))
    direct = _int_or_none(counts.get("aggregate_inherited_flag"))
    if direct is not None:
        return direct
    return sum(
        1
        for row in _as_list(payload.get("audited_artifacts"))
        if _as_mapping(row).get("inherited_flag") is True
    )


def _false_accept_recovery_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    matrix: Mapping[str, Any],
) -> JsonDict:
    statuses = _row_statuses(rows)
    v26_recovery = _as_mapping(matrix.get("false_accept_recovery_summary"))
    corrigendum = payloads["exp3150"]
    preflight = payloads["exp3151"]
    live_trusted = corrigendum.get("live_verifier_evidence_trusted") is True
    clean_status = statuses.get("dot293:exp3152_clean_live_rerun", "missing")
    if live_trusted and clean_status == "clean":
        claim_status = "clean_live_verifier_recovery_ready"
    elif corrigendum.get("known_false_accept_recovery_preserved") is True:
        claim_status = "exact_replay_preserved_but_live_verifier_untrusted"
    else:
        claim_status = "blocked_or_missing_recovery_evidence"
    return {
        "v26_recovery_claim_status": str(v26_recovery.get("recovery_claim_status") or ""),
        "source_false_accept_rate": _float_or_none(v26_recovery.get("source_false_accept_rate")),
        "v26_rerun_false_accept_rate": _float_or_none(v26_recovery.get("rerun_false_accept_rate")),
        "known_false_accept_rows_blocked": v26_recovery.get("known_false_accept_rows_blocked")
        is True,
        "known_false_accept_recovery_preserved": corrigendum.get(
            "known_false_accept_recovery_preserved"
        )
        is True,
        "live_verifier_evidence_trusted": live_trusted,
        "preflight_passed": preflight.get("preflight_passed") is True,
        "preflight_status": statuses.get("dot293:exp3151_live_preflight", "missing"),
        "clean_live_rerun_status": clean_status,
        "repair_gate_implication": str(corrigendum.get("repair_gate_implication") or ""),
        "blocked_downstream_field_count": len(
            _as_list(corrigendum.get("blocked_downstream_fields"))
        ),
        "safe_downstream_field_count": len(_as_list(corrigendum.get("safe_downstream_fields"))),
        "methodology_requirement_count": len(
            _as_list(corrigendum.get("methodology_requirements_for_rerun"))
        ),
        "recovery_claim_status": claim_status,
    }


def _repair_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    gate = payloads["exp3153"]
    ladder = payloads["exp3154"]
    tracefix = payloads["exp3155"]
    gate_state = str(gate.get("repair_gate_state") or gate.get("status") or "")
    return {
        "repair_gate_status": statuses.get("dot293:exp3153_repair_gate_v2", "missing"),
        "repair_ladder_status": statuses.get("dot293:exp3154_repair_ladder_v3", "missing"),
        "tracefix_status": statuses.get("dot293:exp3155_tracefix_repair", "missing"),
        "repair_gate_state": gate_state,
        "blocked_at_layer": str(gate.get("blocked_at_layer") or ""),
        "gate_check_summary": str(gate.get("gate_check_summary") or ""),
        "gates_evaluated_count": len(_as_list(gate.get("gates_evaluated"))),
        "repair_attempt_count": _int_or_none(ladder.get("repair_attempt_count")) or 0,
        "tracefix_counterexample_count": _int_or_none(tracefix.get("counterexample_count")) or 0,
        "repair_claim_allowed": statuses.get("dot293:exp3153_repair_gate_v2") == "clean"
        and statuses.get("dot293:exp3154_repair_ladder_v3") == "clean",
        "live_repair_executed": (_int_or_none(ladder.get("live_call_count")) or 0) > 0,
        "selected_repair_rows": _as_list(gate.get("selected_repair_rows")),
    }


def _fr11_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    ledger = payloads["exp3156"]
    residual = payloads["exp3157"]
    ledger_rate = _float_or_none(ledger.get("ledger_consistency_rate"))
    no_weight_update = (
        ledger.get("no_weight_update_claim") is True
        or residual.get("no_weight_update_claim") is True
    )
    return {
        "ledger_status": statuses.get("dot293:exp3156_fr11_ledger_closure", "missing"),
        "residual_memory_status": statuses.get("dot293:exp3157_fr11_residual_memory", "missing"),
        "continuous_self_learning_targeted": ledger.get("continuous_self_learning_targeted") is True
        or residual.get("continuous_self_learning_targeted") is True,
        "ledger_consistency_rate": ledger_rate,
        "ledger_consistent_count": _int_or_none(ledger.get("ledger_consistent_count")) or 0,
        "replay_panel_count": _int_or_none(ledger.get("replay_panel_count"))
        or _int_or_none(residual.get("replay_panel_count"))
        or 0,
        "soundness_errors": _int_or_none(ledger.get("soundness_errors")) or 0,
        "completeness_errors": _int_or_none(ledger.get("completeness_errors")) or 0,
        "residual_mismatch_count": len(_as_list(ledger.get("residual_mismatch_rows"))),
        "no_weight_update_claim": no_weight_update,
        "model_weight_learning_allowed": (
            not no_weight_update
            and ledger_rate == 1.0
            and (_int_or_none(ledger.get("soundness_errors")) or 0) == 0
            and (_int_or_none(ledger.get("completeness_errors")) or 0) == 0
        ),
        "promotion_recommendation": str(
            ledger.get("promotion_recommendation") or residual.get("promotion_recommendation") or ""
        ),
        "unsafe_skip_count": _int_or_none(residual.get("unsafe_skip_count")) or 0,
        "risky_family_escalation_rate": _float_or_none(
            residual.get("risky_family_escalation_rate")
        ),
        "redundant_check_suppression_rate": _float_or_none(
            residual.get("redundant_check_suppression_rate")
        ),
        "self_learning_claim_status": "controller_memory_only_blocked_until_ledger_consistency_1"
        if no_weight_update or ledger_rate != 1.0
        else "model_weight_learning_ready",
    }


def _architecture_boundary_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    energy = payloads["exp3158"]
    kan = payloads["exp3159"]
    hardware = payloads["exp3160"]
    return {
        "energy_sidecar_status": statuses.get("dot293:exp3158_ebcn_energy_sidecar", "missing"),
        "kan_status": statuses.get("dot293:exp3159_kan_monitor_expansion", "missing"),
        "hardware_status": statuses.get("dot293:exp3160_hardware_boundary", "missing"),
        "exact_labeled_row_count": _int_or_none(energy.get("exact_labeled_row_count")) or 0,
        "known_false_accept_rows_scored": _int_or_none(energy.get("known_false_accept_rows_scored"))
        or 0,
        "scalar_energy_auc": _float_or_none(energy.get("scalar_energy_auc")),
        "violation_localization_coverage": _float_or_none(
            energy.get("violation_localization_coverage")
        ),
        "live_integration_claim_allowed": energy.get("live_integration_claim_allowed") is True,
        "energy_residual_blocker_count": len(_as_list(energy.get("residual_blockers"))),
        "monitor_record_count": _int_or_none(kan.get("monitor_record_count")) or 0,
        "new_monitor_record_count": _int_or_none(kan.get("new_monitor_record_count")) or 0,
        "exact_row_coverage_count": _int_or_none(kan.get("exact_row_coverage_count")) or 0,
        "deployed_verifier_claim_allowed": kan.get("deployed_verifier_claim_allowed") is True,
        "kan_residual_blocker_count": len(_as_list(kan.get("residual_blockers"))),
        "kan_does_not_prove": _text_list(
            _as_mapping(kan.get("claim_boundary")).get("does_not_prove")
        ),
        "authenticated_speedup_claim_allowed": hardware.get("authenticated_speedup_claim_allowed")
        is True,
        "no_hardware_commands_run": hardware.get("no_hardware_commands_run") is True,
        "cuda_status": str(hardware.get("cuda_status") or ""),
        "kv260_status": str(hardware.get("kv260_status") or ""),
        "gatemate_status": str(hardware.get("gatemate_status") or ""),
        "polarfire_status": str(hardware.get("polarfire_status") or ""),
        "extropic_thrml_status": str(hardware.get("extropic_thrml_status") or ""),
        "kona_status": str(hardware.get("kona_status") or ""),
        "missing_operator_evidence_count": len(_as_list(hardware.get("missing_operator_evidence"))),
        "hardware_commands_run": _as_list(hardware.get("hardware_commands_run")),
    }


def _paper_readiness_implications(
    rows: list[Mapping[str, Any]],
    publication_blocker_count: int,
    payloads: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    blocked: list[str] = []
    if (
        statuses.get("dot293:exp3150_adversarial_corrigendum") != "clean"
        or statuses.get("dot293:exp3152_clean_live_rerun") != "clean"
    ):
        blocked.append("live_verifier_headline")
    if (
        statuses.get("dot293:exp3153_repair_gate_v2") != "clean"
        or statuses.get("dot293:exp3154_repair_ladder_v3") != "clean"
    ):
        blocked.append("repair_headline")
    if _fr11_summary(payloads, rows)["model_weight_learning_allowed"] is not True:
        blocked.append("fr11_model_weight_learning")
    architecture = _architecture_boundary_summary(payloads, rows)
    if architecture["live_integration_claim_allowed"] is not True:
        blocked.append("energy_sidecar_live_integration")
    if architecture["deployed_verifier_claim_allowed"] is not True:
        blocked.append("kan_deployed_verifier")
    if architecture["authenticated_speedup_claim_allowed"] is not True:
        blocked.append("hardware_speedup")
    return {
        "paper_ready": publication_blocker_count == 0 and not blocked,
        "publication_blocker_count": publication_blocker_count,
        "blocked_headline_claims": blocked,
    }


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
            "loaded_path": str(row["loaded_path"]),
            "role": str(row["role"]),
            "required": row.get("required") is True,
            "ready_field": str(row.get("ready_field") or ""),
            "present": row.get("present") is True,
            "primary_present": row.get("primary_present") is True,
            "readable_json_object": row.get("readable_json_object") is True,
            "alias_loaded": row.get("alias_loaded") is True,
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
    if matrix and matrix.get("matrix_v26_ready") is not True:
        violations.append("matrix v26 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone v292 authority is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v27 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to rows_total")
    if len(publication_blockers) != sum(
        count for status, count in status_counts.items() if status in PUBLICATION_BLOCKING_STATUSES
    ):
        violations.append("publication_blocker_count does not match row statuses")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot293_artifacts",
        "source": "matrix_v26_capstone_v292_and_dot293_artifacts",
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
    if artifact.get("matrix_v27_ready") is not True:
        return (
            "blocked_matrix_v27_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"invariant_violations={len(_as_list(artifact.get('invariant_violations')))}"
        )
    return (
        "complete: matrix_v27_ready=true; "
        f"rows_total={artifact.get('rows_total')}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v26={artifact.get('blocker_delta_from_v26')}; "
        f"inherited_adversarial_flag_count={artifact.get('inherited_adversarial_flag_count')}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
