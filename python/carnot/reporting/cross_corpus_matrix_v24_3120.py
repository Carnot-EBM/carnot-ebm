"""Build the Exp 3120 cross-corpus matrix v24 artifact.

Spec refs: REQ-REPORT-3120, SCENARIO-REPORT-3120.

Matrix v24 is an aggregation ledger over checked-in `.290` artifacts. It does
not run models, repairs, solvers, the conductor, or hardware. Its job is to
preserve the claim boundaries that the source artifacts already state.
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
MILESTONE = "2026.05.290"
SCHEMA = "carnot.cross_corpus_matrix.v24_290_claim_aggregation.v1"
ARTIFACT = "experiment_3120_cross_corpus_matrix_v24"
OUTPUT_REL_PATH = Path("results/experiment_3120_cross_corpus_matrix_v24.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3120_cross_corpus_matrix_v24.py"

MATRIX_V23_REL_PATH = Path("results/experiment_3107_cross_corpus_matrix_v23.json")
CAPSTONE_V289_REL_PATH = Path("results/experiment_3108_capstone_v289.json")
EXP3099_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
EXP3109_REL_PATH = Path("results/experiment_3109_archive_v289_activate_v290.json")
EXP3110_REL_PATH = Path("results/experiment_3110_sota_model_spec_cache_manifest_corrigendum_v1.json")
EXP3111_REL_PATH = Path("results/experiment_3111_certified_coherence_z3_mcs_feedback_v3.json")
EXP3112_REL_PATH = Path("results/experiment_3112_logic_regularized_verifier_pilot_v1.json")
EXP3113_REL_PATH = Path("results/experiment_3113_diagnostic_local_sota_verifier_calibration_v5.json")
EXP3114_REL_PATH = Path("results/experiment_3114_fragment_level_code_constraint_verification_pilot_v1.json")
EXP3115_REL_PATH = Path("results/experiment_3115_explicit_repair_gate_micro_panel_v4.json")
EXP3116_REL_PATH = Path("results/experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.json")
EXP3117_REL_PATH = Path("results/experiment_3117_ebt_arm_sidecar_score_correlation_boundary_v3.json")
EXP3118_REL_PATH = Path("results/experiment_3118_clut_sampler_backend_integration_boundary_v2.json")
EXP3119_REL_PATH = Path("results/experiment_3119_gatemate_ssqa_operator_evidence_ingestion_v4.json")

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

REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "capstone:v288_paper_readiness",
        "capstone:v289_paper_readiness",
        "Capstone .289 supersedes the prior paper-readiness authority row.",
    ),
    (
        "dot289:exp3099_local_sota_confidence_abstention_panel",
        "dot290:exp3110_sota_model_spec_cache_manifest",
        "Exp 3110 replaces the v23 model-spec metadata gap with explicit cache policy evidence.",
    ),
    (
        "dot289:exp3100_z3_oracle_feedback",
        "dot290:exp3111_certified_coherence_feedback",
        "Exp 3111 replaces blocked live-pair formal feedback with solver-certified feedback.",
    ),
    (
        "dot289:exp3101_local_sota_verifier_calibration_gate",
        "dot290:exp3113_diagnostic_verifier_calibration",
        "Exp 3113 replaces the v23 calibration gate with a diagnostic calibration artifact.",
    ),
    (
        "dot289:exp3102_structured_repair_micro_panel",
        "dot290:exp3115_explicit_repair_gate_micro_panel",
        "Exp 3115 replaces the missing repair micro-panel with an explicit boundary artifact.",
    ),
    (
        "dot289:exp3103_fr11_stress_promotion_boundary",
        "dot290:exp3116_fr11_curriculum_retention_guard",
        "Exp 3116 supersedes the FR-11 stress boundary with controller-only retention evidence.",
    ),
    (
        "dot289:exp3104_ebt_arm_sidecar_pipeline_boundary",
        "dot290:exp3117_ebt_arm_sidecar_score_correlation",
        "Exp 3117 replaces the sidecar projection row with score-correlation boundary evidence.",
    ),
    (
        "dot289:exp3105_clut_random_variate_sampler_microbench",
        "dot290:exp3118_clut_sampler_backend_integration",
        "Exp 3118 replaces the cLUT microbench row with CPU backend integration evidence.",
    ),
    (
        "dot289:exp3106_gatemate_operator_evidence",
        "dot290:exp3119_gatemate_operator_evidence",
        "Exp 3119 supersedes GateMate operator-evidence ingestion v3.",
    ),
    (
        "dot289:exp3106_ssqa_readback_evidence",
        "dot290:exp3119_ssqa_readback_evidence",
        "Exp 3119 supersedes SSQA readback gate evidence v3.",
    ),
)
REPLACED_ROW_IDS = {old for old, _new, _reason in REPLACEMENTS}


@dataclass(frozen=True)
class SourceSpec:
    """One source artifact that matrix v24 cites without mutating it."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    ready_field: str = ""


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3107", MATRIX_V23_REL_PATH, "matrix_v23_authority", True, "matrix_v23_ready"),
    SourceSpec("exp3108", CAPSTONE_V289_REL_PATH, "capstone_v289_authority", True, "capstone_ready"),
    SourceSpec("exp3109", EXP3109_REL_PATH, "archive_v289_activate_v290", False, "archive_v289_activate_v290_ready"),
    SourceSpec("exp3110", EXP3110_REL_PATH, "sota_model_spec_cache_manifest", False, "sota_model_manifest_ready"),
    SourceSpec("exp3111", EXP3111_REL_PATH, "certified_coherence_feedback", False, "certified_coherence_feedback_v3_ready"),
    SourceSpec("exp3112", EXP3112_REL_PATH, "logic_regularized_verifier_pilot", False, "logic_regularized_verifier_pilot_ready"),
    SourceSpec("exp3113", EXP3113_REL_PATH, "diagnostic_verifier_calibration", False, "diagnostic_verifier_calibration_v5_ready"),
    SourceSpec("exp3114", EXP3114_REL_PATH, "fragment_level_verification", False, "fragment_verification_pilot_ready"),
    SourceSpec("exp3115", EXP3115_REL_PATH, "explicit_repair_gate_micro_panel", False, "repair_micro_panel_v4_artifact_ready"),
    SourceSpec("exp3116", EXP3116_REL_PATH, "fr11_curriculum_retention_guard", False, "fr11_unsolvable_curriculum_ready"),
    SourceSpec("exp3117", EXP3117_REL_PATH, "ebt_arm_sidecar_score_correlation", False, "sidecar_score_correlation_boundary_v3_ready"),
    SourceSpec("exp3118", EXP3118_REL_PATH, "clut_sampler_backend_integration", False, "clut_backend_integration_boundary_v2_ready"),
    SourceSpec("exp3119", EXP3119_REL_PATH, "gatemate_ssqa_operator_evidence", False, "operator_evidence_ingestion_v4_ready"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return empty evidence on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source checksum for auditable matrix provenance."""

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
    """REQ-REPORT-3120: aggregate matrix v24 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3107"]
    capstone = payloads["exp3108"]
    rows = _carry_forward_rows(matrix) + _dot290_rows(payloads) if matrix and capstone else []
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    before_count = _before_count(matrix, capstone)
    missing_artifacts = _missing_artifacts(sources)
    headline_model_spec_gaps = _headline_model_spec_gaps(rows)
    required_source_errors = _required_source_errors(sources)
    contradictions = _verdict_status_contradictions(sources)
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
        "matrix_v24_ready": ready,
        "rows_total": len(rows),
        "status_counts": status_counts,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v23": len(publication_blockers) - before_count,
        "publication_blockers": publication_blockers,
        "rows": rows,
        "missing_artifacts": missing_artifacts,
        "headline_model_spec_gaps": headline_model_spec_gaps,
        "verifier_repair_status": _verifier_repair_status(payloads, rows),
        "fr11_status": _fr11_status(payloads["exp3116"]),
        "architecture_boundary_status": _architecture_boundary_status(payloads),
        "blocker_reconciliation_from_v23": _blocker_reconciliation(before_count, rows),
        "source_artifacts": _public_sources(sources),
        "source_checksums": {
            str(row["path"]): row.get("sha256") for row in _public_sources(sources)
        },
        "required_source_errors": required_source_errors,
        "honest_verdict_status_contradictions": contradictions,
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
    """Build and persist the Exp 3120 deliverable JSON."""

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
        original_status = row["status"]
        summary = _as_mapping(row.get("summary"))
        if row["row_id"] in REPLACED_ROW_IDS:
            row["status"] = "retired"
            summary.update(
                {
                    "previous_status": original_status,
                    "v24_status_rationale": "superseded_by_current_dot290_row",
                    "superseding_row_id": _replacement_row_id(row["row_id"]),
                }
            )
        else:
            summary.setdefault("v24_status_rationale", "carried_forward_from_matrix_v23")
        row["blocker_class"] = blocker_class(row["status"])
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v23")
        rows.append(row)
    return rows


def _dot290_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _capstone_v289_row(payloads["exp3108"]),
        _ready_row(
            "dot290:exp3109_archive_activation",
            payloads["exp3109"],
            EXP3109_REL_PATH,
            "archive_v289_activate_v290_ready",
            "archive_v289_activate_v290",
            "milestone_activation",
        ),
        _model_manifest_row(payloads["exp3110"]),
        _certified_coherence_row(payloads["exp3111"]),
        _logic_pilot_row(payloads["exp3112"]),
        _diagnostic_calibration_row(payloads["exp3113"]),
        _fragment_verification_row(payloads["exp3114"]),
        _repair_micro_panel_row(payloads["exp3115"]),
        _fr11_curriculum_row(payloads["exp3116"]),
        _ebt_arm_sidecar_row(payloads["exp3117"]),
        _clut_backend_row(payloads["exp3118"]),
        *_gatemate_ssqa_rows(payloads["exp3119"]),
    ]


def _capstone_v289_row(capstone: Mapping[str, Any]) -> JsonDict:
    status = "missing"
    if capstone:
        status = "clean" if capstone.get("paper_ready") is True else "bounded"
        if capstone.get("capstone_ready") is not True:
            status = "blocked"
    return _row(
        row_id="capstone:v289_paper_readiness",
        status=status,
        source_artifact=CAPSTONE_V289_REL_PATH.as_posix(),
        source_field="paper_ready",
        evidence_class="capstone_v289_authority",
        claim_scope="paper_readiness",
        summary={
            "capstone_ready": capstone.get("capstone_ready") is True,
            "paper_ready": capstone.get("paper_ready") is True,
            "publication_blocker_count": _int_or_none(capstone.get("publication_blocker_count")),
            "honest_verdict": str(capstone.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _ready_row(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    ready_field: str,
    evidence_class: str,
    claim_scope: str,
) -> JsonDict:
    return _row(
        row_id=row_id,
        status=_status_for_ready(bool(payload), payload, ready_field),
        source_artifact=source_path.as_posix(),
        source_field=ready_field,
        evidence_class=evidence_class,
        claim_scope=claim_scope,
        summary={
            "ready": payload.get(ready_field) is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _model_manifest_row(payload: Mapping[str, Any]) -> JsonDict:
    present = _text_list(payload.get("present_model_ids"))
    missing = _text_list(payload.get("missing_model_ids"))
    selected = _text_list(payload.get("selected_headline_model_ids"))
    cached_pair = payload.get("cached_sota_pair_available") is True
    ready = payload.get("sota_model_manifest_ready") is True
    gaps = []
    if payload and ready and (missing or not cached_pair):
        gaps.append(
            {
                "row_id": "dot290:exp3110_sota_model_spec_cache_manifest",
                "source_artifact": EXP3110_REL_PATH.as_posix(),
                "missing_model_ids": missing,
                "present_model_ids": present,
                "reason": "mandated headline model cache coverage incomplete; cached SOTA pair unavailable",
            }
        )
    if not payload:
        status = "missing"
    elif not ready:
        status = "blocked"
    elif not selected or payload.get("headline_claim_allowed") is not True:
        status = "model_spec_gap"
    elif gaps:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot290:exp3110_sota_model_spec_cache_manifest",
        status=status,
        source_artifact=EXP3110_REL_PATH.as_posix(),
        source_field="sota_model_manifest_ready",
        evidence_class="sota_model_spec_cache_manifest",
        claim_scope="local_sota_model_cache_policy",
        summary={
            "sota_model_manifest_ready": ready,
            "present_model_ids": present,
            "missing_model_ids": missing,
            "cached_sota_pair_available": cached_pair,
            "selected_headline_model_ids": selected,
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "model_spec_gaps": gaps,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _certified_coherence_row(payload: Mapping[str, Any]) -> JsonDict:
    return _ready_row(
        "dot290:exp3111_certified_coherence_feedback",
        payload,
        EXP3111_REL_PATH,
        "certified_coherence_feedback_v3_ready",
        "certified_coherence_feedback",
        "solver_certified_feedback",
    )


def _logic_pilot_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("logic_regularized_verifier_pilot_ready") is not True:
        status = "blocked"
    elif payload.get("promotion_claim_made") is True:
        status = "bounded"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot290:exp3112_logic_regularized_verifier_pilot",
        status=status,
        source_artifact=EXP3112_REL_PATH.as_posix(),
        source_field="logic_regularized_verifier_pilot_ready",
        evidence_class="logic_regularized_verifier_pilot",
        claim_scope="verifier_diagnostic_no_promotion",
        summary={
            "logic_regularized_verifier_pilot_ready": payload.get(
                "logic_regularized_verifier_pilot_ready"
            )
            is True,
            "exact_ground_truth_count": _int_or_none(payload.get("exact_ground_truth_count")),
            "verifier_recall_delta": _float_or_none(payload.get("verifier_recall_delta")),
            "false_positive_delta": _float_or_none(payload.get("false_positive_delta")),
            "promotion_claim_made": payload.get("promotion_claim_made") is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "status_rationale": "diagnostic_pilot_no_promotion_claim",
        },
        row_origin="milestone_290",
    )


def _diagnostic_calibration_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("diagnostic_verifier_calibration_v5_ready") is True:
        status = "diagnostic_only"
    else:
        status = "blocked"
    return _row(
        row_id="dot290:exp3113_diagnostic_verifier_calibration",
        status=status,
        source_artifact=EXP3113_REL_PATH.as_posix(),
        source_field="diagnostic_verifier_calibration_v5_ready",
        evidence_class="diagnostic_verifier_calibration",
        claim_scope="repair_gate_diagnostic",
        summary={
            "diagnostic_verifier_calibration_v5_ready": payload.get(
                "diagnostic_verifier_calibration_v5_ready"
            )
            is True,
            "repair_gate_state": str(payload.get("repair_gate_state") or ""),
            "verifier_gain_delta_with_certified_coherence": _float_or_none(
                payload.get("verifier_gain_delta_with_certified_coherence")
            ),
            "false_accept_rate": _float_or_none(payload.get("false_accept_rate")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _fragment_verification_row(payload: Mapping[str, Any]) -> JsonDict:
    return _row(
        row_id="dot290:exp3114_fragment_level_verification",
        status=_status_for_ready(bool(payload), payload, "fragment_verification_pilot_ready"),
        source_artifact=EXP3114_REL_PATH.as_posix(),
        source_field="fragment_verification_pilot_ready",
        evidence_class="fragment_level_code_constraint_verification",
        claim_scope="repair_target_localization",
        summary={
            "fragment_verification_pilot_ready": payload.get("fragment_verification_pilot_ready")
            is True,
            "failing_fragment_count": _int_or_none(payload.get("failing_fragment_count")),
            "repair_target_manifest_path": str(payload.get("repair_target_manifest_path") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _repair_micro_panel_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("repair_micro_panel_v4_artifact_ready") is not True:
        status = "blocked"
    elif (
        payload.get("repair_unblocked") is True
        and payload.get("repair_run_executed") is True
        and _float_or_none(payload.get("repair_success_delta")) not in (None, 0.0)
        and _float_or_none(payload.get("intent_preservation_rate")) not in (None, 0.0)
    ):
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="dot290:exp3115_explicit_repair_gate_micro_panel",
        status=status,
        source_artifact=EXP3115_REL_PATH.as_posix(),
        source_field="repair_micro_panel_v4_artifact_ready",
        evidence_class="explicit_repair_gate_micro_panel",
        claim_scope="repair_live_rerun",
        summary={
            "repair_micro_panel_v4_artifact_ready": payload.get(
                "repair_micro_panel_v4_artifact_ready"
            )
            is True,
            "repair_unblocked": payload.get("repair_unblocked") is True,
            "repair_run_executed": payload.get("repair_run_executed") is True,
            "repair_success_delta": _float_or_none(payload.get("repair_success_delta")),
            "false_repair_accept_rate": _float_or_none(payload.get("false_repair_accept_rate")),
            "intent_preservation_rate": _float_or_none(payload.get("intent_preservation_rate")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "status_rationale": "artifact_exists_but_repair_claim_remains_bounded",
        },
        row_origin="milestone_290",
    )


def _fr11_curriculum_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("fr11_unsolvable_curriculum_ready") is not True:
        status = "blocked"
    elif payload.get("controller_only") is True or payload.get("no_weight_update_claim") is True:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot290:exp3116_fr11_curriculum_retention_guard",
        status=status,
        source_artifact=EXP3116_REL_PATH.as_posix(),
        source_field="fr11_unsolvable_curriculum_ready",
        evidence_class="fr11_unsolvable_curriculum_retention_guard",
        claim_scope="controller_only_self_learning_boundary",
        summary={
            "fr11_unsolvable_curriculum_ready": payload.get("fr11_unsolvable_curriculum_ready")
            is True,
            "controller_only": payload.get("controller_only") is True,
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "promotion_decision": str(payload.get("promotion_decision") or ""),
            "soundness_mistakes": _int_or_none(payload.get("soundness_mistakes")),
            "completeness_mistakes": _int_or_none(payload.get("completeness_mistakes")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _ebt_arm_sidecar_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("sidecar_score_correlation_boundary_v3_ready") is True
    bounded = (
        payload.get("no_live_model_integration_claim") is True
        or payload.get("no_weight_update_claim") is True
        or payload.get("no_speedup_claim") is True
    )
    status = "missing" if not payload else "projection_only" if ready and bounded else "clean" if ready else "blocked"
    return _row(
        row_id="dot290:exp3117_ebt_arm_sidecar_score_correlation",
        status=status,
        source_artifact=EXP3117_REL_PATH.as_posix(),
        source_field="sidecar_score_correlation_boundary_v3_ready",
        evidence_class="ebt_arm_sidecar_score_correlation_boundary",
        claim_scope="future_adapter_context",
        summary={
            "sidecar_score_correlation_boundary_v3_ready": ready,
            "no_live_model_integration_claim": payload.get("no_live_model_integration_claim")
            is True,
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "no_speedup_claim": payload.get("no_speedup_claim") is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _clut_backend_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("clut_backend_integration_boundary_v2_ready") is True or str(
        payload.get("status") or ""
    ).lower() == "success"
    cpu_only = payload.get("hardware_claim_made") is False and not _as_list(
        payload.get("hardware_commands_run")
    )
    status = "missing" if not payload else "bounded" if ready and cpu_only else "clean" if ready else "blocked"
    return _row(
        row_id="dot290:exp3118_clut_sampler_backend_integration",
        status=status,
        source_artifact=EXP3118_REL_PATH.as_posix(),
        source_field="clut_backend_integration_boundary_v2_ready",
        evidence_class="clut_sampler_backend_integration_boundary",
        claim_scope="cpu_backend_no_hardware_speedup",
        summary={
            "clut_backend_integration_boundary_v2_ready": ready,
            "default_backend_preserved": payload.get("default_backend_preserved") is True,
            "distribution_checks_passed": payload.get("distribution_checks_passed") is True,
            "hardware_claim_made": payload.get("hardware_claim_made") is True,
            "hardware_command_count": len(_as_list(payload.get("hardware_commands_run"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_290",
    )


def _gatemate_ssqa_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    ready = payload.get("operator_evidence_ingestion_v4_ready") is True
    gatemate_allowed = payload.get("gatemate_rerun_allowed") is True
    ssqa_allowed = payload.get("ssqa_readback_allowed") is True
    return [
        _row(
            row_id="dot290:exp3119_gatemate_operator_evidence",
            status="missing" if not payload else "clean" if gatemate_allowed else "blocked",
            source_artifact=EXP3119_REL_PATH.as_posix(),
            source_field="gatemate_rerun_allowed",
            evidence_class="gatemate_ssqa_operator_evidence_ingestion",
            claim_scope="hardware_rerun_gate",
            summary={
                "operator_evidence_ingestion_v4_ready": ready,
                "gatemate_rerun_allowed": gatemate_allowed,
                "missing_operator_action_count": len(_as_list(payload.get("missing_operator_actions"))),
                "hardware_command_count": len(_as_list(payload.get("hardware_commands_run"))),
                "speedup_claim_made": payload.get("speedup_claim_made") is True,
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            },
            row_origin="milestone_290",
        ),
        _row(
            row_id="dot290:exp3119_ssqa_readback_evidence",
            status="missing" if not payload else "clean" if ssqa_allowed else "gated_skipped" if ready else "blocked",
            source_artifact=EXP3119_REL_PATH.as_posix(),
            source_field="ssqa_readback_allowed",
            evidence_class="gatemate_ssqa_operator_evidence_ingestion",
            claim_scope="host_visible_readback_gate",
            summary={
                "operator_evidence_ingestion_v4_ready": ready,
                "ssqa_readback_allowed": ssqa_allowed,
                "hardware_readback_attempted": payload.get("hardware_readback_attempted") is True,
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            },
            row_origin="milestone_290",
        ),
    ]


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
        "row_origin": str(row.get("row_origin") or "matrix_v23"),
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
    row_origin: str,
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
        "row_origin": row_origin,
    }


def _status_for_ready(present: bool, payload: Mapping[str, Any], ready_field: str) -> str:
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


def _before_count(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> int:
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
            else "missing_or_malformed_dot290_artifact",
        }
        for row in sources
        if row.get("readable_json_object") is not True
    ]


def _headline_model_spec_gaps(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    gaps: list[JsonDict] = []
    for row in rows:
        if normal_status(str(row.get("status") or "")) == "retired":
            continue
        summary = _as_mapping(row.get("summary"))
        gaps.extend(
            dict(gap)
            for gap in _as_list(summary.get("model_spec_gaps"))
            if isinstance(gap, Mapping)
        )
    return gaps


def _verdict_status_contradictions(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    contradictions: list[JsonDict] = []
    for source in sources:
        ready_field = str(source.get("ready_field") or "")
        payload = _as_mapping(source.get("payload"))
        if not ready_field or not payload:
            continue
        ready_value = payload.get(ready_field)
        verdict = str(payload.get("honest_verdict") or "")
        if ready_value is False and _has_success_verdict(verdict):
            contradictions.append(
                {
                    "experiment_id": str(source["experiment_id"]),
                    "path": str(source["path"]),
                    "ready_field": ready_field,
                    "ready_value": False,
                    "honest_verdict": verdict,
                    "reason": "success verdict contradicts false readiness field",
                }
            )
        elif ready_value is True and _has_blocked_verdict(verdict):
            contradictions.append(
                {
                    "experiment_id": str(source["experiment_id"]),
                    "path": str(source["path"]),
                    "ready_field": ready_field,
                    "ready_value": True,
                    "honest_verdict": verdict,
                    "reason": "blocked verdict contradicts true readiness field",
                }
            )
    return contradictions


def _verifier_repair_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = {str(row.get("row_id") or ""): normal_status(str(row.get("status") or "")) for row in rows}
    repair = payloads["exp3115"]
    repair_status = statuses.get("dot290:exp3115_explicit_repair_gate_micro_panel", "missing")
    if repair_status == "clean":
        claim_status = "clean_repair_claim"
    elif repair_status == "bounded":
        claim_status = "bounded_no_positive_repair_delta"
    elif repair_status == "missing":
        claim_status = "missing_repair_artifact"
    else:
        claim_status = "blocked_repair_claim"
    blocker_rows = [
        row["row_id"]
        for row in _publication_blockers(rows)
        if str(row.get("claim_scope") or "") in {"repair_live_rerun", "local_sota_model_cache_policy"}
    ]
    return {
        "model_manifest_status": statuses.get("dot290:exp3110_sota_model_spec_cache_manifest", "missing"),
        "certified_coherence_status": statuses.get("dot290:exp3111_certified_coherence_feedback", "missing"),
        "logic_pilot_status": statuses.get("dot290:exp3112_logic_regularized_verifier_pilot", "missing"),
        "diagnostic_calibration_status": statuses.get("dot290:exp3113_diagnostic_verifier_calibration", "missing"),
        "fragment_verification_status": statuses.get("dot290:exp3114_fragment_level_verification", "missing"),
        "repair_micro_panel_status": repair_status,
        "repair_claim_status": claim_status,
        "repair_unblocked": repair.get("repair_unblocked") is True,
        "repair_run_executed": repair.get("repair_run_executed") is True,
        "repair_success_delta": _float_or_none(repair.get("repair_success_delta")) or 0.0,
        "false_repair_accept_rate": _float_or_none(repair.get("false_repair_accept_rate")) or 0.0,
        "intent_preservation_rate": _float_or_none(repair.get("intent_preservation_rate")) or 0.0,
        "publication_blocker_row_ids": blocker_rows,
    }


def _fr11_status(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing_fr11_artifact"
    elif payload.get("fr11_unsolvable_curriculum_ready") is not True:
        status = "blocked_fr11_precondition"
    elif payload.get("controller_only") is True or payload.get("no_weight_update_claim") is True:
        status = "bounded_controller_only_no_weight_update_claim"
    else:
        status = "clean_weight_update_claim_supported"
    return {
        "status": status,
        "fr11_unsolvable_curriculum_ready": payload.get("fr11_unsolvable_curriculum_ready") is True,
        "controller_only": payload.get("controller_only") is True,
        "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
        "promotion_decision": str(payload.get("promotion_decision") or ""),
        "soundness_mistakes": _int_or_none(payload.get("soundness_mistakes")),
        "completeness_mistakes": _int_or_none(payload.get("completeness_mistakes")),
    }


def _architecture_boundary_status(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp3117 = payloads["exp3117"]
    exp3118 = payloads["exp3118"]
    exp3119 = payloads["exp3119"]
    return {
        "ebt_arm_status": "projection_only_no_live_model_integration"
        if exp3117.get("no_live_model_integration_claim") is True
        else "clean_live_integration_claimed",
        "clut_status": "bounded_cpu_only_no_hardware_speedup"
        if exp3118.get("hardware_claim_made") is False
        else "clean_hardware_claimed",
        "gatemate_status": "clean_operator_evidence_complete"
        if exp3119.get("gatemate_rerun_allowed") is True
        else "blocked_operator_evidence_incomplete",
        "ssqa_status": "clean_host_visible_readback_complete"
        if exp3119.get("ssqa_readback_allowed") is True
        else "gated_skipped_host_visible_readback_missing",
    }


def _blocker_reconciliation(before_count: int, rows: list[Mapping[str, Any]]) -> JsonDict:
    after_count = len(_publication_blockers(rows))
    row_by_id = {str(row.get("row_id") or ""): row for row in rows}
    retired = []
    added = []
    neutral = []
    for old, new, reason in REPLACEMENTS:
        old_row = _as_mapping(row_by_id.get(old))
        new_row = _as_mapping(row_by_id.get(new))
        old_status = normal_status(
            str(_as_mapping(old_row.get("summary")).get("previous_status") or old_row.get("status") or "missing")
        )
        new_status = normal_status(str(new_row.get("status") or "missing"))
        if old_status in PUBLICATION_BLOCKING_STATUSES and old_row.get("status") == "retired":
            retired.append(old)
        if new_status in PUBLICATION_BLOCKING_STATUSES:
            added.append(new)
        neutral.append(
            {
                "old_row_id": old,
                "new_row_id": new,
                "old_status": old_status,
                "new_status": new_status,
                "reason": reason,
            }
        )
    return {
        "publication_blocker_count_before": before_count,
        "publication_blocker_count_after": after_count,
        "blocker_delta_from_v23": after_count - before_count,
        "retired_blocking_rows": retired,
        "added_blocking_rows": added,
        "neutral_replacements": neutral,
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


def _required_source_errors(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "path": str(row["path"]),
            "reason": "missing_or_malformed_required_artifact",
        }
        for row in sources
        if row.get("required") is True and row.get("readable_json_object") is not True
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
    if matrix and matrix.get("matrix_v23_ready") is not True:
        violations.append("matrix v23 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone .289 authority is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v24 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to rows_total")
    observed_blockers = sum(
        1
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    )
    if observed_blockers != len(publication_blockers):
        violations.append("publication_blocker_count does not match row statuses")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_artifacts",
        "source": "matrix_v23_capstone_v289_and_dot290_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "executes_solvers": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("matrix_v24_ready") is not True:
        return (
            "blocked_matrix_v24_preconditions: "
            f"invariant_violations={_as_list(artifact.get('invariant_violations'))}"
        )
    return (
        "complete: "
        "matrix_v24_ready=true; "
        f"rows_total={artifact['rows_total']}; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"blocker_delta_from_v23={artifact['blocker_delta_from_v23']}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}"
    )


def _replacement_row_id(row_id: str) -> str:
    for old, new, _reason in REPLACEMENTS:
        if old == row_id:
            return new
    return ""


def normal_status(status: str) -> str:
    """Normalize legacy labels into the v24 status vocabulary."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    if normalized == "diagnostic":
        return "diagnostic_only"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map each status to the matrix blocker class."""

    return {
        "clean": "none",
        "flagged": "adversarial_or_methodology_flag",
        "bounded": "bounded_claim",
        "blocked": "required_blocker",
        "gated_skipped": "structured_gate_skip",
        "missing": "missing_artifact",
        "retired": "retired_claim",
        "projection_only": "projection_only",
        "diagnostic_only": "diagnostic_only",
        "model_spec_gap": "model_spec_gap",
    }[normal_status(status)]


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if str(item)]


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_success_verdict(verdict: str) -> bool:
    lowered = verdict.lower()
    return lowered.startswith(
        ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped_")
    )


def _has_blocked_verdict(verdict: str) -> bool:
    lowered = verdict.lower()
    return lowered.startswith(("blocked", "failed", "failure"))


__all__ = [
    "CAPSTONE_V289_REL_PATH",
    "EXP3099_REL_PATH",
    "EXP3109_REL_PATH",
    "EXP3110_REL_PATH",
    "EXP3111_REL_PATH",
    "EXP3112_REL_PATH",
    "EXP3113_REL_PATH",
    "EXP3114_REL_PATH",
    "EXP3115_REL_PATH",
    "EXP3116_REL_PATH",
    "EXP3117_REL_PATH",
    "EXP3118_REL_PATH",
    "EXP3119_REL_PATH",
    "MATRIX_V23_REL_PATH",
    "OUTPUT_REL_PATH",
    "PUBLICATION_BLOCKING_STATUSES",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "STATUSES",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
