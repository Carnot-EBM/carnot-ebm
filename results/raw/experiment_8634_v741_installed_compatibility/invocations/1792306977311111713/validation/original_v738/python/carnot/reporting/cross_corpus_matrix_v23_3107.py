"""Build the Exp 3107 cross-corpus matrix v23 artifact.

Spec refs: REQ-REPORT-3107, SCENARIO-REPORT-3107.

Matrix v23 is an authority ledger over checked-in artifacts. It does not run
models, solvers, repair loops, hardware tools, or the conductor; it only
preserves the difference between evidence that is complete, blocked, bounded,
gated, missing, retired, projection-only, diagnostic-only, or still missing
the mandated live-headline model specifications.
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
MILESTONE = "2026.05.289"
NEXT_CAPSTONE_EXPERIMENT = "exp3108"
SCHEMA = "carnot.cross_corpus_matrix.v23_289_claim_aggregation.v1"
ARTIFACT = "experiment_3107_cross_corpus_matrix_v23"
OUTPUT_REL_PATH = Path("results/experiment_3107_cross_corpus_matrix_v23.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3107_cross_corpus_matrix_v23.py"

MATRIX_V22_REL_PATH = Path("results/experiment_3093_cross_corpus_matrix_v22.json")
CAPSTONE_V288_REL_PATH = Path("results/experiment_3094_capstone_v288.json")
EXP3095_REL_PATH = Path("results/experiment_3095_archive_v288_activate_v289.json")
EXP3096_REL_PATH = Path(
    "results/experiment_3096_publication_blocker_triage_and_retirement_ledger_v2.json"
)
EXP3097_REL_PATH = Path("results/experiment_3097_exact_fixture_eval_protocol_audit_v1.json")
EXP3098_REL_PATH = Path("results/experiment_3098_maxsat_abstention_routing_policy_v1.json")
EXP3099_REL_PATH = Path("results/experiment_3099_local_sota_confidence_abstention_panel_v3.json")
EXP3100_REL_PATH = Path("results/experiment_3100_z3_oracle_feedback_v2.json")
EXP3101_REL_PATH = Path("results/experiment_3101_local_sota_verifier_calibration_v4.json")
EXP3102_REL_PATH = Path("results/experiment_3102_gated_structured_repair_micro_panel_v3.json")
EXP3103_REL_PATH = Path("results/experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.json")
EXP3104_REL_PATH = Path("results/experiment_3104_ebt_arm_sidecar_pipeline_boundary_v2.json")
EXP3105_REL_PATH = Path("results/experiment_3105_clut_random_variate_sampler_microbench_v1.json")
EXP3106_REL_PATH = Path("results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json")

LEGACY_STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "retired",
    "projection_only",
)
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
        "capstone:v287_paper_readiness",
        "capstone:v288_paper_readiness",
        "capstone .288 supersedes the prior capstone paper-readiness row.",
    ),
    (
        "dot288:exp3085_icalm_task_abstention_sota_panel",
        "dot289:exp3099_local_sota_confidence_abstention_panel",
        "Exp 3099 supersedes the .288 iCalm abstention panel row.",
    ),
    (
        "dot288:exp3086_dafny_z3_formal_feedback_pilot",
        "dot289:exp3100_z3_oracle_feedback",
        "Exp 3100 supersedes the .288 formal-feedback pilot row.",
    ),
    (
        "dot288:exp3087_local_sota_verifier_calibration_gate",
        "dot289:exp3101_local_sota_verifier_calibration_gate",
        "Exp 3101 supersedes the .288 verifier calibration gate row.",
    ),
    (
        "dot288:exp3089_xgrammar_sota_repair_micro_panel",
        "dot289:exp3102_structured_repair_micro_panel",
        "Exp 3102 supersedes the missing .288 repair micro-panel request.",
    ),
    (
        "dot288:exp3091_ebt_arm_sidecar_adapter_schema_prototype",
        "dot289:exp3104_ebt_arm_sidecar_pipeline_boundary",
        "Exp 3104 supersedes the .288 EBT/ARM sidecar projection row.",
    ),
    (
        "dot288:exp3092_gatemate_operator_evidence",
        "dot289:exp3106_gatemate_operator_evidence",
        "Exp 3106 supersedes the .288 GateMate operator-evidence row.",
    ),
    (
        "dot288:exp3092_ssqa_readback_evidence",
        "dot289:exp3106_ssqa_readback_evidence",
        "Exp 3106 supersedes the .288 SSQA readback-evidence row.",
    ),
)
REPLACED_ROW_IDS = {old for old, _, _ in REPLACEMENTS}


@dataclass(frozen=True)
class SourceSpec:
    """One checked-in source artifact that v23 cites without mutating it."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    source_type: str = "json"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3093", MATRIX_V22_REL_PATH, "matrix_v22_authority", required=True),
    SourceSpec("exp3094", CAPSTONE_V288_REL_PATH, "capstone_v288_authority", required=True),
    SourceSpec("exp3096", EXP3096_REL_PATH, "blocker_triage_ledger_authority", required=True),
    SourceSpec("exp3095", EXP3095_REL_PATH, "archive_v289_activation"),
    SourceSpec("exp3097", EXP3097_REL_PATH, "exact_fixture_eval_protocol_audit"),
    SourceSpec("exp3098", EXP3098_REL_PATH, "maxsat_abstention_routing_policy"),
    SourceSpec("exp3099", EXP3099_REL_PATH, "local_sota_confidence_abstention_panel"),
    SourceSpec("exp3100", EXP3100_REL_PATH, "z3_oracle_feedback"),
    SourceSpec("exp3101", EXP3101_REL_PATH, "local_sota_verifier_calibration_gate"),
    SourceSpec("exp3102", EXP3102_REL_PATH, "gated_structured_repair_micro_panel"),
    SourceSpec("exp3103", EXP3103_REL_PATH, "fr11_resyn_kancl_stress_promotion_boundary"),
    SourceSpec("exp3104", EXP3104_REL_PATH, "ebt_arm_sidecar_pipeline_boundary"),
    SourceSpec("exp3105", EXP3105_REL_PATH, "clut_random_variate_sampler_microbench"),
    SourceSpec("exp3106", EXP3106_REL_PATH, "gatemate_ssqa_operator_evidence_ingestion"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return empty evidence on missing or malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a source checksum for matrix traceability."""

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
    """REQ-REPORT-3107: aggregate matrix v23 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3093"]
    capstone = payloads["exp3094"]
    ledger = payloads["exp3096"]
    rows = _carry_forward_rows(matrix) + _dot289_rows(payloads) if matrix and capstone and ledger else []
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    before_count = _before_count(matrix, capstone, ledger)
    missing_artifacts = _missing_artifacts(sources)
    headline_model_spec_gaps = _headline_model_spec_gaps(rows)
    required_source_errors = _required_source_errors(sources)
    reconciliation = _blocker_reconciliation(
        before_count=before_count,
        after_count=len(publication_blockers),
        rows=rows,
    )
    invariant_violations = _invariant_violations(
        matrix,
        capstone,
        ledger,
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
        "matrix_v23_ready": ready,
        "rows_total": len(rows),
        "status_counts": status_counts,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v22": len(publication_blockers) - before_count,
        "publication_blockers": publication_blockers,
        "rows": rows,
        "missing_artifacts": missing_artifacts,
        "headline_model_spec_gaps": headline_model_spec_gaps,
        "blocker_reconciliation_from_exp3096": reconciliation,
        "capstone_input_artifacts": _capstone_input_artifacts(),
        "source_artifacts": _public_sources(sources),
        "source_checksums": {
            str(row["path"]): row.get("sha256") for row in _public_sources(sources)
        },
        "required_source_errors": required_source_errors,
        "invariant_violations": invariant_violations,
        "inference_substrate": _inference_substrate(),
        "next_capstone_experiment": NEXT_CAPSTONE_EXPERIMENT,
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
    """Build and persist the Exp 3107 matrix artifact."""

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
    payload = read_json_object(path) if spec.source_type == "json" else {}
    readable_json = bool(payload) if spec.source_type == "json" else path.is_file()
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "source_type": spec.source_type,
        "present": path.is_file(),
        "readable_json_object": readable_json,
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
                    "v23_status_rationale": "superseded_by_current_dot289_row",
                    "superseding_row_id": _replacement_row_id(row["row_id"]),
                }
            )
        else:
            summary.setdefault("v23_status_rationale", "carried_forward_from_matrix_v22")
        row["blocker_class"] = blocker_class(row["status"])
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v22")
        rows.append(row)
    return rows


def _dot289_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _capstone_v288_row(payloads["exp3094"]),
        _ready_row(
            "dot289:exp3095_archive_activation",
            payloads["exp3095"],
            EXP3095_REL_PATH,
            "archive_v288_activate_v289_ready",
            "archive_v288_activate_v289",
            "milestone_activation",
        ),
        _diagnostic_row(
            "dot289:exp3096_blocker_triage_ledger",
            payloads["exp3096"],
            EXP3096_REL_PATH,
            "blocker_triage_ready",
            "publication_blocker_triage_ledger",
            "matrix_v23_planning",
        ),
        _ready_row(
            "dot289:exp3097_exact_fixture_eval_protocol_audit",
            payloads["exp3097"],
            EXP3097_REL_PATH,
            "eval_protocol_ready",
            "exact_fixture_eval_protocol_audit",
            "exact_fixture_protocol",
        ),
        _diagnostic_row(
            "dot289:exp3098_maxsat_abstention_routing_policy",
            payloads["exp3098"],
            EXP3098_REL_PATH,
            "maxsat_policy_ready",
            "maxsat_abstention_routing_policy",
            "policy_design_diagnostic",
        ),
        _local_sota_panel_row(payloads["exp3099"]),
        _formal_feedback_row(payloads["exp3100"]),
        _gate_record_row(payloads["exp3101"]),
        _repair_panel_row(payloads["exp3102"]),
        _fr11_stress_row(payloads["exp3103"]),
        _ebt_arm_row(payloads["exp3104"]),
        _diagnostic_row(
            "dot289:exp3105_clut_random_variate_sampler_microbench",
            payloads["exp3105"],
            EXP3105_REL_PATH,
            "clut_microbench_ready",
            "clut_random_variate_sampler_microbench",
            "cpu_microbench_diagnostic",
        ),
        *_gatemate_ssqa_rows(payloads["exp3106"]),
    ]


def _capstone_v288_row(capstone: Mapping[str, Any]) -> JsonDict:
    status = "missing"
    if capstone:
        status = "clean" if capstone.get("paper_ready") is True else "bounded"
        if capstone.get("capstone_ready") is not True:
            status = "blocked"
    return _row(
        row_id="capstone:v288_paper_readiness",
        status=status,
        source_artifact=CAPSTONE_V288_REL_PATH.as_posix(),
        source_field="paper_ready",
        evidence_class="capstone_v288_authority",
        claim_scope="paper_readiness",
        summary={
            "capstone_ready": capstone.get("capstone_ready") is True,
            "paper_ready": capstone.get("paper_ready") is True,
            "publication_blocker_count": _int_or_none(capstone.get("publication_blocker_count")),
            "honest_verdict": str(capstone.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _ready_row(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    ready_field: str,
    evidence_class: str,
    claim_scope: str,
) -> JsonDict:
    status = "missing" if not payload else "clean" if payload.get(ready_field) is True else "blocked"
    return _row(
        row_id=row_id,
        status=status,
        source_artifact=source_path.as_posix(),
        source_field=ready_field,
        evidence_class=evidence_class,
        claim_scope=claim_scope,
        summary={
            "ready": payload.get(ready_field) is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _diagnostic_row(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    ready_field: str,
    evidence_class: str,
    claim_scope: str,
) -> JsonDict:
    ready = payload.get(ready_field) is True or str(payload.get("status") or "") == "success"
    status = "missing" if not payload else "diagnostic_only" if ready else "blocked"
    return _row(
        row_id=row_id,
        status=status,
        source_artifact=source_path.as_posix(),
        source_field=ready_field,
        evidence_class=evidence_class,
        claim_scope=claim_scope,
        summary={
            "ready": ready,
            "diagnostic_only": status == "diagnostic_only",
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "status_rationale": "diagnostic_or_policy_only_not_headline_evidence",
        },
        row_origin="milestone_289",
    )


def _local_sota_panel_row(payload: Mapping[str, Any]) -> JsonDict:
    status, gaps = _live_llm_status_and_gaps(
        "dot289:exp3099_local_sota_confidence_abstention_panel",
        payload,
        EXP3099_REL_PATH,
        "abstention_panel_v3_ready",
    )
    return _row(
        row_id="dot289:exp3099_local_sota_confidence_abstention_panel",
        status=status,
        source_artifact=EXP3099_REL_PATH.as_posix(),
        source_field="abstention_panel_v3_ready",
        evidence_class="local_sota_confidence_abstention_panel",
        claim_scope="local_sota_solution_verifier_gain",
        summary={
            "ready": payload.get("abstention_panel_v3_ready") is True,
            "exact_ground_truth_count": _int_or_none(payload.get("exact_ground_truth_count")),
            "minimum_live_eval_count": _int_or_none(payload.get("minimum_live_eval_count")),
            "abstention_precision": _float_or_none(payload.get("abstention_precision")),
            "false_accept_rate": _float_or_none(payload.get("false_accept_rate")),
            "model_spec_gate_passed": not _has_model_spec_gap(gaps),
            "model_spec_gap_count": sum(1 for gap in gaps if _is_model_spec_gap(gap)),
            "model_spec_gaps": gaps,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _formal_feedback_row(payload: Mapping[str, Any]) -> JsonDict:
    status = "missing" if not payload else "clean" if payload.get("formal_feedback_v2_ready") is True else "blocked"
    return _row(
        row_id="dot289:exp3100_z3_oracle_feedback",
        status=status,
        source_artifact=EXP3100_REL_PATH.as_posix(),
        source_field="formal_feedback_v2_ready",
        evidence_class="z3_oracle_feedback",
        claim_scope="solver_grounded_repair_feedback",
        summary={
            "formal_feedback_v2_ready": payload.get("formal_feedback_v2_ready") is True,
            "headline_blocked_reason": str(payload.get("headline_blocked_reason") or ""),
            "guided_success_count": _int_or_none(payload.get("guided_success_count")),
            "solver_only_success_count": _int_or_none(payload.get("solver_only_success_count")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _gate_record_row(payload: Mapping[str, Any]) -> JsonDict:
    return _row(
        row_id="dot289:exp3101_local_sota_verifier_calibration_gate",
        status=_gate_status(payload),
        source_artifact=EXP3101_REL_PATH.as_posix(),
        source_field="gate_check_summary",
        evidence_class="local_sota_verifier_calibration_gate",
        claim_scope="verifier_gain_recovery_gate",
        summary={
            "status": str(payload.get("status") or ""),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _repair_panel_row(payload: Mapping[str, Any]) -> JsonDict:
    return _row(
        row_id="dot289:exp3102_structured_repair_micro_panel",
        status=_repair_panel_status(payload),
        source_artifact=EXP3102_REL_PATH.as_posix(),
        source_field="gated_structured_repair_micro_panel_ready",
        evidence_class="gated_structured_repair_micro_panel",
        claim_scope="repair_live_rerun",
        summary={
            "artifact_present": bool(payload),
            "gated_structured_repair_micro_panel_ready": payload.get(
                "gated_structured_repair_micro_panel_ready"
            )
            is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "status_rationale": "expected .289 structured repair micro-panel artifact is absent"
            if not payload
            else "artifact_payload_classification",
        },
        row_origin="milestone_289",
    )


def _fr11_stress_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("fr11_stress_ready") is True
    return _row(
        row_id="dot289:exp3103_fr11_stress_promotion_boundary",
        status="missing" if not payload else "clean" if ready else "blocked",
        source_artifact=EXP3103_REL_PATH.as_posix(),
        source_field="fr11_stress_ready",
        evidence_class="fr11_resyn_kancl_stress_promotion_boundary",
        claim_scope="controller_only_stress_boundary_no_promotion",
        summary={
            "fr11_stress_ready": ready,
            "promotion_decision": str(payload.get("promotion_decision") or ""),
            "soundness_mistakes": _int_or_none(payload.get("soundness_mistakes")),
            "completeness_mistakes": _int_or_none(payload.get("completeness_mistakes")),
            "status_rationale": "boundary_complete_but_broader_promotion_blocked",
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _ebt_arm_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("sidecar_boundary_v2_ready") is True
    no_live = payload.get("no_live_model_integration_claim") is True
    status = "missing" if not payload else "projection_only" if ready and no_live else "clean" if ready else "blocked"
    return _row(
        row_id="dot289:exp3104_ebt_arm_sidecar_pipeline_boundary",
        status=status,
        source_artifact=EXP3104_REL_PATH.as_posix(),
        source_field="sidecar_boundary_v2_ready",
        evidence_class="ebt_arm_sidecar_pipeline_boundary",
        claim_scope="future_adapter_context",
        summary={
            "sidecar_boundary_v2_ready": ready,
            "no_live_model_integration_claim": no_live,
            "remaining_integration_blocker_count": len(_as_list(payload.get("remaining_integration_blockers"))),
            "status_rationale": "projection_only_no_live_model_integration",
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_289",
    )


def _gatemate_ssqa_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    ingestion_ready = payload.get("operator_evidence_ingestion_v3_ready") is True
    gatemate_allowed = payload.get("gatemate_rerun_allowed") is True
    ssqa_allowed = payload.get("ssqa_readback_allowed") is True
    return [
        _row(
            row_id="dot289:exp3106_gatemate_operator_evidence",
            status="missing" if not payload else "clean" if gatemate_allowed else "blocked",
            source_artifact=EXP3106_REL_PATH.as_posix(),
            source_field="gatemate_rerun_allowed",
            evidence_class="gatemate_ssqa_operator_evidence_ingestion",
            claim_scope="hardware_rerun_gate",
            summary={
                "operator_evidence_ingestion_v3_ready": ingestion_ready,
                "gatemate_rerun_allowed": gatemate_allowed,
                "missing_operator_action_count": len(_as_list(payload.get("missing_operator_actions"))),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            },
            row_origin="milestone_289",
        ),
        _row(
            row_id="dot289:exp3106_ssqa_readback_evidence",
            status="missing" if not payload else "clean" if ssqa_allowed else "gated_skipped" if ingestion_ready else "blocked",
            source_artifact=EXP3106_REL_PATH.as_posix(),
            source_field="ssqa_readback_allowed",
            evidence_class="gatemate_ssqa_operator_evidence_ingestion",
            claim_scope="host_visible_readback_gate",
            summary={
                "operator_evidence_ingestion_v3_ready": ingestion_ready,
                "ssqa_readback_allowed": ssqa_allowed,
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            },
            row_origin="milestone_289",
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
        "row_origin": str(row.get("row_origin") or "matrix_v22"),
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


def _gate_status(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "missing"
    failed = any(_as_mapping(gate).get("passed") is not True for gate in _as_list(payload.get("gates_evaluated")))
    blocked = str(payload.get("status") or "").lower() == "blocked"
    verdict_failed = "failed" in str(payload.get("honest_verdict") or "").lower()
    return "gated_skipped" if blocked or failed or verdict_failed else "clean"


def _repair_panel_status(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "missing"
    return "clean" if payload.get("gated_structured_repair_micro_panel_ready") is True else "blocked"


def _live_llm_status_and_gaps(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    ready_field: str,
) -> tuple[str, list[JsonDict]]:
    if not payload:
        return "missing", []
    gaps = _model_spec_gaps(row_id, payload, source_path) + _exact_count_gaps(row_id, payload, source_path)
    if payload.get(ready_field) is not True:
        return "blocked", gaps
    if _has_model_spec_gap(gaps):
        return "model_spec_gap", gaps
    if gaps or payload.get("flagged_adversarial") is True or _as_list(payload.get("corrigendum_pending")):
        return "flagged", gaps
    return "clean", []


def _model_spec_gaps(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
) -> list[JsonDict]:
    if not payload or not _is_live_llm_payload(payload):
        return []
    mandatory = [str(value) for value in _as_list(payload.get("mandatory_headline_model_ids"))]
    specs = [_as_mapping(row) for row in _as_list(payload.get("model_specs"))]
    present = {
        str(spec.get("hf_id") or spec.get("model_id") or spec.get("name") or "")
        for spec in specs
        if spec
    }
    if not mandatory:
        return [
            {
                "row_id": row_id,
                "source_artifact": source_path.as_posix(),
                "missing_model_ids": [],
                "present_model_ids": sorted(present),
                "reason": "mandatory_headline_model_ids missing for live LLM artifact",
            }
        ]
    missing = [model_id for model_id in mandatory if model_id not in present]
    if not missing:
        return []
    return [
        {
            "row_id": row_id,
            "source_artifact": source_path.as_posix(),
            "missing_model_ids": missing,
            "present_model_ids": sorted(present),
            "reason": "mandated model_specs missing for live LLM artifact",
        }
    ]


def _exact_count_gaps(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
) -> list[JsonDict]:
    minimum = _int_or_none(payload.get("minimum_live_eval_count"))
    exact = _int_or_none(payload.get("exact_ground_truth_count"))
    if minimum is None:
        return [
            {
                "row_id": row_id,
                "source_artifact": source_path.as_posix(),
                "missing_model_ids": [],
                "present_model_ids": [],
                "reason": "minimum_live_eval_count missing for .289 protocol",
            }
        ]
    if exact is None or exact < minimum:
        return [
            {
                "row_id": row_id,
                "source_artifact": source_path.as_posix(),
                "missing_model_ids": [],
                "present_model_ids": [],
                "reason": "exact fixture count below .289 protocol minimum",
                "exact_ground_truth_count": exact,
                "minimum_live_eval_count": minimum,
            }
        ]
    return []


def _is_live_llm_payload(payload: Mapping[str, Any]) -> bool:
    substrate = _as_mapping(payload.get("inference_substrate"))
    text = json.dumps(substrate, sort_keys=True).lower()
    return (
        substrate.get("live_llm_inference") is True
        or substrate.get("local_gguf_inference") is True
        or substrate.get("executes_models") is True
        or "live_llm_inference" in text
        or "llama_cpp" in text
        or "gguf" in text
    )


def _is_model_spec_gap(gap: Mapping[str, Any]) -> bool:
    return "model" in str(gap.get("reason") or "").lower()


def _has_model_spec_gap(gaps: list[Mapping[str, Any]]) -> bool:
    return any(_is_model_spec_gap(gap) for gap in gaps)


def _replacement_row_id(row_id: str) -> str:
    for old, new, _ in REPLACEMENTS:
        if old == row_id:
            return new
    return ""


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


def _before_count(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> int:
    return (
        _int_or_none(ledger.get("publication_blocker_count_before"))
        or _int_or_none(matrix.get("publication_blocker_count"))
        or _int_or_none(capstone.get("publication_blocker_count"))
        or 0
    )


def _blocker_reconciliation(
    *,
    before_count: int,
    after_count: int,
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    row_by_id = {str(row.get("row_id") or ""): row for row in rows}
    retired_old = []
    added_new = []
    neutral = []
    for old, new, reason in REPLACEMENTS:
        old_row = _as_mapping(row_by_id.get(old))
        new_row = _as_mapping(row_by_id.get(new))
        old_status = normal_status(str(_as_mapping(old_row.get("summary")).get("previous_status") or old_row.get("status") or "missing"))
        new_status = normal_status(str(new_row.get("status") or "missing"))
        if old_status in PUBLICATION_BLOCKING_STATUSES and old_row.get("status") == "retired":
            retired_old.append(old)
        if new_status in PUBLICATION_BLOCKING_STATUSES:
            added_new.append(new)
        neutral.append(
            {
                "old_row_id": old,
                "new_row_id": new,
                "old_status": old_status,
                "new_status": new_status,
                "reason": reason,
            }
        )
    decreases = []
    if retired_old:
        decreases.append(
            {
                "count": len(retired_old),
                "row_ids": retired_old,
                "reason": "Superseded v22 blocker rows retired after concrete v23 replacement rows were added.",
            }
        )
    increases = []
    if added_new:
        increases.append(
            {
                "count": len(added_new),
                "row_ids": added_new,
                "reason": "Replacement v23 rows preserve unresolved blockers instead of promoting .289 evidence.",
            }
        )
    return {
        "publication_blocker_count_before": before_count,
        "publication_blocker_count_after": after_count,
        "blocker_delta_from_v22": after_count - before_count,
        "decreases": decreases,
        "increases": increases,
        "neutral_replacements": neutral,
        "net_explanation": (
            "No net blocker-count movement: retired v22 blockers are offset by "
            "replacement v23 blockers with the same unresolved evidence boundaries."
            if after_count == before_count
            else "Net blocker-count movement follows the increase/decrease entries above."
        ),
    }


def _headline_model_spec_gaps(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    gaps: list[JsonDict] = []
    for row in rows:
        if normal_status(str(row.get("status") or "")) == "retired":
            continue
        summary = _as_mapping(row.get("summary"))
        gaps.extend(
            dict(gap)
            for gap in _as_list(summary.get("model_spec_gaps"))
            if isinstance(gap, Mapping) and _is_model_spec_gap(gap)
        )
    return gaps


def _public_sources(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row["experiment_id"]),
            "path": str(row["path"]),
            "role": str(row["role"]),
            "required": row.get("required") is True,
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


def _missing_artifacts(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    result: list[JsonDict] = []
    for row in sources:
        if row.get("path") == EXP3102_REL_PATH.as_posix() and row.get("present") is not True:
            result.append(
                {
                    "path": EXP3102_REL_PATH.as_posix(),
                    "reason": "expected .289 structured repair micro-panel artifact is absent",
                }
            )
    return result


def _capstone_input_artifacts() -> list[str]:
    return [OUTPUT_REL_PATH.as_posix(), *[spec.path.as_posix() for spec in SOURCE_SPECS]]


def _invariant_violations(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    ledger: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blockers: list[Mapping[str, Any]],
    required_source_errors: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if required_source_errors:
        violations.append("required source artifacts missing or malformed")
    if matrix and matrix.get("matrix_v22_ready") is not True:
        violations.append("matrix v22 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone .288 authority is not ready")
    if ledger and ledger.get("blocker_triage_ready") is not True:
        violations.append("Exp 3096 blocker triage ledger is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v23 statuses")
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
        "source": "matrix_v22_capstone_v288_ledger_and_dot289_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("matrix_v23_ready") is not True:
        return (
            "blocked_matrix_v23_preconditions: "
            f"invariant_violations={_as_list(artifact.get('invariant_violations'))}"
        )
    return (
        "complete: "
        "matrix_v23_ready=true; "
        f"rows_total={artifact['rows_total']}; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"blocker_delta_from_v22={artifact['blocker_delta_from_v22']}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}"
    )


def normal_status(status: str) -> str:
    """Normalize legacy labels into the v23 status vocabulary."""

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


__all__ = [
    "CAPSTONE_V288_REL_PATH",
    "EXP3095_REL_PATH",
    "EXP3096_REL_PATH",
    "EXP3097_REL_PATH",
    "EXP3098_REL_PATH",
    "EXP3099_REL_PATH",
    "EXP3100_REL_PATH",
    "EXP3101_REL_PATH",
    "EXP3102_REL_PATH",
    "EXP3103_REL_PATH",
    "EXP3104_REL_PATH",
    "EXP3105_REL_PATH",
    "EXP3106_REL_PATH",
    "LEGACY_STATUSES",
    "MATRIX_V22_REL_PATH",
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
