"""Build the Exp 3093 cross-corpus matrix v22 artifact.

Spec refs: REQ-REPORT-3093, SCENARIO-REPORT-3093.

Matrix v22 is an evidence ledger, not a fresh experiment. It reads matrix v21,
the .287 capstone, the Exp 3082 blocker ledger, and checked-in .288 artifacts
to decide which claim rows are clean, flagged, bounded, blocked, gated,
missing, retired, or projection-only. That separation matters because a
capstone matrix should summarize what the repository already proves, not
quietly rerun models, hardware, solvers, or repair loops while building a
paper-readiness table.
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
MILESTONE = "2026.05.288"
NEXT_CAPSTONE_EXPERIMENT = "exp3094"
SCHEMA = "carnot.cross_corpus_matrix.v22_288_claim_aggregation.v1"
ARTIFACT = "experiment_3093_cross_corpus_matrix_v22"
OUTPUT_REL_PATH = Path("results/experiment_3093_cross_corpus_matrix_v22.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3093_cross_corpus_matrix_v22.py"

MATRIX_V21_REL_PATH = Path("results/experiment_3079_cross_corpus_matrix_v21.json")
CAPSTONE_V287_REL_PATH = Path("results/experiment_3080_capstone_v287.json")
EXP3081_REL_PATH = Path("results/experiment_3081_archive_v287_activate_v288.json")
EXP3082_REL_PATH = Path("results/experiment_3082_publication_blocker_reduction_ledger_v1.json")
EXP3083_REL_PATH = Path("results/experiment_3083_verifier_hardness_autopsy_protocol_v1.json")
EXP3084_REL_PATH = Path("results/experiment_3084_resyn_exact_fixture_bank_generator_v1.json")
EXP3085_REL_PATH = Path("results/experiment_3085_icalm_task_abstention_sota_panel_v2.json")
EXP3086_REL_PATH = Path("results/experiment_3086_dafny_z3_formal_feedback_pilot_v1.json")
EXP3087_REL_PATH = Path("results/experiment_3087_gated_local_sota_verifier_calibration_v3.json")
EXP3088_REL_PATH = Path("results/experiment_3088_xgrammar2_structured_repair_emitter_preflight_v1.json")
EXP3089_REL_PATH = Path("results/experiment_3089_gated_xgrammar_sota_repair_micro_panel_v2.json")
EXP3090_REL_PATH = Path("results/experiment_3090_fr11_resyn_kancl_completeness_repair_v1.json")
EXP3091_REL_PATH = Path("results/experiment_3091_ebt_arm_sidecar_adapter_schema_prototype_v1.json")
EXP3092_REL_PATH = Path("results/experiment_3092_gatemate_ssqa_operator_evidence_ingestion_v2.json")

STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "retired",
    "projection_only",
)
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "missing",
    "projection_only",
}
LEDGER_CATEGORIES = (
    "verifier_gain",
    "repair_gate",
    "fr11_budget",
    "hardware_evidence",
    "adapter_projection",
    "missing_artifact",
    "bounded_status",
    "retired_status",
    "documentation_hygiene",
)

REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    (
        "capstone:v286_paper_readiness",
        "capstone:v287_paper_readiness",
        "capstone .287 supersedes the prior capstone paper-readiness row.",
    ),
    (
        "dot287:exp3070_first_token_abstention",
        "dot288:exp3085_icalm_task_abstention_sota_panel",
        "Exp 3085 supersedes the .287 first-token abstention row.",
    ),
    (
        "dot287:exp3071_verge_mcs_feedback",
        "dot288:exp3086_dafny_z3_formal_feedback_pilot",
        "Exp 3086 supersedes the .287 formal-feedback row.",
    ),
    (
        "dot287:exp3072_verifier_calibration_gate",
        "dot288:exp3087_local_sota_verifier_calibration_gate",
        "Exp 3087 supersedes the .287 verifier calibration gate row.",
    ),
    (
        "dot287:exp3073_ebt_arm_adapter_feasibility",
        "dot288:exp3091_ebt_arm_sidecar_adapter_schema_prototype",
        "Exp 3091 supersedes the adapter feasibility row without claiming live integration.",
    ),
    (
        "dot287:exp3075_repair_micro_panel",
        "dot288:exp3089_xgrammar_sota_repair_micro_panel",
        "The .288 repair micro-panel request supersedes the .287 gated repair row.",
    ),
    (
        "dot287:exp3078_gatemate_operator_refresh",
        "dot288:exp3092_gatemate_operator_evidence",
        "Exp 3092 supersedes the .287 GateMate operator-evidence refresh row.",
    ),
    (
        "dot287:exp3078_ssqa_readback_refresh",
        "dot288:exp3092_ssqa_readback_evidence",
        "Exp 3092 supersedes the .287 SSQA readback refresh row.",
    ),
)
REPLACED_ROW_IDS = {old for old, _, _ in REPLACEMENTS}


@dataclass(frozen=True)
class SourceSpec:
    """One checked-in source artifact that v22 cites without mutating it."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    source_type: str = "json"


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3079", MATRIX_V21_REL_PATH, "matrix_v21_authority", required=True),
    SourceSpec("exp3080", CAPSTONE_V287_REL_PATH, "capstone_v287_authority", required=True),
    SourceSpec("exp3082", EXP3082_REL_PATH, "blocker_ledger_authority", required=True),
    SourceSpec("exp3081", EXP3081_REL_PATH, "archive_v288_activation"),
    SourceSpec("exp3083", EXP3083_REL_PATH, "verifier_hardness_protocol"),
    SourceSpec("exp3084", EXP3084_REL_PATH, "resyn_exact_fixture_bank"),
    SourceSpec("exp3085", EXP3085_REL_PATH, "icalm_task_abstention_panel"),
    SourceSpec("exp3086", EXP3086_REL_PATH, "dafny_z3_formal_feedback"),
    SourceSpec("exp3087", EXP3087_REL_PATH, "gated_verifier_calibration"),
    SourceSpec("exp3088", EXP3088_REL_PATH, "structured_repair_emitter_preflight"),
    SourceSpec("exp3089", EXP3089_REL_PATH, "gated_xgrammar_repair_micro_panel"),
    SourceSpec("exp3090", EXP3090_REL_PATH, "fr11_resyn_kancl_completeness_repair"),
    SourceSpec("exp3091", EXP3091_REL_PATH, "ebt_arm_sidecar_adapter_schema"),
    SourceSpec("exp3092", EXP3092_REL_PATH, "gatemate_ssqa_operator_evidence_ingestion"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object and return empty evidence when the source is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a file checksum for source traceability."""

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
    """REQ-REPORT-3093: aggregate matrix v22 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3079"]
    capstone = payloads["exp3080"]
    ledger = payloads["exp3082"]
    ledger_categories = _ledger_categories(ledger)
    fr11_clear_ids = _fr11_clear_ids(ledger_categories, payloads["exp3090"])
    rows = (
        _carry_forward_rows(matrix, fr11_clear_ids) + _dot288_rows(payloads)
        if matrix and capstone and ledger
        else []
    )
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    required_source_errors = _required_source_errors(sources)
    missing_artifacts = _missing_artifacts(sources)
    headline_model_spec_gaps = _headline_model_spec_gaps(rows)
    reconciliation = _blocker_reconciliation(
        before_count=_before_count(matrix, ledger),
        after_count=len(publication_blockers),
        fr11_clear_ids=fr11_clear_ids,
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
        "matrix_v22_ready": ready,
        "rows_total": len(rows),
        "status_counts": status_counts,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v21": reconciliation["blocker_delta_from_v21"],
        "publication_blockers": publication_blockers,
        "rows": rows,
        "missing_artifacts": missing_artifacts,
        "headline_model_spec_gaps": headline_model_spec_gaps,
        "blocker_reconciliation_from_ledger": reconciliation,
        "capstone_input_artifacts": _capstone_input_artifacts(),
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
    """Build and persist the Exp 3093 matrix artifact."""

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


def _carry_forward_rows(matrix: Mapping[str, Any], fr11_clear_ids: set[str]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for raw in _as_list(matrix.get("rows")):
        if not isinstance(raw, Mapping):
            continue
        row = _claim_entry(raw)
        original_status = row["status"]
        summary = _as_mapping(row.get("summary"))
        if row["row_id"] in fr11_clear_ids:
            row["status"] = "clean"
            summary.update(
                {
                    "previous_status": original_status,
                    "v22_status_rationale": (
                        "Exp 3090 passed the controller-only FR-11 soundness, "
                        "completeness, and non-vacuity gates."
                    ),
                    "superseding_artifact": EXP3090_REL_PATH.as_posix(),
                    "claim_boundary": "controller_only_no_model_weight_update",
                }
            )
        elif row["row_id"] in REPLACED_ROW_IDS:
            row["status"] = "retired"
            summary.update(
                {
                    "previous_status": original_status,
                    "v22_status_rationale": "superseded_by_current_dot288_row",
                    "superseding_row_id": _replacement_row_id(row["row_id"]),
                }
            )
        else:
            summary.setdefault("v22_status_rationale", "carried_forward_from_matrix_v21")
        row["blocker_class"] = blocker_class(row["status"])
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v21")
        rows.append(row)
    return rows


def _dot288_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _capstone_v287_row(payloads["exp3080"]),
        _ready_row(
            "dot288:exp3081_archive_activation",
            payloads["exp3081"],
            EXP3081_REL_PATH,
            "archive_v287_activate_v288_ready",
            "archive_activation",
            "milestone_activation",
        ),
        _ready_row(
            "dot288:exp3082_blocker_reduction_ledger",
            payloads["exp3082"],
            EXP3082_REL_PATH,
            "blocker_ledger_ready",
            "publication_blocker_reduction_ledger",
            "matrix_v22_accounting",
        ),
        _ready_row(
            "dot288:exp3083_verifier_hardness_protocol",
            payloads["exp3083"],
            EXP3083_REL_PATH,
            "verifier_hardness_protocol_ready",
            "verifier_hardness_protocol",
            "recovery_protocol",
        ),
        _ready_row(
            "dot288:exp3084_resyn_exact_fixture_bank",
            payloads["exp3084"],
            EXP3084_REL_PATH,
            "resyn_fixture_bank_ready",
            "resyn_exact_fixture_bank",
            "exact_fixture_bank",
        ),
        _live_llm_row(
            "dot288:exp3085_icalm_task_abstention_sota_panel",
            payloads["exp3085"],
            EXP3085_REL_PATH,
            ready_field="abstention_panel_v2_ready",
            evidence_class="icalm_task_abstention_sota_panel",
            claim_scope="local_sota_solution_verifier_gain",
            numeric_gate_field="abstention_precision",
            numeric_gate_min=0.7,
        ),
        _live_llm_row(
            "dot288:exp3086_dafny_z3_formal_feedback_pilot",
            payloads["exp3086"],
            EXP3086_REL_PATH,
            ready_field="formal_feedback_ready",
            evidence_class="dafny_z3_formal_feedback_pilot",
            claim_scope="solver_grounded_repair_feedback",
        ),
        _gate_record_row(payloads["exp3087"]),
        _ready_row(
            "dot288:exp3088_structured_repair_emitter_preflight",
            payloads["exp3088"],
            EXP3088_REL_PATH,
            "structured_generation_ready",
            "structured_repair_emitter_preflight",
            "repair_payload_preflight",
        ),
        _repair_micro_panel_row(payloads["exp3089"]),
        _fr11_repair_row(payloads["exp3090"]),
        _adapter_schema_row(payloads["exp3091"]),
        *_gatemate_ssqa_rows(payloads["exp3092"]),
    ]


def _capstone_v287_row(capstone: Mapping[str, Any]) -> JsonDict:
    status = "missing"
    if capstone:
        status = "blocked" if capstone.get("capstone_ready") is not True else "clean"
        if capstone.get("capstone_ready") is True and capstone.get("paper_ready") is not True:
            status = "bounded"
    return _row(
        row_id="capstone:v287_paper_readiness",
        status=status,
        source_artifact=CAPSTONE_V287_REL_PATH.as_posix(),
        source_field="paper_ready",
        evidence_class="capstone_v287_authority",
        claim_scope="paper_readiness",
        summary={
            "capstone_ready": capstone.get("capstone_ready") is True,
            "paper_ready": capstone.get("paper_ready") is True,
            "publication_blocker_count": _int_or_none(capstone.get("publication_blocker_count")),
        },
        row_origin="milestone_288",
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
        row_origin="milestone_288",
    )


def _live_llm_row(
    row_id: str,
    payload: Mapping[str, Any],
    source_path: Path,
    *,
    ready_field: str,
    evidence_class: str,
    claim_scope: str,
    numeric_gate_field: str | None = None,
    numeric_gate_min: float | None = None,
) -> JsonDict:
    gaps = _model_spec_gaps(row_id, payload, source_path)
    gate_value = _float_or_none(payload.get(numeric_gate_field)) if numeric_gate_field else None
    numeric_gate_failed = (
        numeric_gate_field is not None
        and numeric_gate_min is not None
        and gate_value is not None
        and gate_value < numeric_gate_min
    )
    flagged = (
        payload.get("flagged_adversarial") is True
        or bool(_as_list(payload.get("corrigendum_pending")))
        or numeric_gate_failed
    )
    status = "missing" if not payload else "flagged" if flagged else "clean"
    if payload and status == "clean" and gaps:
        status = "flagged"
    if payload and not flagged and payload.get(ready_field) is not True:
        status = "blocked"
    return _row(
        row_id=row_id,
        status=status,
        source_artifact=source_path.as_posix(),
        source_field=ready_field,
        evidence_class=evidence_class,
        claim_scope=claim_scope,
        summary={
            "ready": payload.get(ready_field) is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "numeric_gate_field": numeric_gate_field or "",
            "numeric_gate_value": gate_value,
            "numeric_gate_min": numeric_gate_min,
            "model_spec_gate_passed": not gaps,
            "model_spec_gap_count": len(gaps),
            "model_spec_gaps": gaps,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_288",
    )


def _gate_record_row(payload: Mapping[str, Any]) -> JsonDict:
    return _row(
        row_id="dot288:exp3087_local_sota_verifier_calibration_gate",
        status=_gate_status(payload),
        source_artifact=EXP3087_REL_PATH.as_posix(),
        source_field="gate_check_summary",
        evidence_class="local_sota_verifier_calibration_gate",
        claim_scope="verifier_gain_recovery_gate",
        summary={
            "status": str(payload.get("status") or ""),
            "gate_check_summary": str(payload.get("gate_check_summary") or ""),
            "failed_gate_count": sum(
                1
                for gate in _as_list(payload.get("gates_evaluated"))
                if _as_mapping(gate).get("passed") is not True
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_288",
    )


def _gate_status(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "missing"
    failed = any(_as_mapping(gate).get("passed") is not True for gate in _as_list(payload.get("gates_evaluated")))
    blocked = str(payload.get("status") or "").lower() == "blocked"
    verdict_failed = "failed" in str(payload.get("honest_verdict") or "").lower()
    return "gated_skipped" if blocked or failed or verdict_failed else "clean"


def _repair_micro_panel_row(panel: Mapping[str, Any]) -> JsonDict:
    return _row(
        row_id="dot288:exp3089_xgrammar_sota_repair_micro_panel",
        status=_repair_panel_status(panel),
        source_artifact=EXP3089_REL_PATH.as_posix(),
        source_field="repair_panel_ready",
        evidence_class="gated_xgrammar_sota_repair_micro_panel",
        claim_scope="repair_live_rerun",
        summary={
            "artifact_present": bool(panel),
            "repair_panel_ready": panel.get("repair_panel_ready") is True,
            "flagged_adversarial": panel.get("flagged_adversarial") is True,
            "honest_verdict": str(panel.get("honest_verdict") or ""),
            "status_rationale": "expected .288 repair micro-panel artifact is absent"
            if not panel
            else "artifact_payload_classification",
        },
        row_origin="milestone_288",
    )


def _repair_panel_status(panel: Mapping[str, Any]) -> str:
    if not panel:
        return "missing"
    flagged = panel.get("flagged_adversarial") is True or bool(_as_list(panel.get("corrigendum_pending")))
    return "flagged" if flagged else "clean" if panel.get("repair_panel_ready") is True else "blocked"


def _fr11_repair_row(payload: Mapping[str, Any]) -> JsonDict:
    passed = _fr11_repair_passed(payload)
    return _row(
        row_id="dot288:exp3090_fr11_resyn_kancl_completeness_repair",
        status="missing" if not payload else "clean" if passed else "blocked",
        source_artifact=EXP3090_REL_PATH.as_posix(),
        source_field="budget_gates",
        evidence_class="fr11_resyn_kancl_completeness_repair",
        claim_scope="controller_only_online_learning_budget",
        summary={
            "fr11_resyn_kancl_ready": payload.get("fr11_resyn_kancl_ready") is True,
            "promotion_decision": str(payload.get("promotion_decision") or ""),
            "controller_only": True,
            "fr11_repair_passed": passed,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
        row_origin="milestone_288",
    )


def _adapter_schema_row(payload: Mapping[str, Any]) -> JsonDict:
    ready = payload.get("adapter_schema_ready") is True and payload.get("sidecar_replay_scorer_ready") is True
    return _row(
        row_id="dot288:exp3091_ebt_arm_sidecar_adapter_schema_prototype",
        status="missing" if not payload else "projection_only" if ready else "blocked",
        source_artifact=EXP3091_REL_PATH.as_posix(),
        source_field="implementation_claim_boundary",
        evidence_class="ebt_arm_sidecar_adapter_schema_prototype",
        claim_scope="future_adapter_context",
        summary={
            "adapter_schema_ready": payload.get("adapter_schema_ready") is True,
            "sidecar_replay_scorer_ready": payload.get("sidecar_replay_scorer_ready") is True,
            "implementation_claim_boundary": str(payload.get("implementation_claim_boundary") or ""),
            "status_rationale": "prototype_only_no_live_inference_integration",
        },
        row_origin="milestone_288",
    )


def _gatemate_ssqa_rows(payload: Mapping[str, Any]) -> list[JsonDict]:
    ingestion_ready = payload.get("operator_evidence_ingestion_ready") is True
    gatemate_allowed = payload.get("gatemate_rerun_allowed") is True
    ssqa_allowed = payload.get("ssqa_readback_allowed") is True
    return [
        _row(
            row_id="dot288:exp3092_gatemate_operator_evidence",
            status="missing" if not payload else "clean" if gatemate_allowed else "blocked",
            source_artifact=EXP3092_REL_PATH.as_posix(),
            source_field="gatemate_rerun_allowed",
            evidence_class="gatemate_ssqa_operator_evidence_ingestion",
            claim_scope="hardware_rerun_gate",
            summary={
                "operator_evidence_ingestion_ready": ingestion_ready,
                "gatemate_rerun_allowed": gatemate_allowed,
                "missing_operator_action_count": len(_as_list(payload.get("missing_operator_actions"))),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            },
            row_origin="milestone_288",
        ),
        _row(
            row_id="dot288:exp3092_ssqa_readback_evidence",
            status="missing" if not payload else "clean" if ssqa_allowed else "gated_skipped" if ingestion_ready else "blocked",
            source_artifact=EXP3092_REL_PATH.as_posix(),
            source_field="ssqa_readback_allowed",
            evidence_class="gatemate_ssqa_operator_evidence_ingestion",
            claim_scope="host_visible_readback_gate",
            summary={
                "operator_evidence_ingestion_ready": ingestion_ready,
                "ssqa_readback_allowed": ssqa_allowed,
                "honest_verdict": str(payload.get("honest_verdict") or ""),
            },
            row_origin="milestone_288",
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
        "row_origin": str(row.get("row_origin") or "matrix_v21"),
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


def _ledger_categories(ledger: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    raw_categories = _as_mapping(ledger.get("blocker_categories"))
    return {
        category: [
            _as_mapping(row)
            for row in _as_list(raw_categories.get(category))
            if isinstance(row, Mapping)
        ]
        for category in LEDGER_CATEGORIES
    }


def _fr11_clear_ids(
    ledger_categories: Mapping[str, list[Mapping[str, Any]]],
    exp3090: Mapping[str, Any],
) -> set[str]:
    if not _fr11_repair_passed(exp3090):
        return set()
    return {str(row.get("row_id") or "") for row in ledger_categories.get("fr11_budget", [])}


def _fr11_repair_passed(payload: Mapping[str, Any]) -> bool:
    gates = _as_mapping(payload.get("budget_gates"))
    if payload.get("fr11_resyn_kancl_ready") is not True and gates.get("all_gates_passed") is not True:
        return False
    soundness = _as_mapping(gates.get("soundness_mistakes"))
    completeness = _as_mapping(gates.get("completeness_mistakes"))
    controls = _as_mapping(gates.get("controls_non_vacuous"))
    return (
        gates.get("all_gates_passed") is True
        and _int_or_none(soundness.get("observed")) == 0
        and _int_or_none(completeness.get("observed")) == 0
        and controls.get("passed") is True
    )


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


def _before_count(matrix: Mapping[str, Any], ledger: Mapping[str, Any]) -> int:
    return (
        _int_or_none(ledger.get("publication_blocker_count_before"))
        or _int_or_none(matrix.get("publication_blocker_count"))
        or 0
    )


def _blocker_reconciliation(
    *,
    before_count: int,
    after_count: int,
    fr11_clear_ids: set[str],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    row_by_id = {str(row.get("row_id") or ""): row for row in rows}
    decreases = []
    if fr11_clear_ids:
        decreases.append(
            {
                "count": len(fr11_clear_ids),
                "ledger_category": "fr11_budget",
                "row_ids": sorted(fr11_clear_ids),
                "reason": (
                    "Exp 3090 passed zero-soundness, zero-completeness, "
                    "non-vacuous controller-only FR-11 gates."
                ),
            }
        )
    increases = []
    if after_count > before_count:
        increases.append(
            {
                "count": after_count - before_count,
                "reason": "New v22 blocker rows exceeded retired or cleaned v21 blockers.",
            }
        )
    neutral = [
        {
            "old_row_id": old,
            "new_row_id": new,
            "old_status": str(_as_mapping(row_by_id.get(old)).get("status") or "missing"),
            "new_status": str(_as_mapping(row_by_id.get(new)).get("status") or "missing"),
            "reason": reason,
        }
        for old, new, reason in REPLACEMENTS
        if old in row_by_id or new in row_by_id
    ]
    return {
        "publication_blocker_count_before": before_count,
        "publication_blocker_count_after": after_count,
        "blocker_delta_from_v21": after_count - before_count,
        "decreases": decreases,
        "increases": increases,
        "neutral_replacements": neutral,
    }


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
    return [
        {
            "row_id": row_id,
            "source_artifact": source_path.as_posix(),
            "missing_model_ids": missing,
            "present_model_ids": sorted(present),
            "reason": "mandated model_specs missing for live LLM artifact",
        }
    ] if missing else []


def _headline_model_spec_gaps(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    gaps: list[JsonDict] = []
    for row in rows:
        summary = _as_mapping(row.get("summary"))
        gaps.extend(
            dict(gap)
            for gap in _as_list(summary.get("model_spec_gaps"))
            if isinstance(gap, Mapping)
        )
    return gaps


def _is_live_llm_payload(payload: Mapping[str, Any]) -> bool:
    substrate = _as_mapping(payload.get("inference_substrate"))
    text = json.dumps(substrate, sort_keys=True).lower()
    return (
        substrate.get("live_llm_inference") is True
        or substrate.get("local_gguf_inference") is True
        or "live_llm_inference" in text
        or "llama_cpp" in text
    )


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
        if row.get("path") == EXP3089_REL_PATH.as_posix() and row.get("present") is not True:
            result.append(
                {
                    "path": EXP3089_REL_PATH.as_posix(),
                    "reason": "expected .288 repair micro-panel artifact is absent",
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
    if matrix and matrix.get("matrix_v21_ready") is not True:
        violations.append("matrix v21 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone .287 authority is not ready")
    if ledger and ledger.get("blocker_ledger_ready") is not True:
        violations.append("Exp 3082 blocker ledger is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v22 statuses")
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
        "source": "matrix_v21_capstone_v287_ledger_and_dot288_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
        "local_repo_only": True,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("matrix_v22_ready") is not True:
        return (
            "blocked_matrix_v22_preconditions: "
            f"invariant_violations={_as_list(artifact.get('invariant_violations'))}"
        )
    return (
        "complete: "
        "matrix_v22_ready=true; "
        f"rows_total={artifact['rows_total']}; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"blocker_delta_from_v21={artifact['blocker_delta_from_v21']}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}"
    )


def normal_status(status: str) -> str:
    """Normalize legacy labels into the v22 status vocabulary."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
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
    "CAPSTONE_V287_REL_PATH",
    "EXP3081_REL_PATH",
    "EXP3082_REL_PATH",
    "EXP3083_REL_PATH",
    "EXP3084_REL_PATH",
    "EXP3085_REL_PATH",
    "EXP3086_REL_PATH",
    "EXP3087_REL_PATH",
    "EXP3088_REL_PATH",
    "EXP3089_REL_PATH",
    "EXP3090_REL_PATH",
    "EXP3091_REL_PATH",
    "EXP3092_REL_PATH",
    "LEDGER_CATEGORIES",
    "MATRIX_V21_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "STATUSES",
    "SourceSpec",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
