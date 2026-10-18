"""Build the Exp 3065 cross-corpus matrix v20 artifact.

Spec refs: REQ-REPORT-3065, SCENARIO-REPORT-3065.

This module is deliberately only an accounting step. It reads checked-in
artifacts from the previous capstone and the current milestone, classifies the
claims those artifacts support, and records the blockers that still prevent a
paper-ready result. It does not try to repair anything live, rerun a model,
invoke a solver, synthesize RTL, flash a board, or reinterpret a missing
hardware transcript as success.
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
MILESTONE = "2026.05.286"
SCHEMA = "carnot.cross_corpus_matrix.v20_286_claim_aggregation.v1"
ARTIFACT = "experiment_3065_cross_corpus_matrix_v20"
OUTPUT_REL_PATH = Path("results/experiment_3065_cross_corpus_matrix_v20.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3065_cross_corpus_matrix_v20.py"

MATRIX_V19_REL_PATH = Path("results/experiment_3052_cross_corpus_matrix_v19.json")
CAPSTONE_V285_REL_PATH = Path("results/experiment_3053_capstone_v285.json")
EXP3054_REL_PATH = Path("results/experiment_3054_archive_v285_activate_v286.json")
EXP3055_REL_PATH = Path(
    "results/experiment_3055_repair_headline_retirement_and_blocker_ledger_v1.json"
)
EXP3056_REL_PATH = Path("results/experiment_3056_repair_de_tautology_protocol_v1.json")
EXP3057_REL_PATH = Path("results/experiment_3057_local_sota_solution_verifier_gain_panel_v1.json")
EXP3058_REL_PATH = Path("results/experiment_3058_aquaforte_style_llm_guided_smt_pilot_v1.json")
EXP3059_REQUESTED_REL_PATH = Path(
    "results/experiment_3059_gated_sota_repair_de_tautology_rerun_v1.json"
)
EXP3059_ACTUAL_REL_PATH = Path("results/experiment_3059_gated_sota_repair_de_tautology_rerun.json")
EXP3060_REL_PATH = Path("results/experiment_3060_fr11_solver_self_model_trace_schema_v1.json")
EXP3061_REL_PATH = Path(
    "results/experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.json"
)
EXP3062_REL_PATH = Path("results/experiment_3062_kan_pwa_locality_verification_audit_v1.json")
EXP3063_REL_PATH = Path("results/experiment_3063_gatemate_no_rerun_operator_action_ledger_v1.json")
EXP3064_REL_PATH = Path(
    "results/experiment_3064_ssqa_host_visible_readback_boundary_ledger_v1.json"
)

STATUSES = (
    "clean",
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
    "retired",
)
CLASS_FIELDS = {
    "clean": "clean_rows",
    "flagged": "flagged_rows",
    "bounded": "bounded_rows",
    "blocked": "blocked_rows",
    "gated_skipped": "gated_skipped_rows",
    "projection_only": "projection_only_rows",
    "missing": "missing_rows",
    "retired": "retired_rows",
}
PUBLICATION_BLOCKING_STATUSES = {
    "flagged",
    "bounded",
    "blocked",
    "gated_skipped",
    "projection_only",
    "missing",
}
REQUIRED_ROW_KEYS = {
    "row_id",
    "status",
    "source_artifact",
    "source_field",
    "evidence_class",
    "blocker_class",
    "claim_scope",
    "summary",
}


@dataclass(frozen=True)
class SourceSpec:
    """One artifact path that matrix v20 is expected to inspect or record.

    Required sources are matrix v19 and capstone v285 because v20 has no claim
    boundary without them. Milestone .286 sources are optional in the sense that
    a missing artifact is itself evidence; v20 stays usable only if that
    absence is represented as a machine-readable missing row.
    """

    experiment_id: str
    path: Path
    role: str
    required: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3052", MATRIX_V19_REL_PATH, "matrix_v19_authority", required=True),
    SourceSpec("exp3053", CAPSTONE_V285_REL_PATH, "capstone_v285_authority", required=True),
    SourceSpec("exp3054", EXP3054_REL_PATH, "archive_v286_activation"),
    SourceSpec("exp3055", EXP3055_REL_PATH, "repair_headline_blocker_ledger"),
    SourceSpec("exp3056", EXP3056_REL_PATH, "repair_de_tautology_protocol"),
    SourceSpec("exp3057", EXP3057_REL_PATH, "local_sota_solution_verifier_gain_panel"),
    SourceSpec("exp3058", EXP3058_REL_PATH, "aquaforte_style_smt_pilot"),
    SourceSpec("exp3059_requested_v1_alias", EXP3059_REQUESTED_REL_PATH, "requested_exp3059_alias"),
    SourceSpec("exp3059", EXP3059_ACTUAL_REL_PATH, "gated_sota_repair_rerun_gate_result"),
    SourceSpec("exp3060", EXP3060_REL_PATH, "fr11_trace_schema"),
    SourceSpec("exp3061", EXP3061_REL_PATH, "fr11_delayed_regression_pilot"),
    SourceSpec("exp3062", EXP3062_REL_PATH, "kan_pwa_locality_audit"),
    SourceSpec("exp3063", EXP3063_REL_PATH, "gatemate_no_rerun_ledger"),
    SourceSpec("exp3064", EXP3064_REL_PATH, "ssqa_readback_boundary_ledger"),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed on absence, arrays, or bad JSON."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 digest for a present source artifact."""

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
    """REQ-REPORT-3065: aggregate v20 rows from checked-in artifacts only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    loaded = {spec.experiment_id: _load_source(root_path, spec) for spec in SOURCE_SPECS}
    payloads = {experiment_id: row["payload"] for experiment_id, row in loaded.items()}
    source_artifacts = [
        _source_artifact(root_path, spec, loaded[spec.experiment_id]) for spec in SOURCE_SPECS
    ]
    rows = _build_rows(payloads, source_artifacts)
    row_classes = _classify_rows(rows)
    publication_blockers = _publication_blockers(rows)
    required_errors = _required_source_errors(source_artifacts)
    ready = (
        not required_errors
        and _rows_machine_readable(rows)
        and _source_records_machine_readable(source_artifacts)
        and all(isinstance(row_classes[field], list) for field in CLASS_FIELDS.values())
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v20_ready": ready,
        "paper_ready": ready and not publication_blockers,
        "rows_total": len(rows),
        **row_classes,
        "publication_blocker_count": len(publication_blockers),
        "publication_blockers": publication_blockers,
        "status_summaries": _status_summaries(payloads),
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row.get("sha256") for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row.get("present") is not True
        ],
        "required_source_errors": required_errors,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "status_updates_written": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": _honest_verdict(ready, len(rows), publication_blockers, required_errors),
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3065 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _load_source(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    return {"payload": read_json_object(path), "present": path.is_file()}


def _source_artifact(root: Path, spec: SourceSpec, loaded: Mapping[str, Any]) -> JsonDict:
    path = root / spec.path
    present = bool(loaded.get("present"))
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "present": present,
        "readable_json_object": bool(loaded.get("payload")),
        "sha256": sha256_file(path),
        "missing_recorded": not present,
        "machine_readable_record": True,
    }


def _required_source_errors(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row.get("experiment_id")),
            "reason": "missing_or_malformed_required_artifact",
        }
        for row in source_artifacts
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _build_rows(
    payloads: Mapping[str, Mapping[str, Any]],
    source_artifacts: list[Mapping[str, Any]],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    rows.extend(_missing_source_rows(source_artifacts))
    rows.extend(_matrix_v19_rows(payloads.get("exp3052", {})))
    rows.extend(_capstone_v285_rows(payloads.get("exp3053", {})))
    rows.extend(_archive_rows(payloads.get("exp3054", {})))
    rows.extend(_repair_ledger_rows(payloads.get("exp3055", {})))
    rows.extend(_repair_protocol_rows(payloads.get("exp3056", {})))
    rows.extend(_solution_verifier_rows(payloads.get("exp3057", {})))
    rows.extend(_aquaforte_rows(payloads.get("exp3058", {})))
    rows.extend(_gated_repair_rows(payloads.get("exp3059", {})))
    rows.extend(_fr11_schema_rows(payloads.get("exp3060", {})))
    rows.extend(_fr11_delayed_rows(payloads.get("exp3061", {})))
    rows.extend(_kan_pwa_rows(payloads.get("exp3062", {})))
    rows.extend(_gatemate_rows(payloads.get("exp3063", {})))
    rows.extend(_ssqa_rows(payloads.get("exp3064", {})))
    return rows


def _missing_source_rows(source_artifacts: list[Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in source_artifacts:
        if source.get("present") is True:
            continue
        rows.append(
            _row(
                row_id=f"source:{source.get('experiment_id')}",
                status="missing",
                source_artifact=str(source.get("path") or ""),
                source_field="source_artifacts.present",
                evidence_class="source_artifact_presence",
                claim_scope="source_artifact_accounting",
                summary={
                    "role": str(source.get("role") or ""),
                    "required": source.get("required") is True,
                    "missing_recorded": True,
                },
            )
        )
    return rows


def _matrix_v19_rows(matrix: Mapping[str, Any]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for index, source_row in enumerate(_as_list(matrix.get("rows"))):
        row = _as_mapping(source_row)
        original_id = str(row.get("row_id") or f"row_{index}")
        status = normal_status(str(row.get("status") or "missing"))
        rows.append(
            _row(
                row_id=f"v19:{original_id}",
                status=status,
                source_artifact=str(row.get("source_artifact") or MATRIX_V19_REL_PATH.as_posix()),
                source_field=str(row.get("source_field") or f"rows[{original_id}]"),
                evidence_class=str(row.get("evidence_class") or "matrix_v19_row"),
                claim_scope=str(row.get("claim_scope") or "matrix_v19_carry_forward"),
                summary={
                    "matrix_v19_row_id": original_id,
                    "matrix_v19_status": status,
                    "matrix_v19_ready": matrix.get("matrix_v19_ready") is True,
                    "summary": _as_mapping(row.get("summary")),
                },
            )
        )
    return rows


def _capstone_v285_rows(capstone: Mapping[str, Any]) -> list[JsonDict]:
    if not capstone:
        return []
    status = "clean" if capstone.get("paper_ready") is True else "bounded"
    if capstone.get("capstone_ready") is not True:
        status = "blocked"
    return [
        _row(
            row_id="capstone:v285_paper_readiness",
            status=status,
            source_artifact=CAPSTONE_V285_REL_PATH.as_posix(),
            source_field="paper_ready",
            evidence_class="capstone_synthesis",
            claim_scope="paper_readiness",
            summary={
                "capstone_ready": capstone.get("capstone_ready") is True,
                "paper_ready": capstone.get("paper_ready") is True,
                "repair_claim_status": str(capstone.get("repair_claim_status") or ""),
                "fr11_self_learning_status": str(capstone.get("fr11_self_learning_status") or ""),
                "gatemate_status": str(capstone.get("gatemate_status") or ""),
                "ssqa_status": str(capstone.get("ssqa_status") or ""),
            },
        )
    ]


def _archive_rows(archive: Mapping[str, Any]) -> list[JsonDict]:
    if not archive:
        return []
    status = "clean" if archive.get("archive_v285_activate_v286_ready") is True else "blocked"
    return [
        _row(
            row_id="archive:v286_activation",
            status=status,
            source_artifact=EXP3054_REL_PATH.as_posix(),
            source_field="archive_v285_activate_v286_ready",
            evidence_class="archive_activation",
            claim_scope="milestone_activation",
            summary={
                "prior_capstone_ready": archive.get("prior_capstone_ready") is True,
                "prior_paper_ready": archive.get("prior_paper_ready") is True,
                "carry_forward_blocker_count": len(_as_list(archive.get("carry_forward_blockers"))),
            },
        )
    ]


def _repair_ledger_rows(ledger: Mapping[str, Any]) -> list[JsonDict]:
    if not ledger:
        return []
    rows = [
        _row(
            row_id="repair:headline_retirement_ledger",
            status="clean" if ledger.get("repair_headline_retirement_ready") is True else "blocked",
            source_artifact=EXP3055_REL_PATH.as_posix(),
            source_field="repair_headline_retirement_ready",
            evidence_class="repair_blocker_ledger",
            claim_scope="repair_claim_boundary",
            summary={
                "repair_claim_status": str(ledger.get("repair_claim_status") or ""),
                "retired_count": len(_as_list(ledger.get("retired_repair_claims"))),
                "bounded_count": len(_as_list(ledger.get("still_bounded_repair_claims"))),
                "rerun_prerequisite_count": len(_as_list(ledger.get("rerun_prerequisites"))),
            },
        )
    ]
    blockers = [_as_mapping(row) for row in _as_list(ledger.get("extracted_repair_blockers"))]
    if blockers:
        rows.append(
            _row(
                row_id="repair:prior_adversarial_blockers",
                status="flagged",
                source_artifact=EXP3055_REL_PATH.as_posix(),
                source_field="extracted_repair_blockers",
                evidence_class="repair_adversarial_blocker_ledger",
                claim_scope="repair_headline_boundary",
                summary={
                    "blocker_count": len(blockers),
                    "blocking_row_ids": [str(row.get("row_id") or "") for row in blockers],
                    "flagged_adversarial": ledger.get("flagged_adversarial") is True,
                },
            )
        )
    for item in _as_list(ledger.get("still_bounded_repair_claims")):
        claim = _as_mapping(item)
        rows.append(
            _row(
                row_id=str(claim.get("row_id") or f"repair:bounded_{len(rows)}"),
                status="bounded",
                source_artifact=str(claim.get("source_artifact") or EXP3055_REL_PATH.as_posix()),
                source_field=str(claim.get("source_field") or "still_bounded_repair_claims"),
                evidence_class="repair_bounded_claim",
                claim_scope="repair_headline_boundary",
                summary={
                    "claim_id": str(claim.get("claim_id") or ""),
                    "repair_claim_status": str(claim.get("repair_claim_status") or ""),
                    "allowed_wording": str(claim.get("allowed_wording") or ""),
                },
            )
        )
    for item in _as_list(ledger.get("retired_repair_claims")):
        claim = _as_mapping(item)
        claim_id = str(claim.get("claim_id") or claim.get("row_id") or f"retired_{len(rows)}")
        rows.append(
            _row(
                row_id=f"repair:{claim_id}",
                status="retired",
                source_artifact=str(claim.get("source_artifact") or EXP3055_REL_PATH.as_posix()),
                source_field=str(claim.get("source_field") or "retired_repair_claims"),
                evidence_class="repair_retired_claim",
                claim_scope="retired_repair_headline_wording",
                summary={
                    "claim_id": claim_id,
                    "allowed_wording": str(claim.get("allowed_wording") or ""),
                },
            )
        )
    return rows


def _repair_protocol_rows(protocol: Mapping[str, Any]) -> list[JsonDict]:
    if not protocol:
        return []
    rows = [
        _row(
            row_id="repair:de_tautology_protocol",
            status="clean"
            if protocol.get("repair_de_tautology_protocol_ready") is True
            else "blocked",
            source_artifact=EXP3056_REL_PATH.as_posix(),
            source_field="repair_de_tautology_protocol_ready",
            evidence_class="repair_de_tautology_protocol",
            claim_scope="repair_rerun_protocol",
            summary={
                "required_live_run_field_count": len(
                    _as_list(protocol.get("required_live_run_fields"))
                )
            },
        )
    ]
    disqualifiers = _as_list(protocol.get("promotion_disqualifiers"))
    if disqualifiers:
        rows.append(
            _row(
                row_id="repair:de_tautology_disqualifiers",
                status="blocked",
                source_artifact=EXP3056_REL_PATH.as_posix(),
                source_field="promotion_disqualifiers",
                evidence_class="repair_promotion_disqualifiers",
                claim_scope="repair_headline_boundary",
                summary={"disqualifier_count": len(disqualifiers)},
            )
        )
    return rows


def _solution_verifier_rows(panel: Mapping[str, Any]) -> list[JsonDict]:
    if not panel:
        return []
    gain = _float_or_none(panel.get("verifier_gain_delta"))
    status = (
        "flagged"
        if panel.get("flagged_adversarial") is True or (gain is not None and gain <= 0)
        else "clean"
    )
    return [
        _row(
            row_id="solver:local_sota_solution_verifier_gain_panel",
            status=status,
            source_artifact=EXP3057_REL_PATH.as_posix(),
            source_field="verifier_gain_delta",
            evidence_class="solver_grounded_verification",
            claim_scope="local_sota_solution_verifier_gain",
            summary={
                "solution_verifier_calibration_ready": panel.get(
                    "solution_verifier_calibration_ready"
                )
                is True,
                "verifier_gain_delta": gain,
                "false_positive_rate": _float_or_none(panel.get("false_positive_rate")),
                "false_negative_rate": _float_or_none(panel.get("false_negative_rate")),
                "flagged_adversarial": panel.get("flagged_adversarial") is True,
            },
        )
    ]


def _aquaforte_rows(pilot: Mapping[str, Any]) -> list[JsonDict]:
    if not pilot:
        return []
    guidance = _as_mapping(pilot.get("guidance_vs_solver_only"))
    delta = _int_or_none(guidance.get("guided_minus_solver_only_success_count"))
    status = (
        "flagged"
        if pilot.get("flagged_adversarial") is True
        else "bounded"
        if delta == 0
        else "clean"
    )
    return [
        _row(
            row_id="solver:aquaforte_smt_pilot",
            status=status,
            source_artifact=EXP3058_REL_PATH.as_posix(),
            source_field="guidance_vs_solver_only.guided_minus_solver_only_success_count",
            evidence_class="solver_grounded_verification",
            claim_scope="llm_guided_smt_pilot",
            summary={
                "llm_guided_smt_pilot_ready": pilot.get("llm_guided_smt_pilot_ready") is True,
                "guided_success_count": _int_or_none(pilot.get("guided_success_count")),
                "solver_only_success_count": _int_or_none(pilot.get("solver_only_success_count")),
                "guided_minus_solver_only_success_count": delta,
                "flagged_adversarial": pilot.get("flagged_adversarial") is True,
            },
        )
    ]


def _gated_repair_rows(gate: Mapping[str, Any]) -> list[JsonDict]:
    if not gate:
        return []
    return [
        _row(
            row_id="repair:gated_sota_rerun",
            status="gated_skipped"
            if _status_from_gate_payload(gate) == "gated_skipped"
            else "blocked",
            source_artifact=EXP3059_ACTUAL_REL_PATH.as_posix(),
            source_field="gate_check_summary",
            evidence_class="repair_rerun_gate",
            claim_scope="repair_live_rerun",
            summary={
                "schema": str(gate.get("schema") or ""),
                "gate_check_summary": str(gate.get("gate_check_summary") or ""),
                "failed_gate_count": sum(
                    1
                    for row in _as_list(gate.get("gates_evaluated"))
                    if _as_mapping(row).get("passed") is not True
                ),
            },
        )
    ]


def _fr11_schema_rows(schema: Mapping[str, Any]) -> list[JsonDict]:
    if not schema:
        return []
    return [
        _row(
            row_id="fr11:solver_self_model_trace_schema",
            status="clean" if schema.get("solver_self_model_trace_ready") is True else "blocked",
            source_artifact=EXP3060_REL_PATH.as_posix(),
            source_field="solver_self_model_trace_ready",
            evidence_class="fr11_trace_schema",
            claim_scope="controller_only_self_learning_schema",
            summary={
                "allowed_edit_target_count": len(_as_list(schema.get("allowed_edit_targets")))
            },
        )
    ]


def _fr11_delayed_rows(pilot: Mapping[str, Any]) -> list[JsonDict]:
    if not pilot:
        return []
    substrate = _as_mapping(pilot.get("inference_substrate"))
    scope_violation = (
        substrate.get("model_weight_training") is True
        or substrate.get("model_weight_mutation") is True
    )
    if scope_violation:
        status = "blocked"
    elif pilot.get("flagged_adversarial") is True:
        status = "flagged"
    elif pilot.get("fr11_delayed_regression_ready") is True:
        status = "bounded"
    else:
        status = "blocked"
    return [
        _row(
            row_id="fr11:delayed_regression",
            status=status,
            source_artifact=EXP3061_REL_PATH.as_posix(),
            source_field="fr11_delayed_regression_ready",
            evidence_class="fr11_controller_self_learning",
            claim_scope="controller_only_delayed_regression",
            summary={
                "fr11_delayed_regression_ready": pilot.get("fr11_delayed_regression_ready") is True,
                "promotion_decision": str(pilot.get("promotion_decision") or ""),
                "edit_targets_used": _as_list(pilot.get("edit_targets_used")),
                "family_holdout_delta": _float_or_none(pilot.get("family_holdout_delta")),
                "delayed_regression_delta": _float_or_none(pilot.get("delayed_regression_delta")),
                "prior_retention_delta": _float_or_none(pilot.get("prior_retention_delta")),
                "flagged_adversarial": pilot.get("flagged_adversarial") is True,
                "model_weight_scope_violation": scope_violation,
            },
        )
    ]


def _kan_pwa_rows(audit: Mapping[str, Any]) -> list[JsonDict]:
    if not audit:
        return []
    exact_bound = audit.get("exact_controller_anchor_bound_available") is True
    promoted = (
        audit.get("kan_pwa_verification_ready") is True
        and audit.get("claim_promotion_useful") is True
    )
    status = "clean" if promoted else "bounded" if exact_bound else "blocked"
    return [
        _row(
            row_id="kan:pwa_locality_audit",
            status=status,
            source_artifact=EXP3062_REL_PATH.as_posix(),
            source_field="kan_pwa_verification_ready",
            evidence_class="kan_pwa_controller_anchor_audit",
            claim_scope="controller_locality_not_model_weight_verification",
            summary={
                "kan_pwa_verification_ready": audit.get("kan_pwa_verification_ready") is True,
                "claim_promotion_useful": audit.get("claim_promotion_useful") is True,
                "exact_controller_anchor_bound_available": exact_bound,
                "locality_bound": _float_or_none(audit.get("locality_bound")),
                "promotion_decision": str(audit.get("promotion_decision") or ""),
            },
        )
    ]


def _gatemate_rows(ledger: Mapping[str, Any]) -> list[JsonDict]:
    if not ledger:
        return []
    rerun_allowed = ledger.get("gatemate_rerun_allowed") is True
    return [
        _row(
            row_id="gatemate:no_rerun_ledger",
            status="clean" if rerun_allowed else "blocked",
            source_artifact=EXP3063_REL_PATH.as_posix(),
            source_field="gatemate_rerun_allowed",
            evidence_class="gatemate_operator_action_ledger",
            claim_scope="hardware_rerun_gate",
            summary={
                "gatemate_no_rerun_ledger_ready": ledger.get("gatemate_no_rerun_ledger_ready")
                is True,
                "gatemate_rerun_allowed": rerun_allowed,
                "downstream_blocked_count": len(_as_list(ledger.get("downstream_tasks_blocked"))),
                "missing_operator_action_count": len(
                    _as_list(ledger.get("missing_operator_actions"))
                ),
            },
        )
    ]


def _ssqa_rows(ledger: Mapping[str, Any]) -> list[JsonDict]:
    if not ledger:
        return []
    readback_allowed = ledger.get("ssqa_readback_allowed") is True
    status_text = str(ledger.get("ssqa_status") or "")
    status = (
        "clean" if readback_allowed else "gated_skipped" if "gated" in status_text else "blocked"
    )
    return [
        _row(
            row_id="ssqa:host_visible_readback_boundary",
            status=status,
            source_artifact=EXP3064_REL_PATH.as_posix(),
            source_field="ssqa_status",
            evidence_class="ssqa_readback_boundary",
            claim_scope="host_visible_readback_gate",
            summary={
                "ssqa_boundary_ledger_ready": ledger.get("ssqa_boundary_ledger_ready") is True,
                "ssqa_readback_allowed": readback_allowed,
                "ssqa_status": status_text,
                "host_visible_smoke_present": _as_mapping(
                    ledger.get("host_visible_smoke_evidence")
                ).get("present")
                is True,
            },
        )
    ]


def _row(
    *,
    row_id: str,
    status: str,
    source_artifact: str,
    source_field: str,
    evidence_class: str,
    claim_scope: str,
    summary: Mapping[str, Any] | None = None,
) -> JsonDict:
    row_status = normal_status(status)
    return {
        "row_id": row_id,
        "status": row_status,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence_class": evidence_class,
        "blocker_class": blocker_class(row_status),
        "claim_scope": claim_scope,
        "summary": dict(summary or {}),
    }


def normal_status(status: str) -> str:
    """Normalize legacy labels into the eight matrix v20 row classes."""

    normalized = status.replace("-", "_")
    if normalized == "gate_skipped":
        return "gated_skipped"
    if normalized == "pilot_only":
        return "bounded"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str) -> str:
    """Map one normalized row class to the publication-boundary reason class."""

    return {
        "clean": "none",
        "flagged": "adversarial_or_methodology_flag",
        "bounded": "bounded_claim",
        "blocked": "required_blocker",
        "gated_skipped": "structured_gate_skip",
        "projection_only": "projection_only",
        "missing": "missing_artifact",
        "retired": "retired_claim",
    }[normal_status(status)]


def _classify_rows(rows: list[JsonDict]) -> dict[str, list[JsonDict]]:
    classes = {field: [] for field in CLASS_FIELDS.values()}
    for row in rows:
        classes[CLASS_FIELDS[normal_status(str(row.get("status") or "missing"))]].append(row)
    return classes


def _publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_id": str(row.get("row_id") or ""),
            "status": normal_status(str(row.get("status") or "missing")),
            "blocker_class": str(row.get("blocker_class") or ""),
            "source_artifact": str(row.get("source_artifact") or ""),
            "source_field": str(row.get("source_field") or ""),
            "claim_scope": str(row.get("claim_scope") or ""),
        }
        for row in rows
        if normal_status(str(row.get("status") or "missing")) in PUBLICATION_BLOCKING_STATUSES
    ]


def _status_summaries(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    repair = payloads.get("exp3055", {})
    gate = payloads.get("exp3059", {})
    verifier = payloads.get("exp3057", {})
    smt = payloads.get("exp3058", {})
    fr11 = payloads.get("exp3061", {})
    kan = payloads.get("exp3062", {})
    gatemate = payloads.get("exp3063", {})
    ssqa = payloads.get("exp3064", {})
    return {
        "repair": {
            "status": "bounded_and_gated_skipped",
            "citations": [
                _citation(
                    EXP3055_REL_PATH, "repair_claim_status", repair.get("repair_claim_status")
                ),
                _citation(
                    EXP3059_ACTUAL_REL_PATH, "gate_check_summary", gate.get("gate_check_summary")
                ),
            ],
        },
        "solver_grounded_verification": {
            "status": "flagged_solver_grounded_no_gain",
            "citations": [
                _citation(
                    EXP3057_REL_PATH, "verifier_gain_delta", verifier.get("verifier_gain_delta")
                ),
                _citation(
                    EXP3058_REL_PATH,
                    "guidance_vs_solver_only.guided_minus_solver_only_success_count",
                    _as_mapping(smt.get("guidance_vs_solver_only")).get(
                        "guided_minus_solver_only_success_count"
                    ),
                ),
            ],
        },
        "fr11": {
            "status": "controller_only_delayed_regression_ready_flagged",
            "citations": [
                _citation(
                    EXP3061_REL_PATH,
                    "fr11_delayed_regression_ready",
                    fr11.get("fr11_delayed_regression_ready"),
                ),
                _citation(EXP3061_REL_PATH, "promotion_decision", fr11.get("promotion_decision")),
            ],
        },
        "kan_pwa": {
            "status": "bounded_controller_anchor_audit_not_promoted",
            "citations": [
                _citation(
                    EXP3062_REL_PATH,
                    "kan_pwa_verification_ready",
                    kan.get("kan_pwa_verification_ready"),
                ),
                _citation(
                    EXP3062_REL_PATH, "claim_promotion_useful", kan.get("claim_promotion_useful")
                ),
            ],
        },
        "gatemate": {
            "status": "blocked_no_rerun_operator_actions_required",
            "citations": [
                _citation(
                    EXP3063_REL_PATH,
                    "gatemate_rerun_allowed",
                    gatemate.get("gatemate_rerun_allowed"),
                ),
                _citation(
                    EXP3063_REL_PATH,
                    "missing_operator_actions",
                    len(_as_list(gatemate.get("missing_operator_actions"))),
                ),
            ],
        },
        "ssqa": {
            "status": "gated_skipped_host_visible_smoke_missing",
            "citations": [
                _citation(EXP3064_REL_PATH, "ssqa_status", ssqa.get("ssqa_status")),
                _citation(
                    EXP3064_REL_PATH, "ssqa_readback_allowed", ssqa.get("ssqa_readback_allowed")
                ),
            ],
        },
    }


def _citation(path: Path, source_field: str, value: Any) -> JsonDict:
    return {"source_artifact": path.as_posix(), "source_field": source_field, "value": value}


def _rows_machine_readable(rows: list[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        REQUIRED_ROW_KEYS <= set(row)
        and normal_status(str(row.get("status") or "missing")) in STATUSES
        and bool(row.get("source_artifact"))
        and bool(row.get("source_field"))
        for row in rows
    )


def _source_records_machine_readable(source_artifacts: list[Mapping[str, Any]]) -> bool:
    required = {"experiment_id", "path", "role", "required", "present", "readable_json_object"}
    return len(source_artifacts) == len(SOURCE_SPECS) and all(
        required <= set(row) and row.get("machine_readable_record") is True
        for row in source_artifacts
    )


def _status_from_gate_payload(payload: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(payload.get("schema") or ""),
            str(payload.get("status") or ""),
            str(payload.get("blocked_at_layer") or ""),
            str(payload.get("honest_verdict") or ""),
            str(payload.get("gate_check_summary") or ""),
        ]
    ).lower()
    if "gate" in text and ("blocked" in text or "failed" in text):
        return "gated_skipped"
    return "blocked"


def _honest_verdict(
    ready: bool,
    rows_total: int,
    publication_blockers: list[Mapping[str, Any]],
    required_errors: list[Mapping[str, Any]],
) -> str:
    if required_errors:
        return (
            "blocked_matrix_v20_preconditions: "
            f"required_source_errors={len(required_errors)}; rows_total={rows_total}"
        )
    if not ready:
        return f"blocked_matrix_v20_preconditions: rows_machine_readable=false; rows_total={rows_total}"
    return (
        "complete: "
        f"matrix_v20_ready=true; rows_total={rows_total}; "
        f"publication_blocker_count={len(publication_blockers)}; paper_ready=false"
    )


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_upstream_artifacts",
        "source": "checked_in_artifacts",
        "executes_models": False,
        "executes_hardware": False,
        "executes_conductor": False,
        "executes_live_repair": False,
        "no_live_llm_inference": True,
    }


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
    "CAPSTONE_V285_REL_PATH",
    "EXP3054_REL_PATH",
    "EXP3055_REL_PATH",
    "EXP3056_REL_PATH",
    "EXP3057_REL_PATH",
    "EXP3058_REL_PATH",
    "EXP3059_ACTUAL_REL_PATH",
    "EXP3059_REQUESTED_REL_PATH",
    "EXP3060_REL_PATH",
    "EXP3061_REL_PATH",
    "EXP3062_REL_PATH",
    "EXP3063_REL_PATH",
    "EXP3064_REL_PATH",
    "MATRIX_V19_REL_PATH",
    "OUTPUT_REL_PATH",
    "REPO_ROOT",
    "SCRIPT_REL_PATH",
    "blocker_class",
    "build_artifact",
    "normal_status",
    "read_json_object",
    "sha256_file",
    "write_artifact",
]
