"""Build the Exp 3175 cross-corpus matrix v28 artifact.

Spec refs: REQ-REPORT-3175, SCENARIO-REPORT-3175.

Matrix v28 is a checked-in evidence ledger over the `.294` milestone. Its
main job is to make blocked and gated-skip work visible as rows, so the next
capstone can distinguish "we proved this", "we skipped this because a gate
failed", and "the artifact is genuinely absent" without rerunning models,
repairs, solvers, hardware, or the conductor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from carnot.reporting import cross_corpus_matrix_v27_3161 as v27


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.294"
SCHEMA = "carnot.cross_corpus_matrix.v28_294_artifact_aggregation.v1"
ARTIFACT = "experiment_3175_cross_corpus_matrix_v28"
OUTPUT_REL_PATH = Path("results/experiment_3175_cross_corpus_matrix_v28.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3175_cross_corpus_matrix_v28.py"

MATRIX_V27_REL_PATH = Path("results/experiment_3161_cross_corpus_matrix_v27.json")
CAPSTONE_V293_REL_PATH = Path("results/experiment_3162_capstone_v293.json")
EXP3163_REL_PATH = Path("results/experiment_3163_archive_v293_activate_v294.json")
EXP3164_REL_PATH = Path("results/experiment_3164_duration_corrected_authenticity_contract_v2.json")
EXP3165_REL_PATH = Path("results/experiment_3165_live_sota_authenticity_replay_v2.json")
EXP3166_REL_PATH = Path(
    "results/experiment_3166_verifier_invariance_token_suspicion_audit_v1.json"
)
EXP3167_REL_PATH = Path("results/experiment_3167_clean_live_sota_verifier_rerun_v9.json")
EXP3168_REL_PATH = Path("results/experiment_3168_repair_gate_decision_v3.json")
EXP3169_REL_PATH = Path("results/experiment_3169_repair_ladder_materializer_v4.json")
EXP3170_REL_PATH = Path("results/experiment_3170_counterexample_certificate_repair_pilot_v2.json")
EXP3171_REL_PATH = Path("results/experiment_3171_fr11_ledger_counterexample_isolation_v1.json")
EXP3172_REL_PATH = Path("results/experiment_3172_fr11_nonforgetting_self_learning_pilot_v2.json")
EXP3173_REL_PATH = Path("results/experiment_3173_ebcn_kan_bounded_diagnostic_expansion_v2.json")
EXP3174_REL_PATH = Path("results/experiment_3174_hardware_tooling_boundary_v8.json")

STATUSES = (
    "clean",
    "blocked",
    "gated_skipped",
    "flagged",
    "diagnostic_only",
    "projection_only",
    "missing",
    "retired",
)
PUBLICATION_BLOCKING_STATUSES = {
    "blocked",
    "gated_skipped",
    "flagged",
    "projection_only",
    "missing",
}
PROJECTION_BOUNDARY_TOKENS = (
    "architecture",
    "sidecar",
    "future_adapter",
    "energy",
    "kan",
)

read_json_object = v27.read_json_object
sha256_file = v27.sha256_file
_as_mapping = v27._as_mapping
_as_list = v27._as_list
_text_list = v27._text_list
_int_or_none = v27._int_or_none
_float_or_none = v27._float_or_none


@dataclass(frozen=True)
class SourceSpec:
    """A JSON artifact that v28 reads, checksums, and cites without mutation."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    ready_field: str = ""


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3161", MATRIX_V27_REL_PATH, "matrix_v27_authority", True, "matrix_v27_ready"),
    SourceSpec("exp3162", CAPSTONE_V293_REL_PATH, "capstone_v293_authority", True, "capstone_ready"),
    SourceSpec(
        "exp3163",
        EXP3163_REL_PATH,
        "archive_v293_activate_v294",
        False,
        "archive_v293_activate_v294_ready",
    ),
    SourceSpec(
        "exp3164",
        EXP3164_REL_PATH,
        "duration_corrected_authenticity_contract",
        False,
        "duration_corrected_authenticity_contract_v2_ready",
    ),
    SourceSpec(
        "exp3165",
        EXP3165_REL_PATH,
        "live_sota_authenticity_replay",
        False,
        "live_sota_authenticity_replay_v2_ready",
    ),
    SourceSpec(
        "exp3166",
        EXP3166_REL_PATH,
        "verifier_invariance_token_suspicion_audit",
        False,
        "verifier_invariance_token_suspicion_audit_ready",
    ),
    SourceSpec(
        "exp3167",
        EXP3167_REL_PATH,
        "clean_live_verifier_rerun",
        False,
        "clean_live_verifier_rerun_v9_ready",
    ),
    SourceSpec(
        "exp3168", EXP3168_REL_PATH, "repair_gate_v3", False, "repair_gate_decision_v3_ready"
    ),
    SourceSpec(
        "exp3169",
        EXP3169_REL_PATH,
        "repair_ladder_v4",
        False,
        "repair_ladder_materializer_v4_ready",
    ),
    SourceSpec(
        "exp3170",
        EXP3170_REL_PATH,
        "counterexample_certificate_repair",
        False,
        "counterexample_certificate_repair_pilot_v2_ready",
    ),
    SourceSpec(
        "exp3171",
        EXP3171_REL_PATH,
        "fr11_ledger_counterexample_isolation",
        False,
        "fr11_ledger_counterexample_isolation_ready",
    ),
    SourceSpec(
        "exp3172",
        EXP3172_REL_PATH,
        "fr11_nonforgetting_self_learning",
        False,
        "fr11_nonforgetting_self_learning_pilot_v2_ready",
    ),
    SourceSpec(
        "exp3173",
        EXP3173_REL_PATH,
        "ebcn_kan_bounded_diagnostics",
        False,
        "ebcn_kan_bounded_diagnostic_expansion_v2_ready",
    ),
    SourceSpec(
        "exp3174",
        EXP3174_REL_PATH,
        "hardware_tooling_boundary",
        False,
        "hardware_tooling_boundary_v8_ready",
    ),
)

V27_MISSING_MATERIALIZERS = {
    "results/experiment_3152_clean_live_sota_verifier_rerun_v8.json": "exp3167",
    "results/experiment_3154_multi_turn_repair_ladder_v3.json": "exp3169",
    "results/experiment_3155_tracefix_counterexample_repair_pilot_v1.json": "exp3170",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3175: aggregate matrix v28 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3161"]
    capstone = payloads["exp3162"]
    rows = _carry_forward_rows(matrix) + _dot294_rows(payloads) if matrix else []
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    prior_count = _prior_publication_blocker_count(matrix, capstone)
    missing_artifacts, missing_comparison = _missing_artifacts(matrix, sources)
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
    paper_implications = _paper_readiness_implications(rows, len(publication_blockers), payloads)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_v28_ready": ready,
        "rows_total": len(rows),
        "status_counts": status_counts,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v27": len(publication_blockers) - prior_count,
        "clean_rows": status_counts["clean"],
        "flagged_rows": status_counts["flagged"],
        "blocked_rows": status_counts["blocked"],
        "gated_skip_rows": status_counts["gated_skipped"],
        "diagnostic_only_rows": status_counts["diagnostic_only"],
        "projection_only_rows": status_counts["projection_only"],
        "missing_rows": status_counts["missing"],
        "retired_rows": status_counts["retired"],
        "missing_artifacts": missing_artifacts,
        "missing_artifact_comparison": missing_comparison,
        "inherited_adversarial_flag_count": _inherited_adversarial_flag_count(matrix),
        "verifier_status": _verifier_status(payloads, rows),
        "repair_status": _repair_status(payloads, rows),
        "fr11_status": _fr11_status(payloads, rows),
        "sidecar_status": _sidecar_status(payloads, rows),
        "hardware_status": _hardware_status(payloads, rows),
        "paper_ready": paper_implications["paper_ready"],
        "publication_blockers": publication_blockers,
        "rows": rows,
        "paper_readiness_implications": paper_implications,
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
    """Build and persist the Exp 3175 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def normal_status(status: str, claim_scope: str = "", evidence_class: str = "") -> str:
    """Normalize historical row statuses into the v28 closed vocabulary."""

    normalized = str(status or "").strip()
    if normalized == "model_spec_gap":
        return "blocked"
    if normalized == "bounded":
        boundary = f"{claim_scope} {evidence_class}".lower()
        if any(token in boundary for token in PROJECTION_BOUNDARY_TOKENS):
            return "projection_only"
        return "blocked"
    return normalized if normalized in STATUSES else "missing"


def blocker_class(status: str, claim_scope: str = "", evidence_class: str = "") -> str:
    """Map a row status to the class used by publication summaries."""

    normalized = normal_status(status, claim_scope, evidence_class)
    if normalized == "clean":
        return "none"
    if normalized == "retired":
        return "retired_non_blocking"
    if normalized == "diagnostic_only":
        return "diagnostic_non_blocking"
    return f"publication_blocker_{normalized}"


def _source_payload(root: Path, spec: SourceSpec) -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "loaded_path": spec.path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "ready_field": spec.ready_field,
        "source_type": "json",
        "present": path.is_file(),
        "primary_present": path.is_file(),
        "readable_json_object": bool(payload),
        "alias_loaded": False,
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
        source_status = str(raw.get("status") or "missing")
        if source_status != row["status"]:
            summary["v28_status_rationale"] = f"normalized_from_v27_{source_status}"
        else:
            summary.setdefault("v28_status_rationale", "carried_forward_from_matrix_v27")
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v27")
        rows.append(row)
    return rows


def _dot294_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _archive_row(payloads["exp3163"]),
        _contract_row(payloads["exp3164"]),
        _live_sota_row(payloads["exp3165"]),
        _invariance_audit_row(payloads["exp3166"]),
        _clean_rerun_row(payloads["exp3167"]),
        _repair_gate_row(payloads["exp3168"]),
        _repair_ladder_row(payloads["exp3169"]),
        _certificate_repair_row(payloads["exp3170"]),
        _fr11_isolation_row(payloads["exp3171"]),
        _fr11_nonforgetting_row(payloads["exp3172"]),
        _sidecar_row(payloads["exp3173"]),
        _hardware_row(payloads["exp3174"]),
    ]


def _archive_row(payload: Mapping[str, Any]) -> JsonDict:
    status = _ready_status(payload, "archive_v293_activate_v294_ready")
    return _row(
        row_id="dot294:exp3163_archive_handoff",
        status=status,
        source_artifact=EXP3163_REL_PATH.as_posix(),
        source_field="archive_v293_activate_v294_ready",
        evidence_class="archive_v293_activate_v294",
        claim_scope="milestone_handoff",
        summary={
            "archive_v293_activate_v294_ready": payload.get("archive_v293_activate_v294_ready")
            is True,
            "prior_publication_blocker_count": _int_or_none(
                payload.get("prior_publication_blocker_count")
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _contract_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("duration_corrected_authenticity_contract_v2_ready") is not True:
        status = "blocked"
    elif payload.get("flagged_adversarial") is True or _as_list(payload.get("corrigendum_pending")):
        status = "flagged"
    else:
        status = "clean"
    return _row(
        row_id="dot294:exp3164_duration_contract",
        status=status,
        source_artifact=EXP3164_REL_PATH.as_posix(),
        source_field="flagged_adversarial",
        evidence_class="duration_corrected_authenticity_contract_v2",
        claim_scope="live_verifier_authenticity_contract",
        summary={
            "duration_corrected_authenticity_contract_v2_ready": payload.get(
                "duration_corrected_authenticity_contract_v2_ready"
            )
            is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "old_fixed_duration_rule_retired_as_hard_gate": payload.get(
                "old_fixed_duration_rule_retired_as_hard_gate"
            )
            is True,
            "observed_source_assessment_passed": _as_mapping(
                payload.get("observed_source_assessment")
            ).get("passed")
            is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _live_sota_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("live_sota_authenticity_replay_v2_ready") is True
        and payload.get("preflight_passed") is True
        and payload.get("headline_claim_allowed") is True
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot294:exp3165_live_sota_replay",
        status=status,
        source_artifact=EXP3165_REL_PATH.as_posix(),
        source_field="preflight_passed",
        evidence_class="live_sota_authenticity_replay_v2",
        claim_scope="live_sota_authenticity_preflight",
        summary={
            "live_sota_authenticity_replay_v2_ready": payload.get(
                "live_sota_authenticity_replay_v2_ready"
            )
            is True,
            "preflight_passed": payload.get("preflight_passed") is True,
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "measured_work_policy_passed": payload.get("measured_work_policy_passed") is True,
            "blocked_reason": str(payload.get("blocked_reason") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _invariance_audit_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("verifier_invariance_token_suspicion_audit_ready") is not True
        or _as_list(payload.get("source_errors"))
    ):
        status = "blocked"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot294:exp3166_invariance_audit",
        status=status,
        source_artifact=EXP3166_REL_PATH.as_posix(),
        source_field="verifier_invariance_token_suspicion_audit_ready",
        evidence_class="verifier_invariance_token_suspicion_audit",
        claim_scope="exact_authority_diagnostic",
        summary={
            "verifier_invariance_token_suspicion_audit_ready": payload.get(
                "verifier_invariance_token_suspicion_audit_ready"
            )
            is True,
            "trusted_exact_row_count": len(_as_list(payload.get("trusted_exact_rows"))),
            "blocked_check_count": len(_as_list(payload.get("blocked_checks"))),
            "source_error_count": len(_as_list(payload.get("source_errors"))),
            "diagnostic_acceptance_authority": False,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _clean_rerun_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("clean_live_verifier_rerun_v9_ready") is not True:
        status = "blocked"
    elif payload.get("gated_skip") is True:
        status = "gated_skipped"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif (
        payload.get("controlled_invariance_passed") is True
        and payload.get("false_accept_gate_passed") is True
        and payload.get("headline_claim_allowed") is True
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot294:exp3167_clean_live_rerun",
        status=status,
        source_artifact=EXP3167_REL_PATH.as_posix(),
        source_field="clean_live_verifier_rerun_v9_ready",
        evidence_class="clean_live_sota_verifier_rerun_v9",
        claim_scope="live_verifier_headline_evidence",
        summary={
            "clean_live_verifier_rerun_v9_ready": payload.get(
                "clean_live_verifier_rerun_v9_ready"
            )
            is True,
            "gated_skip": payload.get("gated_skip") is True,
            "gated_skip_reason": str(payload.get("gated_skip_reason") or ""),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "controlled_invariance_passed": payload.get("controlled_invariance_passed") is True,
            "false_accept_gate_passed": payload.get("false_accept_gate_passed") is True,
            "false_accept_rate": _float_or_none(payload.get("false_accept_rate")),
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "exact_ground_truth_count": _int_or_none(payload.get("exact_ground_truth_count")) or 0,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_gate_row(payload: Mapping[str, Any]) -> JsonDict:
    gate_state = str(payload.get("repair_gate_state") or "")
    if not payload:
        status = "missing"
    elif payload.get("repair_gate_decision_v3_ready") is True and gate_state == "unblocked":
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot294:exp3168_repair_gate_v3",
        status=status,
        source_artifact=EXP3168_REL_PATH.as_posix(),
        source_field="repair_gate_state",
        evidence_class="repair_gate_decision_v3",
        claim_scope="repair_gate_decision",
        summary={
            "repair_gate_decision_v3_ready": payload.get("repair_gate_decision_v3_ready") is True,
            "repair_gate_state": gate_state,
            "gated_skip": payload.get("gated_skip") is True,
            "repair_blocker_count": len(_as_list(payload.get("repair_blockers"))),
            "selected_repair_row_count": len(_as_list(payload.get("selected_repair_rows"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_ladder_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("repair_ladder_materializer_v4_ready") is not True:
        status = "blocked"
    elif payload.get("gated_skip") is True:
        status = "gated_skipped"
    elif payload.get("headline_repair_claim_allowed") is True:
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot294:exp3169_repair_ladder_v4",
        status=status,
        source_artifact=EXP3169_REL_PATH.as_posix(),
        source_field="repair_ladder_materializer_v4_ready",
        evidence_class="repair_ladder_materializer_v4",
        claim_scope="repair_execution",
        summary={
            "repair_ladder_materializer_v4_ready": payload.get(
                "repair_ladder_materializer_v4_ready"
            )
            is True,
            "gated_skip": payload.get("gated_skip") is True,
            "gated_skip_reason": str(payload.get("gated_skip_reason") or ""),
            "gate_state": str(payload.get("gate_state") or ""),
            "headline_repair_claim_allowed": payload.get("headline_repair_claim_allowed") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "repair_attempt_count": _int_or_none(payload.get("repair_attempt_count")) or 0,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _certificate_repair_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("counterexample_certificate_repair_pilot_v2_ready") is not True:
        status = "blocked"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif payload.get("repair_call_required_for_next_step") is True:
        status = "blocked"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot294:exp3170_certificate_repair",
        status=status,
        source_artifact=EXP3170_REL_PATH.as_posix(),
        source_field="counterexample_certificate_repair_pilot_v2_ready",
        evidence_class="counterexample_certificate_repair_pilot_v2",
        claim_scope="formal_counterexample_repair_certificate",
        summary={
            "counterexample_certificate_repair_pilot_v2_ready": payload.get(
                "counterexample_certificate_repair_pilot_v2_ready"
            )
            is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "counterexample_count": _int_or_none(payload.get("counterexample_count")) or 0,
            "exact_accept_count": _int_or_none(payload.get("exact_accept_count")) or 0,
            "exact_row_count": _int_or_none(payload.get("exact_row_count")) or 0,
            "prior_repair_candidates_scored": _int_or_none(
                payload.get("prior_repair_candidates_scored")
            )
            or 0,
            "repair_call_required_for_next_step": payload.get("repair_call_required_for_next_step")
            is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_isolation_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("fr11_ledger_counterexample_isolation_ready") is not True:
        status = "blocked"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot294:exp3171_fr11_counterexample_isolation",
        status=status,
        source_artifact=EXP3171_REL_PATH.as_posix(),
        source_field="fr11_ledger_counterexample_isolation_ready",
        evidence_class="fr11_ledger_counterexample_isolation",
        claim_scope="fr11_controller_memory_diagnostic",
        summary={
            "fr11_ledger_counterexample_isolation_ready": payload.get(
                "fr11_ledger_counterexample_isolation_ready"
            )
            is True,
            "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
            "prior_ledger_consistency_rate": _float_or_none(
                payload.get("prior_ledger_consistency_rate")
            ),
            "ledger_consistent_count": _int_or_none(payload.get("ledger_consistent_count")) or 0,
            "ledger_inconsistent_count": _int_or_none(payload.get("ledger_inconsistent_count"))
            or 0,
            "replay_panel_count": _int_or_none(payload.get("replay_panel_count")) or 0,
            "no_model_weight_update_claimed": payload.get("no_model_weight_update_claimed") is True,
            "promotion_allowed": payload.get("promotion_allowed") is True,
            "isolated_counterexample_family_count": len(
                _as_list(payload.get("isolated_counterexample_families"))
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_nonforgetting_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("fr11_nonforgetting_self_learning_pilot_v2_ready") is True
        and payload.get("nonforgetting_passed") is True
        and payload.get("promotion_allowed") is True
        and payload.get("model_weight_update_claimed") is not True
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot294:exp3172_fr11_nonforgetting",
        status=status,
        source_artifact=EXP3172_REL_PATH.as_posix(),
        source_field="fr11_nonforgetting_self_learning_pilot_v2_ready",
        evidence_class="fr11_nonforgetting_self_learning_pilot_v2",
        claim_scope="fr11_controller_memory_promotion",
        summary={
            "fr11_nonforgetting_self_learning_pilot_v2_ready": payload.get(
                "fr11_nonforgetting_self_learning_pilot_v2_ready"
            )
            is True,
            "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
            "before_ledger_consistency_rate": _float_or_none(
                payload.get("before_ledger_consistency_rate")
            ),
            "after_ledger_consistency_rate": _float_or_none(
                payload.get("after_ledger_consistency_rate")
            ),
            "heldout_consistency_rate": _float_or_none(payload.get("heldout_consistency_rate")),
            "nonforgetting_passed": payload.get("nonforgetting_passed") is True,
            "controller_memory_update_applied": payload.get("controller_memory_update_applied")
            is True,
            "model_weight_update_claimed": payload.get("model_weight_update_claimed") is True,
            "promotion_allowed": payload.get("promotion_allowed") is True,
            "promotion_recommendation": str(payload.get("promotion_recommendation") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _sidecar_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("ebcn_kan_bounded_diagnostic_expansion_v2_ready") is not True:
        status = "blocked"
    elif (
        payload.get("live_integration_claim_allowed") is True
        and payload.get("deployed_verifier_claim_allowed") is True
    ):
        status = "clean"
    else:
        status = "projection_only"
    return _row(
        row_id="dot294:exp3173_ebcn_kan_diagnostics",
        status=status,
        source_artifact=EXP3173_REL_PATH.as_posix(),
        source_field="live_integration_claim_allowed",
        evidence_class="ebcn_kan_bounded_diagnostic_expansion_v2",
        claim_scope="architecture_energy_kan_sidecar_boundary",
        summary={
            "ebcn_kan_bounded_diagnostic_expansion_v2_ready": payload.get(
                "ebcn_kan_bounded_diagnostic_expansion_v2_ready"
            )
            is True,
            "live_integration_claim_allowed": payload.get("live_integration_claim_allowed") is True,
            "deployed_verifier_claim_allowed": payload.get("deployed_verifier_claim_allowed")
            is True,
            "exact_labeled_row_count": _int_or_none(payload.get("exact_labeled_row_count")) or 0,
            "known_false_accept_rows_scored": _int_or_none(
                payload.get("known_false_accept_rows_scored")
            )
            or 0,
            "kan_monitor_record_count": _int_or_none(payload.get("kan_monitor_record_count")) or 0,
            "promotion_blocker_count": len(_as_list(payload.get("promotion_blockers"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _hardware_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("hardware_tooling_boundary_v8_ready") is True
        and payload.get("authenticated_speedup_claim_allowed") is True
        and payload.get("speedup_claim_made") is True
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot294:exp3174_hardware_tooling",
        status=status,
        source_artifact=EXP3174_REL_PATH.as_posix(),
        source_field="authenticated_speedup_claim_allowed",
        evidence_class="hardware_tooling_boundary_v8",
        claim_scope="hardware_speedup_boundary",
        summary={
            "hardware_tooling_boundary_v8_ready": payload.get("hardware_tooling_boundary_v8_ready")
            is True,
            "authenticated_speedup_claim_allowed": payload.get(
                "authenticated_speedup_claim_allowed"
            )
            is True,
            "speedup_claim_made": payload.get("speedup_claim_made") is True,
            "hardware_commands_run": _as_list(payload.get("hardware_commands_run")),
            "cuda_status": str(payload.get("cuda_status") or ""),
            "kv260_status": str(payload.get("kv260_status") or ""),
            "gatemate_status": str(payload.get("gatemate_status") or ""),
            "polarfire_status": str(payload.get("polarfire_status") or ""),
            "extropic_thrml_status": str(payload.get("extropic_thrml_status") or ""),
            "kona_status": str(payload.get("kona_status") or ""),
            "missing_required_source_artifacts": _text_list(
                payload.get("missing_required_source_artifacts")
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _claim_entry(row: Mapping[str, Any]) -> JsonDict:
    claim_scope = str(row.get("claim_scope") or "")
    evidence_class = str(row.get("evidence_class") or "")
    status = normal_status(str(row.get("status") or "missing"), claim_scope, evidence_class)
    return {
        "row_id": str(row.get("row_id") or ""),
        "status": status,
        "source_artifact": str(row.get("source_artifact") or ""),
        "source_field": str(row.get("source_field") or ""),
        "evidence_class": evidence_class,
        "blocker_class": blocker_class(status, claim_scope, evidence_class),
        "claim_scope": claim_scope,
        "summary": _as_mapping(row.get("summary")),
        "row_origin": str(row.get("row_origin") or "matrix_v27"),
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
    normalized = normal_status(status, claim_scope, evidence_class)
    return {
        "row_id": row_id,
        "status": normalized,
        "source_artifact": source_artifact,
        "source_field": source_field,
        "evidence_class": evidence_class,
        "blocker_class": blocker_class(normalized, claim_scope, evidence_class),
        "claim_scope": claim_scope,
        "summary": dict(summary),
        "row_origin": "milestone_294",
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
) -> tuple[list[JsonDict], JsonDict]:
    missing: list[JsonDict] = []
    materialized: list[str] = []
    source_present = {
        str(row["experiment_id"]): row.get("readable_json_object") is True for row in sources
    }
    v27_missing = [row for row in _as_list(matrix.get("missing_artifacts")) if isinstance(row, Mapping)]
    for row in v27_missing:
        path = str(row.get("path") or "")
        materializer = V27_MISSING_MATERIALIZERS.get(path)
        if materializer and source_present.get(materializer):
            materialized.append(path)
            continue
        missing.append(
            {
                "path": path,
                "experiment_id": str(row.get("experiment_id") or ""),
                "reason": "carried_forward_unresolved_missing_artifact_from_v27",
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
                    "reason": "missing_expected_dot294_artifact",
                }
            )
    comparison = {
        "v27_missing_artifact_count": len(v27_missing),
        "v28_missing_artifact_count": len(missing),
        "missing_artifact_delta_from_v27": len(missing) - len(v27_missing),
        "materialized_v27_missing_artifacts": materialized,
    }
    return missing, comparison


def _required_source_errors(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {"path": str(row["path"]), "reason": "missing_or_malformed_required_artifact"}
        for row in sources
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _inherited_adversarial_flag_count(matrix: Mapping[str, Any]) -> int:
    return _int_or_none(matrix.get("inherited_adversarial_flag_count")) or 0


def _verifier_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    rerun = payloads["exp3167"]
    replay = payloads["exp3165"]
    if statuses.get("dot294:exp3167_clean_live_rerun") == "missing":
        return "missing_clean_live_verifier_rerun_v9"
    if rerun.get("gated_skip") is True:
        return "gated_skip_preflight_failed_flagged_adversarial_exact_authority_only"
    if rerun.get("flagged_adversarial") is True:
        return "flagged_adversarial_clean_rerun_not_headline_safe"
    if statuses.get("dot294:exp3167_clean_live_rerun") == "clean":
        return "clean_live_verifier_ready"
    if replay.get("preflight_passed") is False:
        return "blocked_live_sota_replay_preflight_failed"
    return "blocked_live_verifier_not_headline_safe"


def _repair_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    gate = payloads["exp3168"]
    ladder = payloads["exp3169"]
    gate_state = str(gate.get("repair_gate_state") or gate.get("status") or "")
    if statuses.get("dot294:exp3168_repair_gate_v3") == "missing":
        return "missing_repair_gate_decision_v3"
    if (
        gate_state.startswith("blocked_flagged")
        and ladder.get("gated_skip") is True
        and statuses.get("dot294:exp3170_certificate_repair") == "flagged"
    ):
        return "blocked_flagged_verifier_repair_ladder_gated_skipped_certificate_pilot_flagged"
    if statuses.get("dot294:exp3168_repair_gate_v3") == "clean" and statuses.get(
        "dot294:exp3169_repair_ladder_v4"
    ) == "clean":
        return "repair_ready"
    if ladder.get("gated_skip") is True:
        return "blocked_repair_gate_ladder_gated_skipped"
    return "blocked_repair_gate"


def _fr11_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    nonforgetting = payloads["exp3172"]
    isolation = payloads["exp3171"]
    if statuses.get("dot294:exp3172_fr11_nonforgetting") == "clean":
        return "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
    if isolation.get("fr11_ledger_counterexample_isolation_ready") is True:
        return "blocked_fr11_promotion_counterexamples_isolated"
    return "missing_or_blocked_fr11_nonforgetting_evidence"


def _sidecar_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    sidecar = payloads["exp3173"]
    if statuses.get("dot294:exp3173_ebcn_kan_diagnostics") == "clean":
        return "clean_ebcn_kan_live_integration_and_deployed_verifier_allowed"
    if sidecar.get("ebcn_kan_bounded_diagnostic_expansion_v2_ready") is True:
        return "projection_only_ebcn_kan_diagnostics_no_live_integration_or_deployed_verifier"
    return "missing_or_blocked_ebcn_kan_diagnostic_evidence"


def _hardware_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    hardware = payloads["exp3174"]
    if statuses.get("dot294:exp3174_hardware_tooling") == "clean":
        return "clean_authenticated_speedup_claim_present"
    if hardware.get("hardware_tooling_boundary_v8_ready") is True:
        speedup_claim = "speedup_claim_made" if hardware.get("speedup_claim_made") else "no_speedup_claim_made"
        command_status = (
            "no_hardware_commands"
            if not _as_list(hardware.get("hardware_commands_run"))
            else "hardware_commands_present"
        )
        return f"blocked_no_authenticated_speedup_{command_status}_{speedup_claim}"
    return "missing_or_blocked_hardware_tooling_boundary"


def _paper_readiness_implications(
    rows: list[Mapping[str, Any]],
    publication_blocker_count: int,
    payloads: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    blocked: list[str] = []
    if statuses.get("dot294:exp3167_clean_live_rerun") != "clean":
        blocked.append("live_verifier_headline")
    if (
        statuses.get("dot294:exp3168_repair_gate_v3") != "clean"
        or statuses.get("dot294:exp3169_repair_ladder_v4") != "clean"
    ):
        blocked.append("repair_headline")
    if _fr11_status(payloads, rows) != (
        "controller_memory_update_promotable_nonforgetting_passed_no_model_weight_update"
    ):
        blocked.append("fr11_controller_memory_self_learning")
    if statuses.get("dot294:exp3173_ebcn_kan_diagnostics") != "clean":
        blocked.append("energy_sidecar_live_integration")
        blocked.append("kan_deployed_verifier")
    if statuses.get("dot294:exp3174_hardware_tooling") != "clean":
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
    if matrix and matrix.get("matrix_v27_ready") is not True:
        violations.append("matrix v27 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone v293 authority is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v28 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to rows_total")
    if len(publication_blockers) != sum(
        count for status, count in status_counts.items() if status in PUBLICATION_BLOCKING_STATUSES
    ):
        violations.append("publication_blocker_count does not match row statuses")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot294_artifacts",
        "source": "matrix_v27_capstone_v293_and_dot294_artifacts",
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
    if artifact.get("matrix_v28_ready") is not True:
        return (
            "blocked_matrix_v28_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"invariant_violations={len(_as_list(artifact.get('invariant_violations')))}"
        )
    comparison = _as_mapping(artifact.get("missing_artifact_comparison"))
    return (
        "complete: matrix_v28_ready=true; "
        f"rows_total={artifact.get('rows_total')}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v27={artifact.get('blocker_delta_from_v27')}; "
        f"inherited_adversarial_flag_count={artifact.get('inherited_adversarial_flag_count')}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}; "
        f"missing_artifact_delta_from_v27={comparison.get('missing_artifact_delta_from_v27')}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
