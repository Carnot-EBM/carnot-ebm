"""Build the Exp 3147 cross-corpus matrix v26 artifact.

Spec refs: REQ-REPORT-3147, SCENARIO-REPORT-3147.

Matrix v26 is a checked-in artifact aggregator. It reads the .291 matrix and
capstone plus the .292 evidence files, then records the status each source can
honestly support. It does not rerun models, repairs, solvers, the conductor, or
hardware; this matters because several .292 artifacts contain useful evidence
but also carry adversarial flags, missing integration, or bounded-only claim
boundaries that must remain visible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from carnot.reporting import cross_corpus_matrix_v25_3133 as v25


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
MILESTONE = "2026.05.292"
SCHEMA = "carnot.cross_corpus_matrix.v26_292_artifact_aggregation.v1"
ARTIFACT = "experiment_3147_cross_corpus_matrix_v26"
OUTPUT_REL_PATH = Path("results/experiment_3147_cross_corpus_matrix_v26.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3147_cross_corpus_matrix_v26.py"

MATRIX_V25_REL_PATH = Path("results/experiment_3133_cross_corpus_matrix_v25.json")
CAPSTONE_V291_REL_PATH = Path("results/experiment_3134_capstone_v291.json")
EXP3135_REL_PATH = Path("results/experiment_3135_archive_v291_activate_v292.json")
EXP3136_REL_PATH = Path("results/experiment_3136_false_accept_root_cause_autopsy_v1.json")
EXP3137_REL_PATH = Path("results/experiment_3137_exact_safe_accept_abstain_contract_v1.json")
EXP3138_REL_PATH = Path(
    "results/experiment_3138_canonical_answer_vericot_grounding_pilot_v1.json"
)
EXP3139_REL_PATH = Path("results/experiment_3139_live_sota_verifier_rerun_v7.json")
EXP3140_REL_PATH = Path("results/experiment_3140_repair_gate_unlock_decision_v1.json")
EXP3141_REL_PATH = Path("results/experiment_3141_multi_turn_repair_ladder_v2.json")
EXP3142_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3143_REL_PATH = Path(
    "results/experiment_3143_fr11_experience_driven_verifier_memory_v1.json"
)
EXP3144_REL_PATH = Path(
    "results/experiment_3144_ebt_arm_false_accept_calibration_boundary_v3.json"
)
EXP3145_REL_PATH = Path("results/experiment_3145_kan_proof_carrying_monitor_boundary_v2.json")
EXP3146_REL_PATH = Path("results/experiment_3146_hardware_sampler_evidence_boundary_v6.json")

STATUSES = v25.STATUSES
PUBLICATION_BLOCKING_STATUSES = v25.PUBLICATION_BLOCKING_STATUSES
read_json_object = v25.read_json_object
sha256_file = v25.sha256_file
normal_status = v25.normal_status
blocker_class = v25.blocker_class
_as_mapping = v25._as_mapping
_as_list = v25._as_list
_text_list = v25._text_list
_int_or_none = v25._int_or_none
_float_or_none = v25._float_or_none


@dataclass(frozen=True)
class SourceSpec:
    """One JSON artifact that v26 reads and cites without mutating it."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    ready_field: str = ""


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3133", MATRIX_V25_REL_PATH, "matrix_v25_authority", True, "matrix_v25_ready"),
    SourceSpec("exp3134", CAPSTONE_V291_REL_PATH, "capstone_v291_authority", True, "capstone_ready"),
    SourceSpec("exp3135", EXP3135_REL_PATH, "archive_v291_activate_v292", False, "archive_v291_activate_v292_ready"),
    SourceSpec("exp3136", EXP3136_REL_PATH, "false_accept_autopsy", False, "false_accept_autopsy_v1_ready"),
    SourceSpec("exp3137", EXP3137_REL_PATH, "accept_abstain_contract", False, "acceptance_contract_v1_ready"),
    SourceSpec("exp3138", EXP3138_REL_PATH, "canonical_grounding", False, "canonical_grounding_pilot_v1_ready"),
    SourceSpec("exp3139", EXP3139_REL_PATH, "live_verifier_rerun", False, "live_verifier_rerun_v7_ready"),
    SourceSpec("exp3140", EXP3140_REL_PATH, "repair_gate", False, "repair_gate_decision_v1_ready"),
    SourceSpec("exp3141", EXP3141_REL_PATH, "repair_ladder", False, "multi_turn_repair_ladder_v2_ready"),
    SourceSpec("exp3142", EXP3142_REL_PATH, "fr11_vera_evoenv", False, "fr11_vera_evoenv_v2_ready"),
    SourceSpec("exp3143", EXP3143_REL_PATH, "fr11_experience_memory", False, "fr11_experience_verifier_memory_v1_ready"),
    SourceSpec("exp3144", EXP3144_REL_PATH, "ebt_arm_calibration", False, "ebt_arm_false_accept_calibration_v3_ready"),
    SourceSpec("exp3145", EXP3145_REL_PATH, "kan_monitor_boundary", False, "kan_proof_carrying_monitor_v2_ready"),
    SourceSpec("exp3146", EXP3146_REL_PATH, "hardware_boundary", False, "hardware_sampler_evidence_boundary_v6_ready"),
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3147: aggregate matrix v26 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3133"]
    capstone = payloads["exp3134"]
    rows = _carry_forward_rows(matrix) + _dot292_rows(payloads) if matrix else []
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
        "matrix_v26_ready": ready,
        "rows_total": len(rows),
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v25": len(publication_blockers) - prior_count,
        "missing_artifacts": missing_artifacts,
        "status_counts": status_counts,
        "publication_blockers": publication_blockers,
        "rows": rows,
        "headline_claim_allowance_summary": _headline_claim_allowance_summary(payloads, rows),
        "false_accept_recovery_summary": _false_accept_recovery_summary(payloads, rows),
        "repair_gate_summary": _repair_gate_summary(payloads, rows),
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
    """Build and persist the Exp 3147 deliverable JSON."""

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
        summary.setdefault("v26_status_rationale", "carried_forward_from_matrix_v25")
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v25")
        rows.append(row)
    return rows


def _dot292_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _false_accept_autopsy_row(payloads["exp3136"]),
        _contract_row(payloads["exp3137"]),
        _canonical_grounding_row(payloads["exp3138"]),
        _live_verifier_row(payloads["exp3139"]),
        _repair_gate_row(payloads["exp3140"]),
        _repair_ladder_row(payloads["exp3141"], payloads["exp3140"]),
        _fr11_vera_row(payloads["exp3142"]),
        _fr11_experience_row(payloads["exp3143"]),
        _ebt_arm_row(payloads["exp3144"]),
        _kan_monitor_row(payloads["exp3145"]),
        _hardware_row(payloads["exp3146"]),
    ]


def _false_accept_autopsy_row(payload: Mapping[str, Any]) -> JsonDict:
    status = _ready_flagged_status(payload, "false_accept_autopsy_v1_ready")
    return _row(
        row_id="dot292:exp3136_false_accept_autopsy",
        status=status,
        source_artifact=EXP3136_REL_PATH.as_posix(),
        source_field="false_accept_autopsy_v1_ready",
        evidence_class="false_accept_root_cause_autopsy",
        claim_scope="false_accept_recovery_diagnostic",
        summary={
            "false_accept_autopsy_v1_ready": payload.get("false_accept_autopsy_v1_ready") is True,
            "source_false_accept_rate": _float_or_none(payload.get("source_false_accept_rate")),
            "recomputed_false_accept_rate": _float_or_none(
                payload.get("recomputed_false_accept_rate")
            ),
            "source_false_accept_count": _int_or_none(payload.get("source_false_accept_count")),
            "source_live_row_count": _int_or_none(payload.get("source_live_row_count")),
            "false_accept_row_ids": _text_list(payload.get("false_accept_row_ids")),
            "false_accept_mechanism_counts": _as_mapping(
                payload.get("false_accept_mechanism_counts")
            ),
            "regression_row_set": _text_list(payload.get("regression_row_set")),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _contract_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("acceptance_contract_v1_ready") is not True:
        status = "blocked"
    elif (
        payload.get("known_false_accept_rows_blocked") is True
        and (_float_or_none(payload.get("replay_false_accept_rate")) or 0.0) == 0.0
        and (_float_or_none(payload.get("replay_false_reject_rate")) or 0.0) == 0.0
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot292:exp3137_accept_abstain_contract",
        status=status,
        source_artifact=EXP3137_REL_PATH.as_posix(),
        source_field="acceptance_contract_v1_ready",
        evidence_class="exact_safe_accept_abstain_contract",
        claim_scope="exact_safe_accept_abstain_contract",
        summary={
            "acceptance_contract_v1_ready": payload.get("acceptance_contract_v1_ready") is True,
            "known_false_accept_rows_blocked": payload.get("known_false_accept_rows_blocked")
            is True,
            "replay_false_accept_rate": _float_or_none(payload.get("replay_false_accept_rate")),
            "replay_false_reject_rate": _float_or_none(payload.get("replay_false_reject_rate")),
            "replay_abstention_rate": _float_or_none(payload.get("replay_abstention_rate")),
            "replay_counts": _as_mapping(payload.get("replay_counts")),
            "repair_gate_prerequisites": _as_mapping(payload.get("repair_gate_prerequisites")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _canonical_grounding_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("canonical_grounding_pilot_v1_ready") is not True:
        status = "blocked"
    elif _as_list(payload.get("residual_false_accept_rows")):
        status = "blocked"
    elif (
        payload.get("canonicalizer_implemented") is True
        and (_int_or_none(payload.get("false_accept_rows_blocked")) or 0)
        >= (_int_or_none(payload.get("regression_rows_evaluated")) or 1)
    ):
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="dot292:exp3138_canonical_grounding",
        status=status,
        source_artifact=EXP3138_REL_PATH.as_posix(),
        source_field="canonical_grounding_pilot_v1_ready",
        evidence_class="canonical_answer_vericot_grounding_pilot",
        claim_scope="canonical_grounding_regression_replay",
        summary={
            "canonical_grounding_pilot_v1_ready": payload.get(
                "canonical_grounding_pilot_v1_ready"
            )
            is True,
            "false_accept_rows_blocked": _int_or_none(payload.get("false_accept_rows_blocked")),
            "regression_rows_evaluated": _int_or_none(payload.get("regression_rows_evaluated")),
            "residual_false_accept_rows": _text_list(payload.get("residual_false_accept_rows")),
            "canonicalizer_implemented": payload.get("canonicalizer_implemented") is True,
            "premise_grounding_block_count": _int_or_none(
                payload.get("premise_grounding_block_count")
            ),
            "canonicalization_block_count": _int_or_none(
                payload.get("canonicalization_block_count")
            ),
            "ledger_replay_block_count": _int_or_none(payload.get("ledger_replay_block_count")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _live_verifier_row(payload: Mapping[str, Any]) -> JsonDict:
    false_accept = _float_or_none(payload.get("false_accept_rate")) or 0.0
    if not payload:
        status = "missing"
    elif payload.get("live_verifier_rerun_v7_ready") is not True:
        status = "blocked"
    elif false_accept > 0.0 or payload.get("false_accept_gate_passed") is not True:
        status = "blocked"
    elif _has_flags(payload):
        status = "flagged"
    elif payload.get("headline_claim_allowed") is True:
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="dot292:exp3139_live_verifier_rerun",
        status=status,
        source_artifact=EXP3139_REL_PATH.as_posix(),
        source_field="live_verifier_rerun_v7_ready",
        evidence_class="live_sota_verifier_rerun_v7",
        claim_scope="live_sota_verifier_lift",
        summary={
            "live_verifier_rerun_v7_ready": payload.get("live_verifier_rerun_v7_ready") is True,
            "false_accept_gate_passed": payload.get("false_accept_gate_passed") is True,
            "headline_claim_allowed_source": payload.get("headline_claim_allowed") is True,
            "false_accept_rate": false_accept,
            "false_reject_rate": _float_or_none(payload.get("false_reject_rate")) or 0.0,
            "abstention_rate": _float_or_none(payload.get("abstention_rate")),
            "verifier_gain_delta": _float_or_none(payload.get("verifier_gain_delta")) or 0.0,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "repair_gate_candidate_state": str(payload.get("repair_gate_candidate_state") or ""),
            "source_false_accept_rate": _float_or_none(payload.get("source_false_accept_rate")),
            "regression_rows_included": payload.get("regression_rows_included") is True,
            "selected_model_ids": _text_list(payload.get("selected_model_ids")),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_gate_row(payload: Mapping[str, Any]) -> JsonDict:
    gate_state = str(payload.get("repair_gate_state") or "")
    blockers = _as_list(payload.get("repair_blockers")) + _as_list(
        payload.get("headline_disqualifiers")
    )
    if not payload:
        status = "missing"
    elif payload.get("repair_gate_decision_v1_ready") is not True:
        status = "blocked"
    elif gate_state == "unblocked" and not blockers:
        status = "clean"
    elif gate_state.startswith("blocked"):
        status = "blocked"
    else:
        status = "bounded"
    return _row(
        row_id="dot292:exp3140_repair_gate",
        status=status,
        source_artifact=EXP3140_REL_PATH.as_posix(),
        source_field="repair_gate_state",
        evidence_class="repair_gate_unlock_decision",
        claim_scope="repair_gate_decision",
        summary={
            "repair_gate_decision_v1_ready": payload.get("repair_gate_decision_v1_ready") is True,
            "repair_gate_state": gate_state,
            "false_accept_gate_passed": payload.get("false_accept_gate_passed") is True,
            "false_accept_rate": _float_or_none(payload.get("false_accept_rate")),
            "known_false_accepts_blocked": payload.get("known_false_accepts_blocked") is True,
            "regression_rows_included": payload.get("regression_rows_included") is True,
            "exact_authority_ready": payload.get("exact_authority_ready") is True,
            "live_model_ready": payload.get("live_model_ready") is True,
            "monitor_ledger_ready": payload.get("monitor_ledger_ready") is True,
            "repair_rows_available": payload.get("repair_rows_available") is True,
            "repair_blocker_count": len(_as_list(payload.get("repair_blockers"))),
            "headline_disqualifier_count": len(_as_list(payload.get("headline_disqualifiers"))),
            "selected_repair_rows": _as_list(payload.get("selected_repair_rows")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_ladder_row(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> JsonDict:
    source_status = normal_status(str(payload.get("status") or "missing"))
    gate_state = str(gate.get("repair_gate_state") or "")
    if not payload and gate_state and gate_state != "unblocked":
        status = "gated_skipped"
    elif not payload:
        status = "missing"
    elif payload.get("multi_turn_repair_ladder_v2_ready") is True:
        status = "clean"
    elif source_status != "missing":
        status = source_status
    else:
        status = "blocked"
    return _row(
        row_id="dot292:exp3141_repair_ladder",
        status=status,
        source_artifact=EXP3141_REL_PATH.as_posix(),
        source_field="multi_turn_repair_ladder_v2_ready",
        evidence_class="multi_turn_repair_ladder_v2",
        claim_scope="repair_live_rerun",
        summary={
            "multi_turn_repair_ladder_v2_ready": payload.get("multi_turn_repair_ladder_v2_ready")
            is True,
            "source_status": str(payload.get("status") or ""),
            "repair_gate_state": gate_state,
            "gated_skip": status == "gated_skipped",
            "gate_check_summary": "repair gate did not unlock"
            if status == "gated_skipped"
            else str(payload.get("gate_check_summary") or ""),
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "repair_attempt_count": _int_or_none(payload.get("repair_attempt_count")) or 0,
            "headline_repair_claim_allowed": payload.get("headline_repair_claim_allowed") is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_vera_row(payload: Mapping[str, Any]) -> JsonDict:
    ledger = _float_or_none(payload.get("ledger_consistency_rate"))
    if not payload:
        status = "missing"
    elif payload.get("fr11_vera_evoenv_v2_ready") is not True:
        status = "blocked"
    elif _has_flags(payload):
        status = "flagged"
    elif (
        payload.get("no_weight_update_claim") is True
        or payload.get("live_model_variant_generation") is not True
        or (ledger is not None and ledger < 1.0)
    ):
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot292:exp3142_fr11_vera_evoenv",
        status=status,
        source_artifact=EXP3142_REL_PATH.as_posix(),
        source_field="fr11_vera_evoenv_v2_ready",
        evidence_class="fr11_vera_evoenv_hardening",
        claim_scope="controller_only_environment_memory",
        summary={
            "fr11_vera_evoenv_v2_ready": payload.get("fr11_vera_evoenv_v2_ready") is True,
            "continuous_self_learning_targeted": payload.get("continuous_self_learning_targeted")
            is True,
            "live_model_variant_generation": payload.get("live_model_variant_generation") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "admitted_environment_count": _int_or_none(payload.get("admitted_environment_count")),
            "hardened_variant_count": _int_or_none(payload.get("hardened_variant_count")),
            "equivalent_variant_count": _int_or_none(payload.get("equivalent_variant_count")),
            "ledger_consistency_rate": ledger,
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "promotion_recommendation": str(payload.get("promotion_recommendation") or ""),
            "soundness_errors": _int_or_none(payload.get("soundness_errors")) or 0,
            "completeness_errors": _int_or_none(payload.get("completeness_errors")) or 0,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_experience_row(payload: Mapping[str, Any]) -> JsonDict:
    ledger = _float_or_none(payload.get("ledger_consistency_rate"))
    if not payload:
        status = "missing"
    elif payload.get("fr11_experience_verifier_memory_v1_ready") is not True:
        status = "blocked"
    elif _has_flags(payload):
        status = "flagged"
    elif payload.get("no_weight_update_claim") is True or (ledger is not None and ledger < 1.0):
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot292:exp3143_fr11_experience_memory",
        status=status,
        source_artifact=EXP3143_REL_PATH.as_posix(),
        source_field="fr11_experience_verifier_memory_v1_ready",
        evidence_class="fr11_experience_driven_verifier_memory",
        claim_scope="controller_only_experience_memory",
        summary={
            "fr11_experience_verifier_memory_v1_ready": payload.get(
                "fr11_experience_verifier_memory_v1_ready"
            )
            is True,
            "continuous_self_learning_targeted": payload.get("continuous_self_learning_targeted")
            is True,
            "replay_row_count": _int_or_none(payload.get("replay_row_count")) or 0,
            "ledger_consistency_rate": ledger,
            "no_weight_update_claim": payload.get("no_weight_update_claim") is True,
            "promotion_recommendation": str(payload.get("promotion_recommendation") or ""),
            "suppressed_check_count": _int_or_none(payload.get("suppressed_check_count")) or 0,
            "escalated_check_count": _int_or_none(payload.get("escalated_check_count")) or 0,
            "estimated_check_savings_rate": _float_or_none(
                payload.get("estimated_check_savings_rate")
            ),
            "residual_false_accept_risk": _float_or_none(
                payload.get("residual_false_accept_risk")
            ),
            "residual_false_reject_risk": _float_or_none(
                payload.get("residual_false_reject_risk")
            ),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _ebt_arm_row(payload: Mapping[str, Any]) -> JsonDict:
    blockers = _as_list(payload.get("integration_blockers"))
    if not payload:
        status = "missing"
    elif payload.get("ebt_arm_false_accept_calibration_v3_ready") is not True:
        status = "blocked"
    elif payload.get("live_integration") is not True or blockers:
        status = "projection_only"
    elif _has_flags(payload):
        status = "flagged"
    else:
        status = "clean"
    return _row(
        row_id="dot292:exp3144_ebt_arm_calibration",
        status=status,
        source_artifact=EXP3144_REL_PATH.as_posix(),
        source_field="ebt_arm_false_accept_calibration_v3_ready",
        evidence_class="ebt_arm_false_accept_calibration_boundary",
        claim_scope="architecture_energy_budget_boundary",
        summary={
            "ebt_arm_false_accept_calibration_v3_ready": payload.get(
                "ebt_arm_false_accept_calibration_v3_ready"
            )
            is True,
            "live_integration": payload.get("live_integration") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "live_row_count": _int_or_none(payload.get("live_row_count")) or 0,
            "false_accept_rows_evaluated": _int_or_none(
                payload.get("false_accept_rows_evaluated")
            )
            or 0,
            "false_accept_row_ids": _text_list(payload.get("false_accept_row_ids")),
            "integration_blocker_count": len(blockers),
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "corrigendum_pending_count": len(_as_list(payload.get("corrigendum_pending"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _kan_monitor_row(payload: Mapping[str, Any]) -> JsonDict:
    claim_boundary = _as_mapping(payload.get("claim_boundary"))
    does_not_prove = _text_list(claim_boundary.get("does_not_prove"))
    if not payload:
        status = "missing"
    elif payload.get("kan_proof_carrying_monitor_v2_ready") is not True or _as_list(
        payload.get("implementation_blockers")
    ):
        status = "blocked"
    elif payload.get("deployed_verifier_claim") is not True or does_not_prove:
        status = "bounded"
    else:
        status = "clean"
    return _row(
        row_id="dot292:exp3145_kan_monitor_boundary",
        status=status,
        source_artifact=EXP3145_REL_PATH.as_posix(),
        source_field="kan_proof_carrying_monitor_v2_ready",
        evidence_class="kan_proof_carrying_monitor_boundary",
        claim_scope="architecture_kan_verifier_boundary",
        summary={
            "kan_proof_carrying_monitor_v2_ready": payload.get(
                "kan_proof_carrying_monitor_v2_ready"
            )
            is True,
            "kan_code_present": payload.get("kan_code_present") is True,
            "deployed_verifier_claim": payload.get("deployed_verifier_claim") is True,
            "attached_monitor_record_count": _int_or_none(
                payload.get("attached_monitor_record_count")
            )
            or 0,
            "milp_property_check_count": _int_or_none(payload.get("milp_property_check_count"))
            or 0,
            "implementation_blocker_count": len(_as_list(payload.get("implementation_blockers"))),
            "claim_boundary_proves": str(claim_boundary.get("proves") or ""),
            "claim_boundary_does_not_prove": does_not_prove,
            "false_accept_relevance": _as_mapping(payload.get("false_accept_relevance")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _hardware_row(payload: Mapping[str, Any]) -> JsonDict:
    missing_operator = _as_list(payload.get("missing_operator_evidence"))
    missing_sources = _as_list(payload.get("missing_required_source_artifacts"))
    ready = payload.get("hardware_sampler_evidence_boundary_v6_ready") is True
    gatemate = payload.get("gatemate_evidence_complete") is True
    ssqa = payload.get("ssqa_readback_ready") is True
    speedup = payload.get("speedup_claim_allowed") is True
    if not payload:
        status = "missing"
    elif not ready or missing_sources or not gatemate or not ssqa or missing_operator:
        status = "blocked"
    elif speedup:
        status = "clean"
    else:
        status = "bounded"
    return _row(
        row_id="dot292:exp3146_hardware_boundary",
        status=status,
        source_artifact=EXP3146_REL_PATH.as_posix(),
        source_field="hardware_sampler_evidence_boundary_v6_ready",
        evidence_class="hardware_sampler_evidence_boundary",
        claim_scope="architecture_hardware_sampler_boundary",
        summary={
            "hardware_sampler_evidence_boundary_v6_ready": ready,
            "gatemate_evidence_complete": gatemate,
            "ssqa_readback_ready": ssqa,
            "speedup_claim_allowed": speedup,
            "kona_claim_allowed": payload.get("kona_claim_allowed") is True,
            "thrml_tsu_claim_allowed": payload.get("thrml_tsu_claim_allowed") is True,
            "hardware_commands_run": _as_list(payload.get("hardware_commands_run")),
            "missing_operator_evidence_count": len(missing_operator),
            "missing_required_source_artifacts": _text_list(
                payload.get("missing_required_source_artifacts")
            ),
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
        "row_origin": str(row.get("row_origin") or "matrix_v25"),
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
        "row_origin": "milestone_292",
    }


def _ready_flagged_status(payload: Mapping[str, Any], ready_field: str) -> str:
    if not payload:
        return "missing"
    if payload.get(ready_field) is not True:
        return "blocked"
    if _has_flags(payload):
        return "flagged"
    return "clean"


def _has_flags(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True or bool(
        _as_list(payload.get("corrigendum_pending"))
    )


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
            else "missing_or_malformed_dot292_artifact",
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


def _headline_claim_allowance_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    matrix_allowance = _as_mapping(payloads["exp3133"].get("headline_claim_allowance_summary"))
    statuses = _row_statuses(rows)
    live_status = statuses.get("dot292:exp3139_live_verifier_rerun", "missing")
    repair_status = statuses.get("dot292:exp3140_repair_gate", "missing")
    ladder_status = statuses.get("dot292:exp3141_repair_ladder", "missing")
    fr11_statuses = {
        statuses.get("dot292:exp3142_fr11_vera_evoenv", "missing"),
        statuses.get("dot292:exp3143_fr11_experience_memory", "missing"),
    }
    architecture = _architecture_boundary_summary(payloads, rows)
    blocked: list[str] = []
    if matrix_allowance.get("comparative_sota_pair_allowed") is not True:
        blocked.append("comparative_sota_pair")
    if live_status != "clean":
        blocked.append("live_verifier_lift_adversarial_flag")
    if repair_status != "clean" or ladder_status != "clean":
        blocked.append("repair_headline")
    if any(status != "clean" for status in fr11_statuses):
        blocked.append("fr11_model_weight_learning")
    if architecture["live_integration"] is not True:
        blocked.append("ebt_arm_live_integration")
    if architecture["deployed_kan_verifier_claim"] is not True:
        blocked.append("kan_deployed_verifier")
    if architecture["speedup_claim_allowed"] is not True:
        blocked.append("hardware_speedup")
    return {
        "sota_cache_headline_allowed": matrix_allowance.get("sota_cache_headline_allowed") is True,
        "comparative_sota_pair_allowed": matrix_allowance.get("comparative_sota_pair_allowed")
        is True,
        "live_verifier_headline_allowed": live_status == "clean",
        "repair_headline_claim_allowed": repair_status == "clean" and ladder_status == "clean",
        "exact_safe_contract_claim_allowed": statuses.get(
            "dot292:exp3137_accept_abstain_contract"
        )
        == "clean",
        "canonical_grounding_claim_allowed": statuses.get("dot292:exp3138_canonical_grounding")
        == "clean",
        "fr11_model_weight_learning_allowed": all(status == "clean" for status in fr11_statuses),
        "ebt_arm_live_integration_allowed": architecture["live_integration"] is True,
        "kan_deployed_verifier_allowed": architecture["deployed_kan_verifier_claim"] is True,
        "hardware_speedup_claim_allowed": architecture["speedup_claim_allowed"] is True,
        "present_model_ids": _text_list(matrix_allowance.get("present_model_ids")),
        "missing_model_ids": _text_list(matrix_allowance.get("missing_model_ids")),
        "selected_headline_model_ids": _text_list(
            matrix_allowance.get("selected_headline_model_ids")
        ),
        "blocked_headline_claims": blocked,
    }


def _false_accept_recovery_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    autopsy = payloads["exp3136"]
    contract = payloads["exp3137"]
    canonical = payloads["exp3138"]
    live = payloads["exp3139"]
    flagged_count = sum(
        1 for payload in (autopsy, live) if payload.get("flagged_adversarial") is True
    )
    corrigendum_count = sum(
        len(_as_list(payload.get("corrigendum_pending"))) for payload in (autopsy, live)
    )
    if flagged_count or corrigendum_count:
        claim_status = "blocked_by_adversarial_corrigendum"
    elif (
        contract.get("known_false_accept_rows_blocked") is True
        and canonical.get("residual_false_accept_rows") == []
        and (_float_or_none(live.get("false_accept_rate")) or 0.0) == 0.0
    ):
        claim_status = "exact_safe_recovery_ready"
    else:
        claim_status = "blocked_or_bounded"
    return {
        "autopsy_status": statuses.get("dot292:exp3136_false_accept_autopsy", "missing"),
        "accept_abstain_contract_status": statuses.get(
            "dot292:exp3137_accept_abstain_contract", "missing"
        ),
        "canonical_grounding_status": statuses.get("dot292:exp3138_canonical_grounding", "missing"),
        "live_verifier_status": statuses.get("dot292:exp3139_live_verifier_rerun", "missing"),
        "source_false_accept_rate": _float_or_none(autopsy.get("source_false_accept_rate")),
        "recomputed_false_accept_rate": _float_or_none(
            autopsy.get("recomputed_false_accept_rate")
        ),
        "false_accept_row_ids": _text_list(autopsy.get("false_accept_row_ids")),
        "known_false_accept_rows_blocked": contract.get("known_false_accept_rows_blocked")
        is True,
        "canonical_false_accept_rows_blocked": _int_or_none(
            canonical.get("false_accept_rows_blocked")
        )
        or 0,
        "rerun_false_accept_rate": _float_or_none(live.get("false_accept_rate")),
        "rerun_abstention_rate": _float_or_none(live.get("abstention_rate")),
        "rerun_verifier_gain_delta": _float_or_none(live.get("verifier_gain_delta")),
        "false_accept_gate_passed": live.get("false_accept_gate_passed") is True,
        "flagged_adversarial_artifact_count": flagged_count,
        "corrigendum_pending_count": corrigendum_count,
        "recovery_claim_status": claim_status,
    }


def _repair_gate_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    gate = payloads["exp3140"]
    ladder = payloads["exp3141"]
    return {
        "repair_gate_status": statuses.get("dot292:exp3140_repair_gate", "missing"),
        "repair_ladder_status": statuses.get("dot292:exp3141_repair_ladder", "missing"),
        "repair_gate_state": str(gate.get("repair_gate_state") or ""),
        "false_accept_gate_passed": gate.get("false_accept_gate_passed") is True,
        "false_accept_rate": _float_or_none(gate.get("false_accept_rate")),
        "known_false_accepts_blocked": gate.get("known_false_accepts_blocked") is True,
        "regression_rows_included": gate.get("regression_rows_included") is True,
        "selected_repair_rows": _as_list(gate.get("selected_repair_rows")),
        "repair_blocker_count": len(_as_list(gate.get("repair_blockers"))),
        "headline_disqualifier_count": len(_as_list(gate.get("headline_disqualifiers"))),
        "repair_ladder_present": bool(ladder),
        "repair_ladder_missing_path": EXP3141_REL_PATH.as_posix() if not ladder else "",
        "headline_repair_claim_allowed": statuses.get("dot292:exp3140_repair_gate") == "clean"
        and statuses.get("dot292:exp3141_repair_ladder") == "clean",
    }


def _fr11_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    vera = payloads["exp3142"]
    memory = payloads["exp3143"]
    no_weight_update = (
        vera.get("no_weight_update_claim") is True or memory.get("no_weight_update_claim") is True
    )
    vera_ledger = _float_or_none(vera.get("ledger_consistency_rate"))
    memory_ledger = _float_or_none(memory.get("ledger_consistency_rate"))
    return {
        "vera_evoenv_status": statuses.get("dot292:exp3142_fr11_vera_evoenv", "missing"),
        "experience_memory_status": statuses.get(
            "dot292:exp3143_fr11_experience_memory", "missing"
        ),
        "continuous_self_learning_targeted": vera.get("continuous_self_learning_targeted") is True
        or memory.get("continuous_self_learning_targeted") is True,
        "admitted_environment_count": _int_or_none(vera.get("admitted_environment_count")) or 0,
        "hardened_variant_count": _int_or_none(vera.get("hardened_variant_count")) or 0,
        "replay_row_count": _int_or_none(memory.get("replay_row_count")) or 0,
        "no_weight_update_claim": no_weight_update,
        "model_weight_learning_allowed": (
            not no_weight_update
            and (vera_ledger is None or vera_ledger >= 1.0)
            and (memory_ledger is None or memory_ledger >= 1.0)
        ),
        "vera_ledger_consistency_rate": vera_ledger,
        "experience_ledger_consistency_rate": memory_ledger,
        "promotion_recommendations": [
            text
            for text in (
                str(vera.get("promotion_recommendation") or ""),
                str(memory.get("promotion_recommendation") or ""),
            )
            if text
        ],
        "residual_false_accept_risk": _float_or_none(memory.get("residual_false_accept_risk")),
        "residual_false_reject_risk": _float_or_none(memory.get("residual_false_reject_risk")),
    }


def _architecture_boundary_summary(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> JsonDict:
    statuses = _row_statuses(rows)
    ebt = payloads["exp3144"]
    kan = payloads["exp3145"]
    hardware = payloads["exp3146"]
    return {
        "ebt_arm_status": statuses.get("dot292:exp3144_ebt_arm_calibration", "missing"),
        "kan_monitor_status": statuses.get("dot292:exp3145_kan_monitor_boundary", "missing"),
        "hardware_boundary_status": statuses.get("dot292:exp3146_hardware_boundary", "missing"),
        "live_integration": ebt.get("live_integration") is True,
        "integration_blocker_count": len(_as_list(ebt.get("integration_blockers"))),
        "false_accept_rows_evaluated": _int_or_none(ebt.get("false_accept_rows_evaluated")) or 0,
        "deployed_kan_verifier_claim": kan.get("deployed_verifier_claim") is True,
        "kan_attached_monitor_record_count": _int_or_none(
            kan.get("attached_monitor_record_count")
        )
        or 0,
        "kan_does_not_prove": _text_list(_as_mapping(kan.get("claim_boundary")).get("does_not_prove")),
        "speedup_claim_allowed": hardware.get("speedup_claim_allowed") is True,
        "kona_claim_allowed": hardware.get("kona_claim_allowed") is True,
        "thrml_tsu_claim_allowed": hardware.get("thrml_tsu_claim_allowed") is True,
        "gatemate_evidence_complete": hardware.get("gatemate_evidence_complete") is True,
        "ssqa_readback_ready": hardware.get("ssqa_readback_ready") is True,
        "hardware_commands_run": _as_list(hardware.get("hardware_commands_run")),
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
    if matrix and matrix.get("matrix_v25_ready") is not True:
        violations.append("matrix v25 authority is not ready")
    if capstone and capstone.get("capstone_ready") is not True:
        violations.append("capstone v291 authority is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v26 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to rows_total")
    if len(publication_blockers) != sum(
        count for status, count in status_counts.items() if status in PUBLICATION_BLOCKING_STATUSES
    ):
        violations.append("publication_blocker_count does not match row statuses")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot292_artifacts",
        "source": "matrix_v25_capstone_v291_and_dot292_artifacts",
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
    if artifact.get("matrix_v26_ready") is not True:
        return (
            "blocked_matrix_v26_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"invariant_violations={len(_as_list(artifact.get('invariant_violations')))}"
        )
    return (
        "complete: matrix_v26_ready=true; "
        f"rows_total={artifact.get('rows_total')}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v25={artifact.get('blocker_delta_from_v25')}; "
        f"missing_artifacts={len(_as_list(artifact.get('missing_artifacts')))}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
