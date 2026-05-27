"""Build the Exp 3189 cross-corpus matrix v29 artifact.

Spec refs: REQ-REPORT-3189, SCENARIO-REPORT-3189.

Matrix v29 is an evidence ledger over the `.295` milestone. It appends the
receipt, verifier, repair, FR-11, sidecar, and THRML boundary rows to matrix
v28 without re-running models, verifier scoring, repairs, solvers, hardware, or
the conductor. That separation matters because the matrix is claim accounting,
not a new source of empirical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

from carnot.reporting import cross_corpus_matrix_v28_3175 as v28


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.295"
SCHEMA = "carnot.cross_corpus_matrix.v29_295_artifact_aggregation.v1"
ARTIFACT = "experiment_3189_cross_corpus_matrix_v29"
OUTPUT_REL_PATH = Path("results/experiment_3189_cross_corpus_matrix_v29.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3189_cross_corpus_matrix_v29.py"

MATRIX_V28_REL_PATH = Path("results/experiment_3175_cross_corpus_matrix_v28.json")
CAPSTONE_V294_REL_PATH = Path("results/experiment_3176_capstone_v294.json")
ARCHIVE_V295_REL_PATH = Path("results/experiment_3177_archive_v294_activate_v295.json")
EXP3178_REL_PATH = Path("results/experiment_3178_receipt_backed_authenticity_contract_v3.json")
EXP3179_REL_PATH = Path("results/experiment_3179_local_sota_receipt_smoke_v3.json")
EXP3180_REL_PATH = Path("results/experiment_3180_controlled_invariance_executor_v2.json")
EXP3181_REL_PATH = Path("results/experiment_3181_clean_live_sota_verifier_rerun_v10.json")
EXP3182_REL_PATH = Path("results/experiment_3182_distributional_ebm_exact_row_sidecar_v1.json")
EXP3183_REL_PATH = Path("results/experiment_3183_counterexample_certificate_expansion_v3.json")
EXP3184_REL_PATH = Path("results/experiment_3184_repair_gate_decision_v4.json")
EXP3185_REL_PATH = Path("results/experiment_3185_multi_turn_repair_ladder_v5.json")
EXP3186_REL_PATH = Path("results/experiment_3186_fr11_controller_memory_promotion_pack_v1.json")
EXP3187_REL_PATH = Path("results/experiment_3187_fr11_cross_environment_drift_replay_v1.json")
EXP3188_REL_PATH = Path("results/experiment_3188_thrml_factor_graph_api_boundary_v1.json")

STATUSES = v28.STATUSES
PUBLICATION_BLOCKING_STATUSES = v28.PUBLICATION_BLOCKING_STATUSES

read_json_object = v28.read_json_object
sha256_file = v28.sha256_file
normal_status = v28.normal_status
blocker_class = v28.blocker_class
_as_mapping = v28._as_mapping
_as_list = v28._as_list
_text_list = v28._text_list
_int_or_none = v28._int_or_none
_float_or_none = v28._float_or_none


@dataclass(frozen=True)
class SourceSpec:
    """A checked-in source artifact that v29 reads without mutating."""

    experiment_id: str
    path: Path
    role: str
    required: bool = False
    ready_field: str = ""


REQUIRED_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3175", MATRIX_V28_REL_PATH, "matrix_v28_authority", True, "matrix_v28_ready"),
    SourceSpec(
        "exp3176", CAPSTONE_V294_REL_PATH, "capstone_v294_authority", True, "capstone_v294_ready"
    ),
    SourceSpec(
        "exp3177",
        ARCHIVE_V295_REL_PATH,
        "archive_v294_activate_v295",
        True,
        "archive_v294_activate_v295_ready",
    ),
)

DOT295_SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3178",
        EXP3178_REL_PATH,
        "receipt_backed_authenticity_contract",
        False,
        "receipt_backed_authenticity_contract_v3_ready",
    ),
    SourceSpec(
        "exp3179",
        EXP3179_REL_PATH,
        "local_sota_receipt_smoke",
        False,
        "local_sota_receipt_smoke_v3_ready",
    ),
    SourceSpec(
        "exp3180",
        EXP3180_REL_PATH,
        "controlled_invariance_executor",
        False,
        "controlled_invariance_executor_v2_ready",
    ),
    SourceSpec(
        "exp3181",
        EXP3181_REL_PATH,
        "clean_live_sota_verifier_rerun",
        False,
        "clean_live_sota_verifier_rerun_v10_ready",
    ),
    SourceSpec(
        "exp3182",
        EXP3182_REL_PATH,
        "distributional_ebm_exact_row_sidecar",
        False,
        "distributional_ebm_exact_row_sidecar_v1_ready",
    ),
    SourceSpec(
        "exp3183",
        EXP3183_REL_PATH,
        "counterexample_certificate_expansion",
        False,
        "counterexample_certificate_expansion_v3_ready",
    ),
    SourceSpec(
        "exp3184", EXP3184_REL_PATH, "repair_gate_v4", False, "repair_gate_decision_v4_ready"
    ),
    SourceSpec(
        "exp3185", EXP3185_REL_PATH, "repair_ladder_v5", False, "multi_turn_repair_ladder_v5_ready"
    ),
    SourceSpec(
        "exp3186",
        EXP3186_REL_PATH,
        "fr11_controller_memory_promotion_pack",
        False,
        "fr11_controller_memory_promotion_pack_v1_ready",
    ),
    SourceSpec(
        "exp3187",
        EXP3187_REL_PATH,
        "fr11_cross_environment_drift_replay",
        False,
        "fr11_cross_environment_drift_replay_v1_ready",
    ),
    SourceSpec(
        "exp3188",
        EXP3188_REL_PATH,
        "thrml_factor_graph_api_boundary",
        False,
        "thrml_factor_graph_api_boundary_v1_ready",
    ),
)

SOURCE_SPECS: tuple[SourceSpec, ...] = REQUIRED_SOURCE_SPECS + DOT295_SOURCE_SPECS


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3189: aggregate matrix v29 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    matrix = payloads["exp3175"]
    capstone = payloads["exp3176"]
    archive = payloads["exp3177"]
    rows = _carry_forward_rows(matrix) + _dot295_rows(payloads) if matrix else []
    status_counts = _status_counts(rows)
    publication_blockers = _publication_blockers(rows)
    prior_count = _prior_publication_blocker_count(matrix)
    missing_artifacts, missing_comparison = _missing_artifacts(matrix, sources)
    required_source_errors = _required_source_errors(sources)
    narrowing = _paper_v6_narrowing(payloads)
    narrowing_preserved = not any(narrowing.values())
    invariant_violations = _invariant_violations(
        matrix,
        capstone,
        archive,
        rows,
        status_counts,
        publication_blockers,
        required_source_errors,
        narrowing_preserved,
    )
    ready = not invariant_violations
    paper_implications = _paper_readiness_implications(
        rows,
        len(publication_blockers),
        narrowing_preserved,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "cross_corpus_matrix_v29_ready": ready,
        "prior_matrix_version": "v28",
        "prior_matrix_artifact": MATRIX_V28_REL_PATH.as_posix(),
        "rows_total": len(rows),
        "status_counts": status_counts,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": len(publication_blockers),
        "blocker_delta_from_v28": len(publication_blockers) - prior_count,
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
        "verifier_status": _verifier_status(payloads, rows),
        "repair_status": _repair_status(payloads, rows),
        "fr11_status": _fr11_status(payloads, rows),
        "hardware_status": _hardware_status(payloads, rows),
        "sidecar_status": _sidecar_status(payloads, rows),
        "next_top_gap": _next_top_gap(payloads, rows),
        "paper_ready": paper_implications["paper_ready"],
        "paper_readiness_implications": paper_implications,
        "paper_v6_narrowing": narrowing,
        "paper_v6_narrowing_preserved": narrowing_preserved,
        "publication_blockers": publication_blockers,
        "rows": rows,
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
    """Build and persist the Exp 3189 deliverable JSON."""

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
        "loaded_path": spec.path.as_posix(),
        "role": spec.role,
        "required": spec.required,
        "ready_field": spec.ready_field,
        "source_type": "json",
        "present": path.is_file(),
        "primary_present": path.is_file(),
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
        summary.setdefault("v29_status_rationale", "carried_forward_from_matrix_v28")
        row["summary"] = summary
        row["row_origin"] = str(row.get("row_origin") or "matrix_v28")
        rows.append(row)
    return rows


def _dot295_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[JsonDict]:
    return [
        _receipt_contract_row(payloads["exp3178"]),
        _sota_receipt_smoke_row(payloads["exp3179"]),
        _controlled_invariance_row(payloads["exp3180"]),
        _clean_rerun_v10_row(payloads["exp3181"]),
        _distributional_sidecar_row(payloads["exp3182"]),
        _certificate_expansion_row(payloads["exp3183"]),
        _repair_gate_v4_row(payloads["exp3184"]),
        _repair_ladder_v5_row(payloads["exp3185"]),
        _fr11_promotion_pack_row(payloads["exp3186"]),
        _fr11_drift_replay_row(payloads["exp3187"]),
        _thrml_boundary_row(payloads["exp3188"]),
    ]


def _receipt_contract_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("receipt_backed_authenticity_contract_v3_ready") is not True:
        status = "blocked"
    elif payload.get("flagged_adversarial") is True or _as_list(payload.get("corrigendum_pending")):
        status = "flagged"
    else:
        status = "clean"
    return _row(
        row_id="dot295:exp3178_receipt_contract",
        status=status,
        source_artifact=EXP3178_REL_PATH.as_posix(),
        source_field="receipt_backed_authenticity_contract_v3_ready",
        evidence_class="receipt_backed_authenticity_contract_v3",
        claim_scope="live_verifier_authenticity_contract",
        summary={
            "receipt_backed_authenticity_contract_v3_ready": payload.get(
                "receipt_backed_authenticity_contract_v3_ready"
            )
            is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "contract_blocker_count": len(_as_list(payload.get("contract_blockers"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _sota_receipt_smoke_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("local_sota_receipt_smoke_v3_ready") is not True
        or payload.get("preflight_passed") is not True
    ):
        status = "blocked"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif (
        payload.get("clean_rerun_allowed") is True
        and payload.get("substrate_classification") == "full_local_sota_receipt"
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3179_sota_receipt_smoke",
        status=status,
        source_artifact=EXP3179_REL_PATH.as_posix(),
        source_field="clean_rerun_allowed",
        evidence_class="local_sota_receipt_smoke_v3",
        claim_scope="live_sota_receipt_precondition",
        summary={
            "local_sota_receipt_smoke_v3_ready": payload.get("local_sota_receipt_smoke_v3_ready")
            is True,
            "preflight_passed": payload.get("preflight_passed") is True,
            "clean_rerun_allowed": payload.get("clean_rerun_allowed") is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "substrate_classification": str(payload.get("substrate_classification") or ""),
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "proof_receipt_count": len(_as_list(payload.get("proof_receipts"))),
            "throughput_plausibility_passed": payload.get("throughput_plausibility_passed") is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _controlled_invariance_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("controlled_invariance_executor_v2_ready") is not True or _as_list(
        payload.get("source_errors")
    ):
        status = "blocked"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif payload.get("controlled_invariance_passed") is True:
        status = "diagnostic_only"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3180_controlled_invariance",
        status=status,
        source_artifact=EXP3180_REL_PATH.as_posix(),
        source_field="controlled_invariance_passed",
        evidence_class="controlled_invariance_executor_v2",
        claim_scope="exact_authority_diagnostic",
        summary={
            "controlled_invariance_executor_v2_ready": payload.get(
                "controlled_invariance_executor_v2_ready"
            )
            is True,
            "controlled_invariance_passed": payload.get("controlled_invariance_passed") is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "known_false_accept_regression_count": _int_or_none(
                payload.get("known_false_accept_regression_count")
            )
            or 0,
            "semantic_false_accept_count": _int_or_none(payload.get("semantic_false_accept_count"))
            or 0,
            "shortcut_failure_count": _int_or_none(payload.get("shortcut_failure_count")) or 0,
            "exact_row_count": _int_or_none(payload.get("exact_row_count")) or 0,
            "source_error_count": len(_as_list(payload.get("source_errors"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _clean_rerun_v10_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("clean_live_sota_verifier_rerun_v10_ready") is not True:
        status = "blocked"
    elif payload.get("gated_skip") is True:
        status = "gated_skipped"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif (
        payload.get("controlled_invariance_passed") is True
        and payload.get("headline_claim_allowed") is True
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3181_clean_live_rerun_v10",
        status=status,
        source_artifact=EXP3181_REL_PATH.as_posix(),
        source_field="clean_live_sota_verifier_rerun_v10_ready",
        evidence_class="clean_live_sota_verifier_rerun_v10",
        claim_scope="live_verifier_headline_evidence",
        summary={
            "clean_live_sota_verifier_rerun_v10_ready": payload.get(
                "clean_live_sota_verifier_rerun_v10_ready"
            )
            is True,
            "gated_skip": payload.get("gated_skip") is True,
            "gate_reasons": _text_list(payload.get("gate_reasons")),
            "controlled_invariance_passed": payload.get("controlled_invariance_passed") is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "false_accept_rate": _float_or_none(payload.get("false_accept_rate")),
            "false_reject_rate": _float_or_none(payload.get("false_reject_rate")),
            "abstention_rate": _float_or_none(payload.get("abstention_rate")),
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "live_call_count": _int_or_none(payload.get("live_call_count")) or 0,
            "known_false_accept_regression_count": _int_or_none(
                payload.get("known_false_accept_regression_count")
            )
            or 0,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _distributional_sidecar_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("distributional_ebm_exact_row_sidecar_v1_ready") is not True or _as_list(
        payload.get("source_errors")
    ):
        status = "blocked"
    elif payload.get("deployed_verifier_claim_allowed") is True:
        status = "clean"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot295:exp3182_distributional_sidecar",
        status=status,
        source_artifact=EXP3182_REL_PATH.as_posix(),
        source_field="deployed_verifier_claim_allowed",
        evidence_class="distributional_ebm_exact_row_sidecar_v1",
        claim_scope="diagnostic_sidecar_boundary",
        summary={
            "distributional_ebm_exact_row_sidecar_v1_ready": payload.get(
                "distributional_ebm_exact_row_sidecar_v1_ready"
            )
            is True,
            "deployed_verifier_claim_allowed": payload.get("deployed_verifier_claim_allowed")
            is True,
            "known_false_accept_rows_scored": _int_or_none(
                payload.get("known_false_accept_rows_scored")
            )
            or 0,
            "exact_labeled_row_count": _int_or_none(payload.get("exact_labeled_row_count")) or 0,
            "false_accept_separation_auc": _float_or_none(
                payload.get("false_accept_separation_auc")
            ),
            "source_error_count": len(_as_list(payload.get("source_errors"))),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _certificate_expansion_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("counterexample_certificate_expansion_v3_ready") is not True or _as_list(
        payload.get("source_errors")
    ):
        status = "blocked"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif payload.get("repair_call_ready") is True:
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3183_certificate_expansion",
        status=status,
        source_artifact=EXP3183_REL_PATH.as_posix(),
        source_field="repair_call_ready",
        evidence_class="counterexample_certificate_expansion_v3",
        claim_scope="formal_counterexample_repair_certificate",
        summary={
            "counterexample_certificate_expansion_v3_ready": payload.get(
                "counterexample_certificate_expansion_v3_ready"
            )
            is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "repair_call_ready": payload.get("repair_call_ready") is True,
            "counterexample_count": _int_or_none(payload.get("counterexample_count")) or 0,
            "exact_row_count": _int_or_none(payload.get("exact_row_count")) or 0,
            "known_false_accept_rows_covered": _int_or_none(
                payload.get("known_false_accept_rows_covered")
            )
            or 0,
            "blocker_reasons": _text_list(payload.get("blocker_reasons")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_gate_v4_row(payload: Mapping[str, Any]) -> JsonDict:
    gate_state = str(payload.get("repair_gate_state") or "")
    if not payload:
        status = "missing"
    elif payload.get("repair_gate_decision_v4_ready") is True and gate_state == "unblocked":
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3184_repair_gate_v4",
        status=status,
        source_artifact=EXP3184_REL_PATH.as_posix(),
        source_field="repair_gate_state",
        evidence_class="repair_gate_decision_v4",
        claim_scope="repair_gate_decision",
        summary={
            "repair_gate_decision_v4_ready": payload.get("repair_gate_decision_v4_ready") is True,
            "repair_gate_state": gate_state,
            "blocker_reasons": _text_list(payload.get("blocker_reasons")),
            "missing_artifact_count": len(_as_list(payload.get("missing_artifacts"))),
            "repair_attempt_budget_enabled": _as_mapping(
                payload.get("allowed_repair_attempt_budget")
            ).get("enabled")
            is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _repair_ladder_v5_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif payload.get("multi_turn_repair_ladder_v5_ready") is not True:
        status = "blocked"
    elif payload.get("gated_skip") is True:
        status = "gated_skipped"
    elif payload.get("flagged_adversarial") is True:
        status = "flagged"
    elif payload.get("headline_claim_allowed") is True:
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3185_repair_ladder_v5",
        status=status,
        source_artifact=EXP3185_REL_PATH.as_posix(),
        source_field="multi_turn_repair_ladder_v5_ready",
        evidence_class="multi_turn_repair_ladder_v5",
        claim_scope="repair_execution",
        summary={
            "multi_turn_repair_ladder_v5_ready": payload.get("multi_turn_repair_ladder_v5_ready")
            is True,
            "gated_skip": payload.get("gated_skip") is True,
            "gate_state": str(payload.get("gate_state") or ""),
            "headline_claim_allowed": payload.get("headline_claim_allowed") is True,
            "flagged_adversarial": payload.get("flagged_adversarial") is True,
            "repair_attempt_count": _int_or_none(payload.get("repair_attempt_count")) or 0,
            "repair_success_delta": _float_or_none(payload.get("repair_success_delta")),
            "remaining_blockers": _text_list(payload.get("remaining_blockers")),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_promotion_pack_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("fr11_controller_memory_promotion_pack_v1_ready") is True
        and payload.get("promotion_allowed") is True
        and payload.get("no_model_weight_update_claimed") is True
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3186_fr11_promotion_pack",
        status=status,
        source_artifact=EXP3186_REL_PATH.as_posix(),
        source_field="promotion_allowed",
        evidence_class="fr11_controller_memory_promotion_pack_v1",
        claim_scope="fr11_controller_memory_promotion",
        summary={
            "fr11_controller_memory_promotion_pack_v1_ready": payload.get(
                "fr11_controller_memory_promotion_pack_v1_ready"
            )
            is True,
            "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
            "learning_tier": str(payload.get("learning_tier") or ""),
            "promotion_allowed": payload.get("promotion_allowed") is True,
            "no_model_weight_update_claimed": payload.get("no_model_weight_update_claimed") is True,
            "promotion_decision": str(
                _as_mapping(payload.get("promotion_manifest")).get("promotion_decision") or ""
            ),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _fr11_drift_replay_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("fr11_cross_environment_drift_replay_v1_ready") is True
        and payload.get("promotion_allowed") is True
        and payload.get("no_model_weight_update_claimed") is True
        and payload.get("rollback_triggered") is not True
        and (_int_or_none(payload.get("negative_control_regression_count")) or 0) == 0
    ):
        status = "clean"
    else:
        status = "blocked"
    return _row(
        row_id="dot295:exp3187_fr11_drift_replay",
        status=status,
        source_artifact=EXP3187_REL_PATH.as_posix(),
        source_field="promotion_allowed",
        evidence_class="fr11_cross_environment_drift_replay_v1",
        claim_scope="fr11_controller_memory_drift_replay",
        summary={
            "fr11_cross_environment_drift_replay_v1_ready": payload.get(
                "fr11_cross_environment_drift_replay_v1_ready"
            )
            is True,
            "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
            "replay_mode_only": payload.get("replay_mode_only") is True,
            "promotion_allowed": payload.get("promotion_allowed") is True,
            "no_model_weight_update_claimed": payload.get("no_model_weight_update_claimed") is True,
            "cross_environment_row_count": _int_or_none(payload.get("cross_environment_row_count"))
            or 0,
            "heldout_row_count": _int_or_none(payload.get("heldout_row_count")) or 0,
            "negative_control_regression_count": _int_or_none(
                payload.get("negative_control_regression_count")
            )
            or 0,
            "rollback_triggered": payload.get("rollback_triggered") is True,
            "honest_verdict": str(payload.get("honest_verdict") or ""),
        },
    )


def _thrml_boundary_row(payload: Mapping[str, Any]) -> JsonDict:
    if not payload:
        status = "missing"
    elif (
        payload.get("thrml_factor_graph_api_boundary_v1_ready") is not True
        or _as_list(payload.get("source_errors"))
        or payload.get("hardware_speedup_claim_allowed") is True
        or payload.get("kona_or_tsu_execution_claimed") is True
    ):
        status = "blocked"
    else:
        status = "diagnostic_only"
    return _row(
        row_id="dot295:exp3188_thrml_boundary",
        status=status,
        source_artifact=EXP3188_REL_PATH.as_posix(),
        source_field="thrml_factor_graph_api_boundary_v1_ready",
        evidence_class="thrml_factor_graph_api_boundary_v1",
        claim_scope="hardware_api_boundary_diagnostic",
        summary={
            "thrml_factor_graph_api_boundary_v1_ready": payload.get(
                "thrml_factor_graph_api_boundary_v1_ready"
            )
            is True,
            "thrml_import_available": payload.get("thrml_import_available") is True,
            "thrml_version": str(payload.get("thrml_version") or ""),
            "local_api_smoke_passed": payload.get("local_api_smoke_passed") is True,
            "hardware_speedup_claim_allowed": payload.get("hardware_speedup_claim_allowed") is True,
            "kona_or_tsu_execution_claimed": payload.get("kona_or_tsu_execution_claimed") is True,
            "selected_exact_row_count": len(_as_list(payload.get("selected_exact_rows"))),
            "api_gap_count": len(_as_list(payload.get("api_gap_records"))),
            "source_error_count": len(_as_list(payload.get("source_errors"))),
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
        "row_origin": str(row.get("row_origin") or "matrix_v28"),
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
        "row_origin": "milestone_295",
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


def _prior_publication_blocker_count(matrix: Mapping[str, Any]) -> int:
    return _int_or_none(matrix.get("publication_blocker_count")) or 0


def _missing_artifacts(
    matrix: Mapping[str, Any],
    sources: list[Mapping[str, Any]],
) -> tuple[list[JsonDict], JsonDict]:
    missing: list[JsonDict] = []
    for row in _as_list(matrix.get("missing_artifacts")):
        if isinstance(row, Mapping):
            missing.append(
                {
                    "path": str(row.get("path") or ""),
                    "experiment_id": str(row.get("experiment_id") or ""),
                    "reason": "carried_forward_unresolved_missing_artifact_from_v28",
                }
            )
    new_missing: list[str] = []
    for row in sources:
        if row.get("required") is True or row.get("readable_json_object") is True:
            continue
        path = str(row["path"])
        new_missing.append(path)
        missing.append(
            {
                "path": path,
                "experiment_id": str(row["experiment_id"]),
                "reason": "missing_expected_dot295_artifact",
            }
        )
    v28_missing_count = len(_as_list(matrix.get("missing_artifacts")))
    comparison = {
        "v28_missing_artifact_count": v28_missing_count,
        "v29_missing_artifact_count": len(missing),
        "missing_artifact_delta_from_v28": len(missing) - v28_missing_count,
        "new_missing_dot295_artifacts": new_missing,
    }
    return missing, comparison


def _required_source_errors(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {"path": str(row["path"]), "reason": "missing_or_malformed_required_artifact"}
        for row in sources
        if row.get("required") is True and row.get("readable_json_object") is not True
    ]


def _verifier_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    smoke = payloads["exp3179"]
    invariance = payloads["exp3180"]
    rerun = payloads["exp3181"]
    rerun_status = statuses.get("dot295:exp3181_clean_live_rerun_v10")
    if rerun_status == "missing":
        return "missing_clean_live_sota_verifier_rerun_v10"
    if rerun_status == "clean":
        return "clean_live_sota_verifier_ready"
    if rerun.get("gated_skip") is True:
        substrate = str(smoke.get("substrate_classification") or "unknown_substrate")
        controlled = (
            "controlled_invariance_passed"
            if invariance.get("controlled_invariance_passed")
            else "controlled_invariance_not_passed"
        )
        return f"gated_skip_{substrate}_flagged_adversarial_{controlled}_exact_authority_only"
    if rerun_status == "flagged":
        return "flagged_adversarial_clean_rerun_not_headline_safe"
    if smoke.get("clean_rerun_allowed") is not True:
        return "blocked_receipt_precondition_clean_rerun_not_allowed"
    return "blocked_live_verifier_not_headline_safe"


def _repair_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    gate = payloads["exp3184"]
    ladder = payloads["exp3185"]
    gate_status = statuses.get("dot295:exp3184_repair_gate_v4")
    ladder_status = statuses.get("dot295:exp3185_repair_ladder_v5")
    if gate_status == "missing":
        return "missing_repair_gate_decision_v4"
    if gate_status == "clean" and ladder_status == "clean":
        return "repair_ready"
    if (
        str(gate.get("repair_gate_state") or "").startswith("blocked_receipt")
        and ladder.get("gated_skip") is True
    ):
        suffix = (
            "certificate_expansion_flagged"
            if _row_statuses(rows).get("dot295:exp3183_certificate_expansion") == "flagged"
            else "certificate_expansion_not_ready"
        )
        return f"blocked_receipt_precondition_repair_ladder_gated_skipped_{suffix}"
    if ladder.get("gated_skip") is True:
        return "blocked_repair_gate_ladder_gated_skipped"
    return "blocked_repair_gate"


def _fr11_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    promotion_status = statuses.get("dot295:exp3186_fr11_promotion_pack")
    drift_status = statuses.get("dot295:exp3187_fr11_drift_replay")
    if promotion_status == "clean" and drift_status == "clean":
        return "controller_memory_promotion_allowed_cross_environment_replay_passed_no_model_weight_update"
    if promotion_status == "clean":
        return "controller_memory_promotion_pack_ready_pending_drift_replay"
    if payloads["exp3186"].get("fr11_controller_memory_promotion_pack_v1_ready") is True:
        return "blocked_fr11_controller_memory_promotion_pack"
    return "missing_or_blocked_fr11_controller_memory_promotion"


def _sidecar_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    sidecar_status = statuses.get("dot295:exp3182_distributional_sidecar")
    if sidecar_status == "clean":
        return "deployed_verifier_sidecar_claim_allowed"
    if sidecar_status == "diagnostic_only":
        return "diagnostic_only_distributional_sidecar_no_deployed_verifier_claim"
    if payloads["exp3182"].get("distributional_ebm_exact_row_sidecar_v1_ready") is True:
        return "blocked_distributional_sidecar"
    return "missing_distributional_sidecar"


def _hardware_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    statuses = _row_statuses(rows)
    boundary_status = statuses.get("dot295:exp3188_thrml_boundary")
    boundary = payloads["exp3188"]
    if boundary_status == "diagnostic_only":
        return "diagnostic_only_thrml_local_api_smoke_no_kv260_speedup_no_tsu_kona_execution"
    if (
        boundary.get("hardware_speedup_claim_allowed") is True
        or boundary.get("kona_or_tsu_execution_claimed") is True
    ):
        return "blocked_unsupported_thrml_or_hardware_claim"
    if boundary_status == "missing":
        return "missing_thrml_factor_graph_api_boundary"
    return "blocked_thrml_factor_graph_api_boundary"


def _next_top_gap(payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    statuses = _row_statuses(rows)
    if payloads["exp3179"].get("clean_rerun_allowed") is not True:
        return "full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    if statuses.get("dot295:exp3181_clean_live_rerun_v10") != "clean":
        return "clean_live_verifier_adversarial_flag_clearance"
    if statuses.get("dot295:exp3184_repair_gate_v4") != "clean":
        return "repair_gate_unblock_live_repair_attempts"
    if statuses.get("dot295:exp3182_distributional_sidecar") != "clean":
        return "deployed_verifier_sidecar_evidence"
    if statuses.get("dot295:exp3188_thrml_boundary") != "clean":
        return "authenticated_hardware_speedup_or_explicit_no_speedup_boundary"
    return "publication_blocker_retirement_review"


def _paper_readiness_implications(
    rows: list[Mapping[str, Any]],
    publication_blocker_count: int,
    narrowing_preserved: bool,
) -> JsonDict:
    statuses = _row_statuses(rows)
    blocked: list[str] = []
    if statuses.get("dot295:exp3181_clean_live_rerun_v10") != "clean":
        blocked.append("live_verifier_headline")
    if (
        statuses.get("dot295:exp3184_repair_gate_v4") != "clean"
        or statuses.get("dot295:exp3185_repair_ladder_v5") != "clean"
    ):
        blocked.append("repair_headline")
    if statuses.get("dot295:exp3182_distributional_sidecar") != "clean":
        blocked.append("deployed_verifier_sidecar")
    if statuses.get("dot295:exp3188_thrml_boundary") != "clean":
        blocked.append("hardware_speedup")
    if not narrowing_preserved:
        blocked.append("paper_v6_narrowing")
    return {
        "paper_ready": publication_blocker_count == 0 and not blocked,
        "publication_blocker_count": publication_blocker_count,
        "blocked_headline_claims": blocked,
    }


def _paper_v6_narrowing(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    sidecar = payloads["exp3182"]
    promotion = payloads["exp3186"]
    drift = payloads["exp3187"]
    hardware = payloads["exp3188"]
    return {
        "kv260_speedup_claimed": hardware.get("hardware_speedup_claim_allowed") is True,
        "tsu_or_kona_execution_claimed": hardware.get("kona_or_tsu_execution_claimed") is True,
        "deployed_verifier_sidecar_claimed": sidecar.get("deployed_verifier_claim_allowed") is True,
        "model_weight_self_learning_claimed": (
            promotion.get("no_model_weight_update_claimed") is False
            or drift.get("no_model_weight_update_claimed") is False
        ),
        "paper_ready_streak_claimed": False,
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
            "sha256": row.get("sha256"),
            "source_type": str(row.get("source_type") or "json"),
        }
        for row in sources
    ]


def _invariant_violations(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    archive: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blockers: list[Mapping[str, Any]],
    required_source_errors: list[Mapping[str, Any]],
    narrowing_preserved: bool,
) -> list[str]:
    violations: list[str] = []
    if required_source_errors:
        violations.append("required source artifacts missing or malformed")
    if matrix and matrix.get("matrix_v28_ready") is not True:
        violations.append("matrix v28 authority is not ready")
    if capstone and capstone.get("capstone_v294_ready") is not True:
        violations.append("capstone v294 authority is not ready")
    if archive and archive.get("archive_v294_activate_v295_ready") is not True:
        violations.append("archive v295 handoff is not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v29 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to rows_total")
    if len(publication_blockers) != sum(
        count for status, count in status_counts.items() if status in PUBLICATION_BLOCKING_STATUSES
    ):
        violations.append("publication_blocker_count does not match row statuses")
    if not narrowing_preserved:
        violations.append("paper-v6 narrowing was not preserved")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot295_artifacts",
        "source": "matrix_v28_capstone_v294_archive_v295_and_dot295_artifacts",
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
    if artifact.get("cross_corpus_matrix_v29_ready") is not True:
        return (
            "blocked_matrix_v29_preconditions: "
            f"required_source_errors={len(_as_list(artifact.get('required_source_errors')))}; "
            f"invariant_violations={len(_as_list(artifact.get('invariant_violations')))}"
        )
    return (
        "complete: cross_corpus_matrix_v29_ready=true; "
        f"prior_matrix_version={artifact.get('prior_matrix_version')}; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v28={artifact.get('blocker_delta_from_v28')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
