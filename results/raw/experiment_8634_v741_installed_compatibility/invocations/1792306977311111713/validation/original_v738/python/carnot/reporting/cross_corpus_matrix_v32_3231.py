"""Build the Exp 3231 cross-corpus matrix v32 artifact.

Spec refs: REQ-REPORT-3231, SCENARIO-REPORT-3231.

Matrix v32 is claim accounting for milestone `.298`.  It reads the prior v31
matrix, the checked-in `.298` result JSON files, and conductor gate evidence.
It does not rerun models or repairs because this artifact's job is to keep the
publication boundary honest when upstream evidence is missing or blocked.
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
RUN_DATE = "20260528"
MILESTONE = "2026.05.298"
SCHEMA_VERSION = "carnot.cross_corpus_matrix.v32_298_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3231"
ARTIFACT = "experiment_3231_cross_corpus_matrix_v32"
MATRIX_VERSION = "v32"
PREVIOUS_MATRIX_REL_PATH = Path("results/experiment_3217_cross_corpus_matrix_v31.json")
OUTPUT_REL_PATH = Path("results/experiment_3231_cross_corpus_matrix_v32.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3231_cross_corpus_matrix_v32.py"
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
PREVIOUS_NEXT_TOP_GAP = (
    "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
)
NEXT_TOP_GAP = "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt"

EXP3219_REL_PATH = Path("results/experiment_3219_archive_v297_activate_v298.json")
EXP3220_REL_PATH = Path("results/experiment_3220_hermetic_cuda_runtime_repair_ledger_v1.json")
EXP3221_REL_PATH = Path("results/experiment_3221_llama_cpp_cuda_offload_receipt_smoke_v1.json")
EXP3222_REL_PATH = Path("results/experiment_3222_full_local_sota_receipt_v6.json")
EXP3223_REL_PATH = Path(
    "results/experiment_3223_distributional_ebm_exact_row_uncertainty_sidecar_v2.json"
)
EXP3224_REL_PATH = Path(
    "results/experiment_3224_logitext_partial_smt_context_coverage_pilot_v1.json"
)
EXP3225_REL_PATH = Path("results/experiment_3225_clean_live_sota_verifier_rerun_v13.json")
EXP3226_REL_PATH = Path("results/experiment_3226_structured_repair_proposal_preflight_v2.json")
EXP3227_REL_PATH = Path("results/experiment_3227_repair_gate_decision_v7.json")
EXP3228_REL_PATH = Path("results/experiment_3228_multi_turn_repair_ladder_v8.json")
EXP3229_REL_PATH = Path("results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json")
EXP3230_REL_PATH = Path("results/experiment_3230_kan_cl_certificate_boundary_audit_v2.json")

STATUSES = ("complete", "blocked", "gate_blocked", "missing", "partial")
PUBLICATION_BLOCKING_STATUSES = {"blocked", "gate_blocked", "missing"}


@dataclass(frozen=True)
class SourceSpec:
    """One `.298` artifact that matrix v32 must account for even when absent."""

    experiment_id: str
    path: Path
    role: str
    source_field: str
    gate_title: str = ""


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec("exp3219", EXP3219_REL_PATH, "archive_v297_activate_v298", "activation_ready"),
    SourceSpec(
        "exp3220", EXP3220_REL_PATH, "hermetic_cuda_runtime_repair", "cuda_receipt_ready_candidate"
    ),
    SourceSpec(
        "exp3221",
        EXP3221_REL_PATH,
        "llama_cpp_cuda_offload_receipt_smoke",
        "cuda_receipt_ready",
        "llama.cpp CUDA offload receipt smoke gated on herm",
    ),
    SourceSpec(
        "exp3222",
        EXP3222_REL_PATH,
        "full_local_sota_receipt_v6",
        "clean_rerun_allowed",
        "Full local SOTA GGUF receipt v6 gated on llama.cpp",
    ),
    SourceSpec(
        "exp3223", EXP3223_REL_PATH, "exact_row_uncertainty_sidecar", "uncertainty_sidecar_ready"
    ),
    SourceSpec("exp3224", EXP3224_REL_PATH, "partial_smt_context_coverage", "coverage_ready"),
    SourceSpec(
        "exp3225",
        EXP3225_REL_PATH,
        "clean_live_sota_verifier_v13",
        "clean_verifier_ready",
        "Clean live SOTA verifier rerun v13 using exact-row",
    ),
    SourceSpec(
        "exp3226",
        EXP3226_REL_PATH,
        "structured_repair_proposal_preflight_v2",
        "ready_for_repair_gate",
        "Structured repair proposal preflight v2 with schem",
    ),
    SourceSpec("exp3227", EXP3227_REL_PATH, "repair_gate_decision_v7", "repair_gate_state"),
    SourceSpec(
        "exp3228",
        EXP3228_REL_PATH,
        "multi_turn_repair_ladder_v8",
        "repair_ladder_complete",
        "Multi-turn repair ladder v8 gated on repair gate u",
    ),
    SourceSpec(
        "exp3229",
        EXP3229_REL_PATH,
        "fr11_nonforgetting_promotion_controller_v3",
        "promotion_allowed",
    ),
    SourceSpec(
        "exp3230",
        EXP3230_REL_PATH,
        "kan_cl_certificate_boundary_audit_v2",
        "certificate_boundary_ready",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence defensively so a bad artifact cannot become success."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a checksum for every readable source row in the matrix."""

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
    """REQ-REPORT-3231: aggregate v32 from checked-in `.298` evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    previous_matrix = read_json_object(root_path / PREVIOUS_MATRIX_REL_PATH)
    conductor_log = _read_text(root_path / CONDUCTOR_LOG_REL_PATH)
    sources = [_source_payload(root_path, spec, conductor_log) for spec in SOURCE_SPECS]
    rows = [_public_source(source) for source in sources]
    payloads = {source["experiment_id"]: _as_mapping(source.get("payload")) for source in sources}
    status_counts = _status_counts(rows)
    prior_count = _prior_publication_blocker_count(previous_matrix)
    blockers = _publication_blockers(rows, payloads)
    publication_blocker_count = prior_count + len(blockers)
    blocker_delta = publication_blocker_count - prior_count
    local_sota_receipt_state = _local_sota_receipt_state(payloads, rows)
    clean_verifier_state = _clean_verifier_state(rows)
    repair_gate_state = _repair_gate_state(payloads, rows)
    repair_ladder_state = _repair_ladder_state(rows)
    continuous_self_learning_state = _continuous_self_learning_state(payloads)
    hardware_claim_boundary = _hardware_claim_boundary(payloads, rows)
    criteria = _paper_ready_criteria(
        local_sota_receipt_state,
        clean_verifier_state,
        repair_gate_state,
        repair_ladder_state,
        continuous_self_learning_state,
    )
    invariant_violations = _invariant_violations(
        previous_matrix,
        rows,
        status_counts,
        publication_blocker_count,
        prior_count,
        blockers,
    )
    ready = not invariant_violations
    paper_ready = bool(
        ready
        and _all_ready(criteria)
        and publication_blocker_count == 0
        and previous_matrix.get("paper_ready") is True
    )
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_version": MATRIX_VERSION,
        "cross_corpus_matrix_v32_ready": ready,
        "previous_matrix_artifact": PREVIOUS_MATRIX_REL_PATH.as_posix(),
        "previous_matrix_summary": _previous_matrix_summary(previous_matrix),
        "input_artifacts": rows,
        "source_checksums": {
            row["path"]: row["sha256"] for row in rows if row.get("readable_json_object") is True
        },
        "missing_artifacts": [_blocker_record(row) for row in rows if row["status"] == "missing"],
        "gate_blocked_artifacts": _gate_blocked_artifacts(rows),
        "blocked_artifacts": [_blocker_record(row) for row in rows if row["status"] == "blocked"],
        "complete_artifacts": [_blocker_record(row) for row in rows if row["status"] == "complete"],
        "partial_artifacts": [_blocker_record(row) for row in rows if row["status"] == "partial"],
        "status_counts": status_counts,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v31": blocker_delta,
        "blocker_delta_explanation": blockers,
        "publication_blocker_accounting": {
            "previous_matrix": PREVIOUS_MATRIX_REL_PATH.as_posix(),
            "previous_publication_blocker_count": prior_count,
            "new_blocking_artifact_count": len(blockers),
            "publication_blocker_count": publication_blocker_count,
        },
        "local_sota_receipt_state": local_sota_receipt_state,
        "clean_verifier_state": clean_verifier_state,
        "repair_gate_state": repair_gate_state,
        "repair_ladder_state": repair_ladder_state,
        "continuous_self_learning_state": continuous_self_learning_state,
        "hardware_claim_boundary": hardware_claim_boundary,
        "paper_ready_criteria": criteria,
        "paper_ready": paper_ready,
        "next_top_gap": _next_top_gap(
            local_sota_receipt_state,
            clean_verifier_state,
            repair_gate_state,
            repair_ladder_state,
            continuous_self_learning_state,
            hardware_claim_boundary,
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "invariant_violations": invariant_violations,
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
    """Build and persist the Exp 3231 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_payload(root: Path, spec: SourceSpec, conductor_log: str = "") -> JsonDict:
    path = root / spec.path
    payload = read_json_object(path)
    status, rationale = _classify(spec, payload)
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "role": spec.role,
        "source_field": spec.source_field,
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "payload": payload,
        "status": status,
        "status_rationale": rationale,
        "sha256": sha256_file(path),
        "gated_skip_evidence": _gate_evidence(spec.experiment_id, conductor_log, payload),
    }


def _classify(spec: SourceSpec, payload: Mapping[str, Any]) -> tuple[str, str]:
    if not payload:
        return "missing", "expected `.298` artifact is absent or malformed"
    if _is_gate_blocked(payload):
        return "gate_blocked", "artifact was blocked by a conductor pre-gate"
    exp = spec.experiment_id
    if exp == "exp3219":
        return _status_pair(
            payload.get("activation_ready") is True,
            "archive activation confirmed",
            "archive activation is not ready",
        )
    if exp == "exp3220":
        return _status_pair(
            payload.get("cuda_receipt_ready_candidate") is True,
            "hermetic CUDA runtime candidate is ready",
            "hermetic CUDA runtime remains blocked",
        )
    if exp == "exp3221":
        return _status_pair(
            payload.get("cuda_receipt_ready") is True
            or payload.get("offload_receipt_ready") is True,
            "llama.cpp CUDA offload receipt is ready",
            "llama.cpp CUDA offload receipt is not ready",
        )
    if exp == "exp3222":
        return _status_pair(
            payload.get("clean_rerun_allowed") is True,
            "full local SOTA receipt allows clean rerun",
            "full local SOTA receipt does not allow clean rerun",
        )
    if exp == "exp3223":
        return _status_pair(
            payload.get("uncertainty_sidecar_ready") is True
            and payload.get("exact_verifier_authority_preserved") is True,
            "exact-row uncertainty sidecar is ready as triage metadata",
            "exact-row uncertainty sidecar is not ready",
        )
    if exp == "exp3224":
        if payload.get("coverage_ready") is not True:
            return "blocked", "partial SMT coverage pilot is not ready"
        if _int_value(payload.get("partially_formalizable_count")) > 0:
            return "partial", "partial SMT coverage ready without full extraction claim"
        return "complete", "SMT coverage pilot ready"
    if exp == "exp3225":
        return _status_pair(
            payload.get("clean_verifier_ready") is True
            or payload.get("headline_claim_allowed") is True,
            "clean live SOTA verifier evidence is ready",
            "clean live SOTA verifier evidence is not ready",
        )
    if exp == "exp3226":
        return _status_pair(
            payload.get("ready_for_repair_gate") is True,
            "structured repair preflight is ready for gate use",
            "structured repair preflight is not ready",
        )
    if exp == "exp3227":
        return _status_pair(
            str(payload.get("repair_gate_state") or "").startswith("unblocked"),
            "repair gate is unblocked",
            "repair gate remains blocked",
        )
    if exp == "exp3228":
        return _status_pair(
            payload.get("repair_ladder_complete") is True
            or payload.get("repair_ladder_ready") is True,
            "repair ladder completed",
            "repair ladder is not complete",
        )
    if exp == "exp3229":
        if _model_weight_update_claimed(payload):
            return "blocked", "FR-11 controller claims a model-weight update"
        return _status_pair(
            payload.get("promotion_allowed") is True
            and payload.get("controller_memory_promotion_allowed") is True
            and payload.get("nonforgetting_budget_exceeded") is not True
            and _int_value(payload.get("negative_control_regression_count")) == 0,
            "FR-11 controller-memory promotion is allowed without training",
            "FR-11 controller-memory promotion is not allowed",
        )
    if exp == "exp3230":
        if _model_weight_update_claimed(payload):
            return "blocked", "KAN-CL boundary audit claims a model-weight update"
        return (
            "complete",
            "certificate boundary audit complete; sidecar promotion may still be blocked",
        )
    return "blocked", "unknown expected `.298` source"


def _status_pair(condition: bool, complete_reason: str, blocked_reason: str) -> tuple[str, str]:
    return ("complete", complete_reason) if condition else ("blocked", blocked_reason)


def _public_source(source: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(source.get("payload"))
    status = _normal_status(str(source.get("status") or "missing"))
    gate_evidence = _as_mapping(source.get("gated_skip_evidence"))
    row = {
        "experiment_id": str(source.get("experiment_id") or ""),
        "path": str(source.get("path") or ""),
        "role": str(source.get("role") or ""),
        "source_field": str(source.get("source_field") or ""),
        "present": source.get("present") is True,
        "readable_json_object": source.get("readable_json_object") is True,
        "status": status,
        "status_rationale": str(source.get("status_rationale") or ""),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "reported_experiment_id": _experiment_id(payload, str(source.get("experiment_id") or "")),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": source.get("sha256"),
        "gated_skip_evidence": gate_evidence,
        "summary": _payload_summary(payload),
    }
    row["publication_blocker"] = _row_blocks_publication(row, payload)
    return row


def _payload_summary(payload: Mapping[str, Any]) -> JsonDict:
    summary_keys = (
        "activation_ready",
        "cuda_receipt_ready_candidate",
        "cuda_receipt_ready",
        "clean_rerun_allowed",
        "uncertainty_sidecar_ready",
        "coverage_ready",
        "clean_verifier_ready",
        "ready_for_repair_gate",
        "repair_gate_state",
        "repair_ladder_allowed",
        "promotion_allowed",
        "controller_memory_promotion_allowed",
        "accepted_trace_count",
        "certificate_boundary_ready",
        "kan_sidecar_promotion_allowed",
        "missing_certificate_count",
        "model_weight_update_claimed",
        "status",
        "blocked_at_layer",
    )
    return {key: payload.get(key) for key in summary_keys if key in payload}


def _previous_matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "cross_corpus_matrix_v31_ready": matrix.get("cross_corpus_matrix_v31_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count": _prior_publication_blocker_count(matrix),
        "next_top_gap": str(matrix.get("next_top_gap") or ""),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _publication_blockers(
    rows: list[Mapping[str, Any]], payloads: Mapping[str, Mapping[str, Any]]
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    for row in rows:
        payload = payloads.get(str(row.get("experiment_id") or ""), {})
        if _row_blocks_publication(row, payload):
            blockers.append(_blocker_record(row))
    return blockers


def _row_blocks_publication(row: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    status = _normal_status(str(row.get("status") or "missing"))
    if status in PUBLICATION_BLOCKING_STATUSES:
        return True
    return (
        str(row.get("experiment_id") or "") == "exp3230"
        and payload.get("certificate_boundary_ready") is not True
    )


def _blocker_record(row: Mapping[str, Any]) -> JsonDict:
    record = {
        "experiment_id": str(row.get("experiment_id") or ""),
        "path": str(row.get("path") or ""),
        "role": str(row.get("role") or ""),
        "status": _normal_status(str(row.get("status") or "missing")),
        "source_field": str(row.get("source_field") or ""),
        "status_rationale": str(row.get("status_rationale") or ""),
    }
    gate_evidence = _as_mapping(row.get("gated_skip_evidence"))
    if gate_evidence.get("status") == "gate_blocked":
        record["gated_skip_evidence"] = gate_evidence
    return record


def _gate_blocked_artifacts(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    records: list[JsonDict] = []
    for row in rows:
        gate_evidence = _as_mapping(row.get("gated_skip_evidence"))
        if row.get("status") == "gate_blocked" or gate_evidence.get("status") == "gate_blocked":
            records.append(_blocker_record(row))
    return records


def _local_sota_receipt_state(
    payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]
) -> str:
    receipt = payloads["exp3222"]
    if receipt.get("clean_rerun_allowed") is True and _row_status(rows, "exp3222") == "complete":
        return "clean_rerun_allowed_full_local_sota_receipt_v6"
    if _row_status(rows, "exp3222") == "missing":
        if _row_status(rows, "exp3221") == "gate_blocked":
            return "missing_full_local_sota_receipt_v6_after_exp3221_gate_blocked"
        return "missing_full_local_sota_receipt_v6"
    if _row_status(rows, "exp3222") == "gate_blocked":
        return "gate_blocked_full_local_sota_receipt_v6_no_clean_rerun_allowed"
    return "blocked_full_local_sota_receipt_v6_no_clean_rerun_allowed"


def _clean_verifier_state(rows: list[Mapping[str, Any]]) -> str:
    status = _row_status(rows, "exp3225")
    if status == "complete":
        return "clean_live_sota_verifier_v13_ready"
    if status == "gate_blocked" and _row_status(rows, "exp3222") == "missing":
        return "gate_blocked_on_missing_full_local_sota_receipt_v6_no_clean_verifier_evidence"
    if status == "missing":
        return "missing_clean_live_sota_verifier_v13"
    return "blocked_clean_live_sota_verifier_v13"


def _repair_gate_state(
    payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]
) -> str:
    status = _row_status(rows, "exp3227")
    gate = payloads["exp3227"]
    if status == "missing":
        return "missing_repair_gate_v7"
    if str(gate.get("repair_gate_state") or "").startswith("unblocked"):
        return "unblocked_v7"
    return f"blocked_v7_blocker_count_{_int_value(gate.get('blocker_count'))}"


def _repair_ladder_state(rows: list[Mapping[str, Any]]) -> str:
    status = _row_status(rows, "exp3228")
    if status == "complete":
        return "complete_repair_ladder_v8"
    if status == "gate_blocked":
        return "gate_blocked_repair_gate_v7_blocked"
    if status == "missing":
        return "missing_repair_ladder_v8"
    return "blocked_repair_ladder_v8"


def _continuous_self_learning_state(payloads: Mapping[str, Mapping[str, Any]]) -> str:
    controller = payloads["exp3229"]
    boundary = payloads["exp3230"]
    if not controller or not boundary:
        return "missing_fr11_nonforgetting_or_certificate_boundary_artifacts"
    if _model_weight_update_claimed(controller) or _model_weight_update_claimed(boundary):
        return "blocked_model_weight_update_claimed"
    if (
        controller.get("promotion_allowed") is True
        and boundary.get("certificate_boundary_ready") is True
    ):
        return (
            "controller_memory_promotion_allowed_no_model_weight_update_certificate_boundary_ready"
        )
    if controller.get("promotion_allowed") is True:
        return (
            "controller_memory_promotion_allowed_"
            f"{_int_value(controller.get('accepted_trace_count'))}_accepted_no_model_weight_update_"
            f"kan_sidecar_blocked_missing_certificates_{_int_value(boundary.get('missing_certificate_count'))}"
        )
    return "blocked_fr11_nonforgetting_promotion_controller"


def _hardware_claim_boundary(
    payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]
) -> str:
    runtime = payloads["exp3220"]
    if (
        runtime.get("nvidia_smi_available") is True
        and runtime.get("cuda_receipt_ready_candidate") is not True
    ):
        return (
            "cuda_runtime_visible_but_not_usable_no_llama_cpp_offload_receipt_"
            "no_hardware_speedup_tsu_or_kona_claim_allowed"
        )
    return "no_authenticated_runtime_or_hardware_evidence_no_speedup_tsu_or_kona_claim_allowed"


def _paper_ready_criteria(
    local_sota_receipt_state: str,
    clean_verifier_state: str,
    repair_gate_state: str,
    repair_ladder_state: str,
    continuous_self_learning_state: str,
) -> dict[str, bool]:
    return {
        "local_sota_receipt": local_sota_receipt_state.startswith("clean_rerun_allowed"),
        "clean_verifier": clean_verifier_state == "clean_live_sota_verifier_v13_ready",
        "repair": repair_gate_state.startswith("unblocked")
        and repair_ladder_state.startswith("complete"),
        "fr11": continuous_self_learning_state.startswith("controller_memory_promotion_allowed")
        and "no_model_weight_update" in continuous_self_learning_state,
        "claim_boundary": "certificate_boundary_ready" in continuous_self_learning_state,
    }


def _all_ready(criteria: Mapping[str, bool]) -> bool:
    return all(criteria.values())


def _next_top_gap(
    local_sota_receipt_state: str,
    clean_verifier_state: str,
    repair_gate_state: str,
    repair_ladder_state: str,
    continuous_self_learning_state: str,
    hardware_claim_boundary: str,
) -> str:
    if not local_sota_receipt_state.startswith("clean_rerun_allowed"):
        return NEXT_TOP_GAP
    if clean_verifier_state != "clean_live_sota_verifier_v13_ready":
        return "clean_live_verifier_v13_gate_clearance"
    if not (
        repair_gate_state.startswith("unblocked") and repair_ladder_state.startswith("complete")
    ):
        return "repair_gate_v7_unblock_and_ladder_v8_execution"
    if "certificate_boundary_ready" not in continuous_self_learning_state:
        return "fr11_certificate_boundary_for_sidecar_promotion"
    if hardware_claim_boundary != "authenticated_hardware_claim_allowed":
        return "authenticated_hardware_claim_boundary_or_explicit_no_speedup_disclosure"
    return "publication_blocker_retirement_review"


def _invariant_violations(
    previous_matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blocker_count: int,
    prior_count: int,
    blockers: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if previous_matrix.get("cross_corpus_matrix_v31_ready") is not True:
        violations.append("previous matrix v31 is missing or not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v32 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to input artifact rows")
    if publication_blocker_count != prior_count + len(blockers):
        violations.append("publication_blocker_count does not reconcile with v31 delta")
    return violations


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("cross_corpus_matrix_v32_ready") is not True:
        return "blocked_matrix_v32_preconditions: " + "; ".join(
            str(item) for item in artifact.get("invariant_violations", [])
        )
    return (
        "complete: cross_corpus_matrix_v32_ready=true; "
        f"previous_matrix_artifact={artifact['previous_matrix_artifact']}; "
        f"paper_ready={str(artifact['paper_ready']).lower()}; "
        f"publication_blocker_count={artifact['publication_blocker_count']}; "
        f"blocker_delta_from_v31={artifact['blocker_delta_from_v31']}; "
        f"next_top_gap={artifact['next_top_gap']}"
    )


def _gate_evidence(
    experiment_id: str, conductor_log: str, payload: Mapping[str, Any] | None = None
) -> JsonDict:
    payload_map = _as_mapping(payload)
    if _is_gate_blocked(payload_map):
        return {
            "status": "gate_blocked",
            "source": "artifact",
            "summary": str(payload_map.get("gate_check_summary") or ""),
        }
    title = next(
        (spec.gate_title for spec in SOURCE_SPECS if spec.experiment_id == experiment_id), ""
    )
    if not title or not conductor_log:
        return {"status": "absent"}
    for line in conductor_log.splitlines():
        if title in line and "GATE_BLOCK" in line:
            return {
                "status": "gate_blocked",
                "source": CONDUCTOR_LOG_REL_PATH.as_posix(),
                "line": line.strip(),
            }
    return {"status": "absent"}


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_")
    return normalized if normalized in STATUSES else "missing"


def _row_status(rows: list[Mapping[str, Any]], experiment_id: str) -> str:
    for row in rows:
        if row.get("experiment_id") == experiment_id:
            return _normal_status(str(row.get("status") or "missing"))
    return "missing"


def _is_gate_blocked(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("schema") == "blocked_gate_check_v1"
        or payload.get("blocked_at_layer") == "conductor_pre_gate"
        or str(payload.get("honest_verdict") or "").startswith("blocked_gate_check")
    )


def _prior_publication_blocker_count(matrix: Mapping[str, Any]) -> int:
    value = matrix.get("publication_blocker_count")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _model_weight_update_claimed(payload: Mapping[str, Any]) -> bool:
    if payload.get("model_weight_update_claimed") is True:
        return True
    if payload.get("model_weight_update_performed") is True:
        return True
    substrate = _as_mapping(payload.get("inference_substrate"))
    return any(
        substrate.get(key) is True
        for key in (
            "base_model_weights_updated",
            "hidden_state_mutation_claimed",
            "model_weight_learning",
            "model_weight_mutation",
            "model_weight_training",
            "kan_model_weight_training",
        )
    )


def _experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    value = payload.get("experiment_id")
    if value:
        return str(value)
    experiment = payload.get("experiment")
    if isinstance(experiment, int) and not isinstance(experiment, bool):
        return f"exp{experiment}"
    return fallback


def _bool_value(value: Any) -> bool:
    return value if isinstance(value, bool) else False


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return max(0.0, round(end - started_s, 6))
