"""Build the Exp 3203 cross-corpus matrix v30 artifact.

Spec refs: REQ-REPORT-3203, SCENARIO-REPORT-3203.

Matrix v30 is a claim-accounting pass over the `.296` artifacts. It reads the
prior matrix and checked-in source JSONs, then records which parts are clean,
blocked, gate-skipped, diagnostic-only, retired, or missing. It deliberately
does not run models, verifiers, repairs, hardware, or the conductor because a
matrix is only allowed to reconcile evidence that already exists.
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
RUN_DATE = "20260527"
MILESTONE = "2026.05.296"
SCHEMA_VERSION = "carnot.cross_corpus_matrix.v30_296_artifact_aggregation.v1"
EXPERIMENT_ID = "exp3203"
MATRIX_VERSION = "v30"
ARTIFACT = "experiment_3203_cross_corpus_matrix_v30"
OUTPUT_REL_PATH = Path("results/experiment_3203_cross_corpus_matrix_v30.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3203_cross_corpus_matrix_v30.py"

MATRIX_V29_REL_PATH = Path("results/experiment_3189_cross_corpus_matrix_v29.json")
EXP3191_REL_PATH = Path("results/experiment_3191_archive_v295_activate_v296.json")
EXP3192_REL_PATH = Path("results/experiment_3192_receipt_adversarial_contract_v4.json")
EXP3193_REL_PATH = Path("results/experiment_3193_llama_cpp_cuda_offload_health_probe_v1.json")
EXP3194_REL_PATH = Path("results/experiment_3194_clean_live_sota_verifier_rerun_v11.json")
EXP3195_REL_PATH = Path("results/experiment_3195_adaptive_verification_granularity_policy_v1.json")
EXP3196_REL_PATH = Path("results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json")
EXP3197_REL_PATH = Path("results/experiment_3197_exverus_inductive_certificate_expansion_v1.json")
EXP3198_REL_PATH = Path("results/experiment_3198_repair_gate_decision_v5.json")
EXP3199_REL_PATH = Path("results/experiment_3199_multi_turn_repair_ladder_v6.json")
EXP3200_REL_PATH = Path("results/experiment_3200_fr11_verify_trace_memory_controller_v1.json")
EXP3201_REL_PATH = Path("results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json")
EXP3202_REL_PATH = Path("results/experiment_3202_sparse_potts_paoa_thrml_factor_boundary_v1.json")

STATUSES = ("clean", "blocked", "gated_skipped", "diagnostic_only", "retired", "missing")
PUBLICATION_BLOCKING_STATUSES = {"blocked", "gated_skipped", "missing"}


@dataclass(frozen=True)
class SourceSpec:
    """A checked-in source artifact that v30 must account for explicitly."""

    experiment_id: str
    path: Path
    role: str
    row_id: str
    source_field: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "exp3189",
        MATRIX_V29_REL_PATH,
        "matrix_v29_authority",
        "authority:exp3189_matrix_v29",
        "cross_corpus_matrix_v29_ready",
    ),
    SourceSpec(
        "exp3191",
        EXP3191_REL_PATH,
        "archive_v295_activate_v296",
        "dot296:exp3191_archive_activation",
        "activation_ready",
    ),
    SourceSpec(
        "exp3192",
        EXP3192_REL_PATH,
        "receipt_adversarial_contract_v4",
        "dot296:exp3192_receipt_contract_v4",
        "current_evidence_assessment.clean_rerun_allowed",
    ),
    SourceSpec(
        "exp3193",
        EXP3193_REL_PATH,
        "llama_cpp_cuda_offload_health_probe",
        "dot296:exp3193_cuda_offload_health_probe",
        "clean_rerun_allowed",
    ),
    SourceSpec(
        "exp3194",
        EXP3194_REL_PATH,
        "clean_live_sota_verifier_rerun_v11",
        "dot296:exp3194_clean_verifier_rerun_v11",
        "status",
    ),
    SourceSpec(
        "exp3195",
        EXP3195_REL_PATH,
        "adaptive_verification_granularity_policy",
        "dot296:exp3195_adaptive_granularity_policy",
        "adaptive_verification_granularity_policy_v1_ready",
    ),
    SourceSpec(
        "exp3196",
        EXP3196_REL_PATH,
        "gencp_domain_preview_repair_compiler",
        "dot296:exp3196_domain_preview_compiler",
        "repair_call_ready",
    ),
    SourceSpec(
        "exp3197",
        EXP3197_REL_PATH,
        "exverus_inductive_certificate_expansion",
        "dot296:exp3197_inductive_certificate_expansion",
        "repair_call_ready",
    ),
    SourceSpec(
        "exp3198",
        EXP3198_REL_PATH,
        "repair_gate_decision_v5",
        "dot296:exp3198_repair_gate_v5",
        "repair_gate_state",
    ),
    SourceSpec(
        "exp3199",
        EXP3199_REL_PATH,
        "multi_turn_repair_ladder_v6",
        "dot296:exp3199_repair_ladder_v6",
        "status",
    ),
    SourceSpec(
        "exp3200",
        EXP3200_REL_PATH,
        "fr11_verify_trace_memory_controller",
        "dot296:exp3200_fr11_trace_memory_controller",
        "promotion_allowed",
    ),
    SourceSpec(
        "exp3201",
        EXP3201_REL_PATH,
        "kan_cl_nonforgetting_sidecar_audit",
        "dot296:exp3201_kan_cl_sidecar_audit",
        "sidecar_promotion_allowed",
    ),
    SourceSpec(
        "exp3202",
        EXP3202_REL_PATH,
        "sparse_potts_paoa_thrml_factor_boundary",
        "dot296:exp3202_sparse_potts_thrml_boundary",
        "speedup_claim_allowed",
    ),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a source JSON object and fail closed for absent or malformed data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Checksum source evidence so matrix lineage can be independently checked."""

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
    """REQ-REPORT-3203: aggregate matrix v30 from checked-in evidence only."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = [_source_payload(root_path, spec) for spec in SOURCE_SPECS]
    payloads = {str(row["experiment_id"]): _as_mapping(row.get("payload")) for row in sources}
    rows = [_classify_source(row) for row in sources]
    status_counts = _status_counts(rows)
    matrix = payloads["exp3189"]
    prior_count = _prior_publication_blocker_count(matrix)
    new_blockers = _new_publication_blockers(rows)
    publication_blocker_count = (
        prior_count + len(new_blockers) if prior_count is not None else len(new_blockers)
    )
    missing_artifacts = [
        {
            "experiment_id": str(row["experiment_id"]),
            "path": str(row["path"]),
            "reason": "missing_or_malformed_expected_dot296_artifact",
        }
        for row in sources
        if row.get("readable_json_object") is not True
    ]
    local_sota_receipt_status = _local_sota_receipt_status(payloads)
    clean_verifier_status = _clean_verifier_status(rows)
    repair_status = _repair_status(payloads, rows)
    fr11_self_learning_status = _fr11_self_learning_status(payloads, rows)
    hardware_sampler_status = _hardware_sampler_status(payloads, rows)
    blocked_required = _required_evidence_blocked_or_missing(
        local_sota_receipt_status,
        clean_verifier_status,
        repair_status,
        fr11_self_learning_status,
        hardware_sampler_status,
    )
    paper_v6_narrowing = _paper_v6_narrowing(payloads)
    paper_v6_narrowing_preserved = not any(paper_v6_narrowing.values())
    paper_ready = (
        not blocked_required
        and publication_blocker_count == 0
        and paper_v6_narrowing_preserved
        and matrix.get("paper_ready") is True
    )
    invariant_violations = _invariant_violations(
        matrix,
        rows,
        status_counts,
        publication_blocker_count,
        prior_count,
        new_blockers,
    )
    ready = not invariant_violations
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_version": MATRIX_VERSION,
        "cross_corpus_matrix_v30_ready": ready,
        "prior_matrix_version": "v29",
        "prior_matrix_artifact": MATRIX_V29_REL_PATH.as_posix(),
        "source_artifacts_expected": [spec.path.as_posix() for spec in SOURCE_SPECS],
        "source_artifacts_loaded": [
            str(row["path"]) for row in sources if row.get("readable_json_object") is True
        ],
        "source_artifact_records": _public_sources(sources),
        "source_checksums": {
            str(row["path"]): row.get("sha256")
            for row in sources
            if row.get("readable_json_object") is True
        },
        "missing_artifacts": missing_artifacts,
        "missing_artifact_count": len(missing_artifacts),
        "status_counts": status_counts,
        "artifact_classifications": rows,
        "prior_publication_blocker_count": prior_count,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v29": (
            publication_blocker_count - prior_count if prior_count is not None else None
        ),
        "new_publication_blockers": new_blockers,
        "publication_blocker_accounting": {
            "prior_matrix": MATRIX_V29_REL_PATH.as_posix(),
            "prior_publication_blocker_count": prior_count,
            "new_blocking_artifact_count": len(new_blockers),
            "retired_prior_blocker_count": 0,
            "publication_blocker_count": publication_blocker_count,
        },
        "local_sota_receipt_status": local_sota_receipt_status,
        "clean_verifier_status": clean_verifier_status,
        "repair_status": repair_status,
        "fr11_self_learning_status": fr11_self_learning_status,
        "hardware_sampler_status": hardware_sampler_status,
        "required_evidence_blocked_or_missing": blocked_required,
        "paper_ready": paper_ready,
        "paper_v6_narrowing": paper_v6_narrowing,
        "paper_v6_narrowing_preserved": paper_v6_narrowing_preserved,
        "next_top_gap": _next_top_gap(
            local_sota_receipt_status,
            clean_verifier_status,
            repair_status,
            fr11_self_learning_status,
            hardware_sampler_status,
        ),
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "ops_docs_updated": False,
        "status_updates_written": False,
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
    """Build and persist the Exp 3203 deliverable JSON."""

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
    status = "clean" if payload else "missing"
    return {
        "experiment_id": spec.experiment_id,
        "path": spec.path.as_posix(),
        "loaded_path": spec.path.as_posix(),
        "role": spec.role,
        "row_id": spec.row_id,
        "source_field": spec.source_field,
        "source_type": "json",
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "payload": payload,
        "sha256": sha256_file(path),
        "status": status,
    }


def _classify_source(source: Mapping[str, Any]) -> JsonDict:
    payload = _as_mapping(source.get("payload"))
    status, rationale = _classification_status(str(source.get("experiment_id") or ""), payload)
    return {
        "row_id": str(source.get("row_id") or ""),
        "experiment_id": _experiment_id(payload, str(source.get("experiment_id") or "")),
        "expected_experiment_id": str(source.get("experiment_id") or ""),
        "status": status,
        "status_rationale": rationale,
        "source_artifact": str(source.get("path") or ""),
        "source_field": str(source.get("source_field") or ""),
        "role": str(source.get("role") or ""),
        "blocker_class": _blocker_class(status, str(source.get("role") or "")),
        "contract_v4_methodology": _contract_v4_methodology(payload),
        "contract_v4_adversarial": _contract_v4_adversarial(payload),
    }


def _classification_status(experiment_id: str, payload: Mapping[str, Any]) -> tuple[str, str]:
    if not payload:
        return "missing", "expected artifact is absent or malformed"
    if experiment_id == "exp3189":
        if payload.get("cross_corpus_matrix_v29_ready") is True:
            return "clean", "matrix_v29 authority loaded"
        return "blocked", "matrix_v29 authority is not ready"
    if experiment_id == "exp3191":
        if payload.get("activation_ready") is True:
            return "clean", "archive activation authority loaded"
        return "blocked", "archive activation is not ready"
    if experiment_id == "exp3192":
        clean = _bool_field(payload, "current_evidence_assessment.clean_rerun_allowed")
        headline = _bool_field(payload, "current_evidence_assessment.headline_claim_allowed")
        substrate = str(_nested(payload, "current_evidence_assessment.substrate_classification") or "")
        if clean is True and headline is True and substrate == "full_local_sota_receipt":
            return "clean", "contract allows full local SOTA clean rerun"
        return "blocked", "contract keeps clean rerun blocked"
    if experiment_id == "exp3193":
        if (
            payload.get("clean_rerun_allowed") is True
            and payload.get("headline_claim_allowed") is True
            and payload.get("substrate_classification") == "full_local_sota_receipt"
        ):
            return "clean", "CUDA/offload receipt can unlock clean rerun"
        return "blocked", "CUDA/offload receipt does not unlock clean rerun"
    if experiment_id in {"exp3194", "exp3199"}:
        if _is_gate_skip(payload):
            return "gated_skipped", "conductor pre-gate prevented unsafe execution"
        if payload.get("headline_claim_allowed") is True:
            return "clean", "live gated workflow materialized headline-safe evidence"
        return "blocked", "gated workflow is blocked without a structured skip"
    if experiment_id == "exp3195":
        if payload.get("adaptive_verification_granularity_policy_v1_ready") is not True:
            return "blocked", "adaptive policy artifact is not ready"
        if payload.get("promotion_allowed") is True:
            return "clean", "adaptive policy promotion is allowed"
        return "diagnostic_only", "adaptive policy is artifact-only and not promoted"
    if experiment_id == "exp3196":
        if _as_list(payload.get("source_errors")):
            return "blocked", "domain preview compiler reports source errors"
        if payload.get("repair_call_ready") is True:
            return "clean", "domain preview can feed repair calls"
        return "diagnostic_only", "domain preview is bounded but does not unlock repair"
    if experiment_id == "exp3197":
        if _as_list(payload.get("source_errors")):
            return "blocked", "inductive certificate artifact reports source errors"
        if payload.get("repair_call_ready") is True:
            return "clean", "invariant certificates can feed repair calls"
        return "diagnostic_only", "invariant certificates remain pre-repair evidence"
    if experiment_id == "exp3198":
        if str(payload.get("repair_gate_state") or "") == "unblocked_for_bounded_repair_ladder":
            return "clean", "repair gate is unblocked"
        return "blocked", "repair gate remains blocked"
    if experiment_id == "exp3200":
        if (
            payload.get("promotion_allowed") is True
            and payload.get("model_weight_update_performed") is not True
            and _int_value(payload.get("negative_control_regression_count")) == 0
        ):
            return "clean", "FR-11 trace-memory controller promoted without model-weight update"
        return "blocked", "FR-11 trace-memory controller failed a promotion guard"
    if experiment_id == "exp3201":
        if payload.get("sidecar_promotion_allowed") is True:
            return "clean", "KAN-CL sidecar promotion is allowed"
        if (
            payload.get("model_weight_update_performed") is not True
            and _int_value(payload.get("heldout_regression_count")) == 0
            and _int_value(payload.get("drift_regression_count")) == 0
            and _int_value(payload.get("negative_control_regression_count")) == 0
            and _int_value(payload.get("locality_violation_count")) == 0
        ):
            return "diagnostic_only", "KAN-CL sidecar audit passed but did not promote"
        return "blocked", "KAN-CL sidecar audit reports regressions or mutation"
    if experiment_id == "exp3202":
        if _as_list(payload.get("source_errors")):
            return "blocked", "hardware boundary artifact reports source errors"
        if (
            payload.get("authenticated_hardware_transcript_present") is True
            and payload.get("speedup_claim_allowed") is True
        ):
            return "clean", "authenticated hardware speedup claim is allowed"
        if payload.get("speedup_claim_allowed") is True:
            return "blocked", "speedup claim lacks authenticated transcript"
        return "diagnostic_only", "factor boundary denies hardware speedup claims"
    return "blocked", "unknown expected source classification"


def _contract_v4_methodology(payload: Mapping[str, Any]) -> JsonDict:
    checks = {
        "schema_version_present": "schema_version" in payload,
        "experiment_id_present": bool(_experiment_id(payload, "")),
        "source_artifacts_present": "source_artifacts" in payload,
        "source_checksums_present": "source_checksums" in payload,
        "inference_substrate_present": "inference_substrate" in payload,
        "duration_s_present": "duration_s" in payload,
        "honest_verdict_present": bool(str(payload.get("honest_verdict") or "")),
    }
    checks["methodology_field_gaps"] = [
        key.removesuffix("_present") for key, value in checks.items() if value is False
    ]
    return checks


def _contract_v4_adversarial(payload: Mapping[str, Any]) -> JsonDict:
    verdict = str(payload.get("honest_verdict") or "")
    clean_rerun = payload.get("clean_rerun_allowed")
    if clean_rerun is None:
        clean_rerun = _nested(payload, "current_evidence_assessment.clean_rerun_allowed")
    headline = payload.get("headline_claim_allowed")
    if headline is None:
        headline = _nested(payload, "current_evidence_assessment.headline_claim_allowed")
    substrate = payload.get("substrate_classification")
    if substrate is None:
        substrate = _nested(payload, "current_evidence_assessment.substrate_classification")
    return {
        "flagged_adversarial": payload.get("flagged_adversarial"),
        "corrigendum_pending": _corrigendum_pending(payload.get("corrigendum_pending")),
        "clean_rerun_allowed": clean_rerun,
        "headline_claim_allowed": headline,
        "substrate_classification": str(substrate or ""),
        "blocked_verdict": verdict.startswith("blocked_"),
        "complete_verdict": _complete_verdict(verdict),
        "blocker_reasons": _blocker_reasons(payload),
    }


def _local_sota_receipt_status(payloads: Mapping[str, Mapping[str, Any]]) -> str:
    probe = payloads["exp3193"]
    contract = payloads["exp3192"]
    if not probe and not contract:
        return "missing_local_sota_receipt_evidence"
    if (
        probe.get("clean_rerun_allowed") is True
        and probe.get("headline_claim_allowed") is True
        and probe.get("substrate_classification") == "full_local_sota_receipt"
    ):
        return "passed_full_local_sota_receipt_clean_rerun_allowed"
    if probe.get("substrate_classification") == "cuda_unavailable":
        return "blocked_cuda_unavailable_no_full_local_sota_receipt"
    if _nested(contract, "current_evidence_assessment.substrate_classification") == (
        "cpu_fallback_receipt_only"
    ):
        return "blocked_cpu_fallback_receipt_only_non_headline"
    return "blocked_no_full_local_sota_receipt"


def _clean_verifier_status(rows: list[Mapping[str, Any]]) -> str:
    status = _row_status(rows, "dot296:exp3194_clean_verifier_rerun_v11")
    if status == "clean":
        return "clean_live_verifier_ready"
    if status == "gated_skipped":
        return "gated_skipped_clean_verifier_v11_waiting_on_clean_rerun_allowed"
    if status == "missing":
        return "missing_clean_verifier_rerun_v11"
    return "blocked_clean_verifier_v11"


def _repair_status(payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    gate_status = _row_status(rows, "dot296:exp3198_repair_gate_v5")
    ladder_status = _row_status(rows, "dot296:exp3199_repair_ladder_v6")
    if gate_status == "clean" and ladder_status == "clean":
        return "repair_ready"
    if gate_status == "blocked" and ladder_status == "gated_skipped":
        return "blocked_clean_verifier_gate_repair_ladder_gated_skipped"
    if gate_status == "missing" or ladder_status == "missing":
        return "missing_repair_gate_or_ladder"
    if payloads["exp3198"].get("downstream_gated_skip_expected") is True:
        return "blocked_repair_gate_downstream_gated_skip_expected"
    return "blocked_repair"


def _fr11_self_learning_status(
    payloads: Mapping[str, Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> str:
    controller_status = _row_status(rows, "dot296:exp3200_fr11_trace_memory_controller")
    sidecar_status = _row_status(rows, "dot296:exp3201_kan_cl_sidecar_audit")
    if controller_status == "clean" and sidecar_status == "clean":
        return "controller_memory_and_sidecar_promoted_no_model_weight_update"
    if controller_status == "clean" and sidecar_status == "diagnostic_only":
        return "controller_memory_trace_policy_promoted_no_model_weight_update_sidecar_promotion_blocked"
    if controller_status == "missing":
        return "missing_fr11_trace_memory_controller"
    if payloads["exp3200"].get("model_weight_update_performed") is True:
        return "blocked_fr11_model_weight_update_claimed"
    return "blocked_fr11_trace_memory_controller"


def _hardware_sampler_status(payloads: Mapping[str, Mapping[str, Any]], rows: list[Mapping[str, Any]]) -> str:
    status = _row_status(rows, "dot296:exp3202_sparse_potts_thrml_boundary")
    hardware = payloads["exp3202"]
    if status == "clean":
        return "authenticated_hardware_speedup_claim_allowed"
    if status == "diagnostic_only":
        return "diagnostic_only_sparse_potts_thrml_factor_boundary_no_authenticated_speedup"
    if status == "missing":
        return "missing_sparse_potts_thrml_factor_boundary"
    if hardware.get("speedup_claim_allowed") is True:
        return "blocked_unsupported_hardware_speedup_claim"
    return "blocked_hardware_sampler_boundary"


def _required_evidence_blocked_or_missing(
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_status: str,
    fr11_self_learning_status: str,
    hardware_sampler_status: str,
) -> list[str]:
    blocked: list[str] = []
    if local_sota_receipt_status != "passed_full_local_sota_receipt_clean_rerun_allowed":
        blocked.append("local_sota_receipt")
    if clean_verifier_status != "clean_live_verifier_ready":
        blocked.append("clean_verifier")
    if repair_status != "repair_ready":
        blocked.append("repair")
    if not fr11_self_learning_status.startswith("controller_memory_and_sidecar_promoted"):
        blocked.append("deployed_verifier_sidecar")
    if hardware_sampler_status != "authenticated_hardware_speedup_claim_allowed":
        blocked.append("hardware_sampler")
    return blocked


def _paper_v6_narrowing(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    sidecar = payloads["exp3201"]
    controller = payloads["exp3200"]
    hardware = payloads["exp3202"]
    substrate = _as_mapping(hardware.get("inference_substrate"))
    return {
        "kv260_speedup_claimed": hardware.get("speedup_claim_allowed") is True,
        "tsu_or_kona_execution_claimed": (
            substrate.get("tsu_z1_xtr0_kona_execution_claimed") is True
            or substrate.get("kona_execution_claimed") is True
        ),
        "deployed_verifier_sidecar_claimed": sidecar.get("sidecar_promotion_allowed") is True,
        "model_weight_self_learning_claimed": (
            controller.get("model_weight_update_performed") is True
            or sidecar.get("model_weight_update_performed") is True
        ),
        "paper_ready_streak_claimed": False,
    }


def _next_top_gap(
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_status: str,
    fr11_self_learning_status: str,
    hardware_sampler_status: str,
) -> str:
    if local_sota_receipt_status != "passed_full_local_sota_receipt_clean_rerun_allowed":
        return "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"
    if clean_verifier_status != "clean_live_verifier_ready":
        return "clean_live_verifier_adversarial_flag_clearance"
    if repair_status != "repair_ready":
        return "repair_gate_unblock_live_repair_attempts"
    if not fr11_self_learning_status.startswith("controller_memory_and_sidecar_promoted"):
        return "deployed_verifier_sidecar_promotion"
    if hardware_sampler_status != "authenticated_hardware_speedup_claim_allowed":
        return "authenticated_hardware_speedup_or_explicit_no_speedup_boundary"
    return "publication_blocker_retirement_review"


def _new_publication_blockers(rows: list[Mapping[str, Any]]) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    for row in rows:
        status = _normal_status(str(row.get("status") or "missing"))
        if status not in PUBLICATION_BLOCKING_STATUSES:
            continue
        blockers.append(
            {
                "row_id": str(row.get("row_id") or ""),
                "experiment_id": str(row.get("experiment_id") or ""),
                "status": status,
                "blocker_class": str(row.get("blocker_class") or _blocker_class(status)),
                "source_artifact": str(row.get("source_artifact") or ""),
                "source_field": str(row.get("source_field") or ""),
            }
        )
    return blockers


def _prior_publication_blocker_count(matrix: Mapping[str, Any]) -> int | None:
    value = matrix.get("publication_blocker_count")
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _status_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in STATUSES}
    for row in rows:
        counts[_normal_status(str(row.get("status") or "missing"))] += 1
    return counts


def _normal_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_")
    if normalized == "gated_skip":
        return "gated_skipped"
    return normalized if normalized in STATUSES else "missing"


def _blocker_class(status: str, role: str = "") -> str:
    normalized = _normal_status(status)
    if normalized not in PUBLICATION_BLOCKING_STATUSES:
        return ""
    normalized_role = role.strip().lower().replace(" ", "_") or "artifact"
    return f"publication_blocker_{normalized_role}_{normalized}"


def _public_sources(sources: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row["experiment_id"]),
            "path": str(row["path"]),
            "loaded_path": str(row["loaded_path"]),
            "role": str(row["role"]),
            "source_type": str(row["source_type"]),
            "present": row.get("present") is True,
            "readable_json_object": row.get("readable_json_object") is True,
            "sha256": row.get("sha256"),
        }
        for row in sources
    ]


def _invariant_violations(
    matrix: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    status_counts: Mapping[str, int],
    publication_blocker_count: int,
    prior_count: int | None,
    new_blockers: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if matrix.get("cross_corpus_matrix_v29_ready") is not True:
        violations.append("matrix_v29 authority is missing or not ready")
    if set(status_counts) != set(STATUSES):
        violations.append("status_counts keys do not match required v30 statuses")
    if sum(status_counts.values()) != len(rows):
        violations.append("status_counts do not sum to artifact classification rows")
    if prior_count is not None and publication_blocker_count != prior_count + len(new_blockers):
        violations.append("publication_blocker_count does not reconcile with v29 delta")
    return violations


def _inference_substrate() -> JsonDict:
    return {
        "kind": "aggregation_from_checked_in_dot296_artifacts",
        "source": "matrix_v29_and_dot296_artifacts",
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
    if artifact.get("cross_corpus_matrix_v30_ready") is not True:
        return (
            "blocked_matrix_v30_preconditions: "
            f"missing_artifact_count={artifact.get('missing_artifact_count')}; "
            f"invariant_violations={len(_as_list(artifact.get('invariant_violations')))}"
        )
    return (
        "complete: cross_corpus_matrix_v30_ready=true; "
        f"prior_matrix_version={artifact.get('prior_matrix_version')}; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v29={artifact.get('blocker_delta_from_v29')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _row_status(rows: list[Mapping[str, Any]], row_id: str) -> str:
    for row in rows:
        if row.get("row_id") == row_id:
            return _normal_status(str(row.get("status") or "missing"))
    return "missing"


def _is_gate_skip(payload: Mapping[str, Any]) -> bool:
    if payload.get("gated_skip") is True:
        return True
    if payload.get("blocked_at_layer") == "conductor_pre_gate":
        return True
    return str(payload.get("schema") or "") == "blocked_gate_check_v1"


def _blocker_reasons(payload: Mapping[str, Any]) -> list[str]:
    reasons = _text_list(payload.get("blocker_reasons"))
    if not reasons:
        reasons = _text_list(payload.get("blocked_reasons"))
    if not reasons and payload.get("gate_check_summary"):
        reasons = _text_list(payload.get("gate_check_summary"))
    return reasons


def _corrigendum_pending(value: Any) -> bool:
    if isinstance(value, list):
        return bool(value)
    return bool(value)


def _complete_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "complete_", "success:", "success_", "passed:", "passed_"))


def _experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    value = payload.get("experiment_id")
    if value:
        return str(value)
    experiment = payload.get("experiment")
    if isinstance(experiment, int):
        return f"exp{experiment}"
    return fallback


def _nested(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _bool_field(payload: Mapping[str, Any], dotted: str) -> bool | None:
    value = _nested(payload, dotted)
    return value if isinstance(value, bool) else None


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
