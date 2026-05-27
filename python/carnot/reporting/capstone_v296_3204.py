"""Build the Exp 3204 milestone .296 capstone artifact.

Spec refs: REQ-REPORT-3204, SCENARIO-REPORT-3204.

This module is intentionally a ledger reader. The .296 capstone closes the
milestone by copying publication authority from matrix v30 and summarizing the
already-written source artifacts, so it must never create fresh model, verifier,
repair, solver, hardware, roadmap, conductor, or ops-document evidence.
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
SCHEMA_VERSION = "carnot.milestone_capstone.v296_matrix_v30_terminal_aggregation.v1"
EXPERIMENT_ID = "exp3204"
ARTIFACT = "experiment_3204_capstone_v296"
MATRIX_V30_REL_PATH = Path("results/experiment_3203_cross_corpus_matrix_v30.json")
OUTPUT_REL_PATH = Path("results/experiment_3204_capstone_v296.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3204_capstone_v296.py"

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

REPAIR_GATE_ROW_ID = "dot296:exp3198_repair_gate_v5"
REPAIR_LADDER_ROW_ID = "dot296:exp3199_repair_ladder_v6"


@dataclass(frozen=True)
class SourceSpec:
    """A checked-in `.296` artifact that the capstone should load explicitly."""

    path: Path
    role: str


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(EXP3191_REL_PATH, "archive_v295_activate_v296"),
    SourceSpec(EXP3192_REL_PATH, "receipt_adversarial_contract_v4"),
    SourceSpec(EXP3193_REL_PATH, "llama_cpp_cuda_offload_health_probe"),
    SourceSpec(EXP3194_REL_PATH, "clean_live_sota_verifier_rerun_v11"),
    SourceSpec(EXP3195_REL_PATH, "adaptive_verification_granularity_policy"),
    SourceSpec(EXP3196_REL_PATH, "gencp_domain_preview_repair_compiler"),
    SourceSpec(EXP3197_REL_PATH, "exverus_inductive_certificate_expansion"),
    SourceSpec(EXP3198_REL_PATH, "repair_gate_decision_v5"),
    SourceSpec(EXP3199_REL_PATH, "multi_turn_repair_ladder_v6"),
    SourceSpec(EXP3200_REL_PATH, "fr11_verify_trace_memory_controller"),
    SourceSpec(EXP3201_REL_PATH, "kan_cl_nonforgetting_sidecar_audit"),
    SourceSpec(EXP3202_REL_PATH, "sparse_potts_paoa_thrml_factor_boundary"),
)
CRITICAL_SOURCE_PATHS = tuple(spec.path for spec in SOURCE_SPECS)


def read_json_object(path: Path) -> JsonDict:
    """Read source evidence as a JSON object and fail closed on bad inputs."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Checksum evidence so the capstone can be reproduced from the same inputs."""

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
    """REQ-REPORT-3204: close .296 using matrix v30 as publication authority."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V30_REL_PATH)
    matrix_rows = _as_list(matrix.get("artifact_classifications"))
    source_records = _critical_source_records(root_path)
    sources_by_path = {
        str(record["path"]): _as_mapping(record.get("payload")) for record in source_records
    }

    publication_blocker_count = _int_or_none(matrix.get("publication_blocker_count")) or 0
    blocker_delta_from_v29 = _int_or_none(matrix.get("blocker_delta_from_v29"))
    paper_ready = matrix.get("paper_ready") is True
    local_sota_receipt_status = _field_str(
        matrix, "local_sota_receipt_status", "missing_local_sota_receipt_status"
    )
    clean_verifier_status = _field_str(
        matrix, "clean_verifier_status", "missing_clean_verifier_status"
    )
    repair_gate_status = _repair_gate_status(
        sources_by_path.get(EXP3198_REL_PATH.as_posix(), {}), matrix_rows
    )
    repair_ladder_status = _repair_ladder_status(
        sources_by_path.get(EXP3199_REL_PATH.as_posix(), {}), matrix_rows
    )
    fr11_self_learning_status = _field_str(
        matrix, "fr11_self_learning_status", "missing_fr11_self_learning_status"
    )
    hardware_sampler_status = _field_str(
        matrix, "hardware_sampler_status", "missing_hardware_sampler_status"
    )
    next_top_gap = _field_str(matrix, "next_top_gap", "matrix_v30_missing_next_top_gap")
    invariant_violations = _invariant_violations(
        matrix,
        source_records,
        publication_blocker_count,
        paper_ready,
    )
    capstone_ready = not invariant_violations
    claim_boundaries = _claim_boundaries(matrix, paper_ready, hardware_sampler_status)
    artifact: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "schema": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_artifact": MATRIX_V30_REL_PATH.as_posix(),
        "matrix_ready": matrix.get("cross_corpus_matrix_v30_ready") is True,
        "capstone_v296_ready": capstone_ready,
        "paper_ready": paper_ready,
        "paper_ready_source": f"{MATRIX_V30_REL_PATH.as_posix()}.paper_ready",
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v29": blocker_delta_from_v29,
        "local_sota_receipt_status": local_sota_receipt_status,
        "clean_verifier_status": clean_verifier_status,
        "repair_gate_status": repair_gate_status,
        "repair_ladder_status": repair_ladder_status,
        "fr11_self_learning_status": fr11_self_learning_status,
        "hardware_sampler_status": hardware_sampler_status,
        "next_top_gap": next_top_gap,
        "next_milestone_theme": _next_milestone_theme(next_top_gap),
        "ops_docs_updated": False,
        "active_roadmap_modified": False,
        "conductor_file_modified": False,
        "protected_file_policy": {
            "ops_status_modified": False,
            "ops_changelog_modified": False,
            "traceability_modified": False,
            "research_roadmap_modified": False,
            "research_conductor_modified": False,
            "reason": "conductor stop rule delegates ops/status/changelog/traceability reconciliation",
        },
        "critical_artifacts_expected": [path.as_posix() for path in CRITICAL_SOURCE_PATHS],
        "critical_artifacts_loaded": [
            str(row["path"]) for row in source_records if row.get("readable_json_object") is True
        ],
        "critical_source_artifacts": _public_source_records(source_records),
        "source_checksums": {
            str(row["path"]): row.get("sha256")
            for row in source_records
            if row.get("readable_json_object") is True
        },
        "matrix_summary": _matrix_summary(matrix),
        "phase_outcomes": _phase_outcomes(
            matrix,
            sources_by_path,
            local_sota_receipt_status,
            clean_verifier_status,
            repair_gate_status,
            repair_ladder_status,
            fr11_self_learning_status,
            hardware_sampler_status,
        ),
        "claim_boundaries_preserved": claim_boundaries,
        "required_evidence_blocked_or_missing": _as_list(
            matrix.get("required_evidence_blocked_or_missing")
        ),
        "invariant_violations": invariant_violations,
        "inference_substrate": _inference_substrate(),
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
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
    """Build and persist the Exp 3204 capstone deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _critical_source_records(root: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        payload = read_json_object(path)
        records.append(
            {
                "experiment_id": _source_experiment_id(payload, spec.path.stem),
                "path": spec.path.as_posix(),
                "role": spec.role,
                "present": path.is_file(),
                "readable_json_object": bool(payload),
                "sha256": sha256_file(path),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
                "payload": payload,
            }
        )
    return records


def _public_source_records(records: list[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "experiment_id": str(row.get("experiment_id") or ""),
            "path": str(row.get("path") or ""),
            "role": str(row.get("role") or ""),
            "present": row.get("present") is True,
            "readable_json_object": row.get("readable_json_object") is True,
            "sha256": row.get("sha256"),
            "honest_verdict": str(row.get("honest_verdict") or ""),
        }
        for row in records
    ]


def _repair_gate_status(payload: Mapping[str, Any], rows: list[Any]) -> str:
    state = str(payload.get("repair_gate_state") or "")
    if state:
        return state
    row_status = _row_status(rows, REPAIR_GATE_ROW_ID)
    if row_status == "clean":
        return "clean_repair_gate_unblocked"
    if row_status == "blocked":
        return "blocked_repair_gate_v5"
    return "missing_repair_gate_decision_v5"


def _repair_ladder_status(payload: Mapping[str, Any], rows: list[Any]) -> str:
    if _is_conductor_gate_skip(payload):
        return "gated_skipped_repair_gate_blocked"
    if payload.get("headline_claim_allowed") is True:
        return "repair_ladder_executed"
    if payload:
        return "blocked_repair_ladder_v6"
    row_status = _row_status(rows, REPAIR_LADDER_ROW_ID)
    if row_status == "clean":
        return "repair_ladder_executed"
    if row_status == "gated_skipped":
        return "gated_skipped_repair_gate_blocked"
    return "missing_repair_ladder_v6"


def _is_conductor_gate_skip(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("blocked_at_layer") == "conductor_pre_gate"
        or str(payload.get("schema") or "") == "blocked_gate_check_v1"
    )


def _row_status(rows: list[Any], row_id: str) -> str:
    for row in rows:
        item = _as_mapping(row)
        if item.get("row_id") == row_id:
            return str(item.get("status") or "missing")
    return "missing"


def _phase_outcomes(
    matrix: Mapping[str, Any],
    sources_by_path: Mapping[str, Mapping[str, Any]],
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_gate_status: str,
    repair_ladder_status: str,
    fr11_self_learning_status: str,
    hardware_sampler_status: str,
) -> JsonDict:
    contract = _as_mapping(sources_by_path.get(EXP3192_REL_PATH.as_posix()))
    cuda = _as_mapping(sources_by_path.get(EXP3193_REL_PATH.as_posix()))
    adaptive = _as_mapping(sources_by_path.get(EXP3195_REL_PATH.as_posix()))
    domain_preview = _as_mapping(sources_by_path.get(EXP3196_REL_PATH.as_posix()))
    certificates = _as_mapping(sources_by_path.get(EXP3197_REL_PATH.as_posix()))
    controller = _as_mapping(sources_by_path.get(EXP3200_REL_PATH.as_posix()))
    sidecar = _as_mapping(sources_by_path.get(EXP3201_REL_PATH.as_posix()))
    hardware = _as_mapping(sources_by_path.get(EXP3202_REL_PATH.as_posix()))
    hardware_substrate = _as_mapping(hardware.get("inference_substrate"))
    return {
        "receipt_cuda": {
            "status": local_sota_receipt_status,
            "verdict": _phase_verdict(local_sota_receipt_status),
            "receipt_substrate": str(
                _as_mapping(contract.get("current_evidence_assessment")).get(
                    "substrate_classification"
                )
                or ""
            ),
            "cuda_substrate": str(cuda.get("substrate_classification") or ""),
            "clean_rerun_allowed": cuda.get("clean_rerun_allowed") is True,
        },
        "clean_verifier": {
            "status": clean_verifier_status,
            "verdict": _phase_verdict(clean_verifier_status),
        },
        "adaptive_repair_control": {
            "status": str(matrix.get("repair_status") or ""),
            "verdict": _phase_verdict(str(matrix.get("repair_status") or "")),
            "adaptive_policy_promotion_allowed": adaptive.get("promotion_allowed") is True,
            "estimated_verifier_call_delta": _int_or_none(
                adaptive.get("estimated_verifier_call_delta")
            ),
            "preview_domain_count": _int_or_none(domain_preview.get("preview_domain_count")),
            "invariant_record_count": _int_or_none(certificates.get("invariant_record_count")),
            "repair_gate_status": repair_gate_status,
            "repair_ladder_status": repair_ladder_status,
        },
        "fr11_self_learning": {
            "status": fr11_self_learning_status,
            "controller_promotion_allowed": controller.get("promotion_allowed") is True,
            "sidecar_promotion_allowed": sidecar.get("sidecar_promotion_allowed") is True,
            "trace_count": _int_or_none(controller.get("trace_count")),
            "model_weight_update_claimed": (
                controller.get("model_weight_update_performed") is True
                or sidecar.get("model_weight_update_performed") is True
                or _as_mapping(matrix.get("paper_v6_narrowing")).get(
                    "model_weight_self_learning_claimed"
                )
                is True
            ),
        },
        "hardware_boundary": {
            "status": hardware_sampler_status,
            "authenticated_hardware_transcript_present": (
                hardware.get("authenticated_hardware_transcript_present") is True
            ),
            "speedup_claim_allowed": hardware.get("speedup_claim_allowed") is True,
            "tsu_or_kona_execution_claimed": (
                hardware_substrate.get("tsu_z1_xtr0_kona_execution_claimed") is True
                or hardware_substrate.get("kona_execution_claimed") is True
            ),
            "factor_record_count": _int_or_none(hardware.get("factor_record_count")),
        },
    }


def _phase_verdict(status: str) -> str:
    if status.startswith(("passed", "clean", "repair_ready", "authenticated")):
        return "passed"
    if status.startswith("blocked"):
        return "blocked"
    if "gated_skipped" in status:
        return "gated_skipped"
    if status.startswith("diagnostic_only"):
        return "diagnostic_only"
    return "blocked"


def _claim_boundaries(
    matrix: Mapping[str, Any],
    paper_ready: bool,
    hardware_sampler_status: str,
) -> JsonDict:
    narrowing = _as_mapping(matrix.get("paper_v6_narrowing"))
    return {
        "paper_ready_claim_allowed": paper_ready,
        "repair_claim_allowed": str(matrix.get("repair_status") or "") == "repair_ready",
        "hardware_speedup_claim_allowed": (
            hardware_sampler_status == "authenticated_hardware_speedup_claim_allowed"
        ),
        "tsu_or_kona_claim_allowed": narrowing.get("tsu_or_kona_execution_claimed") is True,
        "model_weight_self_learning_claim_allowed": (
            narrowing.get("model_weight_self_learning_claimed") is True
        ),
    }


def _invariant_violations(
    matrix: Mapping[str, Any],
    source_records: list[Mapping[str, Any]],
    publication_blocker_count: int,
    paper_ready: bool,
) -> list[str]:
    violations: list[str] = []
    if matrix.get("cross_corpus_matrix_v30_ready") is not True:
        violations.append("matrix_v30 authority is missing or not ready")
    missing_sources = [
        str(row.get("path") or "") for row in source_records if not row.get("readable_json_object")
    ]
    if missing_sources:
        violations.append(
            f"critical .296 source artifacts missing or malformed: {len(missing_sources)}"
        )
    loaded = set(str(item) for item in _as_list(matrix.get("source_artifacts_loaded")))
    absent_from_matrix = [
        path.as_posix() for path in CRITICAL_SOURCE_PATHS if path.as_posix() not in loaded
    ]
    if matrix and absent_from_matrix:
        violations.append("matrix_v30 source_artifacts_loaded omits critical .296 artifacts")
    if paper_ready and publication_blocker_count > 0:
        violations.append("matrix_v30 paper_ready=true while publication blockers remain")
    return violations


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "matrix_version": str(matrix.get("matrix_version") or ""),
        "status_counts": _as_mapping(matrix.get("status_counts")),
        "missing_artifact_count": _int_or_none(matrix.get("missing_artifact_count")),
        "publication_blocker_accounting": _as_mapping(matrix.get("publication_blocker_accounting")),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _next_milestone_theme(next_top_gap: str) -> str:
    if next_top_gap.startswith("cuda_offload") or "local_sota_receipt" in next_top_gap:
        return "cuda_offload_full_local_sota_receipt_and_clean_rerun_unblock"
    if "repair" in next_top_gap:
        return "bounded_live_repair_ladder_execution_after_gate_unblock"
    if "sidecar" in next_top_gap or "fr11" in next_top_gap.lower():
        return "fr11_sidecar_promotion_without_model_weight_updates"
    if "hardware" in next_top_gap:
        return "authenticated_hardware_speedup_or_explicit_no_speedup_boundary"
    return "publication_blocker_retirement_review"


def _inference_substrate() -> JsonDict:
    return {
        "kind": "capstone_aggregation_from_checked_in_matrix_v30_and_dot296_artifacts",
        "source": "matrix_v30_and_dot296_source_artifacts",
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
    if artifact.get("capstone_v296_ready") is not True:
        first = str(_as_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_v296_ready=false; {first}"
    return (
        "complete: capstone_v296_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v29={artifact.get('blocker_delta_from_v29')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _source_experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    value = payload.get("experiment_id")
    if value:
        return str(value)
    experiment = payload.get("experiment")
    if isinstance(experiment, int):
        return f"exp{experiment}"
    return fallback


def _field_str(payload: Mapping[str, Any], field: str, fallback: str) -> str:
    value = payload.get(field)
    return str(value) if value not in (None, "") else fallback


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
