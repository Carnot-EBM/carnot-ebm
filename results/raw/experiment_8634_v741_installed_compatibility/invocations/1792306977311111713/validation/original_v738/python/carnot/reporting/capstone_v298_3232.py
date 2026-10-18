"""Build the Exp 3232 milestone .298 capstone artifact.

Spec refs: REQ-REPORT-3232, SCENARIO-REPORT-3232.

The capstone is a terminal publication-readiness decision. It reads matrix v32,
the prior `.297` capstone, and the `.298` artifacts referenced by the matrix.
It deliberately performs no model, verifier, repair, solver, hardware,
conductor, roadmap, or ops-document work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
MILESTONE = "2026.05.298"
SCHEMA_VERSION = "carnot.milestone_capstone.v298_matrix_v32_terminal_aggregation.v1"
EXPERIMENT_ID = "exp3232"
ARTIFACT = "experiment_3232_capstone_v298"
MATRIX_V32_REL_PATH = Path("results/experiment_3231_cross_corpus_matrix_v32.json")
PRIOR_CAPSTONE_REL_PATH = Path("results/experiment_3218_capstone_v297.json")
OUTPUT_REL_PATH = Path("results/experiment_3232_capstone_v298.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3232_capstone_v298.py"
NEXT_TOP_GAP = "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt"
INFERENCE_SUBSTRATE = (
    "terminal_aggregation_from_checked_in_matrix_v32_prior_capstone_and_referenced_artifacts"
)

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

REQUIRED_FIELD_TYPES: tuple[tuple[str, type], ...] = (
    ("schema_version", str),
    ("experiment_id", str),
    ("milestone", str),
    ("matrix_artifact", str),
    ("prior_capstone_artifact", str),
    ("capstone_ready", bool),
    ("paper_ready", bool),
    ("publication_blocker_count", int),
    ("blocker_delta_from_v31", int),
    ("local_sota_receipt_status", str),
    ("clean_verifier_status", str),
    ("repair_gate_status", str),
    ("repair_ladder_status", str),
    ("continuous_self_learning_status", str),
    ("hardware_claim_status", str),
    ("what_this_milestone_proved", list),
    ("next_top_gap", str),
    ("recommended_next_milestone_theme", str),
    ("inference_substrate", str),
    ("conductor_file_modified", bool),
    ("active_roadmap_modified", bool),
    ("honest_verdict", str),
)


def read_json_object(path: Path) -> JsonDict:
    """Read source evidence as a JSON object and fail closed on bad inputs."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Checksum evidence so the capstone decision can be reproduced."""

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
    """REQ-REPORT-3232: close `.298` using matrix v32 as claim authority."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V32_REL_PATH)
    prior_capstone = read_json_object(root_path / PRIOR_CAPSTONE_REL_PATH)
    source_artifacts = _source_artifacts(root_path, matrix, prior_capstone)

    publication_blocker_count = _int_value(matrix.get("publication_blocker_count"))
    blocker_delta_from_v31 = _int_value(matrix.get("blocker_delta_from_v31"))
    local_sota_receipt_status = _field_str(
        matrix, "local_sota_receipt_state", "missing_local_sota_receipt_status"
    )
    clean_verifier_status = _field_str(
        matrix, "clean_verifier_state", "missing_clean_verifier_status"
    )
    repair_gate_status = _field_str(matrix, "repair_gate_state", "missing_repair_gate_status")
    repair_ladder_status = _field_str(
        matrix, "repair_ladder_state", "missing_repair_ladder_status"
    )
    continuous_self_learning_status = _field_str(
        matrix, "continuous_self_learning_state", "missing_continuous_self_learning_status"
    )
    hardware_claim_status = _field_str(
        matrix, "hardware_claim_boundary", "missing_hardware_claim_status"
    )
    next_top_gap = _field_str(matrix, "next_top_gap", "matrix_v32_missing_next_top_gap")
    readiness_criteria = _readiness_criteria(matrix)
    publication_blockers = _publication_blockers(matrix)
    paper_ready_candidate = _paper_ready_from_evidence(
        matrix, publication_blocker_count, readiness_criteria
    )
    claim_boundaries = _claim_boundaries(
        paper_ready_candidate,
        repair_gate_status,
        repair_ladder_status,
        continuous_self_learning_status,
        hardware_claim_status,
    )
    invariant_violations = _invariant_violations(
        matrix=matrix,
        prior_capstone=prior_capstone,
        publication_blocker_count=publication_blocker_count,
        blocker_delta_from_v31=blocker_delta_from_v31,
        paper_ready_candidate=paper_ready_candidate,
        readiness_criteria=readiness_criteria,
        source_artifacts=source_artifacts,
    )

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_artifact": MATRIX_V32_REL_PATH.as_posix(),
        "prior_capstone_artifact": PRIOR_CAPSTONE_REL_PATH.as_posix(),
        "capstone_ready": not invariant_violations,
        "paper_ready": paper_ready_candidate and not invariant_violations,
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v31": blocker_delta_from_v31,
        "local_sota_receipt_status": local_sota_receipt_status,
        "clean_verifier_status": clean_verifier_status,
        "repair_gate_status": repair_gate_status,
        "repair_ladder_status": repair_ladder_status,
        "continuous_self_learning_status": continuous_self_learning_status,
        "hardware_claim_status": hardware_claim_status,
        "what_this_milestone_proved": _what_this_milestone_proved(
            local_sota_receipt_status,
            clean_verifier_status,
            repair_gate_status,
            repair_ladder_status,
            continuous_self_learning_status,
            hardware_claim_status,
            source_artifacts,
        ),
        "next_top_gap": next_top_gap,
        "recommended_next_milestone_theme": _recommended_next_milestone_theme(next_top_gap),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "conductor_file_modified": matrix.get("conductor_file_modified") is True,
        "active_roadmap_modified": matrix.get("active_roadmap_modified") is True,
        "source_artifacts": source_artifacts,
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in source_artifacts
            if row.get("readable_json_object") is True
        },
        "publication_blockers": publication_blockers,
        "readiness_criteria": readiness_criteria,
        "claim_boundaries_preserved": claim_boundaries,
        "prior_capstone_summary": _prior_capstone_summary(prior_capstone),
        "matrix_summary": _matrix_summary(matrix),
        "invariant_violations": [],
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_repair_run": True,
        "no_new_solver_run": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "ops_docs_reconciliation_left_to_conductor": True,
        "duration_s": _duration(start, now_s),
        "honest_verdict": "",
    }
    artifact["invariant_violations"] = invariant_violations + _required_fields_are_typed(artifact)
    artifact["capstone_ready"] = not artifact["invariant_violations"]
    artifact["paper_ready"] = paper_ready_candidate and artifact["capstone_ready"]
    artifact["claim_boundaries_preserved"]["paper_ready_claim_allowed"] = artifact["paper_ready"]
    artifact["honest_verdict"] = _honest_verdict(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3232 capstone deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifacts(
    root: Path, matrix: Mapping[str, Any], prior_capstone: Mapping[str, Any]
) -> list[JsonDict]:
    records = [
        _named_source_record(root, "matrix_v32", MATRIX_V32_REL_PATH, matrix),
        _named_source_record(root, "prior_capstone_v297", PRIOR_CAPSTONE_REL_PATH, prior_capstone),
    ]
    for row in _as_list(matrix.get("input_artifacts")):
        row_map = _as_mapping(row)
        experiment_id = str(row_map.get("experiment_id") or "unknown_matrix_input")
        rel_path = Path(str(row_map.get("path") or ""))
        payload = read_json_object(root / rel_path) if rel_path.as_posix() else {}
        record = _named_source_record(root, experiment_id, rel_path, payload)
        record.update(
            {
                "matrix_status": str(row_map.get("status") or ""),
                "matrix_role": str(row_map.get("role") or ""),
                "matrix_present": row_map.get("present") is True,
                "matrix_status_rationale": str(row_map.get("status_rationale") or ""),
                "reported_experiment_id": _source_experiment_id(payload, experiment_id),
            }
        )
        records.append(record)
    return records


def _named_source_record(
    root: Path,
    experiment_id: str,
    rel_path: Path,
    payload: Mapping[str, Any],
) -> JsonDict:
    path = root / rel_path
    return {
        "experiment_id": experiment_id,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "readable_json_object": bool(payload),
        "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(path),
    }


def _publication_blockers(matrix: Mapping[str, Any]) -> list[JsonDict]:
    blockers = _as_list(matrix.get("blocker_delta_explanation"))
    if blockers:
        return [_as_mapping(row) for row in blockers]
    derived: list[JsonDict] = []
    for row in _as_list(matrix.get("input_artifacts")):
        row_map = _as_mapping(row)
        if str(row_map.get("status") or "") in {"blocked", "gate_blocked", "missing"}:
            derived.append(
                {
                    "experiment_id": str(row_map.get("experiment_id") or ""),
                    "status": str(row_map.get("status") or ""),
                    "path": str(row_map.get("path") or ""),
                }
            )
    return derived


def _readiness_criteria(matrix: Mapping[str, Any]) -> JsonDict:
    criteria = _as_mapping(matrix.get("paper_ready_criteria"))
    return {
        "local_sota_receipt": criteria.get("local_sota_receipt") is True,
        "clean_verifier": criteria.get("clean_verifier") is True,
        "repair": criteria.get("repair") is True,
        "fr11": criteria.get("fr11") is True,
        "claim_boundary": criteria.get("claim_boundary") is True,
    }


def _paper_ready_from_evidence(
    matrix: Mapping[str, Any],
    publication_blocker_count: int,
    readiness_criteria: Mapping[str, bool],
) -> bool:
    return (
        matrix.get("paper_ready") is True
        and publication_blocker_count == 0
        and all(readiness_criteria.values())
    )


def _claim_boundaries(
    paper_ready: bool,
    repair_gate_status: str,
    repair_ladder_status: str,
    continuous_self_learning_status: str,
    hardware_claim_status: str,
) -> JsonDict:
    repair_success = repair_gate_status.startswith("unblocked") and repair_ladder_status.startswith(
        "complete"
    )
    hardware_allowed = hardware_claim_status == "authenticated_hardware_claim_allowed"
    model_weight_learning_allowed = (
        "model_weight_update_certified" in continuous_self_learning_status
        and "no_model_weight_update" not in continuous_self_learning_status
    )
    return {
        "paper_ready_claim_allowed": paper_ready,
        "repair_success_claim_allowed": repair_success,
        "hardware_speedup_claim_allowed": hardware_allowed,
        "tsu_or_kona_claim_allowed": hardware_allowed
        and "tsu_or_kona_claim_allowed" in hardware_claim_status,
        "model_weight_learning_claim_allowed": model_weight_learning_allowed,
    }


def _what_this_milestone_proved(
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_gate_status: str,
    repair_ladder_status: str,
    continuous_self_learning_status: str,
    hardware_claim_status: str,
    source_artifacts: list[Mapping[str, Any]],
) -> list[JsonDict]:
    return [
        {
            "domain": "cuda_receipt",
            "status": _proof_status(local_sota_receipt_status),
            "evidence": _evidence(source_artifacts, ("exp3220", "exp3221", "exp3222")),
            "conclusion": (
                "CUDA was visible to the host, but the selected receipt path did not unlock "
                "clean local SOTA execution."
            ),
        },
        {
            "domain": "live_sota_verification",
            "status": _proof_status(clean_verifier_status),
            "evidence": _evidence(source_artifacts, ("exp3225", "exp3222")),
            "conclusion": (
                "The clean live SOTA verifier did not produce headline evidence because the "
                "full local SOTA receipt gate remained blocked or absent."
            ),
        },
        {
            "domain": "repair",
            "status": _proof_status(repair_gate_status),
            "evidence": _evidence(source_artifacts, ("exp3226", "exp3227")),
            "conclusion": (
                "Structured repair remained blocked; missing preflight and failed gate "
                "evidence cannot be counted as semantic repair success."
            ),
        },
        {
            "domain": "repair_ladder",
            "status": _proof_status(repair_ladder_status),
            "evidence": _evidence(source_artifacts, ("exp3228", "exp3227")),
            "conclusion": (
                "The repair ladder did not run as a success path because the repair gate "
                "did not unblock."
            ),
        },
        {
            "domain": "continuous_self_learning",
            "status": _proof_status(continuous_self_learning_status),
            "evidence": _evidence(source_artifacts, ("exp3229", "exp3230")),
            "conclusion": (
                "FR-11 advanced at the controller-memory governance layer with no "
                "model-weight update claim; KAN-CL sidecar promotion stayed certificate-blocked."
            ),
        },
        {
            "domain": "hardware_boundary",
            "status": _proof_status(hardware_claim_status),
            "evidence": _evidence(source_artifacts, ("exp3220", "exp3221")),
            "conclusion": (
                "No authenticated hardware speedup, TSU execution, or Kona execution claim "
                "is supported by this milestone."
            ),
        },
    ]


def _evidence(source_artifacts: list[Mapping[str, Any]], experiment_ids: tuple[str, ...]) -> str:
    pieces: list[str] = []
    for experiment_id in experiment_ids:
        row = next(
            (candidate for candidate in source_artifacts if candidate.get("experiment_id") == experiment_id),
            {},
        )
        if row:
            status = str(row.get("matrix_status") or ("present" if row.get("present") else "missing"))
            pieces.append(f"{experiment_id}={status}")
        else:
            pieces.append(f"{experiment_id}=not_referenced")
    return "; ".join(pieces)


def _proof_status(status: str) -> str:
    normalized = status.lower()
    if normalized.startswith("missing_full_local_sota_receipt") and "gate_blocked" in normalized:
        return "blocked"
    if normalized.startswith(("clean_rerun_allowed", "clean_live", "unblocked", "complete")):
        return "ready"
    if "gate_blocked" in normalized or "gated_skipped" in normalized:
        return "gate_blocked"
    if normalized.startswith("missing"):
        return "missing"
    if normalized.startswith("blocked"):
        return "blocked"
    if normalized.startswith("controller_memory_promotion_allowed"):
        return "controller_only"
    if "no_hardware_speedup" in normalized or "tsu_or_kona" in normalized:
        return "claim_denied"
    if "cuda_runtime_visible_but_not_usable" in normalized:
        return "claim_denied"
    return "blocked"


def _invariant_violations(
    *,
    matrix: Mapping[str, Any],
    prior_capstone: Mapping[str, Any],
    publication_blocker_count: int,
    blocker_delta_from_v31: int,
    paper_ready_candidate: bool,
    readiness_criteria: Mapping[str, bool],
    source_artifacts: list[Mapping[str, Any]],
) -> list[str]:
    violations: list[str] = []
    if matrix.get("cross_corpus_matrix_v32_ready") is not True:
        violations.append("matrix_v32 authority is missing or not ready")
    if prior_capstone.get("capstone_v297_ready") is not True:
        violations.append("prior capstone v297 authority is missing or not ready")
    if matrix.get("paper_ready") is True and publication_blocker_count > 0:
        violations.append("matrix_v32 paper_ready=true while publication blockers remain")
    if matrix.get("paper_ready") is True and not all(readiness_criteria.values()):
        violations.append("matrix_v32 paper_ready=true while readiness criteria are incomplete")
    prior_count = _int_value(prior_capstone.get("publication_blocker_count"))
    if matrix and blocker_delta_from_v31 != publication_blocker_count - prior_count:
        violations.append("blocker_delta_from_v31 does not reconcile with prior capstone count")
    if paper_ready_candidate and publication_blocker_count != 0:
        violations.append("paper_ready candidate has nonzero blockers")
    if matrix.get("conductor_file_modified") is True:
        violations.append("matrix_v32 reports conductor file modification")
    if matrix.get("active_roadmap_modified") is True:
        violations.append("matrix_v32 reports active roadmap modification")
    input_count = len(_as_list(matrix.get("input_artifacts")))
    if matrix and len(source_artifacts) != input_count + 2:
        violations.append("not all matrix-referenced .298 artifacts are accounted for")
    return violations


def _required_fields_are_typed(artifact: Mapping[str, Any]) -> list[str]:
    violations: list[str] = []
    for field, expected_type in REQUIRED_FIELD_TYPES:
        value = artifact.get(field)
        if expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                violations.append(f"{field} missing_or_wrong_type")
        elif not isinstance(value, expected_type):
            violations.append(f"{field} missing_or_wrong_type")
    return violations


def _recommended_next_milestone_theme(next_top_gap: str) -> str:
    if next_top_gap == NEXT_TOP_GAP or "cuda_runtime" in next_top_gap:
        return "hermetic_cuda_offload_receipt_repair_for_clean_local_sota"
    if "clean_live_verifier" in next_top_gap:
        return "clean_live_verifier_v13_gate_clearance_after_receipt"
    if "repair" in next_top_gap:
        return "structured_repair_gate_v7_unblock_and_ladder_v8_execution"
    if "fr11" in next_top_gap.lower() or "certificate_boundary" in next_top_gap:
        return "fr11_certificate_boundary_for_sidecar_promotion"
    if "hardware" in next_top_gap:
        return "authenticated_hardware_boundary_or_explicit_no_speedup_disclosure"
    return "publication_blocker_retirement_review"


def _prior_capstone_summary(prior_capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "capstone_v297_ready": prior_capstone.get("capstone_v297_ready") is True,
        "paper_ready": prior_capstone.get("paper_ready") is True,
        "publication_blocker_count": _int_value(prior_capstone.get("publication_blocker_count")),
        "next_top_gap": str(prior_capstone.get("next_top_gap") or ""),
        "honest_verdict": str(prior_capstone.get("honest_verdict") or ""),
    }


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "cross_corpus_matrix_v32_ready": matrix.get("cross_corpus_matrix_v32_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count": _int_value(matrix.get("publication_blocker_count")),
        "blocker_delta_from_v31": _int_value(matrix.get("blocker_delta_from_v31")),
        "next_top_gap": str(matrix.get("next_top_gap") or ""),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact.get("capstone_ready") is not True:
        first = str(_as_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_ready=false; {first}"
    return (
        "complete: capstone_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v31={artifact.get('blocker_delta_from_v31')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _source_experiment_id(payload: Mapping[str, Any], fallback: str) -> str:
    value = payload.get("experiment_id")
    if value:
        return str(value)
    experiment = payload.get("experiment")
    if isinstance(experiment, int) and not isinstance(experiment, bool):
        return f"exp{experiment}"
    return fallback


def _field_str(payload: Mapping[str, Any], field: str, fallback: str) -> str:
    value = payload.get(field)
    return str(value) if value not in (None, "") else fallback


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
