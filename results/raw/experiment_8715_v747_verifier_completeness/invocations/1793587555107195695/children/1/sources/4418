"""Build the Exp 3218 milestone .297 capstone artifact.

Spec refs: REQ-REPORT-3218, SCENARIO-REPORT-3218.

The capstone is the terminal claim ledger for milestone `.297`. It summarizes
what the already-written matrix v31 proves and deliberately avoids creating new
model, verifier, repair, solver, hardware, conductor, roadmap, or ops-document
evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.297"
SCHEMA_VERSION = "carnot.milestone_capstone.v297_matrix_v31_terminal_aggregation.v1"
EXPERIMENT_ID = "exp3218"
ARTIFACT = "experiment_3218_capstone_v297"
MATRIX_V31_REL_PATH = Path("results/experiment_3217_cross_corpus_matrix_v31.json")
OUTPUT_REL_PATH = Path("results/experiment_3218_capstone_v297.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3218_capstone_v297.py"
NEXT_TOP_GAP = "cuda_offload_full_local_sota_receipt_clean_rerun_allowed_repair_gate_unblock"

EXP3205_REL_PATH = Path("results/experiment_3205_archive_v296_activate_v297.json")
EXP3206_REL_PATH = Path("results/experiment_3206_cuda_env_forensics_ledger_v1.json")
EXP3207_REL_PATH = Path("results/experiment_3207_llama_cpp_cuda_rebuild_clean_subprocess_v1.json")
EXP3208_REL_PATH = Path("results/experiment_3208_full_local_sota_receipt_v5.json")
EXP3209_REL_PATH = Path("results/experiment_3209_clean_live_sota_verifier_rerun_v12.json")
EXP3210_REL_PATH = Path(
    "results/experiment_3210_context_cot_clbench_parametric_shortcut_fixtures_v1.json"
)
EXP3211_REL_PATH = Path("results/experiment_3211_constraintbench_feasibility_objective_pilot_v1.json")
EXP3212_REL_PATH = Path("results/experiment_3212_structured_repair_proposal_preflight_v1.json")
EXP3213_REL_PATH = Path("results/experiment_3213_repair_gate_decision_v6.json")
EXP3214_REL_PATH = Path("results/experiment_3214_multi_turn_repair_ladder_v7.json")
EXP3215_REL_PATH = Path("results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json")
EXP3216_REL_PATH = Path("results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json")

SOURCE_PATHS: tuple[Path, ...] = (
    EXP3205_REL_PATH,
    EXP3206_REL_PATH,
    EXP3207_REL_PATH,
    EXP3208_REL_PATH,
    EXP3209_REL_PATH,
    EXP3210_REL_PATH,
    EXP3211_REL_PATH,
    EXP3212_REL_PATH,
    EXP3213_REL_PATH,
    EXP3214_REL_PATH,
    EXP3215_REL_PATH,
    EXP3216_REL_PATH,
)


def read_json_object(path: Path) -> JsonDict:
    """Read source evidence as a JSON object and fail closed on bad inputs."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Checksum source evidence so the capstone can be reproduced exactly."""

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
    """REQ-REPORT-3218: close `.297` using matrix v31 as publication authority."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V31_REL_PATH)
    source_records = _source_records(root_path)

    publication_blocker_count = _int_or_none(matrix.get("publication_blocker_count")) or 0
    blocker_delta_from_v30 = _int_or_none(matrix.get("blocker_delta_from_v30"))
    paper_ready = matrix.get("paper_ready") is True
    local_sota_receipt_status = _field_str(
        matrix, "local_sota_receipt_status", "missing_local_sota_receipt_status"
    )
    clean_verifier_status = _field_str(
        matrix, "clean_verifier_status", "missing_clean_verifier_status"
    )
    repair_gate_status = _field_str(matrix, "repair_gate_status", "missing_repair_gate_status")
    repair_ladder_status = _field_str(
        matrix, "repair_ladder_status", "missing_repair_ladder_status"
    )
    context_fixture_status = _field_str(
        matrix, "context_fixture_status", "missing_context_fixture_status"
    )
    constraintbench_fixture_status = _field_str(
        matrix, "constraintbench_fixture_status", "missing_constraintbench_fixture_status"
    )
    fr11_self_learning_status = _field_str(
        matrix, "fr11_self_learning_status", "missing_fr11_self_learning_status"
    )
    hardware_sampler_status = _field_str(
        matrix, "hardware_sampler_status", "missing_hardware_sampler_status"
    )
    next_top_gap = _field_str(matrix, "next_top_gap", "matrix_v31_missing_next_top_gap")
    claim_boundaries = _claim_boundaries(matrix, paper_ready)
    invariant_violations = _invariant_violations(
        matrix, publication_blocker_count, paper_ready, claim_boundaries
    )
    capstone_ready = not invariant_violations

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "matrix_artifact": MATRIX_V31_REL_PATH.as_posix(),
        "matrix_ready": matrix.get("cross_corpus_matrix_v31_ready") is True,
        "capstone_v297_ready": capstone_ready,
        "paper_ready": paper_ready,
        "paper_ready_source": f"{MATRIX_V31_REL_PATH.as_posix()}.paper_ready",
        "publication_blocker_count": publication_blocker_count,
        "blocker_delta_from_v30": blocker_delta_from_v30,
        "local_sota_receipt_status": local_sota_receipt_status,
        "clean_verifier_status": clean_verifier_status,
        "repair_gate_status": repair_gate_status,
        "repair_ladder_status": repair_ladder_status,
        "context_fixture_status": context_fixture_status,
        "constraintbench_fixture_status": constraintbench_fixture_status,
        "fr11_self_learning_status": fr11_self_learning_status,
        "hardware_sampler_status": hardware_sampler_status,
        "next_top_gap": next_top_gap,
        "recommended_next_milestone_theme": _recommended_next_milestone_theme(next_top_gap),
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "conductor_file_modified": matrix.get("conductor_file_modified") is True,
        "active_roadmap_modified": matrix.get("active_roadmap_modified") is True,
        "source_artifacts": source_records,
        "source_checksums": {
            row["path"]: row["sha256"]
            for row in source_records
            if row.get("readable_json_object") is True
        },
        "matrix_summary": _matrix_summary(matrix),
        "phase_outcomes": _phase_outcomes(
            matrix,
            local_sota_receipt_status,
            clean_verifier_status,
            repair_gate_status,
            repair_ladder_status,
            context_fixture_status,
            constraintbench_fixture_status,
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
        "protected_file_policy": {
            "ops_status_modified": False,
            "ops_changelog_modified": False,
            "traceability_modified": False,
            "research_roadmap_modified": False,
            "research_conductor_modified": False,
            "reason": "conductor reconciliation owns ops/status/changelog/traceability updates",
        },
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
    """Build and persist the Exp 3218 capstone deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_records(root: Path) -> list[JsonDict]:
    records: list[JsonDict] = []
    for rel_path in SOURCE_PATHS:
        path = root / rel_path
        payload = read_json_object(path)
        records.append(
            {
                "experiment_id": _source_experiment_id(payload, rel_path.stem),
                "path": rel_path.as_posix(),
                "present": path.is_file(),
                "readable_json_object": bool(payload),
                "schema_version": str(payload.get("schema_version") or payload.get("schema") or ""),
                "honest_verdict": str(payload.get("honest_verdict") or ""),
                "sha256": sha256_file(path),
            }
        )
    return records


def _phase_outcomes(
    matrix: Mapping[str, Any],
    local_sota_receipt_status: str,
    clean_verifier_status: str,
    repair_gate_status: str,
    repair_ladder_status: str,
    context_fixture_status: str,
    constraintbench_fixture_status: str,
    fr11_self_learning_status: str,
    hardware_sampler_status: str,
) -> JsonDict:
    fr11_boundaries = _as_mapping(matrix.get("fr11_claim_boundaries"))
    hardware_boundaries = _as_mapping(matrix.get("hardware_claim_boundaries"))
    return {
        "cuda_receipt_recovery": {
            "status": local_sota_receipt_status,
            "verdict": _phase_verdict(local_sota_receipt_status),
        },
        "clean_verifier": {
            "status": clean_verifier_status,
            "verdict": _phase_verdict(clean_verifier_status),
        },
        "context_fixtures": {
            "status": context_fixture_status,
            "verdict": _phase_verdict(context_fixture_status),
        },
        "constraintbench_fixtures": {
            "status": constraintbench_fixture_status,
            "verdict": _phase_verdict(constraintbench_fixture_status),
        },
        "structured_repair": {
            "repair_gate_status": repair_gate_status,
            "repair_ladder_status": repair_ladder_status,
            "repair_status": str(matrix.get("repair_status") or ""),
        },
        "fr11_self_learning": {
            "status": fr11_self_learning_status,
            "verdict": _phase_verdict(fr11_self_learning_status),
            "controller_memory_promotion_allowed": (
                fr11_boundaries.get("controller_memory_promotion_allowed") is True
            ),
            "queue_promotion_allowed": fr11_boundaries.get("queue_promotion_allowed") is True,
            "model_weight_update_claimed": (
                fr11_boundaries.get("model_weight_update_claimed") is True
            ),
        },
        "hardware_boundary": {
            "status": hardware_sampler_status,
            "verdict": _phase_verdict(hardware_sampler_status),
            "authenticated_hardware_transcript_present": (
                hardware_boundaries.get("authenticated_hardware_transcript_present") is True
            ),
            "speedup_claim_allowed": hardware_boundaries.get("speedup_claim_allowed") is True,
            "tsu_or_kona_claim_allowed": (
                hardware_boundaries.get("tsu_or_kona_claim_allowed") is True
            ),
        },
    }


def _phase_verdict(status: str) -> str:
    if status.startswith(("passed", "clean", "repair_ready", "authenticated")):
        return "passed"
    if status.startswith("available"):
        return "available"
    if status.startswith("diagnostic_only"):
        return "diagnostic_only"
    if status.startswith(("blocked", "missing", "no_authenticated")):
        return "blocked"
    if "gated_skipped" in status:
        return "gated_skipped"
    return "blocked"


def _claim_boundaries(matrix: Mapping[str, Any], paper_ready: bool) -> JsonDict:
    hardware = _as_mapping(matrix.get("hardware_claim_boundaries"))
    fr11 = _as_mapping(matrix.get("fr11_claim_boundaries"))
    transcript_present = hardware.get("authenticated_hardware_transcript_present") is True
    return {
        "paper_ready_claim_allowed": paper_ready,
        "repair_claim_allowed": str(matrix.get("repair_status") or "") == "repair_ready",
        "hardware_speedup_claim_allowed": (
            hardware.get("speedup_claim_allowed") is True and transcript_present
        ),
        "tsu_or_kona_claim_allowed": (
            hardware.get("tsu_or_kona_claim_allowed") is True and transcript_present
        ),
        "model_weight_self_learning_claim_allowed": (
            matrix.get("model_weight_self_learning_claim_allowed") is True
            and fr11.get("model_weight_update_claimed") is True
        ),
    }


def _invariant_violations(
    matrix: Mapping[str, Any],
    publication_blocker_count: int,
    paper_ready: bool,
    claim_boundaries: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    hardware = _as_mapping(matrix.get("hardware_claim_boundaries"))
    fr11 = _as_mapping(matrix.get("fr11_claim_boundaries"))
    if matrix.get("cross_corpus_matrix_v31_ready") is not True:
        violations.append("matrix_v31 authority is missing or not ready")
    if paper_ready and publication_blocker_count > 0:
        violations.append("matrix_v31 paper_ready=true while publication blockers remain")
    if hardware.get("speedup_claim_allowed") is True and (
        hardware.get("authenticated_hardware_transcript_present") is not True
    ):
        violations.append("hardware speedup claim lacks authenticated transcript in matrix_v31")
    if hardware.get("tsu_or_kona_claim_allowed") is True and (
        hardware.get("authenticated_hardware_transcript_present") is not True
    ):
        violations.append("TSU/Kona claim lacks authenticated transcript in matrix_v31")
    if fr11.get("model_weight_update_claimed") is True and (
        claim_boundaries.get("model_weight_self_learning_claim_allowed") is not True
    ):
        violations.append("FR-11 model-weight self-learning claim is not proved by matrix_v31")
    if matrix.get("conductor_file_modified") is True:
        violations.append("matrix_v31 reports conductor file modification")
    if matrix.get("active_roadmap_modified") is True:
        violations.append("matrix_v31 reports active roadmap modification")
    return violations


def _matrix_summary(matrix: Mapping[str, Any]) -> JsonDict:
    return {
        "cross_corpus_matrix_v31_ready": matrix.get("cross_corpus_matrix_v31_ready") is True,
        "paper_ready": matrix.get("paper_ready") is True,
        "publication_blocker_count": _int_or_none(matrix.get("publication_blocker_count")),
        "blocker_delta_from_v30": _int_or_none(matrix.get("blocker_delta_from_v30")),
        "next_top_gap": str(matrix.get("next_top_gap") or ""),
        "honest_verdict": str(matrix.get("honest_verdict") or ""),
    }


def _recommended_next_milestone_theme(next_top_gap: str) -> str:
    if next_top_gap.startswith("cuda_offload") or "local_sota_receipt" in next_top_gap:
        return "cuda_environment_repair_and_clean_local_sota_receipt_recovery"
    if "clean_live_verifier" in next_top_gap:
        return "clean_live_verifier_v12_gate_clearance_after_receipt"
    if "repair" in next_top_gap:
        return "structured_repair_gate_unblock_and_ladder_execution"
    if "fr11" in next_top_gap.lower():
        return "fr11_controller_memory_nonforgetting_promotion"
    if "hardware" in next_top_gap:
        return "authenticated_hardware_transcript_or_explicit_no_speedup_boundary"
    return "publication_blocker_retirement_review"


def _inference_substrate() -> JsonDict:
    return {
        "kind": "capstone_aggregation_from_checked_in_matrix_v31_and_dot297_artifacts",
        "source": "matrix_v31_and_dot297_source_artifacts",
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
    if artifact.get("capstone_v297_ready") is not True:
        first = str(_as_list(artifact.get("invariant_violations"))[0])
        return f"blocked: capstone_v297_ready=false; {first}"
    return (
        "complete: capstone_v297_ready=true; "
        f"paper_ready={str(artifact.get('paper_ready')).lower()}; "
        f"publication_blocker_count={artifact.get('publication_blocker_count')}; "
        f"blocker_delta_from_v30={artifact.get('blocker_delta_from_v30')}; "
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


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _duration(started_s: float, now_s: float | None) -> float:
    end = float(now_s) if now_s is not None else time.perf_counter()
    return round(max(0.0, end - started_s), 6)
