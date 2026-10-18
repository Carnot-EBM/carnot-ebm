"""Build the Exp 3109 archive and .290 handoff artifact.

Spec refs: REQ-REPORT-3109, SCENARIO-REPORT-3109.

This module turns the completed .289 capstone into a machine-readable .290
handoff without activating anything itself. The artifact is intentionally
evidence-only: it reads checked-in JSON/YAML/Markdown files, records the
remaining blockers, and declares that no live model, solver, repair,
synthesis, conductor, or hardware path was executed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
PRIOR_MILESTONE = "2026.05.289"
NEXT_MILESTONE = "2026.05.290"
SCHEMA = "carnot.archive_activation.v289_to_v290.v1"
ARTIFACT = "experiment_3109_archive_v289_activate_v290"
OUTPUT_REL_PATH = Path("results/experiment_3109_archive_v289_activate_v290.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3109_archive_v289_activate_v290.py"

MATRIX_V23_REL_PATH = Path("results/experiment_3107_cross_corpus_matrix_v23.json")
CAPSTONE_V289_REL_PATH = Path("results/experiment_3108_capstone_v289.json")
FORMAL_FEEDBACK_REL_PATH = Path("results/experiment_3100_z3_oracle_feedback_v2.json")
VERIFIER_CALIBRATION_REL_PATH = Path(
    "results/experiment_3101_local_sota_verifier_calibration_v4.json"
)
REPAIR_MICRO_PANEL_REL_PATH = Path(
    "results/experiment_3102_gated_structured_repair_micro_panel_v3.json"
)
FR11_STRESS_REL_PATH = Path(
    "results/experiment_3103_fr11_resyn_kancl_stress_promotion_boundary_v2.json"
)
EBT_ARM_REL_PATH = Path("results/experiment_3104_ebt_arm_sidecar_pipeline_boundary_v2.json")
CLUT_SAMPLER_REL_PATH = Path(
    "results/experiment_3105_clut_random_variate_sampler_microbench_v1.json"
)
GATEMATE_SSQA_REL_PATH = Path(
    "results/experiment_3106_gatemate_ssqa_operator_evidence_ingestion_v3.json"
)
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
AGENTS_REL_PATH = Path("AGENTS.md")
CODEX_REL_PATH = Path("CODEX.md")
CLAUDE_REL_PATH = Path("CLAUDE.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
OPS_STATUS_REL_PATH = Path("ops/status.md")
OPS_CHANGELOG_REL_PATH = Path("ops/changelog.md")
TRACEABILITY_REL_PATH = Path("_bmad/traceability.md")

SOURCE_PATHS = (
    ("matrix_v23", MATRIX_V23_REL_PATH),
    ("capstone_v289", CAPSTONE_V289_REL_PATH),
    ("formal_feedback_v2", FORMAL_FEEDBACK_REL_PATH),
    ("verifier_calibration_gate", VERIFIER_CALIBRATION_REL_PATH),
    ("repair_micro_panel", REPAIR_MICRO_PANEL_REL_PATH),
    ("fr11_stress_boundary", FR11_STRESS_REL_PATH),
    ("ebt_arm_sidecar_boundary", EBT_ARM_REL_PATH),
    ("clut_sampler_microbench", CLUT_SAMPLER_REL_PATH),
    ("gatemate_ssqa_operator_evidence", GATEMATE_SSQA_REL_PATH),
    ("staged_roadmap", STAGED_ROADMAP_REL_PATH),
    ("active_roadmap", ACTIVE_ROADMAP_REL_PATH),
    ("vnext_doc", VNEXT_DOC_REL_PATH),
    ("agents_instructions", AGENTS_REL_PATH),
    ("codex_instructions", CODEX_REL_PATH),
    ("claude_instructions", CLAUDE_REL_PATH),
    ("research_conductor", CONDUCTOR_REL_PATH),
    ("ops_status", OPS_STATUS_REL_PATH),
    ("ops_changelog", OPS_CHANGELOG_REL_PATH),
    ("traceability", TRACEABILITY_REL_PATH),
)
INFERENCE_SUBSTRATE = {
    "kind": "aggregation_from_upstream_artifacts",
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "executes_live_repair": False,
    "local_repo_only": True,
    "no_live_llm_inference": True,
    "source": "checked_in_artifacts",
}
EXPECTED_MODEL_GAP_ROW_IDS = ["dot289:exp3099_local_sota_confidence_abstention_panel"]
EXPECTED_MISSING_REPAIR_MICRO_PANEL = REPAIR_MICRO_PANEL_REL_PATH.as_posix()


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while keeping missing evidence visibly empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_mapping(path: Path) -> JsonDict:
    """Read roadmap YAML without treating malformed content as a valid plan."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a checksum for later audit of each source artifact."""

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
    """REQ-REPORT-3109: synthesize the .289 archive and .290 handoff record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V23_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V289_REL_PATH)
    formal_feedback = read_json_object(root_path / FORMAL_FEEDBACK_REL_PATH)
    verifier_calibration = read_json_object(root_path / VERIFIER_CALIBRATION_REL_PATH)
    fr11_stress = read_json_object(root_path / FR11_STRESS_REL_PATH)
    ebt_arm = read_json_object(root_path / EBT_ARM_REL_PATH)
    clut_sampler = read_json_object(root_path / CLUT_SAMPLER_REL_PATH)
    gatemate_ssqa = read_json_object(root_path / GATEMATE_SSQA_REL_PATH)
    staged = read_yaml_mapping(root_path / STAGED_ROADMAP_REL_PATH)
    active = read_yaml_mapping(root_path / ACTIVE_ROADMAP_REL_PATH)
    source_artifacts = [
        _source_artifact(root_path, role, rel_path) for role, rel_path in SOURCE_PATHS
    ]
    roadmap_handoff = _roadmap_handoff(root_path, staged, active)
    prior_capstone_ready = capstone.get("capstone_ready") is True
    prior_paper_ready = capstone.get("paper_ready") is True
    blocked_reasons = _blocked_reasons(
        capstone_present=bool(capstone),
        prior_capstone_ready=prior_capstone_ready,
        roadmap_handoff=roadmap_handoff,
        vnext_doc_present=(root_path / VNEXT_DOC_REL_PATH).is_file(),
    )
    ready = not blocked_reasons

    artifact: JsonDict = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "prior_milestone": PRIOR_MILESTONE,
        "next_milestone": _next_milestone(roadmap_handoff),
        "archive_v289_activate_v290_ready": ready,
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": prior_paper_ready,
        "prior_paper_ready_source_field_present": "paper_ready" in capstone,
        "status_summary_289": _status_summary_289(matrix, capstone),
        "carry_forward_blockers": _carry_forward_blockers(
            capstone,
            matrix,
            formal_feedback,
            verifier_calibration,
            fr11_stress,
            ebt_arm,
            clut_sampler,
            gatemate_ssqa,
        ),
        "roadmap_handoff": roadmap_handoff,
        "source_artifacts": source_artifacts,
        "source_checksums": {str(row["path"]): row["sha256"] for row in source_artifacts},
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row["present"] is not True
        ],
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "activation_performed_by_this_task": False,
        "research_roadmap_yaml_modified": False,
        "research_roadmap_next_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "traceability_updated": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
        "no_live_repair_rerun": True,
        "no_historical_artifact_rewrite": True,
        "no_push": True,
        "blocked_reasons": blocked_reasons,
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
    """Build and persist the Exp 3109 deliverable JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _source_artifact(root: Path, role: str, rel_path: Path) -> JsonDict:
    path = root / rel_path
    return {
        "role": role,
        "path": rel_path.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path),
    }


def _roadmap_handoff(
    root: Path,
    staged: Mapping[str, Any],
    active: Mapping[str, Any],
) -> JsonDict:
    staged_present = (root / STAGED_ROADMAP_REL_PATH).is_file()
    if staged_present:
        source_path = STAGED_ROADMAP_REL_PATH
        source_payload: Mapping[str, Any] = staged
        used_fallback = False
    else:
        source_path = ACTIVE_ROADMAP_REL_PATH
        source_payload = active
        used_fallback = True
    milestone = str(source_payload.get("milestone") or "")
    milestone_doc = str(source_payload.get("milestone_doc") or "")
    task_ids = _task_ids(source_payload)
    return {
        "requested_staged_roadmap_path": STAGED_ROADMAP_REL_PATH.as_posix(),
        "requested_staged_roadmap_present": staged_present,
        "source_path": source_path.as_posix(),
        "source_present": (root / source_path).is_file(),
        "used_active_roadmap_fallback": used_fallback,
        "active_roadmap_milestone": str(active.get("milestone") or ""),
        "observed_milestone": milestone,
        "expected_milestone": NEXT_MILESTONE,
        "milestone_matches": milestone == NEXT_MILESTONE,
        "observed_milestone_doc": milestone_doc,
        "expected_milestone_doc": VNEXT_DOC_REL_PATH.as_posix(),
        "milestone_doc_matches": milestone_doc == VNEXT_DOC_REL_PATH.as_posix(),
        "task_ids": task_ids,
        "non_empty_tasks": bool(task_ids),
    }


def _blocked_reasons(
    *,
    capstone_present: bool,
    prior_capstone_ready: bool,
    roadmap_handoff: Mapping[str, Any],
    vnext_doc_present: bool,
) -> list[str]:
    reasons: list[str] = []
    if not capstone_present:
        reasons.append("prior capstone artifact missing or malformed")
    elif not prior_capstone_ready:
        reasons.append("prior capstone is not capstone_ready=true")
    if not roadmap_handoff.get("source_present"):
        reasons.append("roadmap handoff source is missing")
    if not roadmap_handoff.get("milestone_matches"):
        reasons.append("roadmap milestone is not 2026.05.290")
    if not roadmap_handoff.get("milestone_doc_matches"):
        reasons.append(
            "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        )
    if not roadmap_handoff.get("non_empty_tasks"):
        reasons.append("roadmap has no tasks")
    if not vnext_doc_present:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _status_summary_289(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    count, count_source = _publication_blocker_count(capstone, matrix)
    return {
        "paper_ready": capstone.get("paper_ready") is True,
        "capstone_ready": capstone.get("capstone_ready") is True,
        "matrix_v23_ready": matrix.get("matrix_v23_ready") is True
        or _as_mapping(capstone.get("matrix_v23_summary")).get("matrix_v23_ready") is True,
        "publication_blocker_count": count,
        "publication_blocker_count_source": count_source,
        "verifier_gain_status": _status_value("verifier_gain_status", matrix, capstone),
        "repair_claim_status": _status_value("repair_claim_status", matrix, capstone),
        "fr11_self_learning_status": _status_value(
            "fr11_self_learning_status",
            matrix,
            capstone,
        ),
        "ebt_arm_status": _status_value("ebt_arm_status", matrix, capstone),
        "sampler_hardware_status": _status_value(
            "sampler_hardware_status",
            matrix,
            capstone,
        ),
        "gatemate_status": _status_value("gatemate_status", matrix, capstone),
        "ssqa_status": _status_value("ssqa_status", matrix, capstone),
        "source_artifacts": _dict_rows(capstone.get("source_artifacts")),
        "missing_capstone_input_artifacts": _dict_rows(
            capstone.get("missing_capstone_input_artifacts") or matrix.get("missing_artifacts")
        ),
        "headline_model_spec_gaps": _dict_rows(
            capstone.get("headline_model_spec_gaps") or matrix.get("headline_model_spec_gaps")
        ),
    }


def _status_value(
    field: str,
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
) -> str:
    value = capstone.get(field)
    if value in (None, ""):
        value = matrix.get(field)
    return str(value or "")


def _publication_blocker_count(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> tuple[int, str]:
    capstone_count = capstone.get("publication_blocker_count")
    if not isinstance(capstone_count, bool) and isinstance(capstone_count, int):
        return capstone_count, "capstone_publication_blocker_count"
    blockers = capstone.get("publication_blockers")
    if isinstance(blockers, list):
        return len(blockers), "capstone_publication_blockers_length"
    matrix_summary = _as_mapping(capstone.get("matrix_v23_summary"))
    summary_count = matrix_summary.get("publication_blocker_count")
    if not isinstance(summary_count, bool) and isinstance(summary_count, int):
        return summary_count, "capstone_matrix_v23_summary"
    count = _first_int_from_text(str(capstone.get("honest_verdict") or ""))
    if count is not None:
        return count, "capstone_honest_verdict"
    count = _first_int_from_text(str(matrix.get("honest_verdict") or ""))
    if count is not None:
        return count, "matrix_honest_verdict"
    return 0, "missing"


def _first_int_from_text(text: str) -> int | None:
    match = re.search(r"publication_blocker_count=(\d+)", text)
    return int(match.group(1)) if match else None


def _carry_forward_blockers(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
    formal_feedback: Mapping[str, Any],
    verifier_calibration: Mapping[str, Any],
    fr11_stress: Mapping[str, Any],
    ebt_arm: Mapping[str, Any],
    clut_sampler: Mapping[str, Any],
    gatemate_ssqa: Mapping[str, Any],
) -> list[JsonDict]:
    status = _status_summary_289(matrix, capstone)
    missing_paths = [
        str(row.get("path") or "")
        for row in status["missing_capstone_input_artifacts"]
        if row.get("path")
    ]
    model_gap_row_ids = [
        str(row.get("row_id") or "")
        for row in status["headline_model_spec_gaps"]
        if row.get("row_id")
    ]
    formal_ready = _formal_feedback_ready(formal_feedback)
    calibration_value = {
        "status": str(verifier_calibration.get("status") or ""),
        "honest_verdict": str(verifier_calibration.get("honest_verdict") or ""),
        "gate_check_summary": str(verifier_calibration.get("gate_check_summary") or ""),
    }
    fr11_value = {
        "fr11_self_learning_status": status["fr11_self_learning_status"],
        "promotion_decision": str(fr11_stress.get("promotion_decision") or ""),
        "soundness_mistakes": _int_or(fr11_stress.get("soundness_mistakes"), 0),
        "completeness_mistakes": _int_or(fr11_stress.get("completeness_mistakes"), 0),
    }
    clut_value = {
        "sampler_hardware_status": status["sampler_hardware_status"],
        "hardware_claim_made": clut_sampler.get("hardware_claim_made") is True,
        "hardware_commands_run": _as_list(clut_sampler.get("hardware_commands_run")),
    }
    gate_values = {
        "gatemate_status": status["gatemate_status"],
        "ssqa_status": status["ssqa_status"],
        "gatemate_rerun_allowed": gatemate_ssqa.get("gatemate_rerun_allowed") is True,
        "ssqa_readback_allowed": gatemate_ssqa.get("ssqa_readback_allowed") is True,
    }
    _ = ebt_arm
    return [
        {
            "blocker_id": "publication_blockers_36",
            "description": "36 publication blockers",
            "source": CAPSTONE_V289_REL_PATH.as_posix(),
            "source_field": "publication_blocker_count",
            "value": status["publication_blocker_count"],
            "expected_carry_forward_value": 36,
            "matches_expected": status["publication_blocker_count"] == 36,
        },
        {
            "blocker_id": "local_sota_metadata_cache_gap",
            "description": "local SOTA metadata/cache gap",
            "source": CAPSTONE_V289_REL_PATH.as_posix(),
            "source_field": "headline_model_spec_gaps",
            "value": model_gap_row_ids,
            "expected_carry_forward_value": EXPECTED_MODEL_GAP_ROW_IDS,
            "matches_expected": model_gap_row_ids == EXPECTED_MODEL_GAP_ROW_IDS,
        },
        {
            "blocker_id": "formal_feedback_v2_not_ready",
            "description": "formal_feedback_v2_ready=false",
            "source": FORMAL_FEEDBACK_REL_PATH.as_posix(),
            "source_field": "formal_feedback_v2_ready",
            "value": formal_ready,
            "expected_carry_forward_value": False,
            "matches_expected": formal_ready is False,
        },
        {
            "blocker_id": "gated_verifier_calibration",
            "description": "gated verifier calibration",
            "source": VERIFIER_CALIBRATION_REL_PATH.as_posix(),
            "source_field": "honest_verdict,gate_check_summary",
            "value": calibration_value,
            "expected_carry_forward_value": "blocked_gate_check_failed",
            "matches_expected": calibration_value["honest_verdict"] == "blocked_gate_check_failed",
        },
        {
            "blocker_id": "missing_repair_micro_panel",
            "description": "missing repair micro-panel",
            "source": CAPSTONE_V289_REL_PATH.as_posix(),
            "source_field": "missing_capstone_input_artifacts",
            "value": missing_paths,
            "expected_carry_forward_value": [EXPECTED_MISSING_REPAIR_MICRO_PANEL],
            "matches_expected": EXPECTED_MISSING_REPAIR_MICRO_PANEL in missing_paths,
        },
        {
            "blocker_id": "fr11_completeness_retention_promotion_block",
            "description": "FR-11 completeness/retention promotion block",
            "source": FR11_STRESS_REL_PATH.as_posix(),
            "source_field": "promotion_decision,soundness_mistakes,completeness_mistakes",
            "value": fr11_value,
            "expected_carry_forward_value": {
                "promotion_decision": "blocked",
                "soundness_mistakes": 0,
                "completeness_mistakes_min": 1,
            },
            "matches_expected": (
                fr11_value["promotion_decision"] == "blocked"
                and fr11_value["soundness_mistakes"] == 0
                and fr11_value["completeness_mistakes"] > 0
            ),
        },
        {
            "blocker_id": "ebt_arm_projection_only",
            "description": "EBT/ARM projection-only",
            "source": EBT_ARM_REL_PATH.as_posix(),
            "source_field": "capstone.ebt_arm_status",
            "value": status["ebt_arm_status"],
            "expected_carry_forward_value": "projection_only_sidecar_pipeline_no_model_integration",
            "matches_expected": (
                status["ebt_arm_status"] == "projection_only_sidecar_pipeline_no_model_integration"
            ),
        },
        {
            "blocker_id": "clut_cpu_diagnostic_only",
            "description": "cLUT CPU diagnostic only",
            "source": CLUT_SAMPLER_REL_PATH.as_posix(),
            "source_field": "hardware_claim_made,hardware_commands_run",
            "value": clut_value,
            "expected_carry_forward_value": {
                "sampler_hardware_status": "diagnostic_only_cpu_microbench_no_hardware_speedup",
                "hardware_claim_made": False,
                "hardware_commands_run": [],
            },
            "matches_expected": clut_value
            == {
                "sampler_hardware_status": "diagnostic_only_cpu_microbench_no_hardware_speedup",
                "hardware_claim_made": False,
                "hardware_commands_run": [],
            },
        },
        {
            "blocker_id": "gatemate_ssqa_missing_operator_evidence",
            "description": "GateMate/SSQA missing operator evidence",
            "source": GATEMATE_SSQA_REL_PATH.as_posix(),
            "source_field": "gatemate_rerun_allowed,ssqa_readback_allowed",
            "value": gate_values,
            "expected_carry_forward_value": {
                "gatemate_status": "blocked_no_rerun_operator_actions_required_no_speedup_claim",
                "ssqa_status": "gated_skipped_host_visible_readback_missing",
                "gatemate_rerun_allowed": False,
                "ssqa_readback_allowed": False,
            },
            "matches_expected": gate_values
            == {
                "gatemate_status": "blocked_no_rerun_operator_actions_required_no_speedup_claim",
                "ssqa_status": "gated_skipped_host_visible_readback_missing",
                "gatemate_rerun_allowed": False,
                "ssqa_readback_allowed": False,
            },
        },
    ]


def _formal_feedback_ready(payload: Mapping[str, Any]) -> bool | None:
    value = payload.get("formal_feedback_v2_ready")
    if isinstance(value, bool):
        return value
    verdict = str(payload.get("honest_verdict") or "")
    if "formal_feedback_v2_ready=false" in verdict:
        return False
    if "formal_feedback_v2_ready=true" in verdict:
        return True
    return None


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["archive_v289_activate_v290_ready"] is True:
        status_summary = _as_mapping(artifact.get("status_summary_289"))
        roadmap = _as_mapping(artifact.get("roadmap_handoff"))
        return (
            "complete: archive_v289_activate_v290_ready=true; "
            f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
            f"prior_paper_ready={str(artifact['prior_paper_ready']).lower()}; "
            f"next_milestone={artifact['next_milestone']}; "
            f"publication_blocker_count={status_summary.get('publication_blocker_count')}; "
            f"roadmap_source={roadmap.get('source_path')}"
        )
    reasons = _as_list(artifact.get("blocked_reasons"))
    if any("capstone" in str(reason) for reason in reasons):
        prefix = "blocked_prior_capstone_not_ready"
    else:
        prefix = "blocked_roadmap_handoff_not_ready"
    return (
        f"{prefix}: "
        f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
        f"next_milestone={artifact['next_milestone']}; "
        f"reasons={'; '.join(str(reason) for reason in reasons)}"
    )


def _next_milestone(roadmap_handoff: Mapping[str, Any]) -> str:
    observed = str(roadmap_handoff.get("observed_milestone") or "")
    return observed or NEXT_MILESTONE


def _duration(start: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - start), 6)


def _task_ids(payload: Mapping[str, Any]) -> list[str]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [
        str(task["id"])
        for task in tasks
        if isinstance(task, Mapping) and task.get("id") not in (None, "")
    ]


def _dict_rows(value: Any) -> list[JsonDict]:
    return [dict(row) for row in _as_list(value) if isinstance(row, Mapping)]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
