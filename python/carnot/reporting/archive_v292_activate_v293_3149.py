"""Build the Exp 3149 archive and .293 handoff artifact.

Spec refs: REQ-REPORT-3149, SCENARIO-REPORT-3149.

This module archives the completed .292 milestone without activating anything.
It reads checked-in result, roadmap, and ops files, then states what those
files support. No model, verifier, repair, solver, conductor, synthesis, or
hardware path is executed here; this is deliberately aggregation-only work.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
PRIOR_MILESTONE = "2026.05.292"
NEXT_MILESTONE = "2026.05.293"
SCHEMA = "carnot.archive_activation.v292_to_v293.v1"
ARTIFACT = "experiment_3149_archive_v292_activate_v293"
OUTPUT_REL_PATH = Path("results/experiment_3149_archive_v292_activate_v293.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3149_archive_v292_activate_v293.py"

MATRIX_V26_REL_PATH = Path("results/experiment_3147_cross_corpus_matrix_v26.json")
CAPSTONE_V292_REL_PATH = Path("results/experiment_3148_capstone_v292.json")
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
    ("matrix_v26", MATRIX_V26_REL_PATH),
    ("capstone_v292", CAPSTONE_V292_REL_PATH),
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
    "executes_verifiers": False,
    "executes_repairs": False,
    "executes_solvers": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "local_repo_only": True,
    "no_live_llm_inference": True,
    "source": "checked_in_artifacts",
    "live_model_calls": 0,
    "hardware_commands_run": [],
}


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed on absent or malformed evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_mapping(path: Path) -> JsonDict:
    """Read roadmap YAML without treating malformed content as a valid handoff."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return a checksum tying summary claims back to exact source bytes."""

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
    """REQ-REPORT-3149: synthesize the .292 archive and .293 handoff record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V26_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V292_REL_PATH)
    staged = read_yaml_mapping(root_path / STAGED_ROADMAP_REL_PATH)
    active = read_yaml_mapping(root_path / ACTIVE_ROADMAP_REL_PATH)
    source_artifacts = [
        _source_artifact(root_path, role, rel_path) for role, rel_path in SOURCE_PATHS
    ]
    roadmap_handoff = _roadmap_handoff(root_path, staged, active)
    prior_capstone_ready = capstone.get("capstone_ready") is True
    prior_paper_ready = capstone.get("paper_ready") is True
    prior_publication_blocker_count, blocker_source = _publication_blocker_count(capstone)
    blocker_delta_from_v25, delta_source = _blocker_delta_from_v25(capstone)
    status_summary = _status_summary_292(
        matrix,
        capstone,
        blocker_count=prior_publication_blocker_count,
        blocker_source=blocker_source,
        blocker_delta=blocker_delta_from_v25,
        delta_source=delta_source,
    )
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
        "archive_v292_activate_v293_ready": ready,
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": prior_paper_ready,
        "prior_paper_ready_source_field_present": "paper_ready" in capstone,
        "prior_publication_blocker_count": prior_publication_blocker_count,
        "prior_publication_blocker_count_source": blocker_source,
        "blocker_delta_from_v25": blocker_delta_from_v25,
        "blocker_delta_from_v25_source": delta_source,
        "status_summary_292": status_summary,
        "carry_forward_blockers": _carry_forward_blockers(status_summary),
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
    """Build and persist the Exp 3149 deliverable JSON."""

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
        reasons.append("roadmap milestone is not 2026.05.293")
    if not roadmap_handoff.get("milestone_doc_matches"):
        reasons.append(
            "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        )
    if not roadmap_handoff.get("non_empty_tasks"):
        reasons.append("roadmap has no tasks")
    if not vnext_doc_present:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _status_summary_292(
    matrix: Mapping[str, Any],
    capstone: Mapping[str, Any],
    *,
    blocker_count: int,
    blocker_source: str,
    blocker_delta: int,
    delta_source: str,
) -> JsonDict:
    return {
        "paper_ready": capstone.get("paper_ready") is True,
        "capstone_ready": capstone.get("capstone_ready") is True,
        "matrix_v26_ready": matrix.get("matrix_v26_ready") is True,
        "publication_blocker_count": blocker_count,
        "publication_blocker_count_source": blocker_source,
        "blocker_delta_from_v25": blocker_delta,
        "blocker_delta_source": delta_source,
        "next_top_gap": _text_field(capstone, "next_top_gap"),
        "false_accept_recovery_status": _text_field(capstone, "false_accept_recovery_status"),
        "live_verifier_status": _text_field(capstone, "live_verifier_status"),
        "repair_gate_status": _text_field(capstone, "repair_gate_status"),
        "repair_ladder_status": _text_field(capstone, "repair_ladder_status"),
        "fr11_self_learning_status": _text_field(capstone, "fr11_self_learning_status"),
        "ebt_arm_status": _text_field(capstone, "ebt_arm_status"),
        "kan_status": _text_field(capstone, "kan_status"),
        "sampler_hardware_status": _text_field(capstone, "sampler_hardware_status"),
        "source_artifacts": _dict_rows(capstone.get("source_artifacts")),
    }


def _publication_blocker_count(capstone: Mapping[str, Any]) -> tuple[int, str]:
    return _int_field(capstone, "publication_blocker_count")


def _blocker_delta_from_v25(capstone: Mapping[str, Any]) -> tuple[int, str]:
    return _int_field(capstone, "blocker_delta_from_v25")


def _int_field(payload: Mapping[str, Any], field: str) -> tuple[int, str]:
    value = payload.get(field)
    if not isinstance(value, bool) and isinstance(value, int):
        return value, f"capstone_{field}"
    return 0, "missing"


def _text_field(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    return str(value) if value not in (None, "") else ""


def _carry_forward_blockers(status: Mapping[str, Any]) -> list[JsonDict]:
    return [
        _blocker(
            "publication_blockers_55",
            "55 publication blockers",
            "publication_blocker_count",
            status["publication_blocker_count"],
            55,
            status["publication_blocker_count"] == 55,
        ),
        _blocker(
            "flagged_live_verifier_evidence",
            "flagged live verifier evidence",
            "live_verifier_status",
            status["live_verifier_status"],
            "flagged",
            status["live_verifier_status"] == "flagged",
        ),
        _blocker(
            "repair_gate_blocked_other",
            "repair gate blocked_other",
            "repair_gate_status",
            status["repair_gate_status"],
            "contains:blocked_other",
            "blocked_other" in str(status["repair_gate_status"]),
        ),
        _blocker(
            "repair_ladder_missing_gated",
            "missing/gated repair ladder",
            "repair_ladder_status",
            status["repair_ladder_status"],
            "gated_skipped_missing_artifact",
            status["repair_ladder_status"] == "gated_skipped_missing_artifact",
        ),
        _blocker(
            "fr11_controller_only_memory",
            "FR-11 controller-only memory",
            "fr11_self_learning_status",
            status["fr11_self_learning_status"],
            "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667",
            status["fr11_self_learning_status"]
            == "bounded_controller_memory_only_no_weight_update_vera_0.833333_experience_0.666667",
        ),
        _blocker(
            "ebt_arm_projection_only",
            "EBT/ARM projection-only",
            "ebt_arm_status",
            status["ebt_arm_status"],
            "projection_only_no_live_integration_blockers_6",
            status["ebt_arm_status"] == "projection_only_no_live_integration_blockers_6",
        ),
        _blocker(
            "kan_bounded_only",
            "KAN bounded only",
            "kan_status",
            status["kan_status"],
            "bounded_monitor_records_2_no_deployed_verifier",
            status["kan_status"] == "bounded_monitor_records_2_no_deployed_verifier",
        ),
        _blocker(
            "no_authenticated_hardware_speedup",
            "no authenticated hardware speedup",
            "sampler_hardware_status",
            status["sampler_hardware_status"],
            "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8",
            status["sampler_hardware_status"]
            == "blocked_no_authenticated_speedup_no_hardware_commands_missing_operator_evidence_8",
        ),
    ]


def _blocker(
    blocker_id: str,
    description: str,
    source_field: str,
    value: Any,
    expected: Any,
    matches: bool,
) -> JsonDict:
    return {
        "blocker_id": blocker_id,
        "description": description,
        "source": CAPSTONE_V292_REL_PATH.as_posix(),
        "source_field": source_field,
        "value": value,
        "expected_carry_forward_value": expected,
        "matches_expected": matches,
    }


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["archive_v292_activate_v293_ready"] is True:
        roadmap = _as_mapping(artifact.get("roadmap_handoff"))
        return (
            "complete: archive_v292_activate_v293_ready=true; "
            f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
            f"prior_paper_ready={str(artifact['prior_paper_ready']).lower()}; "
            f"prior_publication_blocker_count={artifact['prior_publication_blocker_count']}; "
            f"blocker_delta_from_v25={artifact['blocker_delta_from_v25']}; "
            f"next_milestone={artifact['next_milestone']}; "
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
