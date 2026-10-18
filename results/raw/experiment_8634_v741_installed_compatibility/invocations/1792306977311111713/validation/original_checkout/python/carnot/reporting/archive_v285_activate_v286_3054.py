"""Build the Exp 3054 archive and .286 handoff artifact.

Spec refs: REQ-REPORT-3054, SCENARIO-REPORT-3054.

The archive task is an auditable bookkeeping record. It reads the completed
.285 capstone, matrix v19, and roadmap handoff files, then records whether the
.286 handoff is ready without activating the roadmap, running the conductor, or
turning any bounded claim into a paper-ready claim.
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
RUN_DATE = "20260525"
PRIOR_MILESTONE = "2026.05.285"
NEXT_MILESTONE = "2026.05.286"
SCHEMA = "carnot.archive_activation.v285_to_v286.v1"
ARTIFACT = "experiment_3054_archive_v285_activate_v286"
OUTPUT_REL_PATH = Path("results/experiment_3054_archive_v285_activate_v286.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3054_archive_v285_activate_v286.py"

MATRIX_V19_REL_PATH = Path("results/experiment_3052_cross_corpus_matrix_v19.json")
CAPSTONE_V285_REL_PATH = Path("results/experiment_3053_capstone_v285.json")
STAGED_ROADMAP_REL_PATH = Path("research-roadmap-next.yaml")
ACTIVE_ROADMAP_REL_PATH = Path("research-roadmap.yaml")
VNEXT_DOC_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_REL_PATH = Path("scripts/research_conductor.py")
OPS_STATUS_REL_PATH = Path("ops/status.md")
OPS_CHANGELOG_REL_PATH = Path("ops/changelog.md")

COUNT_FIELDS = {
    "clean": "clean_count",
    "flagged": "flagged_count",
    "bounded": "bounded_count",
    "blocked": "blocked_count",
    "gated_skipped": "gated_skipped_count",
    "projection_only": "projection_only_count",
    "missing": "missing_count",
    "retired": "retired_count",
}
STATUS_FIELDS = (
    "repair_claim_status",
    "fr11_self_learning_status",
    "gatemate_status",
    "ssqa_status",
)
EXPECTED_BLOCKERS = (
    {
        "blocker_id": "repair_bounded",
        "status_field": "repair_claim_status",
        "expected_value": "bounded",
        "description": "repair bounded",
    },
    {
        "blocker_id": "gatemate_output_contract_blocked",
        "status_field": "gatemate_status",
        "expected_value": "blocked_output_contract",
        "description": "GateMate output contract blocked",
    },
    {
        "blocker_id": "ssqa_host_visible_smoke_missing",
        "status_field": "ssqa_status",
        "expected_value": "gated_skipped_host_visible_smoke_missing",
        "description": "SSQA host-visible smoke missing",
    },
    {
        "blocker_id": "model_weight_self_learning_out_of_scope",
        "status_field": "fr11_self_learning_status",
        "expected_value": "controller_only_solver_feedback_and_locality_ready",
        "description": "model-weight self-learning out of scope",
    },
)
SOURCE_PATHS = (
    ("matrix_v19", MATRIX_V19_REL_PATH),
    ("capstone_v285", CAPSTONE_V285_REL_PATH),
    ("staged_roadmap", STAGED_ROADMAP_REL_PATH),
    ("active_roadmap", ACTIVE_ROADMAP_REL_PATH),
    ("vnext_doc", VNEXT_DOC_REL_PATH),
    ("research_conductor", CONDUCTOR_REL_PATH),
    ("ops_status", OPS_STATUS_REL_PATH),
    ("ops_changelog", OPS_CHANGELOG_REL_PATH),
)
INFERENCE_SUBSTRATE = {
    "kind": "aggregation_from_upstream_artifacts",
    "executes_models": False,
    "executes_hardware": False,
    "executes_conductor": False,
    "no_live_llm_inference": True,
    "source": "checked_in_artifacts",
}


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object while treating absence or malformed files as no evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_yaml_mapping(path: Path) -> JsonDict:
    """Read a YAML mapping while treating absence, malformed YAML, and lists as no evidence."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) if text.strip() else {}
    except (OSError, yaml.YAMLError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for an existing file."""

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
    """REQ-REPORT-3054: synthesize the .285 archive and .286 handoff record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V19_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V285_REL_PATH)
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
        "archive_v285_activate_v286_ready": ready,
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": prior_paper_ready,
        "prior_paper_ready_source_field_present": "paper_ready" in capstone,
        "status_summary_285": _status_summary_285(matrix, capstone),
        "carry_forward_blockers": _carry_forward_blockers(capstone, matrix),
        "roadmap_handoff": roadmap_handoff,
        "source_artifacts": source_artifacts,
        "source_checksums": {
            str(row["path"]): row["sha256"] for row in source_artifacts
        },
        "missing_source_artifacts": [
            str(row["path"]) for row in source_artifacts if row["present"] is not True
        ],
        "inference_substrate": dict(INFERENCE_SUBSTRATE),
        "activation_performed_by_this_task": False,
        "research_roadmap_yaml_modified": False,
        "scripts_research_conductor_modified": False,
        "ops_status_updated": False,
        "ops_changelog_updated": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "no_new_model_execution": True,
        "no_new_verifier_run": True,
        "no_new_solver_run": True,
        "no_new_synthesis_run": True,
        "no_new_board_flash": True,
        "no_new_hardware_run": True,
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
    """Build and persist the Exp 3054 deliverable JSON."""

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
        reasons.append("roadmap milestone is not 2026.05.286")
    if not roadmap_handoff.get("milestone_doc_matches"):
        reasons.append(
            "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        )
    if not roadmap_handoff.get("non_empty_tasks"):
        reasons.append("roadmap has no tasks")
    if not vnext_doc_present:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _status_summary_285(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    return {
        "paper_ready": capstone.get("paper_ready") is True,
        "capstone_ready": capstone.get("capstone_ready") is True,
        "matrix_v19_ready": matrix.get("matrix_v19_ready") is True,
        "repair_claim_status": _status_value("repair_claim_status", matrix, capstone),
        "fr11_self_learning_status": _status_value(
            "fr11_self_learning_status",
            matrix,
            capstone,
        ),
        "gatemate_status": _status_value("gatemate_status", matrix, capstone),
        "ssqa_status": _status_value("ssqa_status", matrix, capstone),
        "counts": _count_summary(matrix, capstone),
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


def _count_summary(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    capstone_summary = _as_mapping(capstone.get("matrix_v19_summary"))
    return {
        status: _int_or(
            matrix.get(field_name, capstone_summary.get(status)),
            0,
        )
        for status, field_name in COUNT_FIELDS.items()
    }


def _carry_forward_blockers(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    for spec in EXPECTED_BLOCKERS:
        field = str(spec["status_field"])
        value = _status_value(field, matrix, capstone)
        expected = str(spec["expected_value"])
        blockers.append(
            {
                "blocker_id": spec["blocker_id"],
                "description": spec["description"],
                "status_field": field,
                "value": value,
                "expected_carry_forward_value": expected,
                "source": CAPSTONE_V285_REL_PATH.as_posix(),
                "matches_expected": value == expected,
            }
        )
    return blockers


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["archive_v285_activate_v286_ready"] is True:
        return (
            "complete: archive_v285_activate_v286_ready=true; "
            f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
            f"prior_paper_ready={str(artifact['prior_paper_ready']).lower()}; "
            f"next_milestone={artifact['next_milestone']}; "
            f"roadmap_source={artifact['roadmap_handoff']['source_path']}"
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
