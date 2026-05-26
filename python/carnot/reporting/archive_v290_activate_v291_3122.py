"""Build the Exp 3122 archive and .291 handoff artifact.

Spec refs: REQ-REPORT-3122, SCENARIO-REPORT-3122.

This module converts the completed .290 capstone into a machine-readable .291
handoff without activating the roadmap itself. The work is intentionally
evidence-only: it reads checked-in JSON/YAML/Markdown files, carries forward
the unresolved blocker classes, and declares that no model, verifier, repair,
solver, synthesis, conductor, or hardware path was executed.
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
PRIOR_MILESTONE = "2026.05.290"
NEXT_MILESTONE = "2026.05.291"
SCHEMA = "carnot.archive_activation.v290_to_v291.v1"
ARTIFACT = "experiment_3122_archive_v290_activate_v291"
OUTPUT_REL_PATH = Path("results/experiment_3122_archive_v290_activate_v291.json")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3122_archive_v290_activate_v291.py"

MATRIX_V24_REL_PATH = Path("results/experiment_3120_cross_corpus_matrix_v24.json")
CAPSTONE_V290_REL_PATH = Path("results/experiment_3121_capstone_v290.json")
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
    ("matrix_v24", MATRIX_V24_REL_PATH),
    ("capstone_v290", CAPSTONE_V290_REL_PATH),
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
EXPECTED_MISSING_HEADLINE_MODELS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
]
EXPECTED_PRESENT_HEADLINE_MODELS = ["unsloth/gemma-4-26B-A4B-it-GGUF"]


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
    """Return a checksum so every source file can be audited later."""

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
    """REQ-REPORT-3122: synthesize the .290 archive and .291 handoff record."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    matrix = read_json_object(root_path / MATRIX_V24_REL_PATH)
    capstone = read_json_object(root_path / CAPSTONE_V290_REL_PATH)
    staged = read_yaml_mapping(root_path / STAGED_ROADMAP_REL_PATH)
    active = read_yaml_mapping(root_path / ACTIVE_ROADMAP_REL_PATH)
    source_artifacts = [
        _source_artifact(root_path, role, rel_path) for role, rel_path in SOURCE_PATHS
    ]
    roadmap_handoff = _roadmap_handoff(root_path, staged, active)
    prior_capstone_ready = capstone.get("capstone_ready") is True
    prior_paper_ready = capstone.get("paper_ready") is True
    prior_publication_blocker_count, blocker_source = _publication_blocker_count(
        capstone,
        matrix,
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
        "archive_v290_activate_v291_ready": ready,
        "prior_capstone_ready": prior_capstone_ready,
        "prior_paper_ready": prior_paper_ready,
        "prior_paper_ready_source_field_present": "paper_ready" in capstone,
        "prior_publication_blocker_count": prior_publication_blocker_count,
        "prior_publication_blocker_count_source": blocker_source,
        "status_summary_290": _status_summary_290(matrix, capstone),
        "carry_forward_blockers": _carry_forward_blockers(capstone, matrix),
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
    """Build and persist the Exp 3122 deliverable JSON."""

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
        reasons.append("roadmap milestone is not 2026.05.291")
    if not roadmap_handoff.get("milestone_doc_matches"):
        reasons.append(
            "roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md"
        )
    if not roadmap_handoff.get("non_empty_tasks"):
        reasons.append("roadmap has no tasks")
    if not vnext_doc_present:
        reasons.append("openspec/change-proposals/research-roadmap-vNEXT.md is missing")
    return reasons


def _status_summary_290(matrix: Mapping[str, Any], capstone: Mapping[str, Any]) -> JsonDict:
    count, count_source = _publication_blocker_count(capstone, matrix)
    return {
        "paper_ready": capstone.get("paper_ready") is True,
        "capstone_ready": capstone.get("capstone_ready") is True,
        "matrix_v24_ready": matrix.get("matrix_v24_ready") is True
        or _as_mapping(capstone.get("matrix_v24_summary")).get("matrix_v24_ready") is True,
        "publication_blocker_count": count,
        "publication_blocker_count_source": count_source,
        "next_top_gap": _next_top_gap(capstone),
        "model_cache_status": _model_cache_status(capstone, matrix),
        "verifier_gain_status": _status_value("verifier_gain_status", matrix, capstone),
        "formal_feedback_status": _status_value("formal_feedback_status", matrix, capstone),
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
    matrix_summary = _as_mapping(capstone.get("matrix_v24_summary"))
    summary_count = matrix_summary.get("publication_blocker_count")
    if not isinstance(summary_count, bool) and isinstance(summary_count, int):
        return summary_count, "capstone_matrix_v24_summary"
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


def _next_top_gap(capstone: Mapping[str, Any]) -> str:
    gaps = _as_list(capstone.get("remaining_top_gaps"))
    if gaps:
        return str(gaps[0])
    match = re.search(r"next_top_gap=([^;]+)", str(capstone.get("honest_verdict") or ""))
    return match.group(1) if match else ""


def _model_cache_status(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> JsonDict:
    rows = _dict_rows(
        capstone.get("headline_model_spec_gaps") or matrix.get("headline_model_spec_gaps")
    )
    first = rows[0] if rows else {}
    return {
        "present_model_ids": list(_as_list(first.get("present_model_ids"))),
        "missing_model_ids": list(_as_list(first.get("missing_model_ids"))),
        "headline_model_spec_gaps": rows,
    }


def _carry_forward_blockers(
    capstone: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> list[JsonDict]:
    status = _status_summary_290(matrix, capstone)
    model_cache = _as_mapping(status["model_cache_status"])
    repair_success_delta = _repair_success_delta(capstone, matrix)
    return [
        {
            "blocker_id": "publication_blockers_36",
            "description": "36 publication blockers",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "publication_blocker_count",
            "value": status["publication_blocker_count"],
            "expected_carry_forward_value": 36,
            "matches_expected": status["publication_blocker_count"] == 36,
        },
        {
            "blocker_id": "missing_headline_cache_coverage",
            "description": "missing Qwen3.6/Gemma-4-31B headline cache coverage",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "headline_model_spec_gaps",
            "value": {
                "missing_model_ids": list(_as_list(model_cache.get("missing_model_ids"))),
                "present_model_ids": list(_as_list(model_cache.get("present_model_ids"))),
            },
            "expected_carry_forward_value": {
                "missing_model_ids": EXPECTED_MISSING_HEADLINE_MODELS,
                "present_model_ids": EXPECTED_PRESENT_HEADLINE_MODELS,
            },
            "matches_expected": (
                list(_as_list(model_cache.get("missing_model_ids")))
                == EXPECTED_MISSING_HEADLINE_MODELS
                and list(_as_list(model_cache.get("present_model_ids")))
                == EXPECTED_PRESENT_HEADLINE_MODELS
            ),
        },
        {
            "blocker_id": "diagnostic_only_verifier_lift",
            "description": "diagnostic-only verifier lift",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "verifier_gain_status",
            "value": status["verifier_gain_status"],
            "expected_carry_forward_value": (
                "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags"
            ),
            "matches_expected": status["verifier_gain_status"]
            == "diagnostic_gain_recovered_but_headline_bounded_by_cache_and_prior_flags",
        },
        {
            "blocker_id": "zero_repair_delta",
            "description": "zero repair delta",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "repair_claim_status,repair_success_delta",
            "value": {
                "repair_claim_status": status["repair_claim_status"],
                "repair_success_delta": repair_success_delta,
            },
            "expected_carry_forward_value": {
                "repair_claim_status": "bounded_micro_panel_executed_zero_delta_no_promotion",
                "repair_success_delta": 0.0,
            },
            "matches_expected": (
                status["repair_claim_status"]
                == "bounded_micro_panel_executed_zero_delta_no_promotion"
                and repair_success_delta == 0.0
            ),
        },
        {
            "blocker_id": "fr11_controller_only_learning",
            "description": "FR-11 controller-only learning",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "fr11_self_learning_status",
            "value": status["fr11_self_learning_status"],
            "expected_carry_forward_value": (
                "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update"
            ),
            "matches_expected": status["fr11_self_learning_status"]
            == "bounded_controller_only_soundness_zero_completeness_zero_no_weight_update",
        },
        {
            "blocker_id": "ebt_arm_projection_only",
            "description": "EBT/ARM projection-only",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "ebt_arm_status",
            "value": status["ebt_arm_status"],
            "expected_carry_forward_value": (
                "projection_only_sidecar_correlation_no_live_model_integration"
            ),
            "matches_expected": status["ebt_arm_status"]
            == "projection_only_sidecar_correlation_no_live_model_integration",
        },
        {
            "blocker_id": "cpu_only_clut",
            "description": "CPU-only cLUT",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "sampler_hardware_status",
            "value": status["sampler_hardware_status"],
            "expected_carry_forward_value": "bounded_clut_cpu_only_no_hardware_speedup",
            "matches_expected": (
                status["sampler_hardware_status"] == "bounded_clut_cpu_only_no_hardware_speedup"
            ),
        },
        {
            "blocker_id": "missing_operator_visible_hardware_evidence",
            "description": "missing operator-visible hardware evidence",
            "source": CAPSTONE_V290_REL_PATH.as_posix(),
            "source_field": "gatemate_status,ssqa_status",
            "value": {
                "gatemate_status": status["gatemate_status"],
                "ssqa_status": status["ssqa_status"],
            },
            "expected_carry_forward_value": {
                "gatemate_status": "blocked_operator_evidence_incomplete_no_hardware_run",
                "ssqa_status": "gated_skipped_host_visible_readback_missing",
            },
            "matches_expected": {
                "gatemate_status": status["gatemate_status"],
                "ssqa_status": status["ssqa_status"],
            }
            == {
                "gatemate_status": "blocked_operator_evidence_incomplete_no_hardware_run",
                "ssqa_status": "gated_skipped_host_visible_readback_missing",
            },
        },
    ]


def _repair_success_delta(capstone: Mapping[str, Any], matrix: Mapping[str, Any]) -> float:
    direct = capstone.get("repair_success_delta")
    if isinstance(direct, int | float) and not isinstance(direct, bool):
        return float(direct)
    status = _as_mapping(matrix.get("verifier_repair_status"))
    value = status.get("repair_success_delta")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if artifact["archive_v290_activate_v291_ready"] is True:
        roadmap = _as_mapping(artifact.get("roadmap_handoff"))
        return (
            "complete: archive_v290_activate_v291_ready=true; "
            f"prior_capstone_ready={str(artifact['prior_capstone_ready']).lower()}; "
            f"prior_paper_ready={str(artifact['prior_paper_ready']).lower()}; "
            f"prior_publication_blocker_count={artifact['prior_publication_blocker_count']}; "
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


def _int_or(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
