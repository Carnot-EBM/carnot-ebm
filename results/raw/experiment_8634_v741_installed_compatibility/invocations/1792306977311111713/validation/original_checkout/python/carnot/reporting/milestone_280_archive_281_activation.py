"""Generate the Exp 2988 archive artifact for milestone 2026.05.280.

Spec refs: REQ-REPORT-2988, SCENARIO-REPORT-2988.

This module performs the narrow milestone-boundary bookkeeping from .280 to
.281. It reads the terminal .280 capstone and its referenced result artifacts,
checks the historical archive ledger, confirms the .281 roadmap state, and
writes the JSON acceptance artifact. It does not rerun research, edit the
conductor, update ops docs, or push commits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2988-archive-v280-activate-v281"
ARCHIVED_MILESTONE = "2026.05.280"
ACTIVATED_MILESTONE = "2026.05.281"
RUN_DATE = "20260524"
COMPLETED = "2026-05-24"
MILESTONE_TITLE = "Intent-Preserving Repair + Solver Feedback + Readback-Grounded Self-Learning"
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
CAPSTONE_SOURCE = "results/experiment_2987_capstone_v280.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2988_archive_v280_activate_v281.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: archive_ready=true; archived_milestone=2026.05.280; "
    "activated_milestone=2026.05.281; status_updates_written=false"
)

DEFAULT_VALIDATION_COMMANDS = [
    ".venv/bin/python - <<'PY'\n"
    "from pathlib import Path\n"
    "import yaml\n"
    "for name in ('research-roadmap.yaml', 'research-complete.yaml'):\n"
    "    yaml.safe_load(Path(name).read_text(encoding='utf-8'))\n"
    "print('yaml parse ok')\n"
    "PY",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_2988_archive_v280.py",
    "COVERAGE_FILE=/tmp/carnot_exp2988.coverage .venv/bin/coverage run --branch "
    "--include='*/milestone_280_archive_281_activation.py' -m pytest -o addopts='' "
    "tests/python/test_experiment_2988_archive_v280.py -q",
    "COVERAGE_FILE=/tmp/carnot_exp2988.coverage .venv/bin/coverage report "
    "--fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
]

REQUIRED_ARTIFACT_FIELDS = {
    "archive_ready",
    "archived_milestone",
    "activated_milestone",
    "research_complete_updated",
    "status_updates_written",
    "n_tasks_archived",
    "blocked_or_flagged_rows_carried_forward",
    "validation_commands",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "archive_ready": "True only after the completed .280 archive row is explicit and countable.",
    "archived_milestone": "Unambiguous identifier for the completed milestone being archived.",
    "activated_milestone": "Unambiguous identifier for the next milestone confirmed from roadmap state.",
    "research_complete_updated": "True when completed .280 experiments are discoverable in the archive ledger.",
    "status_updates_written": "Read-only check for ops docs; false when conductor reconciliation is deferred.",
    "n_tasks_archived": "Count of completed .280 task entries in research-complete.yaml.",
    "blocked_or_flagged_rows_carried_forward": "Unresolved .280 rows carried forward without hiding flags.",
    "validation_commands": "The exact validation commands this task runs before closeout.",
    "honest_verdict": "Terminal-prefix verdict suitable for conductor consumption.",
}

MILESTONE_TASKS = [
    {
        "id": "exp2975",
        "title": "Archive .279 + Activate .280",
        "deliverable": "results/experiment_2975_archive_v279_activate_v280.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2976",
        "title": "DCCD AdapTrack + TraceCoder Repair Protocol v1",
        "deliverable": "results/experiment_2976_dccd_adaptrack_tracecoder_protocol_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2977",
        "title": "Gated SOTA Intent-Preserving Code Repair Rerun v1",
        "deliverable": "results/experiment_2977_sota_intent_preserving_code_repair_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2978",
        "title": "First-Step + Semantic-Energy Repair Telemetry Panel v1",
        "deliverable": "results/experiment_2978_first_step_semantic_energy_repair_telemetry_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2979",
        "title": "Solver Feedback Schema + MCS/MUS Frontier Upgrade v1",
        "deliverable": "results/experiment_2979_solver_feedback_mcs_frontier_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2980",
        "title": "Gated SOTA Solver Formalization with Feedback v2",
        "deliverable": "results/experiment_2980_sota_solver_formalization_feedback_v2.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2981",
        "title": "Interwhen Partial Monitor Promotion v2",
        "deliverable": "results/experiment_2981_interwhen_partial_monitor_promotion_v2.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2982",
        "title": "FR-11 Independent Metric Continuous Self-Learning Gate v4",
        "deliverable": "results/experiment_2982_fr11_independent_metric_utility_gate_v4.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2983",
        "title": "Trace-to-Skill Repair Memory Pilot v1",
        "deliverable": "results/experiment_2983_trace_to_skill_repair_memory_pilot_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2984",
        "title": "GateMate Readback + Smoke Vector Harness v4",
        "deliverable": "results/experiment_2984_gatemate_readback_smoke_vector_v4.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2985",
        "title": "SSQA Dual-BRAM Hardware Projection + Register-Map Plan v1",
        "deliverable": "results/experiment_2985_ssqa_dual_bram_register_map_plan_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2986",
        "title": "Cross-Corpus Matrix v14 + Claim Boundary Audit",
        "deliverable": "results/experiment_2986_cross_corpus_matrix_v14.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp2987",
        "title": "Milestone .280 Capstone + Paper-Readiness Decision",
        "deliverable": CAPSTONE_SOURCE,
        "result": "OK (conductor)",
    },
]

CLASSIFICATION_KEYS = ("blocked", "clean", "flagged", "missing", "pilot-only", "projection-only")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text.strip():
        return {}
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _as_str_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _as_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
    }


def _audit_rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _classification_counts_from_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in CLASSIFICATION_KEYS}
    for row in rows:
        classification = str(row.get("classification") or row.get("status") or "")
        if classification in counts:
            counts[classification] += 1
    return counts


def _blocked_or_flagged_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    carried: list[dict[str, Any]] = []
    for row in rows:
        classification = str(row.get("classification") or row.get("status") or "")
        if classification not in {"blocked", "flagged"}:
            continue
        carried.append(
            {
                "experiment_id": str(row.get("experiment_id") or ""),
                "path": str(row.get("path") or ""),
                "classification": classification,
                "honest_verdict": str(row.get("honest_verdict") or ""),
                "upstream_flags": _as_str_list(row.get("upstream_flags")),
                "prior_failure_outcome": str(row.get("prior_failure_outcome") or ""),
            }
        )
    return carried


def _load_all_referenced_artifacts(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    read_paths: list[str] = []
    missing_paths: list[str] = []
    for row in rows:
        rel_path = str(row.get("path") or "")
        if not rel_path:
            continue
        payload = _read_json_mapping(root / rel_path)
        if payload:
            read_paths.append(rel_path)
        else:
            missing_paths.append(rel_path)
    return {
        "n_referenced_artifacts": len([row for row in rows if row.get("path")]),
        "n_source_artifacts_read": len(read_paths),
        "source_artifacts_read": read_paths,
        "missing_referenced_artifacts": missing_paths,
    }


def _archive_completed_block_count(text: str) -> int:
    lines = text.splitlines()
    pattern = re.compile(rf"^- id:\s*['\"]?{re.escape(ARCHIVED_MILESTONE)}['\"]?\s*$")
    count = 0
    for index, line in enumerate(lines):
        if not pattern.match(line):
            continue
        end = len(lines)
        for next_index in range(index + 1, len(lines)):
            if lines[next_index].startswith("- id: "):
                end = next_index
                break
        if any(row.lstrip().startswith("completed:") for row in lines[index:end]):
            count += 1
    return count


def _archive_completed_block_present(text: str) -> bool:
    """REQ-REPORT-2988: detect an existing completed .280 archive block."""

    return _archive_completed_block_count(text) > 0


def _archive_task_count(text: str) -> int:
    payload = yaml.safe_load(text) if text.strip() else {}
    rows = payload.get("milestones", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return 0
    for row in rows:
        if (
            isinstance(row, dict)
            and str(row.get("id")) == ARCHIVED_MILESTONE
            and "completed" in row
        ):
            tasks = row.get("tasks")
            return len(tasks) if isinstance(tasks, list) else 0
    return 0


def _minimal_archive_entry(capstone: Mapping[str, Any]) -> dict[str, Any]:
    finding = str(capstone.get("honest_verdict") or "Exp 2987 capstone archived.")
    return {
        "id": ARCHIVED_MILESTONE,
        "title": MILESTONE_TITLE,
        "doc": MILESTONE_DOC,
        "completed": COMPLETED,
        "finding": finding,
        "tasks": MILESTONE_TASKS,
    }


def _append_archive_entry(path: Path, capstone: Mapping[str, Any]) -> None:
    original = _read_text(path)
    entry = yaml.safe_dump(
        [_minimal_archive_entry(capstone)],
        sort_keys=False,
        allow_unicode=False,
        width=120,
    ).rstrip()
    prefix = original if original.strip() else "milestones:\n"
    separator = "" if prefix.endswith("\n") else "\n"
    path.write_text(prefix + separator + entry + "\n", encoding="utf-8")


def _roadmap_metadata(root: Path) -> dict[str, Any]:
    next_path = root / "research-roadmap-next.yaml"
    if next_path.exists():
        path = next_path
        fallback = False
    else:
        path = root / "research-roadmap.yaml"
        fallback = True

    payload = _load_yaml_mapping(path)
    milestone = str(payload.get("milestone", ""))
    milestone_doc = str(payload.get("milestone_doc", ""))
    tasks = payload.get("tasks")
    non_empty_tasks = isinstance(tasks, list) and len(tasks) > 0
    return {
        "roadmap_source": path.name,
        "roadmap_exists": path.exists(),
        "research_roadmap_next_exists": next_path.exists(),
        "used_active_roadmap_fallback": fallback,
        "observed_milestone": milestone,
        "expected_milestone": ACTIVATED_MILESTONE,
        "milestone_matches": milestone == ACTIVATED_MILESTONE,
        "observed_milestone_doc": milestone_doc,
        "expected_milestone_doc": MILESTONE_DOC,
        "milestone_doc_matches": milestone_doc == MILESTONE_DOC,
        "n_tasks": len(tasks) if isinstance(tasks, list) else 0,
        "non_empty_tasks": non_empty_tasks,
    }


def _status_updates_written(root: Path) -> bool:
    status = _read_text(root / "ops" / "status.md")
    changelog = _read_text(root / "ops" / "changelog.md")
    return (
        ARCHIVED_MILESTONE in status
        and ACTIVATED_MILESTONE in status
        and ARCHIVED_MILESTONE in changelog
        and ACTIVATED_MILESTONE in changelog
    )


def _capstone_summary(capstone: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capstone_loaded": bool(capstone),
        "capstone_milestone": str(capstone.get("milestone") or ""),
        "capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "paper_ready_from_capstone": bool(capstone.get("paper_ready", False)),
        "headline_outcome_from_capstone": str(capstone.get("headline_outcome") or ""),
        "clean_artifacts_from_capstone": _as_str_list(capstone.get("clean_artifacts")),
        "flagged_artifacts_from_capstone": _as_str_list(capstone.get("flagged_artifacts")),
        "blocked_artifacts_from_capstone": _as_str_list(capstone.get("blocked_artifacts")),
        "missing_artifacts_from_capstone": _as_str_list(capstone.get("missing_artifacts")),
        "pilot_only_artifacts_from_capstone": _as_str_list(capstone.get("pilot_only_artifacts")),
        "projection_only_artifacts_from_capstone": _as_str_list(
            capstone.get("projection_only_artifacts")
        ),
        "artifact_classification_counts_from_capstone": _as_int_mapping(
            capstone.get("artifact_classification_counts")
        ),
        "gaps_remaining_from_capstone": _as_str_list(capstone.get("gaps_remaining")),
        "next_milestone_recommendations_from_capstone": _as_str_list(
            capstone.get("next_milestone_recommendations")
        ),
    }


def _honest_verdict(
    *,
    archive_ready: bool,
    research_complete_updated: bool,
    roadmap: Mapping[str, Any],
    capstone_loaded: bool,
) -> tuple[str, list[str]]:
    blocked_reasons: list[str] = []
    if not capstone_loaded:
        blocked_reasons.append("capstone source missing or invalid")
    if not archive_ready or not research_complete_updated:
        blocked_reasons.append("research-complete.yaml does not archive 2026.05.280")
    if not roadmap["milestone_matches"]:
        blocked_reasons.append("roadmap milestone is not 2026.05.281")
    if not roadmap["non_empty_tasks"]:
        blocked_reasons.append("roadmap has no tasks for 2026.05.281")

    if blocked_reasons:
        return "blocked: " + "; ".join(blocked_reasons), blocked_reasons
    return COMPLETE_VERDICT, blocked_reasons


def _base_artifact(duration_s: float, validation_commands: Sequence[str]) -> dict[str, Any]:
    return {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "capstone_source": CAPSTONE_SOURCE,
        "run_date": RUN_DATE,
        "duration_s": duration_s,
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "validation_commands": list(validation_commands),
        "pushed": False,
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "files_not_modified": [
            "research-roadmap.yaml",
            "scripts/research_conductor.py",
            "ops/changelog.md",
            "ops/status.md",
            "_bmad/traceability.md",
        ],
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    clock: Callable[[], float] = time.perf_counter,
    validation_commands: Sequence[str] = DEFAULT_VALIDATION_COMMANDS,
) -> dict[str, Any]:
    """SCENARIO-REPORT-2988: write the .280 archive and .281 activation JSON."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_before = _read_text(root_path / "research-roadmap.yaml")
    complete_before = _read_text(root_path / "research-complete.yaml")
    status_before = _read_text(root_path / "ops" / "status.md")
    changelog_before = _read_text(root_path / "ops" / "changelog.md")
    capstone = _read_json_mapping(root_path / CAPSTONE_SOURCE)

    archive_already_present = _archive_completed_block_present(complete_before)
    archive_appended_this_run = False
    if capstone and not archive_already_present:
        _append_archive_entry(root_path / "research-complete.yaml", capstone)
        archive_appended_this_run = True

    complete_after = _read_text(root_path / "research-complete.yaml")
    archive_count = _archive_completed_block_count(complete_after)
    n_tasks_archived = _archive_task_count(complete_after)
    research_complete_updated = archive_count == 1 and n_tasks_archived >= len(MILESTONE_TASKS)
    archive_ready = research_complete_updated
    roadmap = _roadmap_metadata(root_path)
    audit_rows = _audit_rows(capstone.get("artifact_audit") if capstone else [])
    read_summary = _load_all_referenced_artifacts(root_path, audit_rows)
    status_updates_written = _status_updates_written(root_path)
    honest_verdict, blocked_reasons = _honest_verdict(
        archive_ready=archive_ready,
        research_complete_updated=research_complete_updated,
        roadmap=roadmap,
        capstone_loaded=bool(capstone),
    )
    duration_s = round(clock() - start_s, 6)
    roadmap_after = _read_text(root_path / "research-roadmap.yaml")
    status_after = _read_text(root_path / "ops" / "status.md")
    changelog_after = _read_text(root_path / "ops" / "changelog.md")

    artifact = {
        **_base_artifact(duration_s, validation_commands),
        "honest_verdict": honest_verdict,
        "archive_ready": archive_ready,
        "research_complete_updated": research_complete_updated,
        "status_updates_written": status_updates_written,
        "n_tasks_archived": n_tasks_archived if research_complete_updated else 0,
        "archive_already_present": archive_already_present,
        "archive_appended_this_run": archive_appended_this_run,
        "archive_completed_block_count": archive_count,
        "activation": roadmap,
        "archive": {
            "research_complete_path": "research-complete.yaml",
            "ready_after_run": archive_ready,
        },
        "roadmap_verification": {
            "research_roadmap_yaml_sha256_before": _sha256_text(roadmap_before),
            "research_roadmap_yaml_sha256_after": _sha256_text(roadmap_after),
            "research_roadmap_yaml_modified": roadmap_before != roadmap_after,
        },
        "ops_doc_verification": {
            "ops_status_modified": status_before != status_after,
            "ops_changelog_modified": changelog_before != changelog_after,
        },
        "artifact_classification_counts_from_audit": _classification_counts_from_audit(
            audit_rows
        ),
        "blocked_or_flagged_rows_carried_forward": _blocked_or_flagged_rows(audit_rows),
        "blocked_reasons": blocked_reasons,
        "notes": [
            "research-roadmap-next.yaml was checked first when present.",
            "research-roadmap.yaml was read only when research-roadmap-next.yaml was absent.",
            "research-roadmap-next.yaml is absent in the current checkout because .281 is active.",
            "Ops status/changelog/traceability docs were left for the conductor reconciler.",
            "scripts/research_conductor.py was not modified.",
            "No push was performed.",
        ],
        **_capstone_summary(capstone),
        **read_summary,
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
