"""Generate the Exp 3012 archive artifact for milestone 2026.05.282.

Spec refs: REQ-REPORT-3012, SCENARIO-REPORT-3012.

This module performs the narrow milestone-boundary bookkeeping from .282 to
.283. It reads the terminal .282 capstone, re-reads the local .282 artifacts
that the capstone references, checks the historical archive ledger, confirms
the .283 roadmap state, and writes the JSON acceptance artifact. It does not
rerun research, edit the conductor, update ops docs, call an LLM, or push.
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
EXPERIMENT_ID = "exp3012-archive-v282-activate-v283"
ARCHIVED_MILESTONE = "2026.05.282"
ACTIVATED_MILESTONE = "2026.05.283"
RUN_DATE = "20260524"
COMPLETED = "2026-05-24"
MILESTONE_TITLE = "Claim Repair + Metamorphic Validation + Attractor Memory + GateMate IO"
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
CAPSTONE_SOURCE = "results/experiment_3011_capstone_v282.json"
MATRIX_SOURCE = "results/experiment_3010_cross_corpus_matrix_v16.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_3012_archive_v282_activate_v283.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: archive_ready=true; archived_milestone=2026.05.282; "
    "activated_milestone=2026.05.283; status_updates_written=false"
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
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3012_archive_v282.py",
    "COVERAGE_FILE=/tmp/carnot_exp3012.coverage .venv/bin/coverage run --branch "
    "--include='*/milestone_282_archive_283_activation.py' -m pytest -o addopts='' "
    "tests/python/test_experiment_3012_archive_v282.py -q",
    "COVERAGE_FILE=/tmp/carnot_exp3012.coverage .venv/bin/coverage report "
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
    "adversarial_flags_carried_forward",
    "inference_substrate",
    "validation_commands",
    "honest_verdict",
}

FIELD_PRINCIPLES = {
    "archive_ready": "True only after the completed .282 archive row is explicit and countable.",
    "archived_milestone": "Unambiguous identifier for the completed milestone being archived.",
    "activated_milestone": "Unambiguous identifier for the next milestone confirmed from roadmap state.",
    "research_complete_updated": "True when completed .282 experiments are discoverable in the archive ledger.",
    "status_updates_written": "Read-only check for ops docs; false when conductor reconciliation is deferred.",
    "n_tasks_archived": "Count of completed .282 task entries in research-complete.yaml.",
    "blocked_or_flagged_rows_carried_forward": "Unresolved .282 rows carried forward without hiding flags.",
    "adversarial_flags_carried_forward": "Audit flags copied forward so false-positive and true-positive rows stay visible.",
    "inference_substrate": "Aggregation must not masquerade as live inference.",
    "validation_commands": "The exact validation commands this task runs before closeout.",
    "honest_verdict": "Terminal-prefix verdict suitable for conductor consumption.",
}

MILESTONE_TASKS = [
    {
        "id": "exp3000-archive-v281-activate-v282",
        "title": "Archive .281 and activate .282",
        "deliverable": "results/experiment_3000_archive_v281_activate_v282.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3001-sota-gguf-cache-carry-forward-checksum-refresh",
        "title": "SOTA GGUF cache carry-forward and checksum refresh",
        "deliverable": "results/experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3002-metamorphic-repair-oracle-audit",
        "title": "Metamorphic repair-oracle audit for hard-set repair",
        "deliverable": "results/experiment_3002_metamorphic_repair_oracle_audit_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3003-gated-sota-repair-metamorphic-false-accept-rerun",
        "title": "Gated SOTA repair rerun with metamorphic false-accept checks",
        "deliverable": "results/experiment_3003_gated_sota_repair_metamorphic_false_accept_rerun_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3004-aquaforte-beaver-live-retry-provenance-v2",
        "title": "AquaForte/BEAVER live retry provenance v2",
        "deliverable": "results/experiment_3004_aquaforte_beaver_live_retry_provenance_v2.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3005-solver-to-validator-tree-expansion",
        "title": "Solver-to-validator tree expansion",
        "deliverable": "results/experiment_3005_solver_to_validator_tree_expansion_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3006-eqr-fixed-point-energy-diagnostic",
        "title": "EqR fixed-point energy diagnostic",
        "deliverable": "results/experiment_3006_eqr_fixed_point_energy_diagnostic_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3007-fr11-attractor-trace-memory-stability",
        "title": "FR-11 attractor trace-memory stability",
        "deliverable": "results/experiment_3007_fr11_attractor_trace_memory_stability_v1.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3008-gatemate-host-visible-io-transport-v2",
        "title": "GateMate host-visible IO transport v2",
        "deliverable": "results/experiment_3008_gatemate_host_visible_io_transport_v2.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3009-ssqa-dual-bram-rtl-pnr-resource-report-v2",
        "title": "SSQA dual-BRAM RTL PnR resource report v2",
        "deliverable": "results/experiment_3009_ssqa_dual_bram_rtl_pnr_resource_report_v2.json",
        "result": "OK (conductor)",
    },
    {
        "id": "exp3010-cross-corpus-matrix-v16",
        "title": "Cross-corpus matrix v16",
        "deliverable": MATRIX_SOURCE,
        "result": "OK (conductor)",
    },
    {
        "id": "exp3011-milestone-282-capstone",
        "title": "Milestone .282 capstone",
        "deliverable": CAPSTONE_SOURCE,
        "result": "OK (conductor)",
    },
]

CLASSIFICATION_KEYS = (
    "blocked",
    "clean",
    "flagged",
    "gated-skipped",
    "missing",
    "pilot-only",
    "projection-only",
)
UNRESOLVED_STATUSES = {"blocked", "flagged", "gated-skipped", "missing"}
MILESTONE_282_EXPERIMENT_IDS = {f"exp{number}" for number in range(3000, 3011)}


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


def _matrix_rows(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        rows = value.get("rows")
    else:
        rows = value
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _classification_counts_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {key: 0 for key in CLASSIFICATION_KEYS}
    for row in rows:
        status = str(row.get("classification") or row.get("status") or "")
        if status in counts:
            counts[status] += 1
    return counts


def _blocked_or_flagged_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    carried: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("classification") or row.get("status") or "")
        if status not in UNRESOLVED_STATUSES:
            continue
        carried.append(
            {
                "row_id": str(row.get("row_id") or ""),
                "experiment_id": str(row.get("experiment_id") or row.get("source_experiment_id") or ""),
                "classification": status,
                "honest_verdict": str(row.get("honest_verdict") or row.get("source_honest_verdict") or ""),
                "upstream_flags": _as_str_list(row.get("upstream_flags")),
            }
        )
    return carried


def _adversarial_flag_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    carried: list[dict[str, Any]] = []
    for row in rows:
        flags = _as_str_list(row.get("upstream_flags"))
        if not flags:
            continue
        carried.append(
            {
                "row_id": str(row.get("row_id") or ""),
                "experiment_id": str(row.get("experiment_id") or row.get("source_experiment_id") or ""),
                "classification": str(row.get("classification") or row.get("status") or ""),
                "honest_verdict": str(row.get("honest_verdict") or row.get("source_honest_verdict") or ""),
                "upstream_flags": flags,
            }
        )
    return carried


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
    """REQ-REPORT-3012: detect an existing completed .282 archive block."""

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
    finding = str(capstone.get("honest_verdict") or "Exp 3011 capstone archived.")
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


def _source_artifact_entries(capstone: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = capstone.get("source_artifacts_read")
    if not isinstance(value, list):
        return []
    entries = [entry for entry in value if isinstance(entry, Mapping)]
    return [
        entry
        for entry in entries
        if str(entry.get("experiment_id") or "") in MILESTONE_282_EXPERIMENT_IDS
    ]


def _load_all_referenced_artifacts(root: Path, capstone: Mapping[str, Any]) -> dict[str, Any]:
    read_paths: list[str] = []
    missing_paths: list[str] = []
    checked: list[dict[str, Any]] = []
    for entry in _source_artifact_entries(capstone):
        rel_path = str(entry.get("path") or "")
        if not rel_path:
            continue
        payload = _read_json_mapping(root / rel_path)
        present = bool(payload)
        if present:
            read_paths.append(rel_path)
        else:
            missing_paths.append(rel_path)
        checked.append(
            {
                "experiment_id": str(entry.get("experiment_id") or ""),
                "path": rel_path,
                "present": (root / rel_path).exists(),
                "readable_json_object": present,
            }
        )
    return {
        "n_milestone_282_source_artifacts_referenced": len(checked),
        "n_milestone_282_source_artifacts_read": len(read_paths),
        "milestone_282_source_artifacts_read": read_paths,
        "missing_referenced_artifacts": missing_paths,
        "referenced_source_artifact_checks": checked,
    }


def _capstone_summary(capstone: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capstone_loaded": bool(capstone),
        "capstone_milestone": str(capstone.get("milestone") or ""),
        "capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "capstone_ready_from_capstone": bool(capstone.get("capstone_ready", False)),
        "paper_ready_from_capstone": bool(capstone.get("paper_ready", False)),
        "clean_task_rows_from_capstone": _as_str_list(capstone.get("clean_task_rows")),
        "flagged_task_rows_from_capstone": _as_str_list(capstone.get("flagged_task_rows")),
        "blocked_task_rows_from_capstone": _as_str_list(capstone.get("blocked_task_rows")),
        "missing_task_rows_from_capstone": _as_str_list(capstone.get("missing_task_rows")),
        "gated_skipped_task_rows_from_capstone": _as_str_list(
            capstone.get("gated_skipped_task_rows")
        ),
        "pilot_only_task_rows_from_capstone": _as_str_list(capstone.get("pilot_only_task_rows")),
        "projection_only_task_rows_from_capstone": _as_str_list(
            capstone.get("projection_only_task_rows")
        ),
        "artifact_classification_counts_from_capstone": _as_int_mapping(
            capstone.get("matrix_status_counts")
        ),
        "task_classification_counts_from_capstone": _as_int_mapping(
            capstone.get("task_classification_counts")
        ),
        "next_milestone_recommendation_from_capstone": str(
            capstone.get("next_milestone_recommendation") or ""
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
        blocked_reasons.append("research-complete.yaml does not archive 2026.05.282")
    if not roadmap["milestone_matches"]:
        blocked_reasons.append("roadmap milestone is not 2026.05.283")
    if not roadmap["non_empty_tasks"]:
        blocked_reasons.append("roadmap has no tasks for 2026.05.283")

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
            "research-roadmap-next.yaml",
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
    """SCENARIO-REPORT-3012: write the .282 archive and .283 activation JSON."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_before = _read_text(root_path / "research-roadmap.yaml")
    next_roadmap_before = _read_text(root_path / "research-roadmap-next.yaml")
    complete_before = _read_text(root_path / "research-complete.yaml")
    status_before = _read_text(root_path / "ops" / "status.md")
    changelog_before = _read_text(root_path / "ops" / "changelog.md")
    traceability_before = _read_text(root_path / "_bmad" / "traceability.md")
    capstone = _read_json_mapping(root_path / CAPSTONE_SOURCE)
    matrix = _read_json_mapping(root_path / MATRIX_SOURCE) if capstone else {}

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
    rows = _matrix_rows(matrix)
    adversarial_flags = _adversarial_flag_rows(rows)
    read_summary = _load_all_referenced_artifacts(root_path, capstone)
    status_updates_written = _status_updates_written(root_path)
    honest_verdict, blocked_reasons = _honest_verdict(
        archive_ready=archive_ready,
        research_complete_updated=research_complete_updated,
        roadmap=roadmap,
        capstone_loaded=bool(capstone),
    )
    duration_s = round(clock() - start_s, 6)
    roadmap_after = _read_text(root_path / "research-roadmap.yaml")
    next_roadmap_after = _read_text(root_path / "research-roadmap-next.yaml")
    status_after = _read_text(root_path / "ops" / "status.md")
    changelog_after = _read_text(root_path / "ops" / "changelog.md")
    traceability_after = _read_text(root_path / "_bmad" / "traceability.md")

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
            "research_roadmap_next_yaml_sha256_before": _sha256_text(next_roadmap_before),
            "research_roadmap_next_yaml_sha256_after": _sha256_text(next_roadmap_after),
            "research_roadmap_next_yaml_modified": next_roadmap_before != next_roadmap_after,
        },
        "ops_doc_verification": {
            "ops_status_modified": status_before != status_after,
            "ops_changelog_modified": changelog_before != changelog_after,
            "bmad_traceability_modified": traceability_before != traceability_after,
        },
        "artifact_classification_counts_from_matrix": _classification_counts_from_rows(rows),
        "blocked_or_flagged_rows_carried_forward": _blocked_or_flagged_rows(rows),
        "adversarial_flags_carried_forward": adversarial_flags,
        "adversarial_flag_count": len(adversarial_flags),
        "blocked_reasons": blocked_reasons,
        "notes": [
            "research-roadmap-next.yaml was checked first when present.",
            "research-roadmap.yaml was read only when research-roadmap-next.yaml was absent.",
            "research-roadmap-next.yaml is absent in the current checkout because .283 is active.",
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
