"""Generate the Exp 2847 archive artifact for milestone 2026.05.269.

Spec: REQ-REPORT-2847, SCENARIO-REPORT-2847.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2847-archive-v269-activate-v270"
ARCHIVED_MILESTONE = "2026.05.269"
ACTIVATED_MILESTONE = "2026.05.270"
RUN_DATE = "20260522"
COMPLETED = "2026-05-22"
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
CAPSTONE_SOURCE = "results/experiment_2846_capstone_v269.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2847_archive_v269_activate_v270.json")

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "archived_milestone",
    "activated_milestone",
    "archive_already_present",
    "capstone_source",
    "paper_ready_from_capstone",
    "blocked_artifacts_from_capstone",
    "missing_artifacts_from_capstone",
    "top_3_next_actions",
    "run_date",
    "duration_s",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


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


def _archive_completed_block_present(text: str) -> bool:
    """Return true when any `.269` archive row records a completion date.

    WHY: `research-complete.yaml` is large and can contain historical duplicate
    rows. A narrow block scan answers the bookkeeping question without
    round-tripping the full file through a YAML formatter.
    """

    lines = text.splitlines()
    pattern = re.compile(rf"^- id:\s*['\"]?{re.escape(ARCHIVED_MILESTONE)}['\"]?\s*$")
    for index, line in enumerate(lines):
        if not pattern.match(line):
            continue
        end = len(lines)
        for next_index in range(index + 1, len(lines)):
            if lines[next_index].startswith("- id: "):
                end = next_index
                break
        if any(row.lstrip().startswith("completed:") for row in lines[index:end]):
            return True
    return False


def _minimal_archive_entry(capstone: Mapping[str, Any]) -> dict[str, Any]:
    finding = str(capstone.get("honest_verdict") or "Exp 2846 capstone archived.")
    return {
        "id": ARCHIVED_MILESTONE,
        "title": "SOTA Runtime Gate + Multi-Corpus Evidence + LoopUS Self-Learning",
        "doc": MILESTONE_DOC,
        "completed": COMPLETED,
        "finding": finding,
        "tasks": [
            {
                "id": "exp2846-capstone-v269",
                "title": "Terminal Milestone Summary v269",
                "deliverable": CAPSTONE_SOURCE,
                "result": "OK (capstone)",
            }
        ],
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
    return {
        "roadmap_source": path.name,
        "roadmap_exists": path.exists(),
        "used_active_roadmap_fallback": fallback,
        "observed_milestone": milestone,
        "expected_milestone": ACTIVATED_MILESTONE,
        "milestone_matches": milestone == ACTIVATED_MILESTONE,
        "observed_milestone_doc": milestone_doc,
        "expected_milestone_doc": MILESTONE_DOC,
        "milestone_doc_matches": milestone_doc == MILESTONE_DOC,
    }


def _capstone_summary(capstone: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capstone_loaded": bool(capstone),
        "sota_runtime_ready_from_capstone": bool(capstone.get("sota_runtime_ready", False)),
        "paper_ready_from_capstone": bool(capstone.get("paper_ready", False)),
        "blocked_artifacts_from_capstone": _as_str_list(capstone.get("blocked_artifacts")),
        "missing_artifacts_from_capstone": _as_str_list(capstone.get("missing_artifacts")),
        "top_3_next_actions": _as_str_list(capstone.get("top_3_next_actions"))[:3],
    }


def _honest_verdict(
    *,
    archive_ready: bool,
    roadmap: Mapping[str, Any],
    capstone: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blocked_reasons: list[str] = []
    if not archive_ready:
        blocked_reasons.append("research-complete.yaml does not archive 2026.05.269")
    if not roadmap["milestone_matches"]:
        blocked_reasons.append("roadmap milestone is not 2026.05.270")
    if not roadmap["milestone_doc_matches"]:
        blocked_reasons.append("roadmap milestone_doc is not openspec/change-proposals/research-roadmap-vNEXT.md")
    if not capstone:
        blocked_reasons.append("capstone source missing or invalid")

    if blocked_reasons:
        return "blocked: " + "; ".join(blocked_reasons), blocked_reasons

    return (
        "complete: "
        f"archive_ready={str(archive_ready).lower()}; "
        f"archived_milestone={ARCHIVED_MILESTONE}; "
        f"activated_milestone={ACTIVATED_MILESTONE}; "
        f"paper_ready={str(bool(summary['paper_ready_from_capstone'])).lower()}; "
        f"blocked_artifacts={len(summary['blocked_artifacts_from_capstone'])}; "
        f"missing_artifacts={len(summary['missing_artifacts_from_capstone'])}",
        blocked_reasons,
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-2847: write the .269 archive and .270 activation JSON."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    complete_path = root_path / "research-complete.yaml"
    complete_before = _read_text(complete_path)
    archive_already_present = _archive_completed_block_present(complete_before)
    capstone = _read_json_mapping(root_path / CAPSTONE_SOURCE)

    archive_appended_this_run = False
    if not archive_already_present and capstone:
        _append_archive_entry(complete_path, capstone)
        archive_appended_this_run = True

    archive_ready = _archive_completed_block_present(_read_text(complete_path))
    roadmap = _roadmap_metadata(root_path)
    summary = _capstone_summary(capstone)
    honest_verdict, blocked_reasons = _honest_verdict(
        archive_ready=archive_ready,
        roadmap=roadmap,
        capstone=capstone,
        summary=summary,
    )
    duration_s = round(clock() - start_s, 6)

    artifact = {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": honest_verdict,
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "archive_already_present": archive_already_present,
        "archive_appended_this_run": archive_appended_this_run,
        "capstone_source": CAPSTONE_SOURCE,
        "run_date": RUN_DATE,
        "duration_s": duration_s,
        "activation": roadmap,
        "archive": {
            "research_complete_path": "research-complete.yaml",
            "ready_after_run": archive_ready,
        },
        "blocked_reasons": blocked_reasons,
        "notes": [
            "research-roadmap.yaml was read only when research-roadmap-next.yaml was absent.",
            "scripts/research_conductor.py was not modified.",
            "No push was performed.",
            "Ops status/changelog/traceability docs were left for the conductor reconciler.",
        ],
        **summary,
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
