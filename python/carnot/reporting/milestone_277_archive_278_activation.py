"""Generate the Exp 2949 archive artifact for milestone 2026.05.277.

Spec refs: REQ-REPORT-2949, SCENARIO-REPORT-2949.

This module is intentionally boring bookkeeping. It reads the terminal .277
capstone, ensures the historical completion ledger has a .277 row, confirms
that .278 is either staged or already active, and writes the JSON acceptance
object the conductor expects.
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
EXPERIMENT_ID = "exp2949-archive-v277-activate-v278"
ARCHIVED_MILESTONE = "2026.05.277"
ACTIVATED_MILESTONE = "2026.05.278"
RUN_DATE = "20260523"
COMPLETED = "2026-05-23"
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
CAPSTONE_SOURCE = "results/experiment_2948_capstone_v277.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2949_archive_v277_activate_v278.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: archive_ready=true; archived_milestone=2026.05.277; activated_milestone=2026.05.278"
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "archived_milestone",
    "activated_milestone",
    "capstone_source",
    "paper_ready_from_capstone",
    "headline_outcome_from_capstone",
    "clean_artifacts_from_capstone",
    "flagged_artifacts_from_capstone",
    "blocked_artifacts_from_capstone",
    "missing_artifacts_from_capstone",
    "next_gaps_from_capstone",
    "archive_ready",
    "inference_substrate",
    "duration_s",
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Self-declared terminal state per Verdict Terminal-Prefix Discipline.",
    "archive_ready": "True only after research-complete.yaml contains exactly one completed .277 archive row.",
    "archived_milestone": "Audit trail for the completed milestone moved into the archive ledger.",
    "activated_milestone": "Audit trail for the next milestone confirmed from roadmap state.",
    "capstone_source": "The terminal .277 capstone artifact used as archive evidence.",
    "paper_ready_from_capstone": "Copied directly from the .277 capstone paper_ready field.",
    "headline_outcome_from_capstone": (
        "Copied from deep_think_corrigenda_outcomes.headline_outcome without inference."
    ),
    "clean_artifacts_from_capstone": "Clean .277 artifacts copied without reclassification.",
    "flagged_artifacts_from_capstone": "Flagged .277 artifacts copied without laundering.",
    "blocked_artifacts_from_capstone": "Blocked .277 artifacts copied without reclassification.",
    "missing_artifacts_from_capstone": "Missing .277 artifacts copied without imputation.",
    "next_gaps_from_capstone": "The .278 carry-forward gaps copied from the capstone action list.",
    "inference_substrate": "Declares this as pure aggregation from upstream artifacts.",
    "duration_s": "Real wall-clock duration for the admin run; no sleep padding.",
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


def _as_int_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, int) and not isinstance(item, bool)
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
    """REQ-REPORT-2949: detect an existing completed .277 archive block."""

    return _archive_completed_block_count(text) > 0


def _minimal_archive_entry(capstone: Mapping[str, Any]) -> dict[str, Any]:
    finding = str(capstone.get("honest_verdict") or "Exp 2948 capstone archived.")
    return {
        "id": ARCHIVED_MILESTONE,
        "title": "Phase-3 Deep Think Corrigenda + Paper-v6 Narrowing Verification",
        "doc": MILESTONE_DOC,
        "completed": COMPLETED,
        "finding": finding,
        "tasks": [
            {
                "id": "exp2948",
                "title": "Capstone .277: Deep Think Corrigenda + Paper-v6 Narrowing Verification",
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
        "research_roadmap_next_exists": next_path.exists(),
        "used_active_roadmap_fallback": fallback,
        "observed_milestone": milestone,
        "expected_milestone": ACTIVATED_MILESTONE,
        "milestone_matches": milestone == ACTIVATED_MILESTONE,
        "observed_milestone_doc": milestone_doc,
        "expected_milestone_doc": MILESTONE_DOC,
        "milestone_doc_matches": milestone_doc == MILESTONE_DOC,
    }


def _headline_outcome(capstone: Mapping[str, Any]) -> str:
    outcomes = capstone.get("deep_think_corrigenda_outcomes")
    if not isinstance(outcomes, Mapping):
        return ""
    return str(outcomes.get("headline_outcome") or "")


def _next_gaps(capstone: Mapping[str, Any]) -> list[str]:
    gaps = _as_str_list(capstone.get("gaps_for_278"))
    if gaps:
        return gaps
    return _as_str_list(capstone.get("top_3_next_actions"))


def _capstone_summary(capstone: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "capstone_loaded": bool(capstone),
        "capstone_milestone": str(capstone.get("milestone") or ""),
        "capstone_honest_verdict": str(capstone.get("honest_verdict") or ""),
        "paper_ready_from_capstone": bool(capstone.get("paper_ready", False)),
        "headline_outcome_from_capstone": _headline_outcome(capstone),
        "clean_artifacts_from_capstone": _as_str_list(capstone.get("clean_artifacts")),
        "flagged_artifacts_from_capstone": _as_str_list(capstone.get("flagged_artifacts")),
        "blocked_artifacts_from_capstone": _as_str_list(capstone.get("blocked_artifacts")),
        "missing_artifacts_from_capstone": _as_str_list(capstone.get("missing_artifacts")),
        "artifact_classification_counts_from_capstone": _as_int_mapping(
            capstone.get("artifact_classification_counts")
        ),
        "next_gaps_from_capstone": _next_gaps(capstone),
    }


def _honest_verdict(
    *,
    archive_ready: bool,
    roadmap: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blocked_reasons: list[str] = []
    if not archive_ready:
        blocked_reasons.append("research-complete.yaml does not archive 2026.05.277")
    if not roadmap["milestone_matches"]:
        blocked_reasons.append("roadmap milestone is not 2026.05.278")

    if blocked_reasons:
        return "blocked: " + "; ".join(blocked_reasons), blocked_reasons

    return COMPLETE_VERDICT, blocked_reasons


def _base_artifact(duration_s: float) -> dict[str, Any]:
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
        "pushed": False,
        "scripts_research_conductor_modified": False,
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
) -> dict[str, Any]:
    """SCENARIO-REPORT-2949: write the .277 archive and .278 activation JSON."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    capstone = _read_json_mapping(root_path / CAPSTONE_SOURCE)
    if not capstone:
        duration_s = round(clock() - start_s, 6)
        summary = _capstone_summary({})
        artifact = {
            **_base_artifact(duration_s),
            "honest_verdict": "blocked_capstone_missing",
            "archive_ready": False,
            "archive_already_present": _archive_completed_block_present(
                _read_text(root_path / "research-complete.yaml")
            ),
            "archive_appended_this_run": False,
            "archive_completed_block_count": _archive_completed_block_count(
                _read_text(root_path / "research-complete.yaml")
            ),
            "activation": _roadmap_metadata(root_path),
            "archive": {
                "research_complete_path": "research-complete.yaml",
                "ready_after_run": False,
            },
            "blocked_reasons": ["capstone source missing or invalid"],
            "notes": ["Capstone precondition failed; archive mutation was skipped."],
            **summary,
        }
        return _write_json(output, artifact)

    complete_path = root_path / "research-complete.yaml"
    complete_before = _read_text(complete_path)
    archive_already_present = _archive_completed_block_present(complete_before)

    archive_appended_this_run = False
    if not archive_already_present:
        _append_archive_entry(complete_path, capstone)
        archive_appended_this_run = True

    archive_count = _archive_completed_block_count(_read_text(complete_path))
    archive_ready = archive_count == 1
    roadmap = _roadmap_metadata(root_path)
    summary = _capstone_summary(capstone)
    honest_verdict, blocked_reasons = _honest_verdict(
        archive_ready=archive_ready,
        roadmap=roadmap,
    )
    duration_s = round(clock() - start_s, 6)

    artifact = {
        **_base_artifact(duration_s),
        "honest_verdict": honest_verdict,
        "archive_ready": archive_ready,
        "archive_already_present": archive_already_present,
        "archive_appended_this_run": archive_appended_this_run,
        "archive_completed_block_count": archive_count,
        "activation": roadmap,
        "archive": {
            "research_complete_path": "research-complete.yaml",
            "ready_after_run": archive_ready,
        },
        "blocked_reasons": blocked_reasons,
        "notes": [
            "research-roadmap-next.yaml was checked first when present.",
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
