"""Generate the Exp 2962 archive artifact for milestone 2026.05.278.

Spec refs: REQ-REPORT-2962, SCENARIO-REPORT-2962.

This module performs the routine milestone-boundary bookkeeping from .278 to
.279. It reads the terminal .278 capstone, ensures the historical completion
ledger has one completed .278 row, confirms the .279 roadmap state, and writes
the JSON acceptance object. It does not rerun research work or edit the
conductor.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2962-archive-v278-activate-v279"
ARCHIVED_MILESTONE = "2026.05.278"
ACTIVATED_MILESTONE = "2026.05.279"
RUN_DATE = "20260524"
COMPLETED = "2026-05-24"
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
CAPSTONE_SOURCE = "results/experiment_2961_capstone_v278.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2962_archive_v278_activate_v279.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: archive_ready=true; archived_milestone=2026.05.278; activated_milestone=2026.05.279"
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
    "archive_ready": "True only after research-complete.yaml contains exactly one completed .278 archive row.",
    "archived_milestone": "Audit trail for the completed milestone moved into the archive ledger.",
    "activated_milestone": "Audit trail for the next milestone confirmed from roadmap state.",
    "capstone_source": "The terminal .278 capstone artifact used as archive evidence.",
    "paper_ready_from_capstone": "Copied directly from the .278 capstone paper_ready field.",
    "headline_outcome_from_capstone": "Copied directly from the .278 capstone headline_outcome field.",
    "clean_artifacts_from_capstone": "Clean .278 artifacts copied without reclassification.",
    "flagged_artifacts_from_capstone": "Flagged .278 artifacts copied without laundering.",
    "blocked_artifacts_from_capstone": "Blocked .278 artifacts copied without reclassification.",
    "missing_artifacts_from_capstone": "Missing .278 artifacts copied without imputation.",
    "next_gaps_from_capstone": "The .279 carry-forward gaps copied from the capstone gap lists.",
    "inference_substrate": "Declares this as pure aggregation from upstream artifacts.",
    "duration_s": "Real wall-clock duration for the admin run; no sleep padding.",
}


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
    """REQ-REPORT-2962: detect an existing completed .278 archive block."""

    return _archive_completed_block_count(text) > 0


def _minimal_archive_entry(capstone: Mapping[str, Any]) -> dict[str, Any]:
    finding = str(capstone.get("honest_verdict") or "Exp 2961 capstone archived.")
    return {
        "id": ARCHIVED_MILESTONE,
        "title": "Structured Code Repair + Utility-Gated Self-Learning + GateMate Materialization",
        "doc": MILESTONE_DOC,
        "completed": COMPLETED,
        "finding": finding,
        "tasks": [
            {
                "id": "exp2961",
                "title": "Capstone .278",
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
    return str(capstone.get("headline_outcome") or "")


def _next_gaps(capstone: Mapping[str, Any]) -> list[str]:
    gaps = _as_str_list(capstone.get("gaps_remaining"))
    if gaps:
        return gaps
    return _as_str_list(capstone.get("next_milestone_recommendations"))


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
        blocked_reasons.append("research-complete.yaml does not archive 2026.05.278")
    if not roadmap["milestone_matches"]:
        blocked_reasons.append("roadmap milestone is not 2026.05.279")

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
    """SCENARIO-REPORT-2962: write the .278 archive and .279 activation JSON."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_before = _read_text(root_path / "research-roadmap.yaml")
    capstone = _read_json_mapping(root_path / CAPSTONE_SOURCE)
    if not capstone:
        duration_s = round(clock() - start_s, 6)
        summary = _capstone_summary({})
        complete_text = _read_text(root_path / "research-complete.yaml")
        roadmap_after = _read_text(root_path / "research-roadmap.yaml")
        artifact = {
            **_base_artifact(duration_s),
            "honest_verdict": "blocked_capstone_missing",
            "archive_ready": False,
            "archive_already_present": _archive_completed_block_present(complete_text),
            "archive_appended_this_run": False,
            "archive_completed_block_count": _archive_completed_block_count(complete_text),
            "activation": _roadmap_metadata(root_path),
            "archive": {
                "research_complete_path": "research-complete.yaml",
                "ready_after_run": False,
            },
            "roadmap_verification": {
                "research_roadmap_yaml_sha256_before": _sha256_text(roadmap_before),
                "research_roadmap_yaml_sha256_after": _sha256_text(roadmap_after),
                "research_roadmap_yaml_modified": roadmap_before != roadmap_after,
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
    roadmap_after = _read_text(root_path / "research-roadmap.yaml")

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
        "roadmap_verification": {
            "research_roadmap_yaml_sha256_before": _sha256_text(roadmap_before),
            "research_roadmap_yaml_sha256_after": _sha256_text(roadmap_after),
            "research_roadmap_yaml_modified": roadmap_before != roadmap_after,
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
