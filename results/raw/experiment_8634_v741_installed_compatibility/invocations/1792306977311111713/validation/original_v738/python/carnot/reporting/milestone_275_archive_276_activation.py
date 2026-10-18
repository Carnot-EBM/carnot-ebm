"""Generate the Exp 2923 archive artifact for milestone 2026.05.275.

Spec: REQ-REPORT-2923, SCENARIO-REPORT-2923.

This module performs milestone-boundary bookkeeping only. It verifies that the
terminal .275 capstone exists, checks whether the historical archive ledger
already contains .275, appends a small archive row only when it is missing, and
confirms that .276 is the next or active milestone.
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
EXPERIMENT_ID = "exp2923-archive-v275-activate-v276"
ARCHIVED_MILESTONE = "2026.05.275"
ACTIVATED_MILESTONE = "2026.05.276"
RUN_DATE = "20260523"
COMPLETED = "2026-05-23"
MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"
CAPSTONE_SOURCE = "results/experiment_2922_capstone_v275.json"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2923_archive_v275_activate_v276.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
COMPLETE_VERDICT = (
    "complete: archive_ready=true; archived_milestone=2026.05.275; "
    "activated_milestone=2026.05.276"
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "archive_ready",
    "archived_milestone",
    "activated_milestone",
    "capstone_source",
    "paper_ready_from_capstone",
    "hardware_speedup_claim_eligible_from_capstone",
    "clean_artifacts_from_capstone",
    "flagged_artifacts_from_capstone",
    "blocked_artifacts_from_capstone",
    "missing_artifacts_from_capstone",
    "diagnostic_artifacts_from_capstone",
    "recommended_next_actions",
    "inference_substrate",
    "duration_s",
    "run_date",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Self-declared terminal state lets the reconciler classify the verdict "
        "without re-running. Successful archive artifacts use the exact "
        "complete archive/activation verdict required by the roadmap task."
    ),
    "archive_ready": "True only after research-complete.yaml contains a completed .275 archive row.",
    "archived_milestone": "Audit trail for the completed milestone moved into the archive ledger.",
    "activated_milestone": "Audit trail for the next milestone confirmed from roadmap state.",
    "capstone_source": "The terminal .275 capstone artifact used as archive evidence.",
    "paper_ready_from_capstone": "Copied directly from the .275 capstone paper_ready field.",
    "hardware_speedup_claim_eligible_from_capstone": (
        "Copied directly from the .275 capstone hardware claim eligibility field."
    ),
    "clean_artifacts_from_capstone": "Clean .275 artifacts copied without reclassification.",
    "flagged_artifacts_from_capstone": "Flagged .275 artifacts copied without laundering.",
    "blocked_artifacts_from_capstone": "Blocked .275 artifacts copied without reclassification.",
    "missing_artifacts_from_capstone": "Missing .275 artifacts copied without imputation.",
    "diagnostic_artifacts_from_capstone": (
        "Diagnostic-only .275 artifacts copied without headline promotion."
    ),
    "recommended_next_actions": (
        "Concrete .276 follow-up actions copied from the capstone action list."
    ),
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


def _archive_completed_block_present(text: str) -> bool:
    """Return true when any `.275` archive row records a completion date.

    WHY: ``research-complete.yaml`` is a long historical ledger written by many
    agents. A narrow text scan confirms the milestone state without letting a
    YAML serializer reformat unrelated completed experiments.
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
    finding = str(capstone.get("honest_verdict") or "Exp 2922 capstone archived.")
    return {
        "id": ARCHIVED_MILESTONE,
        "title": (
            "Hardware Baselines + Code Hallucination Repair + "
            "Verifier-Grounded Self-Learning"
        ),
        "doc": MILESTONE_DOC,
        "completed": COMPLETED,
        "finding": finding,
        "tasks": [
            {
                "id": "exp2922",
                "title": "Milestone 2026.05.275 Capstone + Claim Boundary",
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


def _recommended_next_actions(capstone: Mapping[str, Any]) -> list[str]:
    recommended = _as_str_list(capstone.get("recommended_next_actions"))
    if recommended:
        return recommended
    return _as_str_list(capstone.get("top_3_next_actions"))


def _capstone_summary(capstone: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic = _as_str_list(capstone.get("diagnostic_artifacts"))
    if not diagnostic:
        diagnostic = _as_str_list(capstone.get("diagnostic_only_artifacts"))
    return {
        "capstone_loaded": bool(capstone),
        "paper_ready_from_capstone": bool(capstone.get("paper_ready", False)),
        "hardware_speedup_claim_eligible_from_capstone": bool(
            capstone.get("hardware_speedup_claim_eligible", False)
        ),
        "clean_artifacts_from_capstone": _as_str_list(capstone.get("clean_artifacts")),
        "flagged_artifacts_from_capstone": _as_str_list(capstone.get("flagged_artifacts")),
        "blocked_artifacts_from_capstone": _as_str_list(capstone.get("blocked_artifacts")),
        "missing_artifacts_from_capstone": _as_str_list(capstone.get("missing_artifacts")),
        "diagnostic_artifacts_from_capstone": diagnostic,
        "recommended_next_actions": _recommended_next_actions(capstone),
    }


def _honest_verdict(
    *,
    archive_ready: bool,
    roadmap: Mapping[str, Any],
) -> tuple[str, list[str]]:
    blocked_reasons: list[str] = []
    if not archive_ready:
        blocked_reasons.append("research-complete.yaml does not archive 2026.05.275")
    if not roadmap["milestone_matches"]:
        blocked_reasons.append("roadmap milestone is not 2026.05.276")

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
    """REQ-REPORT-2923: write the .275 archive and .276 activation JSON."""

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

    archive_ready = _archive_completed_block_present(_read_text(complete_path))
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
