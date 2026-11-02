"""Generate the Exp 2827 archive artifact for milestone 2026.05.267.

Spec: REQ-REPORT-2827, SCENARIO-REPORT-2827.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "carnot.archive_activation.v1"
EXPERIMENT_ID = "exp2827-archive-v267-activate-v268"
ARCHIVED_MILESTONE = "2026.05.267"
ACTIVATED_MILESTONE = "2026.05.268"
COMPLETED = "2026-05-21"
DEFAULT_OUTPUT_PATH = Path("results/experiment_2827_archive_v267.json")

PARTIAL_STATUS_NOTE = (
    ".267 was a partial milestone: exp2819-exp2822 all SKIP "
    "(pre-restart gemini-cli crash storm); exp2823 retired as fabrication "
    "(legacy/fabricated/, exclusion manifest entry); exp2824 + exp2825 "
    "complete; capstone exp2826 complete."
)

MILESTONE_267_ENTRY: dict[str, Any] = {
    "id": ARCHIVED_MILESTONE,
    "title": "Partial milestone: Gemini CLI crash storm and fabricated TruthfulQA retirement",
    "doc": "",
    "completed": COMPLETED,
    "finding": (
        "Partial milestone. Four early tasks skipped during the pre-restart "
        "gemini-cli crash storm; one artifact was retired as fabricated; "
        "three tasks produced non-fabricated artifacts."
    ),
    "partial_status_note": PARTIAL_STATUS_NOTE,
    "archived_milestone_experiments_completed": 3,
    "tasks": [
        {
            "id": "exp2819",
            "title": "Archive .266 + Activate .267",
            "deliverable": "results/experiment_2819_archive_v266.json",
            "result": "SKIP (pre-restart gemini-cli crash storm)",
        },
        {
            "id": "exp2820",
            "title": "FoVer Memory-Leakage Isolation",
            "deliverable": "results/experiment_2820_fover_memory_leakage_isolation.json",
            "result": "SKIP (pre-restart gemini-cli crash storm)",
        },
        {
            "id": "exp2821",
            "title": "MBPP Corpus Dual-Condition Evaluation",
            "deliverable": "results/experiment_2821_mbpp_ensemble_eval.json",
            "result": "SKIP (pre-restart gemini-cli crash storm)",
        },
        {
            "id": "exp2822",
            "title": "HumanEval Full Dual-Condition Evaluation",
            "deliverable": "results/experiment_2822_humaneval_full_ensemble_eval.json",
            "result": "SKIP (pre-restart gemini-cli crash storm)",
        },
        {
            "id": "exp2823",
            "title": "TruthfulQA Corpus Dual-Condition Evaluation",
            "deliverable": "legacy/fabricated/experiment_2823_truthfulqa_ensemble_eval.json",
            "result": (
                "RETIRED as fabrication (legacy/fabricated/; "
                "ops/exclusion_manifest.yaml entry)"
            ),
        },
        {
            "id": "exp2824",
            "title": "Cross-Corpus Per-Verifier Dual-Condition Matrix",
            "deliverable": "results/experiment_2824_cross_corpus_verifier_matrix.json",
            "result": "OK (non-fabricated artifact)",
        },
        {
            "id": "exp2825",
            "title": "Paper v6 Section 5 Results Table",
            "deliverable": "results/experiment_2825_paper_v6_multicorpus_table.json",
            "result": "OK (non-fabricated artifact)",
        },
        {
            "id": "exp2826",
            "title": "Capstone v267",
            "deliverable": "results/experiment_2826_capstone_v267.json",
            "result": "OK (non-fabricated capstone artifact)",
        },
    ],
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix per Verdict Terminal-Prefix Discipline.",
    "archived_milestone": "Records which milestone was archived.",
    "archived_milestone_experiments_completed": (
        "Honest count: .267 had 3 of 8 tasks produce non-fabricated artifacts."
    ),
    "activated_milestone": "Records which milestone is now active.",
    "duration_s": "Wall-time measurement for this admin action.",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _load_first_yaml_mapping(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    if not text.strip():
        return {}
    payload = yaml.safe_load(text) or {}
    return payload if isinstance(payload, dict) else {}


def _roadmap_milestone(path: Path) -> str:
    return str(_load_first_yaml_mapping(path).get("milestone", "not_found"))


def _archive_entry_block() -> str:
    return yaml.safe_dump([MILESTONE_267_ENTRY], sort_keys=False, allow_unicode=False, width=120)


def _find_milestone_block(text: str, milestone: str) -> tuple[int, int] | None:
    """Return line indexes for a top-level milestone row.

    WHY: `research-complete.yaml` is large and mostly historical. Rewriting the
    entire file through PyYAML would create noisy churn, so the generator parses
    the target block boundaries and replaces only the .267 row.
    """

    lines = text.splitlines()
    start = None
    pattern = re.compile(rf"^- id:\s*['\"]?{re.escape(milestone)}['\"]?\s*$")
    for index, line in enumerate(lines):
        if pattern.match(line):
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if lines[index].startswith("- id: "):
            end = index
            break
    return start, end


def _count_milestone_entries(text: str, milestone: str) -> int:
    pattern = re.compile(rf"^- id:\s*['\"]?{re.escape(milestone)}['\"]?\s*$", re.MULTILINE)
    return len(pattern.findall(text))


def _load_archive_entry_from_text(text: str) -> dict[str, Any] | None:
    block = _find_milestone_block(text, ARCHIVED_MILESTONE)
    if block is None:
        return None
    lines = text.splitlines()
    payload = yaml.safe_load("\n".join(lines[block[0] : block[1]])) or []
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return None  # pragma: no cover - defensive guard for malformed YAML blocks.


def _replace_or_append_archive_entry(path: Path) -> dict[str, Any]:
    original = _read_text(path)
    block = _find_milestone_block(original, ARCHIVED_MILESTONE)
    entry_block = _archive_entry_block().rstrip()
    if block is None:
        separator = "" if not original or original.endswith("\n") else "\n"
        prefix = original if original else "milestones:\n"
        path.write_text(prefix + separator + entry_block + "\n", encoding="utf-8")
        return {"appended_this_run": True, "existing_entry_corrected": False}

    lines = original.splitlines()
    replacement = entry_block.splitlines()
    corrected = lines[block[0] : block[1]] != replacement
    updated = lines[: block[0]] + replacement + lines[block[1] :]
    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return {"appended_this_run": False, "existing_entry_corrected": corrected}


def _fabricated_retirement_present(root: Path) -> bool:
    exclusion_text = _read_text(root / "ops" / "exclusion_manifest.yaml")
    fabricated_readme = _read_text(root / "legacy" / "fabricated" / "README.md")
    return "experiment_id: 2823" in exclusion_text and "experiment_2823" in fabricated_readme


def _honest_verdict(archive_ready: bool, active_milestone: str) -> str:
    if active_milestone != ACTIVATED_MILESTONE:
        return (
            "complete: unexpected_active_milestone; "
            f"archived_milestone={ARCHIVED_MILESTONE}; active_milestone={active_milestone}"
        )
    readiness = str(archive_ready).lower()
    return (
        f"complete: archive_ready={readiness}; "
        f"archived_milestone={ARCHIVED_MILESTONE}; active_milestone={active_milestone}; "
        "completed_non_fabricated=3_of_8"
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """REQ-REPORT-2827: write the idempotent .267 archive activation artifact."""

    start_s = clock()
    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output

    roadmap_path = root_path / "research-roadmap.yaml"
    complete_path = root_path / "research-complete.yaml"
    active_milestone_before = _roadmap_milestone(roadmap_path)
    complete_before = _read_text(complete_path)
    existing_entry_before = _load_archive_entry_from_text(complete_before)
    existing_count_before = _count_milestone_entries(complete_before, ARCHIVED_MILESTONE)

    archive_update = {"appended_this_run": False, "existing_entry_corrected": False}
    if active_milestone_before == ACTIVATED_MILESTONE:
        archive_update = _replace_or_append_archive_entry(complete_path)

    complete_after = _read_text(complete_path)
    existing_count_after = _count_milestone_entries(complete_after, ARCHIVED_MILESTONE)
    existing_entry_after = _load_archive_entry_from_text(complete_after)
    active_milestone_after = _roadmap_milestone(roadmap_path)
    archive_ready = bool(
        active_milestone_after == ACTIVATED_MILESTONE
        and existing_count_after == 1
        and existing_entry_after
        and existing_entry_after.get("archived_milestone_experiments_completed") == 3
    )
    duration_s = round(clock() - start_s, 6)

    artifact = {
        "id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "honest_verdict": _honest_verdict(archive_ready, active_milestone_after),
        "archived_milestone": ARCHIVED_MILESTONE,
        "archived_milestone_experiments_completed": 3,
        "activated_milestone": active_milestone_after,
        "archive_ready": archive_ready,
        "completed": COMPLETED,
        "duration_s": duration_s,
        "preconditions_checked": {
            "roadmap_milestone": {
                "command": "yaml.safe_load(research-roadmap.yaml).milestone",
                "observed": active_milestone_before,
                "expected": ACTIVATED_MILESTONE,
                "passed": active_milestone_before == ACTIVATED_MILESTONE,
            },
            "research_complete_archive": {
                "command": "search research-complete.yaml for 2026.05.267",
                "observed_before_count": existing_count_before,
                "observed_after_count": existing_count_after,
                "passed": existing_count_after == 1,
            },
            "fabricated_retirement": {
                "command": "check ops/exclusion_manifest.yaml and legacy/fabricated/README.md",
                "passed": _fabricated_retirement_present(root_path),
            },
        },
        "archive": {
            "research_complete_path": str(complete_path),
            "existing_entry_before_run": existing_entry_before is not None,
            "existing_entry_after_run": existing_entry_after is not None,
            "duplicate_count_after": existing_count_after,
            "partial_status_note": PARTIAL_STATUS_NOTE,
            "decision": (
                "Corrected the existing .267 archive row."
                if archive_update["existing_entry_corrected"]
                else "Appended the .267 archive row."
                if archive_update["appended_this_run"]
                else "No archive mutation performed for this roadmap state."
            ),
            **archive_update,
        },
        "activation": {
            "research_roadmap_path": str(roadmap_path),
            "observed_active_milestone_before": active_milestone_before,
            "observed_active_milestone_after": active_milestone_after,
            "expected_active_milestone": ACTIVATED_MILESTONE,
            "confirmed": active_milestone_after == ACTIVATED_MILESTONE,
        },
        "field_principles": FIELD_PRINCIPLES,
        "notes": [
            "scripts/research_conductor.py was not modified.",
            "No push was performed.",
            "Ops status/changelog/traceability docs were left for the conductor reconciler.",
        ],
    }
    return _write_json(output, artifact)


if __name__ == "__main__":  # pragma: no cover
    run()
