"""Exp 1204 retro-template STEP 0 documentation fix.

This module makes the .94 operational fix auditable: it finds the known-issues
entry for retros that ran out of turn budget before writing artifacts, appends
the resolution note without pruning historical text, verifies that Exp 1215 is
configured with the STEP 0 skeleton pattern and opus/100 turns, and writes the
machine-readable Exp 1204 artifact.

Spec: REQ-REPORT-014, SCENARIO-REPORT-011.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
KNOWN_ISSUES_PATH = REPO_ROOT / "ops" / "known-issues.md"
ROADMAP_PATH = REPO_ROOT / "research-roadmap.yaml"
DELIVERABLE_PATH = REPO_ROOT / "results" / "experiment_1204_retro_template_step0_fix.json"

ISSUE_TITLE = "Retro Task Boundary Too Tight"
RESOLUTION_NOTE = "RESOLVED .94 (2026-05-03): exp1215 uses STEP 0 skeleton + opus/100 turns"


def _read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _issue_heading_match(text: str) -> re.Match[str] | None:
    pattern = rf"^### .*\b{re.escape(ISSUE_TITLE)}\b.*$"
    return re.search(pattern, text, flags=re.MULTILINE)


def retro_boundary_issue_found(text: str) -> bool:
    """Return whether the known-issues text contains the retro boundary entry."""

    return _issue_heading_match(text) is not None


def resolution_note_present(text: str) -> bool:
    """Return whether the exact .94 resolution note is already present."""

    return RESOLUTION_NOTE in text


def add_resolution_note(text: str) -> tuple[str, bool]:
    """Add the .94 resolution note immediately under the issue heading.

    The known-issues file is an operational history, not a mutable ticket list.
    This function therefore appends only the resolution line and keeps every
    existing detail about the original failure pattern intact.
    """

    heading_match = _issue_heading_match(text)
    if heading_match is None or resolution_note_present(text):
        return text, False

    insert_at = heading_match.end()
    suffix = text[insert_at:]
    separator = "" if suffix.startswith("\n") else "\n"
    updated = f"{text[:insert_at]}\n{RESOLUTION_NOTE}{separator}{suffix}"
    return updated, True


def _exp1215_task_block(roadmap_text: str) -> str:
    match = re.search(
        r"^- id:\s*exp1215[^\n]*\n.*?(?=^- id:|\Z)",
        roadmap_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return match.group(0) if match else ""


def exp1215_step0_pattern_found(roadmap_text: str) -> bool:
    """Return whether Exp 1215 explicitly writes a STEP 0 skeleton artifact."""

    block = _exp1215_task_block(roadmap_text).lower()
    return bool(block) and "step 0" in block and "write skeleton artifact" in block


def exp1215_opus_100_found(roadmap_text: str) -> bool:
    """Return whether Exp 1215 is configured for opus with max_turns 100."""

    block = _exp1215_task_block(roadmap_text)
    return (
        bool(block)
        and "model: opus" in block
        and re.search(r"max_turns:\s*100\b", block) is not None
    )


def _honest_verdict(
    issue_found: bool,
    resolution_note_added: bool,
    already_resolved: bool,
    exp1215_template_fix_found: bool,
) -> str:
    if not issue_found or not exp1215_template_fix_found:
        return "blocked"
    if resolution_note_added:
        return "template_updated"
    if already_resolved:
        return "already_resolved"
    return "blocked"


def build_artifact(
    known_issues_text: str,
    roadmap_text: str,
    *,
    resolution_note_added: bool,
    known_issues_file_updated: bool,
) -> dict[str, object]:
    """Build the Exp 1204 artifact from updated source texts."""

    issue_found = retro_boundary_issue_found(known_issues_text)
    step0_found = exp1215_step0_pattern_found(roadmap_text)
    opus_100_found = exp1215_opus_100_found(roadmap_text)
    template_fix_found = step0_found and opus_100_found
    already_resolved = resolution_note_present(known_issues_text) and not resolution_note_added

    return {
        "experiment": "1204_retro_template_step0_fix",
        "schema": "retro_template_step0_fix_v1",
        "run_date": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retro_boundary_issue_found": issue_found,
        "resolution_note_added": resolution_note_added,
        "known_issues_file_updated": known_issues_file_updated,
        "retro_template_updated": resolution_note_added,
        "exp1215_step0_pattern_found": step0_found,
        "exp1215_opus_100_found": opus_100_found,
        "honest_verdict": _honest_verdict(
            issue_found,
            resolution_note_added,
            already_resolved,
            template_fix_found,
        ),
    }


def run(
    *,
    known_issues_path: Path = KNOWN_ISSUES_PATH,
    roadmap_path: Path = ROADMAP_PATH,
    out_path: Path = DELIVERABLE_PATH,
) -> dict[str, object]:
    """Update known-issues and write the Exp 1204 JSON deliverable."""

    known_issues_text = _read_optional(known_issues_path)
    updated_text, note_added = add_resolution_note(known_issues_text)
    if note_added:
        known_issues_path.write_text(updated_text, encoding="utf-8")

    roadmap_text = _read_optional(roadmap_path)
    artifact = build_artifact(
        updated_text,
        roadmap_text,
        resolution_note_added=note_added,
        known_issues_file_updated=note_added,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-issues", type=Path, default=KNOWN_ISSUES_PATH)
    parser.add_argument("--roadmap", type=Path, default=ROADMAP_PATH)
    parser.add_argument("--out", type=Path, default=DELIVERABLE_PATH)
    args = parser.parse_args(argv)

    artifact = run(
        known_issues_path=args.known_issues,
        roadmap_path=args.roadmap,
        out_path=args.out,
    )
    print(
        f"[exp1204] verdict={artifact['honest_verdict']} "
        f"resolution_note_added={artifact['resolution_note_added']} out={args.out}"
    )
    return 1 if artifact["honest_verdict"] == "blocked" else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
