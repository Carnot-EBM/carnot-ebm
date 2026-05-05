"""Write the Exp 1321 publication-hold related-work delta artifact.

This runner is intentionally non-submission code. It records the current local
publication state, folds the 2026-05-05 literature sweep into related-work
notes when a suitable local notes file exists, and otherwise keeps the delta in
the JSON artifact.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from carnot.reporting.arxiv_hold_receipt_v2 import detect_operator_hold

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = (
    DEFAULT_RESULTS_DIR / "experiment_1321_publication_hold_related_work_delta_v11.json"
)
DEFAULT_EXP1307_PATH = DEFAULT_RESULTS_DIR / "experiment_1307_arxiv_v10_hold_receipt_v2.json"
DEFAULT_KNOWN_ISSUES_PATH = REPO_ROOT / "ops" / "known-issues.md"
DEFAULT_REFERENCES_PATH = REPO_ROOT / "research-references.md"
DEFAULT_RELATED_WORK_NOTES_PATH = (
    REPO_ROOT / "docs" / "research-notes" / "literature-priority-audit.md"
)

EXPERIMENT = 1321
SCHEMA = "publication_hold_related_work_delta_v11"
RUN_DATE = "20260505"
DELTA_HEADING = "## 2026-05-05 Related-Work Delta (Exp 1321)"
REQUIRED_FIELDS = (
    "status",
    "publication_state",
    "operator_hold_active",
    "credentialed_submission_attempted",
    "related_work_delta_written",
    "new_references_count",
    "honest_verdict",
)


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-PUBLISH-016: create the auditable skeleton before reading inputs."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "status": "in_progress",
            "publication_state": "unknown",
            "operator_hold_active": False,
            "credentialed_submission_attempted": False,
            "related_work_delta_written": False,
            "new_references_count": 0,
            "honest_verdict": "in_progress",
        },
    )


def _field_from_entry(body: str, field_name: str) -> str:
    pattern = re.compile(rf"^- \*\*{field_name}s?:\*\*\s*(.+)$", re.MULTILINE)
    match = pattern.search(body)
    return " ".join(match.group(1).split()) if match else ""


def extract_20260505_reference_entries(references_text: str) -> list[dict[str, str]]:
    """REQ-PUBLISH-016: extract material 2025-2026 entries from the 2026-05-05 sweep."""

    entries: list[dict[str, str]] = []
    capture = False
    current_heading = ""
    current_title = ""
    current_body: list[str] = []

    def flush_entry() -> None:
        if not current_title:
            return
        body = "\n".join(current_body)
        if not re.search(r"\b(?:2025|2026|25\d{2}\.|26\d{2}\.|arXiv\s+2[56])", body, re.I):
            return
        entries.append(
            {
                "sweep": current_heading,
                "title": current_title,
                "paper": _field_from_entry(body, "Paper"),
                "source": _field_from_entry(body, "Source"),
                "what": _field_from_entry(body, "What"),
                "relevance": _field_from_entry(body, "Relevance to Carnot"),
            }
        )

    for line in references_text.splitlines():
        if line.startswith("## "):
            flush_entry()
            capture = line.startswith("## 2026-05-05")
            current_heading = line[3:].strip() if capture else ""
            current_title = ""
            current_body = []
        elif capture and line.startswith("### "):
            flush_entry()
            current_title = line[4:].strip()
            current_body = []
        elif capture:
            current_body.append(line)
    flush_entry()
    return entries


def _reference_cluster(title: str) -> str:
    lower = title.lower()
    if any(term in lower for term in ("constraint", "falcon", "grammar", "infeasibility")):
        return "constraint and certificate generation"
    if any(term in lower for term in ("satquest", "semantic", "drift")):
        return "verifier-backed reasoning and semantic control"
    if any(term in lower for term in ("cerce", "dvi", "querybandits", "garbage")):
        return "online self-learning and non-forgetting"
    if any(term in lower for term in ("kan", "p-bit", "ising", "extropic", "kona")):
        return "hardware-portable energy and KAN context"
    return "other publication-context updates"


def build_related_work_delta(entries: list[dict[str, str]]) -> str:
    """REQ-PUBLISH-016: summarize the 2026-05-05 sweep without implying submission."""

    clusters: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        clusters[_reference_cluster(entry["title"])].append(entry["title"].split(":", 1)[0])

    lines = [
        DELTA_HEADING,
        "",
        "Publication remains under operator hold; this notes delta does not record,",
        "authorize, or imply an arXiv submission.",
        "",
        f"New material 2025-2026 references counted: {len(entries)}.",
        "",
        "Compact related-work impact:",
    ]
    for cluster, titles in clusters.items():
        lines.append(f"- {cluster}: {', '.join(titles)}.")
    lines.extend(
        [
            "",
            "Honest impact: the sweep strengthens the paper's related-work framing around",
            "constraint-backed generation, verifier-grounded reasoning, continual-learning",
            "safety, and hardware-portable energy models, but it does not lift the",
            "operator hold or justify credentialed submission.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_related_work_delta(
    *,
    notes_path: Path,
    project_root: Path,
    delta: str,
) -> str:
    if not notes_path.exists():
        return "artifact"
    text = notes_path.read_text(encoding="utf-8")
    if DELTA_HEADING not in text:
        separator = "\n\n" if text.strip() else ""
        notes_path.write_text(text.rstrip() + separator + delta, encoding="utf-8")
    return _relative_path(notes_path, project_root)


def _publication_state(exp1307: dict[str, Any], operator_hold_active: bool) -> str:
    if exp1307.get("publication_state") == "submitted":
        return "submitted"
    if operator_hold_active or exp1307.get("operator_hold_active") is True:
        return "operator_hold"
    return str(exp1307.get("publication_state") or "blocked")


def validate_artifact(artifact: dict[str, Any]) -> None:
    """REQ-PUBLISH-016: enforce the terminal schema and non-submission boundary."""

    missing = sorted(set(REQUIRED_FIELDS).difference(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["credentialed_submission_attempted"] is False, (
        "credentialed_submission_attempted must remain false"
    )
    assert artifact["status"] in {"in_progress", "complete"}
    if artifact["status"] == "complete":
        assert artifact["related_work_delta_written"] is True
        assert artifact["new_references_count"] > 0
        assert artifact["honest_verdict"]


def run(
    *,
    project_root: Path | str = REPO_ROOT,
    exp1307_path: Path | str | None = None,
    known_issues_path: Path | str | None = None,
    references_path: Path | str | None = None,
    related_work_notes_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """SCENARIO-PUBLISH-017: write the final artifact from local evidence only."""

    root = Path(project_root)
    results_dir = root / "results"
    output = Path(out_path) if out_path is not None else results_dir / DEFAULT_OUT_PATH.name
    write_in_progress_artifact(output)

    exp1307_file = (
        Path(exp1307_path)
        if exp1307_path is not None
        else root / DEFAULT_EXP1307_PATH.relative_to(REPO_ROOT)
    )
    known_file = (
        Path(known_issues_path)
        if known_issues_path is not None
        else root / DEFAULT_KNOWN_ISSUES_PATH.relative_to(REPO_ROOT)
    )
    references_file = (
        Path(references_path)
        if references_path is not None
        else root / DEFAULT_REFERENCES_PATH.relative_to(REPO_ROOT)
    )
    notes_file = (
        Path(related_work_notes_path)
        if related_work_notes_path is not None
        else root / DEFAULT_RELATED_WORK_NOTES_PATH.relative_to(REPO_ROOT)
    )

    exp1307 = _read_json(exp1307_file)
    known_text = known_file.read_text(encoding="utf-8") if known_file.exists() else ""
    operator_hold_active = (
        detect_operator_hold(known_text) or exp1307.get("operator_hold_active") is True
    )
    entries = extract_20260505_reference_entries(references_file.read_text(encoding="utf-8"))
    delta = build_related_work_delta(entries)
    delta_target = _write_related_work_delta(notes_path=notes_file, project_root=root, delta=delta)
    publication_state = _publication_state(exp1307, operator_hold_active)

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": "complete",
        "publication_state": publication_state,
        "operator_hold_active": operator_hold_active,
        "credentialed_submission_attempted": False,
        "related_work_delta_written": True,
        "new_references_count": len(entries),
        "honest_verdict": (
            "operator_hold_active_related_work_delta_written_no_submission"
            if publication_state == "operator_hold"
            else "related_work_delta_written_no_submission"
        ),
        "related_work_delta_target": delta_target,
        "related_work_delta": delta,
        "material_reference_titles": [entry["title"] for entry in entries],
        "source_artifacts": [
            _relative_path(exp1307_file, root),
            _relative_path(known_file, root),
            _relative_path(references_file, root),
        ],
        "exp1307_honest_verdict": exp1307.get("honest_verdict"),
        "receipt_check_scope": "local_repository_files_only",
    }
    validate_artifact(artifact)
    return _write_json(output, artifact)


def main() -> None:  # pragma: no cover
    artifact = run()
    print(
        artifact["publication_state"],
        artifact["credentialed_submission_attempted"],
        artifact["new_references_count"],
        artifact["honest_verdict"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
