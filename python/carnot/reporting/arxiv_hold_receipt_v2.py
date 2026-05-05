"""Write the Exp 1307 arXiv v10 hold/receipt artifact.

The publication boundary here is deliberately narrow: inspect checked-in local
files, report whether a receipt already exists, and otherwise preserve the
operator hold or prior blocker without attempting any credentialed arXiv action.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_KNOWN_ISSUES_PATH = REPO_ROOT / "ops" / "known-issues.md"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1307_arxiv_v10_hold_receipt_v2.json"

EXPERIMENT = 1307
SCHEMA = "arxiv_v10_hold_receipt_v2"
RUN_DATE = "20260505"
PRIOR_BLOCKER_FILENAMES = (
    "experiment_1294_arxiv_v10_submission_receipt_or_blocker.json",
    "experiment_1295_milestone_retro_100.json",
)
REQUIRED_FIELDS = (
    "status",
    "publication_state",
    "arxiv_receipt_present",
    "operator_hold_active",
    "credentialed_submission_attempted",
    "blocker",
    "honest_verdict",
)


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Create the auditable skeleton before any local receipt evaluation."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "status": "in_progress",
            "publication_state": "unknown",
            "arxiv_receipt_present": False,
            "operator_hold_active": False,
            "credentialed_submission_attempted": False,
            "blocker": "local receipt check not yet completed",
            "honest_verdict": "in_progress",
        },
    )


def detect_operator_hold(known_issues_text: str) -> bool:
    """Return true only for an active, unresolved publication-hold section."""

    lines = known_issues_text.splitlines()
    for index, line in enumerate(lines):
        upper_line = line.upper()
        if "PUBLICATION HOLD" not in upper_line:
            continue
        if "RESOLVED" in upper_line or "~~" in line:
            continue
        section: list[str] = []
        for body_line in lines[index:]:
            if body_line.startswith("## ") and section:
                break
            section.append(body_line)
        section_text = "\n".join(section).upper()
        if "ARXIV SUBMISSION IS ON HOLD" in section_text:
            return True
    return False


def _has_receipt_signal(path: Path, payload: dict[str, Any]) -> bool:
    if path.name.startswith("arxiv_submission_receipt"):
        return True
    if payload.get("arxiv_receipt_present") is True and payload.get("publication_state") == "submitted":
        return True
    if payload.get("arxiv_submission_id"):
        return True
    if payload.get("arxiv_submitted") is True and (
        payload.get("arxiv_receipt") or payload.get("submission_receipt")
    ):
        return True
    return False


def find_local_arxiv_receipt(
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    *,
    project_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Scan checked-in result JSON files for a concrete arXiv receipt signal."""

    results_path = Path(results_dir)
    root = Path(project_root) if project_root is not None else results_path.parent
    for path in sorted(results_path.rglob("*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        if _has_receipt_signal(path, payload):
            return {"path": _relative_path(path, root), "payload": payload}
    return None


def _blocker_from_payload(payload: dict[str, Any]) -> str | None:
    publication_state = payload.get("publication_state")
    if isinstance(publication_state, dict):
        return publication_state.get("external_blocker") or publication_state.get("blocker")
    return (
        payload.get("external_blocker")
        or payload.get("gate_check_summary")
        or payload.get("blocker")
        or payload.get("honest_verdict")
    )


def _prior_blockers(results_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for filename in PRIOR_BLOCKER_FILENAMES:
        path = results_dir / filename
        if path.exists():
            payload = _read_json(path)
            blockers.append(
                {
                    "path": _relative_path(path, project_root),
                    "status": payload.get("status"),
                    "honest_verdict": payload.get("honest_verdict"),
                    "blocker": _blocker_from_payload(payload),
                }
            )
    return blockers


def _final_artifact(
    *,
    publication_state: str,
    arxiv_receipt_present: bool,
    operator_hold_active: bool,
    blocker: str | None,
    honest_verdict: str,
    receipt: dict[str, Any] | None,
    prior_blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "status": "complete",
        "publication_state": publication_state,
        "arxiv_receipt_present": arxiv_receipt_present,
        "operator_hold_active": operator_hold_active,
        "credentialed_submission_attempted": False,
        "blocker": blocker,
        "honest_verdict": honest_verdict,
        "local_receipt_path": receipt["path"] if receipt else None,
        "prior_publication_blockers": prior_blockers,
        "receipt_check_scope": "local_repository_files_only",
    }


def run(
    *,
    project_root: Path | str = REPO_ROOT,
    results_dir: Path | str | None = None,
    known_issues_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Write the terminal Exp 1307 artifact from local repository evidence only."""

    root = Path(project_root)
    results_path = Path(results_dir) if results_dir is not None else root / "results"
    known_path = Path(known_issues_path) if known_issues_path is not None else root / "ops" / "known-issues.md"
    output = Path(out_path) if out_path is not None else results_path / DEFAULT_OUT_PATH.name
    write_in_progress_artifact(output)

    known_issues_text = known_path.read_text(encoding="utf-8") if known_path.exists() else ""
    operator_hold_active = detect_operator_hold(known_issues_text)
    prior_blockers = _prior_blockers(results_path, root)
    receipt = find_local_arxiv_receipt(results_path, project_root=root)

    if receipt:
        artifact = _final_artifact(
            publication_state="submitted",
            arxiv_receipt_present=True,
            operator_hold_active=False,
            blocker=None,
            honest_verdict="local_arxiv_receipt_present_submitted",
            receipt=receipt,
            prior_blockers=prior_blockers,
        )
    elif operator_hold_active:
        artifact = _final_artifact(
            publication_state="operator_hold",
            arxiv_receipt_present=False,
            operator_hold_active=True,
            blocker="operator_publication_hold_active_no_local_receipt",
            honest_verdict="operator_hold_active_no_local_arxiv_receipt",
            receipt=None,
            prior_blockers=prior_blockers,
        )
    else:
        artifact = _final_artifact(
            publication_state="blocked",
            arxiv_receipt_present=False,
            operator_hold_active=False,
            blocker=next((entry["blocker"] for entry in prior_blockers if entry.get("blocker")), "no_local_arxiv_receipt_recorded"),
            honest_verdict="blocked_no_local_arxiv_receipt",
            receipt=None,
            prior_blockers=prior_blockers,
        )

    return _write_json(output, artifact)
