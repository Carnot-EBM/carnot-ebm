"""Build the Exp 1412 browser-only arXiv operator action sheet.

This module intentionally does not know how to submit to arXiv. Exp 1390
already proved that non-interactive credentials were unavailable, so the useful
operator-facing next step is a compact sheet that points at the verified source
archive and the exact metadata block to paste into the browser form.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_PATH = Path("results") / "experiment_1412_arxiv_operator_action_sheet_v3.json"
DEFAULT_BUNDLE_PATH = Path("results") / "arxiv_bundle_v11.tar.gz"
DEFAULT_MANUAL_CHECKLIST_PATH = Path("docs") / "arxiv-manual-submission-checklist.md"
DEFAULT_OPERATOR_ACTION_SHEET_PATH = Path("docs") / "arxiv-submit-now.md"
RUN_DATE = "20260506"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "bundle_path",
    "bundle_exists",
    "bundle_size_bytes",
    "manual_checklist_path",
    "operator_action_sheet_path",
    "submission_ready_for_operator",
    "credentialed_submission_attempted",
    "honest_verdict",
}


def _relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def _base_artifact(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "bundle_path": "results/arxiv_bundle_v11.tar.gz",
        "bundle_exists": False,
        "bundle_size_bytes": 0,
        "manual_checklist_path": "docs/arxiv-manual-submission-checklist.md",
        "operator_action_sheet_path": "docs/arxiv-submit-now.md",
        "submission_ready_for_operator": False,
        "credentialed_submission_attempted": False,
        "honest_verdict": status,
    }


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """Write the durable placeholder before bundle or checklist validation."""

    artifact = _base_artifact("in_progress")
    artifact["honest_verdict"] = "in_progress_operator_action_sheet_not_yet_validated"
    return _write_json(Path(out_path), artifact)


def _required_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"manual checklist missing {label}")
    return " ".join(match.group(1).strip().split())


def _fenced_value(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}:\s*```text\s*(.*?)\s*```"
    return _required_match(pattern, text, label)


def extract_checklist_metadata(checklist_text: str) -> dict[str, Any]:
    """Extract the exact browser-form metadata from the manual checklist."""

    return {
        "upload_url": _required_match(
            r"Upload URL:\s*(https://arxiv\.org/submit)", checklist_text, "Upload URL"
        ),
        "title": _fenced_value(checklist_text, "Title"),
        "author": _fenced_value(checklist_text, "Authors"),
        "primary_category": _fenced_value(checklist_text, "Primary category"),
        "license": _fenced_value(checklist_text, "License"),
        "abstract_present": bool(_fenced_value(checklist_text, "Abstract")),
    }


def build_action_sheet(
    *,
    metadata: dict[str, Any],
    bundle_path: str,
    manual_checklist_path: str,
    run_date: str = RUN_DATE,
) -> str:
    """Render the terse, under-five-minute browser action sheet."""

    return f"""# Submit Carnot to arXiv Now

Run date: {run_date}

1. Open {metadata["upload_url"]} and sign in.
2. Start a new submission and choose the compressed TeX/source upload path.
3. Upload `{bundle_path}`.
4. Let AutoTeX process the archive, then inspect the generated PDF preview.
5. Use this metadata:
   - Title: {metadata["title"]}
   - Author: {metadata["author"]}
   - Primary category: {metadata["primary_category"]}
   - License: {metadata["license"]}
   - Abstract: paste the Abstract block from `{manual_checklist_path}`.
6. Final-review the title, author, category, license, abstract, figures, and references.
7. Submit in the browser and record the arXiv identifier in the Exp 1390 artifact.

No SWORD/API submission is attempted by this action sheet.
"""


def run(
    *,
    project_root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
    bundle_path: Path | str = DEFAULT_BUNDLE_PATH,
    manual_checklist_path: Path | str = DEFAULT_MANUAL_CHECKLIST_PATH,
    operator_action_sheet_path: Path | str = DEFAULT_OPERATOR_ACTION_SHEET_PATH,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Validate the ready bundle, write the action sheet, and emit the artifact."""

    root = Path(project_root)
    output = root / out_path
    bundle = root / bundle_path
    checklist = root / manual_checklist_path
    action_sheet = root / operator_action_sheet_path
    write_in_progress_artifact(output)

    bundle_exists = bundle.exists()
    bundle_size = bundle.stat().st_size if bundle_exists else 0
    common = {
        "bundle_path": _relative_path(bundle, root),
        "bundle_exists": bundle_exists,
        "bundle_size_bytes": bundle_size,
        "manual_checklist_path": _relative_path(checklist, root),
        "operator_action_sheet_path": _relative_path(action_sheet, root),
    }
    if bundle_size <= 0:
        artifact = _base_artifact("blocked")
        artifact.update(
            {
                **common,
                "submission_ready_for_operator": False,
                "credentialed_submission_attempted": False,
                "honest_verdict": "blocked_bundle_missing_or_empty",
            }
        )
        return _write_json(output, artifact)

    metadata = extract_checklist_metadata(checklist.read_text(encoding="utf-8"))
    action_sheet.parent.mkdir(parents=True, exist_ok=True)
    action_sheet.write_text(
        build_action_sheet(
            metadata=metadata,
            bundle_path=common["bundle_path"],
            manual_checklist_path=common["manual_checklist_path"],
            run_date=run_date,
        ),
        encoding="utf-8",
    )

    artifact = _base_artifact("complete")
    artifact.update(
        {
            **common,
            "submission_ready_for_operator": True,
            "credentialed_submission_attempted": False,
            "honest_verdict": "operator_action_sheet_ready_no_api_attempt",
        }
    )
    return _write_json(output, artifact)
