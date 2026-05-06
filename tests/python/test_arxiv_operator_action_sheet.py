"""Tests for the Exp 1412 arXiv operator action sheet.

Spec traces: REQ-PUBLISH-020, SCENARIO-PUBLISH-022.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import arxiv_operator_action_sheet as exp1412


CHECKLIST_TEXT = """# Carnot arXiv Manual Submission Checklist

Run date: 2026-05-06

Upload URL: https://arxiv.org/submit

Ready bundle:
- Relative path: `results/arxiv_bundle_v11.tar.gz`
- Absolute path: `/repo/results/arxiv_bundle_v11.tar.gz`
- Verified non-empty source archive: yes

## Pre-Filled Metadata

Title:

```text
Carnot: An Architectural Framework for Mapping the Empirical Bounds of LLM Verification
```

Authors:

```text
Ian Blenke <ian@blenke.com>
```

Primary category:

```text
cs.LG
```

License:

```text
CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)
```

Abstract:

```text
Long abstract body that should not be copied into the terse action sheet.
```
"""


def _write_checklist(root: Path) -> None:
    checklist = root / "docs" / "arxiv-manual-submission-checklist.md"
    checklist.parent.mkdir(parents=True, exist_ok=True)
    checklist.write_text(CHECKLIST_TEXT, encoding="utf-8")


def _write_ready_bundle(root: Path) -> Path:
    bundle = root / "results" / "arxiv_bundle_v11.tar.gz"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(b"ready source archive")
    return bundle


def test_write_in_progress_artifact_has_required_fields_req_publish_020(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-020: the runner writes the required durable placeholder first."""

    out_path = tmp_path / "results" / "experiment_1412_arxiv_operator_action_sheet_v3.json"

    artifact = exp1412.write_in_progress_artifact(out_path)

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert exp1412.REQUIRED_ARTIFACT_FIELDS <= set(written)
    assert written["credentialed_submission_attempted"] is False


def test_extract_checklist_metadata_req_publish_020() -> None:
    """REQ-PUBLISH-020: the operator sheet uses the existing manual checklist metadata."""

    metadata = exp1412.extract_checklist_metadata(CHECKLIST_TEXT)

    assert metadata == {
        "upload_url": "https://arxiv.org/submit",
        "title": (
            "Carnot: An Architectural Framework for Mapping the Empirical Bounds of "
            "LLM Verification"
        ),
        "author": "Ian Blenke <ian@blenke.com>",
        "primary_category": "cs.LG",
        "license": "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "abstract_present": True,
    }


def test_missing_checklist_metadata_is_rejected_req_publish_020() -> None:
    """REQ-PUBLISH-020: missing checklist metadata cannot be silently condensed."""

    with pytest.raises(ValueError, match="Authors"):
        exp1412.extract_checklist_metadata(CHECKLIST_TEXT.replace("Authors:", ""))


def test_run_writes_action_sheet_and_complete_artifact_scenario_publish_022(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-022: a non-empty bundle produces the browser action sheet."""

    bundle = _write_ready_bundle(tmp_path)
    _write_checklist(tmp_path)

    artifact = exp1412.run(project_root=tmp_path, run_date="20260506")

    written = json.loads(
        (tmp_path / "results" / "experiment_1412_arxiv_operator_action_sheet_v3.json").read_text(
            encoding="utf-8"
        )
    )
    sheet = tmp_path / artifact["operator_action_sheet_path"]
    sheet_text = sheet.read_text(encoding="utf-8")

    assert written == artifact
    assert artifact["status"] == "complete"
    assert artifact["bundle_exists"] is True
    assert artifact["bundle_size_bytes"] == bundle.stat().st_size
    assert artifact["submission_ready_for_operator"] is True
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["honest_verdict"] == "operator_action_sheet_ready_no_api_attempt"
    assert "https://arxiv.org/submit" in sheet_text
    assert "results/arxiv_bundle_v11.tar.gz" in sheet_text
    assert "cs.LG" in sheet_text
    assert "CC-BY-4.0" in sheet_text
    assert "Ian Blenke <ian@blenke.com>" in sheet_text
    assert "Carnot: An Architectural Framework" in sheet_text
    assert "Abstract: paste the Abstract block from" in sheet_text
    assert "Long abstract body" not in sheet_text


def test_missing_bundle_blocks_without_action_sheet_req_publish_020(tmp_path: Path) -> None:
    """REQ-PUBLISH-020: a missing bundle blocks without attempting submission."""

    _write_checklist(tmp_path)

    artifact = exp1412.run(project_root=tmp_path, run_date="20260506")

    assert artifact["status"] == "blocked"
    assert artifact["bundle_exists"] is False
    assert artifact["bundle_size_bytes"] == 0
    assert artifact["submission_ready_for_operator"] is False
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["honest_verdict"] == "blocked_bundle_missing_or_empty"
    assert not (tmp_path / artifact["operator_action_sheet_path"]).exists()
