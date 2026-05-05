"""Tests for the Exp 1307 arXiv v10 hold/receipt artifact."""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.arxiv_hold_receipt_v2 import (
    REQUIRED_FIELDS,
    detect_operator_hold,
    find_local_arxiv_receipt,
    run,
    write_in_progress_artifact,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_write_in_progress_artifact_has_required_fields_req_publish_015(tmp_path: Path) -> None:
    """REQ-PUBLISH-015: the runner writes a durable in-progress skeleton first."""

    out_path = tmp_path / "results" / "experiment_1307_arxiv_v10_hold_receipt_v2.json"

    artifact = write_in_progress_artifact(out_path=out_path)

    assert out_path.exists()
    assert artifact["status"] == "in_progress"
    assert set(REQUIRED_FIELDS).issubset(artifact)
    assert artifact["credentialed_submission_attempted"] is False


def test_find_local_receipt_requires_actual_submission_signal_req_publish_015(tmp_path: Path) -> None:
    """REQ-PUBLISH-015: blocker artifacts mentioning receipts are not receipts."""

    results = tmp_path / "results"
    (results / "000_list_payload.json").parent.mkdir(parents=True, exist_ok=True)
    (results / "000_list_payload.json").write_text("[]", encoding="utf-8")
    _write_json(
        results / "experiment_1294_arxiv_v10_submission_receipt_or_blocker.json",
        {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"},
    )
    _write_json(
        results / "arxiv_submission_receipt_20260505.json",
        {"arxiv_submission_id": "submit/1234567", "arxiv_submitted": True},
    )

    receipt = find_local_arxiv_receipt(results_dir=results)

    assert receipt is not None
    assert receipt["path"] == "results/arxiv_submission_receipt_20260505.json"


def test_no_receipt_leaves_publication_on_operator_hold_scenario_publish_015(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-015: no local receipt plus active hold records terminal hold."""

    project = tmp_path
    known_issues = project / "ops" / "known-issues.md"
    known_issues.parent.mkdir(parents=True)
    known_issues.write_text(
        "## PUBLICATION HOLD\n\narXiv submission is ON HOLD until operator review.\n\n## NEXT\n",
        encoding="utf-8",
    )
    _write_json(
        project / "results" / "experiment_1294_arxiv_v10_submission_receipt_or_blocker.json",
        {"status": "blocked", "gate_check_summary": "prior failure metadata missing"},
    )

    artifact = run(project_root=project)

    assert artifact["status"] == "complete"
    assert artifact["publication_state"] == "operator_hold"
    assert artifact["arxiv_receipt_present"] is False
    assert artifact["operator_hold_active"] is True
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["blocker"] == "operator_publication_hold_active_no_local_receipt"
    assert artifact["honest_verdict"] == "operator_hold_active_no_local_arxiv_receipt"


def test_existing_local_receipt_marks_submitted_req_publish_015(tmp_path: Path) -> None:
    """REQ-PUBLISH-015: a checked-in receipt is enough to mark submitted."""

    project = tmp_path
    known_issues = project / "ops" / "known-issues.md"
    known_issues.parent.mkdir(parents=True)
    known_issues.write_text("## PUBLICATION HOLD\narXiv submission is ON HOLD.\n", encoding="utf-8")
    _write_json(
        project / "results" / "arxiv_submission_receipt_20260505.json",
        {"arxiv_submitted": True, "arxiv_submission_id": "submit/7654321"},
    )

    artifact = run(project_root=project)

    assert artifact["publication_state"] == "submitted"
    assert artifact["arxiv_receipt_present"] is True
    assert artifact["operator_hold_active"] is False
    assert artifact["credentialed_submission_attempted"] is False
    assert artifact["blocker"] is None
    assert artifact["honest_verdict"] == "local_arxiv_receipt_present_submitted"


def test_local_receipt_detection_accepts_structured_receipt_fields_req_publish_015(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-015: prior terminal receipt schemas count as local receipts."""

    results = tmp_path / "results"
    _write_json(
        results / "experiment_1306_prior_hold_receipt.json",
        {"arxiv_receipt_present": True, "publication_state": "submitted"},
    )

    receipt = find_local_arxiv_receipt(results_dir=results)

    assert receipt is not None
    assert receipt["path"] == "results/experiment_1306_prior_hold_receipt.json"


def test_local_receipt_detection_accepts_submission_id_and_receipt_req_publish_015(
    tmp_path: Path,
) -> None:
    """REQ-PUBLISH-015: checked-in submission IDs or receipt bodies are enough."""

    results = tmp_path / "results_id"
    _write_json(results / "experiment_1.json", {"arxiv_submission_id": "submit/1"})
    receipt = find_local_arxiv_receipt(results_dir=results)

    assert receipt is not None
    assert receipt["path"] == "results_id/experiment_1.json"

    results = tmp_path / "results_receipt"
    _write_json(
        results / "experiment_2.json",
        {"arxiv_submitted": True, "arxiv_receipt": {"submission_id": "submit/2"}},
    )
    receipt = find_local_arxiv_receipt(results_dir=results)

    assert receipt is not None
    assert receipt["path"] == "results_receipt/experiment_2.json"


def test_no_hold_uses_exact_prior_blocker_req_publish_015(tmp_path: Path) -> None:
    """REQ-PUBLISH-015: without a hold, the exact prior blocker is preserved."""

    project = tmp_path
    known_issues = project / "ops" / "known-issues.md"
    known_issues.parent.mkdir(parents=True)
    known_issues.write_text("No active publication hold here.\n", encoding="utf-8")
    _write_json(
        project / "results" / "experiment_1294_arxiv_v10_submission_receipt_or_blocker.json",
        {"status": "blocked", "gate_check_summary": "prior failure metadata missing"},
    )
    _write_json(
        project / "results" / "experiment_1295_milestone_retro_100.json",
        {
            "status": "complete",
            "publication_state": {"external_blocker": "retro publication blocker"},
        },
    )

    artifact = run(project_root=project)

    assert artifact["publication_state"] == "blocked"
    assert artifact["operator_hold_active"] is False
    assert artifact["blocker"] == "prior failure metadata missing"
    assert artifact["honest_verdict"] == "blocked_no_local_arxiv_receipt"


def test_detect_operator_hold_ignores_resolved_hold_heading_req_publish_015() -> None:
    """REQ-PUBLISH-015: resolved historical hold text does not keep the hold active."""

    text = "## ~~PUBLICATION HOLD~~ (RESOLVED)\n\narXiv submission is ON HOLD in old notes."

    assert detect_operator_hold(text) is False
