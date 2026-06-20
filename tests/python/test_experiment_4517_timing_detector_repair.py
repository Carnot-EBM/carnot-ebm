"""Tests for Exp 4517 milestone timing-detector repair.

Spec refs: REQ-REPORT-4517, SCENARIO-REPORT-4517-FALSE-ZERO,
SCENARIO-REPORT-4517-DISAGREEMENT, SCENARIO-REPORT-4517-WRITE-STAMP.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

from carnot import experiment_4517_timing_detector_repair as runner_mod
from carnot.reporting import timing_detector_repair_4517 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _touch_iso(path: Path, iso_text: str) -> None:
    timestamp = datetime.fromisoformat(iso_text).timestamp()
    os.utime(path, (timestamp, timestamp))


def _write_json(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"honest_verdict": "complete: fixture"}), encoding="utf-8")


def test_req_report_4517_spec_anchor_exists() -> None:
    """REQ-REPORT-4517: OpenSpec declares the mtime/changelog detector contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4517" in spec
    assert "SCENARIO-REPORT-4517-FALSE-ZERO" in spec
    assert "SCENARIO-REPORT-4517-DISAGREEMENT" in spec
    assert "SCENARIO-REPORT-4517-WRITE-STAMP" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_4517_disagreement_uses_union_and_flags_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4517-DISAGREEMENT: source mismatch still returns true count."""

    results_dir = tmp_path / "results"
    first = results_dir / "experiment_4517001_mtime_only.json"
    shared = results_dir / "experiment_4517002_shared.json"
    changelog_only = "results/experiment_4517003_changelog_only.json"
    _write_json(first)
    _write_json(shared)
    _touch_iso(first, "2026-06-20T10:00:00+00:00")
    _touch_iso(shared, "2026-06-20T10:01:00+00:00")

    changelog_text = "\n".join(
        [
            "- 2026-06-20: fixture shared; results/experiment_4517002_shared.json",
            f"- 2026-06-20: fixture changelog only; {changelog_only}",
        ]
    )
    window = mod.MilestoneWindow(
        milestone="2026.06.999",
        start_iso="2026-06-20T09:59:00+00:00",
        end_iso="2026-06-20T10:02:00+00:00",
        experiment_id_min=4517001,
        experiment_id_max=4517003,
        changelog_date="2026-06-20",
    )

    detection = mod.detect_milestone(tmp_path, window, changelog_text=changelog_text)

    assert detection.mtime_count == 2
    assert detection.changelog_count == 2
    assert detection.corrected_count == 3
    assert detection.detector_gap_suspected is True
    assert detection.mtime_only_paths == ("results/experiment_4517001_mtime_only.json",)
    assert detection.changelog_only_paths == ("results/experiment_4517003_changelog_only.json",)


def test_scenario_report_4517_changelog_fallback_recovers_touched_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4517-FALSE-ZERO: changelog recovers an artifact touched outside mtime window."""

    path = tmp_path / "results/experiment_4517100_touched_later.json"
    _write_json(path)
    _touch_iso(path, "2026-06-20T12:00:00+00:00")
    changelog_text = "- 2026-06-20: fixture; results/experiment_4517100_touched_later.json"
    window = mod.MilestoneWindow(
        milestone="2026.06.998",
        start_iso="2026-06-20T09:00:00+00:00",
        end_iso="2026-06-20T10:00:00+00:00",
        experiment_id_min=4517100,
        experiment_id_max=4517100,
        changelog_date="2026-06-20",
    )

    detection = mod.detect_milestone(tmp_path, window, changelog_text=changelog_text)

    assert detection.mtime_count == 0
    assert detection.changelog_count == 1
    assert detection.corrected_count == 1
    assert detection.fallback_used is True
    assert detection.detector_gap_suspected is True


def test_scenario_report_4517_real_false_zero_repaired_counts() -> None:
    """SCENARIO-REPORT-4517-FALSE-ZERO: real .415/.416 artifacts repair to 10 each."""

    retro_415 = json.loads((REPO_ROOT / "results/operational_retro_2026_06_415.json").read_text())
    retro_416 = json.loads((REPO_ROOT / "results/operational_retro_2026_06_416.json").read_text())
    assert retro_415["experiments_completed"] == 0
    assert retro_416["experiments_completed"] == 0

    payload = mod.build_payload(REPO_ROOT, tests_added_pass=True)

    assert payload["detector_true_count_415"] == 10
    assert payload["detector_true_count_416"] == 10
    assert payload["milestone_detections"]["2026.06.415"]["corrected_count"] == 10
    assert payload["milestone_detections"]["2026.06.416"]["corrected_count"] == 10
    assert "results/experiment_4494_adapter_deepen_l2.json" in payload["milestone_detections"]["2026.06.415"]["corrected_paths"]
    assert "results/experiment_4508_arc_affordance_sota_416.json" in payload["milestone_detections"]["2026.06.416"]["corrected_paths"]
    assert payload["milestone_detections"]["2026.06.415"]["legacy_reported_count"] == 0
    assert payload["milestone_detections"]["2026.06.416"]["legacy_reported_count"] == 0
    assert payload["milestone_detections"]["2026.06.416"]["detector_gap_suspected"] is True


def test_scenario_report_4517_write_helper_stamps_duration_and_compute_bound(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4517-WRITE-STAMP: persisted artifact gets write-time fields."""

    payload = {
        "schema": mod.SCHEMA,
        "experiment": "experiment_4517_timing_detector_repair",
        "honest_verdict": "shipped: timing_detector_repaired_true_counts",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "detector_true_count_415": 10,
        "detector_true_count_416": 10,
        "tests_added_pass": True,
        "preconditions_checked": {"retro_glob_count": 2},
        "field_principles": mod.FIELD_PRINCIPLES,
    }

    output_path = mod.write_payload(tmp_path, payload, started_s=100.0, now_s=lambda: 100.25)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted["duration_s"] == 0.25
    assert persisted["compute_bound"] is False
    assert payload.get("duration_s") is None
    mod.validate_artifact(persisted)


def test_req_report_4517_payload_principles_and_validation() -> None:
    """REQ-REPORT-4517: required artifact fields are terminal and principle-annotated."""

    payload = mod.build_payload(REPO_ROOT, tests_added_pass=True)

    assert payload["honest_verdict"] == "shipped: timing_detector_repaired_true_counts"
    assert payload["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert payload["tests_added_pass"] is True
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]

    bad = dict(payload)
    bad["detector_true_count_415"] = 9
    try:
        mod.validate_artifact(bad)
    except ValueError as exc:
        assert "detector_true_count_415" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("validate_artifact accepted a false .415 count")


def test_req_report_4517_blocked_precondition_payload(tmp_path: Path) -> None:
    """REQ-REPORT-4517: missing preconditions block without fabricating true counts."""

    payload = mod.build_payload(tmp_path, tests_added_pass=False)

    assert payload["honest_verdict"].startswith("blocked_")
    assert payload["detector_true_count_415"] == 0
    assert payload["detector_true_count_416"] == 0
    assert payload["tests_added_pass"] is False
    assert payload["preconditions_checked"]["results_dir_exists"] is False


def test_req_report_4517_experiment_entrypoint_prints_payload(tmp_path: Path, monkeypatch, capsys) -> None:
    """REQ-REPORT-4517: requested direct script command has a thin entrypoint."""

    output_path = tmp_path / mod.OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"honest_verdict": "shipped: ok"}\n', encoding="utf-8")

    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_mod, "run", lambda root: output_path)

    assert runner_mod.main() == 0
    assert '"honest_verdict": "shipped: ok"' in capsys.readouterr().out
