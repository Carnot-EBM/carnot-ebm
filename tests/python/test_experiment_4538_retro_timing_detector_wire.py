"""Tests for Exp 4538 retro timing-data detector wiring.

Spec refs: REQ-REPORT-4538, SCENARIO-REPORT-4538-RETRO-PATH,
SCENARIO-REPORT-4538-FALLBACK, SCENARIO-REPORT-4538-ARTIFACT.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_4538_retro_timing_detector_wire as runner_mod
from carnot.reporting import retro_timing_detector_wire_4538 as mod
from carnot.reporting.timing_detector_repair_4517 import MilestoneWindow


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def _touch_iso(path: Path, iso_text: str) -> None:
    timestamp = datetime.fromisoformat(iso_text).timestamp()
    os.utime(path, (timestamp, timestamp))


def _write_json(path: Path, payload: dict[str, object] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload or {"honest_verdict": "complete: fixture"}),
        encoding="utf-8",
    )


def _write_changelog(root: Path, *lines: str) -> None:
    path = root / "ops" / "changelog.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_legacy_retro(root: Path, milestone: str, count: int) -> None:
    suffix = milestone.rsplit(".", 1)[1]
    _write_json(
        root / "results" / f"operational_retro_2026_06_{suffix}.json",
        {"experiments_completed": count},
    )


def _fixture_window() -> MilestoneWindow:
    return MilestoneWindow(
        milestone="2026.06.999",
        start_iso="2026-06-21T10:00:00+00:00",
        end_iso="2026-06-21T10:10:00+00:00",
        experiment_id_min=4538001,
        experiment_id_max=4538003,
        changelog_date="2026-06-21",
    )


def test_req_report_4538_spec_anchor_exists() -> None:
    """REQ-REPORT-4538: OpenSpec declares the retro timing-data contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4538" in spec
    assert "SCENARIO-REPORT-4538-RETRO-PATH" in spec
    assert "SCENARIO-REPORT-4538-FALLBACK" in spec
    assert "SCENARIO-REPORT-4538-ARTIFACT" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_4538_retro_path_reports_fixture_count(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4538-RETRO-PATH: retro TIMING DATA reports artifact truth."""

    artifacts = [
        ("experiment_4538001_alpha.json", 60.0, False, "2026-06-21T10:01:00+00:00"),
        ("experiment_4538002_beta.json", 120.0, True, "2026-06-21T10:02:00+00:00"),
        ("experiment_4538003_gamma.json", 30.0, False, "2026-06-21T10:03:00+00:00"),
    ]
    for filename, duration_s, compute_bound, mtime in artifacts:
        path = tmp_path / "results" / filename
        _write_json(
            path,
            {
                "experiment": filename.removesuffix(".json"),
                "duration_s": duration_s,
                "compute_bound": compute_bound,
            },
        )
        _touch_iso(path, mtime)
    _write_changelog(
        tmp_path,
        "- 2026-06-21: fixture alpha; results/experiment_4538001_alpha.json",
        "- 2026-06-21: fixture beta; results/experiment_4538002_beta.json",
        "- 2026-06-21: fixture gamma; results/experiment_4538003_gamma.json",
    )
    _write_legacy_retro(tmp_path, "2026.06.999", 0)

    timing = mod.build_retro_timing_data(tmp_path, _fixture_window())

    assert timing.experiments_completed == 3
    assert timing.reported_in_window_count == 3
    assert timing.on_disk_in_window_count == 3
    assert timing.regression_assert_passed is True
    assert timing.detector_gap_suspected is True
    assert timing.legacy_reported_count == 0
    assert timing.compute_bound_experiments_count == 1
    assert timing.total_wall_time_minutes == 3.5
    assert "Experiments completed: 3" in timing.timing_summary
    assert "detector_gap_suspected=true" in timing.timing_summary


def test_scenario_report_4538_changelog_fallback_reports_touched_artifact(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4538-FALLBACK: changelog fallback repairs a missed mtime."""

    path = tmp_path / "results" / "experiment_4538001_touched_later.json"
    _write_json(path, {"duration_s": 12.0, "compute_bound": False})
    _touch_iso(path, "2026-06-21T11:00:00+00:00")
    _write_changelog(
        tmp_path,
        "- 2026-06-21: touched later; results/experiment_4538001_touched_later.json",
    )
    _write_legacy_retro(tmp_path, "2026.06.999", 0)

    timing = mod.build_retro_timing_data(tmp_path, _fixture_window())

    assert timing.experiments_completed == 1
    assert timing.on_disk_in_window_count == 1
    assert timing.reported_in_window_count == 1
    assert timing.repaired_detection["mtime_count"] == 0
    assert timing.repaired_detection["changelog_count"] == 1
    assert timing.repaired_detection["fallback_used"] is True
    assert timing.regression_assert_passed is True


def test_scenario_report_4538_regression_assert_catches_future_false_zero(tmp_path: Path) -> None:
    """REQ-REPORT-4538: injected count must equal on-disk in-window artifacts."""

    path = tmp_path / "results" / "experiment_4538001_unreported.json"
    _write_json(path, {"duration_s": 1.0, "compute_bound": False})
    _touch_iso(path, "2026-06-21T11:00:00+00:00")
    _write_changelog(tmp_path, "- 2026-06-21: no experiment artifact on this line")
    _write_legacy_retro(tmp_path, "2026.06.999", 0)

    with pytest.raises(mod.DetectorRegressionError) as exc_info:
        mod.build_retro_timing_data(tmp_path, _fixture_window())

    assert "reported in-window count 0 != on-disk in-window artifact count 1" in str(
        exc_info.value
    )


def test_scenario_report_4538_payload_fields_and_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-4538-ARTIFACT: required fields are principle-annotated."""

    payload = mod.build_payload(
        REPO_ROOT,
        tests_added_pass={"command": "fixture-targeted-pytest --no-cov", "passed": True},
    )

    assert payload["honest_verdict"] == "shipped: retro_timing_detector_wired_regression_asserted"
    assert payload["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert payload["retro_path_wired"]["consumer_path"] == mod.RETRO_TIMING_DATA_PATH
    assert payload["retro_path_wired"]["repaired_detector_module"].endswith(
        "timing_detector_repair_4517"
    )
    assert payload["regression_assert_added"]["assert_passed"] is True
    assert payload["regression_assert_added"]["reported_in_window_count"] == payload[
        "regression_assert_added"
    ]["on_disk_in_window_count"]
    assert payload["retro_timing_data"]["experiments_completed"] == 10
    assert payload["retro_timing_data"]["detector_gap_suspected"] is True
    assert len(payload["cited_upstream_artifacts"]) == 3
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in payload
        assert field in payload["field_principles"]

    mod.validate_artifact(payload)
    output_path = mod.write_payload(tmp_path, payload, started_s=100.0, now_s=lambda: 100.5)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["duration_s"] == 0.5
    assert persisted["compute_bound"] is False
    mod.validate_artifact(persisted)


def test_req_report_4538_experiment_entrypoint_prints_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4538: requested direct script command has a thin entrypoint."""

    output_path = tmp_path / mod.OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"honest_verdict": "shipped: ok"}\n', encoding="utf-8")

    monkeypatch.setattr(runner_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner_mod, "run", lambda root: output_path)

    assert runner_mod.main() == 0
    assert '"honest_verdict": "shipped: ok"' in capsys.readouterr().out
