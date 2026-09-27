"""REQ-REPORT-7763 and REQ-ARC-WMTE-7763 qualification regression tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from carnot.experiment_7708_v671_arc_generalization_runner import _FixtureArcade
from carnot.experiment_7763_v675_arc_runner_qualification import (
    ARMS,
    GAMES,
    SEEDS,
    cold_reduce,
    run_probe,
    schedule_rows,
)


def test_scenario_report_7763_custody_rows_and_counts() -> None:
    """SCENARIO-REPORT-7763-CUSTODY: all ordered units survive a cold reduction."""
    rows = schedule_rows()
    assert len(rows) == 48
    assert [row[k] for k in ("game", "seed", "arm") for row in rows[:1]] == [
        GAMES[0],
        SEEDS[0],
        ARMS[0],
    ]
    assert rows[-1]["episode_id"] == "sc25:67502:organic"
    assert cold_reduce(rows, []) == {"intended": 48, "started": 0, "completed": 0, "actions": 0}
    with pytest.raises(ValueError, match="schedule_changed"):
        cold_reduce(rows[:-1], [])
    with pytest.raises(ValueError, match="duplicate_probe"):
        cold_reduce(rows, [{"episode_id": rows[0]["episode_id"]}] * 2)


def test_scenario_arc_wmte_7763_provenance_negative_controls() -> None:
    """SCENARIO-ARC-WMTE-7763-PROVENANCE: replay cannot inflate organic seen."""
    rows = schedule_rows()
    probe = {
        "episode_id": rows[0]["episode_id"],
        "actions": [{"action": "RESET"}],
        "actions_charged": 1,
        "error": None,
        "counter_event_rows": [
            {
                "event": "observation",
                "provenance": "organic",
                "seen": 1,
                "organic_seen": 1,
                "replay_seen": 0,
                "reset_seen": 0,
            },
            {
                "event": "observation",
                "provenance": "replay",
                "seen": 2,
                "organic_seen": 1,
                "replay_seen": 1,
                "reset_seen": 0,
            },
        ],
    }
    summary = cold_reduce(rows, [probe])
    assert summary == {"intended": 48, "started": 1, "completed": 1, "actions": 1}
    assert (
        cold_reduce(rows, [dict(probe, counter_event_rows=[{"event": "selection"}])])["started"]
        == 1
    )
    bad = dict(probe, actions_charged=2)
    with pytest.raises(ValueError, match="action_count"):
        cold_reduce(rows, [bad])
    wrong = dict(
        probe,
        counter_event_rows=[
            *probe["counter_event_rows"][:1],
            dict(probe["counter_event_rows"][1], organic_seen=2),
        ],
    )
    with pytest.raises(ValueError, match="counter_provenance"):
        cold_reduce(rows, [wrong])
    unknown = dict(
        probe, counter_event_rows=[dict(probe["counter_event_rows"][0], provenance="unknown")]
    )
    with pytest.raises(ValueError, match="counter_provenance"):
        cold_reduce(rows, [unknown])


def test_scenario_report_7763_validation_real_fixture_transport() -> None:
    """SCENARIO-REPORT-7763-VALIDATION: the scored E3 wrapper crosses SDK transitions."""
    row = run_probe("fixture", 67501, "off", _FixtureArcade(), 3)
    assert row["policy_entry"]["policy_class"] == "E3AgentPolicy"
    assert row["counts"]["sdk_transitions"] > 0
    assert row["max_seconds"] == 75
    assert row["new_solve_credit"] is False


def test_scenario_report_7763_validation_private_basetemp_child(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7763-VALIDATION: pytest's nested basetemp has a real parent."""
    parent = tmp_path / "nested"
    parent.mkdir(parents=True)
    test_file = tmp_path / "test_child.py"
    test_file.write_text("def test_child():\n    assert True\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            f"--basetemp={parent / 'child'}",
            str(test_file),
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
