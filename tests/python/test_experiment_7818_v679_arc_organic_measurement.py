"""Scored public ARC measurement checks for REQ-REPORT-7818."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_7818_v679_arc_organic_measurement as measure


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "results/raw/experiment_7818_v679_arc_organic_measurement/validation_command_manifest.json"
)
CLI = "scripts/experiments/experiment_7818_v679_arc_organic_measurement.py"


def test_scenario_report_7818_dispatch_real_cli(tmp_path):
    """SCENARIO-REPORT-7818-DISPATCH: capture every declared child through the real CLI."""
    output = tmp_path / "dispatch.json"
    result = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-u", CLI, "--record-dispatch", str(output)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": "python:."},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    expected = json.loads(MANIFEST.read_text())["commands"]
    assert json.loads(output.read_text()) == [
        {key: row[key] for key in ("name", "argv", "classification")} for row in expected
    ]
    assert len(expected) == 66
    assert expected[-1]["name"] == "cold_replay"
    assert expected[-2]["name"] == "strict_row_lint"
    assert [r["name"] for r in expected if r["classification"] == "diagnostic"] == [
        "repository_health_full_python_suite"
    ]
    assert not any(
        x == "scripts/experiments/experiment_7803_v678_arc_runner_qualification.py"
        for r in expected
        for x in r["argv"]
    )
    assert measure.command_plan(MANIFEST) == json.loads(output.read_text())


def test_scenario_report_7818_dispatch_rejects_appended_child(tmp_path):
    """SCENARIO-REPORT-7818-DISPATCH: no appended command can run."""
    data = json.loads(MANIFEST.read_text())
    data["commands"].append(
        {"name": "surprise", "argv": ["true"], "classification": "required", "timeout_s": 1}
    )
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manifest_mutation"):
        measure.command_plan(bad)


def test_scenario_report_7818_dispatch_order_and_budget():
    """SCENARIO-REPORT-7818-DISPATCH: six randomized arms finish per game."""
    plan = measure.command_plan(MANIFEST)
    episodes = plan[:48]
    assert len({r["name"] for r in episodes}) == 48
    games = [r["name"].split("_")[1] for r in episodes]
    assert all(games[i : i + 6] == [games[i]] * 6 for i in range(0, 48, 6))
    for i in range(0, 48, 6):
        assert {r["name"].split("_")[2] for r in episodes[i : i + 3]} == {"67815"}
        assert {r["name"].split("_")[2] for r in episodes[i + 3 : i + 6]} == {"67816"}
        assert {r["name"].split("_")[3] for r in episodes[i : i + 3]} == {"off", "total", "organic"}
        assert {r["name"].split("_")[3] for r in episodes[i + 3 : i + 6]} == {
            "off",
            "total",
            "organic",
        }
        assert all(r["argv"][7:] == ["--date", "20260928"] for r in episodes[i : i + 6])


def test_scenario_report_7818_custody_retry_parent_and_mutation(tmp_path):
    """SCENARIO-REPORT-7818-DISPATCH: sealed bytes identify one attempt only."""
    parent = tmp_path / "missing" / "logs"
    with pytest.raises(FileNotFoundError):
        measure.seal_log(parent, "unit", 1, b"first")
    parent.mkdir(parents=True)
    first = measure.seal_log(parent, "unit", 1, b"first")
    later = measure.seal_log(parent, "unit", 2, b"second")
    assert first != later
    assert measure.verify_log(first, hashlib.sha256(b"first").hexdigest())
    assert measure.verify_log(later, hashlib.sha256(b"second").hexdigest())
    first.write_bytes(b"firsu")
    assert not measure.verify_log(first, hashlib.sha256(b"first").hexdigest())
    assert later.read_bytes() == b"second"
    with pytest.raises(FileExistsError):
        measure.seal_log(parent, "unit", 2, b"second")


def test_scenario_report_7818_measurement_sdk_formula_and_reset():
    """SCENARIO-REPORT-7818-MEASUREMENT: level score uses SDK weights and human actions."""
    actions = [
        {"action": "RESET", "actual_observation": {"level": 0}},
        {"action": "ACTION1", "actual_observation": {"level": 0}},
        {"action": "ACTION1", "actual_observation": {"level": 1}},
        {"action": "ACTION1", "actual_observation": {"level": 1}},
        {"action": "ACTION1", "actual_observation": {"level": 2}},
    ]
    row = measure.score_episode(actions, [2, 2, 3])
    assert row["first_level_up_actions"] == 3
    assert row["total_actions"] == 5
    assert row["reset_count"] == 1
    assert row["level_actions_charged"][:2] == [3, 2]
    assert row["level_actions_uncharged"][:2] == [2, 2]
    assert row["score_charged"] < row["score_uncharged"]
    assert row["score_uncharged"] == pytest.approx(50.0)


def test_scenario_report_7818_measurement_incomplete_forbids_benefit():
    """SCENARIO-REPORT-7818-MEASUREMENT: missing pairs cannot be called a win."""
    rows = [
        {
            "game": "cd82",
            "seed": 67815,
            "arm": arm,
            "status": "completed",
            "score_charged": 0.0,
            "score_uncharged": 0.0,
            "peak_level": 0,
            "total_actions": 10,
        }
        for arm in ("off", "total", "organic")
    ]
    result = measure.reduce_rows(rows)
    assert result["independent_n"] == 0
    assert result["organic_benefit_score"] == 0
    assert result["complete_pairs"] is False
