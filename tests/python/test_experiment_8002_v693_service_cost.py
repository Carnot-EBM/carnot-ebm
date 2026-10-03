"""REQ-REPORT-8002: private CLI publication and frozen validation paths."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_8002_v693_service_cost as e
from carnot.reporting import service_cost_8002 as s
from carnot.reporting.current_work_receipt import atomic_json


def test_real_cli(tmp_path):
    """SCENARIO-REPORT-8002-CLI: actual success, blocked and cold routes."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, s.fixture())
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    prefix = [str(e.ROOT / ".venv/bin/python"), str(e.ROOT / e.OWNED[-1]), "--date", "20261002"]
    output = tmp_path / "success" / (e.NAME + ".json")
    commands = [
        ["--fixture-input", str(fixture), "--validation-worker", "--output", str(output)],
        ["--cold-replay", str(output)],
        [
            "--root",
            str(tmp_path / "absent"),
            "--validation-worker",
            "--output",
            str(tmp_path / "blocked" / output.name),
        ],
        ["--date", "invalid"],
    ]
    for index, command in enumerate(commands):
        child = subprocess.run(
            prefix + command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        assert child.returncode == (1 if index == 3 else 0), child.stdout + child.stderr
    value = json.loads(output.read_text())
    storage = Path(value["environmental_observations"]["filesystem"])
    assert not storage.is_relative_to(Path("/tmp"))
    last_write = next(r for r in reversed(value["rows"]) if r["case"] != "no_write")
    assert (storage / "state.json").stat().st_size == last_write["bytes_written"]
    assert value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert (
        json.loads((tmp_path / "blocked" / output.name).read_text())["verdict_class"] == "blocked"
    )
    value["rows"][0]["wall_ns"] = 0
    atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1


def test_owned_validation_and_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8002-CLI: owned failures retain readiness zero."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, s.fixture())
    monkeypatch.setattr(e, "execute", lambda *a: [dict(required=True, passed=False)])
    monkeypatch.setattr(
        e, "coverage_counts", lambda *a: dict(owned=dict(num_statements=1, covered_lines=1))
    )
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(fixture), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=False))
    with pytest.raises(ValueError, match="candidate_rejected"):
        e.publish(tmp_path / "rejected" / output.name, e.base(s.fixture()))
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="primary_resolution"):
        e.publish(tmp_path / "reader" / output.name, e.base(s.fixture()))


def test_failed_candidate_and_real_counts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8002-CLI: real counts and failed candidate remain visible."""
    from carnot.reporting.service_validation_8002 import coverage_counts

    atomic_json(
        tmp_path / "coverage.json",
        dict(files={"owned": dict(summary=dict(num_statements=1, covered_lines=1))}),
    )
    assert coverage_counts(tmp_path)["owned"]["num_statements"] == 1
    assert e.main(["--date", "invalid"]) == 1
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=False))
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--validation-worker",
                "--output",
                str(tmp_path / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_valid_owned_cli_and_empty_coverage(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8002-CLI: completeness requires a nonempty denominator."""
    monkeypatch.setattr(s, "authenticate", lambda *a: s.fixture())
    monkeypatch.setattr(e, "execute", lambda *a: [dict(required=True, passed=True)])
    monkeypatch.setattr(
        e, "coverage_counts", lambda *a: dict(owned=dict(num_statements=1, covered_lines=1))
    )
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "null"
    monkeypatch.setattr(e, "coverage_counts", lambda *a: {})
    output = tmp_path / "invalid" / output.name
    assert e.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["service_measurement_ready_score"] == 0
