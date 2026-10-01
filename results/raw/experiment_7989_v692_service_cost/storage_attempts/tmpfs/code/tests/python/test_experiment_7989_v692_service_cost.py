"""REQ-REPORT-7989: real private service CLI and publication routes."""

import json
import os
import subprocess

import pytest

from carnot import experiment_7989_v692_service_cost as e
from carnot.reporting import service_cost_7989 as s
from carnot.reporting.current_work_receipt import atomic_json


def test_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7989-CLI: replay, request, block and invalid arguments."""
    inputs, heads = s.fixture()
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, dict(requests=inputs, heads=heads))
    output = tmp_path / f"{e.NAME}.json"
    assert e.main(["--date", "bad"]) == 1
    assert e.main(["--fixture-e2e", str(tmp_path / "absent")]) == 1
    assert e.main(["--fixture-e2e", str(fixture), "--output", str(output)]) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--terminal-recheck", str(output)]) == 0
    blocked = tmp_path / "blocked" / output.name
    assert (
        e.main(
            ["--data-root", str(tmp_path / "absent"), "--skip-validation", "--output", str(blocked)]
        )
        == 0
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=False))
    assert e.main(["--terminal-recheck", str(output)]) == 1
    with pytest.raises(ValueError, match="candidate_rejected"):
        e.publish(tmp_path / "reject" / output.name, e.base({}))
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=True))
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="primary_resolution"):
        e.publish(tmp_path / "readfail" / output.name, e.base({}))


def test_validation_route(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7989-GATES: required failures remain disqualified."""

    def exited(manifest, raw, scratch):
        atomic_json(
            scratch / "coverage.json",
            dict(files={"owned.py": dict(summary=dict(num_statements=1, covered_lines=1))}),
        )
        (scratch / ".coverage.exited").write_bytes(b"archive")
        return [dict(required=True, passed=False)]

    inputs, heads = s.fixture()
    monkeypatch.setattr(s, "authenticate", lambda root: dict(requests=inputs, heads=heads))
    monkeypatch.setattr(e.prior, "execute_commands", exited)
    output = tmp_path / f"{e.NAME}.json"
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["service_measurement_ready_score"] == 0


def test_real_cli(tmp_path):
    """SCENARIO-REPORT-7989-CLI: script bootstraps outside the checkout."""
    inputs, heads = s.fixture()
    atomic_json(tmp_path / "request.json", inputs[0])
    atomic_json(tmp_path / "heads.json", heads)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [
        str(e.ROOT / ".venv/bin/python"),
        str(e.ROOT / e.OWNED[1]),
        "--request",
        str(tmp_path / "request.json"),
        "--heads",
        str(tmp_path / "heads.json"),
        "--output",
        str(tmp_path / "response.json"),
    ]
    result = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "response.json").read_text())["verified"] is False
