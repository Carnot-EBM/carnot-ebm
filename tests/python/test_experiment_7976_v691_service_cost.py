"""REQ-REPORT-7976: actual private CLI, blocking and cold publication routes."""

import json
from pathlib import Path
import os
import subprocess

import pytest

from carnot import experiment_7976_v691_service_cost as e
from carnot.reporting import service_cost_7976 as s
from carnot.reporting.current_work_receipt import atomic_json


def test_main_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7976-3: owned expected failures preserve safe terminal state."""
    fixture = tmp_path / "fixture.json"
    atomic_json(fixture, dict(requests=s.fixture()[0], heads=s.fixture()[1]))
    output = tmp_path / "success" / f"{e.NAME}.json"
    assert e.main(["--date", "bad"]) == 1
    assert e.main(["--fixture-e2e", str(tmp_path / "missing")]) == 1
    assert e.main(["--fixture-e2e", str(fixture), "--output", str(output)]) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--terminal-recheck", str(output)]) == 0
    blocked = tmp_path / "blocked" / f"{e.NAME}.json"
    assert (
        e.main(
            [
                "--data-root",
                str(tmp_path / "missing"),
                "--output",
                str(blocked),
                "--skip-validation",
            ]
        )
        == 0
    )
    assert (
        json.loads(blocked.read_text())["honest_verdict"] == "complete_blocked_no_qualified_service"
    )
    value = json.loads(output.read_text())
    value["rows"][0]["p50_s"] = 100
    atomic_json(tmp_path / "drift.json", value)
    assert e.main(["--cold-replay", str(tmp_path / "drift.json")]) == 1
    p = tmp_path / "request.json"
    h = tmp_path / "heads.json"
    atomic_json(p, s.fixture()[0][0])
    h.write_text(json.dumps(s.fixture()[1]))
    assert (
        e.main(["--request", str(p), "--heads", str(h), "--output", str(tmp_path / "reply.json")])
        == 0
    )
    bad = e.base({})
    e.apply_checks(bad, [dict(required=True, passed=False)])
    assert bad["verdict_class"] == "disqualified"
    assert bad["service_measurement_ready_score"] == 0
    e.apply_checks(bad, [dict(required=False, passed=False)])
    assert bad["repository_health"]["status"] == "degraded_open"
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=False, receipts=[]))
    with pytest.raises(ValueError, match="candidate_rejected"):
        e.publish(tmp_path / "reject" / f"{e.NAME}.json", e.base({}))


def test_actual_request_cli(tmp_path):
    """SCENARIO-REPORT-7976-3: execute script without repository PYTHONPATH."""
    path = tmp_path / "request.json"
    heads = tmp_path / "heads.json"
    atomic_json(path, s.fixture()[0][0])
    heads.write_text(json.dumps(s.fixture()[1]))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    run = subprocess.run(
        [
            str(e.ROOT / ".venv/bin/python"),
            str(e.ROOT / e.OWNED[1]),
            "--request",
            str(path),
            "--heads",
            str(heads),
            "--output",
            str(tmp_path / "response.json"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads((tmp_path / "response.json").read_text())["action"] in (
        "accept",
        "reject",
        "escalate",
    )


def test_manifest(tmp_path):
    """SCENARIO-REPORT-7976-3: freeze private commands before measurement."""
    manifest = e.freeze_commands(tmp_path / "raw", tmp_path / "scratch")
    assert manifest["coverage_includes"] == e.INCLUDE
    assert all(c["deadline_s"] <= 300 for c in manifest["commands"])


def test_owned_validation_branch(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7976-3: owned receipts archive only exited child evidence."""

    def exited(manifest, raw, scratch):
        atomic_json(
            scratch / "coverage.json",
            dict(files={"owned.py": dict(summary=dict(covered_lines=1, num_statements=1))}),
        )
        (scratch / ".coverage.exited").write_bytes(b"archive")
        return [dict(required=True, passed=True)]

    monkeypatch.setattr(e.prior, "execute_commands", exited)
    assert (
        e.main(
            ["--data-root", str(tmp_path / "absent"), "--output", str(tmp_path / f"{e.NAME}.json")]
        )
        == 0
    )
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=False))
    assert e.main(["--terminal-recheck", str(tmp_path / f"{e.NAME}.json")]) == 1
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    monkeypatch.setattr(e, "terminal_check", lambda p: dict(passed=True))
    with pytest.raises(ValueError, match="primary_resolution"):
        e.publish(tmp_path / "read-failure" / f"{e.NAME}.json", e.base({}))
