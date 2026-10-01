"""REQ-REPORT-7993: owned checks and private real CLI publication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_7993_v693_capture_custody as producer
from carnot.reporting import validation_7993 as validation
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar


@pytest.fixture
def private_output(tmp_path):
    return tmp_path / "results" / (producer.NAME + ".json")


def test_private_main_success_blocked_replay(private_output, tmp_path):
    """SCENARIO-REPORT-7993-CUSTODY: diagnostic CLI never asserts readiness."""
    args = ["--private-run", "--output", str(private_output)]
    assert producer.main(args) == 0
    value = json.loads(private_output.read_text())
    assert value["verdict_class"] == "null"
    assert value["recovered_stream_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert all(r["imported_fields"] for r in value["cited_upstream_artifacts"])
    assert producer.main(["--cold-replay", str(private_output)]) == 0
    assert read_bound_sidecar(private_output, Path(value["terminal_validation_sidecar_path"]))[
        "primary_sha256"
    ] == sha256_file(private_output)
    value["rows"][0]["denominator"] = 2
    bad = tmp_path / "bad.json"
    atomic_json(bad, value)
    assert producer.main(["--cold-replay", str(bad)]) == 1
    blocked = tmp_path / "blocked" / (producer.NAME + ".json")
    assert producer.main(["--private-run", "--root", str(tmp_path), "--output", str(blocked)]) == 0
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["observed"] is None
    assert producer.main(["--cold-replay", str(blocked)]) == 0
    assert producer.main(["--private-run"]) == 1
    with pytest.raises(SystemExit):
        producer.main(["--date", "20260930"])


def test_owned_validation_and_frozen_checks(private_output, tmp_path, monkeypatch):
    """REQ-REPORT-7993: owned failures disqualify without altering custody."""
    value = producer.build(producer.ROOT, 0)
    log = tmp_path / "owned.log"
    log.write_text("owned validation fixture\n")
    passed = dict(
        name="owned", passed=True, required=True, log_path=str(log), log_sha256=sha256_file(log)
    )
    producer.apply_validation(value, [passed], dict(statements=1, covered=1, missing=0))
    assert value["recovered_stream_ready_score"] == 1
    assert value["positive_control_results"]["owned_fixture_checks_passed"]
    broken = deepcopy(value)
    producer.apply_validation(broken, [dict(passed=False, required=True)], {})
    assert broken["verdict_class"] == "disqualified" and broken["recovered_stream_ready_score"] == 0
    manifest = validation.freeze(tmp_path)
    assert any(c["name"] == "repository_health" for c in manifest["commands"])
    assert any(c["name"] == "e2e015" and c["required"] for c in manifest["commands"])
    assert manifest["owned"] == producer.OWNED
    monkeypatch.setattr(
        validation, "execute", lambda *_: ([passed], dict(statements=1, covered=1, missing=0))
    )
    assert producer.main(["--output", str(private_output)]) == 0
    assert json.loads(private_output.read_text())["recovered_stream_ready_score"] == 1
    with pytest.raises(ValueError, match="reconstruction"):
        value["historical_token_totals"]["completion_tokens"] += 1
        producer.replay(value)


def test_real_cli_success_blocked_cold(private_output, tmp_path):
    """SCENARIO-REPORT-7993-CUSTODY: private execution has no PYTHONPATH."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cli = str(producer.ROOT / producer.OWNED[-1])
    for args in [
        ["--private-run", "--output", str(private_output)],
        ["--cold-replay", str(private_output)],
        [
            "--private-run",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(tmp_path / "blocked" / private_output.name),
        ],
    ]:
        child = subprocess.run(
            [str(producer.ROOT / ".venv/bin/python"), "-u", cli, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert child.returncode == 0, child.stdout.decode() + child.stderr.decode()


def test_validation_execution_receipts(tmp_path, monkeypatch):
    """REQ-REPORT-7993: receipt logs and nonempty denominators are checked."""
    manifest = validation.freeze(tmp_path)
    monkeypatch.setattr(
        validation,
        "run_commands",
        lambda root, specs, **kw: [
            dict(
                name=s.name,
                passed=True,
                exit_code=0,
                log_path=str(tmp_path / "log"),
                log_sha256="sha256:log",
            )
            for s in specs
        ],
    )
    atomic_json(
        tmp_path / "coverage.json",
        dict(totals=dict(num_statements=1, covered_lines=1, missing_lines=0)),
    )
    receipts, counts = validation.execute(manifest, tmp_path)
    assert counts["statements"] == counts["covered"] == 1 and counts["missing"] == 0
    assert receipts[-2]["required"] is False
    (tmp_path / "coverage.json").unlink()
    receipts, counts = validation.execute(manifest, tmp_path)
    assert counts == {} and any(not r["passed"] for r in receipts)
    manifest["code_config_hashes"][0]["sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="frozen"):
        validation.execute(manifest, tmp_path)


def test_terminal_failure_and_replay_receipt_tamper(private_output, monkeypatch):
    """REQ-VERIFY-7993: a critical flag cannot be self-cleared."""
    assert producer.main(["--private-run", "--output", str(private_output)]) == 0
    value = json.loads(private_output.read_text())
    value["model_invocation_counts"]["generation_calls_attempted"] = 1
    with pytest.raises(ValueError, match="current"):
        producer.replay(value)
    monkeypatch.setattr(
        producer, "run_commands", lambda *a, **kw: [dict(name="adversarial", passed=False)]
    )
    assert not producer.terminal_check(private_output)["passed"]


def test_readiness_receipt_and_reader_failure_paths(tmp_path, private_output, monkeypatch):
    """REQ-REPORT-7993: critical flags, log drift and either reader fail closed."""
    value = producer.build(producer.ROOT, 0)
    log = tmp_path / "receipt.log"
    log.write_text("passed\n")
    producer.apply_validation(
        value,
        [dict(passed=True, required=True, log_path=str(log), log_sha256=sha256_file(log))],
        dict(statements=1, covered=1, missing=0),
    )
    flagged = deepcopy(value)
    flagged["flagged_adversarial"] = True
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.replay(flagged)
    log.write_text("tampered\n")
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        producer.replay(value)
    value["recovered_stream_ready_score"] = 0
    for mode, reason in [
        ("validator", "private_candidate"),
        ("private", "private_reader"),
        ("final", "primary_reader"),
    ]:
        monkeypatch.setattr(producer, "terminal_check", lambda p: dict(passed=mode != "validator"))
        monkeypatch.setattr(
            producer,
            "reader_receipt",
            lambda task, results, **kw: dict(
                passed=mode == "final" and results != private_output.parent
            ),
        )
        monkeypatch.setattr(
            producer,
            "publish_primary",
            lambda *a: dict(primary_sha256="sha256:test", sidecar_path="private"),
        )
        with pytest.raises(ValueError, match=reason):
            producer.publish(private_output, value, tmp_path / mode)
