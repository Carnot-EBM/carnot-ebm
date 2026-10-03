"""SCENARIO-REPORT-8003-VALIDATION: real CLI and frozen private checks."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_8003_v693_hardware_sparse_boundary as e
from carnot.reporting import hardware_sparse_8003 as h
from carnot.reporting.current_work_receipt import atomic_json


def test_private_cli(tmp_path: Path) -> None:
    """REQ-REPORT-8003: success, blocked, date and cold replay use real children."""
    source = tmp_path / "fixture.json"
    atomic_json(source, h.fixture())
    script = str(e.ROOT / e.OWNED[-1])
    output = tmp_path / "success" / (e.NAME + ".json")
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", script, *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    result = run(["--fixture-input", str(source), "--validation-worker", "--output", str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert run(["--cold-replay", str(output)]).returncode == 0
    value["board_custody_ready_score"] = 0
    atomic_json(output, value)
    assert run(["--cold-replay", str(output)]).returncode == 1
    assert run(["--date", "20261001"]).returncode == 1
    blocked = tmp_path / "blocked" / output.name
    result = run(
        ["--root", str(tmp_path / "absent"), "--validation-worker", "--output", str(blocked)]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["hardware_evidence_ready_score"] == 0
    assert run(["--cold-replay", str(blocked)]).returncode == 0


def test_owned_check_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8003: a required failure is disqualified, never ready."""
    source = tmp_path / "fixture.json"
    atomic_json(source, h.fixture())
    monkeypatch.setattr(
        e, "execute", lambda *args: [dict(required=True, passed=False, name="owned_failure")]
    )
    monkeypatch.setattr(e, "coverage_counts", lambda *args: {})
    output = tmp_path / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["hardware_evidence_ready_score"] == 0
    assert value["gate_check_summary"]


def test_inprocess_routes_and_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8003: empty coverage fails while complete owned coverage qualifies."""
    from carnot.reporting import validation_8003 as v

    source = tmp_path / "fixture.json"
    atomic_json(source, h.fixture())
    output = tmp_path / "success" / (e.NAME + ".json")
    monkeypatch.setattr(e, "execute", lambda *args: [])
    monkeypatch.setattr(
        e,
        "coverage_counts",
        lambda *args: {p: dict(num_statements=1, covered_lines=1) for p in e.OWNED},
    )
    assert e.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--date", "20261001"]) == 1
    health = tmp_path / "previous_health.json"
    atomic_json(
        health,
        dict(
            rows=[
                dict(
                    name="full_pytest",
                    required=False,
                    passed=False,
                    exit_code=1,
                    command=".venv/bin/pytest tests/python -q",
                )
            ]
        ),
    )
    assert (
        e.main(
            [
                "--fixture-input",
                str(source),
                "--repository-health-receipt",
                str(health),
                "--output",
                str(tmp_path / "health" / output.name),
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "health" / output.name).read_bytes())
    assert saved["repository_health"][0]["reused_diagnostic"]
    assert not saved["repository_health"][0]["passed"]
    monkeypatch.setattr(e, "coverage_counts", lambda *args: {})
    assert (
        e.main(["--fixture-input", str(source), "--output", str(tmp_path / "empty" / output.name)])
        == 0
    )
    assert v.coverage_counts(tmp_path) == {}
    atomic_json(
        tmp_path / "coverage.json",
        dict(
            files={str(e.ROOT / e.OWNED[0]): dict(summary=dict(num_statements=1, covered_lines=1))}
        ),
    )
    assert v.coverage_counts(tmp_path)[e.OWNED[0]]["num_statements"] == 1


@pytest.mark.parametrize("fault", ["validator", "private_reader", "public_reader", "changed_bytes"])
def test_publication_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """REQ-REPORT-8003: validator rejection and identity drift cannot pass publication."""
    value = dict(
        h.reduce(h.fixture()), replay_inputs=h.fixture(), experiment_id=8003, task_id=e.TASK
    )
    output = tmp_path / (e.NAME + ".json")

    def terminal(path: Path) -> dict:
        return dict(passed=fault != "validator", candidate_sha256=e.sha256_file(path))

    monkeypatch.setattr(e, "terminal_check", terminal)
    real_reader = e.reader_receipt
    calls = 0

    def reader(*args, **kwargs) -> dict:
        nonlocal calls
        calls += 1
        receipt = real_reader(*args, **kwargs)
        if fault == "private_reader" or fault == "public_reader" and calls == 2:
            receipt["passed"] = False
        return receipt

    monkeypatch.setattr(e, "reader_receipt", reader)
    if fault == "changed_bytes":

        def changed(output, value, validator) -> dict:
            atomic_json(tmp_path / "changed.json", dict(value, mutation=True))
            return validator(tmp_path / "changed.json")

        monkeypatch.setattr(e, "publish_primary", changed)
    with pytest.raises(
        ValueError,
        match="candidate_rejected|private_primary_resolution|primary_resolution|publication_drift",
    ):
        e.publish(output, value, tmp_path / "scratch")


def test_consumer_scratch_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8003: historical consumers require private scratch under /tmp."""
    from carnot.reporting import validation_8003 as v

    monkeypatch.setattr(v, "atomic_json", lambda *args: None)
    manifest = v.freeze(tmp_path, Path("/private/cached-scratch"))
    command = next(r for r in manifest["commands"] if r["name"] == "owned_unit_consumers")
    label = next(a.split("=", 1)[1] for a in command["argv"] if a.startswith("--basetemp="))
    assert Path(label).is_relative_to(Path("/tmp"))
    assert Path(label).parent.is_dir()


def test_receipt_log_cwd(tmp_path: Path) -> None:
    """REQ-REPORT-8003: cold replay logs resolve against the actual child cwd."""
    from carnot.reporting import validation_8003 as v

    receipt = dict(name="cold_replay_cli", log_path="evidence/cold.log")
    manifest = dict(commands=[dict(name="cold_replay_cli", cwd=str(tmp_path))])
    v.normalize_receipts([receipt], manifest)
    assert receipt["log_path"] == str(tmp_path / "evidence/cold.log")


def test_configuration_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8003: edits after the freeze require a new invocation."""
    source = tmp_path / "fixture.json"
    atomic_json(source, h.fixture())
    original = e.sha256_file
    calls = {}

    def changed(path: Path) -> str:
        calls[path] = calls.get(path, 0) + 1
        return (
            "sha256:changed" if path == e.ROOT / e.OWNED[0] and calls[path] > 1 else original(path)
        )

    monkeypatch.setattr(e, "sha256_file", changed)
    assert (
        e.main(
            [
                "--fixture-input",
                str(source),
                "--validation-worker",
                "--output",
                str(tmp_path / (e.NAME + ".json")),
            ]
        )
        == 1
    )
