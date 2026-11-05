"""REQ-REPORT-8802: exact source custody and standalone terminal replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import causal_memory_runner_8802 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_gate_missing_zero_and_rejected(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8802-COLD-CUSTODY: absence differs from an observed zero."""
    path = tmp_path / "upstream.json"
    assert r.upstream_gate(path)["observed"] == "missing_file"
    atomic_json(path, {"experiment_id": 8801})
    assert r.upstream_gate(path)["observed"] == "missing_field"
    atomic_json(path, {"experiment_id": 8801, "memory_protocol_ready_score": 0})
    assert r.upstream_gate(path)["observed"] == 0
    atomic_json(
        path,
        {"experiment_id": 8801, "memory_protocol_ready_score": 1, "verdict_class": "disqualified"},
    )
    assert r.upstream_gate(path)["observed"] == "rejected_producer"
    assert r.upstream_gate(r.ROOT / r.UPSTREAM)["passed"]


def test_candidate_replay_and_tamper(tmp_path: Path) -> None:
    """REQ-REPORT-8802: reconstruct counts from retained measured rows."""
    raw = tmp_path / "raw"
    value = r.run(raw, "20261104")
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert r.replay(path)["passed"]
    for field, supplied in [
        ("completed_count", -1),
        ("memory_execution_ready_score", 1),
        ("numerical_training_calls", -1),
    ]:
        changed = deepcopy(value)
        changed[field] = supplied
        changed["reproducibility_checksum"] = canonical_hash(changed["rows"])
        atomic_json(path, changed)
        assert not r.replay(path)["passed"]
    atomic_json(path, value)
    payload = Path(value["raw_shard_hashes"][0]["path"])
    original_bytes = payload.read_bytes()
    changed = deepcopy(value)
    raw_work = json.loads(original_bytes)
    raw_work["rows"][0]["metric"] += 0.5
    atomic_json(payload, raw_work)
    changed["rows"] = raw_work["rows"]
    changed["raw_shard_hashes"][0] = r.reference(payload)
    changed["reproducibility_checksum"] = canonical_hash(changed["rows"])
    atomic_json(path, changed)
    assert not r.replay(path)["passed"]
    payload.write_bytes(original_bytes)
    atomic_json(path, value)
    payload.write_text("{}")
    assert not r.replay(path)["passed"]
    atomic_json(path, {})
    assert not r.replay(path)["passed"]


def test_standalone_cli_and_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8802-COLD-CUSTODY: CLI works outside the checkout."""
    with pytest.raises(SystemExit):
        r.main(["--date", "20261103"])
    monkeypatch.setattr(r, "run", lambda *args: {"fixture": True})
    monkeypatch.setattr(r, "qualify_and_publish", lambda value, raw, out: None)
    assert (
        r.main(
            [
                "--date",
                "20261104",
                "--raw",
                str(tmp_path / "cli"),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        == 0
    )
    candidate = tmp_path / "bad.json"
    candidate.write_text("{}")
    assert r.main(["--replay", str(candidate)]) == 1
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "coverage",
        "run",
        "--rcfile=/dev/null",
        "--parallel-mode",
        "--data-file="
        + str(
            Path(os.environ.get("CARNOT_8802_COVERAGE_DIRECTORY", str(tmp_path)))
            / ".coverage.child"
        ),
        str(r.ROOT / r.CLI),
        "--replay",
        str(candidate),
    ]
    child = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert child.returncode == 1 and "replay" in child.stdout


def test_validation_and_publication_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8802: owned failure closes readiness; retained candidates stay auditable."""
    from types import SimpleNamespace

    receipts = []

    def child(name, argv, raw, **kwargs):
        receipts.append(name)
        return dict(name=name, passed=True)

    monkeypatch.setattr(r, "child", child)
    monkeypatch.setattr(
        r,
        "Coverage",
        lambda **kwargs: SimpleNamespace(combine=lambda **kwargs: None, save=lambda: None),
    )
    monkeypatch.setattr(r, "coverage_summary", lambda *args: dict(complete=True, percent=100))
    checked = r.validate(tmp_path / "validate")
    assert checked["passed"] and "private_E2E_018_021" in receipts
    assert r.terminal_checks(tmp_path / "candidate", tmp_path)["passed"]
    monkeypatch.setattr(r, "validate", lambda raw: checked)
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, validator: dict(passed=validator(out)["passed"])
    )
    value = dict(
        acceptance_gates=dict(preconditions=True, controls=True, owned_validation=False),
        verdict_class="disqualified",
        phase_spans=[],
        duration_s=0,
    )
    r.qualify_and_publish(value, tmp_path / "publish", tmp_path / "out")
    assert value["memory_execution_ready_score"] == 1 and value["verdict_class"] == "null"
    checked["passed"] = False
    r.qualify_and_publish(value, tmp_path / "failed", tmp_path / "out")
    assert value["memory_execution_ready_score"] == 0 and value["verdict_class"] == "disqualified"
    value["acceptance_gates"]["preconditions"] = False
    value["verdict_class"] = "blocked"
    r.qualify_and_publish(value, tmp_path / "blocked", tmp_path / "out")
    assert value["verdict_class"] == "blocked"

    def reject(*args):
        raise ValueError("candidate_rejected")

    monkeypatch.setattr(r, "publish_primary", reject)
    with pytest.raises(ValueError, match="candidate_rejected"):
        r.qualify_and_publish(value, tmp_path / "rejected", tmp_path / "out")
    assert (tmp_path / "rejected/rejected_candidate.json").is_file()


def test_external_absence_blocks_before_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8802-CAUSAL-PROTOTYPE: external absence never becomes partial."""
    monkeypatch.setattr(r, "UPSTREAM", str(tmp_path / "missing.json"))
    result = r.run(tmp_path / "blocked", "20261104")
    assert result["verdict_class"] == "blocked" and not result["rows"]
    assert result["censored_count"] == result["intended_count"]
    assert result["gate_check_summary"][0]["observed"] == "missing_file"
