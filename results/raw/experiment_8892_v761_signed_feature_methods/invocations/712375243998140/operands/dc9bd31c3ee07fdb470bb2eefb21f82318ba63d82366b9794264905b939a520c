"""REQ-REPORT-8856: causal execution retains complete, replayable evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import linear_causal_streams_8856 as m
from carnot.reporting import linear_causal_runner_8856 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def stream_work(tmp_path_factory: Any, fitting_evidence: Any) -> dict[str, Any]:
    """REQ-REPORT-8856: execute real fsynced stores and the complete mixed delay tail."""
    plan = m.a.stream(0)
    heads, _ = r.initialize(fitting_evidence)
    return m.run_stream(tmp_path_factory.mktemp("stream8856"), plan, heads)


def test_stream_integrity(stream_work: Any) -> None:
    """SCENARIO-REPORT-8856-CONTRACT: all arms restart, drain and preserve retention."""
    work = stream_work
    assert m.integrity(work)["passed"]
    assert len(work["rows"]) == 1152
    assert len(work["feedback_rows"]) == 288
    assert len(work["rejected_feedback_rows"]) == 12
    assert len(work["retention_rows"]) == 864
    assert len(work["restart_rows"]) == 6
    assert all(row["unchanged"] for row in work["restart_rows"])
    assert all(row["wall_clock_ns"] > 0 for row in work["rows"])
    assert all(row["learning_state_unchanged"] for row in work["retention_rows"])
    for row in work["feedback_rows"]:
        settings = m.a.ARMS[row["arm"]]
        assert row["weights_changed"] == settings["calibration"]
        assert not row["templates_changed"] or settings["additions"]
        assert row["feedback_label"] in row["released_prefix"]
    assert (
        len(m.result_rows(dict(protocol=m.a.protocol(), stream_results=[work]), "pending")) == 27648
    )
    assert len(m.descriptives(work["rows"])) == 12
    assert {row["feedback_delay"] for row in m.descriptives(work["rows"])} == {0, 8}


def test_tampered_rows(stream_work: Any) -> None:
    """REQ-REPORT-8856: seals and complete census cannot be replaced by counters."""
    bad = deepcopy(stream_work)
    bad["rows"].pop()
    with pytest.raises(ValueError, match="row_census"):
        m.integrity(bad)
    bad = deepcopy(stream_work)
    online = next(row for row in bad["rows"] if row["split"] == "online")
    online["prediction_seal"] = "changed"
    with pytest.raises(ValueError, match="prediction_seal"):
        m.integrity(bad)
    assert m.semantic(dict(wall_clock_ns=1, value=[dict(duration_s=2, x=3)])) == dict(
        value=[dict(x=3)]
    )


def fixture_publish(tmp_path: Path, monkeypatch: Any, failure: Exception | None = None) -> Path:
    """REQ-REPORT-8856: private fixtures exercise publication without granting evidence."""

    def preflight(raw: Path, source: Path) -> Any:
        if failure:
            raise failure
        return {}, {}, []

    monkeypatch.setattr(r, "preflight", preflight)
    monkeypatch.setattr(r, "execute", lambda raw, evidence, work, started: None)
    monkeypatch.setattr(r, "validate", lambda raw: dict(passed=True, receipts=[], coverage={}))
    monkeypatch.setattr(r, "terminal", lambda path, raw: dict(passed=True, receipts=[]))
    output = tmp_path / (r.NAME + ".json")
    assert r.main(["--date", "20261106", "--private-output", str(output)]) == 0
    return output


@pytest.mark.parametrize(
    "failure,status",
    [
        (FileNotFoundError("absent"), "missing_file"),
        (KeyError("field"), "missing_field"),
        (PermissionError("observed_zero"), "observed_zero"),
        (ValueError("changed_bytes"), "rejected_producer"),
    ],
)
def test_blocked_publication(
    tmp_path: Path, monkeypatch: Any, failure: Exception, status: str
) -> None:
    """REQ-REPORT-8856: external absence retains every registered denominator."""
    path = fixture_publish(tmp_path, monkeypatch, failure)
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["status"] == status
    assert value["intended_count"] == value["censored_count"] == 27648
    assert value["stream_rows_ready_score"] == 0
    assert r.replay(path)["replay_passed"]
    assert r.main(["--replay", str(path)]) == 0


def test_private_cli(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-8856-CONTRACT: replay works outside the checkout without PYTHONPATH."""
    path = fixture_publish(tmp_path, monkeypatch, FileNotFoundError("absent"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for argv, expected in (
        (["--replay", str(path)], 0),
        (["--date", "bad"], 1),
        (["--replay", str(tmp_path / "absent")], 1),
    ):
        child = subprocess.run(
            [str(r.ROOT / ".venv/bin/python"), "-u", str(r.ROOT / r.CLI), *argv],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == expected, child.stdout + child.stderr
    value = json.loads(path.read_bytes())
    value["completed_count"] = 1
    atomic_json(path, value)
    with pytest.raises(ValueError):
        r.replay(path)
    assert r.main(["--replay", str(path)]) == 1
    assert r.main(["--date", "bad"]) == 1


def test_authority_rejections(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8856: changed bytes, missing fields and zero readiness cannot run."""
    with pytest.raises(FileNotFoundError):
        r.preflight(tmp_path / "raw", tmp_path / "missing")
    producer = json.loads(r.SOURCE.read_bytes())
    for field in ("linear_methods_ready_score", "required_checks_passed"):
        path = tmp_path / (field + ".json")
        atomic_json(path, dict(producer, **{field: 0}))
        monkeypatch.setattr(r, "SOURCE_PIN", r.reference(path)["sha256"])
        with pytest.raises(PermissionError, match="observed_zero"):
            r.preflight(tmp_path / "raw", path)
    monkeypatch.setattr(r, "SOURCE_PIN", "sha256:bad")
    with pytest.raises(ValueError, match="operand_hash"):
        r.preflight(tmp_path / "raw", r.SOURCE)


@pytest.fixture(scope="module")
def fitting_evidence(tmp_path_factory: Any) -> Any:
    """REQ-REPORT-8856: use authenticated original fitting data, never future labels."""
    raw = tmp_path_factory.mktemp("authority8856")
    script = (
        "from pathlib import Path; from carnot.reporting import linear_causal_runner_8856 as r; "
        "from carnot.reporting.current_work_receipt import atomic_json; "
        f"e,p,refs=r.preflight(Path({str(raw)!r}),r.SOURCE); "
        "assert p['linear_methods_ready_score']==1 and refs; "
        f"atomic_json(Path({str(raw / 'response.json')!r}),dict(development_rows=e['development_rows'],fits=e['fits']))"
    )
    child = subprocess.run(
        [str(r.ROOT / ".venv/bin/python"), "-u", "-c", script],
        cwd=r.ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    return json.loads((raw / "response.json").read_bytes())


def test_original_fits(fitting_evidence: Any) -> None:
    """REQ-REPORT-8856: initial heads reproduce every admitted fit and reject drift."""
    heads, rows = r.initialize(fitting_evidence)
    assert len(heads) == len(rows) == 5
    assert sum(row["optimizer_steps"] for row in rows) == 2000
    bad = deepcopy(fitting_evidence)
    for row in bad["fits"]:
        if row["objective"] == "linear":
            row["parameters"] = []
    with pytest.raises(ValueError, match="original_fit_drift"):
        r.initialize(bad)


def test_execution_schedule(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8856: scheduling keeps all streams; clocks alone can stop execution."""
    monkeypatch.setattr(r, "initialize", lambda evidence: ({}, []))
    monkeypatch.setattr(m.a, "controls", lambda raw: [])
    seen = []

    def run(raw: Path, plan: Any, heads: Any) -> Any:
        seen.append(plan["stream_id"])
        return dict(stream=plan, rows=[], feedback_rows=[])

    monkeypatch.setattr(m, "run_stream", run)
    work = dict(protocol=m.a.protocol(), stream_results=[], stream_refs=[], pilot_refs=[])
    r.execute(tmp_path / "complete", {}, work, r.time.monotonic())
    assert len(work["stream_results"]) == 24 and len(seen) == 26
    assert len(work["pilot_refs"]) == 2
    monkeypatch.setattr(m.a, "projection", lambda pilots, initialization: dict(passed=False))
    with pytest.raises(TimeoutError, match="projected_execution_budget"):
        r.execute(
            tmp_path / "projection",
            {},
            dict(protocol=m.a.protocol(), pilot_refs=[]),
            r.time.monotonic(),
        )
    monkeypatch.setattr(m.a, "projection", lambda pilots, initialization: dict(passed=True))
    with pytest.raises(TimeoutError, match="registered_execution_deadline"):
        r.execute(
            tmp_path / "deadline",
            {},
            dict(protocol=m.a.protocol(), pilot_refs=[]),
            r.time.monotonic() - 2600,
        )


def test_validation_wrappers(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8856: scoped checks and terminal subprocesses retain exact ownership."""

    def validate(raw: Path) -> Any:
        assert m.a.prior.validation.OWNED == r.OWNED
        assert m.a.prior.validation.TESTS == r.TESTS
        return dict(passed=True)

    def commands(plan: Any, logs: Path) -> Any:
        assert len(plan) == 3 and plan[-1][-1] == 1800
        assert "PYTHONPATH" in plan[-1][1] and "/tmp" in plan[-1][1]
        return [dict(passed=True)]

    monkeypatch.setattr(m.a.prior.validation, "validate", validate)
    monkeypatch.setattr(m.a.prior.validation, "commands", commands)
    assert r.validate(tmp_path)["passed"]
    assert r.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def test_partial_and_failed_checks(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8856: owned failures disqualify; unfinished owned work stays partial."""
    path = fixture_publish(tmp_path / "partial", monkeypatch)
    assert json.loads(path.read_bytes())["verdict_class"] == "partial"
    monkeypatch.setattr(r, "validate", lambda raw: dict(passed=False, receipts=[], coverage={}))
    assert r.main(["--output", str(tmp_path / "failed" / (r.NAME + ".json"))]) == 0
    assert (
        json.loads((tmp_path / "failed" / (r.NAME + ".json")).read_bytes())["verdict_class"]
        == "disqualified"
    )
    monkeypatch.setattr(r, "validate", lambda raw: dict(passed=True, receipts=[], coverage={}))
    monkeypatch.setattr(r, "terminal", lambda path, raw: dict(passed=False, receipts=[]))
    out = tmp_path / "terminal" / (r.NAME + ".json")
    assert r.main(["--output", str(out)]) == 0
    assert json.loads(out.read_bytes())["stream_rows_ready_score"] == 0

    def fail(*args: Any) -> None:
        raise RuntimeError("owned_control")

    monkeypatch.setattr(r, "execute", fail)
    out = tmp_path / "owned" / (r.NAME + ".json")
    assert r.main(["--output", str(out)]) == 0
    assert json.loads(out.read_bytes())["gate_check_summary"][0]["status"] == "owned_failure"


def test_real_replay(
    tmp_path: Path, monkeypatch: Any, stream_work: Any, fitting_evidence: Any
) -> None:
    """REQ-REPORT-8856: recompute real fits, gradients and losses in fresh stores."""
    fixture_publish(tmp_path / "setup", monkeypatch)
    _, initial = r.initialize(fitting_evidence)
    controls = m.a.controls(tmp_path / "controls")
    checkpoint = tmp_path / "stream.json"
    atomic_json(checkpoint, stream_work)

    def execute(raw: Path, evidence: Any, work: Any, started: float) -> None:
        work.update(
            stream_results=[stream_work],
            stream_refs=[r.reference(checkpoint)],
            initialization_rows=initial,
            controls=controls,
            pilot_refs=[r.reference(checkpoint)],
        )

    monkeypatch.setattr(r, "preflight", lambda raw, source: (fitting_evidence, {}, []))
    monkeypatch.setattr(r, "execute", execute)
    output = tmp_path / (r.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 1152 and value["censored_count"] == 26496
    assert value["numerical_training_calls"] > 2000
    assert r.replay(output)["replay_passed"]
    log = tmp_path / "receipt.log"
    log.write_text("private actual test log\n")
    value["validation_receipts"] = [dict(log_path=str(log), log_sha256=r.reference(log)["sha256"])]
    atomic_json(output, value)
    monkeypatch.setattr(m, "run_stream", lambda raw, plan, heads: deepcopy(stream_work))
    assert r.replay(output)["replay_passed"]
    log.write_text("changed\n")
    with pytest.raises(ValueError, match="validation_log"):
        r.replay(output)


def test_ready_coverage_gate(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-8856: readiness checks the measured coverage operand, even after rehash."""
    path = fixture_publish(tmp_path, monkeypatch, FileNotFoundError("absent"))
    value = json.loads(path.read_bytes())
    work = r.bound(value["raw_shard_hashes"][0])
    reduced = r.reductions(work, value["row_reason"])
    reduced.update(completed_count=27648, censored_count=0)
    value.update(
        reduced,
        verdict_class="circular_positive",
        stream_rows_ready_score=1,
        required_checks_passed=True,
    )
    coverage_path = tmp_path / "coverage.fixture"
    coverage_path.write_text("coverage validation fixture\n")
    coverage = dict(combined_reference=r.reference(coverage_path), percent=100, complete=True)
    value["owned_statement_coverage"] = coverage
    monkeypatch.setattr(r, "reductions", lambda work, reason: reduced)
    monkeypatch.setattr(m.a.prior.validation, "coverage_summary", lambda path, owned: coverage)
    atomic_json(path, value)
    assert r.replay(path)["replay_passed"]
    value["owned_statement_coverage"]["percent"] = 99
    atomic_json(path, value)
    with pytest.raises(ValueError, match="qualification_drift"):
        r.replay(path)
