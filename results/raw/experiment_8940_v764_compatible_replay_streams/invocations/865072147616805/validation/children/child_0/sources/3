"""REQ-REPORT-8940 / SCENARIO-REPORT-8940-CONTRACT: future replay custody."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import compatible_replay_streams_8940 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_protocol_and_census() -> None:
    """REQ-REPORT-8940: preserve methods and all23040 paired query comparisons."""
    assert m.protocol() == m.methods.protocol()
    result = m.census("missing")
    assert (
        "comparison_training_calls" not in result and "private_control_training_calls" not in result
    )
    assert len(result["rows"]) == 23040
    assert len({(r["unit_id"], r["arm"]) for r in result["rows"]}) == 23040
    assert all(r["status"] == "censored" for r in result["rows"])


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict, dict]:
    """REQ-REPORT-8940: execute genuine tiny-head updates on one sealed stream."""
    raw = tmp_path_factory.mktemp("streams8940")
    plan = m.protocol()["frozen_evaluation_manifest"][0]
    head = m.methods.signed.head(11, "signed", [1 / 3, 1 / 3])
    result = m.run_stream(raw, plan, head)
    return raw, result, head


def test_causal_rows_restart_and_terminal_pending(measured: tuple) -> None:
    """SCENARIO-REPORT-8940-CONTRACT: restart preserves all state and future seals."""
    _, result, head = measured
    assert len(result["rows"]) == 960
    assert len(result["prediction_rows"]) == 240
    assert len(result["retention_rows"]) == 720
    assert len(result["feedback_rows"]) == 200
    assert len(result["pending_feedback_rows"]) == 40
    assert len(result["restart_rows"]) == 5
    assert all(r["passed"] for r in result["control_rows"])
    assert all(r["subsequent_predictions_equal"] for r in result["restart_rows"])
    assert all(r["uninterrupted_query24_state_equal"] for r in result["restart_rows"])
    assert result["numerical_training_calls"] == 2 * len(result["gradient_ledger"])
    assert sum(r["training_calls"] for r in result["private_control_rows"]) == len(
        result["gradient_ledger"]
    )
    for arm, state in result["states"].items():
        assert state["cursor"] == 40 and state["prediction_cursor"] == 48
        assert len(state["pending"]) == 8
        assert all(r["split"] == "train" for r in state["buffer"])
        assert not any("retention" in r["group_id"] for r in state["released"])
        if arm == "frozen_verified":
            assert state["committed"] == head
    assert all(r["value"] == r["numerator"] / r["denominator"] for r in result["rows"])
    m.verify(result, result["plan"])
    changed = deepcopy(result)
    changed["rows"][0]["value"] += 0.1
    with pytest.raises(ValueError):
        m.verify(changed, result["plan"])


def test_missing_and_rejected_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-8940: classify absent, empty, rejected and zero prerequisites."""
    raw = tmp_path / "raw"
    assert m.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({"replay_methods_ready_score": 0}, "observed_zero"),
        ({"replay_methods_ready_score": 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (m.SOURCE + ".json"), value)
        assert m.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == check
    (tmp_path / (m.SOURCE + ".json")).write_text("invalid json")
    assert m.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "rejected_producer"


def test_blocked_cli_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8940-CONTRACT: the real foreign-directory CLI fails closed."""
    import os
    import subprocess
    import sys

    from carnot.reporting import compatible_replay_stream_runner_8940 as r

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    run = subprocess.run(
        argv + ["--sources", str(tmp_path), "--private-output", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["replay_rows_ready_score"] == 0
    assert len(value["rows"]) == 23040 and value["gate_check_summary"]
    valid = subprocess.run(
        argv + ["--replay", str(output)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert valid.returncode == 0 and "replay_passed" in valid.stdout
    assert r.main(["--replay", str(output)]) == 0
    value["rows"][0]["value"] = 0.1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert r.main(["--replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "20261109"])
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


def test_original_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8940: admitted producer bytes cannot hide changed original heads."""
    from carnot.reporting.v746_retained_evidence import reference

    heads = tmp_path / "heads.json"
    atomic_json(heads, {"11": {"fixture": True}})
    producer = tmp_path / (m.SOURCE + ".json")
    atomic_json(
        producer,
        dict(
            experiment_id=8933,
            task_id=m.methods.TASK,
            milestone="2026.11.764",
            replay_methods_ready_score=1,
            required_checks_passed=True,
            protocol=m.protocol(),
            protocol_hash=canonical_hash(m.protocol()),
            owned_statement_coverage=dict(complete=True, percent=100),
            control_rows=[dict(passed=True)],
            raw_directory=str(tmp_path),
            source_artifact_hashes=[reference(heads)],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(m, "PIN", reference(producer)["sha256"])
    pre = m.preconditions(tmp_path / "raw", tmp_path)
    assert pre["external_ready"] and pre["heads"] == {"11": {"fixture": True}}
    atomic_json(heads, {"changed": True})
    assert (
        m.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    heads.unlink()
    assert m.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"


def test_measure_and_recover_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured: tuple
) -> None:
    """REQ-REPORT-8940: preserve complete, failed and partial attempts without replacing seeds."""
    import shutil

    directory, stream, head = measured
    plan = stream["plan"]
    p = deepcopy(m.protocol())
    p["frozen_evaluation_manifest"] = [plan]
    monkeypatch.setattr(m, "protocol", lambda: p)
    monkeypatch.setattr(m, "run_stream", lambda *a: stream)
    result = m.measure(tmp_path / "complete", {str(plan["initial_head_seed"]): head})
    assert result["rows"] and len(result["stream_refs"]) == 1
    assert result["numerical_training_calls"] == stream["numerical_training_calls"]
    assert result == m.recover(tmp_path / "complete", "unstarted")

    def fail(*args: object) -> None:
        raise ValueError("owned_failure")

    monkeypatch.setattr(m, "run_stream", fail)
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="owned_failure"):
        m.measure(failed, {str(plan["initial_head_seed"]): head})
    assert all(r["status"] == "failed" for r in m.recover(failed, "owned_failure")["rows"])
    partial = failed / "states" / plan["stream_id"]
    shutil.copytree(directory, partial)
    progress = json.loads((partial / "progress.json").read_bytes())
    progress["rows"] = progress["rows"][:10]
    atomic_json(partial / "progress.json", progress)
    recovered = m.recover(failed, "owned_failure")
    assert sum(r["status"] == "complete" for r in recovered["rows"]) == 10
    assert recovered["numerical_training_calls"] == stream["numerical_training_calls"]
    assert len(recovered["pending_feedback_rows"]) == 40
    assert recovered["numerical_training_calls"] == len(recovered["gradient_ledger"]) + sum(
        r["training_calls"] for r in recovered["private_control_rows"]
    )


@pytest.mark.parametrize("mode", ["complete", "resources", "measurement", "validation", "terminal"])
def test_owned_terminal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured: tuple, mode: str
) -> None:
    """SCENARIO-REPORT-8940-CONTRACT: failed owned acceptance always disqualifies."""
    from types import SimpleNamespace
    from carnot.reporting import compatible_replay_stream_runner_8940 as r
    from carnot.reporting.v746_retained_evidence import reference

    _, stream, head = measured
    source = tmp_path / (m.SOURCE + ".json")
    atomic_json(source, {"fixture": True})
    saved = tmp_path / "stream.json"
    atomic_json(saved, stream)
    result = m.census("fixture")
    result["stream_refs"] = [reference(saved)] if mode != "measurement" else []
    pre = dict(
        external_ready=True,
        resources=dict(passed=mode != "resources"),
        references=[reference(source)],
        diagnostics=[],
        heads={str(stream["plan"]["initial_head_seed"]): head},
    )
    monkeypatch.setattr(m, "preconditions", lambda *a: pre)
    monkeypatch.setattr(m, "recover", lambda *a: deepcopy(result))
    monkeypatch.setattr(m, "run_stream", lambda *a: stream)

    def measurement(*args: object) -> None:
        if mode == "measurement":
            raise ValueError("owned_failure")

    monkeypatch.setattr(m, "measure", measurement)
    coverage = dict(
        complete=mode != "validation",
        percent=100,
        files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED],
    )
    validation = dict(coverage=coverage, union=dict(passed=True), receipts=[], raw_references=[])
    log = tmp_path / "terminal.log"
    log.write_text("private terminal receipt")
    monkeypatch.setattr(r, "CONTEXT", [("repository_health", log), ("absent", tmp_path / "absent")])
    monkeypatch.setattr(
        r,
        "adapter",
        lambda: SimpleNamespace(
            validate=lambda *a: validation,
            terminal=lambda *a: [
                dict(passed=mode != "terminal", stdout_path=str(log), stderr_path=str(log))
            ],
        ),
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["context_receipts"][0]["scope"] == "repository_health"
    assert value["verdict_class"] == "disqualified" and value["replay_rows_ready_score"] == 0
    assert r.replay(output)["passed"]
    for key, changed in [
        ("completed_count", -1),
        ("protocol_hash", "drift"),
        ("numerical_training_calls", -1),
        ("replay_rows_ready_score", 1),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError):
            r.replay(output)


def test_readiness_null_and_private_import(tmp_path: Path) -> None:
    """REQ-REPORT-8940: complete measurement needs no positive learning effect."""
    import runpy
    from carnot.reporting import compatible_replay_stream_runner_8940 as r

    value = dict(
        owned_statement_coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        test_union=dict(passed=True),
        validation_receipts=[dict(passed=True)],
        preconditions_checked=dict(external_ready=True, resources=dict(passed=True)),
        owned_errors=[],
        completed_count=23040,
        stream_refs=[{}] * 24,
        attempt_rows=[dict(status="complete")] * 24,
        pending_feedback_rows=[{}] * 960,
        control_rows=[dict(passed=True)],
    )
    assert r.readiness(value) == 1
    value["control_rows"][0]["passed"] = False
    assert r.readiness(value) == 0
    assert r.adapter().m is m
    runpy.run_path(str(m.ROOT / m.CLI), run_name="imported_private_control")
