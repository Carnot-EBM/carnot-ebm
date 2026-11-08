"""REQ-REPORT-8922 / SCENARIO-REPORT-8922-CONTRACT: sealed future controls."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import interpolated_causal_streams_8922 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_protocol_and_gate_diagnostics(tmp_path: Path) -> None:
    """REQ-REPORT-8922: external absence never authorizes future labels."""
    p = m.protocol()
    assert [s["seed"] for s in p["frozen_evaluation_manifest"]] == list(range(763200, 763224))
    assert len(m.census("missing")["rows"]) == 18432
    assert m.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    source = tmp_path / (m.SOURCE + ".json")
    for data, check in [
        ({}, "missing_field"),
        ({"methods_ready_score": 0}, "observed_zero"),
        ({"methods_ready_score": True}, "rejected_producer"),
        ({"methods_ready_score": 1, "candidate_ready_score": 0}, "observed_zero"),
        ({"methods_ready_score": 1, "candidate_ready_score": 1}, "rejected_producer"),
    ]:
        atomic_json(source, data)
        assert m.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    source.write_text("invalid")
    assert not m.preconditions(tmp_path / "raw", tmp_path)["external_ready"]


@pytest.fixture(scope="module")
def stream_result(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """SCENARIO-REPORT-8922-CONTRACT: use real arithmetic on private constructed worlds."""
    raw = tmp_path_factory.mktemp("causal8922")
    plan = m.protocol()["frozen_evaluation_manifest"][0]
    head = m.preconditions(raw / "authentication", m.ROOT / "results")["heads"][str(plan["seed"])]
    result = m.run_stream(raw, plan, head)
    return dict(raw=raw, plan=plan, head=head, result=result)


def test_causal_stream_and_tampering(stream_result: dict) -> None:
    """REQ-REPORT-8922: causal scores and permutation survive restart and reject corruption."""
    result, plan = stream_result["result"], stream_result["plan"]
    m.verify(result, plan)
    assert len(result["rows"]) == 768
    assert len(result["prediction_rows"]) == len(result["release_rows"]) == 192
    assert len(result["retention_rows"]) == 576
    assert len(result["gradient_call_ledger"]) == 4
    assert len({r["calls"] for r in result["gradient_call_ledger"]}) == 1
    assert result["optimizer_sequences"] and result["restart_rows"]
    assert all(r["passed"] for r in result["control_rows"])
    for field in ("metric", "prediction"):
        broken = deepcopy(result)
        broken["rows"][0][field] = 99
        broken["stream_checksum"] = canonical_hash(
            {k: v for k, v in broken.items() if k != "stream_checksum"}
        )
        with pytest.raises(ValueError):
            m.verify(broken, plan)
    broken = deepcopy(result)
    broken["states"]["interpolated_permuted"]["blocks"][0]["targets"][0] = "invalid"
    broken["stream_checksum"] = canonical_hash(
        {k: v for k, v in broken.items() if k != "stream_checksum"}
    )
    with pytest.raises(ValueError):
        m.verify(broken, plan)


def test_authentic_producer_and_operand_drift(tmp_path: Path) -> None:
    """REQ-REPORT-8922: current replacement bytes cannot replace original custody."""
    from carnot.reporting.v746_retained_evidence import reference

    pre = m.preconditions(tmp_path / "auth", m.ROOT / "results")
    assert pre["external_ready"], pre["diagnostics"]
    assert pre["heads"]
    source = json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())
    operand = tmp_path / "operand.txt"
    operand.write_text("original")
    source["source_artifact_hashes"] = []
    source["code_config_hashes"] = [reference(operand)]
    source["raw_shard_hashes"] = []
    target = tmp_path / (m.SOURCE + ".json")
    atomic_json(target, source)
    pin = reference(target)["sha256"]
    operand.write_text("changed")
    with patch.object(m, "PIN", pin):
        assert not m.preconditions(tmp_path / "drift", tmp_path)["external_ready"]


def test_recover_and_deadline(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8922: retain completed and failed cells without replacement streams."""
    result, plan = stream_result["result"], stream_result["plan"]
    atomic_json(tmp_path / (plan["stream_id"] + ".json"), result)
    atomic_json(tmp_path / ("attempt-" + plan["stream_id"] + ".json"), dict(status="complete"))
    (tmp_path / "states" / plan["stream_id"]).mkdir(parents=True)
    recovered = m.recover(tmp_path, "unfinished")
    assert sum(r["status"] == "complete" for r in recovered["rows"]) == 768
    assert recovered["numerical_training_calls"] == result["numerical_training_calls"]
    next_plan = m.protocol()["frozen_evaluation_manifest"][1]
    atomic_json(tmp_path / ("attempt-" + next_plan["stream_id"] + ".json"), dict(status="failed"))
    assert sum(r["status"] == "failed" for r in m.recover(tmp_path, "failure")["rows"]) == 768


def test_private_cli_and_rehashed_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8922-CONTRACT: real foreign CLI blocks and rejects rehashed tampering."""
    import os
    import subprocess

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [str(m.ROOT / ".venv/bin/python"), "-u", str(m.ROOT / m.CLI)]
    child = subprocess.run(
        command + ["--sources", str(tmp_path), "--private-output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["context_receipts"]
    assert value["verdict_class"] == "blocked"
    child = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value["completed_count"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    child = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 1


@pytest.mark.parametrize("mode", ["null", "resources", "measurement", "validation", "terminal"])
def test_owned_terminal_paths(tmp_path: Path, mode: str) -> None:
    """REQ-REPORT-8922: completed valid execution is null; owned failures disqualify."""
    from carnot.reporting import interpolated_causal_runner_8922 as r

    result = m.census("unit_control")
    result.update(
        rows=[dict(row, status="complete", missing_reason=None) for row in result["rows"]],
        stream_refs=[{}] * 24,
        attempt_rows=[dict(status="complete")] * 24,
        control_rows=[dict(passed=True)],
        pending_feedback_rows=[],
    )
    pre = dict(
        external_ready=True,
        resources=dict(passed=mode != "resources"),
        references=[],
        diagnostics=[],
        heads={},
    )
    validation = dict(
        coverage=dict(
            complete=mode != "validation",
            percent=100,
            files=[dict(path=p, excluded=[], missing=[]) for p in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[],
        raw_references=[],
    )
    log = tmp_path / "scanner.log"
    log.write_text("retained scanner output")
    reports = (
        [dict(passed=False, stdout_path=str(log), stderr_path=str(log))]
        if mode == "terminal"
        else []
    )
    with (
        patch.object(m, "preconditions", return_value=pre),
        patch.object(
            m,
            "measure",
            side_effect=RuntimeError("owned_failure") if mode == "measurement" else None,
        ),
        patch.object(m, "recover", return_value=result),
        patch.object(r.checks, "validate", return_value=validation),
        patch.object(r.checks, "terminal", return_value=reports),
        patch.object(r, "replay", return_value=dict(passed=True)),
    ):
        output = tmp_path / (m.NAME + ".json")
        assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("null" if mode == "null" else "disqualified")
    assert value["causal_rows_ready_score"] == int(mode == "null")
    assert bool(value["owned_errors"]) == (mode == "measurement")


def test_cli_argument_and_replay_failures(tmp_path: Path) -> None:
    """REQ-REPORT-8922: dates and output custody cannot silently change."""
    from carnot.reporting import interpolated_causal_runner_8922 as r

    with pytest.raises(SystemExit):
        r.main(["--date", "20261109"])
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])
    assert r.main(["--replay", str(tmp_path / "missing.json")]) == 1


def test_measure_success_and_failure(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8922: no replacement is allowed when a scheduled stream fails."""
    plan = stream_result["plan"]
    p = dict(m.protocol(), frozen_evaluation_manifest=[plan])
    pre = dict(heads={str(plan["seed"]): stream_result["head"]})
    with (
        patch.object(m, "protocol", return_value=p),
        patch.object(m, "run_stream", return_value=stream_result["result"]),
    ):
        assert len(m.measure(tmp_path / "complete", pre)["rows"]) == 768
    with (
        patch.object(m, "protocol", return_value=p),
        patch.object(m, "run_stream", side_effect=ValueError("failed_cell")),
    ):
        with pytest.raises(ValueError, match="failed_cell"):
            m.measure(tmp_path / "failed", pre)
    with (
        patch.object(m.time, "monotonic", side_effect=[0, 2500]),
        patch.object(m, "protocol", return_value=p),
    ):
        with pytest.raises(ValueError, match="measurement_deadline"):
            m.measure(tmp_path / "deadline", pre)


def test_partial_custody(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8922: interrupted work retains predictions, optimizer and spent gradients."""
    source = stream_result["raw"]
    plan, saved = stream_result["plan"], stream_result["result"]
    directory = tmp_path / "states" / plan["stream_id"]
    directory.mkdir(parents=True)
    for name in ["progress.json", "transactions.json", "optimizer_sequences.json"]:
        (directory / name).write_bytes((source / name).read_bytes())
    for arm, state in saved["states"].items():
        atomic_json(directory / (arm + ".json"), state)
    result = m.recover(tmp_path, "interrupted")
    assert result["numerical_training_calls"] == saved["numerical_training_calls"]
    assert result["prediction_rows"] and result["state_hashes"] and result["optimizer_sequences"]


def test_score_precedes_update_and_restart_equivalence(tmp_path: Path, stream_result: dict) -> None:
    """SCENARIO-REPORT-8922-CONTRACT: research scoring cannot observe the new committed head."""
    result = stream_result["result"]
    assert len(result["scoring_rows"]) == 192
    releases = {(r["arm"], r["group_id"]): r for r in result["release_rows"]}
    for row in result["scoring_rows"]:
        release = releases[row["arm"], row["group_id"]]
        assert row["scored_before_release"] is True
        assert row["prediction"] == release["prediction"]
    uninterrupted = m.run_stream(
        tmp_path, stream_result["plan"], stream_result["head"], restart=False
    )
    assert uninterrupted["states"] == result["states"]
    assert uninterrupted["rows"] == result["rows"]


def test_real_partial_publication_and_semantic_replay(tmp_path: Path, stream_result: dict) -> None:
    """REQ-REPORT-8922: genuine retained partial work replays without claiming readiness."""
    from carnot.reporting import interpolated_causal_runner_8922 as r

    def measure(raw: Path, pre: dict) -> dict:
        plan = stream_result["plan"]
        atomic_json(raw / (plan["stream_id"] + ".json"), stream_result["result"])
        atomic_json(raw / ("attempt-" + plan["stream_id"] + ".json"), dict(status="complete"))
        return m.recover(raw, "unstarted")

    validation = dict(
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[],
        raw_references=[],
    )
    output = tmp_path / (m.NAME + ".json")
    with (
        patch.object(m, "measure", side_effect=measure),
        patch.object(r.checks, "validate", return_value=validation),
        patch.object(r.checks, "terminal", return_value=[]),
    ):
        assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["independent_count"] == 1
    assert value["completed_count"] == 768 and value["numerical_training_calls"] > 0
    assert r.main(["--replay", str(output)]) == 0
    value["rows"][0]["metric"] = 10
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert r.main(["--replay", str(output)]) == 1
