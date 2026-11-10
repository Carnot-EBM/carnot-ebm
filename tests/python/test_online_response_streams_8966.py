"""REQ-REPORT-8966 / SCENARIO-REPORT-8966-CONTRACT: sealed real-error streams."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import online_response_streams_8966 as d
from carnot.reporting import online_response_stream_runner_8966 as runner
from carnot.memory import online_response_worker_8966 as w
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8966: authenticate retained inputs before any computation."""
    value = d.authenticate(d.ROOT)
    assert not value["failures"]
    return value


@pytest.fixture()
def stream_request(tmp_path):
    """REQ-REPORT-8966: mechanics fixtures never become real-response measurements."""
    labels = tmp_path / "labels.json"
    atomic_json(labels, dict(labels=[0, 1, 0, 1] * 8))
    return dict(
        stream_id="private-stream",
        unit_ids=[f"u{i}" for i in range(32)],
        probabilities=[0.2, 0.8] * 16,
        label_path=str(labels),
        retention_ids=[f"r{i}" for i in range(128)],
        retention_probabilities=[0.7] * 128,
        guard_rows=[[0.2, 0], [0.8, 1]] * 32,
    )


def test_schedule_equations_and_complete_pending_state(stream_request, tmp_path):
    """REQ-REPORT-8966: seal first, release exactly seven blocks, leave four pending."""
    request = stream_request
    state = w.trajectory(request, tmp_path / "control")
    assert len(state["rows"]) == 128
    assert len(state["retention_rows"]) == 1536
    assert state["pending"] == [29, 30, 31, 32]
    assert state["released_through"] == 28
    assert [r["query"] for r in state["feedback_rows"]] == list(range(8, 33, 4))
    assert state["feedback_rows"][0]["ordinals"] == [1, 2, 3, 4]
    assert len(state["promotion_rows"]) == 21
    assert state["params"]["frozen"] == [1.0, 0.0]
    for row in state["promotion_rows"]:
        assert row["candidate"] == w.adapter.step(
            row["before"], row["logits"], row["labels"], row["arm"]
        )
        assert row["after"] == (row["candidate"] if row["passed"] else row["before"])
    for q in range(1, 33):
        seal = json.loads((tmp_path / "control" / "seals" / f"{q:02d}.json").read_text())
        assert len(seal["rows"]) == 4 * q
        assert seal["prediction_hash"] == canonical_hash(seal["rows"])


def test_real_child_restart_and_future_truth_mutations(stream_request, tmp_path):
    """SCENARIO-REPORT-8966-CONTRACT: actual termination retains complete state."""
    request = stream_request
    baseline = w.trajectory(request, tmp_path / "control")
    restarted, receipt = d.restart(request, tmp_path / "restart")
    assert restarted == baseline
    assert receipt["terminated"] and receipt["query"] == 16 and receipt["passed"]
    mutated = deepcopy(request)
    truth = json.loads(Path(request["label_path"]).read_text())["labels"]
    atomic_json(tmp_path / "future.json", dict(labels=truth[:16] + [1 - y for y in truth[16:]]))
    mutated["label_path"] = str(tmp_path / "future.json")
    changed = w.trajectory(mutated, tmp_path / "future")
    assert changed["seals"][:23] == baseline["seals"][:23]
    assert changed["rows"][: 23 * 4] == baseline["rows"][: 23 * 4]
    assert changed["params"] != baseline["params"]


def test_authentic_real_corpus_and_input_failures(evidence, tmp_path, monkeypatch):
    """REQ-REPORT-8966: absent or changed producers name the exact failed field."""
    assert len(evidence["producers"]) == 4
    absent = d.authenticate(tmp_path)
    assert len(absent["failures"]) == 4
    assert all(r["check"] == "missing_file" for r in absent["failures"])
    name, pin, field, task = d.INPUTS[0]
    source = tmp_path / name
    atomic_json(source, {"task_id": task, "required_checks_passed": True})
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(source)["sha256"], field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["check"] == "missing_field"
    value = json.loads(source.read_text())
    value[field] = 0
    atomic_json(source, value)
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(source)["sha256"], field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["check"] == "observed_zero"
    monkeypatch.setattr(d, "INPUTS", [(name, pin, field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["check"] == "rejected_producer"


def test_private_cli_publication_cold_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8966-CONTRACT: private E2E-018/021 stays outside the repo."""
    output = tmp_path / (d.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(d.ROOT / d.CLI)]
    result = subprocess.run(
        command + ["--date", "20261110", "--fixture-e2e", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    replay = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["online_rows_ready_score"] == 0
    value["completed_count"] = 24
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert (
        subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        ).returncode
        == 1
    )


@pytest.fixture(scope="module")
def evaluated(evidence, tmp_path_factory):
    """REQ-REPORT-8966: execute all real streams once, with no synthetic substitution."""
    raw = tmp_path_factory.mktemp("8966-real")
    return raw, d.evaluate(evidence, raw)


def test_real_rows_and_independent_denominator(evaluated):
    """REQ-REPORT-8966: every query carries two distinct, recomputable typed losses."""
    raw, value = evaluated
    assert value["completed_count"] == value["independent_count"] == 24
    assert len(value["prediction_rows"]) == 3072
    assert len(value["retention_rows"]) == 36864
    assert len(value["rows"]) == 39936
    assert len(value["streams"]) == 24
    assert all(r["passed"] for r in value["restart_rows"])
    assert all(r["passed"] for r in value["control_rows"])
    assert all(r["pending"] == [29, 30, 31, 32] for r in value["streams"])
    assert value["scheduled_adapter_training_calls"] == 24 * 7 * 3
    assert value["dependent_control_training_calls"] == 3 * 24 * 7 * 3
    assert value["numerical_training_calls"] == 4 * 24 * 7 * 3
    for row in value["rows"]:
        assert row["metrics"]["brier"]["value"] == (row["p_error"] - row["label"]) ** 2
        assert row["metrics"]["classification_error"]["value"] == int(
            (row["p_error"] >= 0.5) != row["label"]
        )
        assert row["cluster_id"] == row["stream_id"]
    assert len({r["unit_id"] for r in value["prediction_rows"]}) == 768


def test_roster_and_resume_refuse_changed_state(stream_request, tmp_path):
    """REQ-REPORT-8966: durable state and its original request must agree."""
    request = stream_request
    with pytest.raises(ValueError, match="stream_roster"):
        w.trajectory(dict(request, unit_ids=["duplicate"] * 32), tmp_path / "bad")
    raw = tmp_path / "resume"
    w.trajectory(request, raw, stop=16)
    changed = dict(request, probabilities=[0.4] * 32)
    with pytest.raises(ValueError, match="resume_state"):
        w.trajectory(changed, raw, resume=True)
    state = json.loads((raw / "state.json").read_text())
    state["query"] = 15
    atomic_json(raw / "state.json", state)
    with pytest.raises(ValueError, match="resume_state"):
        w.trajectory(request, raw, resume=True)


def test_rejected_proposals_preserve_committed_parameters(stream_request, tmp_path):
    """REQ-REPORT-8966: guard rejection cannot leak a candidate into later decisions."""
    request = dict(stream_request, probabilities=[0.9] * 32, guard_rows=[[0.5, 1]] * 64)
    atomic_json(Path(request["label_path"]), dict(labels=[0] * 32))
    state = w.trajectory(request, tmp_path / "rejected")
    rejected = [r for r in state["promotion_rows"] if not r["passed"]]
    assert rejected and all(r["after"] == r["before"] for r in rejected)
    assert all(p == [1.0, 0.0] for p in state["params"].values())


def test_authentication_rejects_malformed_and_changed_operands(tmp_path, monkeypatch):
    """REQ-REPORT-8966: even rehashed producers cannot hide missing or changed operands."""
    name, _, field, task = d.INPUTS[0]
    source = tmp_path / name
    source.parent.mkdir(parents=True)
    source.write_text("not-json")
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(source)["sha256"], field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["artifact_field"] == "producer_json"
    operand = tmp_path / "operand.json"
    atomic_json(operand, dict(original=True))
    value = dict(
        task_id=task,
        required_checks_passed=True,
        **{field: 1},
        source_artifact_hashes=[d.ref(operand)],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    atomic_json(source, value)
    monkeypatch.setattr(d, "INPUTS", [(name, d.ref(source)["sha256"], field, task)])
    operand.write_text("changed")
    failure = d.authenticate(tmp_path)["failures"][0]
    assert failure["path"] == str(operand) and failure["artifact_field"] == "sha256"


def test_child_midpoint_failures_are_bounded(tmp_path, monkeypatch, stream_request):
    """REQ-REPORT-8966: child failure or silence terminates only the owned process."""
    process = subprocess.Popen([sys.executable, "-c", "pass"], stdout=subprocess.PIPE)
    with pytest.raises(ValueError, match="midpoint_child_closed"):
        d.wait_midpoint(process)
    process.wait(timeout=10)
    process.stdout.close()
    with monkeypatch.context() as m:
        moments = iter([0, 121])
        m.setattr(d.time, "monotonic", lambda: next(moments))
        with pytest.raises(TimeoutError, match="midpoint_timeout"):
            d.wait_midpoint(process)
    with monkeypatch.context() as m:
        moments = iter([0, 1, 121])
        m.setattr(d.time, "monotonic", lambda: next(moments))
        m.setattr(d.select, "select", lambda *args: ([], [], []))
        with pytest.raises(TimeoutError, match="midpoint_timeout"):
            d.wait_midpoint(process)
    with monkeypatch.context() as m:
        m.setattr(d, "run_commands", lambda *args, **kwargs: [dict(passed=False)])
        with pytest.raises(ValueError, match="resume_child_failed"):
            d.restart(stream_request, tmp_path / "child-failure")


def test_worker_main_normal_and_midpoint(stream_request, tmp_path, monkeypatch):
    """REQ-REPORT-8966: the worker exposes its midpoint only after durable commit."""
    import io

    path = tmp_path / "request.json"
    atomic_json(path, stream_request)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    assert w.main([str(path), str(tmp_path / "worker"), "--stop16"]) == 0
    assert w.main([str(path), str(tmp_path / "worker"), "--resume"]) == 0


def test_build_keeps_readiness_and_scientific_benefit_separate(evaluated, tmp_path, monkeypatch):
    """REQ-REPORT-8966: ready causal evidence remains null; owned failure disqualifies."""
    work = runner.freeze(d.ROOT, tmp_path / "blocked", True)
    validation = dict(receipts=[], coverage={})
    value = d.build(work, validation, True)
    work["prior_terminal"] = dict(value, honest_verdict="complete_blocked_preserved")
    assert d.build(work, validation, True)["honest_verdict"] == "complete_blocked_preserved"
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d.prior.admission.shard, "qualified", lambda *args: True)
    assert d.build(work, validation, False)["verdict_class"] == "blocked"
    work["evidence"]["failures"] = []
    work["result"] = evaluated[1]
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "null" and value["online_rows_ready_score"] == 1
    assert value["online_benefit_score"] is None and value["independent_generalization_score"] == 0
    work["owned_failure"] = True
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"


def test_replay_recomputes_actual_streams_and_rejects_changed_request(evidence, evaluated):
    """SCENARIO-REPORT-8966-CONTRACT: saved states must equal independent recomputation."""
    raw, value = evaluated
    assert d.evaluate(evidence, raw, replay=True) == value
    path = raw / "streams" / value["streams"][0]["stream_id"] / "request.json"
    original = json.loads(path.read_text())
    atomic_json(path, dict(original, probabilities=[0.5] * 32))
    try:
        with pytest.raises(ValueError, match="request_drift"):
            d.evaluate(evidence, raw, replay=True)
    finally:
        atomic_json(path, original)


def test_evaluation_disqualifies_owned_failure_and_roster(
    evidence, evaluated, tmp_path, monkeypatch
):
    """REQ-REPORT-8966: external absence and owned failures have distinct dispositions."""

    def broken(*args, **kwargs):
        raise ValueError("owned-control-failed")

    copied = deepcopy(evidence)
    corpus = copied["producers"]["exp8963-corpus-admission"]
    corpus["broker_manifest"]["online"]["streams"] = []
    with pytest.raises(ValueError, match="real_response_roster"):
        d.evaluate(copied, tmp_path / "roster")
    scored = json.loads((evaluated[0] / "scoring" / "response.json").read_text())
    scored["labeling_helper_absent"] = False
    with monkeypatch.context() as m:
        m.setattr(d.controls, "child", lambda *args: scored)
        with pytest.raises(ValueError, match="scoring_custody"):
            d.evaluate(evidence, tmp_path / "custody")
    monkeypatch.setattr(d, "evaluate", broken)
    result, failed = runner.evaluation(deepcopy(evidence), tmp_path / "failure")
    assert failed and result["completed_count"] == 0


def test_semantic_replay_and_source_tamper(evaluated, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8966-CONTRACT: rehashing a false result cannot pass."""
    raw = tmp_path / "raw"
    original = d.empty_result()
    original["prediction_rows"] = [deepcopy(evaluated[1]["prediction_rows"][0])]
    original["rows"] = original["prediction_rows"]
    monkeypatch.setattr(d, "evaluate", lambda *args, **kwargs: deepcopy(original))
    work = runner.freeze(d.ROOT, raw, False)
    monkeypatch.setattr(runner, "evaluation", lambda *args: (original, False))
    work["result"] = deepcopy(original)
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    output = tmp_path / (d.NAME + ".json")
    atomic_json(output, d.build(work, validation, True))
    assert runner.replay(output)["passed"]
    work["result"]["prediction_rows"][0]["value"] += 0.5
    atomic_json(raw / "work.json", work)
    atomic_json(output, d.build(work, validation, True))
    assert not runner.replay(output)["passed"]
    value = json.loads(output.read_text())
    value["raw_shard_hashes"][0]["sha256"] = "wrong-bytes"
    atomic_json(output, value)
    assert not runner.replay(output)["passed"]


def test_main_owned_checks_and_one_global_attempt(tmp_path, monkeypatch):
    """REQ-REPORT-8966: owned coverage and one global attempt have separate receipts."""
    original = runner.freeze(d.ROOT, tmp_path / "initial", True)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", lambda *args: deepcopy(original))
    monkeypatch.setattr(
        runner,
        "validation_plan",
        lambda *args, **kwargs: [
            runner.CommandSpec("owned_unit_child", ("private-unit-command",), "owned", 1)
        ],
    )
    private = tmp_path / "private"
    private.mkdir()
    (private / ".coverage").write_text("private orchestration fixture")
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **kwargs: str(private))
    monkeypatch.setattr(runner, "measured", lambda *args: {})
    calls = []

    def commands(*args, **kwargs):
        calls.append(args[1])
        return []

    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "replay", lambda *args: dict(passed=True))
    assert runner.main(["--date", "20261110"]) == 0
    assert len(calls) == 2 and calls[-1][0].name == "full_python_suite"
    assert runner.main(["--date", "20261110"]) == 0
    with pytest.raises(SystemExit):
        runner.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261110", "--fixture-e2e", str(tmp_path / "unsafe.json")])
