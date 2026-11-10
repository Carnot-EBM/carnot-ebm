"""REQ-REPORT-8968 / SCENARIO-REPORT-8968-CONTRACT: complete cached service costs."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.memory import complete_service_worker_8968 as w
from carnot.reporting import complete_service_cost_8968 as d
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8968: use only authenticated frozen real-response operands."""
    value = d.authenticate(d.ROOT)
    assert value["failures"] == []
    return value


@pytest.fixture(scope="module")
def requests(evidence, tmp_path_factory):
    """REQ-REPORT-8968: freeze a paired census before any service runs."""
    raw = tmp_path_factory.mktemp("service-cost")
    return raw, d.prepare(evidence, raw)


def test_roster_is_fresh_paired_and_balanced(requests):
    """REQ-REPORT-8968: process seeds and shared corpus streams never multiply n."""
    raw, plan = requests
    roster = plan["roster"]
    assert len(roster) == 96 and len({s["seed"] for s in roster}) == 24
    assert len({s["cluster_id"] for s in roster}) == 24
    for arm in w.ARMS:
        assert (
            sorted(s["order"] for s in roster if s["arm"] == arm)
            == [0] * 6 + [1] * 6 + [2] * 6 + [3] * 6
        )
    assert d.prepare(evidence=None, raw=raw) == plan


def test_ordinary_score_matches_existing_frozen_ensemble(requests):
    """REQ-REPORT-8968: confidence service cannot pay artificial graph encoding."""
    from carnot.autoresearch import relational_graph_encoder as g

    raw, plan = requests
    request = d.request_for(plan, plan["roster"][0])
    service = json.loads(Path(request["service_path"]).read_bytes())
    full = json.loads(Path(plan["service_reference"]["path"]).read_bytes())
    row = json.loads(plan["streams"][0]["rows"][0])
    expected = float(
        g.ensemble(
            full["states"]["confidence_logistic"],
            [row],
            full["temperatures"]["confidence_logistic"],
        )[0]
    )
    small = dict(
        states=full["states"]["confidence_logistic"],
        temperature=full["temperatures"]["confidence_logistic"],
    )
    assert w.confidence(small, row["features"]) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        w.confidence(small, dict(row["features"], label=1))
    with pytest.raises(ValueError):
        w.confidence(small, {k: -1 for k in row["features"]})


@pytest.mark.parametrize("arm", w.ARMS)
def test_service_restart_drain_and_real_operations(requests, tmp_path, arm):
    """REQ-REPORT-8968:32 complete requests and the pending tail belong in costs."""
    raw, plan = requests
    spec = next(s for s in plan["roster"] if s["arm"] == arm)
    request = d.request_for(plan, spec)
    atomic_json(tmp_path / "request.json", request)
    first = w.run(tmp_path / "request.json", 16)
    resumed = w.run(tmp_path / "request.json", 32)
    atomic_json(tmp_path / "control/request.json", request)
    full = w.run(tmp_path / "control/request.json", 32)
    assert resumed["initial_state"] == first["state"]
    assert resumed["state"] == full["state"]
    state = resumed["state"]
    assert len(state["rows"]) == 32 and state["query"] == 32 and not state["pending"]
    assert state["counts"]["sampling_call_count"] == 0
    assert state["counts"]["updates"] == (8 if arm in ("bias_only", "adaptive") else 0)
    assert state["counts"]["guard_evaluations"] == (
        8 * 64 * 2 if arm in ("bias_only", "adaptive") else 0
    )
    assert state["counts"]["graph_encoded_rows"] == (
        0 if arm == "confidence_logistic" else 3 * (96 if arm in ("bias_only", "adaptive") else 32)
    )
    assert first["storage"]["file_fsync"] == first["storage"]["directory_fsync"] > 0
    assert all(r["released_after"] >= r["ordinals"][-1] for r in state["feedback_rows"])
    bad = deepcopy(request)
    bad["rows"] = []
    atomic_json(tmp_path / "bad/request.json", bad)
    with pytest.raises(ValueError, match="request_roster"):
        w.run(tmp_path / "bad/request.json", 32)
    atomic_json(tmp_path / "state.json", dict(state, request_hash="changed"))
    with pytest.raises(ValueError, match="resume_state"):
        w.run(tmp_path / "request.json", 32)


def test_seals_labels_and_bounded_children(requests, tmp_path, monkeypatch):
    """REQ-REPORT-8968: trust boundaries reject altered seals and malformed labels."""
    _, plan = requests
    req = d.request_for(plan, next(s for s in plan["roster"] if s["arm"] == "adaptive"))
    seal = tmp_path / "seal.json"
    atomic_json(seal, dict(query=8, rows=[], prediction_hash="wrong"))
    with pytest.raises(ValueError, match="prediction_seal"):
        w.release(req, seal, 8)
    rows = [dict(query=q) for q in range(8)]
    atomic_json(seal, dict(query=8, rows=rows, prediction_hash=d.canonical_hash(rows)))
    atomic_json(tmp_path / "labels.json", dict(labels=[True] * 32))
    with pytest.raises(ValueError, match="online_label_schema"):
        w.release(dict(req, label_path=str(tmp_path / "labels.json")), seal, 8)
    assert w.release(req, seal, 1) == []
    py = str(d.ROOT / ".venv/bin/python")
    receipt = d.child([py, "-u", "-c", "print('child forwarded')"], tmp_path / "ok.log")
    assert receipt["passed"] and receipt["peak_memory_bytes"] > 0
    assert "child forwarded" in (tmp_path / "ok.log").read_text()
    assert not d.child([py, "-c", "raise SystemExit(3)"], tmp_path / "failed.log")["passed"]
    original = d.time.monotonic
    ticks = iter([original(), original() + 16, original() + 32, original() + 48])
    monkeypatch.setattr(d.time, "monotonic", lambda: next(ticks, original() + 64))
    timed = d.child(
        [py, "-c", "import time; time.sleep(5)"], tmp_path / "timeout.log", timeout=0.01
    )
    assert timed["timed_out"] and not timed["passed"]


def test_real_parent_restart_cell_and_reduction(requests, tmp_path):
    """REQ-REPORT-8968: whole-process clocks, memory and durable counters reduce exactly."""
    _, plan = requests
    spec = next(s for s in plan["roster"] if s["arm"] == "confidence_logistic")
    cell = d.execute(tmp_path / spec["cluster_id"] / spec["arm"], spec, plan)
    reduced = d.reduce(tmp_path)
    assert reduced["completed_count"] == 1 and reduced["censored_count"] == 95
    row = next(r for r in reduced["rows"] if r["status"] == "completed")
    assert (
        row["value"]
        == (
            cell["commands"][-1]["ended_monotonic_ns"] - cell["commands"][0]["started_monotonic_ns"]
        )
        / 1e9
    )
    bad = deepcopy(cell)
    bad["segments"][1]["initial_state"]["query"] = 15
    with pytest.raises(ValueError, match="restart_parity"):
        d.verify(bad)
    bad = deepcopy(cell)
    bad["segments"][1]["state"]["pending"] = [32]
    with pytest.raises(ValueError, match="complete_responses"):
        d.verify(bad)
    bad = deepcopy(cell)
    bad["segments"][1]["state"]["counts"]["updates"] = 1
    with pytest.raises(ValueError, match="operation_counts"):
        d.verify(bad)
    bad = deepcopy(cell)
    bad["commands"][0]["peak_memory_bytes"] = 0
    with pytest.raises(ValueError, match="parent_process_receipts"):
        d.verify(bad)
    bad = deepcopy(cell)
    bad["segments"][0]["storage"]["file_fsync"] = 0
    with pytest.raises(ValueError, match="durability"):
        d.verify(bad)


def test_paired_intervals_keep_all_scheduled_units(requests, tmp_path, monkeypatch):
    """REQ-REPORT-8968: timing fixtures test arithmetic, never become research measurements."""
    _, plan = requests
    for spec in plan["roster"]:
        arm = spec["arm"]
        online = arm in ("bias_only", "adaptive")
        state = dict(
            query=32,
            rows=[{}] * 32,
            pending=[],
            counts=dict(
                updates=8 if online else 0,
                guard_evaluations=1024 if online else 0,
                graph_encoded_rows=288 if online else 0 if arm == "confidence_logistic" else 96,
            ),
        )
        first = dict(query=16)
        segments = [
            dict(
                pid=1, state=first, storage=dict(durable_bytes=10, file_fsync=1, directory_fsync=1)
            ),
            dict(
                pid=2,
                initial_state=first,
                state=state,
                storage=dict(durable_bytes=20, file_fsync=1, directory_fsync=1),
            ),
        ]
        duration = 1 if arm == "confidence_logistic" else 2
        commands = [
            dict(
                passed=True,
                peak_memory_bytes=100,
                pid=i,
                started_monotonic_ns=i * 10**9,
                ended_monotonic_ns=(i + duration) * 10**9,
            )
            for i in (1, 2)
        ]
        atomic_json(
            tmp_path / spec["cluster_id"] / arm / "unit.json",
            dict(spec=spec, segments=segments, commands=commands),
        )
    result = d.reduce(tmp_path)
    assert result["completed_count"] == 96 and result["independent_count"] == 24
    assert all(
        r["independent_count"] == 24 and r["mean"] == 1 for r in result["paired_cost_intervals"]
    )
    spec = plan["roster"][0]
    directory = tmp_path / spec["cluster_id"] / spec["arm"]
    (directory / "unit.json").unlink()
    atomic_json(directory / "failure.json", dict(reason="measured_failure"))
    result = d.reduce(tmp_path)
    assert result["failed_count"] == 1 and result["independent_count"] == 23
    assert d.empty_result()["censored_count"] == 96
    monkeypatch.setattr(d, "roster", lambda: [spec])
    monkeypatch.setattr(d, "prepare", lambda evidence, raw: dict(roster=[spec]))
    monkeypatch.setattr(d, "execute", lambda *a: (_ for _ in ()).throw(ValueError("owned_failure")))
    assert d.measure({}, tmp_path / "failed")["failed_count"] == 1
    monkeypatch.setattr(d, "execute", lambda *a: None)
    assert d.measure({}, tmp_path / "censored")["censored_count"] == 1


@pytest.mark.parametrize("gate", [None, 0, False, True])
def test_external_diagnostics_are_typed(tmp_path, monkeypatch, gate):
    """REQ-REPORT-8968: absent, zero and boolean gates remain distinct failures."""
    name, pin, field, task = d.INPUTS[0]
    path = tmp_path / name
    value = dict(task_id=task, required_checks_passed=True)
    if gate is not None:
        value[field] = gate
    atomic_json(path, value)
    monkeypatch.setattr(d.auth, "INPUTS", [(name, d.ref(path)["sha256"], field, task)])
    failure = d.authenticate(tmp_path)["failures"][0]
    assert (
        failure["artifact_field"] == field
        and failure["observed"] == gate
        and failure["expected"] == 1
    )
    assert failure["check"] == (
        "missing_field" if gate is None else "observed_zero" if gate == 0 else "rejected_producer"
    )


def test_missing_malformed_and_operand_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8968: a repaired producer hash cannot authorize changed operands."""
    name, pin, field, task = d.INPUTS[0]
    monkeypatch.setattr(d.auth, "INPUTS", [(name, pin, field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["check"] == "missing_file"
    path = tmp_path / name
    path.parent.mkdir(parents=True)
    path.write_text("invalid")
    monkeypatch.setattr(d.auth, "INPUTS", [(name, d.ref(path)["sha256"], field, task)])
    assert d.authenticate(tmp_path)["failures"][0]["artifact_field"] == "producer_json"
    operand = tmp_path / "operand.json"
    atomic_json(operand, dict(original=True))
    atomic_json(
        path,
        dict(
            task_id=task,
            required_checks_passed=True,
            **{field: 1},
            source_artifact_hashes=[d.ref(operand)],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(d.auth, "INPUTS", [(name, d.ref(path)["sha256"], field, task)])
    operand.write_text("changed")
    assert d.authenticate(tmp_path)["failures"]


def test_build_gates_and_preserves_terminal(requests, tmp_path, monkeypatch):
    """REQ-REPORT-8968: null scientific benefit never erases qualified cost rows."""
    from carnot.reporting import complete_service_cost_runner_8968 as runner

    _, plan = requests
    raw = tmp_path
    atomic_json(raw / "plan.json", plan)
    result = d.empty_result()
    result.update(
        completed_count=96,
        censored_count=0,
        independent_count=24,
        durability_rows=[{}] * 96,
        paired_cost_intervals=[dict(arm="adaptive", independent_count=24, upper=-1)],
    )
    work = dict(
        result=result,
        evidence=dict(failures=[], references=[]),
        owned_failure=False,
        raw=str(raw),
        started=1,
        ended=1000000001,
        phase_spans=[],
        sources=[],
        raw_references=[],
        global_test_debt={},
    )
    validation = dict(receipts=[], coverage={})
    monkeypatch.setattr(d.upstream.prior.admission.shard, "qualified", lambda *args: True)
    value = d.build(work, validation, False)
    assert value["cost_rows_ready_score"] == 1 and not value["deployment_recommendation"]
    assert value["verdict_class"] == "null" and value["duration_s"] == 1
    plan["quality"] = dict(
        detector_benefit_score=1, relational_controls_ready_score=1, online_benefit_score=1
    )
    atomic_json(raw / "plan.json", plan)
    assert d.build(work, validation, False)["deployment_recommendation"]
    plan["quality"]["online_benefit_score"] = True
    atomic_json(raw / "plan.json", plan)
    assert not d.build(work, validation, False)["deployment_recommendation"]
    work["owned_failure"] = True
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    work["owned_failure"] = False
    work["evidence"]["failures"] = [dict(check="missing_file")]
    work["prior_terminal"] = dict(
        verdict_class="blocked",
        honest_verdict="complete_blocked_prior_scope",
        gate_check_summary=work["evidence"]["failures"],
    )
    assert d.build(work, validation, False)["honest_verdict"] == "complete_blocked_prior_scope"
    monkeypatch.setattr(d.upstream.prior.admission.shard, "qualified", lambda *args: False)
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    assert runner.evaluation(dict(failures=[{}]))[1] is False
    monkeypatch.setattr(d, "measure", lambda *args: d.empty_result())
    assert (
        runner.evaluation(
            dict(failures=[], references=[dict(path=str(tmp_path / "inputs/results/x.json"))])
        )[1]
        is True
    )
    monkeypatch.setattr(
        d, "measure", lambda *args: (_ for _ in ()).throw(ValueError("control_failure"))
    )
    assert (
        runner.evaluation(
            dict(failures=[], references=[dict(path=str(tmp_path / "inputs/results/x.json"))])
        )[1]
        is True
    )


def test_private_cli_cold_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8968-CONTRACT: foreign-directory publication and replay refuse altered aggregates."""
    import os
    import subprocess
    import sys
    from carnot.reporting import complete_service_cost_runner_8968 as runner

    output = tmp_path / (d.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(d.ROOT / d.CLI)]

    def invoke(*args):
        return subprocess.run(
            command + list(args), cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )

    published = invoke(
        "--date", "20261110", "--sources", str(tmp_path / "absent"), "--private-output", str(output)
    )
    assert published.returncode == 0, published.stdout + published.stderr
    assert invoke("--cold-replay", str(output)).returncode == 0
    value = json.loads(output.read_bytes())
    assert (
        value["fixture_only"]
        and value["verdict_class"] == "blocked"
        and value["cost_rows_ready_score"] == 0
    )
    value["rows"][0]["value"] = 3
    value["reproducibility_checksum"] = d.upstream.calibration.checksum(value)
    atomic_json(output, value)
    assert invoke("--cold-replay", str(output)).returncode == 1
    assert not runner.replay(output)["passed"]
    output.write_text("broken")
    assert not runner.replay(output)["passed"]
    assert invoke("--date", "wrong").returncode != 0


def test_production_orchestration_checks_are_mandatory(tmp_path, monkeypatch):
    """REQ-REPORT-8968: fake command receipts test routing only; production reruns real checks."""
    from carnot.reporting import complete_service_cost_runner_8968 as runner

    monkeypatch.setattr(d, "ROOT", tmp_path)

    def freeze(root, raw):
        raw.mkdir(parents=True)
        return dict(
            raw=str(raw),
            result=d.empty_result(),
            evidence=dict(failures=[], references=[]),
            owned_failure=False,
            sources=[],
            started=runner.time.monotonic_ns(),
            ended=runner.time.monotonic_ns(),
            phase_spans=[],
            raw_references=[],
            global_test_debt={},
        )

    monkeypatch.setattr(runner.runtime, "freeze", freeze)

    def plan(private, producer):
        (private / ".coverage").touch()
        return [
            runner.CommandSpec("owned_unit_child", ("fixture",), "owned"),
            runner.CommandSpec("coverage_json", ("fixture",), "owned"),
        ]

    monkeypatch.setattr(runner, "validation_plan", plan)
    invoked = []

    def commands(root, planned, *, log_dir, heartbeat_s):
        log_dir.mkdir(parents=True)
        (log_dir / "check.log").write_text("private routing fixture")
        invoked.extend(c.name for c in planned)
        return [dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in planned]

    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(runner, "measured", lambda *a: {})
    published = []
    monkeypatch.setattr(
        runner, "publish_primary", lambda output, value, validator: published.append(value) or {}
    )
    assert runner.main(["--date", "20261110"]) == 0
    assert invoked == [
        "owned_unit_child",
        "coverage_json",
        "full_python_suite",
        "cold_replay_cli",
        "adversarial",
        "strict_rows",
    ]
    assert published[0]["cost_rows_ready_score"] == 0
    assert published[0]["verdict_class"] == "disqualified"
    with pytest.raises(SystemExit):
        runner.main(["--date", "wrong"])
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261110", "--private-output", str(tmp_path / "inside.json")])


def test_replay_refuses_drift_and_requires_measured_coverage(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8968-CONTRACT: owned replay refuses a fabricated coverage claim."""
    from carnot.reporting import complete_service_cost_runner_8968 as runner
    from carnot.reporting.primary_publication import publish_primary

    monkeypatch.setattr(d, "authenticate", lambda root: dict(failures=[], references=[]))
    work = dict(
        raw=str(tmp_path),
        result=d.empty_result(),
        evidence=dict(failures=[], references=[]),
        owned_failure=False,
        sources=[],
        started=1,
        ended=2,
        phase_spans=[],
        raw_references=[],
        global_test_debt={},
    )
    validation = dict(receipts=[], coverage={})
    atomic_json(tmp_path / "work.json", work)
    atomic_json(tmp_path / "validation.json", validation)
    value = d.build(work, validation, True)
    target = tmp_path / (d.NAME + ".json")
    publish_primary(target, value, runner.replay)
    assert runner.replay(target)["passed"]
    value["cost_rows_ready_score"] = 1
    value["reproducibility_checksum"] = d.upstream.calibration.checksum(value)
    atomic_json(target, value)
    assert not runner.replay(target)["passed"]
    value["cost_rows_ready_score"] = 0
    value["code_config_hashes"] = [
        dict(d.ref(Path(d.__file__)), executed_source=True, source_path="/nonexistent/changed")
    ]
    value["reproducibility_checksum"] = d.upstream.calibration.checksum(value)
    atomic_json(target, value)
    assert not runner.replay(target)["passed"]
    work["result"]["completed_count"] = 1
    atomic_json(tmp_path / "work.json", work)
    value = d.build(work, validation, True)
    atomic_json(target, value)
    assert not runner.replay(target)["passed"]
