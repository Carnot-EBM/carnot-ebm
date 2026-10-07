"""REQ-VERIFY-8245 and REQ-REPORT-8245: private tests give no device credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import polarfire_state_dispatch_8245 as d
from carnot.reporting import polarfire_dispatch_execution_8245 as cli
from carnot.reporting import polarfire_packet_evaluator_8245 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference


def panel():
    """Private probabilities exercise mechanics without substituting natural sources."""
    return dict(
        checks=[],
        references=[],
        cited=[],
        learner_block=[],
        state_origin=dict(kind="private_static_fixture", arm="static", mechanics_only=True),
        state=dict(
            schema_version=1,
            model=dict(
                kind="patch",
                base=dict(kind="input"),
                patches=[dict(delta=0.2, group=dict(interval=[0, 1], reject_only=False))],
            ),
        ),
        queries=[
            dict(unit_id=str(i), source_cluster_id=str(i), p=p, baseline_p=p, baseline_action=a)
            for i, (p, a) in enumerate(
                [(0.99, "accept"), (0.1, "reject"), (None, "escalate"), (0.5, "escalate")]
            )
        ],
    )


def invoke(*args):
    """A real child starts outside the checkout so the script must arrange imports."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_packet_and_cli(tmp_path):
    """SCENARIO-VERIFY-8245-PACKET: exact child hashes and negative inputs are replayable."""
    data = panel()
    packet = d.packet(data, tmp_path)
    assert e.evaluate(packet) == d.expected(data)
    source, output = tmp_path / "fixture.json", tmp_path / (d.NAME + ".json")
    atomic_json(source, data)
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == [] and value["current_device_execution_count"] == 0
    assert value["polarfire_validation_ready_score"] == 1
    assert invoke("--cold-replay", output).returncode == 0
    for key, replacement in [
        ("completed_count", 999),
        ("packet_sha256", "bad"),
        ("reproducibility_checksum", "bad"),
    ]:
        changed = deepcopy(value)
        changed[key] = replacement
        if key != "reproducibility_checksum":
            changed["reproducibility_checksum"] = cli.checksum(changed)
        atomic_json(output, changed)
        assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", source, "--output", d.ROOT / "results" / output.name).returncode == 1
    assert invoke("--input", tmp_path / "absent", "--output", output).returncode == 1
    assert invoke("--evaluate", tmp_path / "absent").returncode == 1


def test_frozen_mypy_executes(tmp_path):
    """SCENARIO-REPORT-8245-CLI: the frozen argv must parse and type check real files."""
    plan = cli.commands(tmp_path)
    typed = next(s for s in plan if s.name == "changed_module_mypy")
    run = subprocess.run(typed.argv, cwd=d.ROOT, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert typed.argv[-len(cli.OWNED) :] == tuple(cli.OWNED)
    assert {"e2e015", "e2e019", "affected_consumers"} <= {s.name for s in plan}


def test_receiver_negatives(tmp_path):
    """SCENARIO-VERIFY-8245-PACKET: version and checksum checks precede scoring."""
    data = panel()
    data["state"]["model"] = dict(
        kind="mixture",
        step=0.25,
        base=data["state"]["model"],
        candidate=dict(kind="global", scale=0, intercept=0),
    )
    packet = d.packet(data, tmp_path)
    assert e.evaluate(packet) == d.expected(data)
    for field, replacement, reason in [
        ("schema", "bad", "packet_schema"),
        ("state_sha256", "bad", "state_hash"),
        ("query_sha256", "bad", "query_hash"),
        ("expected_output_sha256", "bad", "output_hash"),
    ]:
        tampered = dict(packet, **{field: replacement})
        with pytest.raises(ValueError, match=reason):
            e.evaluate(tampered)
    for field, replacement, reason in [
        ("version", 2, "state_version"),
        ("payload_sha256", "bad", "payload_hash"),
    ]:
        envelope = json.loads(packet["state_bytes"])
        envelope[field] = replacement
        tampered = dict(packet, state_bytes=e.canonical(envelope).decode())
        tampered["state_sha256"] = e.digest(tampered["state_bytes"].encode())
        with pytest.raises(ValueError, match=reason):
            e.evaluate(tampered)
    assert e.predict(dict(kind="global", scale=0, intercept=-2), data["queries"][0]) < 0.5
    with pytest.raises(ValueError, match="model_kind"):
        e.predict(dict(kind="bad", base=dict(kind="input")), data["queries"][0])
    with pytest.raises(ValueError, match="probability"):
        e.clip(float("nan"))
    with pytest.raises(ValueError, match="baseline_action"):
        e.action(0.5, "bad")
    assert e.action(0.5, "reject") == "escalate"
    good = tmp_path / "packet.json"
    assert e.main([str(good)]) == 0
    packet["evaluator_sha256"] = "bad"
    atomic_json(good, packet)
    assert e.main([str(good)]) == 1
    run = subprocess.run(
        [sys.executable, str(d.ROOT / d.EVALUATOR), str(good)], capture_output=True, timeout=20
    )
    assert run.returncode == 1


def test_board_routes(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8245-BOARD: one attempt, exact blocking, task-only cleanup."""
    data = panel()
    d.packet(data, tmp_path)
    expected = d.expected(data)
    calls = []

    def fake(plan, raw):
        spec = plan[0]
        calls.append(spec)
        raw.mkdir(parents=True, exist_ok=True)
        out = raw / "stdout"
        out.write_text(
            json.dumps(expected)
            if spec.name == "board_evaluate"
            else "/tmp/carnot8245-test\n"
            if spec.name == "board_python_scratch"
            else ""
        )
        return [
            dict(
                name=spec.name,
                passed=True,
                normal_exit=True,
                actual_exit=0,
                stdout_path=str(out),
                stdout_sha256=e.digest(out.read_bytes()),
            )
        ]

    monkeypatch.setattr(d, "execute", fake)
    result = d.board(tmp_path)
    assert result["executed"] and result["output"] == expected
    assert calls[0].argv == (*d.SSH, "true")
    assert calls[-1].name == "board_cleanup"
    assert all(c.timeout_s <= 60 for c in calls)
    for failure in [
        "board_ssh",
        "board_python_scratch",
        "board_transfer",
        "board_evaluate",
        "board_cleanup",
    ]:
        calls.clear()

        def fail(plan, raw):
            receipts = fake(plan, raw)
            if plan[0].name == failure:
                receipts[0].update(passed=False, actual_exit=255)
            return receipts

        monkeypatch.setattr(d, "execute", fail)
        result = d.board(tmp_path)
        assert result["block"] and not result["ready"]
        assert sum(s.name == "board_ssh" for s in calls) == 1

    def unsafe(plan, raw):
        receipts = fake(plan, raw)
        if plan[0].name == "board_python_scratch":
            Path(receipts[0]["stdout_path"]).write_text("/tmp/unowned\n")
        return receipts

    monkeypatch.setattr(d, "execute", unsafe)
    assert d.board(tmp_path)["block"]["artifact_field"] == "private_remote_directory"
    assert calls[-1].name == "board_python_scratch"


def producer(tmp_path, eid, data):
    """Private signed primaries exercise the existing terminal reader without live results."""
    from carnot.reporting.primary_publication import publish_primary

    suffix = "v712_qualified_delayed_learning" if eid == 8240 else "v711_utility_kernel"
    output = tmp_path / "results" / f"experiment_{eid}_{suffix}.json"
    raw = tmp_path / str(eid)
    raw.mkdir(parents=True, exist_ok=True)
    query = raw / "restart-input.json"
    state = raw / "final_states.json"
    atomic_json(query, dict(rows=data["queries"]))
    atomic_json(state, [dict(local_only=data["state"])])
    measurement = raw / "measurement.json"
    atomic_json(measurement, dict(static=dict(model=data["state"]["model"])))
    side = raw / "terminal.json"
    value = dict(
        experiment_id=eid,
        task_id="exp8240-qualified-delayed-learning" if eid == 8240 else "exp8221-utility-kernel",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        code_config_hashes=[],
        utility_trajectory_ready_score=1,
        static_kernel_ready_score=1,
        validation_receipts=[],
        final_states_path=str(state),
        measurement_reference=reference(measurement),
        raw_shard_hashes=[reference(query), reference(state)],
        terminal_validation_sidecar_path=str(side),
    )
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    atomic_json(side, dict(publication=publication))
    return output


def test_source_qualification_and_fallback(tmp_path, monkeypatch):
    """REQ-VERIFY-8245: qualified current state is preferred and failed learners stay blocked."""
    assert d.load(tmp_path, tmp_path / "empty")["state"] is None
    learner = producer(tmp_path, 8240, panel())
    kernel = producer(tmp_path, 8221, panel())
    monkeypatch.setattr(
        d, "PINS", {8240: reference(learner)["sha256"], 8221: reference(kernel)["sha256"]}
    )
    data = d.load(tmp_path, tmp_path / "raw1")
    assert data["state_origin"]["kind"] == "qualified_current_learner"
    value = json.loads(learner.read_bytes())
    bound = Path(
        json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())["publication"][
            "sidecar_path"
        ]
    )
    report = json.loads(bound.read_bytes())
    report["report"]["passed"] = False
    atomic_json(bound, report)
    data = d.load(tmp_path, tmp_path / "raw2")
    assert data["state_origin"]["mechanics_only"] and data["learner_block"]
    assert data["learner_block"][-1]["artifact_field"] == "terminal.report.passed"
    atomic_json(bound, dict(report, report=dict(passed=True)))
    primitive = Path(value["final_states_path"])
    primitive.write_text("[]")
    data = d.load(tmp_path, tmp_path / "raw3")
    assert data["state_origin"]["kind"] == "qualified_static_fixture"
    assert any(c["artifact_field"] == "sha256" and not c["passed"] for c in data["learner_block"])
    learner.write_text("[]")
    kernel.write_text("{}")
    assert d.load(tmp_path, tmp_path / "raw4")["state"] is None
    learner.unlink()
    learner = producer(tmp_path, 8240, panel())
    monkeypatch.setitem(d.PINS, 8240, reference(learner)["sha256"])
    value = json.loads(learner.read_bytes())
    value["validation_receipts"] = [dict(passed=False, normal_exit=True)]
    atomic_json(learner, value)
    monkeypatch.setitem(d.PINS, 8240, reference(learner)["sha256"])
    # Keep the terminal byte binding valid so a failed owned receipt is the operand.
    report = json.loads(bound.read_bytes())
    report["primary_sha256"] = reference(learner)["sha256"]
    atomic_json(bound, report)
    assert d.load(tmp_path, tmp_path / "raw5")["state"] is None


def test_reductions_and_owned_barrier(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8245-CLI: an owned failure forbids actual board contact."""
    data = panel()
    empty = dict(ready=False, executed=False, output=None, block=None, receipts=[])
    host = d.expected(data)
    assert d.reduce(data, host, empty, True)["verdict_class"] == "null"
    assert d.reduce(data, {}, empty, True)["polarfire_validation_ready_score"] == 0
    device = dict(empty, ready=True, executed=True, output=host)
    assert d.reduce(data, host, device, True)["verdict_class"] == "circular_positive"
    assert d.reduce(data, host, dict(device, output={}), True)["verdict_class"] == "disqualified"
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(
        cli,
        "historical",
        lambda raw: dict(
            receipts=[],
            reproduced=True,
            primary=reference(
                d.ROOT / "results/experiment_8231_v711_polarfire_state_boundary.json"
            ),
            archived_sources=[],
        ),
    )
    monkeypatch.setattr(d, "load", lambda root, raw: panel())
    monkeypatch.setattr(
        d, "board", lambda raw: dict(empty, block=d.operand("board_ssh.actual_exit", raw, 0, 255))
    )
    execute = cli.execute

    def checks(plan, raw):
        if plan and plan[0].name == "repository_health_once":
            return []
        return execute(plan, raw)

    monkeypatch.setattr(cli, "execute", checks)
    output = tmp_path / (d.NAME + ".json")
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert cli.replay(output)["passed"]
    monkeypatch.setattr(cli, "commands", lambda p: [object()])
    monkeypatch.setattr(d, "board", lambda raw: pytest.fail("owned failure contacted board"))
    monkeypatch.setattr(
        cli, "commands", lambda p: [cli.CommandSpec("fail", ("false",), "owned", 1)]
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        cli, "validators", lambda p: [cli.CommandSpec("fail_terminal", ("false",), "terminal", 1)]
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 1


def test_historical_and_missing_resources(tmp_path, monkeypatch):
    """REQ-REPORT-8245: original failures are reproduced while absent resources stay blocked."""
    historical = cli.historical(tmp_path / "historical")
    assert historical["reproduced"]
    assert [r["actual_exit"] for r in historical["receipts"]] == [1, 2]
    with patch.object(e, "digest", return_value="bad"):
        with pytest.raises(ValueError, match="historical_primary_hash"):
            cli.historical(tmp_path / "bad")
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(
        cli,
        "historical",
        lambda raw: dict(
            receipts=[],
            reproduced=True,
            primary=reference(
                d.ROOT / "results/experiment_8231_v711_polarfire_state_boundary.json"
            ),
            archived_sources=[],
        ),
    )
    monkeypatch.setattr(d, "load", lambda root, raw: panel())
    original = cli.execute

    def execute(plan, raw):
        if plan and plan[0].name == "resources_and_scratch":
            return [dict(passed=False, normal_exit=True)]
        return original(plan, raw)

    monkeypatch.setattr(cli, "execute", execute)
    monkeypatch.setattr(d, "board", lambda raw: pytest.fail("missing resources contacted board"))
    source = tmp_path / "input.json"
    output = tmp_path / (d.NAME + ".json")
    atomic_json(source, panel())
    assert cli.main(["--input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["polarfire_validation_ready_score"] == 0
    assert cli.replay(output)["passed"]


def rebind(value, path, field=None):
    """Rehashed tampering must still fail deterministic replay, beyond basic byte checks."""
    value["raw_shard_hashes"] = [
        reference(path) if r["path"] == str(path) else r for r in value["raw_shard_hashes"]
    ]
    if field:
        value[field] = reference(path)
    value["reproducibility_checksum"] = cli.checksum(value)


def test_cold_packet_and_device_tamper(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8245-PACKET: even self-consistent replacement packets fail source parity."""
    source, output = tmp_path / "fixture.json", tmp_path / (d.NAME + ".json")
    data = panel()
    atomic_json(source, data)
    assert invoke("--input", source, "--output", output).returncode == 0
    original = json.loads(output.read_bytes())
    raw = Path(original["packet_path"]).parent
    packet_path = Path(original["packet_path"])
    original_packet = packet_path.read_bytes()
    packet = json.loads(original_packet)
    packet["evaluator_sha256"] = "bad"
    atomic_json(packet_path, packet)
    value = deepcopy(original)
    value["packet_sha256"] = reference(packet_path)["sha256"]
    rebind(value, packet_path)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="evaluator_hash"):
        cli.replay(output)
    changed = deepcopy(data)
    changed["queries"][0]["p"] = 0.2
    d.packet(changed, raw)
    value["packet_sha256"] = reference(packet_path)["sha256"]
    rebind(value, packet_path)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="packet_reference_drift"):
        cli.replay(output)
    packet_path.write_bytes(original_packet)
    value = deepcopy(original)
    stdout = tmp_path / "board.stdout"
    atomic_json(stdout, d.expected(data))
    device = dict(
        ready=True,
        executed=True,
        output=d.expected(data),
        block=None,
        receipts=[
            dict(
                name="board_evaluate",
                stdout_path=str(stdout),
                stdout_sha256=reference(stdout)["sha256"],
            )
        ],
    )
    board_path = Path(value["board_reference"]["path"])
    atomic_json(board_path, device)
    data["fixture"] = True
    value.update(d.reduce(data, d.expected(data), device, True))
    rebind(value, board_path, "board_reference")
    atomic_json(output, value)
    assert cli.replay(output)["passed"]
    atomic_json(stdout, dict(tampered=True))
    device["receipts"][0]["stdout_sha256"] = reference(stdout)["sha256"]
    atomic_json(board_path, device)
    rebind(value, board_path, "board_reference")
    atomic_json(output, value)
    with pytest.raises(ValueError, match="board_transcript_drift"):
        cli.replay(output)
    assert (
        d.reduce(data, d.expected(data), dict(device, output={}), True)[
            "polarfire_validation_ready_score"
        ]
        == 0
    )


def test_source_receipts_empty_queries_and_invalid_board_json(tmp_path, monkeypatch):
    """REQ-VERIFY-8245: receipt bytes and empty query schema remain explicit operands."""
    data = panel()
    data["queries"] = []
    learner = producer(tmp_path, 8240, data)
    monkeypatch.setattr(d, "PINS", {8240: reference(learner)["sha256"], 8221: "bad"})
    assert d.load(tmp_path, tmp_path / "raw-empty")["state"] is None
    value = json.loads(learner.read_bytes())
    query = next(
        r for r in value["raw_shard_hashes"] if Path(r["path"]).name == "restart-input.json"
    )
    atomic_json(Path(query["path"]), dict(rows=panel()["queries"]))
    query.update(reference(Path(query["path"])))
    stream = tmp_path / "validation.log"
    stream.write_text("owned pass")
    value["validation_receipts"] = [
        dict(
            passed=True,
            normal_exit=True,
            **{s + "_path": str(stream) for s in ["stdout", "stderr"]},
            **{s + "_sha256": reference(stream)["sha256"] for s in ["stdout", "stderr"]},
        )
    ]
    atomic_json(learner, value)
    monkeypatch.setitem(d.PINS, 8240, reference(learner)["sha256"])
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    bound = Path(terminal["publication"]["sidecar_path"])
    report = json.loads(bound.read_bytes())
    report["primary_sha256"] = reference(learner)["sha256"]
    atomic_json(bound, report)
    assert d.load(tmp_path, tmp_path / "raw-receipts")["state"] is not None

    def badjson(plan, raw):
        raw.mkdir(parents=True, exist_ok=True)
        stdout = raw / "stdout"
        stdout.write_text(
            "/tmp/carnot8245-fixture" if plan[0].name == "board_python_scratch" else "bad-json"
        )
        return [
            dict(
                name=plan[0].name,
                passed=True,
                normal_exit=True,
                actual_exit=0,
                stdout_path=str(stdout),
                stdout_sha256=reference(stdout)["sha256"],
            )
        ]

    monkeypatch.setattr(d, "execute", badjson)
    device = d.board(tmp_path / "board")
    assert device["executed"] and device["block"]["artifact_field"] == "board_output_schema"


def test_qualified_upstream_rejects_unsupported_state_schema(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8245-PACKET: a valid producer cannot grant a new state version."""
    data = panel()
    data["state"]["schema_version"] = 2
    learner = producer(tmp_path, 8240, data)
    monkeypatch.setattr(d, "PINS", {8240: reference(learner)["sha256"], 8221: "bad"})
    loaded = d.load(tmp_path, tmp_path / "raw")
    assert loaded["state"] is None
    failed = [c for c in loaded["checks"] if not c["passed"]]
    assert any(
        c["artifact_field"] == "required_input_schema_and_hash" and c["observed"] == "state_version"
        for c in failed
    )


def test_health_receipt_reuse_and_rejection(tmp_path):
    """SCENARIO-REPORT-8245-CLI: a corrected attempt reuses one byte-bound global diagnostic."""
    source, output = tmp_path / "fixture.json", tmp_path / (d.NAME + ".json")
    atomic_json(source, panel())
    stream = tmp_path / "health.stdout"
    stream.write_text("known collection failure\n")
    cached = tmp_path / "previous.json"
    receipt = dict(
        name="repository_health_once",
        command_argv=[str(d.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        actual_exit=2,
        expected_exit=0,
        passed=False,
        normal_exit=True,
        duration_s=1.0,
        started_monotonic_ns=1,
        ended_monotonic_ns=1000000001,
        **{s + "_path": str(stream) for s in ["stdout", "stderr"]},
        **{s + "_sha256": reference(stream)["sha256"] for s in ["stdout", "stderr"]},
    )
    atomic_json(cached, dict(repository_health=[receipt]))
    run = invoke("--input", source, "--output", output, "--health-receipt", cached)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["repository_health"] == [receipt] and value["required_checks_passed"]
    assert value["repository_health_source"]["sha256"] == reference(cached)["sha256"]
    assert invoke("--cold-replay", output).returncode == 0
    atomic_json(cached, dict(repository_health=[]))
    assert invoke("--input", source, "--output", output, "--health-receipt", cached).returncode == 1
    assert (
        invoke(
            "--input", source, "--output", output, "--health-receipt", tmp_path / "absent"
        ).returncode
        == 1
    )
