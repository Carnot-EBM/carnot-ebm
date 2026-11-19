"""REQ-REPORT-9139-V779 / SCENARIO-REPORT-9139-V779: bounded schedule custody."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import validation_schedule_9139 as s
from carnot.reporting.current_work_receipt import atomic_json


def history() -> dict[str, Any]:
    nodes = ["a.py::test_a", "a.py::test_b", "b.py::test_c"]
    return dict(
        nodes=nodes,
        events=[
            dict(nodeid=nodes[0], when=p, outcome="passed", duration=0.1)
            for p in ["setup", "call", "teardown"]
        ],
        required_command_union=[
            dict(name="owned_tests", argv=["pytest", "a.py"]),
            dict(name="direct_consumers", argv=["pytest", "b.py"]),
        ],
        original_assertion_references=[],
    )


def test_budget_union_fixture_and_unknown_bounds() -> None:
    assert 2100 - 1800 == 300
    plan = s.schedule(history())
    assert plan["budget_equation"]["hard_cap_s"] == 3300 + 1200 + 300 == 4800
    assert plan["budget_equation"]["verification_reservations"] == 1
    assert sorted(n for p in plan["execution_graph"] for n in p["node_ids"]) == history()["nodes"]
    assert len(plan["measured_cost_rows"]) == 1 and len(plan["unknown_cost_rows"]) == 2
    assert plan["budget_equation"]["projected_total_s"] > 60
    assert all(len({n.split("::")[0] for n in p["node_ids"]}) == 1 for p in plan["execution_graph"])
    changed = history()
    changed["nodes"] *= 2
    with pytest.raises(ValueError, match="duplicate_node"):
        s.schedule(changed)
    huge = history()
    huge["nodes"] = [f"a.py::test_{i}" for i in range(200)]
    huge["events"] = []
    assert s.schedule(huge)["budget_equation"]["feasible"] is False


@pytest.mark.parametrize("observed", ["missing_file", "missing_field", 0, True, 1])
def test_gate_operands(tmp_path: Path, monkeypatch: Any, observed: Any) -> None:
    path = tmp_path / s.CONTRACT
    if observed != "missing_file":
        atomic_json(
            path, {} if observed == "missing_field" else dict(contract_ready_score=observed)
        )
        monkeypatch.setitem(s.PINS, s.CONTRACT, s.ref(path)["sha256"])
    work = s.authenticate(tmp_path, tmp_path / "raw")
    operand = work["checks"][0 if observed == "missing_file" else 1]
    assert operand["observed"] == observed
    assert type(operand["observed"]) is type(observed)
    assert not all(c["passed"] for c in work["checks"])


@pytest.fixture(scope="module")
def actual_bundle(tmp_path_factory: Any) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("schedule")
    source = s.ROOT / "tests/python/test_forward_evidence_9111.py"
    nodes = [
        "tests/python/test_forward_evidence_9111.py::test_no_load_and_historical_scope",
        "tests/python/test_forward_evidence_9111.py::test_forward_event_ledger",
    ]
    h = dict(
        nodes=nodes,
        events=[],
        required_command_union=[],
        original_assertion_references=[s.ref(source)],
    )
    plan = s.schedule(h)
    return s.execute(plan, root / "run", "owned-invocation", [s.ref(source)])


def test_actual_partition_consumed_without_rerun(actual_bundle: dict[str, Any]) -> None:
    result = s.consume(actual_bundle, "owned-invocation")
    assert result["completed_count"] == 2
    assert result["duplicate_execution_count"] == 0
    assert result["rerun_commands"] == []
    assert all(r["status"] == "completed" for r in result["rows"])


@pytest.mark.parametrize(
    "mutation",
    [
        "invocation",
        "nodes",
        "duplicate",
        "missing",
        "failed",
        "command",
        "assertion",
        "code",
        "stale",
        "log",
        "events",
    ],
)
def test_rejects_changed_partition_evidence(actual_bundle: dict[str, Any], mutation: str) -> None:
    value = deepcopy(actual_bundle)
    part = value["partitions"][0]
    if mutation == "invocation":
        value["invocation_id"] = "foreign"
    elif mutation == "nodes":
        part["node_ids"].append("foreign::test")
    elif mutation == "duplicate":
        value["partitions"].append(deepcopy(part))
    elif mutation == "missing":
        value["partitions"] = []
    elif mutation == "failed":
        part["receipt"]["exit_code"] = 1
    elif mutation == "command":
        part["receipt"]["argv"].append("--changed")
    elif mutation == "assertion":
        value["schedule"]["original_assertion_hashes"][0]["sha256"] = "bad"
    elif mutation == "code":
        value["code_config_hashes"][0]["sha256"] = "bad"
    elif mutation == "stale":
        part["receipt"]["started_monotonic_ns"] = 0
    elif mutation == "log":
        part["receipt"]["stdout_sha256"] = "bad"
    else:
        part["events_reference"]["sha256"] = "bad"
    with pytest.raises((ValueError, KeyError)):
        s.consume(value, "owned-invocation")


def test_timeout_preserves_censored_nodes(tmp_path: Path) -> None:
    source = tmp_path / "test_wait.py"
    source.write_text("import time\ndef test_wait():\n    time.sleep(3)\n")
    h = dict(
        nodes=[str(source) + "::test_wait"],
        events=[],
        required_command_union=[],
        original_assertion_references=[s.ref(source)],
    )
    plan = s.schedule(h)
    bundle = s.execute(plan, tmp_path / "run", "timeout", [s.ref(source)], node_seconds=1)
    assert bundle["partitions"][0]["receipt"]["exit_code"] == 124
    with pytest.raises(ValueError, match="partition_exit"):
        s.consume(bundle, "timeout")


def test_foreign_and_duplicate_phase_controls() -> None:
    events = [
        dict(nodeid="a::x", when=p, outcome="passed", duration=0.1)
        for p in ["start", "setup", "call", "teardown"]
    ]
    assert s.reduce_nodes(["a::x"], events)[0]["status"] == "completed"
    with pytest.raises(ValueError, match="phase_evidence"):
        s.reduce_nodes(["b::x"], events)
    with pytest.raises(ValueError, match="phase_evidence"):
        s.reduce_nodes(["a::x"], events + events[:1])
    assert s.reduce_nodes(["a::x"], [])[0]["status"] == "censored"
    events[2]["outcome"] = "failed"
    assert s.reduce_nodes(["a::x"], events)[0]["status"] == "failed"


def test_current_runner_seam_and_private_cli(actual_bundle: dict[str, Any], tmp_path: Path) -> None:
    """SCENARIO-REPORT-9139-V779: normal orchestration consumes completed child evidence."""
    from carnot.reporting import validation_schedule_runner_9139 as r

    bundle = tmp_path / "bundle.json"
    atomic_json(bundle, actual_bundle)
    command = r.foreign_argv(
        s.prior.old.CLI, "--schedule-v779", str(bundle), "--schedule-invocation", "owned-invocation"
    )
    receipt = s.prior.run_check("normal_child", command, tmp_path / "logs")
    assert receipt["passed"]
    assert json.loads((tmp_path / "consumed.json").read_bytes())["completed_count"] == 2
    assert s.orchestration(bundle, "owned-invocation")["rerun_commands"] == []
    output = tmp_path / (s.NAME + ".json")
    refused = r.foreign_argv(
        s.CLI, "--input-root", str(tmp_path / "absent"), "--output", str(output)
    )
    assert s.prior.run_check("refusal", refused, tmp_path / "logs")["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["schedule_ready_score"] == 0
    assert value["current_llm_invocations"] == value["generated_tokens"] == 0
    assert s.prior.run_check(
        "cold", r.foreign_argv(s.CLI, "--replay", str(output)), tmp_path / "logs"
    )["passed"]
    value["schedule_ready_score"] = 1
    atomic_json(output, value)
    assert s.prior.run_check(
        "tamper", r.foreign_argv(s.CLI, "--replay", str(output)), tmp_path / "logs", expected=1
    )["passed"]
    assert r.main(["--date", "bad"]) == 1
    assert r.main(["--replay", str(tmp_path / "absent")]) == 1


def test_budget_refuses_science_without_full_qualification(actual_bundle: dict[str, Any]) -> None:
    from carnot.reporting import validation_schedule_runner_9139 as r

    work = dict(
        checks=[dict(passed=True)],
        references=[],
        receipts=[],
        code_config_hashes=[],
        schedule=actual_bundle["schedule"],
        bundle=actual_bundle,
        coverage={},
        isolated_child_receipts=[],
    )
    value = r.build(work, "20261116", 0, 1)
    assert value["schedule_ready_score"] == 0 and value["required_checks_passed"] is False
    assert value["model_work_authorized"] is False
    assert value["full_621_node_qualification_score"] == 0
    assert value["sample_size_budget"]["effective_independent_groups"] == 0


def test_real_authentication_and_historical_drift(tmp_path: Path, monkeypatch: Any) -> None:
    work = s.authenticate(s.ROOT, tmp_path / "auth")
    assert all(c["passed"] for c in work["checks"])
    assert len(work["schedule"]["original_node_manifest"]) == 621
    assert len(work["schedule"]["measured_cost_rows"]) == 48
    assert len(work["schedule"]["unknown_cost_rows"]) == 573
    assert work["schedule"]["budget_equation"]["feasible"] is False
    root = tmp_path / "changed"
    atomic_json(root / s.CONTRACT, dict(contract_ready_score=1))
    value = deepcopy(work["historical"])
    value["replay_input"]["work"]["nodes"].pop()
    atomic_json(root / s.PREVIOUS, value)
    for name in [s.CONTRACT, s.PREVIOUS]:
        monkeypatch.setitem(s.PINS, name, s.ref(root / name)["sha256"])
    with pytest.raises(ValueError, match="historical_node_evidence"):
        s.authenticate(root, tmp_path / "invalid")
    monkeypatch.setattr(s, "HISTORICAL_READER", tmp_path / "missing_reader")
    refused = s.authenticate(root, tmp_path / "drift")
    assert any(c["observed"] == "missing_file" for c in refused["checks"])
    assert "schedule" not in refused


def test_schedule_infeasibility_and_time_boundary(tmp_path: Path, monkeypatch: Any) -> None:
    plan = s.schedule(history())
    plan["budget_equation"]["feasible"] = False
    with pytest.raises(ValueError, match="schedule_infeasible"):
        s.execute(plan, tmp_path, "no_launch", [])
    plan["budget_equation"]["feasible"] = True
    plan["verification_graph"] = [dict(name="must_not_launch", argv=["false"])]
    ticks = iter([0, 3301, 3301])
    monkeypatch.setattr(s.time, "monotonic", lambda: next(ticks))
    bundle = s.execute(plan, tmp_path, "stopped", [])
    assert bundle["partitions"] == []


def test_remaining_command_contract(tmp_path: Path) -> None:
    plan = s.schedule(
        dict(
            nodes=[],
            events=[],
            original_assertion_references=[],
            required_command_union=[
                dict(
                    name="required_cli",
                    argv=[str(s.ROOT / ".venv/bin/python"), "-c", "print('actual command')"],
                    expected=0,
                    deadline=20,
                )
            ],
        )
    )
    bundle = s.execute(plan, tmp_path / "run", "commands", [])
    assert s.consume(bundle, "commands")["completed_count"] == 0
    changed = deepcopy(bundle)
    changed["checks"] = []
    with pytest.raises(ValueError, match="command_union"):
        s.consume(changed, "commands")
    changed = deepcopy(bundle)
    changed["checks"][0]["receipt"]["argv"].append("tamper")
    with pytest.raises(ValueError, match="command_evidence"):
        s.consume(changed, "commands")


def test_incomplete_and_union_rejection(actual_bundle: dict[str, Any], tmp_path: Path) -> None:
    changed = deepcopy(actual_bundle)
    changed["schedule"]["original_node_manifest"] = []
    with pytest.raises(ValueError, match="obligation_union"):
        s.consume(changed, "owned-invocation")
    part = deepcopy(actual_bundle)["partitions"][0]
    events = tmp_path / "events.jsonl"
    events.write_text("")
    changed = deepcopy(actual_bundle)
    changed["partitions"][0]["events_reference"] = s.ref(events)
    with pytest.raises(ValueError, match="unfinished_partition"):
        s.consume(changed, "owned-invocation")
    changed = deepcopy(actual_bundle)
    changed["partitions"][0]["node_ids"] = []
    # Keep the disk receipt unchanged; identity checks must still reject changed scope.
    with pytest.raises(ValueError, match="partition_identity"):
        s.consume(changed, "owned-invocation")
    assert part["receipt"]["exit_code"] == 0


def test_measure_and_replay_paths(
    actual_bundle: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    from carnot.reporting import validation_schedule_runner_9139 as r

    work = dict(
        checks=[dict(passed=True)],
        references=[],
        receipts=[],
        code_config_hashes=[],
        schedule=actual_bundle["schedule"],
    )

    def execute(plan: Any, raw: Path, invocation: str, hashes: Any) -> dict[str, Any]:
        bundle = dict(deepcopy(actual_bundle), invocation_id=invocation)
        atomic_json(raw / "bundle.json", bundle)
        return bundle

    monkeypatch.setattr(s, "execute", execute)
    r.measure(work, tmp_path / "run")
    assert work["rows"] and work["isolated_child_receipts"][0]["passed"]
    value = r.build(work, "20261116", 0, 1)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert r.replay(path)["passed"]
    changed = deepcopy(value)
    changed["reproducibility_checksum"] = "bad"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="replay_hash"):
        r.replay(path)
    changed = deepcopy(value)
    changed["replay_input"]["work"]["rows"] = []
    changed["reproducibility_checksum"] = s.canonical_hash(changed["replay_input"])
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="node_reduction_drift"):
        r.replay(path)
    changed = deepcopy(value)
    changed["replay_input"]["work"]["bundle"]["invocation_id"] = "foreign"
    changed["reproducibility_checksum"] = s.canonical_hash(changed["replay_input"])
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="bundle_drift"):
        r.replay(path)
    monkeypatch.setattr(s.prior, "run_check", lambda *a, **k: dict(passed=False))
    failed = dict(work, references=[], receipts=[])
    r.measure(failed, tmp_path / "failed")
    assert failed["measurement_error"] == "normal_orchestration_failed"

    def refuse(*args: Any) -> Any:
        raise ValueError("unfinished_partition")

    monkeypatch.setattr(s, "consume", refuse)
    failed = dict(work, references=[], receipts=[])
    r.measure(failed, tmp_path / "unfinished")
    assert failed["measurement_error"] == "unfinished_partition"
    atomic_json(path, r.build(failed, "20261116", 0, 1))
    assert r.replay(path)["passed"]


def test_verification_and_coverage_controls(
    actual_bundle: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    from carnot.reporting import validation_schedule_runner_9139 as r

    class Tracer:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def combine(self, **k: Any) -> None:
            pass

        def save(self) -> None:
            pass

        def load(self) -> None:
            pass

        def set_option(self, *a: Any) -> None:
            pass

        def analysis2(self, name: str) -> Any:
            return name, [1], [], [], ""

    monkeypatch.setattr(r, "Coverage", Tracer)
    assert len(r.coverage_rows(Tracer())) == len(r.OWNED) + 1
    data = tmp_path / ".coverage"
    data.write_text("fixture data; never publication evidence")
    rows = [dict(path="fixture", statements=[1], excluded=[], missing=[])]
    monkeypatch.setattr(r, "coverage_rows", lambda t: rows)
    receipt = dict(name="check", argv=["check"], passed=True, normal_exit=True, timed_out=False)
    for stream in ["stdout", "stderr"]:
        path = tmp_path / stream
        path.write_text("fixture log")
        receipt[stream + "_path"] = str(path)
        receipt[stream + "_sha256"] = s.ref(path)["sha256"]
    monkeypatch.setattr(r, "commands", lambda raw: [("check", ["check"])])
    monkeypatch.setattr(s.prior, "run_check", lambda *a, **k: receipt)
    work = dict(receipts=[])
    r.verification(work, tmp_path)
    work["bundle"] = actual_bundle
    consumed = tmp_path / "consumed.json"
    atomic_json(consumed, s.consume(actual_bundle, "owned-invocation"))
    work["consumed_reference"] = s.ref(consumed)
    work["isolated_child_receipts"] = [receipt]
    assert r.qualified(work)
    value = r.build(
        dict(
            work,
            checks=[dict(passed=True)],
            references=[],
            code_config_hashes=[],
            schedule=actual_bundle["schedule"],
        ),
        "20261116",
        0,
        1,
    )
    assert value["schedule_ready_score"] == 1 and value["model_work_authorized"] is False
    work["receipts"] = []
    assert not r.qualified(work)
    work["receipts"] = [dict(receipt, passed=False)]
    assert not r.qualified(work)
    work["receipts"] = [receipt]
    rows[0]["missing"] = [1]
    assert not r.qualified(work)
    ticks = iter([0, 1201])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(ticks))
    r.verification(dict(receipts=[]), tmp_path)


def test_main_orchestration_and_terminal_controls(tmp_path: Path, monkeypatch: Any) -> None:
    from carnot.reporting import validation_schedule_runner_9139 as r

    class Tracer:
        def __init__(self, **k: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def save(self) -> None:
            pass

    work = dict(checks=[dict(passed=True)], references=[], receipts=[], code_config_hashes=[])
    monkeypatch.setattr(s, "authenticate", lambda *a: work)
    monkeypatch.setattr(r, "Coverage", Tracer)
    monkeypatch.setattr(r, "measure", lambda *a: None)
    monkeypatch.setattr(r, "verification", lambda *a: None)
    monkeypatch.setattr(s.prior, "run_check", lambda *a, **k: dict(passed=True))
    monkeypatch.setattr(
        r, "publish_primary", lambda output, value, validator: atomic_json(output, value)
    )
    output = tmp_path / (s.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    assert r.main(["--replay", str(output)]) == 0
    assert r.terminal(output, tmp_path / "checks")["passed"]
    changed = json.loads(output.read_bytes())
    changed["schedule_ready_score"] = 1
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(output)


def test_replay_source_reduction_and_control_drift(tmp_path: Path, monkeypatch: Any) -> None:
    from carnot.reporting import validation_schedule_runner_9139 as r

    work = s.authenticate(s.ROOT, tmp_path / "real")
    value = r.build(work, "20261116", 0, 1)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert r.replay(path)["passed"]
    value["replay_input"]["work"]["schedule"]["units"] = "minutes"
    value["reproducibility_checksum"] = s.canonical_hash(value["replay_input"])
    atomic_json(path, value)
    with pytest.raises(ValueError, match="schedule_drift"):
        r.replay(path)
    monkeypatch.setattr(s.prior.old, "semantic_control_rows", lambda: [])
    with pytest.raises(ValueError, match="historical_control_drift"):
        s.authenticate(s.ROOT, tmp_path / "control_drift")
